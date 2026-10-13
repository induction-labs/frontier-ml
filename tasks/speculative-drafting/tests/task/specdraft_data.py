"""This is where we generate synthetic prompts for eval. 

Commands:
python tasks/speculative-drafting/specdraft_data.py prompts --out WORK/prompts.jsonl --seed SEED   # OpenAI, paid
modal run tasks/speculative-drafting/pipeline/specdraft_modal.py::references --prompts WORK/prompts.jsonl --out WORK/references.jsonl
python tasks/speculative-drafting/specdraft_data.py publish --references WORK/references.jsonl --destination data/speculative-drafting --seed SEED
python tasks/speculative-drafting/specdraft_data.py verify --destination data/speculative-drafting
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import time
import urllib.request

TARGET = {"repo": "Qwen/Qwen3-8B", "revision": "b968826d9c46dd6066d109eabc6255188de91218"}
GENERATION = {"chat_template": "target tokenizer, add_generation_prompt=True, enable_thinking=False",
              "decoding": "greedy (temperature 0)", "max_new_tokens": 256,
              "stop": "the target's end-of-turn token <|im_end|> (id 151645), kept as the last reference token"}
PROMPT_MODEL = "gpt-5.4-mini-2026-03-17"
CATEGORIES = {
    "math": "a math question or word problem (arithmetic, algebra, probability, geometry, calculus or number "
            "theory) that asks for a worked solution",
    "code_write": "a request to write a function, script, class or query in a named language (Python, JavaScript, "
                  "TypeScript, C++, Rust, Go, Java, SQL or Bash), with concrete requirements",
    "code_edit": "a request that pastes 20-60 lines of code in any common language and asks to fix a bug, refactor, "
                 "add a feature, add comments or explain it",
    "qa": "a factual or explanatory question about science, history, geography, technology, health, economics or "
          "everyday life",
    "writing": "a creative or practical writing request: an email, story, poem, speech, product description, "
               "cover letter, social post or outline",
    "rewrite": "a request that pastes a passage of 150-400 words (an article excerpt, notes, a review, a report or "
               "a transcript, written by you) and asks to summarize, rewrite, shorten, change tone, extract facts or "
               "answer questions about it",
    "structured": "a request to produce structured output (JSON, a Markdown table, CSV, YAML or a numbered plan) "
                  "from details given in the message",
    "multilingual": "a request written entirely in one of Chinese, Spanish, French, German, Japanese, Korean, "
                    "Russian, Portuguese, Italian or Arabic, on any everyday or technical topic",
}
TOPICS = (
    "gardening", "astronomy", "personal finance", "cooking", "cycling", "marine biology", "urban planning",
    "classical music", "cybersecurity", "board games", "medieval history", "photography", "chemistry lab work",
    "small business", "travel planning", "machine learning", "home repair", "nutrition", "volcanoes", "poetry",
    "logistics", "climate", "architecture", "linguistics", "sports statistics", "genetics", "film editing",
    "public speaking", "beekeeping", "electric vehicles", "tax paperwork", "mythology", "robotics", "knitting",
    "wine regions", "databases", "job interviews", "parenting", "birdwatching", "renewable energy", "law basics",
    "video games", "mountaineering", "pharmacology", "fashion", "philosophy", "networking protocols", "pottery",
    "supply chains", "insurance", "ancient Egypt", "coffee", "chess", "oceanography", "real estate", "journalism",
    "space missions", "carpentry", "pets", "language learning", "accounting", "earthquakes", "museums", "trains",
    "dentistry", "running", "jazz", "farming", "epidemiology", "3D printing", "event planning", "typography",
    "customer support", "mental health", "aviation", "compilers", "statistics", "bakery", "hiking", "sailing",
)
STYLES = ("terse", "detailed", "casual", "formal", "with a specific audience in mind", "with explicit constraints",
          "as a beginner", "as an expert", "with a word or length limit", "with an example of the desired format")
PER_CALL = 6
# TODO(SJ): 80 of the 320 generated prompts are unused; adding them to the held-out set would cut the SE (~0.10 tau).
DEVELOPMENT_PER_CATEGORY = 5
HELDOUT_PER_CATEGORY = 25
MAX_PROMPT_TOKENS = 1536
SCHEMA = "speculative-drafting-v1"


def sha256(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def chat(messages, *, key, model=PROMPT_MODEL, attempts=4):
    body = json.dumps({"model": model, "messages": messages, "response_format": {"type": "json_object"},
                       "reasoning_effort": "low", "max_completion_tokens": 16000}).encode()
    for attempt in range(attempts):
        request = urllib.request.Request("https://api.openai.com/v1/chat/completions", body,
                                         {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=300) as response:
                return json.load(response)
        except Exception:  # noqa: BLE001 - transient provider errors; the operator reruns on persistent failure
            if attempt + 1 == attempts:
                raise
            time.sleep(5 * (attempt + 1))


def write_prompts(out, *, key, seed, per_category, max_calls_per_category=40):
    """Append accepted prompts to `out` after every call; rerunning resumes from what is already there.

    Each category draws its hints from its own seeded stream and runs in its own thread. A malformed or empty
    model response is skipped (and counted), never parsed into a prompt.
    """
    import threading
    from concurrent.futures import ThreadPoolExecutor
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    existing = [json.loads(line) for line in out.read_text().splitlines() if line.strip()] if out.exists() else []
    seen = {" ".join(r["messages"][0]["content"].lower().split()) for r in existing}
    counts = Counter(r["category"] for r in existing)
    lock = threading.Lock()

    def category_worker(category, description):
        rng = random.Random(f"{seed}:{category}:{counts[category]}")
        made, calls, skipped = counts[category], 0, 0
        while made < per_category and calls < max_calls_per_category:
            calls += 1
            topics = rng.sample(TOPICS, PER_CALL)
            styles = [rng.choice(STYLES) for _ in topics]
            wanted = "\n".join(f"{i + 1}. topic hint: {t}; style: {s}" for i, (t, s) in enumerate(zip(topics, styles)))
            messages = [
                {"role": "system", "content": "You write realistic messages that people send to an AI assistant. "
                 "Return only JSON."},
                {"role": "user", "content": (
                    f"Write {PER_CALL} different user messages. Each must be {description}. Make them varied in "
                    "length, phrasing and difficulty, self-contained, and answerable by a text assistant without "
                    "browsing or files. Do not number them or add any commentary inside the messages.\n"
                    f"Use these hints, one per message:\n{wanted}\n"
                    'Return {"messages": ["...", ...]}.')}]
            try:
                response = chat(messages, key=key)
                texts = json.loads(response["choices"][0]["message"]["content"] or "").get("messages", [])
                if not isinstance(texts, list):
                    raise ValueError("messages is not a list")
            except (ValueError, KeyError, IndexError, TypeError, AttributeError):
                skipped += 1
                continue
            accepted = []
            with lock:
                for text in texts:
                    if made + len(accepted) >= per_category:
                        break
                    if not isinstance(text, str) or not 20 <= len(text) <= 6000:
                        continue
                    norm = " ".join(text.lower().split())
                    if norm in seen:
                        continue
                    seen.add(norm)
                    accepted.append({"category": category, "messages": [{"role": "user", "content": text}],
                                     "prompt_model": response.get("model")})
                with out.open("a") as stream:
                    stream.write("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in accepted))
            made += len(accepted)
            print(f"{category}: {made}/{per_category} (calls {calls}, skipped {skipped})", flush=True)
        return category, made

    with ThreadPoolExecutor(max_workers=len(CATEGORIES)) as pool:
        results = dict(pool.map(lambda item: category_worker(*item), CATEGORIES.items()))
    short = {c: n for c, n in results.items() if n < per_category}
    if short:
        raise RuntimeError(f"Categories short of {per_category} prompts: {short}")
    return results


def public_row(row, index):
    return {"id": f"validation-{index:04d}", "category": row["category"], "messages": row["messages"],
            "prompt_tokens": row["prompt_tokens"], "reference": row["reference"]}


def heldout_row(row, index):
    return {"id": f"heldout-{index:04d}", "category": row["category"], "messages": row["messages"],
            "prompt_tokens": row["prompt_tokens"], "reference": row["reference"],
            "finish_reason": row.get("finish_reason")}


SCHEMA_MD = """# Draft-server evaluation data

`validation.jsonl` has one conversation per line:

- `id`: row identifier.
- `category`: a coarse label for the kind of request (math, code_write, code_edit, qa, writing, rewrite,
  structured, multilingual). The held-out set has the same categories in equal proportion.
- `messages`: the conversation in chat format (one user turn).
- `prompt_tokens`: the target tokenizer's ids for `messages` with `add_generation_prompt=True` and
  `enable_thinking=False`, exactly what the server receives at the start of every request.
- `reference`: the target's greedy continuation of `prompt_tokens`, at most 256 tokens. It ends with the
  end-of-turn token `<|im_end|>` when the target finished within 256 tokens.

Only ids are sent to `/draft`: `prompt_tokens` followed by the part of `reference` accepted so far.
`TARGET.json` records the target model, revision and generation settings.
"""


def publish(references, destination, *, seed):
    rows = [json.loads(line) for line in Path(references).read_text().splitlines() if line.strip()]
    kept, seen, dropped = [], set(), Counter()
    for row in rows:
        key = tuple(row["prompt_tokens"])
        if key in seen:
            dropped["duplicate_prompt"] += 1
            continue
        if len(row["prompt_tokens"]) > MAX_PROMPT_TOKENS:
            dropped["prompt_too_long"] += 1
            continue
        if len(row["reference"]) < 16:
            dropped["reference_too_short"] += 1
            continue
        seen.add(key)
        kept.append(row)
    by_category = {}
    for row in kept:
        by_category.setdefault(row["category"], []).append(row)
    rng = random.Random(f"{seed}:split")
    validation, heldout = [], []
    for category in CATEGORIES:
        pool = by_category.get(category, [])
        rng.shuffle(pool)
        need = DEVELOPMENT_PER_CATEGORY + HELDOUT_PER_CATEGORY
        if len(pool) < need:
            raise ValueError(f"{category}: {len(pool)} usable prompts, need {need}")
        validation += pool[:DEVELOPMENT_PER_CATEGORY]
        heldout += pool[DEVELOPMENT_PER_CATEGORY:need]
    rng.shuffle(validation)
    rng.shuffle(heldout)
    destination = Path(destination)
    (destination / "public").mkdir(parents=True, exist_ok=True)
    (destination / "private").mkdir(parents=True, exist_ok=True)
    public_rows = [public_row(row, i + 1) for i, row in enumerate(validation)]
    private_rows = [heldout_row(row, i + 1) for i, row in enumerate(heldout)]
    (destination / "public/validation.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in public_rows))
    (destination / "private/test.jsonl").write_text(
        "".join(json.dumps(r, ensure_ascii=True) + "\n" for r in private_rows))
    (destination / "public/SCHEMA.md").write_text(SCHEMA_MD)
    (destination / "public/TARGET.json").write_text(json.dumps(
        {"model": TARGET["repo"], "revision": TARGET["revision"], **GENERATION}, indent=2) + "\n")
    prompt_overlap = len({tuple(r["prompt_tokens"]) for r in public_rows} & {tuple(r["prompt_tokens"]) for r in private_rows})
    text_overlap = len({r["messages"][0]["content"] for r in public_rows} & {r["messages"][0]["content"] for r in private_rows})
    manifest = {
        "schema_version": 1, "task": "speculative-drafting", "dataset_schema": SCHEMA,
        "source": {"generator": "specdraft_data.py prompts", "prompt_model": PROMPT_MODEL,
                   "prompt_models_returned": sorted({r.get("prompt_model") or "" for r in rows}),
                   "target": TARGET, "generation": GENERATION, "references_sha256": sha256(references),
                   "reference_engine": sorted({json.dumps(r.get("engine"), sort_keys=True) for r in rows}),
                   "license": "prompts written for this task; references are outputs of the Apache-2.0 target"},
        "splits": {"validation": len(public_rows), "heldout": len(private_rows)},
        "counts": {"validation": dict(Counter(r["category"] for r in public_rows)),
                   "heldout": dict(Counter(r["category"] for r in private_rows)),
                   "dropped": dict(dropped), "generated": len(rows)},
        "reference_tokens": {"validation": sum(len(r["reference"]) for r in public_rows),
                             "heldout": sum(len(r["reference"]) for r in private_rows)},
        "split_audit": {"group_overlap": prompt_overlap, "input_fingerprint_overlap": text_overlap,
                        "passed": prompt_overlap == 0 and text_overlap == 0,
                        "limitations": ("Prompts are model-written and fresh; the held-out prompts were never "
                                        "published. References are one engine's greedy decode; near-tie positions "
                                        "may differ from another engine's greedy argmax (see reference_agreement).")},
        "reference_agreement": _agreement([*validation, *heldout]),
        "files": {},
    }
    if not manifest["split_audit"]["passed"]:
        raise ValueError(f"Cross-split leakage: {manifest['split_audit']}")
    for relative in ("public/validation.jsonl", "public/SCHEMA.md", "public/TARGET.json", "private/test.jsonl"):
        path = destination / relative
        manifest["files"][relative] = {"bytes": path.stat().st_size, "sha256": sha256(path)}
    (destination / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def _agreement(rows):
    checked = [r["argmax_agreement"] for r in rows if isinstance(r.get("argmax_agreement"), dict)]
    if not checked:
        return None
    positions = sum(c["positions"] for c in checked)
    return {"rows": len(checked), "positions": positions,
            "mismatches": sum(c["mismatches"] for c in checked),
            "method": checked[0].get("method")}


def verify(destination):
    destination = Path(destination)
    manifest = json.loads((destination / "manifest.json").read_text())
    for relative, meta in manifest["files"].items():
        path = destination / relative
        if path.stat().st_size != meta["bytes"] or sha256(path) != meta["sha256"]:
            raise ValueError(f"{relative} differs from the manifest")
    public = [json.loads(line) for line in (destination / "public/validation.jsonl").read_text().splitlines()]
    private = [json.loads(line) for line in (destination / "private/test.jsonl").read_text().splitlines()]
    if len(public) != manifest["splits"]["validation"] or len(private) != manifest["splits"]["heldout"]:
        raise ValueError("Split counts differ from the manifest")
    for row in public:
        if set(row) != {"id", "category", "messages", "prompt_tokens", "reference"}:
            raise ValueError("Public rows carry only agent-visible fields")
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    prompts = sub.add_parser("prompts")
    prompts.add_argument("--out", required=True)
    # TODO(SJ): back up data/speculative-drafting.private-seed somewhere durable and private (not the repo or HF).
    prompts.add_argument("--seed", required=True, help="private seed; record it outside the repository")
    prompts.add_argument("--per-category", type=int, default=36)
    publish_cmd = sub.add_parser("publish")
    publish_cmd.add_argument("--references", required=True)
    publish_cmd.add_argument("--destination", required=True)
    publish_cmd.add_argument("--seed", required=True)
    verify_cmd = sub.add_parser("verify")
    verify_cmd.add_argument("--destination", required=True)
    args = parser.parse_args(argv)
    if args.command == "prompts":
        import os
        write_prompts(args.out, key=os.environ["OPENAI_API_KEY"], seed=args.seed, per_category=args.per_category)
    elif args.command == "publish":
        print(json.dumps(publish(args.references, args.destination, seed=args.seed)["splits"]))
    else:
        print(json.dumps(verify(args.destination)["splits"]))


if __name__ == "__main__":
    main()
