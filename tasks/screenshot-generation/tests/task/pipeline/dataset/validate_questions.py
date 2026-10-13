"""Freeze question pairs: keep a pair only if the pinned judge answers both questions correctly on the REAL
screenshot with a clear margin, and a text-only reading of the description gives the same answers
(so every frozen question is inferable from what the generator is told). Also rejects descriptions
that leak literal page text.

Usage: python validate_questions.py processed.jsonl drafts.jsonl validated.jsonl
"""
import argparse
import json
import math
import re
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the task directory: task-local modules by bare name
import screenshot_generation as sg  # noqa: E402

MODELS = "/scratch/frontier-ml-workspace/work/screenshot-generation/models"
RAW = Path("/scratch/frontier-ml-workspace/work/screenshot-generation/crawl/raw")
MARGIN = 0.3
MIN_PAIRS, MAX_PAIRS = 4, 6
TEXT_PROMPT = ("Here is a description of a screenshot:\n\n{description}\n\nBased only on this description, "
               "answer the question with a single word: yes or no. If the description does not say, answer no.\nQuestion: {question}")
STOP = set("a an the and or of to in on at for with by from is are be as its it this that these those into over under "
           "left right top bottom page large small white black blue red green grey gray dark light bold text image".split())


def words(s):
    return re.findall(r"[a-z0-9]+", s.lower())


def leaks(description, texts, domain):
    """Literal page text in the description: any shared 3-gram with a non-stopword, or a rare page token."""
    desc = words(description)
    grams = {tuple(desc[i:i + 3]) for i in range(len(desc) - 2)}
    found = []
    for t in texts:
        w = words(t["text"])
        for i in range(len(w) - 2):
            g = tuple(w[i:i + 3])
            if g in grams and any(x not in STOP for x in g):
                found.append(" ".join(g))
    label = (domain or "").split(".")[0].lower()
    if len(label) >= 4 and label in description.lower().replace(" ", ""):
        found.append("domain:" + label)
    if re.search(r"[\"“”]|\b\d{2,}\b|https?://|www\.", description):
        found.append("quote_or_number")
    lowered = description.lower()
    for t in texts:
        line = " ".join(t["text"].split()).lower()
        # Whole multi-word lines only: single UI words ("settings", "desktop") are generic vocabulary.
        if len(line) >= 8 and len(line.split()) >= 2 and any(c.isalpha() for c in line) and line in lowered:
            found.append("line:" + line[:40])
    return sorted(set(found))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("processed")
    parser.add_argument("drafts")
    parser.add_argument("out")
    parser.add_argument("--raw", default=str(RAW), help="directory of <key>.png screenshots")
    args = parser.parse_args()
    processed = {json.loads(l)["key"]: json.loads(l) for l in open(args.processed) if l.strip()}
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["key"] for l in open(args.out) if l.strip()}
    drafts = [json.loads(l) for l in open(args.drafts) if l.strip()]
    drafts = [d for d in drafts if d["key"] not in done and d["key"] in processed]
    print(len(drafts), "drafts to validate", flush=True)
    judge = sg.PinnedJudge(MODELS, gpu_memory_utilization=0.8)
    from vllm import SamplingParams
    batch = 64
    with open(args.out, "a") as out:
        for start in range(0, len(drafts), batch):
            chunk = drafts[start:start + batch]
            jobs, text_convs, owners = [], [], []
            for d in chunk:
                qs = [q for p in d.get("pairs", [])[:7] for q in (p["yes"], p["no"])] if d["status"] == "ok" else []
                jobs.append((Image.open(Path(args.raw) / f"{d['key']}.png").convert("RGB"), qs))
                for q in qs:
                    text_convs.append([{"role": "user", "content": TEXT_PROMPT.format(description=d.get("description", ""), question=q)}])
            image_answers = judge.answer(jobs)
            text_out = judge.llm.chat(text_convs, judge.params, use_tqdm=False, chat_template_kwargs={"enable_thinking": False}) if text_convs else []
            text_answers, k = [], 0
            for _, qs in jobs:
                row = []
                for _ in qs:
                    top = text_out[k].outputs[0].logprobs[0] if text_out[k].outputs[0].logprobs else {}
                    yes = sum(math.exp(v.logprob) for t, v in top.items() if t in judge.yes)
                    no = sum(math.exp(v.logprob) for t, v in top.items() if t in judge.no)
                    row.append("yes" if yes > no else "no")
                    k += 1
                text_answers.append(row)
            for d, (_, qs), ia, ta in zip(chunk, jobs, image_answers, text_answers):
                p = processed[d["key"]]
                record = {"key": d["key"], "draft_status": d["status"]}
                if d["status"] != "ok" or d.get("unsafe") is not False:
                    record["status"] = "unsafe_or_unparsed"
                    out.write(json.dumps(record) + "\n")
                    continue
                kept, audit = [], []
                for i in range(0, len(qs), 2):
                    (ya, ym), (na, nm) = ia[i], ia[i + 1]
                    ok_image = ya == "yes" and na == "no" and abs(ym) >= MARGIN and abs(nm) >= MARGIN
                    ok_text = ta[i] == "yes" and ta[i + 1] == "no"
                    audit.append({"yes": qs[i], "no": qs[i + 1], "image": [ya, round(ym, 3), na, round(nm, 3)], "text": ta[i:i + 2],
                                  "kept": ok_image and ok_text})
                    if ok_image and ok_text and len(kept) < MAX_PAIRS:
                        kept.append((qs[i], qs[i + 1]))
                leak = leaks(d["description"], p["texts"], p.get("domain"))
                questions = [{"question": q, "answer": a, "pair": j} for j, (y, n) in enumerate(kept) for q, a in ((y, "yes"), (n, "no"))]
                status = "ok"
                if leak:
                    status = "description_leak"
                elif len(kept) < MIN_PAIRS:
                    status = "too_few_pairs"
                record.update(status=status, description=d["description"], page_type=d.get("page_type"), questions=questions,
                              leaks=leak, pair_audit=audit)
                out.write(json.dumps(record) + "\n")
            out.flush()
            print(start + len(chunk), "validated", flush=True)
    judge.close()


if __name__ == "__main__":
    main()
