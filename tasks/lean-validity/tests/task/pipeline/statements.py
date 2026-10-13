"""Candidate theorem statements for both splits (organizer side).

Validation statements are public statements used verbatim (apart from a neutral
theorem name). Held-out statements are drawn with a private seed from large public
pools and then mutated (numeric literal perturbation and/or consistent variable
renaming). A held-out candidate is kept only if the mutation changed it and its body
(statement without the theorem name, whitespace-normalized) appears in none of the
complete public pools (``public_bodies``: every Goedel-Pset shard, Lean Workbook,
FormalMATH-All, miniF2F test and valid), and only if it compiles with ``sorry`` under
the pinned toolchain (checked in ``__main__``). Before 2026-09-26 unmutated candidates
were kept, which put 26% of the shipped held-out statements verbatim in public pools;
``build_data.py prune_public`` removed them and ``stage_assemble`` now audits the shipped
set against the same pools.
"""
from __future__ import annotations

import collections
import glob
import hashlib
import json
import random
import re

HF = "/scratch/hf_home/hub"
IDENT_CHARS = r"[\w'₀-₉!?]"
BANNED_STATEMENT = re.compile(r"\b(sorry|admit|native_decide|axiom|def|abbrev|noncomputable|instance|structure|"
                              r"inductive|class|notation|macro|syntax|elab|import|set_option|variable|namespace|section|open)\b")


def _snapshot(name: str) -> str:
    return glob.glob(f"{HF}/datasets--{name.replace('/', '--')}/snapshots/*")[0]


def _snapshot_or_none(name: str) -> str | None:
    found = glob.glob(f"{HF}/datasets--{name.replace('/', '--')}/snapshots/*")
    return found[0] if found else None


def normalize(formal: str) -> str | None:
    """Reduce a pool entry to a single `theorem NAME ... := by` statement, or None."""
    text = formal.replace("\r\n", "\n")
    # Drop header lines and docstrings/comments that precede the theorem.
    text = re.sub(r"/-.*?-/", "", text, flags=re.S)
    lines = [line for line in text.split("\n")
             if not re.match(r"\s*(import|open|set_option)\b", line) and not re.match(r"\s*--", line)]
    text = "\n".join(lines).strip()
    text = re.sub(r"--[^\n]*", "", text)
    text = re.sub(r"\s*:=\s*by\s*sorry\s*$", " := by", text)
    text = re.sub(r"\s*:=\s*by\s*$", " := by", text)
    text = re.sub(r"\s*:=\s*sorry\s*$", " := by", text)
    if not re.match(r"^(theorem|lemma)\s+\S+", text) or not text.endswith(":= by"):
        return None
    if text.count(":= by") != 1 or BANNED_STATEMENT.search(text[:-5]) or len(text) > 1200:
        return None
    text = re.sub(r"^lemma\b", "theorem", text)
    text = re.sub(r"[ \t]+\n", "\n", text)
    return text


def rename(statement: str, name: str) -> str:
    return re.sub(r"^theorem\s+\S+", f"theorem {name}", statement, count=1)


def theorem_name(statement: str) -> str:
    return re.match(r"^theorem\s+(\S+)", statement).group(1)


def load_pools() -> dict[str, list[str]]:
    import pandas as pd
    pools: dict[str, list[str]] = {}
    minif2f = pd.read_parquet(_snapshot("AI-MO/minif2f_test") + "/data/train-00000-of-00001.parquet")
    pools["minif2f"] = [s for s in (normalize(x) for x in minif2f["formal_statement"]) if s]
    workbook = json.load(open(_snapshot("internlm/Lean-Workbook") + "/lean_workbook.json"))
    pools["workbook"] = [s for s in (normalize(x["formal_statement"]) for x in workbook) if s]
    formalmath = json.load(open(_snapshot("SphereLab/FormalMATH-All") + "/FormalMATH_all.json"))
    pools["formalmath"] = [s for s in (normalize(x["autoformalization"]) for x in formalmath if x.get("compiler_feedback_bool")) if s]
    pset = pd.read_parquet(_snapshot("Goedel-LM/Goedel-Pset-v1") + "/data/train-00000-of-00004.parquet")
    pools["pset"] = [s for s in (normalize(x) for x in pset["formal_statement"].sample(60000, random_state=0)) if s]
    for key in pools:  # stable order, de-duplicated by statement body
        seen, unique = set(), []
        for s in pools[key]:
            body = s.split(None, 2)[2] if len(s.split(None, 2)) == 3 else s
            if body not in seen:
                seen.add(body)
                unique.append(s)
        pools[key] = unique
    return pools


# ---- public-statement audit -------------------------------------------------------------
# The complete public pools a held-out statement must not appear in verbatim. The drawing
# pools above sample 60k Goedel-Pset rows; the audit reads all four shards, both Lean
# Workbook files, all of FormalMATH-All and both miniF2F splits (valid via
# cat-searcher/minif2f-lean4, the Lean 4 port shipped with DeepSeek-Prover-V1.5).
FULL_POOLS = ("minif2f_test", "minif2f_valid", "workbook", "formalmath", "pset")
DECLARATION = re.compile(r"(?:^|\n)[ \t]*(?:theorem|lemma)\s+\S+")
STATEMENT_END = re.compile(r":=\s*(?:by\b|sorry\b|\Z)")


def declaration_bodies(text: str) -> list[str]:
    """Whitespace-normalized bodies (statement without the theorem name, up to `:= by` /
    `:= sorry`) of every theorem or lemma in a Lean text; header lines and comments are
    ignored. This is the unit of the verbatim check."""
    text = text.replace("\r\n", "\n")
    text = re.sub(r"/-.*?-/", " ", text, flags=re.S)
    text = re.sub(r"--[^\n]*", " ", text)
    starts = list(DECLARATION.finditer(text))
    bodies = []
    for index, match in enumerate(starts):
        stop = starts[index + 1].start() if index + 1 < len(starts) else len(text)
        rest = text[match.end():stop]
        end = STATEMENT_END.search(rest)
        body = re.sub(r"\s+", " ", rest[:end.start()] if end else rest).strip()
        if body:
            bodies.append(body)
    return bodies


def statement_body(text: str) -> str | None:
    """Body of the first theorem in `text` (a bare statement or a complete Lean file), or None."""
    bodies = declaration_bodies(text)
    return bodies[0] if bodies else None


def load_full_pool_texts() -> dict[str, list[str]]:
    """Raw statement texts of every public pool, complete (not the drawing sample)."""
    import pandas as pd
    texts: dict[str, list[str]] = {}
    minif2f = pd.read_parquet(_snapshot("AI-MO/minif2f_test") + "/data/train-00000-of-00001.parquet", columns=["formal_statement"])
    texts["minif2f_test"] = list(minif2f["formal_statement"])
    lean4 = _snapshot_or_none("cat-searcher/minif2f-lean4")
    if lean4 is None:  # fail closed: an audit over fewer pools than FULL_POOLS must not report "passed"
        raise FileNotFoundError("cat-searcher/minif2f-lean4 is not in the Hugging Face cache; fetch it before auditing")
    for split in ("valid", "test"):
        with open(f"{lean4}/{split}.jsonl") as stream:
            rows = [json.loads(line)["formal_statement"] for line in stream if line.strip()]
        texts.setdefault(f"minif2f_{split}", []).extend(rows)
    workbook = _snapshot("internlm/Lean-Workbook")
    texts["workbook"] = [x["formal_statement"] for x in json.load(open(workbook + "/lean_workbook.json"))]
    texts["workbook"] += list(pd.read_parquet(workbook + "/wkbk_1009.parquet", columns=["formal_statement"])["formal_statement"])
    formalmath = json.load(open(_snapshot("SphereLab/FormalMATH-All") + "/FormalMATH_all.json"))
    texts["formalmath"] = [x["autoformalization"] for x in formalmath]
    texts["pset"] = []
    for shard in sorted(glob.glob(_snapshot("Goedel-LM/Goedel-Pset-v1") + "/data/train-*.parquet")):
        texts["pset"] += list(pd.read_parquet(shard, columns=["formal_statement"])["formal_statement"])
    missing = [pool for pool in FULL_POOLS if not texts.get(pool)]
    if missing:
        raise RuntimeError(f"Public pools missing or empty: {missing}")
    return texts


def public_bodies(texts: dict[str, list[str]] | None = None) -> dict[str, set[str]]:
    """{pool: set of statement bodies} over the complete public pools (or the given texts)."""
    texts = load_full_pool_texts() if texts is None else texts
    return {pool: {body for text in items for body in declaration_bodies(text)} for pool, items in texts.items()}


def audit_statements(statements: dict[str, str], public: dict[str, set[str]]) -> dict:
    """Which of `statements` ({id: statement or Lean file}) appear verbatim in which public pool.

    Returns {"checked": n, "matches": {id: [pools]}, "by_pool": {pool: n}, "pools": [...]}.
    A statement without a parsable theorem counts as a match of pool "unparsed" (fail closed)."""
    matches: dict[str, list[str]] = {}
    by_pool: collections.Counter = collections.Counter()
    for key, text in statements.items():
        body = statement_body(text)
        if body is None:
            matches[key] = ["unparsed"]
            by_pool["unparsed"] += 1
            continue
        pools = [pool for pool in sorted(public) if body in public[pool]]
        if pools:
            matches[key] = pools
            by_pool.update(pools)
    return {"checked": len(statements), "matches": matches, "by_pool": dict(sorted(by_pool.items())),
            "pools": sorted(public)}


NUMBER = re.compile(r"(?<![\w.'₀-₉^])(\d+)(?![\w.'₀-₉])")


def perturb_numbers(statement: str, rng: random.Random) -> str:
    head, body = statement.split(None, 2)[:2], statement.split(None, 2)[2]
    matches = [m for m in NUMBER.finditer(body) if not body[max(0, m.start() - 1):m.start()] in ("^",)]
    if not matches:
        return statement
    m = rng.choice(matches)
    value = int(m.group(1))
    options = [value + 1, value + 2, value * 2] + ([value - 1] if value > 2 else [])
    new = str(rng.choice([o for o in options if o != value]))
    body = body[:m.start()] + new + body[m.end():]
    return " ".join(head) + " " + body


BINDER = re.compile(r"\(([^():]+?)\s*:\s*[^()]*?\)")
FRESH = list("xyzuvwpqrstmnk")


def rename_variables(statement: str, rng: random.Random) -> str:
    body = statement
    names = []
    for group in BINDER.finditer(body):
        for token in group.group(1).split():
            if re.fullmatch(r"[a-zA-Z]", token):
                names.append(token)
    names = list(dict.fromkeys(names))
    if not names:
        return statement
    used = set(re.findall(r"(?<![\w'₀-₉.])([a-zA-Z])(?![\w'₀-₉])", body))
    fresh = [c for c in FRESH if c not in used]
    rng.shuffle(fresh)
    mapping = {old: fresh.pop() for old in names if fresh}
    if not mapping:
        return statement
    head, rest = body.split(None, 2)[:2], body.split(None, 2)[2]
    pattern = re.compile(r"(?<![\w'₀-₉.])(" + "|".join(map(re.escape, mapping)) + r")(?![\w'₀-₉])")
    rest = pattern.sub(lambda m: mapping[m.group(1)], rest)
    return " ".join(head) + " " + rest


def build_candidates(seed: str, n_validation: int, n_heldout: int, n_train: int = 0, *, pools=None, public=None,
                     report: dict | None = None) -> list[dict]:
    """Candidate rows for the validation, held-out and organizer train splits.

    Held-out and train candidates are drawn until `n_heldout` / `n_train` survive two filters:
    the mutation must have changed the statement, and the statement body must not appear in
    any complete public pool (`public`, default `public_bodies()`) or in an earlier candidate.
    Drop counts are written to `report` when given."""
    pools = load_pools() if pools is None else pools
    public_all = set().union(*public_bodies().values()) if public is None else set(public)
    rng = random.Random(hashlib.sha256(seed.encode()).hexdigest())
    for key in sorted(pools):
        rng.shuffle(pools[key])
    rows = []

    def name_for(split, index):
        return "thm_" + hashlib.sha256(f"{seed}:{split}:{index}".encode()).hexdigest()[:10]

    val_mix = [("minif2f", 0.5), ("workbook", 0.25), ("pset", 0.25)]
    held_mix = [("workbook", 0.45), ("pset", 0.45), ("formalmath", 0.10)]
    cursor = {key: 0 for key in pools}

    def draw(key):
        value = pools[key][cursor[key]]
        cursor[key] += 1
        return value

    for index in range(n_validation):
        key = rng.choices([k for k, _ in val_mix], [w for _, w in val_mix])[0]
        statement = rename(draw(key), name_for("validation", index))
        rows.append({"sid": f"v{index:05d}", "split": "validation", "pool": key, "statement": statement, "mutation": []})
    # Held-out statements, then an organizer-only "train" split generated the same way
    # (used for baselines; disjoint from both evaluation splits by construction). Candidates
    # the mutation left unchanged, or whose body still appears verbatim in a public pool
    # (renaming or perturbation can land on another public statement), are dropped.
    dropped = collections.Counter()
    kept = {"heldout": 0, "train": 0}
    seen_bodies = set()
    index = 0
    while kept["heldout"] < n_heldout or kept["train"] < n_train:
        split = "heldout" if kept["heldout"] < n_heldout else "train"
        key = rng.choices([k for k, _ in held_mix], [w for _, w in held_mix])[0]
        statement = rename(draw(key), name_for(split, index))
        index += 1
        mutation = []
        if rng.random() < 0.6:
            new = perturb_numbers(statement, rng)
            if new != statement:
                statement, mutation = new, mutation + ["numbers"]
        if rng.random() < 0.7:
            new = rename_variables(statement, rng)
            if new != statement:
                statement, mutation = new, mutation + ["variables"]
        body = statement_body(statement)
        if not mutation:
            dropped["unmutated"] += 1
            continue
        if body is None or body in public_all:
            dropped["verbatim_public"] += 1
            continue
        if body in seen_bodies:
            dropped["duplicate"] += 1
            continue
        seen_bodies.add(body)
        sid = f"h{kept[split]:05d}" if split == "heldout" else f"r{kept[split]:05d}"
        kept[split] += 1
        rows.append({"sid": sid, "split": split, "pool": key, "statement": statement, "mutation": mutation})
    if report is not None:
        report.update({"drawn": index, "kept": dict(kept), "dropped": dict(dropped)})
    return rows


if __name__ == "__main__":
    import argparse
    import sys
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from lean_check import HEADER, run_pool

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed-file", required=True)
    parser.add_argument("--validation", type=int, default=200)
    parser.add_argument("--heldout", type=int, default=1600)
    parser.add_argument("--train", type=int, default=0)
    parser.add_argument("--output", required=True)
    parser.add_argument("--workers", type=int, default=64)
    args = parser.parse_args()
    seed = open(args.seed_file).read().strip()
    report = {}
    rows = build_candidates(seed, args.validation, args.heldout, args.train, report=report)
    print("candidates", json.dumps(report))
    jobs = [{"key": r["sid"], "lean": HEADER + r["statement"] + "\n  sorry", "theorem": theorem_name(r["statement"])} for r in rows]
    results = run_pool(jobs, "repl", args.workers, 60)
    kept = 0
    with open(args.output, "w") as stream:
        for row, result in zip(rows, results):
            # Compiles with sorry: no errors, and only the sorry warning makes it invalid.
            ok = result.get("status") == "ok" and not result.get("errors") and result.get("sorry")
            if ok:
                kept += 1
                stream.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"kept {kept}/{len(rows)} statements that compile with sorry")
