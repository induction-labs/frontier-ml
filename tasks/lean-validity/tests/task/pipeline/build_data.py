"""Organizer data pipeline driver for lean-validity. Never shipped to the agent.

Stages (each reads/writes JSONL under --run):
  statements  -> statements.jsonl            (statements.py; compile-with-sorry filter)
  generate    -> gen_<prover>.jsonl          (generate.py, one vLLM process per prover)
  extract     -> originals.jsonl             (canonical items, disallowed attempts dropped)
  check1      -> check1_originals.jsonl      (REPL compile of originals)
  mutate      -> mutants.jsonl               (near-miss + neutral edits of pass and fail parents)
  check1m     -> check1_mutants.jsonl        (REPL compile of mutants)
  check2      -> check2.jsonl                (independent `lean --json` full-file compile)
  assemble    -> data/lean-validity/{public,private} + stats.json
  prune_public -> rewrites data/lean-validity/private/test.jsonl + manifest.json without any
                  theorem whose statement appears verbatim in a public pool (no seed needed)

Pass label: both compiles agree and report no errors, no sorry, and only
propext/Classical.choice/Quot.sound. Disagreements, timeouts and crashes are dropped.
Kept theorems have >=3 matched (passing, failing) pairs, where a pair shares its source
prover and edit kind; at most 8 pairs are kept per theorem.

Held-out statements must not appear verbatim (theorem name removed, whitespace-normalized)
in any complete public pool: `statements.py` filters candidates, `assemble` re-checks the
shipped set against `statements.public_bodies()` and fails closed, and the result is
recorded in manifest.json `split_audit.heldout_public_statements`.
"""
from __future__ import annotations

import argparse
import collections
import datetime
import hashlib
import json
import os
import random
import re
import subprocess
import sys
import tempfile
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))

from lean_check import HEADER, TOOLCHAIN, run_pool  # noqa: E402
from mutate import mutate  # noqa: E402
from statements import audit_statements, public_bodies, theorem_name  # noqa: E402

MIN_PER_CLASS, MAX_PER_CLASS = 3, 8
WRITE_MANIFEST = Path(os.environ.get(
    "FRONTIER_WRITE_MANIFEST",
    "/scratch/frontier-ml-workspace/frontier-mle-clean/.claude/skills/frontier-task/scripts/write_manifest.py"))
SCHEMA = """# Lean proof-validity validation data

`validation.jsonl` has one JSON object per proof attempt:

| field | meaning |
|---|---|
| `id` | attempt id |
| `theorem` | theorem group id; every theorem has equally many compiling and non-compiling attempts (3-8 of each) |
| `lean` | the complete Lean file, exactly the text sent as `{{"lean": ...}}` to `POST /check` |
| `valid` | `true` iff the file compiles under the pinned toolchain (see below) |
| `errors` | up to three Lean error messages for invalid attempts (not available at test time) |

Toolchain: Lean {toolchain} and Mathlib tag {toolchain}. Every file starts with this header:

```lean
{header}
```

`valid` is true only if the file elaborates with no errors, no `sorry`, and
`#print axioms` for the theorem lists nothing beyond `propext`, `Classical.choice` and
`Quot.sound`. Labels were produced by two independent compiles (REPL and a full-file
`lean` run) that had to agree. Attempts using `sorry`, `admit`, `native_decide`, `axiom`,
extra imports, `set_option` or metaprogramming were removed before labeling. Attempts come
from several automated provers and from small edits of their proofs (some edits keep the
proof valid). Validation statements come from public problem sets, about half of them from
miniF2F-test (the rest from Lean Workbook and Goedel-Pset); held-out statements are
private variants of public statements (a numeric literal changed and/or variables renamed)
that do not appear verbatim in the source pools, with attempts produced the same way.

You may use this file however you want, including for training, fine-tuning, calibration
and model selection. It is the only public validation- or test-split material you may use
that way: every other copy of miniF2F (test or valid, including public proof attempts on its
statements) is off limits like any other public validation or test split, as
`/task/CONTRACT.md` says.
"""
LIMITATIONS = ("Audit covers exact ids, theorem groups, exact attempt texts and verbatim presence of every held-out "
               "statement body (theorem name removed, whitespace-normalized) in the complete public pools listed under "
               "heldout_public_statements; held-out statements are one-edit variants of public pool statements, so "
               "near-duplicates of them exist online.")


def read(path):
    with open(path) as stream:
        return [json.loads(line) for line in stream if line.strip()]


def write(path, rows):
    with open(path, "w") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False) + "\n")


def key_of(sid, lean):
    return sid + ":" + hashlib.sha256(lean.encode()).hexdigest()[:16]


def error_category(errors: list[str]) -> str:
    if not errors:
        return "none"
    e = errors[0]
    rules = [("heartbeats", r"heartbeats|maximum recursion"), ("unknown_identifier", r"unknown (identifier|constant)|unknown namespace"),
             ("unsolved_goals", r"unsolved goals"), ("linarith_failed", r"linarith failed"),
             ("type_mismatch", r"type mismatch|application type mismatch|failed to synthesize"),
             ("syntax", r"unexpected token|expected|unknown tactic"), ("simp_failed", r"simp made no progress|simp failed|norm_num failed"),
             ("omega_failed", r"omega could not"), ("rewrite_failed", r"did not find instance of the pattern|motive is not type correct"),
             ("no_goals", r"no goals")]
    for name, pattern in rules:
        if re.search(pattern, e):
            return name
    return "other"


def stage_extract(run: Path):
    from extract import canonical_item
    statements = {r["sid"]: r for r in read(run / "statements.jsonl")}
    reasons = collections.Counter()
    seen, rows = set(), []
    for path in sorted(run.glob("gen_*.jsonl")):
        for row in read(path):
            if row.get("finish_reason") != "stop":
                reasons["truncated"] += 1
                continue
            item, reason = canonical_item(row["text"], statements[row["sid"]]["statement"])
            reasons[reason] += 1
            if item is None:
                continue
            key = key_of(row["sid"], item)
            if key in seen:
                reasons["duplicate"] += 1
                continue
            seen.add(key)
            rows.append({"key": key, "sid": row["sid"], "origin": row["prover"], "kind": "original", "parent": None,
                         "lean": item, "theorem": theorem_name(statements[row["sid"]]["statement"])})
    write(run / "originals.jsonl", rows)
    print("extract", dict(reasons), "items", len(rows))


def stage_check(run: Path, source: str, target: str, mode: str, workers: int, timeout: float, only_keys=None):
    rows = read(run / source)
    if only_keys is not None:
        rows = [r for r in rows if r["key"] in only_keys]
    done = {}
    if (run / target).exists():  # resume
        done = {r["key"]: r for r in read(run / target) if r.get("status") in ("ok", "timeout")}
    jobs = [r for r in rows if r["key"] not in done]
    print(f"{target}: {len(jobs)} to check ({len(done)} cached)", flush=True)
    for start in range(0, len(jobs), 5000):
        chunk = jobs[start:start + 5000]
        results = run_pool(chunk, mode, workers, timeout)
        for result in results:
            done[result["key"]] = result
        write(run / target, list(done.values()))
    write(run / target, list(done.values()))


def stage_mutate(run: Path, seed: str, near_pass=3, neutral_pass=1, fail_parents=4):
    statements = {r["sid"]: r for r in read(run / "statements.jsonl")}
    originals = read(run / "originals.jsonl")
    results = {r["key"]: r for r in read(run / "check1_originals.jsonl")}
    by_sid = collections.defaultdict(list)
    for row in originals:
        result = results.get(row["key"])
        if result and result.get("status") == "ok":
            by_sid[row["sid"]].append((row, bool(result["valid"])))
    existing = {r["key"] for r in originals}
    mutants = []
    for sid, rows in sorted(by_sid.items()):
        if not any(valid for _, valid in rows):
            continue  # no passing parent: the theorem cannot reach the balance filter
        rng = random.Random(f"{seed}:mutate:{sid}")
        statement = statements[sid]["statement"]
        passing = [r for r, v in rows if v]
        failing = [r for r, v in rows if not v]
        rng.shuffle(passing)
        rng.shuffle(failing)
        plan = []
        for parent in passing[:4]:
            plan += [(parent, "near_miss")] * near_pass + [(parent, "neutral")] * neutral_pass
        for parent in failing[:fail_parents]:
            plan += [(parent, "near_miss"), (parent, "neutral")]
        for parent, kind in plan:
            result = mutate(parent["lean"], statement, kind, rng)
            if result is None:
                continue
            text, operator = result
            key = key_of(sid, text)
            if key in existing:
                continue
            existing.add(key)
            mutants.append({"key": key, "sid": sid, "origin": operator, "kind": kind, "parent": parent["key"],
                            "lean": text, "theorem": parent["theorem"]})
    write(run / "mutants.jsonl", mutants)
    print("mutants", len(mutants), collections.Counter(m["kind"] for m in mutants))


def candidate_keys(run: Path, seed: str, spare_pairs: int = 2) -> set:
    """Keys worth a second compile: the pairs `select` would take from first-compile labels,
    plus `spare_pairs` extra pairs per theorem to replace attempts dropped by the second
    compile. Items that are never selectable do not need a second compile."""
    items = read(run / "originals.jsonl") + (read(run / "mutants.jsonl") if (run / "mutants.jsonl").exists() else [])
    first = {r["key"]: r for r in read(run / "check1_originals.jsonl")}
    if (run / "check1_mutants.jsonl").exists():
        first.update({r["key"]: r for r in read(run / "check1_mutants.jsonl")})
    attach_provers(items)
    by_sid = collections.defaultdict(list)
    for item in items:
        r = first.get(item["key"])
        if r and r.get("status") == "ok":
            by_sid[item["sid"]].append(dict(item, valid=bool(r["valid"])))
    keys = set()
    for sid, rows in by_sid.items():
        rng = random.Random(f"{seed}:check2:{sid}")
        pos, neg = select([r for r in rows if r["valid"]], [r for r in rows if not r["valid"]], rng,
                          limit=MAX_PER_CLASS + spare_pairs)
        if len(pos) >= MIN_PER_CLASS:
            keys.update(r["key"] for r in pos + neg)
    return keys


def attach_provers(items):
    """A mutant inherits the source prover of its parent attempt."""
    prover_of = {item["key"]: item["origin"] for item in items if item["kind"] == "original"}
    for item in items:
        item["prover"] = item["origin"] if item["kind"] == "original" else prover_of.get(item["parent"], item["origin"])


def select(passing, failing, rng, limit=MAX_PER_CLASS):
    """Choose matched (compiling, non-compiling) pairs for one theorem, at most MAX_PER_CLASS.

    Attempts are bucketed by (source prover, kind), where a mutant inherits its parent's
    prover and kind is original / near_miss / neutral. Only pairs from the same bucket are
    taken, cycling over buckets, so within every theorem each bucket holds as many
    compiling as non-compiling attempts. Any predictor that depends only on the prover or
    the edit kind therefore scores exactly 0.5, like a statement-only predictor.
    """
    def buckets(items):
        table = collections.defaultdict(list)
        for item in items:
            table[(item["prover"], item["kind"])].append(item)
        for values in table.values():
            rng.shuffle(values)
        return table

    pos, neg = buckets(passing), buckets(failing)
    shared = sorted(set(pos) & set(neg))
    rng.shuffle(shared)
    shared.sort(key=lambda key: key[1] != "original")  # originals first, stable within groups
    chosen_pos, chosen_neg = [], []
    progress = True
    while progress and len(chosen_pos) < limit:
        progress = False
        for key in shared:
            if pos[key] and neg[key] and len(chosen_pos) < limit:
                chosen_pos.append(pos[key].pop())
                chosen_neg.append(neg[key].pop())
                progress = True
    return chosen_pos, chosen_neg


def shortcut_diagnostics(rows):
    """Macro F1 of label-free metadata oracles, thresholded on the split itself (an upper bound).

    prover_oracle: score = the split-wide pass rate of the attempt's source prover.
    kind_oracle:   score = the split-wide pass rate of original / near-miss / neutral.
    origin_oracle: score = the split-wide pass rate of the edit operator or prover (not visible in the text).
    length:        score = -len(text) (shorter proofs look safer).
    Balanced splits put always-yes / always-no at 1/3.
    """
    sys.path.insert(0, str(HERE / "baselines"))
    from binarize import choose_threshold

    def oracle(field):
        table = collections.defaultdict(list)
        for row in rows:
            table[row[field]].append(row["valid"])
        rate = {k: sum(v) / len(v) for k, v in table.items()}
        return [rate[row[field]] for row in rows]

    if not rows:
        return {}
    labels = [row["valid"] for row in rows]
    out = {}
    for name, values in (("prover_oracle", oracle("prover")), ("kind_oracle", oracle("kind")),
                         ("origin_oracle", oracle("origin")), ("length", [-len(r["lean"]) for r in rows])):
        out[name] = choose_threshold(values, labels)[1]
    return out


def audit_heldout_statements(heldout: list[dict], public: dict[str, set[str]]) -> dict:
    """Verbatim check of every held-out theorem statement against the complete public pools.

    One attempt per theorem group is enough (all attempts of a group share the statement).
    Returns the theorem-level result recorded in manifest.json; `passed` is False if any
    statement body appears in any pool (or cannot be parsed)."""
    by_theorem = {}
    for row in heldout:
        by_theorem.setdefault(row["theorem"], row["lean"])
    audit = audit_statements(by_theorem, public)
    return {"checked_theorems": audit["checked"], "verbatim_theorems": len(audit["matches"]),
            "by_pool": audit["by_pool"], "pools": audit["pools"], "matches": audit["matches"],
            "passed": not audit["matches"]}


def record_public_audit(data_root: Path, audit: dict) -> dict:
    """Add the verbatim-statement audit to manifest.json's split_audit (written by write_manifest.py)."""
    path = data_root / "manifest.json"
    manifest = json.loads(path.read_text())
    entry = {k: audit[k] for k in ("checked_theorems", "verbatim_theorems", "by_pool", "pools", "passed")}
    manifest["split_audit"]["heldout_public_statements"] = entry
    manifest["split_audit"]["passed"] = bool(manifest["split_audit"]["passed"] and audit["passed"])
    path.write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    return manifest


def write_manifest(data_root: Path, source: dict, audit: dict):
    """data/<slug>/manifest.json: file hashes, counts and the split audit via the shared skill
    script, with this pipeline's provenance as `source` and the public-statement audit appended."""
    if not audit["passed"]:
        raise SystemExit(f"{audit['verbatim_theorems']} held-out theorem statements appear verbatim in public pools "
                         f"{audit['by_pool']}; manifest not written")
    with tempfile.TemporaryDirectory() as temp:  # outside data_root: the script inventories every file there
        source_path = Path(temp) / "source.json"
        source_path.write_text(json.dumps(source, indent=2) + "\n")
        subprocess.run([sys.executable, str(WRITE_MANIFEST), str(data_root), "--task", "lean-validity",
                        "--source", str(source_path), "--group-field", "theorem", "--input-fields", "lean",
                        "--limitations", LIMITATIONS], check=True, stdout=subprocess.DEVNULL)
    return record_public_audit(data_root, audit)


def summary(rows):
    return {"items": len(rows), "theorems": len({r["theorem"] for r in rows}),
            "valid": sum(r["valid"] for r in rows), "invalid": sum(not r["valid"] for r in rows)}


def stage_prune_public(data_root: Path, run: Path | None = None, public: dict[str, set[str]] | None = None) -> dict:
    """Remove every held-out theorem group whose statement appears verbatim in a public pool.

    Whole groups go, so the pass/fail balance and the per-bucket matching survive; ids are
    unchanged. With --run, groups whose statement the pipeline left unmutated
    (heldout_provenance.jsonl) are removed too, although an unmutated statement is verbatim
    public by construction. Rewrites private/test.jsonl, public/SCHEMA.md and manifest.json
    (source stats, shortcut diagnostics, prune record and the public-statement audit)."""
    public = public_bodies() if public is None else public
    rows = read(data_root / "private/test.jsonl")
    before = audit_heldout_statements(rows, public)
    remove = set(before["matches"])
    unmutated = set()
    if run is not None and (run / "heldout_provenance.jsonl").exists():
        provenance = {r["id"]: r for r in read(run / "heldout_provenance.jsonl")}
        unmutated = {r["theorem"] for r in rows if not provenance[r["id"]]["statement_mutation"]}
        remove |= unmutated
    kept = [r for r in rows if r["theorem"] not in remove]
    removed = [r for r in rows if r["theorem"] in remove]
    write(data_root / "private/test.jsonl", kept)
    (data_root / "public/SCHEMA.md").write_text(SCHEMA.format(header=HEADER.rstrip("\n"), toolchain=TOOLCHAIN))
    after = audit_heldout_statements(kept, public)
    source = json.loads((data_root / "manifest.json").read_text())["source"]
    source["heldout"] = summary(kept)
    source["shortcut_diagnostics"] = shortcut_diagnostics(kept)
    if remove or "public_statement_prune" not in source:  # a clean re-run keeps the record of the original prune
        source["public_statement_prune"] = {
            "date": datetime.date.today().isoformat(),
            "checked_theorems": before["checked_theorems"], "removed_theorems": len(remove), "removed_items": len(removed),
            "verbatim_by_pool": before["by_pool"], "unmutated_theorems": len(unmutated), "pools": before["pools"],
            "rule": "held-out statement body (theorem name removed, whitespace-normalized) verbatim in a complete public pool",
        }
    write_manifest(data_root, source, after)
    result = {"removed_theorems": sorted(remove), "removed_items": len(removed), "by_pool": before["by_pool"],
              "kept": summary(kept), "audit_after": {k: after[k] for k in ("checked_theorems", "verbatim_theorems", "passed")}}
    print(json.dumps(result, indent=2))
    return result


def stage_assemble(run: Path, data_root: Path, seed: str, n_validation: int, n_heldout: int, provisional: bool):
    statements = {r["sid"]: r for r in read(run / "statements.jsonl")}
    items = read(run / "originals.jsonl") + read(run / "mutants.jsonl")
    attach_provers(items)
    first = {r["key"]: r for r in read(run / "check1_originals.jsonl") + read(run / "check1_mutants.jsonl")}
    second = {r["key"]: r for r in read(run / "check2.jsonl")}
    stats = collections.Counter()
    labeled = collections.defaultdict(list)
    for item in items:
        a, b = first.get(item["key"]), second.get(item["key"])
        if not a or not b:
            stats["not_double_compiled"] += 1
            continue
        if a.get("status") != "ok" or b.get("status") != "ok":
            stats["dropped_" + (a.get("status") if a.get("status") != "ok" else b.get("status"))] += 1
            continue
        stats["double_compiled"] += 1
        if bool(a["valid"]) != bool(b["valid"]):
            stats["disagreement"] += 1
            continue
        item = dict(item, valid=bool(a["valid"]), errors=a.get("errors") or [])
        labeled[item["sid"]].append(item)
    splits = {"validation": [], "heldout": [], "train": []}
    theorem_counts = collections.Counter()
    for sid in sorted(labeled):
        rows = labeled[sid]
        passing = [r for r in rows if r["valid"]]
        failing = [r for r in rows if not r["valid"]]
        rng = random.Random(f"{seed}:select:{sid}")
        chosen_pos, chosen_neg = select(passing, failing, rng)
        if len(chosen_pos) < MIN_PER_CLASS:
            continue
        split = statements[sid]["split"]
        theorem_counts[split] += 1
        splits[split].append((sid, chosen_pos + chosen_neg))
    limits = {"validation": n_validation, "heldout": n_heldout, "train": 100000}
    out = {}
    for split, groups in splits.items():
        rng = random.Random(f"{seed}:split:{split}")
        rng.shuffle(groups)
        groups = groups[:limits[split]]
        rows = []
        for t_index, (sid, chosen) in enumerate(sorted(groups)):
            for item in chosen:
                rows.append((f"{split[0]}t{t_index:04d}", sid, item))
        rng.shuffle(rows)  # the evaluation order interleaves theorems
        out[split] = rows
    public, private = data_root / "public", data_root / "private"
    public.mkdir(parents=True, exist_ok=True)
    private.mkdir(parents=True, exist_ok=True)
    validation = [{"id": f"v{idx:05d}", "theorem": t, "lean": item["lean"], "valid": item["valid"],
                   "errors": [e[:600] for e in item["errors"][:3]]} for idx, (t, sid, item) in enumerate(out["validation"])]
    heldout = [{"id": f"h{idx:05d}", "theorem": t, "lean": item["lean"], "valid": item["valid"],
                "error_category": error_category(item["errors"]), "kind": item["kind"], "origin": item["origin"],
                "prover": item["prover"]}
               for idx, (t, sid, item) in enumerate(out["heldout"])]
    write(public / "validation.jsonl", validation)
    (public / "SCHEMA.md").write_text(SCHEMA.format(header=HEADER.rstrip("\n"), toolchain=TOOLCHAIN))
    # Organizer-only baseline training split (never delivered; same generation process as held-out).
    write(run / "train_split.jsonl", [{"id": f"r{idx:05d}", "theorem": t, "lean": item["lean"], "valid": item["valid"],
                                       "errors": item["errors"][:3], "kind": item["kind"], "origin": item["origin"]}
                                      for idx, (t, sid, item) in enumerate(out.get("train", []))])
    write(private / "test.jsonl", heldout)
    # Organizer-only provenance, kept outside the delivered public payload.
    write(run / "heldout_provenance.jsonl", [{"id": row["id"], "sid": sid, "pool": statements[sid]["pool"],
                                               "statement_mutation": statements[sid]["mutation"], "key": item["key"]}
                                              for row, (t, sid, item) in zip(heldout, out["heldout"])])

    # Fail closed: no shipped held-out statement may appear verbatim in a complete public pool.
    audit = audit_heldout_statements(heldout, public_bodies())
    disagreement_rate = stats["disagreement"] / max(1, stats["double_compiled"])
    manifest = {
        "task": "lean-validity", "provisional": provisional,
        "toolchain": {"lean": f"leanprover/lean4:{TOOLCHAIN}", "mathlib": f"leanprover-community/mathlib4 tag {TOOLCHAIN}",
                      "maxHeartbeats": 200000, "header_sha256": hashlib.sha256(HEADER.encode()).hexdigest()},
        "label_rule": "valid iff REPL and full-file `lean --json` compiles both report no errors, no sorry, "
                      "and axioms within {propext, Classical.choice, Quot.sound}; disagreements/timeouts dropped",
        "balance_filter": {"min_per_class": MIN_PER_CLASS, "max_per_class": MAX_PER_CLASS,
                           "matching": "pairs share (source prover, edit kind)"},
        "validation": summary(validation), "heldout": summary(heldout),
        "eligible_theorems": dict(theorem_counts),
        "double_compile": {"compiled": stats["double_compiled"], "disagreements": stats["disagreement"],
                           "disagreement_rate": disagreement_rate},
        "drops": {k: v for k, v in stats.items() if k.startswith("dropped_") or k == "not_double_compiled"},
        "seed_sha256": hashlib.sha256(seed.encode()).hexdigest(),
    }
    manifest["shortcut_diagnostics"] = shortcut_diagnostics(heldout)
    (run / "source.json").write_text(json.dumps(manifest, indent=2) + "\n")
    write_manifest(data_root, manifest, audit)
    held_kind = collections.Counter((r["kind"], r["valid"]) for r in heldout)
    held_origin = collections.Counter((r["origin"], r["valid"]) for r in heldout)
    (run / "stats.json").write_text(json.dumps({"manifest": manifest, "heldout_kind_label": {f"{k}:{v}": n for (k, v), n in held_kind.items()},
                                                "heldout_origin_label": {f"{k}:{v}": n for (k, v), n in held_origin.items()}}, indent=2))
    print(json.dumps(manifest, indent=2))


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("stage", choices=["extract", "check1", "mutate", "check1m", "check2", "assemble", "prune_public"])
    parser.add_argument("--run", type=Path, help="run directory (optional for prune_public)")
    parser.add_argument("--seed-file", type=Path, help="private seed (not needed for prune_public)")
    parser.add_argument("--workers", type=int, default=96)
    parser.add_argument("--data-root", type=Path, default=Path("/scratch/frontier-ml-workspace/frontier-mle-clean/data/lean-validity"))
    parser.add_argument("--validation", type=int, default=50)
    parser.add_argument("--heldout", type=int, default=400)
    parser.add_argument("--provisional", action="store_true")
    args = parser.parse_args()
    if args.stage == "prune_public":
        stage_prune_public(args.data_root, args.run)
        return
    if args.run is None or args.seed_file is None:
        parser.error("--run and --seed-file are required for this stage")
    seed = args.seed_file.read_text().strip()
    if args.stage == "extract":
        stage_extract(args.run)
    elif args.stage == "check1":
        stage_check(args.run, "originals.jsonl", "check1_originals.jsonl", "repl", args.workers, 60)
    elif args.stage == "mutate":
        stage_mutate(args.run, seed)
    elif args.stage == "check1m":
        stage_check(args.run, "mutants.jsonl", "check1_mutants.jsonl", "repl", args.workers, 60)
    elif args.stage == "check2":
        keys = candidate_keys(args.run, seed)
        merged = read(args.run / "originals.jsonl") + read(args.run / "mutants.jsonl")
        write(args.run / "check2_jobs.jsonl", [r for r in merged if r["key"] in keys])
        stage_check(args.run, "check2_jobs.jsonl", "check2.jsonl", "cli", args.workers, 90)
    elif args.stage == "assemble":
        stage_assemble(args.run, args.data_root, seed, args.validation, args.heldout, args.provisional)


if __name__ == "__main__":
    main()
