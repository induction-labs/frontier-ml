"""Prepare fresh proof-shortening data from an organizer's generated proof corpus.

python tasks/lean-simplification/pipeline/build_data.py --source-run DIR \
  --checker-url http://127.0.0.1:8124 --output data/lean-simplification

Read original (unmutated) successful attempts, choose the shortest per statement,
remove cross-split structural duplicates and easy automation, and recheck every
retained input with the final sandboxed kernel checker. Does not publish or spend.
"""
import argparse
from collections import Counter, defaultdict
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simplification_client import CheckerClient
from simplification_metric import lex, proof_length, proof_policy

HEADER = ("import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 200000\n\n"
          "open BigOperators Real Nat Topology Rat\n\n")
TACTICS = ("omega", "rfl", "ring", "norm_num", "simp_all", "linarith", "nlinarith", "decide", "aesop", "exact?")
AUTOMATION = "by\n  first\n" + "".join("  | (solve | " + t + ")\n" for t in TACTICS)
UPSTREAM = [
    {"repo": "AI-MO/minif2f_test", "revision": "5def318348521dfc34875045a9ecddf729a2b49f", "license": "Apache-2.0"},
    {"repo": "internlm/Lean-Workbook", "revision": "2e066e310b2c6d2c27616927ae131f82901c8f1c", "license": "Apache-2.0"},
    {"repo": "Goedel-LM/Goedel-Pset-v1", "revision": "79fc4fbe87b4c694a941bc63ad92dc38d39d41db", "license": "MIT"},
    {"repo": "SphereLab/FormalMATH-All", "revision": "8eea2550f174ce8b95444334da18a7473c2d169c", "license": "MIT"},
]
SCHEMA = """# Proof simplification

Each JSONL row contains `id`, `statement`, and `proof`. The request sends the last
two fields to `POST /simplify`; return exactly `{"proof": "<one Lean term>"}`.
`statement` is the theorem signature between its name and `:=`.

Lean v4.15.0, Mathlib v4.15.0; imports Mathlib and Aesop;
open BigOperators Real Nat Topology Rat. Proofs check under maxHeartbeats 200000,
maxRecDepth 1024, 60 seconds, 1 CPU and 4 GiB RAM, with only propext,
Classical.choice and Quot.sound axioms.
Metaprograms, commands, option changes, native execution, quoted identifiers and
literals are unsupported. Ordinary proof terms, local helpers and tactics work.
Keywords are checked on Lean's parsed syntax, so identifiers that merely contain
one (such as `Polynomial.eval`) and ordinary `$` application are accepted.

Score is mean `max(0, 1 - output_tokens/input_tokens)` for valid outputs, zero for
invalid ones. See `/task/simplification_metric.py` for exact token counting.
The optional separate checker is needed to score new proofs during validation;
without it the validation script reports predictions and no score.
"""


def read(path):
    return [json.loads(line) for line in path.read_text().splitlines() if line.strip()]


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def family(signature):
    """Conservative syntactic family: anonymize identifiers and numeric constants.

This groups alpha-renamings and numeric perturbations (and may overgroup unrelated
statements). It is not a claim of semantic decontamination against the internet.
"""
    names, out = {}, []
    for kind, token in lex(signature):
        if kind == "word" and token[0].isdigit():
            out.append("NUMBER")
        elif kind == "word" and "." not in token and token not in {"ℕ", "ℤ", "ℚ", "ℝ", "ℂ", "Prop", "Type"}:
            if token not in names:
                names[token] = f"v{len(names)}"
            out.append(names[token])
        else:
            out.append(token)
    return hashlib.sha256(json.dumps(out).encode()).hexdigest()


def candidates(source):
    statements = {r["sid"]: r for r in read(source / "statements.jsonl")}
    valid = {r["key"] for r in read(source / "check1_originals.jsonl") if r.get("valid") is True}
    grouped = defaultdict(list)
    for row in read(source / "originals.jsonl"):
        if row["key"] not in valid or row.get("kind") != "original":
            continue
        statement = statements[row["sid"]]
        prefix = HEADER + statement["statement"]
        if not row["lean"].startswith(prefix) or not prefix.endswith(":= by"):
            raise ValueError("Input does not use the pinned trusted statement/header")
        proof = "by" + row["lean"][len(prefix):]
        signature = re.sub(r"^theorem\s+\S+\s*", "", statement["statement"][:-len(":= by")]).strip()
        if proof_policy(proof):
            continue
        grouped[row["sid"]].append({"statement": signature, "proof": proof, "source_key": row["key"],
                                     "source_split": statement["split"], "family": family(signature)})
    shortest = [min(rows, key=lambda r: (proof_length(r["proof"]), r["source_key"])) for rows in grouped.values()]
    # Reserve all public families before selecting private data. The old organizer
    # train split is private and can join this new task's held-out pool.
    public_families = {r["family"] for r in shortest if r["source_split"] == "validation"}
    seen, selected, drops = set(), [], Counter()
    for row in sorted(shortest, key=lambda r: r["source_key"]):
        split = "validation" if row["source_split"] == "validation" else "heldout"
        if split == "heldout" and row["family"] in public_families:
            drops["public_family_overlap"] += 1
            continue
        if row["family"] in seen:
            drops["duplicate_family"] += 1
            continue
        seen.add(row["family"])
        if proof_length(row["proof"]) < 8:
            drops["shorter_than_8_tokens"] += 1
            continue
        selected.append(dict(row, split=split))
    return selected, drops


def prepare(source, output, checker, workers=8):
    rows, drops = candidates(source)
    print(f"Checking {len(rows)} independent candidates", flush=True)
    def screen(row):
        if not checker(row, row["proof"])["valid"]:
            return row, "original_recheck_failed"
        if checker(row, AUTOMATION)["valid"]:
            return row, "fixed_automation_solved"
        return row, None
    output.mkdir(parents=True, exist_ok=True)
    provenance, selected = [], defaultdict(list)
    # Cache every result as it arrives. This makes long CPU preparation reviewable.
    (output / "private").mkdir(exist_ok=True)
    with (output / "private/screening.jsonl").open("w") as log, ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (row, reason) in enumerate(pool.map(screen, rows), 1):
            log.write(json.dumps({"source_key": row["source_key"], "reason": reason}) + "\n")
            log.flush()
            if reason:
                drops[reason] += 1
            else:
                split = row["split"]
                identifier = f"{split}-{len(selected[split]):04d}"
                selected[split].append({"id": identifier, "statement": row["statement"], "proof": row["proof"]})
                provenance.append({"id": identifier, **{k: row[k] for k in ("source_key", "source_split", "family")}})
            if index % 25 == 0:
                print(f"{index}/{len(rows)}: retained {dict((k,len(v)) for k,v in selected.items())}; drops {dict(drops)}", flush=True)
    if not selected["validation"] or not selected["heldout"]:
        raise ValueError("Need both nonempty splits")
    for split, directory, filename in (("validation", "public", "validation.jsonl"), ("heldout", "private", "test.jsonl")):
        folder = output / directory
        folder.mkdir(exist_ok=True)
        (folder / filename).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in selected[split]))
    (output / "public/SCHEMA.md").write_text(SCHEMA)
    (output / "private/provenance.jsonl").write_text("".join(json.dumps(r) + "\n" for r in provenance))
    # Small reference checker sources help development; no model weights or private inputs.
    artifact = output / "artifacts/lean-checker"
    artifact.mkdir(parents=True, exist_ok=True)
    seed = Path(__file__).resolve().parents[1]
    shutil.copytree(seed / "checker", artifact / "checker", dirs_exist_ok=True,
                    ignore=shutil.ignore_patterns("*.olean", "*.ilean", "__pycache__"))
    for name in ("simplification_metric.py", "simplification_client.py", "simplification_checker.py"):
        shutil.copyfile(seed / name, artifact / name)
    group_sets = {split: {r["family"] for r in provenance if r["id"].startswith(split + "-")}
                  for split in ("validation", "heldout")}
    fingerprints = {split: {hashlib.sha256((r["statement"] + "\n" + r["proof"]).encode()).hexdigest()
                           for r in selected[split]} for split in ("validation", "heldout")}
    overlaps = {"group_overlap": len(group_sets["validation"] & group_sets["heldout"]),
                "id_overlap": len({r["id"] for r in selected["validation"]} & {r["id"] for r in selected["heldout"]}),
                "input_fingerprint_overlap": len(fingerprints["validation"] & fingerprints["heldout"])}
    if any(overlaps.values()):
        raise ValueError("Evaluation splits overlap")
    manifest = {"schema_version": 1, "task": "lean-simplification", "source": {
        "kind": "organizer_generated_original_proofs", "provisional": True,
        "local_source_hashes": {f: digest(source / f) for f in ("statements.jsonl", "originals.jsonl", "check1_originals.jsonl")},
        "upstream_pools": UPSTREAM, "licenses": ["Apache-2.0", "MIT"], "checker": checker.health,
        "selection": "shortest valid original per theorem; >=8 tokens; fixed automation screened; one structural family",
        "automation": AUTOMATION, "drops": dict(drops),
        "private_pool": "upstream heldout and organizer-only train; upstream validation stays public",
        "admission": "pending strongest permitted baseline, $50 pilot, timing calibration and larger data if <200 heldout"},
        "splits": {k: len(v) for k,v in selected.items()},
        "split_audit": {**overlaps, "passed": True,
                        "limitations": "Structural decontamination is conservative; public semantic relatives and cross-task private overlap remain possible."},
        "files": {}}
    for folder in ("public", "private", "artifacts"):
        for path in sorted((output / folder).rglob("*")):
            if path.is_file():
                manifest["files"][str(path.relative_to(output))] = {"bytes": path.stat().st_size, "sha256": digest(path)}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps({"splits": manifest["splits"], "drops": dict(drops)}, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-run", type=Path, required=True)
    parser.add_argument("--checker-url", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=8)
    args = parser.parse_args()
    prepare(args.source_run, args.output, CheckerClient(args.checker_url), args.workers)
