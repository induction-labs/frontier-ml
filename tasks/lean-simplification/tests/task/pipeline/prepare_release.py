"""Create the public dataset payload from an organizer's archived research data.

Copies only evaluation inputs and reference checker assets; never uploads anything.
The private/ directory is a harness partition, not a confidentiality boundary once
the payload is published to the public task repository.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil

FILES = (
    "public/validation.jsonl", "public/SCHEMA.md", "private/test.jsonl",
    "artifacts/lean-checker/checker/Dockerfile",
    "artifacts/lean-checker/checker/ProofCheck.lean",
    "artifacts/lean-checker/checker/README.md",
    "artifacts/lean-checker/checker/worker.py",
    "artifacts/lean-checker/simplification_checker.py",
    "artifacts/lean-checker/simplification_client.py",
    "artifacts/lean-checker/simplification_metric.py",
)
SOURCE_FIELDS = ("kind", "licenses", "upstream_pools", "checker", "selection",
                 "automation", "drops", "admission", "provisional", "release_ready")


def prepare(source, output):
    source, output = source.resolve(), output.resolve()
    if output.exists() or output.is_relative_to(source):
        raise ValueError("Use a new output directory outside the organizer archive")
    original = json.loads((source / "manifest.json").read_text())
    if original["task"] != "lean-simplification":
        raise ValueError("Unexpected task manifest")
    files = {}
    for name in FILES:
        path = source / name
        if path.is_symlink() or not path.resolve().is_relative_to(source):
            raise ValueError(f"Escaping or symlinked release file: {name}")
        raw = path.read_bytes()
        record = {"bytes": len(raw), "sha256": hashlib.sha256(raw).hexdigest()}
        if record != original["files"][name]:
            raise ValueError(f"Source file differs from its verified manifest: {name}")
        files[name] = record
    splits = {}
    ids = set()
    for split, name in (("validation", "public/validation.jsonl"), ("heldout", "private/test.jsonl")):
        rows = [json.loads(line) for line in (source / name).read_text().splitlines()]
        if len(rows) != original["splits"][split] or not rows:
            raise ValueError(f"Wrong count for {split}")
        for row in rows:
            if set(row) != {"id", "statement", "proof"} or not all(isinstance(v, str) for v in row.values()):
                raise ValueError("Release rows must contain only id, statement and proof strings")
            if row["id"] in ids:
                raise ValueError("Duplicate release ID")
            ids.add(row["id"])
        splits[split] = len(rows)
    audit = dict(original["split_audit"])
    if audit.get("passed") is not True:
        raise ValueError("Source split audit did not pass")
    audit["limitations"] = (audit.get("limitations", "") + " Both splits are publicly downloadable; "
                            "private/ only controls placement within the evaluation harness.").strip()
    manifest = {"schema_version": 1, "task": original["task"],
                "source": {k: original["source"][k] for k in SOURCE_FIELDS if k in original["source"]},
                "splits": splits, "split_audit": audit, "files": files,
                "publication": {"visibility": "public", "prepared_at": datetime.now(timezone.utc).isoformat(),
                                "includes": "Evaluation inputs and reference checker source assets",
                                "organizer_records": "Provenance, screening logs, baseline responses and diagnostics stay in the local research archive"}}
    output.mkdir(parents=True)
    for name in FILES:
        target = output / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source / name, target)
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True) + "\n")
    print(json.dumps({"files": len(files) + 1, "splits": splits, "output": str(output)}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    prepare(args.source, args.output)
