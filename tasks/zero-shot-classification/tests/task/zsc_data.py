"""Prepare frozen classification data from pinned public parquet files (organizer only).

python tasks/zero-shot-classification/zsc_data.py --cache work/zero-shot-classification/research/data
Requires httpx and pyarrow for preparation, never for the public validator.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import shutil
import unicodedata

HERE = Path(__file__).resolve().parent
LOCK = HERE / "pipeline/source_lock.json"
SEED = "frontier-zero-shot-classification-v1-20260925"
HELDOUT_CAP = 200
VALIDATION_CAP = 500


def fingerprint(text):
    canonical = " ".join(unicodedata.normalize("NFKC", text).casefold().split())
    return hashlib.sha256(canonical.encode()).hexdigest()


def digest_file(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def decode_pairs(rows, dataset):
    """Group by text and actual hypotheses; never infer K from binary-label patterns."""
    groups, inventory, invalid = {}, set(), set()
    input_rows = 0
    for row in rows:
        input_rows += 1
        text, hypothesis, binary = row.get("text"), row.get("hypothesis"), row.get("labels")
        if not isinstance(text, str) or not text.strip() or not isinstance(hypothesis, str) or not hypothesis.strip():
            raise ValueError(f"Malformed source text/hypothesis in {dataset}")
        if type(binary) is not int or binary not in (0, 1):
            raise ValueError(f"Malformed binary annotation in {dataset}")
        key = fingerprint(text)
        inventory.add(hypothesis)
        if key not in groups:
            groups[key] = (text, {})
        labels = groups[key][1]
        if hypothesis in labels and labels[hypothesis] != binary:
            invalid.add(key)
        labels[hypothesis] = binary
    labels = sorted(inventory)
    if not 2 <= len(labels) <= 256:
        raise ValueError(f"Invalid class inventory in {dataset}")
    positions = {label: index for index, label in enumerate(labels)}
    cleaned, excluded = [], Counter()
    for key, (text, annotations) in groups.items():
        positive = [label for label, yes in annotations.items() if yes]
        if key in invalid:
            excluded["conflicting_duplicate_annotations"] += 1
        elif set(annotations) != inventory:
            excluded["incomplete_candidate_set"] += 1
        elif len(positive) != 1:
            excluded["no_positive" if not positive else "multiple_positives"] += 1
        else:
            cleaned.append({"dataset": dataset, "text": text, "labels": labels,
                            "target": positions[positive[0]]})
    audit = {"paired_rows": input_rows, "unique_inputs": len(groups), "classes": len(labels),
             "eligible": len(cleaned), "excluded": dict(sorted(excluded.items()))}
    return cleaned, audit, set(groups)


def select(rows, name, cap):
    return sorted(rows, key=lambda row: hashlib.sha256(
        (SEED + "\0" + name + "\0" + fingerprint(row["text"])).encode()).digest())[:cap]


def source_rows(path):
    import pyarrow.parquet as pq
    for batch in pq.ParquetFile(path).iter_batches(batch_size=8192):
        yield from batch.to_pylist()


def validation_rows(repo, path):
    if repo == "fancyzhx/dbpedia_14":
        labels = ["This text describes " + name + "." for name in (
            "a company", "an educational institution", "an artist", "an athlete", "a holder of public office",
            "a means of transportation", "a building", "a natural place", "a village", "an animal",
            "a plant", "a music album", "a film", "a written work")]
        for row in source_rows(path):
            yield {"dataset": "development-topic", "text": row["title"] + "\n" + row["content"],
                   "labels": labels, "target": row["label"]}
    else:
        raise ValueError("Public development is restricted to DBpedia-14 topic classification")


def fetch(entry, cache):
    path = cache / entry["repo"] / entry["file"]
    if not path.exists():
        import httpx
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".part")
        with httpx.stream("GET", f"https://huggingface.co/datasets/{entry['repo']}/resolve/{entry['revision']}/{entry['file']}",
                          follow_redirects=True, timeout=180) as response:
            response.raise_for_status()
            with temporary.open("wb") as stream:
                for chunk in response.iter_bytes():
                    stream.write(chunk)
        temporary.replace(path)
    if path.stat().st_size != entry["bytes"] or digest_file(path) != entry["sha256"]:
        raise ValueError("Pinned source checksum mismatch: " + str(path))
    return path


def prepare(cache, output):
    from zsc_metric import validate_samples
    lock = json.loads(LOCK.read_text())
    if output.exists():
        raise FileExistsError("Refusing to overwrite existing evaluation data")
    heldout, validation, audits, source_inputs = [], [], {}, set()
    for entry in lock["files"]:
        if entry["repo"] != "btzsc/btzsc":
            continue
        name = entry["file"].split("/")[0]
        rows, audit, fingerprints = decode_pairs(source_rows(fetch(entry, cache)), name)
        selected = select(rows, name, HELDOUT_CAP)
        heldout.extend(selected)
        audits[name] = dict(audit, selected=len(selected))
        source_inputs.update(fingerprints)
        print(name, json.dumps(audits[name]), flush=True)
    if len(audits) != 22 or any(a["selected"] != HELDOUT_CAP for a in audits.values()):
        raise ValueError("All 22 benchmark datasets must contribute exactly 200 examples")
    public_audits = {}
    for entry in lock["files"]:
        if entry["repo"] == "btzsc/btzsc":
            continue
        rows, seen, dropped = [], set(), 0
        for row in validation_rows(entry["repo"], fetch(entry, cache)):
            key = fingerprint(row["text"])
            if key in seen or key in source_inputs:
                dropped += 1
                continue
            seen.add(key)
            rows.append(row)
        selected = select(rows, entry["repo"], VALIDATION_CAP)
        validation.extend(selected)
        public_audits[entry["repo"]] = {"selected": len(selected), "overlap_or_duplicates_excluded": dropped}
    if len(validation) != VALIDATION_CAP or {row["dataset"] for row in validation} != {"development-topic"}:
        raise ValueError("Expected exactly 500 public topic-classification examples")
    for prefix, rows in (("validation", validation), ("heldout", heldout)):
        for index, row in enumerate(rows):
            row["id"] = f"{prefix}-{index:05d}"
        validate_samples(rows)
    overlap = {fingerprint(r["text"]) for r in validation} & {fingerprint(r["text"]) for r in heldout}
    if overlap:
        raise ValueError("Public/held-out input overlap")
    (output / "public").mkdir(parents=True)
    (output / "private").mkdir()
    for part, filename, rows in (("public", "validation.jsonl", validation), ("private", "test.jsonl", heldout)):
        # Escape Unicode line separators: the shared JSONL loader uses splitlines().
        (output / part / filename).write_text("".join(json.dumps(row, ensure_ascii=True) + "\n" for row in rows))
    for name in ("DATA_POLICY.md", "SCHEMA.md", "ATTRIBUTION.md"):
        shutil.copyfile(HERE / "pipeline" / name, output / "public" / name)
    manifest = {"schema_version": 1, "task": "zero-shot-classification", "protocol": "zero-shot-classification-v2", "source": lock,
                "splits": {"validation": len(validation), "heldout": len(heldout)},
                "selection": {"seed": SEED, "heldout_per_dataset_cap": HELDOUT_CAP,
                              "validation_per_dataset_cap": VALIDATION_CAP, "development_bucket": "topic-classification"},
                "source_audits": audits, "public_audits": public_audits,
                "split_audit": {"passed": True, "group_overlap": 0, "input_fingerprint_overlap": 0,
                    "limitations": "Public, explicitly named evaluation sources; provenance review does not prove absence of pretraining contamination. Related held-out datasets reuse texts; bootstrap by shared text. BTZSC redistribution license audit pending."},
                "files": {path.relative_to(output).as_posix(): {"bytes": path.stat().st_size, "sha256": digest_file(path)}
                          for path in sorted(output.rglob("*")) if path.is_file()}}
    (output / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest["splits"]))
    return manifest


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cache", type=Path, required=True)
    parser.add_argument("--output", type=Path, default=Path("data/zero-shot-classification"))
    args = parser.parse_args(argv)
    prepare(args.cache, args.output)


if __name__ == "__main__":
    main()
