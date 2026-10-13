"""Pinned MATH-500 and official MMLU-Pro preparation and integrity checks."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import urllib.request

LOCK_PATH = Path(__file__).with_name("metadata") / "language_sources.json"
ALIASES = {"math_500": "math500", "math500": "math500", "mmlu_pro": "mmlu-pro", "mmlu-pro": "mmlu-pro"}
EXPECTED_COUNTS = {"math500": 500, "mmlu-pro": 12032}
MATH500_SUBJECTS = frozenset({"Algebra", "Counting & Probability", "Geometry", "Intermediate Algebra",
                              "Number Theory", "Prealgebra", "Precalculus"})


def _name(name):
    if name not in ALIASES:
        raise ValueError(f"Unknown benchmark: {name}")
    return ALIASES[name]


def source_lock():
    return json.loads(LOCK_PATH.read_text())


def _digest(path):
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _verify_file(path, expected):
    path = Path(path)
    if path.is_symlink() or not path.is_file() or path.stat().st_size != expected["bytes"] or _digest(path) != expected["sha256"]:
        raise ValueError(f"Pinned benchmark file changed: {path}")


def normalize_math500(rows):
    """The 500 upstream problems in file order: id by position, question/answer/subject/level kept, the worked
    solution dropped, and the upstream unique_id retained as source_id."""
    if len(rows) != EXPECTED_COUNTS["math500"]:
        raise ValueError("MATH-500 requires all 500 source problems")
    normalized = []
    for index, row in enumerate(rows, 1):
        if not isinstance(row.get("unique_id"), str) or not isinstance(row.get("answer"), str):
            raise ValueError("MATH-500 source rows need a string unique_id and answer")
        normalized.append({"id": f"math500-{index:04d}", "question": row["problem"], "answer": row["answer"],
                           "subject": row["subject"], "level": row["level"], "source_id": row["unique_id"]})
    validate_rows("math500", normalized)
    return normalized


def normalize_mmlu(rows, *, validation=False):
    if len(rows) != (70 if validation else EXPECTED_COUNTS["mmlu-pro"]):
        raise ValueError("MMLU-Pro requires the complete official split (12,032 test or 70 validation)")
    normalized = []
    for row in sorted(rows, key=lambda r: r["question_id"]):
        source_id = row["question_id"]
        if type(source_id) is not int or type(row.get("answer_index")) is not int or row["answer"] != chr(ord("A") + row["answer_index"]):
            raise ValueError("MMLU-Pro source answer index mismatch")
        options = [option for option in row["options"] if option != "N/A"]
        if any(option == "N/A" for option in row["options"][:row["answer_index"] + 1]):
            raise ValueError("N/A removal would change the official answer letter")
        record = {"id": f"mmlu-pro-{source_id}", "question": row["question"], "options": options, "answer": row["answer"], "category": row["category"], "source_id": source_id, "source": row["src"]}
        if validation:
            record["cot_content"] = row["cot_content"]
        normalized.append(record)
    validate_rows("mmlu-pro", normalized, validation=validation)
    return normalized


def validate_rows(name, rows, *, validation=False):
    name = _name(name)
    if validation and name != "mmlu-pro":
        raise ValueError("Only MMLU-Pro has an official few-shot validation split")
    expected_count = 70 if validation else EXPECTED_COUNTS[name]
    if len(rows) != expected_count:
        raise ValueError(f"Incomplete {name}: expected {expected_count} rows")
    if len({r.get("id") for r in rows}) != len(rows) or len({r.get("source_id") for r in rows}) != len(rows):
        raise ValueError("Duplicate benchmark ID or source ID")
    for position, row in enumerate(rows, 1):
        if not isinstance(row.get("question"), str) or not row["question"].strip():
            raise ValueError("Empty benchmark question")
        if name == "math500":
            if not isinstance(row.get("answer"), str) or not row["answer"].strip():
                raise ValueError("MATH-500 answers must be nonempty reference expressions")
            if row.get("subject") not in MATH500_SUBJECTS or type(row.get("level")) is not int or not 1 <= row["level"] <= 5:
                raise ValueError("MATH-500 rows need one of the seven subjects and a level from 1 to 5")
            if not isinstance(row.get("source_id"), str) or not row["source_id"].startswith("test/") or not row["source_id"].endswith(".json"):
                raise ValueError("MATH-500 source_id must be the upstream unique_id (test/<subject>/<n>.json)")
            if row["id"] != f"math500-{position:04d}":
                raise ValueError("MATH-500 canonical IDs follow the pinned file order")
        else:
            options = row.get("options")
            answer = row.get("answer")
            if not isinstance(options, list) or not 2 <= len(options) <= 10 or not all(isinstance(o, str) and o.strip() for o in options) or not isinstance(answer, str) or len(answer) != 1 or answer not in "ABCDEFGHIJ"[:len(options)]:
                raise ValueError("Invalid MMLU-Pro options or answer")
            if type(row.get("source_id")) is not int or row["id"] != f"mmlu-pro-{row['source_id']}" or not isinstance(row.get("category"), str) or not row["category"]:
                raise ValueError("Invalid MMLU-Pro source ID/category")
            if validation and (not isinstance(row.get("cot_content"), str) or not row["cot_content"].strip()):
                raise ValueError("MMLU-Pro validation examples require official CoT explanations")
    return rows


def serialize_rows(rows):
    return "".join(json.dumps(row, ensure_ascii=False, sort_keys=True, separators=(",", ":")) + "\n" for row in rows).encode()


def _read_jsonl(path):
    with Path(path).open(encoding="utf-8") as stream:
        return [json.loads(line) for line in stream if line.strip()]


def prepare(destination, *, download=True):
    """Fetch pinned raw files and reproduce exact prepared JSONL; no paid APIs."""
    root = Path(destination)
    lock = source_lock()
    summaries = {}
    for name in EXPECTED_COUNTS:
        record = lock[name]
        raw_root = root / "sources" / name
        for relative, expected in record["files"].items():
            path = raw_root / relative
            if not path.exists():
                if not download:
                    raise FileNotFoundError(path)
                path.parent.mkdir(parents=True, exist_ok=True)
                url = f"https://huggingface.co/datasets/{record['repo']}/resolve/{record['revision']}/{relative}"
                with urllib.request.urlopen(url, timeout=120) as response:
                    payload = response.read(expected["bytes"] + 1)
                if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                    raise ValueError(f"Downloaded bytes do not match source lock: {relative}")
                path.write_bytes(payload)
            _verify_file(path, expected)
        if name == "math500":
            rows = normalize_math500(_read_jsonl(raw_root / "test.jsonl"))
        else:
            import pyarrow.parquet as pq
            rows = normalize_mmlu(pq.read_table(raw_root / "data/test-00000-of-00001.parquet").to_pylist())
        output = root / record["prepared"]["file"]
        payload = serialize_rows(rows)
        if len(payload) != record["prepared"]["bytes"] or hashlib.sha256(payload).hexdigest() != record["prepared"]["sha256"]:
            raise ValueError("Prepared benchmark differs from the pinned normalized bytes")
        if output.exists():
            _verify_file(output, record["prepared"])
        else:
            output.write_bytes(payload)
        if name == "mmlu-pro":
            examples = normalize_mmlu(pq.read_table(raw_root / "data/validation-00000-of-00001.parquet").to_pylist(), validation=True)
            expected = record["validation_prepared"]
            payload = serialize_rows(examples)
            if len(payload) != expected["bytes"] or hashlib.sha256(payload).hexdigest() != expected["sha256"]:
                raise ValueError("Prepared MMLU-Pro validation differs from pinned bytes")
            validation_path = root / expected["file"]
            if validation_path.exists():
                _verify_file(validation_path, expected)
            else:
                validation_path.write_bytes(payload)
        (root / (name + ".manifest.json")).write_text(json.dumps(record, indent=2, sort_keys=True) + "\n")
        summaries[name] = {"samples": len(rows), "path": str(output), "sha256": record["prepared"]["sha256"], "verified": True}
    return summaries


def load_prepared(name, path):
    """Trust the package lock, never an adjacent candidate-controlled manifest."""
    name = _name(name)
    record = source_lock()[name]
    path = Path(path)
    if path.is_dir():
        path = path / record["prepared"]["file"]
    _verify_file(path, record["prepared"])
    # JSON strings may legally contain Unicode line separators; splitlines()
    # would cut a valid question in half. JSONL records are ASCII-newline delimited.
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    return validate_rows(name, rows)


def load_mmlu_validation(path):
    record = source_lock()["mmlu-pro"]["validation_prepared"]
    path = Path(path)
    if path.is_dir():
        path = path / record["file"]
    _verify_file(path, record)
    with path.open(encoding="utf-8") as stream:
        rows = [json.loads(line) for line in stream]
    return validate_rows("mmlu-pro", rows, validation=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("destination", type=Path)
    parser.add_argument("--offline", action="store_true", help="Use only already downloaded pinned raw files")
    parser.add_argument("--verify-only", action="store_true")
    args = parser.parse_args()
    if args.verify_only:
        result = {name: {"samples": len(load_prepared(name, args.destination)), "verified": True} for name in EXPECTED_COUNTS}
        result["mmlu-pro-validation"] = {"samples": len(load_mmlu_validation(args.destination)), "verified": True}
    else:
        result = prepare(args.destination, download=not args.offline)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
