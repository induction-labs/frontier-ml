"""Prepare and audit the local payload from pinned official source files. Trusted only."""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import unicodedata
import urllib.request

from svg_render import render_svg, sanitize_svg, renderer_binary

HERE = Path(__file__).resolve().parent
STRATA = {"easy": "simple", "medium": "detailed", "hard": "complex"}
SEED = "frontier-svg-generation-v2-caption-audit-20260925"
AUDIT = HERE / "pipeline/caption_audit.json"
AUDIT_LOCK = HERE / "pipeline/caption_audit_lock.json"


def digest(data):
    return hashlib.sha256(data).hexdigest()


def canonical_text(text):
    return " ".join(unicodedata.normalize("NFKC", text).split()).casefold()


def load_caption_audit():
    raw = AUDIT.read_bytes()
    lock = json.loads(AUDIT_LOCK.read_text())
    if digest(raw) != lock["sha256"] or len(raw) != lock["bytes"]:
        raise ValueError("Reviewed caption audit digest mismatch")
    audit = json.loads(raw)
    if audit.get("status") != "reviewed" or audit.get("schema_version") != 2:
        raise ValueError("Caption audit must be fully reviewed")
    records = audit["rows"]
    keys = [row["source_key"] for row in records]
    expected = {f"{family}-{index:03d}" for family, count in (("easy", 100), ("medium", 99), ("hard", 100)) for index in range(count)}
    if len(keys) != len(expected) or set(keys) != expected:
        raise ValueError("Caption audit must cover every pinned source row exactly once")
    for row in records:
        decision = row.get("decision")
        if decision not in {"retain", "revise", "exclude"} or not row.get("reason"):
            raise ValueError("Invalid caption audit decision")
        reviewers = row.get("reviewers", [])
        if not isinstance(reviewers, list) or len(set(reviewers)) != len(reviewers) or not all(isinstance(r, str) and r for r in reviewers):
            raise ValueError("Invalid caption audit reviewers")
        if len(reviewers) < (1 if decision == "retain" else 2):
            raise ValueError("Corrections and exclusions require a second visual review")
        original, revised = row.get("original_caption"), row.get("revised_caption")
        if not isinstance(original, str) or digest(original.encode()) != row["original_caption_sha256"]:
            raise ValueError("Original caption audit hash mismatch")
        if decision == "exclude":
            if revised is not None or row["revised_caption_sha256"] is not None:
                raise ValueError("Excluded rows cannot carry a replacement caption")
        elif not isinstance(revised, str) or not revised.strip() or digest(revised.encode()) != row["revised_caption_sha256"]:
            raise ValueError("Revised caption audit hash mismatch")
        elif (decision == "retain") != (revised == original):
            raise ValueError("Caption decision contradicts replacement text")
    return audit, {row["source_key"]: row for row in records}, lock


def audited_caption(row, instruction, canonical, png):
    if (row["original_caption"] != instruction or row["canonical_svg_sha256"] != digest(canonical)
            or row["render_sha256"] != digest(png)):
        raise ValueError("Caption review does not match the source text, SVG or official rendering")
    return row["revised_caption"]


def fetch(source_dir):
    """Download only the four small, ungated, immutable source files."""
    source_dir = Path(source_dir)
    source_dir.mkdir(parents=True, exist_ok=True)
    lock = json.loads((HERE / "pipeline/source_lock.json").read_text())
    specs = [(family + ".json", spec["path"], spec["sha256"], spec["bytes"])
             for family, spec in lock["files"].items()]
    specs.append(("LICENSE", "LICENSE", lock["license_sha256"], 65536))
    for local, remote, expected, limit in specs:
        path = source_dir / local
        if path.exists() and digest(path.read_bytes()) == expected:
            continue
        url = f"https://raw.githubusercontent.com/{lock['repo']}/{lock['revision']}/{remote}"
        with urllib.request.urlopen(url, timeout=60) as response:
            data = response.read(limit + 1)
        if len(data) > limit or digest(data) != expected:
            raise ValueError("Pinned source download mismatch: " + remote)
        path.write_bytes(data)
    return {"source_files_verified": len(specs), "revision": lock["revision"]}


def prepare(source_dir, output_dir, *, dev_count=30):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if output_dir.exists() and any(output_dir.iterdir()):
        raise ValueError("Refusing to replace a nonempty payload directory")
    lock = json.loads((HERE / "pipeline/source_lock.json").read_text())
    license_bytes = (source_dir / "LICENSE").read_bytes()
    if digest(license_bytes) != lock["license_sha256"]:
        raise ValueError("Pinned source license mismatch")
    audit, reviewed, audit_lock = load_caption_audit()
    if audit["source_revision"] != lock["revision"]:
        raise ValueError("Caption review belongs to another source revision")
    binary, version = renderer_binary(official=True)
    if audit["renderer"] != version:
        raise ValueError("Caption review renderer version mismatch")
    raw_rows, rejected = [], []
    for family, spec in lock["files"].items():
        source = (source_dir / (family + ".json")).read_bytes()
        if digest(source) != spec["sha256"] or len(source) != spec["bytes"]:
            raise ValueError("Pinned source file mismatch: " + family)
        for index, item in enumerate(json.loads(source)):
            try:
                if not isinstance(item["question"], list) or len(item["question"]) != 1:
                    raise ValueError("Expected exactly one instruction")
                instruction = " ".join(item["question"][0].split())
                if not instruction:
                    raise ValueError("Empty instruction")
                canonical, _ = sanitize_svg(item["answer"])
                reference = canonical.decode()
                png, info = render_svg(reference, binary=binary)
                if not info["nonuniform"]:
                    raise ValueError("Uniform reference")
                review = reviewed[f"{family}-{index:03d}"]
                instruction = audited_caption(review, instruction, canonical, png)
                if instruction is None:
                    rejected.append({"family": family, "index": index, "reason": review["reason"], "decision": "caption_audit_exclusion"})
                    continue
                raw_rows.append({"instruction": instruction, "reference_svg": reference, "stratum": STRATA[family],
                                 "text_hash": digest(canonical_text(instruction).encode()), "svg_hash": digest(canonical),
                                 "render_hash": digest(png), "source_family": family, "source_index": index})
            except (ValueError, KeyError, TypeError) as error:
                # A frozen reviewed corpus cannot silently lose rows because an
                # input, review, parser or renderer changed during preparation.
                raise ValueError(f"Reviewed source mismatch at {family}-{index:03d}: {error}") from error
    # Components bind identical instructions, canonical SVGs and rendered images.
    # Keep one deterministic representative per component, preventing cross-split overlap.
    parents = list(range(len(raw_rows)))
    def find(i):
        while parents[i] != i:
            parents[i] = parents[parents[i]]
            i = parents[i]
        return i
    keys = {}
    for i, row in enumerate(raw_rows):
        for field in ("text_hash", "svg_hash", "render_hash"):
            key = (field, row[field])
            if key in keys:
                parents[find(i)] = find(keys[key])
            keys[key] = i
    components = {}
    for i, row in enumerate(raw_rows):
        components.setdefault(find(i), []).append(row)
    rank = lambda row: digest((SEED + row["text_hash"] + row["svg_hash"]).encode())
    eligible = sorted((min(group, key=rank) for group in components.values()), key=rank)
    dev = [row for row in eligible if row["stratum"] == "simple"][:dev_count]
    if len(dev) != dev_count:
        raise ValueError("Not enough simple-icon development examples")
    dev_keys = {row["text_hash"] for row in dev}
    heldout = [row for row in eligible if row["text_hash"] not in dev_keys]
    if set(row["stratum"] for row in heldout) != set(STRATA.values()):
        raise ValueError("All evaluation strata are required")
    for folder, rows, prefix in (("public", dev, "validation"), ("private", heldout, "evaluation")):
        directory = output_dir / folder
        directory.mkdir(parents=True, exist_ok=True)
        filename = "validation.jsonl" if folder == "public" else "test.jsonl"
        records = [{"id": f"{prefix}-{i:04d}", **{k: row[k] for k in ("instruction", "reference_svg", "stratum")}} for i, row in enumerate(rows)]
        (directory / filename).write_text("".join(json.dumps(row, ensure_ascii=True, separators=(",", ":")) + "\n" for row in records))
    (output_dir / "public/SCHEMA.md").write_bytes((HERE / "pipeline/SCHEMA.md").read_bytes())
    # Required attribution belongs in the payload provenance, outside the public prompt surface.
    (output_dir / "public/LICENSE.source.txt").write_bytes(license_bytes)
    manifest = {"schema_version": 1, "data_version": "caption-audit-v2", "task": "svg-generation", "source": lock,
                "caption_audit": {**audit_lock, "decisions": dict(Counter(row["decision"] for row in audit["rows"]))},
                "splits": {"validation": len(dev), "heldout": len(heldout)},
                "split_audit": {"raw_rows": len(raw_rows) + len(rejected), "rejected": rejected,
                    "duplicate_rows_removed": len(raw_rows) - len(eligible), "strata": dict(Counter(row["stratum"] for row in heldout)),
                    "group_overlap": 0, "input_fingerprint_overlap": 0, "image_overlap": 0, "passed": True,
                    "selection_seed": SEED, "preparation_renderer": version,
                    "limitations": "Public source remains searchable; neutral IDs do not prove secrecy. Near-duplicate semantic artworks can remain. Underlying artwork rights need a publication audit."},
                "records": [{"id": f"{prefix}-{i:04d}", "source_key": f"{row['source_family']}-{row['source_index']:03d}",
                             "render_sha256": row["render_hash"], "caption_sha256": digest(row["instruction"].encode())}
                            for rows, prefix in ((dev, "validation"), (heldout, "evaluation")) for i, row in enumerate(rows)],
                "files": {}}
    for path in sorted(output_dir.rglob("*")):
        if path.is_file():
            data = path.read_bytes()
            manifest["files"][str(path.relative_to(output_dir))] = {"bytes": len(data), "sha256": digest(data)}
    (output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2) + "\n")
    return manifest


def verify(directory):
    directory = Path(directory)
    manifest = json.loads((directory / "manifest.json").read_text())
    for relative, spec in manifest["files"].items():
        data = (directory / relative).read_bytes()
        if len(data) != spec["bytes"] or digest(data) != spec["sha256"]:
            raise ValueError("Prepared data mismatch: " + relative)
    for relative, key in (("public/validation.jsonl", "validation"), ("private/test.jsonl", "heldout")):
        rows = [json.loads(line) for line in (directory / relative).read_text().splitlines()]
        if len(rows) != manifest["splits"][key] or len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Invalid prepared row count or IDs")
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["fetch", "prepare", "verify"])
    parser.add_argument("--source-dir")
    parser.add_argument("--output-dir")
    args = parser.parse_args()
    if args.mode in {"fetch", "prepare"} and not args.source_dir:
        parser.error("--source-dir is required for fetch and prepare")
    if args.mode != "fetch" and not args.output_dir:
        parser.error("--output-dir is required for prepare and verify")
    if args.mode == "fetch":
        print(json.dumps(fetch(args.source_dir), indent=2))
    else:
        result = prepare(args.source_dir, args.output_dir) if args.mode == "prepare" else verify(args.output_dir)
        print(json.dumps({"splits": result["splits"], "audit": result["split_audit"]}, indent=2))


if __name__ == "__main__":
    main()
