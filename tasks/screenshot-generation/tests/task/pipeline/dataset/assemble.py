"""Assemble data/screenshot-generation/{public,private} from processed pages and validated drafts.

Selection: page passed the DOM/OCR completeness check, the draft is safe and leak-free, and >= 4 question
pairs survived judge validation. One page per registrable domain; exact duplicate text layouts dropped.
Split: deterministic hash order under a private salt; first N_VAL -> public validation, next N_TEST -> private held-out.

Usage: python assemble.py --processed P.jsonl --validated V.jsonl --out DATA_DIR [--n-val 100 --n-test 500]
"""
import argparse
import hashlib
import json
import os
import shutil
from pathlib import Path

RAW = Path("/scratch/frontier-ml-workspace/work/screenshot-generation/crawl/raw")
SALT_FILE = Path("/scratch/frontier-ml-workspace/work/screenshot-generation/dataset/split_salt.txt")


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--processed", nargs="+", required=True)
    parser.add_argument("--validated", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=500)
    args = parser.parse_args()
    if not SALT_FILE.exists():
        SALT_FILE.write_text(os.urandom(16).hex())
    salt = SALT_FILE.read_text().strip()
    processed = {}
    for path in args.processed:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                processed[r["key"]] = r
    validated = {}
    for path in args.validated:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                validated[r["key"]] = r
    candidates = []
    for key, v in validated.items():
        p = processed.get(key)
        if not p or p["status"] != "ok" or v.get("status") != "ok":
            continue
        candidates.append((hashlib.sha256((salt + key).encode()).hexdigest(), key))
    candidates.sort()
    seen_domains, seen_layouts, chosen = set(), set(), []
    for _, key in candidates:
        p = processed[key]
        domain = p.get("domain")
        layout = hashlib.sha256(json.dumps(sorted(t["text"] for t in p["texts"])).encode()).hexdigest()
        if domain in seen_domains or layout in seen_layouts:
            continue
        seen_domains.add(domain)
        seen_layouts.add(layout)
        chosen.append(key)
    val_keys = chosen[:args.n_val]
    test_keys = chosen[args.n_val:args.n_val + args.n_test]
    out = Path(args.out)
    for split in ("public", "private"):
        if (out / split / "images").exists():
            shutil.rmtree(out / split / "images")
        (out / split / "images").mkdir(parents=True)
    shutil.copyfile(Path(__file__).resolve().parents[1] / "static" / "SCHEMA.md", out / "public" / "SCHEMA.md")
    manifest = {"task": "screenshot-generation", "files": {}}
    stats = {}
    for split, keys, prefix, name in (("public", val_keys, "val", "validation.jsonl"), ("private", test_keys, "test", "test.jsonl")):
        rows, provenance = [], []
        for i, key in enumerate(keys, 1):
            p, v = processed[key], validated[key]
            sid = f"{prefix}-{i:04d}"
            image = f"images/{sid}.png"
            shutil.copyfile(RAW / f"{key}.png", out / split / image)
            rows.append({"id": sid, "description": v["description"],
                         "texts": [{"text": t["text"], "bbox": t["bbox"]} for t in p["texts"]],
                         "questions": v["questions"], "ignore_regions": p.get("ignore_regions", []), "image": image, "size": "1280x720"})
            provenance.append({"id": sid, "key": key, "url": p["url"], "domain": p["domain"], "captured_at": p["captured_at"],
                               "page_type": v.get("page_type"), "drafter": "Qwen/Qwen3.6-27B-FP8@e89b16eb",
                               "question_validator": "Qwen/Qwen3.5-9B@c2022362 (image and description-only)",
                               "human_review": "pending", "real_T": p["real_T"], "real_L": p["real_L"],
                               "n_boxes": p["n_boxes"], "n_chars": p["n_chars"]})
        (out / split / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (out / "private" / f"provenance-{prefix}.jsonl").write_text("".join(json.dumps(r) + "\n" for r in provenance))
        n = max(1, len(provenance))
        stats[split] = {"samples": len(rows), "mean_real_T": sum(r["real_T"] for r in provenance) / n,
                        "mean_real_L": sum(r["real_L"] for r in provenance) / n,
                        "mean_boxes": sum(r["n_boxes"] for r in provenance) / n,
                        "mean_questions": sum(len(r["questions"]) for r in rows) / n}
    for path in sorted(out.rglob("*")):
        if path.is_file() and path.name != "manifest.json":
            manifest["files"][str(path.relative_to(out))] = {"sha256": sha256(path), "bytes": path.stat().st_size}
    manifest.update(stats=stats, candidates=len(candidates), unique_domains=len(chosen),
                    provisional=len(test_keys) < args.n_test or len(val_keys) < args.n_val,
                    human_review="pending: descriptions and questions are VLM-drafted and machine-validated only")
    (out / "manifest.json").write_text(json.dumps(manifest, indent=1) + "\n")
    print(json.dumps({k: v for k, v in manifest.items() if k != "files"}, indent=1))


if __name__ == "__main__":
    main()
