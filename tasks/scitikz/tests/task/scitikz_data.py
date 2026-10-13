"""Rebuild the task payload from the two immutable upstream Parquet files."""
import argparse
from collections import defaultdict
import hashlib
from io import BytesIO
import json
from pathlib import Path
import random
import shutil

BENCH = "9bcd2eeee846268dc9e0d75ee8291a141dde57b8"
TRAIN = "3061cc331aa6f71cc7430e0480929d0ccddade67"


def digest(value):
    return hashlib.sha256(value).hexdigest()


def image_hash(data):
    from PIL import Image
    image = Image.open(BytesIO(data)).convert("RGB")
    return digest(str(image.size).encode()+image.tobytes())


def main():
    import pyarrow.parquet as pq
    from PIL import Image
    from scitikz_metric import normalize_tex
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bench", required=True)
    parser.add_argument("--training", required=True)
    parser.add_argument("--output", default="data/scitikz")
    args = parser.parse_args()
    root = Path(args.output)
    for split in ("public", "private"):
        (root/split/"images").mkdir(parents=True, exist_ok=True)
    # The released renderer pads numeric IDs to six digits before sorting files.
    bench = sorted(pq.read_table(args.bench).to_pylist(), key=lambda r: int(r["id"]))
    assert len(bench) == 611
    bench_images = {image_hash(row["image"]["bytes"]) for row in bench}
    bench_codes = {digest(normalize_tex(row["code"]).encode()) for row in bench}
    # Deterministic reservoir per source category: inspect the entire allowed corpus,
    # but decode images only for final candidates and replace any exact overlap.
    randomizer = random.Random(260406079)
    groups, seen = defaultdict(list), defaultdict(int)
    for batch in pq.ParquetFile(args.training).iter_batches(batch_size=256):
        for row in batch.to_pylist():
            if digest(normalize_tex(row["code"]).encode()) in bench_codes:
                continue
            key = row.get("sci") or "unknown"
            seen[key] += 1
            pool = groups[key]
            if len(pool) < 100:
                pool.append(row)
            else:
                index = randomizer.randrange(seen[key])
                if index < 100:
                    pool[index] = row
    selected, used = [], set(bench_images)
    categories = sorted(groups)
    while len(selected) < 50:
        for key in categories:
            while groups[key]:
                row = groups[key].pop()
                fingerprint = image_hash(row["image"]["bytes"])
                if fingerprint not in used:
                    used.add(fingerprint)
                    selected.append(row)
                    break
            if len(selected) == 50:
                break
        if not any(groups.values()) and len(selected) < 50:
            raise ValueError("Insufficient nonoverlapping development examples")
    provenance = {"benchmark_revision": BENCH, "training_revision": TRAIN, "public_sources": [], "private_sources": []}
    for split, rows, filename in (("public", selected, "validation.jsonl"), ("private", bench, "test.jsonl")):
        records = []
        for i, row in enumerate(rows):
            identifier = f"{'dev' if split=='public' else 'test'}-{i:04d}"
            path = f"images/{identifier}.png"
            Image.open(BytesIO(row["image"]["bytes"])).convert("RGB").save(root/split/path)
            record = {"id": identifier, "image": path, "code": row["code"]}
            if split == "private":
                record["level"] = row["level"]
            else:
                record.update(category=row.get("sci"), figure_type=row.get("type"))
            records.append(record)
            provenance[split+"_sources"].append({"id": identifier, "source_id": row["id"]})
        (root/split/filename).write_text("".join(json.dumps(row, ensure_ascii=False)+"\n" for row in records))
    source = {"paper": "https://arxiv.org/abs/2604.06079v2", "license": "Apache-2.0 (upstream dataset declarations)",
              "sources": [{"repo": "JackieLin0123/SciTikZ-Bench", "revision": BENCH, "file": "SciTikZ_Bench.parquet",
                           "sha256": hashlib.file_digest(open(args.bench, "rb"), "sha256").hexdigest()},
                          {"repo": "JackieLin0123/SciTikZ-230K", "revision": TRAIN, "file": "SciTikZ.parquet",
                           "sha256": hashlib.file_digest(open(args.training, "rb"), "sha256").hexdigest()}],
              "selection": "Full 611 benchmark; seed 260406079 stratified reservoir from allowed corpus for 50 development examples",
              "audit": {"decoded_rgb_overlap": 0, "normalized_code_overlap": 0,
                        "limitation": "Exact overlap only, not semantic/template deduplication or pretraining contamination detection"}}
    (root/"source.json").write_text(json.dumps(source, indent=2)+"\n")
    (root/"provenance.json").write_text(json.dumps(provenance, indent=2)+"\n")
    shutil.copyfile(Path(__file__).parent/"scorer/requirements.txt", root/"public/scorer-requirements.txt")
    shutil.copyfile(Path(__file__).parent/"scorer/weights.json", root/"public/scorer-weights.json")
    shutil.copyfile(Path(__file__).parent/"SCHEMA.md", root/"public/SCHEMA.md")
    shutil.copyfile("frontier_ml/licenses/Apache-2.0.txt", root/"public/LICENSE-datasets.txt")
    print(json.dumps({"public": len(selected), "private": len(bench), "categories": categories}))


if __name__ == "__main__":
    main()
