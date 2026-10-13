"""Render organizer training pages (PCA fit, FSQ pilot) with text disjoint from both splits.

Uses raw corpus papers that were NOT selected for the paraphrased evaluation corpus, so no
title, abstract or paragraph of a validation or held-out page appears in training.

    python render_training.py --raw corpus.jsonl --exclude corpus_rewritten.jsonl \
        --fonts FONTS --out DIR --count 20000 --workers 24
"""
import argparse
import json
from multiprocessing import Pool
import os
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import ocr_latent_render as render  # noqa: E402

ARGS = None


def work(indices):
    records = [json.loads(line) for line in Path(ARGS.raw).read_text().splitlines() if line.strip()]
    excluded = {json.loads(line)["arxiv_id"] for line in Path(ARGS.exclude).read_text().splitlines() if line.strip()}
    records = [r for r in records if r["arxiv_id"] not in excluded]
    # The evaluation corpus took every usable full-text paper; synthesize article bodies for
    # training from other raw abstracts (headings from other titles), deterministically.
    import random
    rng = random.Random(ARGS.seed)
    usable = [r for r in records if r["title_plain"] and r["abstract_plain"] and len(r["title"]) < 70]
    for record in records:
        if record["title_plain"] and record["abstract_plain"] and not record["full_text"]:
            blocks = []
            for _ in range(rng.randint(3, 6)):
                blocks.append({"kind": "heading", "text": rng.choice(usable)["title"]})
                blocks.extend({"kind": "paragraph", "text": rng.choice(usable)["abstract"]} for _ in range(rng.randint(1, 3)))
            record["full_text"] = blocks
    corpus = render.Corpus(records, "unused", "all")
    renderer = render.Renderer(render.Fonts(ARGS.fonts))
    plan = render.plan(ARGS.count)
    rows = []
    try:
        for index in indices:
            kind, bucket = plan[index]
            try:
                sample = render.generate(renderer, corpus, ARGS.seed, "train", index, kind, bucket, max_attempts=12)
            except RuntimeError:
                continue
            stem = f"{index:06d}"
            (Path(ARGS.out) / f"{stem}.png").write_bytes(sample["png"])
            rows.append({"id": stem, "image": f"{stem}.png", "text": sample["text"], "kind": kind, "bucket": bucket,
                         "layout": sample["layout"], "chars": sample["chars"]})
    finally:
        renderer.close()
    return rows


def main():
    global ARGS
    parser = argparse.ArgumentParser()
    parser.add_argument("--raw", required=True)
    parser.add_argument("--exclude", required=True)
    parser.add_argument("--fonts", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--count", type=int, default=20000)
    parser.add_argument("--workers", type=int, default=24)
    parser.add_argument("--seed", default="organizer-train-v1")
    ARGS = parser.parse_args()
    Path(ARGS.out).mkdir(parents=True, exist_ok=True)
    chunks = [list(range(i, ARGS.count, ARGS.workers)) for i in range(ARGS.workers)]
    with Pool(ARGS.workers) as pool:
        results = pool.map(work, chunks)
    rows = sorted((r for chunk in results for r in chunk), key=lambda r: r["id"])
    (Path(ARGS.out) / "train.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    print(json.dumps({"rendered": len(rows), "requested": ARGS.count}))


if __name__ == "__main__":
    main()
