"""Reference text lines = the pinned PP-OCRv5's own output on the real screenshot (same settings as scoring).

For every extracted frame, run the metric's full-page detector+recognizer (PinnedOCR.detect). A detection
becomes a requested line only if
  * Google Cloud Vision confirms its string (spaces aside): it equals the Cloud Vision lines centred in the
    detection box, or it lies inside one Cloud Vision line that contains it (two engines agree), and
  * the metric's own crop recognizer (PinnedOCR.recognize at the detection's box) reads the same string,
so the text and box given to the generator are PP-OCRv5's output and a real screenshot scores T = 1 by
construction. Every other detection and every unmatched Cloud Vision line becomes a neutral ignore region.

Usage: CUDA_VISIBLE_DEVICES=4 python reference_ocr.py RAW_DIR OUT_JSONL --shard 0/2 (and GPU 6 with --shard 1/2)
"""
import argparse
import json
import math
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))

MODELS = "/scratch/frontier-ml-workspace/work/screenshot-generation/models"
MIN_REQUESTED = 6
MIN_REQUESTED_CHAR_FRACTION = 0.5
MAX_IGNORE_AREA = 0.25
_ocr = None
_sg = None


def overlap(a, b):
    ix = max(0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0, min(a[3], b[3]) - max(a[1], b[1]))
    inter = ix * iy
    area = lambda r: max(1, (r[2] - r[0]) * (r[3] - r[1]))
    return inter / area(a), inter / area(b)


def int_box(b, sg):
    return [max(0, int(math.floor(b[0]))), max(0, int(math.floor(b[1]))),
            min(sg.WIDTH, int(math.ceil(b[2]))), min(sg.HEIGHT, int(math.ceil(b[3])))]


def squash(text):
    return "".join(text.split())


def cloud_vision_agrees(box, text, gcv, sg):
    """The detection's string is confirmed by Cloud Vision (spaces aside, since the engines segment words
    differently): either it equals the x-ordered concatenation of the Cloud Vision lines centred inside the
    detection box, or the detection lies inside one Cloud Vision line whose text contains it."""
    target = squash(text)
    if len(target) < 1:
        return False
    inside = []
    for l in gcv:
        x1, y1, x2, y2 = l["bbox"]
        cx, cy = (x1 + x2) / 2, (y1 + y2) / 2
        if box[0] <= cx <= box[2] and box[1] <= cy <= box[3]:
            inside.append(l)
    if inside and squash(sg.normalize(" ".join(l["text"] for l in sorted(inside, key=lambda l: l["bbox"][0])))) == target:
        return True
    for l in gcv:
        det_in_line, _ = overlap(box, l["bbox"])
        if det_in_line >= 0.7 and target in squash(sg.normalize(l["text"])) and (len(target) >= 3 or target == squash(sg.normalize(l["text"]))):
            return True
    return False


def union_fraction(regions, sg):
    import numpy as np
    mask = np.zeros((sg.HEIGHT, sg.WIDTH), bool)
    for x1, y1, x2, y2 in regions:
        mask[y1:y2, x1:x2] = True
    return float(mask.mean())


def page_status(record, max_texts=400):
    """Page filters on a processed record (shared with assembly)."""
    if record["n_requested"] < MIN_REQUESTED:
        return "too_few_requested"
    if record["n_requested"] > max_texts:
        return "too_many_requested"
    if record["requested_char_fraction"] < MIN_REQUESTED_CHAR_FRACTION:
        return "low_agreement"
    if record["ignore_area"] > MAX_IGNORE_AREA:
        return "large_ignore_area"
    return "ok"


def process(path):
    global _ocr, _sg
    from PIL import Image
    try:
        if _sg is None:
            import screenshot_generation as sg
            _sg = sg
        sg = _sg
        if _ocr is None:
            _ocr = sg.PinnedOCR(MODELS, threads=2, device="cuda", gpu_mem_limit=8 << 30)
        meta = json.loads(Path(path).read_text())
        image = Image.open(Path(path).with_suffix(".png")).convert("RGB")
        gcv = [l for l in meta["gcv_lines"] if any(c.isalnum() for c in l["text"])]
        all_detections = _ocr.detect(image)
        detections = [d for d in all_detections if any(c.isalnum() for c in d["text"])]
        record = {"key": meta["key"], "video_id": meta["video_id"], "heuristic_type": meta["heuristic_type"],
                  "n_detections": len(detections), "detection_chars": sum(sg.weight(d["text"]) for d in detections)}
        candidates, rest = [], []
        for d in detections:
            box = int_box(d["bbox"], sg)
            text = sg.normalize(d["text"])
            ok = box[2] - box[0] >= 3 and box[3] - box[1] >= 5 and cloud_vision_agrees(box, text, gcv, sg)
            (candidates if ok else rest).append({"text": text, "bbox": box})
        readings = _ocr.recognize(image, candidates) if candidates else []
        requested = [c for c, r in zip(candidates, readings) if sg.normalize(r) == c["text"]]
        requested_boxes = [t["bbox"] for t in requested]
        ignore = [c["bbox"] for c, r in zip(candidates, readings) if sg.normalize(r) != c["text"]]
        ignore += [r["bbox"] for r in rest]
        ignore += [int_box(l["bbox"], sg) for l in gcv
                   if not any(max(overlap(int_box(l["bbox"], sg), b)) >= 0.3 for b in requested_boxes)]
        ignore = [b for b in ignore if b[2] > b[0] and b[3] > b[1]]
        chars = sum(sg.weight(t["text"]) for t in requested)
        record.update(texts=requested, ignore_regions=ignore, n_requested=len(requested), requested_chars=chars,
                      requested_char_fraction=chars / max(1, record["detection_chars"]),
                      ignore_area=union_fraction(ignore, sg))
        if requested:
            # The metric's recognizer is batch-independent, so these readings are exactly what scoring the real
            # screenshot produces: T = 1 by construction. Layout uses the same full-page detections.
            recognized = [r for c, r in zip(candidates, readings) if sg.normalize(r) == c["text"]]
            record["real_T"], _ = sg.text_accuracy(recognized, requested)
            record["real_L"], record["real_layout"] = sg.layout_f1(all_detections, requested, ignore_regions=ignore)
        record["status"] = page_status(record, sg.MAX_TEXTS)
        return record
    except Exception as error:
        return {"key": Path(path).stem, "status": f"error:{type(error).__name__}:{str(error)[-200:]}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("out")
    parser.add_argument("--workers", type=int, default=2)
    parser.add_argument("--shard", default="0/1", help="i/n: process every n-th page starting at i (one process per GPU)")
    parser.add_argument("--keys", help="optional file with one key per line to process first")
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["key"] for l in open(args.out) if l.strip() and not json.loads(l)["status"].startswith("error")}
    paths = [p for p in sorted(Path(args.raw).glob("*.json")) if p.stem not in done]
    if args.keys:
        first = set(Path(args.keys).read_text().split())
        paths.sort(key=lambda p: p.stem not in first)
    index, total = (int(v) for v in args.shard.split("/"))
    paths = paths[index::total][:args.limit]
    print(len(paths), "pages", flush=True)
    counts = {}
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, "a") as out:
        for i, record in enumerate(pool.map(process, [str(p) for p in paths], chunksize=2)):
            out.write(json.dumps(record) + "\n")
            out.flush()
            key = record["status"].split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if i % 100 == 0:
                print(i, counts, flush=True)
    print(counts)


if __name__ == "__main__":
    main()
