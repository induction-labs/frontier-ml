"""Cross-verify Cloud Vision text lines with the pinned PP-OCRv5 and build requested boxes + ignore regions.

A Cloud Vision line becomes a requested box only if it is horizontal and the pinned recognizer reads
exactly the same string (after the metric's NFKC/whitespace normalization) on the metric's own crop,
so two independent OCR engines agree on every requested string. Every other text line (disagreement,
vertical text, non-Latin script, icons) becomes an ignore region: not requested, not rewarded, not
penalized. Pinned full-page detections that no Cloud Vision line explains also become ignore regions.
Pages are kept only if most of their text is requested and ignore regions stay small.

Usage: python verify.py RAW_DIR OUT_JSONL [--workers 8]
"""
import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import screenshot_generation as sg  # noqa: E402

MODELS = "/scratch/frontier-ml-workspace/work/screenshot-generation/models"
PAD = 0.12                   # vertical padding of ink-tight Cloud Vision boxes, fraction of height
MIN_REQUESTED = 6
MIN_REQUESTED_CHAR_FRACTION = 0.6
MAX_IGNORE_AREA = 0.25
_ocr = None


def overlaps_x(a, b):
    return min(a[2], b[2]) - max(a[0], b[0]) > 0


def pad_boxes(boxes):
    """Pad each box vertically by PAD * height, never into another line that shares horizontal extent."""
    out = []
    for i, (x1, y1, x2, y2) in enumerate(boxes):
        p = max(1, int(round(PAD * (y2 - y1))))
        top, bottom = max(0, y1 - p), min(sg.HEIGHT, y2 + p)
        for j, other in enumerate(boxes):
            if i == j or not overlaps_x((x1, y1, x2, y2), other):
                continue
            if other[3] <= y1 and other[3] > top:
                top = other[3]
            if other[1] >= y2 and other[1] < bottom:
                bottom = other[1]
        out.append([int(x1), int(top), int(x2), int(bottom)])
    return out


def covered(box, regions, pad=3):
    x1, y1, x2, y2 = box
    area = max(1, (x2 - x1) * (y2 - y1))
    inter = 0
    for r in regions:
        ix = max(0, min(x2, r[2] + pad) - max(x1, r[0] - pad))
        iy = max(0, min(y2, r[3] + pad) - max(y1, r[1] - pad))
        inter = max(inter, ix * iy)
    return inter / area


def union_fraction(regions):
    import numpy as np
    mask = np.zeros((sg.HEIGHT, sg.WIDTH), bool)
    for x1, y1, x2, y2 in regions:
        mask[max(0, y1):y2, max(0, x1):x2] = True
    return float(mask.mean())


def process(path):
    global _ocr
    from PIL import Image
    try:
        if _ocr is None:
            _ocr = sg.PinnedOCR(MODELS, threads=2, device="cuda", gpu_mem_limit=8 << 30)
        meta = json.loads(Path(path).read_text())
        image = Image.open(Path(path).with_suffix(".png")).convert("RGB")
        gcv = [l for l in meta["gcv_lines"] if any(c.isalnum() for c in l["text"])]
        record = {"key": meta["key"], "video_id": meta["video_id"], "heuristic_type": meta["heuristic_type"],
                  "gcv_lines": len(gcv), "gcv_chars": sum(sg.weight(l["text"]) for l in gcv)}
        flat = [l for l in gcv if l["horizontal"] and l["bbox"][2] - l["bbox"][0] >= 3 and l["bbox"][3] - l["bbox"][1] >= 5]
        boxes = pad_boxes([[max(0, l["bbox"][0]), max(0, l["bbox"][1]), min(sg.WIDTH, l["bbox"][2]), min(sg.HEIGHT, l["bbox"][3])] for l in flat])
        candidates = [{"text": sg.normalize(l["text"]), "bbox": b} for l, b in zip(flat, boxes)]
        readings = _ocr.recognize(image, candidates) if candidates else []
        requested, ignore, sims = [], [], []
        for c, r in zip(candidates, readings):
            s = sg.similarity(r, c["text"])
            sims.append(round(s, 3))
            (requested if s == 1.0 else ignore).append(c if s == 1.0 else c["bbox"])
        flat_ids = {id(l) for l in flat}
        ignore += [l["bbox"] for l in gcv if id(l) not in flat_ids]
        detections = _ocr.detect(image)
        missed = []
        for d in detections:
            b = [int(round(v)) for v in d["bbox"]]
            if sum(ch.isalnum() for ch in d["text"]) < 2:
                continue
            if covered(b, [t["bbox"] for t in requested] + ignore) < 0.5:
                missed.append(b)
        ignore += missed
        ignore = [box for box in ([max(0, int(x1)), max(0, int(y1)), min(sg.WIDTH, int(x2)), min(sg.HEIGHT, int(y2))]
                                  for x1, y1, x2, y2 in ignore) if box[2] > box[0] and box[3] > box[1]]
        req_chars = sum(sg.weight(t["text"]) for t in requested)
        record.update(requested=requested, ignore_regions=ignore, n_requested=len(requested), n_ignored=len(ignore),
                      n_missed=len(missed), requested_chars=req_chars,
                      requested_char_fraction=req_chars / max(1, record["gcv_chars"]),
                      ignore_area=union_fraction(ignore), similarities=sims)
        if requested:
            # Requested lines were read exactly by the metric's own recognizer, so real-page T is 1 by construction;
            # the independent evidence is the Cloud Vision / PP-OCRv5 agreement rate recorded above.
            t_score = 1.0
            l_score, layout = sg.layout_f1(detections, requested, ignore_regions=ignore)
            record.update(real_T=t_score, real_L=l_score, real_layout=layout)
        if len(requested) < MIN_REQUESTED:
            record["status"] = "too_few_requested"
        elif len(requested) > sg.MAX_TEXTS:
            record["status"] = "too_many_requested"
        elif record["requested_char_fraction"] < MIN_REQUESTED_CHAR_FRACTION:
            record["status"] = "low_agreement"
        elif record["ignore_area"] > MAX_IGNORE_AREA:
            record["status"] = "large_ignore_area"
        else:
            record["status"] = "ok"
        return record
    except Exception as error:
        return {"key": Path(path).stem, "status": f"error:{type(error).__name__}:{str(error)[:200]}"}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("out")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--limit", type=int)
    args = parser.parse_args()
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["key"] for l in open(args.out) if l.strip() and not json.loads(l)["status"].startswith("error")}
    paths = [str(p) for p in sorted(Path(args.raw).glob("*.json")) if p.stem not in done]
    paths = paths[:args.limit] if args.limit else paths
    print(len(paths), "pages", flush=True)
    counts = {}
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, "a") as out:
        for i, record in enumerate(pool.map(process, paths, chunksize=4)):
            out.write(json.dumps(record) + "\n")
            key = record["status"].split(":")[0]
            counts[key] = counts.get(key, 0) + 1
            if i % 200 == 0:
                print(i, counts, flush=True)
    print(counts)


if __name__ == "__main__":
    main()
