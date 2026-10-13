"""Turn raw crawled pages into candidate samples.

For each page: normalize DOM boxes, run the pinned OCR on the real screenshot, drop the page if
OCR finds text outside the DOM boxes (the box list must be complete), and record the real-page
T and L so the metric itself can be validated.

Usage: python process_pages.py RAW_DIR OUT_JSONL [--workers 16]
"""
import argparse
import json
import math
import sys
import unicodedata
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the task directory: task-local modules by bare name
import screenshot_generation as sg  # noqa: E402

MODELS = "/scratch/frontier-ml-workspace/work/screenshot-generation/models"
OUTSIDE_MIN_CHARS = 2      # detections with fewer alphanumerics are ignored by the completeness check
OUTSIDE_COVERAGE = 0.5     # a detection must be at least this covered by (dilated) DOM boxes
_ocr = None


def bad_char(c):
    cat = unicodedata.category(c)
    return cat in ("Co", "Cn", "Cs", "So") or 0xE000 <= ord(c) <= 0xF8FF or ord(c) > 0xFFFF


def clean_boxes(boxes):
    out = []
    for b in boxes:
        text = sg.normalize(b["text"])
        if not text or not any(c.isalnum() for c in text) or any(bad_char(c) for c in text):
            continue
        x1, y1, x2, y2 = b["bbox"]
        box = [max(0, math.floor(x1)), max(0, int(round(y1))), min(sg.WIDTH, math.ceil(x2)), min(sg.HEIGHT, int(round(y2)))]
        if box[2] - box[0] < 2 or box[3] - box[1] < 4:
            continue
        out.append({"text": text, "bbox": box, "font_size": b.get("font_size"), "source": b.get("source")})
    return out


def outside_detections(detections, texts):
    import numpy as np
    mask = np.zeros((sg.HEIGHT, sg.WIDTH), bool)
    for t in texts:
        x1, y1, x2, y2 = t["bbox"]
        pad = max(3, int(round(0.25 * (y2 - y1))))
        mask[max(0, y1 - pad):y2 + pad, max(0, x1 - pad):x2 + pad] = True
    outside = []
    for d in detections:
        x1, y1, x2, y2 = (int(round(v)) for v in d["bbox"])
        if sum(c.isalnum() for c in d["text"]) < OUTSIDE_MIN_CHARS:
            continue
        if x1 <= sg.BORDER_PX or y1 <= sg.BORDER_PX or x2 >= sg.WIDTH - sg.BORDER_PX or y2 >= sg.HEIGHT - sg.BORDER_PX:
            continue
        region = mask[max(0, y1):y2, max(0, x1):x2]
        if region.size and region.mean() < OUTSIDE_COVERAGE:
            outside.append({"text": d["text"], "bbox": [x1, y1, x2, y2], "score": round(d["score"], 3)})
    return outside


MAX_REGION_FRACTION = 0.25   # a single imagery ignore region may cover at most this much of the page
MAX_IGNORE_FRACTION = 0.30   # union of ignore regions


def assign_imagery(outside, imagery):
    """Explain outside-DOM text by imagery (img/svg/canvas/iframe/background-image) rectangles.

    Returns (ignore_regions, unexplained_detections)."""
    import numpy as np
    regions = []
    for item in imagery or []:
        x1, y1, x2, y2 = item["bbox"]
        box = [max(0, math.floor(x1)), max(0, math.floor(y1)), min(sg.WIDTH, math.ceil(x2)), min(sg.HEIGHT, math.ceil(y2))]
        area = (box[2] - box[0]) * (box[3] - box[1])
        if area > 0 and area <= MAX_REGION_FRACTION * sg.WIDTH * sg.HEIGHT:
            regions.append((area, box))
    regions.sort()
    chosen, unexplained = [], []
    for det in outside:
        x1, y1, x2, y2 = det["bbox"]
        det_area = max(1, (x2 - x1) * (y2 - y1))
        match = None
        for area, box in regions:
            ix = max(0, min(x2, box[2]) - max(x1, box[0]))
            iy = max(0, min(y2, box[3]) - max(y1, box[1]))
            if ix * iy / det_area >= 0.5:
                match = box
                break
        if match is None:
            unexplained.append(det)
        elif match not in chosen:
            chosen.append(match)
    mask = np.zeros((sg.HEIGHT, sg.WIDTH), bool)
    for x1, y1, x2, y2 in chosen:
        mask[y1:y2, x1:x2] = True
    if mask.mean() > MAX_IGNORE_FRACTION:
        unexplained.append({"text": "<ignore regions too large>", "bbox": [0, 0, 0, 0], "score": 1.0})
    return chosen, unexplained


def process(path):
    try:
        return _process(path)
    except Exception as error:  # recorded, never silently kept
        return {"key": Path(path).stem, "status": f"error:{type(error).__name__}"}


def _process(path):
    global _ocr
    from PIL import Image
    if _ocr is None:
        _ocr = sg.PinnedOCR(MODELS, threads=2, device="cuda", gpu_mem_limit=int(2.5 * (1 << 30)))
    meta = json.loads(Path(path).read_text())
    png = Path(path).with_suffix(".png")
    image = Image.open(png).convert("RGB")
    if image.size != (sg.WIDTH, sg.HEIGHT):
        return {"key": Path(path).stem, "status": "size"}
    texts = clean_boxes(meta["boxes"])
    record = {"key": Path(path).stem, "url": meta.get("final_url"), "domain": meta.get("domain"), "title": meta.get("title"),
              "captured_at": meta.get("captured_at"), "n_boxes": len(texts), "n_chars": sum(sg.weight(t["text"]) for t in texts)}
    if len(texts) < 6 or record["n_chars"] < 60:
        return {**record, "status": "too_few_boxes"}
    if len(texts) > sg.MAX_TEXTS:
        return {**record, "status": "too_many_boxes"}
    detections = _ocr.detect(image)
    outside = outside_detections(detections, texts)
    ignore, unexplained = assign_imagery(outside, meta.get("imagery")) if "imagery" in meta else ([], outside)
    recognized = _ocr.recognize(image, texts)
    t_score, per_box = sg.text_accuracy(recognized, texts)
    l_score, layout = sg.layout_f1(detections, texts, ignore_regions=ignore)
    record.update(real_T=t_score, real_L=l_score, real_layout=layout, outside=unexplained[:20], n_outside=len(unexplained),
                  n_outside_in_imagery=len(outside) - len(unexplained), ignore_regions=ignore, has_imagery="imagery" in meta,
                  texts=texts, box_similarity=[round(s, 4) for s in per_box], recognized=recognized)
    record["status"] = "outside_text" if unexplained else "ok"
    return record


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("raw")
    parser.add_argument("out")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--require-imagery", action="store_true", help="skip pages crawled before imagery regions were recorded")
    args = parser.parse_args()
    done = set()
    if Path(args.out).exists():
        done = {json.loads(l)["key"] for l in open(args.out) if l.strip()}
    paths = [str(p) for p in sorted(Path(args.raw).glob("*.json")) if p.stem not in done
             and (not args.require_imagery or '"imagery":' in p.read_text())]
    print(len(paths), "pages to process", flush=True)
    counts = {}
    with ProcessPoolExecutor(args.workers) as pool, open(args.out, "a") as out:
        for i, record in enumerate(pool.map(process, paths, chunksize=4)):
            out.write(json.dumps(record) + "\n")
            counts[record["status"]] = counts.get(record["status"], 0) + 1
            if i % 100 == 0:
                print(i, counts, flush=True)
    print(counts)


if __name__ == "__main__":
    main()
