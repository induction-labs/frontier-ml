"""Extract candidate frames from the internal OCR-screens parquet (one frame per source video).

Source: /scratch/data/ocr_screens_v2_described_with_images (1280x720 WebP frames of screen recordings,
Google Cloud Vision fullTextAnnotation in `ocr_results`, a Qwen description in `prompt_text`).
For each selected video one frame is chosen with a salted hash; its image is decoded to PNG and its
Cloud Vision words are grouped into text lines (symbol detectedBreak), with each line's box the union
of its word boxes. Non-horizontal lines are kept but flagged; later steps turn them into ignore regions.

Usage: python extract.py OUT_DIR --videos 4000 [--salt-file SALT]
"""
import argparse
import collections
import glob
import hashlib
import io
import json
import re
from pathlib import Path

import pyarrow.parquet as pq
from PIL import Image

SOURCE = "/scratch/data/ocr_screens_v2_described_with_images"
TYPE_RE = [("mobile", r"\b(mobile|smartphone|phone screen|android|ios|iphone)\b"),
           ("ide_terminal", r"\b(terminal|ide|code editor|integrated development|command[- ]line|console)\b"),
           ("game", r"\b(game|gameplay|hud)\b"),
           ("web", r"\b(web page|webpage|website|browser|web-based|web application|web app)\b"),
           ("desktop", r"\b(desktop|window|application|taskbar|dock|spreadsheet|file explorer)\b")]
# Pre-selection weights by (heuristic) screen type so rarer kinds are oversampled.
WEIGHT = {"web": 1.0, "desktop": 1.0, "ide_terminal": 2.0, "mobile": 8.0, "game": 2.0, "other": 1.5}


def heuristic_type(description):
    head = (description or "")[:400].lower()
    for name, rx in TYPE_RE:
        if re.search(rx, head):
            return name
    return "other"


def parse_ocr(raw):
    obj = json.loads(raw)
    if isinstance(obj, str):
        obj = json.loads(obj)
    items = obj if isinstance(obj, list) else [obj]
    return [json.loads(it) if isinstance(it, str) else it for it in items]


def vbox(vertices):
    xs = [v.get("x", 0) for v in vertices]
    ys = [v.get("y", 0) for v in vertices]
    return [min(xs), min(ys), max(xs), max(ys)]


def horizontal(vertices):
    """Cloud Vision lists vertices clockwise from the text's top-left; horizontal text has v0.x < v1.x, same y."""
    if len(vertices) != 4:
        return False
    v0, v1 = vertices[0], vertices[1]
    dx, dy = v1.get("x", 0) - v0.get("x", 0), v1.get("y", 0) - v0.get("y", 0)
    return dx > 0 and abs(dy) <= max(2, 0.1 * dx)


def lines(pages):
    out = []
    for resp in pages:
        for page in resp.get("pages", []):
            for block in page.get("blocks", []):
                for para in block.get("paragraphs", []):
                    text, boxes, flat = "", [], True
                    for word in para.get("words", []):
                        wtext = ""
                        brk = None
                        for sym in word.get("symbols", []):
                            wtext += sym.get("text", "")
                            brk = ((sym.get("property") or {}).get("detectedBreak") or {}).get("type")
                            if brk in ("SPACE", "SURE_SPACE"):
                                wtext += " "
                        boxes.append(vbox(word["boundingBox"]["vertices"]))
                        flat = flat and horizontal(word["boundingBox"]["vertices"])
                        text += wtext
                        if brk in ("EOL_SURE_SPACE", "LINE_BREAK", "HYPHEN"):
                            if brk == "HYPHEN":
                                text += "-"
                            out.append((text, boxes, flat))
                            text, boxes, flat = "", [], True
                    if text.strip():
                        out.append((text, boxes, flat))
    result = []
    for text, boxes, flat in out:
        text = " ".join(text.split())
        if not text:
            continue
        box = [min(b[0] for b in boxes), min(b[1] for b in boxes), max(b[2] for b in boxes), max(b[3] for b in boxes)]
        result.append({"text": text, "bbox": box, "horizontal": flat, "words": len(boxes)})
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("out")
    parser.add_argument("--videos", type=int, default=4000)
    parser.add_argument("--salt-file", default="/scratch/frontier-ml-workspace/work/screenshot-generation/internal/salt.txt")
    args = parser.parse_args()
    salt_path = Path(args.salt_file)
    if not salt_path.exists():
        import os
        salt_path.write_text(os.urandom(16).hex())
    salt = salt_path.read_text().strip()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = sorted(glob.glob(SOURCE + "/*.parquet"))
    # Pass 1: index (file, row, video, heuristic type) without images.
    index = collections.defaultdict(list)
    for f in files:
        table = pq.read_table(f, columns=["video_id", "prompt_text"])
        for row, (video, desc) in enumerate(zip(table.column("video_id").to_pylist(), table.column("prompt_text").to_pylist())):
            index[video].append((f, row, heuristic_type(desc)))
    print(len(index), "videos", sum(len(v) for v in index.values()), "frames", flush=True)
    # One frame per video (salted), then a weighted salted draw of videos.
    chosen = []
    for video, frames in index.items():
        frame = min(frames, key=lambda fr: hashlib.sha256(f"{salt}:{video}:{fr[1]}:{Path(fr[0]).name}".encode()).hexdigest())
        u = int(hashlib.sha256(f"{salt}:pick:{video}".encode()).hexdigest()[:12], 16) / 16 ** 12
        chosen.append((u ** (1.0 / WEIGHT[frame[2]]), video, frame))  # Efraimidis-Spirakis weighted sampling
    chosen.sort(reverse=True)
    chosen = chosen[:args.videos]
    by_file = collections.defaultdict(list)
    for _, video, (f, row, kind) in chosen:
        by_file[f].append((row, video, kind))
    counts = collections.Counter()
    for f, rows in by_file.items():
        table = pq.read_table(f, columns=["video_id", "source_file", "source_row", "image", "ocr_results"])
        for row, video, kind in rows:
            rec = {c: table.column(c)[row].as_py() for c in ("video_id", "source_file", "source_row", "ocr_results")}
            assert rec["video_id"] == video
            image = Image.open(io.BytesIO(table.column("image")[row].as_py())).convert("RGB")
            if image.size != (1280, 720):
                counts["size"] += 1
                continue
            key = hashlib.sha256(f"{salt}:key:{video}".encode()).hexdigest()[:20]
            try:
                text_lines = lines(parse_ocr(rec["ocr_results"]))
            except Exception:
                counts["ocr_parse"] += 1
                continue
            image.save(out / f"{key}.png")
            (out / f"{key}.json").write_text(json.dumps({
                "key": key, "video_id": video, "parquet": Path(f).name, "row": row, "source_file": rec["source_file"],
                "source_row": rec["source_row"], "heuristic_type": kind, "gcv_lines": text_lines}))
            counts[kind] += 1
    print(dict(counts))


if __name__ == "__main__":
    main()
