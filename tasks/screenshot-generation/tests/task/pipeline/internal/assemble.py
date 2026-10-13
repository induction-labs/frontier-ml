"""Assemble data/screenshot-generation-internal/ from verified candidates, drafts and frozen question pairs.

Selection: cross-OCR-verified page, real-page layout F1 >= 0.85, safe and leak-free description, >= 4
validated pairs. One frame per source video already; near-duplicates (CLIP cosine >= 0.95) and identical
text sets are dropped. Stratified by the drafter's screen_type with fixed quotas (shortfalls filled from
the other types), in salted order; the first stratified draw is validation, the next is held-out.
Provenance (video id, parquet row) goes only to private/provenance-*.jsonl.

Usage: python assemble.py --candidates C.jsonl --drafts D.jsonl --validated V.jsonl --raw RAW --out DIR
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
import screenshot_generation as sg  # noqa: E402

QUOTAS = {"web_page": 0.35, "desktop_app": 0.35, "ide_terminal": 0.12, "mobile": 0.06, "other": 0.12}
SALT_FILE = Path("/scratch/frontier-ml-workspace/work/screenshot-generation/internal/salt.txt")
MODELS = "/scratch/frontier-ml-workspace/work/screenshot-generation/models"
STARTING_ARTIFACT = HERE / "pipeline/static/STARTING_ARTIFACT.json"


ALIASES = {"web_app": "web_page", "website": "web_page", "desktop": "desktop_app", "terminal": "ide_terminal", "ide": "ide_terminal"}


def screen_type(draft):
    t = ALIASES.get(draft.get("screen_type"), draft.get("screen_type"))
    return t if t in QUOTAS else "other"


def clean_regions(regions):
    """Clip to the canvas and drop empty rectangles (detector polygons can extend past the image)."""
    out = []
    for x1, y1, x2, y2 in regions:
        box = [max(0, int(x1)), max(0, int(y1)), min(sg.WIDTH, int(x2)), min(sg.HEIGHT, int(y2))]
        if box[2] > box[0] and box[3] > box[1]:
            out.append(box)
    return out


def load(path):
    return {r["key"]: r for r in (json.loads(l) for l in open(path) if l.strip())}


def stratified(pool, n):
    """pool: {type: [keys in salted order]} -> n keys following QUOTAS, filling shortfalls round-robin."""
    want = {t: int(round(q * n)) for t, q in QUOTAS.items()}
    pick = {t: pool.get(t, [])[:want[t]] for t in QUOTAS}
    missing = n - sum(len(v) for v in pick.values())
    while missing > 0:
        progressed = False
        for t in sorted(QUOTAS, key=lambda t: -QUOTAS[t]):
            spare = pool.get(t, [])[len(pick[t]):]
            if spare and missing > 0:
                pick[t].append(spare[0])
                missing -= 1
                progressed = True
        if not progressed:
            break
    chosen = [k for t in QUOTAS for k in pick[t]]
    for t in QUOTAS:
        pool[t] = pool.get(t, [])[len(pick[t]):]
    return chosen


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--candidates", required=True)
    parser.add_argument("--drafts", required=True)
    parser.add_argument("--validated", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-val", type=int, default=100)
    parser.add_argument("--n-test", type=int, default=500)
    args = parser.parse_args()
    salt = SALT_FILE.read_text().strip()
    cands, drafts, validated = load(args.candidates), load(args.drafts), load(args.validated)
    keys = [k for k, v in validated.items() if v.get("status") == "ok" and k in cands and drafts.get(k, {}).get("status") == "ok"]
    keys.sort(key=lambda k: hashlib.sha256(f"{salt}:order:{k}".encode()).hexdigest())
    # Near-duplicate removal (CLIP cosine) and identical text sets.
    from PIL import Image
    clip = sg.PinnedCLIP(MODELS)
    emb = clip.embed([Image.open(Path(args.raw) / f"{k}.png").convert("RGB") for k in keys])
    clip.close()
    kept, kept_emb, seen_text = [], [], set()
    for k, e in zip(keys, emb):
        text_key = hashlib.sha256(json.dumps(sorted(t["text"] for t in cands[k]["texts"])).encode()).hexdigest()
        if text_key in seen_text or any(float(e @ f) >= 0.95 for f in kept_emb):
            continue
        seen_text.add(text_key)
        kept.append(k)
        kept_emb.append(e)
    pool = {}
    for k in kept:
        pool.setdefault(screen_type(drafts[k]), []).append(k)
    available = {t: len(v) for t, v in pool.items()}
    val_keys = stratified(pool, args.n_val)
    test_keys = stratified(pool, args.n_test)
    out = Path(args.out)
    for split in ("public", "private"):
        for sub in ("media", "images"):
            if (out / split / sub).exists():
                shutil.rmtree(out / split / sub)
    (out / "public/media").mkdir(parents=True, exist_ok=True)
    (out / "private/images").mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, keys_, prefix, name, media in (("public", val_keys, "validation", "validation.jsonl", "media"),
                                              ("private", test_keys, "test", "test.jsonl", "images")):
        rows, provenance = [], []
        for i, k in enumerate(keys_, 1):
            c, d, v = cands[k], drafts[k], validated[k]
            sid = f"{prefix}-{i:04d}"
            image = f"{media}/{i:06d}.png" if split == "public" else f"images/{sid}.png"
            shutil.copyfile(Path(args.raw) / f"{k}.png", out / split / image)
            rows.append({"id": sid, "description": v["description"], "screen_type": screen_type(d),
                         "texts": [{"text": t["text"], "bbox": t["bbox"]} for t in c["texts"]],
                         "questions": v["questions"], "ignore_regions": clean_regions(c["ignore_regions"]), "size": "1280x720",
                         "image": image})
            provenance.append({"id": sid, "key": k, "video_id": c["video_id"], "screen_type": screen_type(d),
                               "page_type": d.get("page_type"), "real_T": c["real_T"], "real_L": c["real_L"],
                               "n_boxes": c["n_boxes"], "n_chars": c["n_chars"], "ignore_area": c["ignore_area"],
                               "requested_char_fraction": c["requested_char_fraction"],
                               "source": "ocr_screens_v2_described_with_images", "boxes": "Google Cloud Vision lines, exact-match verified by pinned PP-OCRv5",
                               "drafter": "Qwen/Qwen3.6-27B-FP8@e89b16eb", "human_review": "pending"})
        (out / split / name).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in rows))
        (out / "private" / f"provenance-{'val' if prefix == 'validation' else 'test'}.jsonl").write_text(
            "".join(json.dumps(r) + "\n" for r in provenance))
        n = max(1, len(rows))
        types = {}
        for r in rows:
            types[r["screen_type"]] = types.get(r["screen_type"], 0) + 1
        stats[split] = {"samples": len(rows), "screen_types": types,
                        "mean_real_T": sum(p["real_T"] for p in provenance) / n,
                        "mean_real_L": sum(p["real_L"] for p in provenance) / n,
                        "mean_boxes": sum(p["n_boxes"] for p in provenance) / n,
                        "mean_questions": sum(len(r["questions"]) for r in rows) / n}
    shutil.copyfile(HERE / "pipeline/static/SCHEMA.md", out / "public/SCHEMA.md")
    shutil.copyfile(STARTING_ARTIFACT, out / "public/STARTING_ARTIFACT.json")
    print(json.dumps({"verified_and_validated": len(keys), "after_dedup": len(kept), "available_by_type": available,
                      "stats": stats}, indent=1))


if __name__ == "__main__":
    main()
