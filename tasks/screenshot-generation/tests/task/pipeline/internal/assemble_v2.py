"""Assemble the v2 data: reference lines from the pinned OCR, text-free descriptions, few public samples.

Selection: reference_ocr.py page filters pass (dual-OCR agreement; statuses are recomputed, so an older
personal-identifier flag does not drop a page), a safe and parseable draft, and a description that leaks no
on-screen text (checked against the new reference lines). One frame per source video (already); identical
text sets and near-identical thumbnails are dropped. A pool of n_val + n_test frames is drawn stratified by the
drafter's screen_type with fixed quotas (shortfalls filled from other types) in salted order; the public
validation set is a seeded random sample of that pool and the rest is held-out (no overlap).
Provenance goes only to private/provenance-*.jsonl.

Usage: python assemble_v2.py --reference R0.jsonl R1.jsonl --drafts D.jsonl --raw RAW --out DIR --n-val 25 --n-test 500
"""
import argparse
import hashlib
import json
import shutil
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "pipeline/dataset"))
sys.path.insert(0, str(HERE / "pipeline/internal"))
from validate_questions import leaks  # noqa: E402
from reference_ocr import page_status  # noqa: E402

QUOTAS = {"web_page": 0.35, "desktop_app": 0.35, "ide_terminal": 0.12, "mobile": 0.06, "other": 0.12}
ALIASES = {"web_app": "web_page", "website": "web_page", "desktop": "desktop_app", "terminal": "ide_terminal", "ide": "ide_terminal"}
SALT_FILE = Path("/scratch/frontier-ml-workspace/work/screenshot-generation/internal/salt.txt")


def screen_type(draft):
    t = ALIASES.get(draft.get("screen_type"), draft.get("screen_type"))
    return t if t in QUOTAS else "other"


def load(paths):
    rows = {}
    for path in paths:
        for line in open(path):
            if line.strip():
                r = json.loads(line)
                if not r["status"].startswith("error"):
                    rows[r["key"]] = r
    return rows


def thumbnail(path):
    from PIL import Image
    import numpy as np
    return np.asarray(Image.open(path).convert("L").resize((64, 36)), dtype=np.float32) / 255.0


def stratified(pool, n):
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
    for t in QUOTAS:
        pool[t] = pool.get(t, [])[len(pick[t]):]
    return [k for t in QUOTAS for k in pick[t]]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", nargs="+", required=True)
    parser.add_argument("--drafts", required=True)
    parser.add_argument("--raw", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--n-val", type=int, default=25)
    parser.add_argument("--n-test", type=int, default=500)
    parser.add_argument("--seed", type=int, default=20260924, help="seed of the public validation sample")
    args = parser.parse_args()
    salt = SALT_FILE.read_text().strip()
    ref = load(args.reference)
    drafts = {json.loads(l)["key"]: json.loads(l) for l in open(args.drafts) if l.strip()}
    counts = {"reference_total": len(ref), "reference_ok": 0, "flagged_personal_identifier_kept": 0, "no_draft": 0,
              "unsafe": 0, "description_leak": 0}
    keys = []
    for k, r in ref.items():
        if r["status"] == "personal_identifier":
            counts["flagged_personal_identifier_kept"] += 1
        if "n_requested" not in r or page_status(r) != "ok":
            continue
        counts["reference_ok"] += 1
        d = drafts.get(k)
        if not d or d.get("status") != "ok" or not isinstance(d.get("description"), str):
            counts["no_draft"] += 1
            continue
        if d.get("unsafe") is not False:
            counts["unsafe"] += 1
            continue
        if leaks(d["description"], r["texts"], None):
            counts["description_leak"] += 1
            continue
        keys.append(k)
    keys.sort(key=lambda k: hashlib.sha256(f"{salt}:order:{k}".encode()).hexdigest())
    kept, thumbs, seen_text = [], [], set()
    for k in keys:
        text_key = hashlib.sha256(json.dumps(sorted(t["text"] for t in ref[k]["texts"])).encode()).hexdigest()
        th = thumbnail(Path(args.raw) / f"{k}.png")
        if text_key in seen_text or any(float(abs(th - o).mean()) < 0.01 for o in thumbs):
            continue
        seen_text.add(text_key)
        thumbs.append(th)
        kept.append(k)
    pool = {}
    for k in kept:
        pool.setdefault(screen_type(drafts[k]), []).append(k)
    available = {t: len(v) for t, v in pool.items()}
    import random
    chosen = stratified(pool, args.n_val + args.n_test)
    val_set = set(random.Random(args.seed).sample(chosen, args.n_val))
    val_keys = [k for k in chosen if k in val_set]
    test_keys = [k for k in chosen if k not in val_set]
    out = Path(args.out)
    for stale in ("public/media", "private/images"):
        if (out / stale).exists():
            shutil.rmtree(out / stale)
    for stale in ("public/cmmd_reference.json", "public/real_clip_embeddings.npy", "public/STARTING_ARTIFACT.json",
                  "private/cmmd_reference.json", "private/real_clip_embeddings.npy"):
        (out / stale).unlink(missing_ok=True)
    (out / "public/media").mkdir(parents=True, exist_ok=True)
    (out / "private/images").mkdir(parents=True, exist_ok=True)
    stats = {}
    for split, keys_, prefix, name in (("public", val_keys, "validation", "validation.jsonl"),
                                       ("private", test_keys, "test", "test.jsonl")):
        rows, provenance = [], []
        for i, k in enumerate(keys_, 1):
            r, d = ref[k], drafts[k]
            sid = f"{prefix}-{i:04d}"
            image = f"media/{i:06d}.png" if split == "public" else f"images/{sid}.png"
            shutil.copyfile(Path(args.raw) / f"{k}.png", out / split / image)
            rows.append({"id": sid, "description": d["description"], "screen_type": screen_type(d),
                         "texts": r["texts"], "ignore_regions": r["ignore_regions"], "size": "1280x720", "image": image})
            provenance.append({"id": sid, "key": k, "video_id": r["video_id"], "screen_type": screen_type(d),
                               "page_type": d.get("page_type"), "real_T": r["real_T"], "real_L": r["real_L"],
                               "n_boxes": r["n_requested"], "n_chars": r["requested_chars"],
                               "requested_char_fraction": r["requested_char_fraction"], "ignore_area": r["ignore_area"],
                               "source": "ocr_screens_v2_described_with_images",
                               "texts": "pinned PP-OCRv5 detections on the real screenshot, confirmed by Google Cloud Vision and the metric crop recognizer",
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
                        "min_real_L": min((p["real_L"] for p in provenance), default=0),
                        "mean_boxes": sum(p["n_boxes"] for p in provenance) / n}
    shutil.copyfile(HERE / "pipeline/static/SCHEMA.md", out / "public/SCHEMA.md")
    print(json.dumps({"counts": counts, "eligible": len(keys), "after_dedup": len(kept), "available_by_type": available,
                      "stats": stats}, indent=1))


if __name__ == "__main__":
    main()
