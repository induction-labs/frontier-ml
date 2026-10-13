"""Organizer baselines for ocr-latent-compression, scored with the public metric (no HTTP).

    CUDA_VISIBLE_DEVICES=2 python measure_baselines.py --data data/.../public/validation.jsonl \
        --train-dir TRAIN_PNGS --out results.json

Baselines:
* ceiling_tiny / ceiling_base: uncapped DeepSeek-OCR vision tokens (Tiny 64 tokens ~ 80 KB fp8
  plus separators; Base 256 tokens), frozen decoder. Not compliant (far over 2 KB).
* pca_k{K}_b{B}: per-token PCA of Tiny projector outputs (fit on organizer training renders),
  uniform scalar quantization, 64*K*B bits = 2048 bytes, frozen decoder. Compliant.
* jpeg2k_* / webp2k_*: the page image itself compressed to <= 2048 bytes, decoded and fed to
  the uncapped model. Non-compliant reference showing what 2 KB of pixels carries.
"""
from __future__ import annotations

import argparse
import io
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE / "pca_submission"))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from codec import dequantize, pack, quantize, unpack  # noqa: E402
from deepseek_ocr_ref import VisionDecoder, VisionEncoder, load_model, preprocess  # noqa: E402
from ocr_text import ned, normalize, word_ned  # noqa: E402

BUCKETS = ("sparse", "medium", "dense", "very_dense")
PCA_CONFIGS = ((32, 8), (64, 4), (128, 2), (256, 1))


def summarize(rows, predictions):
    records = []
    for row, text in zip(rows, predictions):
        p, r = normalize(text), normalize(row["text"])
        records.append({"id": row["id"], "kind": row["kind"], "bucket": row["bucket"],
                        "value": 1 - word_ned(p, r), "char": 1 - ned(p, r)})
    mean = lambda rs, key="value": round(sum(r[key] for r in rs) / len(rs), 4) if rs else None
    return {"score": mean(records), "char_similarity": mean(records, "char"),
            "char_by_bucket": {b: mean([r for r in records if r["bucket"] == b], "char") for b in BUCKETS},
            "by_bucket": {b: mean([r for r in records if r["bucket"] == b]) for b in BUCKETS},
            "by_kind": {k: mean([r for r in records if r["kind"] == k]) for k in ("document", "screenshot")},
            "by_kind_bucket": {f"{k}/{b}": mean([r for r in records if r["kind"] == k and r["bucket"] == b])
                               for k in ("document", "screenshot") for b in BUCKETS}}


def features(encoder, images, mode, batch=16):
    out = []
    for start in range(0, len(images), batch):
        out.append(encoder.features(torch.stack([preprocess(i, mode) for i in images[start:start + batch]])).float().cpu())
    return torch.cat(out)


def decode_all(decoder, grids, batch=50, max_new_tokens=1800, lengths=None):
    """Greedy decode; grids are processed in order of expected output length to limit padding waste."""
    order = sorted(range(len(grids)), key=lambda i: lengths[i]) if lengths else list(range(len(grids)))
    texts = [None] * len(grids)
    for start in range(0, len(order), batch):
        chunk = order[start:start + batch]
        for index, text in zip(chunk, decoder.generate([grids[i] for i in chunk], max_new_tokens=max_new_tokens)):
            texts[index] = text
    return texts


def fit_pca(train_features, k_max):
    tokens = train_features.reshape(-1, train_features.shape[-1]).double()
    mean = tokens.mean(0)
    _, _, vh = torch.linalg.svd(tokens - mean, full_matrices=False)
    return mean.float(), vh[:k_max].float()


def pca_roundtrip(val_features, train_features, mean, basis, k, bits):
    b = basis[:k]
    train_coef = ((train_features.reshape(-1, mean.shape[0]) - mean) @ b.T).numpy()
    low, high = np.percentile(train_coef, 0.1, axis=0), np.percentile(train_coef, 99.9, axis=0)
    grids, sizes = [], []
    for feats in val_features:
        coef = ((feats - mean) @ b.T).numpy()
        payload = pack(quantize(coef, low, high, bits), bits)
        sizes.append(len(payload))
        codes = unpack(payload, coef.size, bits).reshape(coef.shape)
        grids.append(torch.from_numpy(dequantize(codes, low, high, bits)).float() @ b + mean)
    return grids, max(sizes)


def compress_to_budget(image, fmt, budget=2048, min_quality=30):
    """Largest grayscale resolution (aspect kept) whose best quality >= min_quality fits the budget."""
    gray = image.convert("L")
    widths = [512, 448, 384, 320, 256, 224, 192, 160, 128, 96, 64]
    fallback = None
    for width in widths:
        height = max(8, round(gray.height * width / gray.width))
        small = gray.resize((width, height), Image.LANCZOS)
        lo, hi, best = 1, 95, None
        while lo <= hi:
            quality = (lo + hi) // 2
            buffer = io.BytesIO()
            small.save(buffer, format=fmt, quality=quality)
            if buffer.tell() <= budget:
                best, lo = (quality, buffer.getvalue()), quality + 1
            else:
                hi = quality - 1
        if best and best[0] >= min_quality:
            return best[1], width, best[0]
        if best and fallback is None:
            fallback = (best[1], width, best[0])
    return fallback


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--limit", type=int)
    parser.add_argument("--only", nargs="*", help="subset of baseline names")
    parser.add_argument("--batch", type=int, default=50, help="decode batch size (memory vs speed)")
    args = parser.parse_args()
    rows = [json.loads(line) for line in args.data.read_text().splitlines() if line.strip()][:args.limit]
    images = [Image.open(args.data.parent / r["image"]).convert("RGB") for r in rows]
    tokenizer, model = load_model()
    encoder, decoder = VisionEncoder(model), VisionDecoder(model, tokenizer)
    results = json.loads(args.out.read_text()) if args.out.exists() else {}
    wanted = lambda name: not args.only or any(name.startswith(o) for o in args.only)

    def record(name, grids, extra=None):
        started = time.time()
        texts = decode_all(decoder, grids, batch=args.batch, lengths=[len(r["text"]) for r in rows])
        predictions = args.out.with_name(args.out.stem + ".predictions.jsonl")
        with predictions.open("a") as stream:
            for row, text in zip(rows, texts):
                stream.write(json.dumps({"baseline": name, "id": row["id"], "prediction": text}) + "\n")
        results[name] = {**summarize(rows, texts), **(extra or {}), "decode_seconds": round(time.time() - started, 1),
                         "samples": len(rows), "examples": [{"id": r["id"], "bucket": r["bucket"], "prediction": t[:400]}
                                                             for r, t in list(zip(rows, texts))[:4]]}
        args.out.write_text(json.dumps(results, indent=2))
        print(name, results[name]["score"], results[name]["by_bucket"], flush=True)

    tiny = features(encoder, images, "tiny")
    if wanted("ceiling_tiny"):
        record("ceiling_tiny", list(tiny), {"payload_bytes_fp8_equivalent": 64 * 1280})
    if wanted("ceiling_base"):
        record("ceiling_base", list(features(encoder, images, "base")), {"payload_bytes_fp8_equivalent": 256 * 1280})
    if any(wanted(f"pca_k{k}_b{b}") for k, b in PCA_CONFIGS):
        train_images = [Image.open(p).convert("RGB") for p in sorted(args.train_dir.glob("*.png"))]
        train = features(encoder, train_images, "tiny")
        mean, basis = fit_pca(train, max(k for k, _ in PCA_CONFIGS))
        for k, bits in PCA_CONFIGS:
            name = f"pca_k{k}_b{bits}"
            if wanted(name):
                grids, max_bytes = pca_roundtrip(list(tiny), train, mean, basis, k, bits)
                record(name, grids, {"payload_bytes_max": max_bytes, "train_images": len(train_images)})
    for fmt in ("JPEG", "WEBP"):
        name = f"{fmt.lower()}2k"
        if not any(wanted(f"{name}_{m}") for m in ("tiny", "base")):
            continue
        decoded, meta = [], []
        for image in images:
            data, width, quality = compress_to_budget(image, fmt)
            meta.append((len(data), width, quality))
            decoded.append(Image.open(io.BytesIO(data)).convert("RGB").resize(image.size, Image.BICUBIC))
        extra = {"payload_bytes_max": max(m[0] for m in meta), "mean_width": float(np.mean([m[1] for m in meta])),
                 "mean_quality": float(np.mean([m[2] for m in meta]))}
        for mode in ("tiny", "base"):
            if wanted(f"{name}_{mode}"):
                record(f"{name}_{mode}", list(features(encoder, decoded, mode)), extra)


if __name__ == "__main__":
    main()
