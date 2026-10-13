"""Fit the organizer PCA baseline and assemble a contract-compliant output/ directory.

    CUDA_VISIBLE_DEVICES=2 python make_pca_submission.py --train-dir TRAIN_PNGS --out SUBMISSION --k 64 --bits 4

The base checkpoint is hard-linked (a regular file, byte-identical to the artifact). The
result is used for the local end-to-end check of validate.py, the verifier and the static
checks; its serve.sh relies on $OCR_PYTHON instead of vendored wheels, so it is not an
offline-image submission.
"""
import argparse
from datetime import datetime, timezone
import json
import os
from pathlib import Path
import shutil
import sys

import numpy as np
import torch
from PIL import Image
from safetensors.torch import save_file

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
from deepseek_ocr_ref import DEFAULT_WEIGHTS, VisionEncoder, load_model  # noqa: E402
from measure_baselines import features, fit_pca  # noqa: E402

BASE_FILES = ["model-00001-of-000001.safetensors", "model.safetensors.index.json", "config.json",
              "configuration_deepseek_v2.py", "conversation.py", "deepencoder.py", "modeling_deepseekocr.py",
              "modeling_deepseekv2.py", "processor_config.json", "special_tokens_map.json", "tokenizer.json",
              "tokenizer_config.json", "LICENSE"]


def link_or_copy(source, target):
    try:
        os.link(source, target)
    except OSError:
        shutil.copy2(source, target)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-dir", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--k", type=int, default=64)
    parser.add_argument("--bits", type=int, default=4)
    args = parser.parse_args()
    if 64 * args.k * args.bits > 2048 * 8:
        raise SystemExit("64 * k * bits must fit in 2048 bytes")
    out = args.out
    if out.exists():
        shutil.rmtree(out)
    (out / "weights").mkdir(parents=True)
    tokenizer, model = load_model()
    train_paths = sorted(args.train_dir.glob("*.png"))
    train = features(VisionEncoder(model), [Image.open(p).convert("RGB") for p in train_paths], "tiny")
    mean, basis = fit_pca(train, args.k)
    coef = ((train.reshape(-1, mean.shape[0]) - mean) @ basis.T).numpy()
    low, high = np.percentile(coef, 0.1, axis=0), np.percentile(coef, 99.9, axis=0)
    save_file({"mean": mean.contiguous(), "basis": basis.contiguous(), "low": torch.from_numpy(low).float(),
               "high": torch.from_numpy(high).float(), "bits": torch.tensor([args.bits], dtype=torch.int32)},
              str(out / "weights" / "pca_tiny.safetensors"))
    weights = Path(DEFAULT_WEIGHTS)
    for name in BASE_FILES:
        link_or_copy(weights / name, out / "weights" / name)
    shutil.copy2(weights / "deepencoder.py", out / "deepencoder.py")
    for name in ("codec.py", "encoder.py", "decoder.py", "server.py", "serve.sh"):
        shutil.copy2(HERE / "pca_submission" / name, out / name)
    (out / "serve.sh").chmod(0o755)
    shutil.copy2(Path(__file__), out / "make_pca_submission.py")
    shutil.copy2(HERE / "measure_baselines.py", out / "measure_baselines.py")
    shutil.copy2(HERE / "deepseek_ocr_ref.py", out / "deepseek_ocr_ref.py")
    log = {"created": datetime.now(timezone.utc).isoformat(), "train_images": len(train_paths), "k": args.k,
           "bits": args.bits, "payload_bytes": 64 * args.k * args.bits // 8, "mode": "tiny",
           "explained_variance_note": "per-token PCA over 64 x 1280 projector outputs"}
    (out / "training_log.json").write_text(json.dumps(log, indent=2) + "\n")
    architecture = {"base_model": "deepseek-ai/DeepSeek-OCR",
                    "base_revision": "9f30c71f441d010e5429c532364a86705536c53a",
                    "training_evidence_files": ["make_pca_submission.py", "measure_baselines.py", "deepseek_ocr_ref.py",
                                                "training_log.json"],
                    "inference_evidence_files": ["serve.sh", "server.py", "encoder.py", "codec.py", "decoder.py",
                                                 "deepencoder.py"]}
    (out / "architecture.json").write_text(json.dumps(architecture, indent=2) + "\n")
    (out / "README.md").write_text(
        "Organizer PCA baseline: DeepEncoder Tiny features -> per-token PCA (k=%d) -> %d-bit uniform codes "
        "(%d bytes). /decode dequantizes into the 64 vision-token positions of the frozen DeepSeek-OCR decoder.\n"
        % (args.k, args.bits, 64 * args.k * args.bits // 8))
    print(json.dumps(log))


if __name__ == "__main__":
    main()
