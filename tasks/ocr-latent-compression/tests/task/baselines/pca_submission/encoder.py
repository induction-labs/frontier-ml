"""/encode path of the organizer PCA baseline.

DeepEncoder in Tiny mode (512x512 resize -> SAM-B + CLIP-L -> linear projector, 64 x 1280
features), then a per-token PCA to k coefficients and fixed-rate uniform quantization
to ``bits`` bits each. The payload is exactly the packed codes. No text is produced.
"""
import io
from pathlib import Path

import numpy as np
import torch
from easydict import EasyDict
from PIL import Image
from safetensors import safe_open
from safetensors.torch import load_file

from codec import pack, quantize
from deepencoder import MlpProjector, build_clip_l, build_sam_vit_b

HERE = Path(__file__).resolve().parent
BASE = HERE / "weights" / "model-00001-of-000001.safetensors"
PCA = HERE / "weights" / "pca_tiny.safetensors"
SIZE = 512
_models = {}


def load_deepencoder(device="cuda"):
    sam, clip = build_sam_vit_b(), build_clip_l()
    projector = MlpProjector(EasyDict(projector_type="linear", input_dim=2048, n_embed=1280))
    parts = {"model.sam_model.": sam, "model.vision_model.": clip, "model.projector.": projector}
    with safe_open(BASE, framework="pt") as checkpoint:
        names = list(checkpoint.keys())
        for prefix, module in parts.items():
            state = {name[len(prefix):]: checkpoint.get_tensor(name) for name in names if name.startswith(prefix)}
            missing, unexpected = module.load_state_dict(state, strict=False)
            if unexpected or [m for m in missing if not m.endswith("position_ids")]:
                raise RuntimeError(f"DeepEncoder weights do not match for {prefix}: {missing[:4]} {unexpected[:4]}")
    return [m.to(device=device, dtype=torch.bfloat16).requires_grad_(False) for m in (sam, clip, projector)]


def models():
    if not _models:
        sam, clip, projector = load_deepencoder()
        pca = load_file(PCA)
        _models.update(sam=sam, clip=clip, projector=projector,
                       mean=pca["mean"].float().cuda(), basis=pca["basis"].float().cuda(),
                       low=pca["low"].numpy(), high=pca["high"].numpy(), bits=int(pca["bits"].item()))
    return _models


def pixels(image_bytes):
    with Image.open(io.BytesIO(image_bytes)) as image:
        image = image.convert("RGB").resize((SIZE, SIZE), Image.BICUBIC)
    array = torch.from_numpy(np.asarray(image, dtype=np.float32) / 255.0).permute(2, 0, 1)
    return (array - 0.5) / 0.5


@torch.no_grad()
def features(batch):
    m = models()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        sam = m["sam"](batch.cuda().to(torch.bfloat16))
        clip = m["clip"](batch.cuda().to(torch.bfloat16), sam)
        return m["projector"](torch.cat((clip[:, 1:], sam.flatten(2).permute(0, 2, 1)), dim=-1)).float()


def encode(images):
    """list of PNG/JPEG bytes -> list of payload bytes."""
    m = models()
    feats = features(torch.stack([pixels(image) for image in images]))
    coefficients = ((feats - m["mean"]) @ m["basis"].T).cpu().numpy()
    return [pack(quantize(c, m["low"], m["high"], m["bits"]), m["bits"]) for c in coefficients]
