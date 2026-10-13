"""Trusted, offline RGB LPIPS-Alex v0.1 and sequence FloLPIPS/PWC metrics.

Not imported by the stdlib-only public HTTP client. Weights must be supplied in a
hash-locked organizer bundle; no downloads or random-weight fallback at runtime.
"""
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.nn import functional as F

from vfi_protocol import media_path, validate_png
from vfi_pwc import PWCNet
from vfi_scoring import PROTOCOL

ASSETS = {"alexnet", "lpips_alex", "pwcnet"}


def file_sha256(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def code_sha256():
    root = Path(__file__).parent
    names = ("vfi_metrics.py", "vfi_pwc.py", "vfi_scoring.py", "vfi_evaluate.py", "vfi_protocol.py", "vfi_selection.py",
             "vfi_runtime.py", "vfi_gpu_worker.py")
    content = {name: file_sha256(root / name) for name in names}
    return hashlib.sha256(json.dumps(content, sort_keys=True).encode()).hexdigest()


def bundle_files(bundle, expected_sha256=None):
    root = Path(bundle).resolve()
    lock_path = root / "lock.json"
    if expected_sha256 is not None and file_sha256(lock_path) != expected_sha256:
        raise ValueError("Metric bundle lock differs from the frozen digest")
    lock = json.loads(lock_path.read_text())
    if lock.get("protocol") != PROTOCOL or set(lock.get("files", {})) != ASSETS:
        raise ValueError("Metric bundle must lock AlexNet, LPIPS v0.1 and PWC-Net")
    files = {}
    for name, record in lock["files"].items():
        path = media_path(root, record["path"])
        if file_sha256(path) != record["sha256"]:
            raise ValueError(f"Metric weight hash mismatch: {name}")
        files[name] = path
    return files


class AlexFeatures(nn.Module):
    """Torchvision AlexNet feature topology; no classifier or implicit hub download."""
    def __init__(self):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, 11, stride=4, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(64, 192, 5, padding=2), nn.ReLU(), nn.MaxPool2d(3, 2),
            nn.Conv2d(192, 384, 3, padding=1), nn.ReLU(),
            nn.Conv2d(384, 256, 3, padding=1), nn.ReLU(),
            nn.Conv2d(256, 256, 3, padding=1), nn.ReLU())

    def forward(self, image):
        outputs = []
        for index, layer in enumerate(self.features):
            image = layer(image)
            if index in (1, 4, 7, 9, 11):
                outputs.append(image / (image.square().sum(1, keepdim=True).sqrt() + 1e-10))
        return outputs


class PerceptualMaps(nn.Module):
    def __init__(self, alexnet, calibration):
        super().__init__()
        self.trunk = AlexFeatures()
        state = torch.load(alexnet, map_location="cpu", weights_only=True)
        self.trunk.load_state_dict({key: value for key, value in state.items() if key.startswith("features.")}, strict=True)
        learned = torch.load(calibration, map_location="cpu", weights_only=True)
        expected = {f"lin{i}.model.1.weight" for i in range(5)}
        if set(learned) != expected:
            raise ValueError("Expected exactly the five official LPIPS-Alex v0.1 linear weights")
        self.linear = nn.ModuleList([nn.Conv2d(c, 1, 1, bias=False) for c in (64, 192, 384, 256, 256)])
        for i, layer in enumerate(self.linear):
            weight = learned[f"lin{i}.model.1.weight"]
            if not torch.isfinite(weight).all() or (weight < 0).any():
                raise ValueError("Invalid LPIPS calibration weights")
            layer.load_state_dict({"weight": weight}, strict=True)
        self.register_buffer("shift", torch.tensor([-.030, -.088, -.188])[None, :, None, None])
        self.register_buffer("scale", torch.tensor([.458, .448, .450])[None, :, None, None])

    def forward(self, reference, prediction):
        features = [self.trunk((2 * image - 1 - self.shift) / self.scale) for image in (reference, prediction)]
        return [layer((left - right).square()) for layer, left, right in zip(self.linear, *features)]


def spatial_lpips(maps):
    return sum(layer.mean((2, 3)) for layer in maps).mean()


def motion_weighted_lpips(maps, flow_difference):
    """Reference FloLPIPS weighting; uniform spatial weights for exactly zero flow difference."""
    if not torch.isfinite(flow_difference).all():
        raise ValueError("Nonfinite optical flow")
    total = 0
    for layer in maps:
        flow = F.interpolate(flow_difference, size=layer.shape[-2:], mode="bilinear", align_corners=False)
        magnitude = flow.square().sum(1, keepdim=True).sqrt()
        mass = magnitude.sum((2, 3), keepdim=True)
        # Avoid upstream's 0/0 on perfect matches and static/identical flow.
        weights = torch.where(mass > 0, magnitude / mass.clamp_min(torch.finfo(mass.dtype).tiny),
                              torch.full_like(magnitude, 1 / (layer.shape[-2] * layer.shape[-1])))
        total = total + (layer * weights).sum((2, 3))
    return total.mean()


def read_rgb(path, width, height, device):
    blob = Path(path).read_bytes()
    if Path(path).suffix == ".rgb":
        if len(blob) != width * height * 3:
            raise ValueError("Raw RGB frame has the wrong byte length")
        pixels = np.frombuffer(blob, dtype=np.uint8).reshape(height, width, 3).copy()
    else:
        validate_png(blob, width, height)
        with Image.open(path) as image:
            pixels = np.array(image, dtype=np.uint8, copy=True)
    return torch.from_numpy(pixels).permute(2, 0, 1).unsqueeze(0).to(device=device, dtype=torch.float32) / 255


class MetricEngine:
    def __init__(self, bundle, *, expected_sha256=None, device="cpu"):
        if device not in ("cpu", "cuda"):
            raise ValueError("Metrics support explicit cpu or cuda only")
        files = bundle_files(bundle, expected_sha256)
        self.device = device
        torch.set_num_threads(8)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        self.perceptual = PerceptualMaps(files["alexnet"], files["lpips_alex"]).eval().requires_grad_(False).to(device)
        self.flow = PWCNet()
        state = torch.load(files["pwcnet"], map_location="cpu", weights_only=True)
        self.flow.load_state_dict({key.replace("module", "net"): value for key, value in state.items()}, strict=True)
        self.flow.eval().requires_grad_(False).to(device)
        for model in (self.perceptual, self.flow):
            if any(not torch.isfinite(p).all() for p in model.parameters()):
                raise ValueError("Nonfinite metric weights")

    @torch.inference_mode()
    def interval(self, references, predictions, width, height, *, check_deadline=lambda: None):
        if len(references) != len(predictions) or len(references) not in (5, 9, 17):
            raise ValueError("Metrics require a complete 4x/8x/16x interval")
        if min(width, height) < 32:
            raise ValueError("LPIPS-Alex requires both native dimensions >=32; no resizing is allowed")
        read = lambda path: read_rgb(path, width, height, self.device)
        reference, prediction = read(references[0]), read(predictions[0])
        lpips, temporal, psnr = [], [], []
        # Each pair belongs to this interval, including transitions adjacent to endpoints.
        for i in range(len(references) - 1):
            check_deadline()
            next_reference, next_prediction = read(references[i + 1]), read(predictions[i + 1])
            maps = self.perceptual(reference, prediction)
            difference = self.flow(reference, next_reference) - self.flow(prediction, next_prediction)
            temporal.append(float(motion_weighted_lpips(maps, difference)))
            if i > 0:  # anchors never enter the framewise LPIPS/PSNR means
                lpips.append(float(spatial_lpips(maps)))
                mse = float((reference.double() - prediction.double()).square().mean())
                psnr.append(-10 * math.log10(max(mse, 1e-10)))
            reference, prediction = next_reference, next_prediction
        check_deadline()
        result = {"lpips": math.fsum(lpips) / len(lpips), "flolpips": math.fsum(temporal) / len(temporal),
                  "psnr": math.fsum(psnr) / len(psnr)}
        if any(not math.isfinite(v) or v < 0 for v in result.values()):
            raise ValueError("Invalid learned metric output")
        return result
