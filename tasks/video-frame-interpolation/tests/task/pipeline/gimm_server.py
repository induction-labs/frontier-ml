"""Organizer baseline: frozen upstream GIMM-VFI-R-P behind the task HTTP interface.

The upstream implementation and checkpoints are acquired separately; see BASELINE.md.
No training, augmentation, resizing, reference access, or cross-request image cache.
"""
import argparse
import base64
from http.server import BaseHTTPRequestHandler, HTTPServer
import io
import json
import os
from pathlib import Path
import sys
import traceback

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from vfi_protocol import MAX_JSON_BYTES, REQUEST_KEYS, RGB8_CONTENT_TYPE, validate_png, validate_timeline


def decode_request(payload):
    import numpy as np
    from PIL import Image

    validate_timeline(payload, image_field="png")
    if set(payload) != REQUEST_KEYS:
        raise ValueError("Unexpected request fields")
    images = []
    for frame in payload["input_frames"]:
        blob = base64.b64decode(frame["png"], validate=True)
        validate_png(blob, payload["width"], payload["height"])
        with Image.open(io.BytesIO(blob)) as image:
            images.append(np.array(image, dtype=np.uint8, copy=True))
    return images


def quantize_prediction(image):
    import numpy as np

    if image.ndim != 3 or image.shape[2] != 3 or not np.isfinite(image).all():
        raise ValueError("Model returned invalid RGB pixels")
    # Match upstream SNU_FILM_arb.py's PNG export: clip, multiply, truncate.
    return (np.clip(image, 0, 1) * 255).astype(np.uint8)


def encode_prediction(image):
    from PIL import Image

    pixels = quantize_prediction(image)
    stream = io.BytesIO()
    Image.fromarray(pixels).save(stream, format="PNG", compress_level=1)
    return base64.b64encode(stream.getvalue()).decode("ascii")


class GIMM:
    def __init__(self, source):
        import numpy as np
        import torch
        from omegaconf import OmegaConf

        if not torch.cuda.is_available():
            raise RuntimeError("GIMM baseline requires CUDA")
        source = Path(source).resolve()
        # RAFT's initializer resolves its checkpoint relative to the upstream root.
        os.chdir(source)
        sys.path.insert(0, str(source / "src"))
        from models import create_model
        from models.generalizable_INR.configs import GIMMVFIConfig

        torch.set_num_threads(8)
        torch.manual_seed(0)
        np.random.seed(0)
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        torch.backends.cudnn.benchmark = False
        config = OmegaConf.load(source / "configs/gimmvfi/gimmvfi_r_arb.yaml")
        arch = GIMMVFIConfig.create(config.arch)
        model, _ = create_model(arch, ema=False)
        checkpoint = torch.load(source / "pretrained_ckpt/gimmvfi_r_arb_lpips.pt",
                                map_location="cpu", weights_only=True)
        state = checkpoint["state_dict"]
        if not all(torch.isfinite(value).all() for value in state.values()):
            raise ValueError("Nonfinite baseline checkpoint")
        model.load_state_dict(state, strict=True)
        self.model = model.eval().requires_grad_(False).cuda()
        # Compile fixed kernels before /health. No evaluation frame is used here.
        rng = np.random.default_rng(0)
        self.predict([rng.integers(0, 256, (128, 128, 3), dtype=np.uint8) for _ in range(2)],
                     list(range(1, 16)), 16)

    def predict(self, images, indices, factor):
        import torch
        from torch.nn import functional as F

        height, width = images[0].shape[:2]
        dh, dw = (-height) % 32, (-width) % 32
        padding = (dw // 2, dw - dw // 2, dh // 2, dh - dh // 2)
        with torch.inference_mode():
            endpoints = [F.pad(torch.from_numpy(im).permute(2, 0, 1).unsqueeze(0).float() / 255,
                               padding, mode="replicate") for im in images]
            xs = torch.stack(endpoints, dim=2).cuda()
            times = [index / factor for index in indices]
            coords = [(self.model.sample_coord_input(1, xs.shape[-2:], [t], device=xs.device), None)
                      for t in times]
            result = self.model(xs, coords, t=[torch.tensor([t], device=xs.device) for t in times])
            outputs = result["imgt_pred"]
            if len(outputs) != len(indices):
                raise ValueError("Incomplete GIMM output")
            predictions = [im[0, :, padding[2]:padding[2] + height, padding[0]:padding[0] + width]
                           .permute(1, 2, 0).contiguous().cpu().numpy() for im in outputs]
            torch.cuda.synchronize()
        return predictions

    def interpolate(self, payload):
        images = decode_request(payload)
        predictions = self.predict(images, payload["target_indices"], payload["frame_count"] - 1)
        return {"frames": [{"index": index, "png": encode_prediction(image)}
                           for index, image in zip(payload["target_indices"], predictions, strict=True)]}

    def interpolate_raw(self, payload):
        images = decode_request(payload)
        predictions = self.predict(images, payload["target_indices"], payload["frame_count"] - 1)
        return b"".join(quantize_prediction(image).tobytes() for image in predictions)


def serve(model, host, port):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def reply(self, status, value):
            body = json.dumps(value, allow_nan=False).encode()
            if len(body) > MAX_JSON_BYTES:
                raise ValueError("Oversized response")
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self.reply(200 if self.path == "/health" else 404, {"ready": True})

        def do_POST(self):
            if self.path != "/interpolate":
                self.reply(404, {"error": "unknown endpoint"})
                return
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= MAX_JSON_BYTES:
                    raise ValueError("Invalid request length")
                payload = json.loads(self.rfile.read(size))
                if self.headers.get("Accept") == RGB8_CONTENT_TYPE:
                    body = model.interpolate_raw(payload)
                    self.send_response(200)
                    self.send_header("Content-Type", RGB8_CONTENT_TYPE)
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.reply(200, model.interpolate(payload))
            except Exception as exc:
                traceback.print_exc()
                self.reply(500, {"error": f"{type(exc).__name__}: {exc}"})

    with HTTPServer((host, port), Handler) as server:
        server.serve_forever()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    serve(GIMM(args.source), args.host, args.port)
