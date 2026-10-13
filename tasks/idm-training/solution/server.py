"""Standard-library interface smoke fixture; contains no trained model.

Endpoints return trivial well-formed responses. Tasks with architecture or training
requirements still reject this fixture before scoring; it is not a compliant model.
"""
import base64
import json
import math
import os
import struct
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer


def blank_png(width=1280, height=720):
    """A white 8-bit RGB PNG built with the standard library (image-generation endpoints)."""
    def chunk(kind, data):
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data) & 0xFFFFFFFF)
    rows = b"".join(b"\x00" + b"\xff" * (3 * width) for _ in range(height))
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(rows, 9)) + chunk(b"IEND", b""))


def flac_seconds(blob):
    """Audio duration from a FLAC STREAMINFO block (end-of-turn); 0.0 when the bytes are not a FLAC stream."""
    if blob[:4] != b"fLaC":
        return 0.0
    offset = 4
    while offset + 4 <= len(blob):
        header, length = blob[offset], int.from_bytes(blob[offset + 1:offset + 4], "big")
        if header & 0x7F == 0 and length == 34:  # STREAMINFO: sample rate (20 bits) ... total samples (36 bits)
            bits = int.from_bytes(blob[offset + 14:offset + 22], "big")
            return (bits & ((1 << 36) - 1)) / (bits >> 44) if bits >> 44 else 0.0
        if header & 0x80:
            break
        offset += 4 + length
    return 0.0


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def respond(self, status, body):
        payload = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(payload)))
        self.end_headers()
        self.wfile.write(payload)

    def do_GET(self):
        if self.path == "/health":
            self.respond(200, {"status": "ready", "fixture": True})
        elif self.path == "/vocab":  # babylm-replay: 256 single-byte tokens
            self.respond(200, {"tokens": [bytes([i]).hex() for i in range(256)]})
        elif self.path == "/v1/models":
            self.respond(200, {"object": "list", "data": [{"id": "submission", "object": "model"}]})
        else:
            self.respond(404, {"error": "not found"})

    def do_POST(self):
        try:
            size = int(self.headers.get("Content-Length", "0"))
            limit = (320 if self.path == "/interpolate" else 256 if self.path in {"/predict", "/actions"} else 32) * 1024 * 1024
            if not 0 < size <= limit:
                return self.respond(413, {"error": "invalid payload size"})
            body = self.rfile.read(size)
            if self.path == "/actions":  # IDM sends multipart video, not JSON.
                if not self.headers.get("Content-Type", "").startswith("multipart/form-data;"):
                    return self.respond(400, {"error": "expected multipart/form-data"})
                return self.respond(200, [])
            payload = json.loads(body)
            if self.path == "/redact":
                self.respond(200, {"id": payload["id"], "spans": []})
            elif self.path == "/check":  # lean-validity
                if not isinstance(payload.get("lean"), str):
                    raise ValueError("lean must be text")
                self.respond(200, {"valid": False})
            elif self.path == "/simplify":  # lean-simplification: unchanged proof is the zero floor
                self.respond(200, {"proof": payload["proof"]})
            elif self.path == "/classify":  # zero-shot-classification: uniform, untrained fixture
                count = len(payload["labels"])
                if not 2 <= count <= 256:
                    raise ValueError("expected 2..256 labels")
                self.respond(200, {"probabilities": [1 / count] * count})
            elif self.path == "/encode":  # ocr-latent-compression: no codes from a two-entry codebook
                self.respond(200, {"codes": [], "codebook_size": 2})
            elif self.path == "/decode":
                self.respond(200, {"text": ""})
            elif self.path == "/predict":  # end-of-turn: all-zero probability streams covering the audio at a 200 ms hop
                self.respond(200, {speaker: {"hop_seconds": 0.2, "eot_probability": [0.0] * max(
                    1, math.ceil(flac_seconds(base64.b64decode(payload[f"{speaker}_audio"])) / 0.2 - 1e-9))}
                    for speaker in ("speaker_1", "speaker_2")})
            elif self.path == "/score":  # babylm-replay: uniform byte model, exactly 8 bits per byte
                ids = list(payload["text"].encode("utf-8"))
                self.respond(200, {"token_ids": ids, "logprobs": [-math.log(256)] * len(ids)})
            elif self.path == "/dist":
                self.respond(200, {"logprobs": [-math.log(256)] * 256})
            elif self.path == "/v1/images/generations":  # screenshot-generation
                self.respond(200, {"data": [{"b64_json": base64.b64encode(blank_png()).decode()}]})
            elif self.path == "/interpolate":  # interface fixture only: blank RGB or nearest PNG
                anchors, targets = payload["input_frames"], payload["target_indices"]
                if (not isinstance(anchors, list) or len(anchors) < 2
                        or not isinstance(targets, list) or not targets
                        or any(not isinstance(frame, dict) or type(frame.get("index")) is not int
                               or not isinstance(frame.get("png"), str) for frame in anchors)
                        or any(type(index) is not int for index in targets)):
                    raise ValueError("expected indexed input frames and target indices")
                if self.headers.get("Accept") == "application/vnd.frontier.rgb8":
                    width, height = payload["width"], payload["height"]
                    if (type(width) is not int or type(height) is not int or not 1 <= width <= 4096
                            or not 1 <= height <= 4096 or not 1 <= len(targets) <= 15
                            or width * height * 3 * len(targets) > 512 * 1024**2):
                        raise ValueError("invalid raw RGB dimensions")
                    body = bytes(width * height * 3 * len(targets))
                    self.send_response(200)
                    self.send_header("Content-Type", "application/vnd.frontier.rgb8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                else:
                    self.respond(200, {"frames": [{"index": index, "png": min(
                        anchors, key=lambda frame: (abs(frame["index"] - index), frame["index"]))["png"]}
                        for index in targets]})
            elif self.path == "/detect":  # ai-text-detection
                self.respond(200, {"p_ai": 0.5})
            elif self.path == "/act":  # pokemon-random-battle: legal first action, no trained policy
                self.respond(200, {"actions": [0 for _ in payload["observations"]]})
            elif self.path == "/draft":  # speculative-drafting: no proposals, acceptance length 1 (the zero floor)
                if not isinstance(payload.get("tokens"), list):
                    raise ValueError("tokens must be a list")
                self.respond(200, {"draft": []})
            elif self.path == "/transcribe":  # music-transcription: no notes, F1 0 (the zero floor)
                if not flac_seconds(base64.b64decode(payload["audio"])):
                    raise ValueError("audio must be a FLAC stream")
                self.respond(200, {"notes": []})
            elif self.path == "/v1/chat/completions":
                self.respond(200, {"id": "smoke", "object": "chat.completion", "model": "submission", "choices": [{"index": 0, "message": {"role": "assistant", "content": "No description available."}, "finish_reason": "stop"}]})
            else:
                self.respond(404, {"error": "not implemented by smoke fixture"})
        except (ValueError, KeyError):
            self.respond(400, {"error": "invalid JSON request"})


if __name__ == "__main__":
    # Trusted clients open up to 64 concurrent connections; the default backlog of 5 resets them.
    ThreadingHTTPServer.request_queue_size = 256
    ThreadingHTTPServer((os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", "8000"))), Handler).serve_forever()
