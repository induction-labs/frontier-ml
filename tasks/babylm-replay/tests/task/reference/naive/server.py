"""Offline scoring server for the babylm-replay reference submissions.

GET /health, GET /vocab, POST /score {"text"}, POST /dist {"prefix_tokens"}.
Loads only files written by train.sh into $WEIGHTS_DIR. Inference runs in float32 so
/score and /dist agree to ~1e-5 nats regardless of window shape.
"""
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm  # noqa: E402
from tokenizers import Tokenizer  # noqa: E402

WEIGHTS = os.environ["WEIGHTS_DIR"]
torch.backends.cuda.matmul.allow_tf32 = False
torch.backends.cudnn.allow_tf32 = False
MODEL, CFG = lm.load_model(WEIGHTS)
TOK = Tokenizer.from_file(os.path.join(WEIGHTS, "tokenizer.json"))
BOS = TOK.token_to_id(lm.BOS)
VOCAB = lm.vocab_bytes(TOK)
VOCAB_JSON = json.dumps({"tokens": [None if b is None else b.hex() for b in VOCAB]}).encode()
CONTEXT = CFG["context"]
STRIDE = CONTEXT // 2
LOCK = threading.Lock()


@torch.no_grad()
def score(text):
    ids = TOK.encode(text).ids
    if b"".join(VOCAB[i] for i in ids) != text.encode("utf-8"):
        raise ValueError("tokenizer round trip failed")
    x = torch.tensor([BOS] + ids, device="cuda")
    out = []
    with LOCK:
        for a, j0, j1 in lm.score_windows(len(ids), CONTEXT, STRIDE):
            logits = MODEL(x[a:j1 - 1][None])[0]
            lp = torch.log_softmax(logits, -1)[j0 - 1 - a:]
            out.append(lp.gather(1, x[j0:j1, None])[:, 0])
    logprobs = torch.cat(out).tolist() if out else []
    return {"token_ids": ids, "logprobs": logprobs}


@torch.no_grad()
def dist(prefix):
    if any(type(i) is not int or not 0 <= i < len(VOCAB) or VOCAB[i] is None for i in prefix):
        raise ValueError("invalid prefix token")
    x = [BOS] + prefix
    j = len(x)
    a = lm.window_start(j, CONTEXT, STRIDE)
    with LOCK:
        logits = MODEL(torch.tensor(x[a:j], device="cuda")[None])[0, -1]
        return {"logprobs": torch.log_softmax(logits, -1).tolist()}


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

    def log_message(self, *args):
        pass

    def send(self, code, body):
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path == "/health":
            self.send(200, b'{"ok":true}')
        elif self.path == "/vocab":
            self.send(200, VOCAB_JSON)
        else:
            self.send(404, b"{}")

    def do_POST(self):
        try:
            body = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))))
            if self.path == "/score":
                result = score(body["text"])
            elif self.path == "/dist":
                result = dist(body["prefix_tokens"])
            else:
                return self.send(404, b"{}")
            self.send(200, json.dumps(result).encode())
        except Exception as exc:  # noqa: BLE001
            self.send(400, json.dumps({"error": str(exc)}).encode())


if __name__ == "__main__":
    ThreadingHTTPServer((os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", 8000))), Handler).serve_forever()
