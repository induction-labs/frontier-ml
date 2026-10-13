"""Contract smoke server for lean-validity (stdlib only). Not a learned model; it cannot pass review.

    HOST=127.0.0.1 PORT=8000 python heuristic_server.py [--mode constant|length|gbdt --model gbdt.json]

constant: always answers {"valid": false} (macro F1 1/3 on balanced data).  length: yes
for short proofs (a surface prior).  gbdt: yes when a model exported by gbdt.py scores
at least --threshold.
"""
import argparse
import json
import os
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def make_predictor(mode, model_path=None, threshold=0.5):
    if mode == "constant":
        return lambda text: False
    if mode == "length":
        return lambda text: len(text) < 1200
    if mode == "gbdt":
        from gbdt import load_predictor
        score = load_predictor(model_path)
        return lambda text: score(text) >= threshold
    raise ValueError(mode)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", default="constant", choices=["constant", "length", "gbdt"])
    parser.add_argument("--model")
    parser.add_argument("--threshold", type=float, default=0.5)
    args = parser.parse_args()
    predict = make_predictor(args.mode, args.model, args.threshold)

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, code, payload):
            body = json.dumps(payload).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send(200 if self.path == "/health" else 404, {"ok": self.path == "/health"})

        def do_POST(self):
            if self.path != "/check":
                return self._send(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length", 0))
            try:
                text = json.loads(self.rfile.read(length))["lean"]
            except (ValueError, KeyError, TypeError):
                return self._send(400, {"error": "expected {\"lean\": str}"})
            self._send(200, {"valid": bool(predict(text))})

    ThreadingHTTPServer.request_queue_size = 256  # the trusted client opens up to 64 connections at once
    server = ThreadingHTTPServer((os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", "8000"))), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
