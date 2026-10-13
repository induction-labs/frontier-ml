"""Contract-following baseline server: python server.py METHOD  (zimage | flux2klein | compositor | html | blank).

Serves GET /health and POST /v1/images/generations on $HOST:$PORT. compositor/html are NON-COMPLIANT
metric sanity checks; zimage/flux2klein are training-free baselines.
"""
import base64
import io
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
import methods  # noqa: E402

GENERATOR = methods.make(sys.argv[1])
LOCK = threading.Lock()


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):
        pass

    def reply(self, status, body):
        data = json.dumps(body).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)

    def do_GET(self):
        self.reply(200, {"status": "ready"}) if self.path == "/health" else self.reply(404, {"error": "not found"})

    def do_POST(self):
        if self.path != "/v1/images/generations":
            return self.reply(404, {"error": "not found"})
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            with LOCK:
                image = GENERATOR(request["description"], request["texts"]).convert("RGB")
            buffer = io.BytesIO()
            image.save(buffer, format="PNG")
            self.reply(200, {"data": [{"b64_json": base64.b64encode(buffer.getvalue()).decode()}]})
        except (KeyError, ValueError) as error:
            self.reply(400, {"error": str(error)})


if __name__ == "__main__":
    HTTPServer((os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))), Handler).serve_forever()
