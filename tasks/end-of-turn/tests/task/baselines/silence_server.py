"""HTTP interface fixture: all-zero probability streams at a 200 ms hop. It is not a trained task solution."""
import base64
import json
import os
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from end_of_turn import flac_info
from turn_taking import silent_stream


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        self.respond({"ready": True})

    def do_POST(self):
        payload = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        streams = {}
        for speaker in ("speaker_1", "speaker_2"):
            info = flac_info(base64.b64decode(payload[f"{speaker}_audio"]))
            streams[speaker] = silent_stream(info["frames"] / info["sample_rate"])
        self.respond(streams)

    def respond(self, value):
        body = json.dumps(value).encode()
        self.send_response(200)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, *args):
        pass


if __name__ == "__main__":
    ThreadingHTTPServer((os.environ.get("HOST", "127.0.0.1"), int(os.environ.get("PORT", "8000"))), Handler).serve_forever()
