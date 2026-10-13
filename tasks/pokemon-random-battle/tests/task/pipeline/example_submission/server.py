"""Minimal learned linear-action endpoint, loaded once at startup."""
import json
import os
from pathlib import Path
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np
from safetensors.numpy import load_file

WEIGHTS = load_file(Path(__file__).with_name("model.safetensors"))["action_weight"]

class Handler(BaseHTTPRequestHandler):
    def log_message(self,*args):
        pass

    def reply(self,status,body):
        raw=json.dumps(body).encode()
        self.send_response(status); self.send_header("Content-Type","application/json")
        self.send_header("Content-Length",str(len(raw))); self.end_headers(); self.wfile.write(raw)

    def do_GET(self):
        self.reply(200 if self.path == "/health" else 404,{"ready":True})

    def do_POST(self):
        try:
            size=int(self.headers.get("Content-Length","0"))
            if self.path != "/act" or not 0<size<=8*1024*1024:
                return self.reply(400,{"error":"invalid request"})
            observations=json.loads(self.rfile.read(size))["observations"]
            if not 1<=len(observations)<=32:
                raise ValueError("invalid batch")
            actions=[int(np.argmax(np.asarray(o["action_features"],dtype=np.float32)@WEIGHTS)) for o in observations]
            self.reply(200,{"actions":actions})
        except (ValueError,KeyError,TypeError):
            self.reply(400,{"error":"invalid observation"})

if __name__ == "__main__":
    ThreadingHTTPServer((os.environ.get("HOST","0.0.0.0"),int(os.environ.get("PORT","8000"))),Handler).serve_forever()
