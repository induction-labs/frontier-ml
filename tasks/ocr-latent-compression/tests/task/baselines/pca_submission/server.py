"""HTTP server for the organizer PCA baseline: /health, /encode, /decode with GPU micro-batching.

One GPU worker thread owns both models. Requests queue per endpoint and are batched (up to
MAX_BATCH, waiting at most BATCH_WAIT seconds for more). /health turns 200 only after both
models are loaded and warmed up.
"""
import base64
import io
import json
import os
import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

from PIL import Image

import encoder
from decoder import Decoder

MAX_BATCH, BATCH_WAIT = 16, 0.02
READY = threading.Event()
QUEUES = {"encode": queue.Queue(), "decode": queue.Queue()}


class Job:
    def __init__(self, data):
        self.data, self.done, self.result, self.error = data, threading.Event(), None, None


def gpu_worker():
    try:
        decoder = Decoder()
        warm = io.BytesIO()
        Image.new("RGB", (64, 64), "white").save(warm, format="PNG")
        decoder.decode(encoder.encode([warm.getvalue()]))
    except BaseException:
        import traceback
        traceback.print_exc()
        os._exit(1)  # fail fast: never serve 503 forever after a failed model load
    READY.set()
    while True:
        busy = False
        for kind, run in (("encode", encoder.encode), ("decode", decoder.decode)):
            jobs = []
            try:
                jobs.append(QUEUES[kind].get_nowait())
            except queue.Empty:
                continue
            deadline = time.monotonic() + BATCH_WAIT
            while len(jobs) < MAX_BATCH:
                try:
                    jobs.append(QUEUES[kind].get(timeout=max(0, deadline - time.monotonic())))
                except queue.Empty:
                    break
            busy = True
            try:
                results = run([job.data for job in jobs])
                for job, result in zip(jobs, results):
                    job.result = result
            except Exception as exc:  # report to every waiting request
                for job in jobs:
                    job.error = repr(exc)
            for job in jobs:
                job.done.set()
        if not busy:
            time.sleep(0.002)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"

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
        if self.path == "/health":
            self.reply(200 if READY.is_set() else 503, {"ready": READY.is_set()})
        else:
            self.reply(404, {"error": "not found"})

    def do_POST(self):
        try:
            request = json.loads(self.rfile.read(int(self.headers.get("Content-Length", "0"))))
            if self.path == "/encode":
                job = Job(base64.b64decode(request["image"], validate=True))
            elif self.path == "/decode":
                job = Job(base64.b64decode(request["payload"], validate=True))
            else:
                return self.reply(404, {"error": "not found"})
        except (ValueError, KeyError, TypeError):
            return self.reply(400, {"error": "bad request"})
        kind = self.path[1:]
        QUEUES[kind].put(job)
        job.done.wait()
        if job.error:
            return self.reply(500, {"error": job.error})
        if kind == "encode":
            self.reply(200, {"payload": base64.b64encode(job.result).decode("ascii")})
        else:
            self.reply(200, {"text": job.result})


if __name__ == "__main__":
    threading.Thread(target=gpu_worker, daemon=True).start()
    ThreadingHTTPServer((os.environ.get("HOST", "0.0.0.0"), int(os.environ.get("PORT", "8000"))), Handler).serve_forever()
