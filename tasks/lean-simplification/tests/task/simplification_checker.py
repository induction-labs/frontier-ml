"""Docker isolation and HTTP checker service.

Official grading deploys an organizer-owned copy. Candidate copies of the public
reference helper are for development and never participate in trusted grading.

python tasks/lean-simplification/simplification_checker.py --image sha256:... --port 8124

Deploy on an organizer-owned CPU host. Do not expose Docker to the candidate or
mount the submission, data, credentials, host filesystem, or Docker socket in jobs.
"""
import argparse
import hashlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import re
import subprocess
import threading
import uuid

from simplification_client import LEAN_VERSION, MATHLIB_REVISION, PROTOCOL


class DockerChecker:
    def __init__(self, image):
        if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
            raise ValueError("Use an immutable local Docker image ID (sha256:...)")
        self.image = image
        result = subprocess.run(["docker", "image", "inspect", image], check=True, capture_output=True, text=True)
        info = json.loads(result.stdout)[0]
        labels = info["Config"].get("Labels") or {}
        for key, value in {"lean": LEAN_VERSION, "mathlib": MATHLIB_REVISION, "protocol": PROTOCOL}.items():
            if labels.get("frontier." + key) != value:
                raise ValueError("Checker image labels do not match pinned toolchain")
        self.fingerprint = hashlib.sha256((PROTOCOL + "\n" + image).encode()).hexdigest()
        # Retain the content even if a development build moves its original tag.
        subprocess.run(["docker", "tag", image, "frontier-lean-checker:" + self.fingerprint], check=True)
        self.slots = threading.BoundedSemaphore(32)

    def command(self, name):
        return ["docker", "run", "--rm", "--name", name, "--network", "none", "--read-only",
                "--user", "65534:65534", "--cap-drop", "ALL", "--security-opt", "no-new-privileges",
                "--pids-limit", "128", "--memory", "4g", "--memory-swap", "4g", "--cpus", "1",
                "--ulimit", "core=0", "--ulimit", "fsize=4194304:4194304",
                "--tmpfs", "/tmp:rw,noexec,nosuid,size=32m", "-i", self.image]

    def __call__(self, request):
        name = "lean-simplification-" + uuid.uuid4().hex
        with self.slots:
            try:
                proc = subprocess.run(self.command(name), input=json.dumps(request, ensure_ascii=False), capture_output=True,
                                      text=True, timeout=70)
            except subprocess.TimeoutExpired:
                # A Docker client timeout alone does NOT kill the container.
                cleanup = subprocess.run(["docker", "rm", "-f", name], capture_output=True, timeout=15)
                if cleanup.returncode:
                    raise RuntimeError("Could not stop timed-out checker container")
                return {"valid": False, "status": "timeout"}
        if proc.returncode:
            raise RuntimeError(f"Checker container failed ({proc.returncode}): {proc.stderr[-1000:]}")
        result = json.loads(proc.stdout)
        if type(result.get("valid")) is not bool or result.get("status") not in {"ok", "invalid", "timeout"}:
            raise RuntimeError("Invalid checker worker verdict")
        return result

    def selftest(self):
        positive = self({"statement": "(n : ℕ) : n = n", "proof": "by rfl"})
        negative = self({"statement": ": False", "proof": "by exact True.intro"})
        if not positive["valid"] or negative["valid"]:
            raise RuntimeError(f"Checker self-test failed: {positive}, {negative}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8124)
    args = parser.parse_args(argv)
    checker = DockerChecker(args.image)
    checker.selftest()

    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *args):
            pass

        def respond(self, status, payload):
            raw = json.dumps(payload).encode()
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(raw)))
            self.end_headers()
            self.wfile.write(raw)

        def do_GET(self):
            if self.path != "/health":
                return self.respond(404, {})
            self.respond(200, {"protocol": PROTOCOL, "lean": LEAN_VERSION, "mathlib": MATHLIB_REVISION,
                               "fingerprint": checker.fingerprint})

        def do_POST(self):
            if self.path != "/check":
                return self.respond(404, {})
            try:
                size = int(self.headers.get("Content-Length", "0"))
                if not 0 < size <= 600_000:
                    return self.respond(400, {"error": "body size"})
                request = json.loads(self.rfile.read(size))
                self.respond(200, checker(request))
            except Exception as exc:
                self.respond(503, {"status": "infrastructure_error", "error": str(exc)})

    print(json.dumps({"url": f"http://{args.host}:{args.port}", "fingerprint": checker.fingerprint}), flush=True)
    ThreadingHTTPServer.request_queue_size = 64
    ThreadingHTTPServer((args.host, args.port), Handler).serve_forever()


if __name__ == "__main__":
    main()
