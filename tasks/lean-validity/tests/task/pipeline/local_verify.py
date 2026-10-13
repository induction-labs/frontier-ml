"""Organizer smoke test: run a submission locally and apply the trusted gates except the LLM review.

    python local_verify.py SUBMISSION_DIR DATA.jsonl [--port 8123] [--review-receipt PATH]

Starts `bash SUBMISSION_DIR/serve.sh` with HOST/PORT, waits up to 300 s for /health, runs
the no-Lean scan, then the verifier's timed batch (600 s gate enforced, validation_mode
off). Without --review-receipt the architecture review is reported as skipped, so the
result is never an official score. This does not provide offline/GPU isolation.
"""
import argparse
import json
import os
import signal
import subprocess
import sys
import time
from pathlib import Path

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK.parents[1]))

from frontier_ml.taskspec import load_task, sibling  # noqa: E402
from frontier_ml.validation.http import HTTPTransport  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("data")
    parser.add_argument("--port", type=int, default=8123)
    parser.add_argument("--concurrency", type=int, default=64)
    parser.add_argument("--review-receipt")
    args = parser.parse_args()
    task = load_task("lean-validity", TASK)
    verifier = task.module
    scan = sibling(TASK / "verifier.py", "lean_scan.py").scan_submission(args.submission)
    if not scan["passed"]:
        print(json.dumps({"score": 0.0, "failure": "no_lean_scan", "scan": scan}, indent=2))
        return 1
    env = dict(os.environ, HOST="127.0.0.1", PORT=str(args.port), MODEL_NAME="submission")
    proc = subprocess.Popen(["bash", str(Path(args.submission) / "serve.sh")], env=env, start_new_session=True)
    transport = HTTPTransport(f"http://127.0.0.1:{args.port}")
    started = time.monotonic()
    try:
        while time.monotonic() - started < 300:
            try:
                if transport.raw_request("GET", "/health", timeout=2)[0] == 200:
                    break
            except Exception:
                time.sleep(0.5)
        else:
            print(json.dumps({"score": 0.0, "failure": "startup"}))
            return 1
        startup = time.monotonic() - started
        config = {"concurrency": args.concurrency, "inference_timeout_seconds": 3600, "validation_mode": False}
        if args.review_receipt:
            config["architecture_review_path"] = args.review_receipt
            result = verifier.verify({"transport": transport, "snapshot": Path(args.submission)}, args.data, config)
        else:
            result = verifier.run_gated(transport, args.data, config)
            result["details"]["gates"] = {"no_lean_scan": {"passed": True}, "architecture": "SKIPPED (no reviewer configured)"}
        result["details"].pop("per_theorem_accuracy", None)
        result["startup_seconds"] = startup
        result["official"] = False
        print(json.dumps(result, indent=2))
        return 0
    finally:
        os.killpg(proc.pid, signal.SIGTERM)


if __name__ == "__main__":
    raise SystemExit(main())
