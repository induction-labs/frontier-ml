"""Local end-to-end run of the trusted verifier against a real HTTP server (no Harbor, local GPU 4).

python run_verifier.py SUBMISSION_DIR SPLIT_JSONL OUT_JSON [--expected N] [--evaluation-samples N]
Runs scorer preflight, HTTP generation, pinned scorers and quality gates.
"""
import argparse
import json
import os
import subprocess
import time
import urllib.request
from pathlib import Path

from frontier_ml.taskspec import load_module
from frontier_ml.validation.http import HTTPTransport

TASK = Path(__file__).resolve().parents[2]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("submission")
    parser.add_argument("split")
    parser.add_argument("out")
    parser.add_argument("--expected", type=int, default=500, help="number of rows in the pinned source pool")
    parser.add_argument("--evaluation-samples", type=int, default=200, help="size of the fixed stratified evaluation subset")
    parser.add_argument("--port", type=int, default=8123)
    args = parser.parse_args()
    sub = Path(args.submission)
    verifier = load_module("screenshot-generation.e2e", TASK / "verifier.py", require=("verify",))
    config = {"validation_mode": False, "predictions_only": False, "concurrency": 1, "expected_samples": args.expected,
              "evaluation_samples": args.evaluation_samples,
              "inference_timeout_seconds": 7200, "request_timeout_seconds": 90,
              "scorer_models_dir": "/scratch/frontier-ml-workspace/work/screenshot-generation/models"}
    verifier.preflight(config, args.split)
    env = dict(os.environ, HOST="127.0.0.1", PORT=str(args.port), MODEL_NAME="submission", PYTHONDONTWRITEBYTECODE="1")
    server = subprocess.Popen(["bash", str(sub / "serve.sh")], env=env, stdout=open(Path(args.out).with_suffix(".server.log"), "w"),
                              stderr=subprocess.STDOUT)
    started = time.monotonic()
    try:
        while time.monotonic() - started < 300:
            try:
                with urllib.request.urlopen(f"http://127.0.0.1:{args.port}/health", timeout=2) as r:
                    if r.status == 200:
                        break
            except Exception:
                time.sleep(1)
        else:
            raise SystemExit("server not healthy in 300 s")
        startup = time.monotonic() - started
        result = verifier.verify({"transport": HTTPTransport(f"http://127.0.0.1:{args.port}"), "snapshot": sub,
                                  "startup_elapsed": startup, "backend": "local"}, args.split, config)
        result["startup_seconds"] = startup
    finally:
        server.terminate()
        server.wait(timeout=60)
    Path(args.out).write_text(json.dumps(result, indent=1))
    d = result["details"]
    print(json.dumps({"score": result["score"], "means": d.get("means"), "failed_gates": d.get("failed_gates"),
                      "latency": d.get("gates", {}).get("latency")}, indent=1))


if __name__ == "__main__":
    main()
