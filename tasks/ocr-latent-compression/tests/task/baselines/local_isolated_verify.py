"""Local end-to-end check of the trusted two-phase verifier against a real submission directory.

Runs the task's ``verify_isolated`` with a launcher that starts a *fresh local process*
(``bash serve.sh``) per phase on this machine's GPU, waits for /health (300 s), and kills
the process group when the phase ends. Static provenance metadata checks run for real
against the trusted lock. The LLM architecture review is NOT run: a local receipt marked
``local-e2e-unreviewed`` is written so the pipeline can proceed. This is a harness test,
not an official score, and it provides no network isolation.

    CUDA_VISIBLE_DEVICES=2 OCR_PYTHON=... python local_isolated_verify.py --submission DIR \
        --data data/ocr-latent-compression/public/validation.jsonl --out result.json [--diagnostic-rate 1e-6]
"""
import argparse
from contextlib import contextmanager
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))
from frontier_ml.scoring.architecture import submission_digest  # noqa: E402
from frontier_ml.taskspec import load_module, sibling  # noqa: E402
from frontier_ml.validation.http import HTTPTransport  # noqa: E402

SEED = ROOT / "tasks" / "ocr-latent-compression"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--submission", type=Path, required=True)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--port", type=int, default=8190)
    parser.add_argument("--diagnostic-rate", type=float,
                        help="override the per-phase images/s floor (e.g. 1e-6) to get a score from a slow server; "
                             "default: the verifier's real floor")
    args = parser.parse_args()
    verifier = load_module("ocr-latent-compression.verifier", SEED / "verifier.py", require=("verify_isolated",))
    checks = list(sibling(SEED / "verifier.py", "review.py").POLICY["checks"])
    work = Path(tempfile.mkdtemp(prefix="ocr-local-verify-", dir=os.environ.get("OCR_E2E_WORK")))
    receipt = work / "receipt.json"
    receipt.write_text(json.dumps({"task": "ocr-latent-compression", "decision": "pass", "reviewer": "local-e2e-unreviewed",
                                   "submission_digest": submission_digest(args.submission),
                                   "evidence": [{"check": c, "note": "not reviewed"} for c in checks],
                                   "checks": {c: True for c in checks}}))
    rows = [line for line in args.data.read_text().splitlines() if line.strip()]
    config = {**verifier.default_config(), "concurrency": 16, "expected_samples": len(rows),
              "inference_timeout_seconds": 3600, "submission_path": str(args.submission),
              "architecture_review_path": str(receipt)}
    if args.diagnostic_rate:
        config["minimum_images_per_second"] = args.diagnostic_rate
    phase_index = [0]

    @contextmanager
    def launch():
        phase_index[0] += 1
        port = args.port + phase_index[0]
        home = work / f"home{phase_index[0]}"
        home.mkdir()
        env = {"PATH": os.environ["PATH"], "HOME": str(home), "TMPDIR": str(home), "HOST": "127.0.0.1",
               "PORT": str(port), "MODEL_NAME": "submission", "CUDA_VISIBLE_DEVICES": os.environ.get("CUDA_VISIBLE_DEVICES", "0"),
               "OCR_PYTHON": os.environ.get("OCR_PYTHON", sys.executable)}
        log = open(work / f"server{phase_index[0]}.log", "wb")
        process = subprocess.Popen(["bash", str(args.submission / "serve.sh")], env=env, stdout=log, stderr=subprocess.STDOUT,
                                   start_new_session=True)
        started = time.monotonic()
        transport = HTTPTransport(f"http://127.0.0.1:{port}")
        try:
            while True:
                if process.poll() is not None:
                    raise RuntimeError(f"server exited during startup; see {log.name}")
                if time.monotonic() - started > 300:
                    raise RuntimeError("server did not become healthy within 300 s")
                try:
                    if transport.raw_request("GET", "/health", timeout=2)[0] == 200:
                        break
                except OSError:
                    pass
                time.sleep(0.5)
            yield {"transport": transport, "startup_elapsed": time.monotonic() - started, "backend": "local-process",
                   "sandbox_id": f"local-{process.pid}"}
        finally:
            os.killpg(process.pid, signal.SIGKILL)
            process.wait()
            log.close()

    result = verifier.verify_isolated(launch, str(args.data), config)
    result["local_e2e_note"] = "LLM review bypassed; local processes, no network isolation; not an official score"
    for record in result["details"]["records"]:
        record.pop("prediction", None)
    args.out.write_text(json.dumps(result, indent=2))
    summary = {k: result["details"][k] for k in ("by_kind", "by_bucket", "payload_bytes", "determinism", "failures")}
    print(json.dumps({"score": result["score"], **summary, "timing": result["details"]["timing"],
                      "instances": result["details"]["instances"]}, indent=2))


if __name__ == "__main__":
    main()
