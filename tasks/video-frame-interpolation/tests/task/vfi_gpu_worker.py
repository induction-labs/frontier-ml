"""Trusted H100 worker. All inputs have been staged onto ephemeral local disk."""
import json
from pathlib import Path
import sys
import time

from vfi_evaluate import calibration, digest, evaluation_identity, evaluate_export, runtime_identity
from vfi_metrics import MetricEngine, code_sha256
from vfi_runtime import check_cuda_runtime


def run(root):
    root = Path(root)
    started = time.monotonic()
    config = json.loads((root / "request.json").read_text())
    gpu = check_cuda_runtime()
    data, predictions = root / "data/test.jsonl", root / "predictions"
    _, identity = evaluation_identity(data, selection=config["evaluation_selection"])
    if identity["evaluation_sha256"] != config["evaluation_sha256"]:
        raise ValueError("Scoring media changed during transfer")
    if digest(predictions / "predictions.jsonl") != config["prediction_manifest_sha256"]:
        raise ValueError("Prediction manifest changed during transfer")
    config["calibration_path"] = str(root / "calibration.json")
    calibration(config, identity)
    engine = MetricEngine(root / "metrics", expected_sha256=config["metric_bundle_sha256"], device="cuda")
    remaining = config["metric_timeout_seconds"] - (time.monotonic() - started)
    report = evaluate_export(data, predictions, engine, timeout_seconds=remaining,
                             selection=config["evaluation_selection"],
                             on_interval=lambda n, total: print(f"metrics: {n}/{total}", flush=True))
    report.update(evaluation_sha256=identity["evaluation_sha256"],
                  metric_bundle_sha256=config["metric_bundle_sha256"],
                  calibration_sha256=config["calibration_sha256"],
                  metric_code_sha256=code_sha256(), runtime=runtime_identity("cuda"), gpu=gpu,
                  worker_seconds=time.monotonic() - started)
    # File transport works with outbound networking disabled, including large reports.
    temporary = root / "result.tmp"
    temporary.write_text(json.dumps(report, allow_nan=False) + "\n")
    temporary.replace(root / "result.json")
    return report


if __name__ == "__main__":
    run(sys.argv[1])
