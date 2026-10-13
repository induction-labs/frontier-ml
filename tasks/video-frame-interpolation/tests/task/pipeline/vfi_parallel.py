"""Parallel organizer scoring of one complete prediction export, using the frozen CPU metric."""
import json
from pathlib import Path
import time

from gimm_baseline import sha256, write_json
from vfi_evaluate import reference_calibration as calibration, evaluate_export, evaluation_identity, runtime_identity
from vfi_metrics import MetricEngine, code_sha256
from vfi_scoring import METRICS, PROTOCOL, aggregate, validate_baseline


def score_shard(root, index, count):
    root = Path(root)
    receipt = json.loads((root / "prepared.json").read_text())
    engine = MetricEngine(root / "metrics", expected_sha256=receipt["metric_bundle_sha256"])
    report = evaluate_export(root / "download/video-frame-interpolation/private/test.jsonl",
                             root / "predictions-1", engine, shard=(index, count), timeout_seconds=7200,
                             selection=receipt["dataset"].get("selection"),
                             on_interval=lambda n, total: print(f"shard {index}: interval {n}/{total}", flush=True))
    report.update(shard_index=index, shard_count=count, runtime=runtime_identity("cpu"),
                  metric_code_sha256=code_sha256(), metric_bundle_sha256=receipt["metric_bundle_sha256"])
    return report


def merge_shards(samples, reports, *, count, manifest_sha256, bundle_sha256):
    if len(reports) != count or {r["shard_index"] for r in reports} != set(range(count)):
        raise ValueError("Missing or duplicate metric shards")
    combined = {}
    for report in reports:
        expected = {"shard_count": count, "protocol": PROTOCOL, "runtime": runtime_identity("cpu"),
                    "metric_code_sha256": code_sha256(), "metric_bundle_sha256": bundle_sha256,
                    "prediction_manifest_sha256": manifest_sha256}
        if any(report.get(k) != v for k, v in expected.items()):
            raise ValueError("Metric shard provenance differs")
        assigned = samples[report["shard_index"]::count]
        if [r["id"] for r in report["intervals"]] != [s["id"] for s in assigned]:
            raise ValueError("Metric shard interval coverage differs")
        for sample, record in zip(assigned, report["intervals"], strict=True):
            if record["interpolation_factor"] != sample["frame_count"] - 1 or record["id"] in combined:
                raise ValueError("Metric shard factor mismatch or duplicated interval")
            combined[record["id"]] = record
    rows = [combined[s["id"]] for s in samples]
    return rows, aggregate(rows)


def finish(root, reports, started):
    root = Path(root)
    receipt = json.loads((root / "prepared.json").read_text())
    samples, identity = evaluation_identity(root / "download/video-frame-interpolation/private/test.jsonl",
                                             selection=receipt["dataset"].get("selection"))
    if identity["evaluation_sha256"] != receipt["dataset"]["evaluation_sha256"]:
        raise ValueError("Evaluation bytes changed after baseline preparation")
    manifest_hash = sha256(root / "predictions-1/predictions.jsonl")
    rows, factors = merge_shards(samples, reports, count=len(reports), manifest_sha256=manifest_hash,
                                 bundle_sha256=receipt["metric_bundle_sha256"])
    report = {"protocol": PROTOCOL, "by_factor": factors, "intervals": rows,
              "evaluation_sha256": identity["evaluation_sha256"],
              "metric_bundle_sha256": receipt["metric_bundle_sha256"],
              "metric_code_sha256": code_sha256(), "runtime": runtime_identity("cpu"),
              "prediction_manifest_sha256": manifest_hash, "workers": len(reports),
              "metric_seconds": time.monotonic() - started,
              "worker_metric_seconds": sum(r["metric_seconds"] for r in reports)}
    baseline = {factor: {name: row[name] for name in METRICS} for factor, row in factors.items()}
    validate_baseline(baseline)
    calibrated = {k: report[k] for k in ("protocol", "evaluation_sha256", "metric_bundle_sha256",
                                        "metric_code_sha256", "runtime", "prediction_manifest_sha256")}
    calibrated.update(baseline=baseline, baseline_id=receipt["baseline_id"],
                      baseline_revision=sha256(root / "prepared.json"))
    write_json(root / "metrics.json", report)
    write_json(root / "calibration.json", calibrated)
    calibration({"calibration_path": str(root / "calibration.json"),
                 "calibration_sha256": sha256(root / "calibration.json"),
                 "metric_bundle_sha256": receipt["metric_bundle_sha256"]}, identity)
    names = ["prepared.json", "pass-1.json", "metrics.json", "calibration.json", "metrics/lock.json"]
    if (root / "generation.json").exists():
        names.append("generation.json")
    return {name: (root / name).read_text() for name in names}
