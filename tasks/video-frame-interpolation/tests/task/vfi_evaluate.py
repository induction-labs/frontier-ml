"""Organizer CLI and trusted scoring of complete prediction exports, never candidate code."""
import argparse
import hashlib
import json
from pathlib import Path
import time

from frontier_ml.validation.evaluation import load_samples
from frontier_ml.validation.errors import VerificationFailure
from vfi_protocol import media_path, normalize_sample, validate_sample
from vfi_scoring import (FACTORS, METRICS, PROTOCOL, SCORING_PROTOCOL, aggregate, number,
                         validate_anchors, validate_baseline)
from vfi_selection import select_intervals


def digest(path):
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def evaluation_identity(data_path, *, selection=None):
    """Bind calibration to rows AND every endpoint/reference byte, not only paths."""
    path = Path(data_path).resolve()
    samples = select_intervals([normalize_sample(sample) for sample in load_samples(path)], selection)
    files = {}
    factors = set()
    for sample in samples:
        validate_sample(sample)
        if min(sample["width"], sample["height"]) < 32 or "reference_frames" not in sample:
            raise VerificationFailure("Scoring requires native dimensions >=32 and every reference frame")
        factors.add(sample["frame_count"] - 1)
        for frame in sample["input_frames"] + sample["reference_frames"]:
            relative = frame["path"]
            if relative not in files:
                files[relative] = digest(media_path(path.parent, relative))
    if factors != set(FACTORS):
        raise VerificationFailure("Evaluation requires all three interpolation factors")
    identity = {"jsonl_sha256": digest(path), "files": files}
    if selection is not None:
        identity.update(selection=selection, selected_ids=[sample["id"] for sample in samples])
    identity["evaluation_sha256"] = hashlib.sha256(json.dumps(identity, sort_keys=True).encode()).hexdigest()
    return samples, identity


def runtime_identity(device):
    import numpy
    import PIL
    import torch
    runtime = {"torch": torch.__version__.split("+")[0], "numpy": numpy.__version__,
               "pillow": PIL.__version__, "device": device, "precision": "float32", "threads": 8}
    if device == "cuda":
        runtime.update(torch_build=torch.__version__, cuda=torch.version.cuda)
    return runtime


def reference_calibration(config, identity):
    """Validate a single measured reference; this alone cannot enable grading."""
    from vfi_metrics import code_sha256
    path = config.get("calibration_path")
    if not path or not config.get("calibration_sha256") or digest(path) != config["calibration_sha256"]:
        raise VerificationFailure("Missing or changed frozen baseline calibration")
    record = json.loads(Path(path).read_text())
    expected = {"protocol": PROTOCOL, "evaluation_sha256": identity["evaluation_sha256"],
                "metric_bundle_sha256": config.get("metric_bundle_sha256"),
                "metric_code_sha256": code_sha256(),
                "runtime": config.get("metric_runtime") or runtime_identity("cpu")}
    if any(record.get(key) != value for key, value in expected.items()):
        raise VerificationFailure("Baseline calibration does not match the data, metrics, weights or scoring runtime")
    for key in ("baseline_id", "baseline_revision", "prediction_manifest_sha256"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise VerificationFailure("Baseline calibration needs measured prediction and model provenance")
    validate_baseline(record.get("baseline"))
    return record


def calibration(config, identity):
    """Production requires both measured anchors, bound to the scoring version."""
    record = reference_calibration(config, identity)
    if record.get("scoring_protocol") != SCORING_PROTOCOL:
        raise VerificationFailure("Calibration does not match the two-anchor scoring protocol")
    for key in ("target_id", "target_revision", "target_prediction_manifest_sha256"):
        if not isinstance(record.get(key), str) or not record[key].strip():
            raise VerificationFailure("Both calibration anchors need measured prediction and model provenance")
    validate_anchors(record.get("baseline"), record.get("target"))
    return record


def compose_calibration(baseline, target):
    """Combine two measured reference receipts without changing their error values."""
    from vfi_metrics import code_sha256
    keys = ("protocol", "evaluation_sha256", "metric_bundle_sha256", "metric_code_sha256", "runtime")
    if any(baseline.get(key) != target.get(key) or baseline.get(key) is None for key in keys):
        raise VerificationFailure("Both anchors must use identical data, metric weights, code and runtime")
    if baseline["protocol"] != PROTOCOL or baseline["metric_code_sha256"] != code_sha256():
        raise VerificationFailure("Reference measurements must use the current metric implementation")
    for reference in (baseline, target):
        for key in ("baseline_id", "baseline_revision", "prediction_manifest_sha256"):
            if not isinstance(reference.get(key), str) or not reference[key].strip():
                raise VerificationFailure("Each reference needs measured prediction and model provenance")
    validate_anchors(baseline.get("baseline"), target.get("baseline"))
    return {**baseline, "scoring_protocol": SCORING_PROTOCOL, "target": target["baseline"],
            "target_id": target["baseline_id"], "target_revision": target["baseline_revision"],
            "target_prediction_manifest_sha256": target["prediction_manifest_sha256"]}


def evaluate_export(data_path, prediction_dir, engine, *, timeout_seconds=10800, on_interval=None, shard=None,
                    selection=None):
    """Stream metrics; reject incomplete/tampered exports before returning any score."""
    number(timeout_seconds, "metric timeout", positive=True)
    if shard is not None:
        if (not isinstance(shard, tuple) or len(shard) != 2
                or any(type(n) is not int for n in shard) or not 0 <= shard[0] < shard[1]):
            raise ValueError("Metric shard must be (index, count) with 0 <= index < count")
    deadline = time.monotonic() + timeout_seconds

    def check_deadline():
        if time.monotonic() >= deadline:
            raise TimeoutError("Trusted metric computation exceeded its deadline")

    data_path, root = Path(data_path).resolve(), Path(prediction_dir).resolve()
    samples = select_intervals([normalize_sample(sample) for sample in load_samples(data_path)], selection)
    status = json.loads((root / "run.json").read_text())
    if (status.get("status") != "complete" or status.get("samples_completed") != len(samples)
            or status.get("samples_expected") != len(samples)):
        raise VerificationFailure("Only complete prediction exports can be evaluated")
    records = []
    with (root / "predictions.jsonl").open() as stream:
        for ordinal, sample in enumerate(samples, 1):
            check_deadline()
            validate_sample(sample)
            line = stream.readline()
            if not line:
                raise VerificationFailure("Missing interval in prediction export")
            row = json.loads(line)
            for key in ("id", "schema_version", "width", "height", "frame_count", "target_indices"):
                if row.get(key) != sample[key]:
                    raise VerificationFailure(f"Prediction export does not match {key}")
            factor = sample["frame_count"] - 1
            if row.get("interpolation_factor") != factor or row.get("input_indices") != [0, factor]:
                raise VerificationFailure("Prediction export has the wrong interpolation schedule")
            frames = row.get("frames", [])
            if len(frames) != sample["frame_count"]:
                raise VerificationFailure("Incomplete predicted interval")
            if shard is not None and (ordinal - 1) % shard[1] != shard[0]:
                continue
            references = {f["index"]: media_path(data_path.parent, f["path"])
                          for f in sample["input_frames"] + sample["reference_frames"]}
            predicted = []
            for index, frame in enumerate(frames):
                if frame.get("index") != index or frame.get("generated") is not (index in sample["target_indices"]):
                    raise VerificationFailure("Reordered frames or incorrect generated flags")
                path = media_path(root, frame["path"])
                actual = digest(path)
                if actual != frame.get("sha256"):
                    raise VerificationFailure("Prediction PNG checksum mismatch")
                if index in (0, factor) and actual != digest(references[index]):
                    raise VerificationFailure("Exported endpoints differ from supplied endpoints")
                predicted.append(path)
            metrics = engine.interval([references[i] for i in range(factor + 1)], predicted,
                                      sample["width"], sample["height"], check_deadline=check_deadline)
            records.append({"id": sample["id"], "interpolation_factor": factor, **metrics})
            check_deadline()
            if on_interval:
                on_interval(ordinal, len(samples))
        if stream.read().strip():
            raise VerificationFailure("Unexpected extra prediction intervals")
    return {"protocol": PROTOCOL, "by_factor": aggregate(records), "intervals": records,
            "prediction_manifest_sha256": digest(root / "predictions.jsonl"),
            "metric_seconds": timeout_seconds - (deadline - time.monotonic())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--predictions-dir", required=True)
    parser.add_argument("--metric-bundle", required=True)
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cpu")
    parser.add_argument("--timeout-seconds", type=float, default=10800)
    parser.add_argument("--output", required=True, help="new raw metric report JSON")
    parser.add_argument("--write-calibration", help="new measured-baseline JSON bound to the selected runtime")
    parser.add_argument("--baseline-id")
    parser.add_argument("--baseline-revision", help="immutable checkpoint revision or submission digest")
    parser.add_argument("--selection-json", help="trusted fixed interval selection JSON; omitted means full export")
    args = parser.parse_args(argv)
    destinations = [Path(p).resolve() for p in (args.output, args.write_calibration) if p]
    if len(set(destinations)) != len(destinations) or any(p.exists() for p in destinations):
        parser.error("Report and calibration destinations must be distinct new files")
    if args.write_calibration and (not args.baseline_id or not args.baseline_revision):
        parser.error("Calibration requires --baseline-id and --baseline-revision")
    if args.write_calibration and args.device == "cuda":
        from vfi_runtime import check_cuda_runtime
        check_cuda_runtime()
    from vfi_metrics import MetricEngine, code_sha256
    selection = json.loads(Path(args.selection_json).read_text()) if args.selection_json else None
    _, identity = evaluation_identity(args.data, selection=selection)
    bundle_hash = digest(Path(args.metric_bundle) / "lock.json")
    engine = MetricEngine(args.metric_bundle, expected_sha256=bundle_hash, device=args.device)
    report = evaluate_export(args.data, args.predictions_dir, engine, timeout_seconds=args.timeout_seconds, selection=selection,
                             on_interval=lambda n, total: print(f"metrics: {n}/{total}", flush=True))
    report.update(evaluation_sha256=identity["evaluation_sha256"], metric_bundle_sha256=bundle_hash,
                  metric_code_sha256=code_sha256(), runtime=runtime_identity(args.device))
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(report, indent=2, allow_nan=False) + "\n")
    if args.write_calibration:
        baseline = {factor: {name: row[name] for name in METRICS} for factor, row in report["by_factor"].items()}
        validate_baseline(baseline)  # perfect references are a control, never a normalization floor
        record = {key: report[key] for key in ("protocol", "evaluation_sha256", "metric_bundle_sha256",
                                               "metric_code_sha256", "runtime", "prediction_manifest_sha256")}
        record.update(baseline=baseline, baseline_id=args.baseline_id, baseline_revision=args.baseline_revision)
        with Path(args.write_calibration).open("x") as stream:
            stream.write(json.dumps(record, indent=2, allow_nan=False) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
