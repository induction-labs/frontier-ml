"""Unscored, stdlib-only endpoint interpolation checks and prediction export."""
import base64
import hashlib
import json
import math
from pathlib import Path
import time

from frontier_ml.validation.cli import public_main
from frontier_ml.validation.evaluation import load_samples
from frontier_ml.validation.progress import progress
from vfi_protocol import (MAX_JSON_BYTES, RGB8_CONTENT_TYPE, build_request, normalize_sample,
                          parse_response, parse_rgb8_response, validate_sample)
from vfi_scoring import require_evaluation
from vfi_selection import select_intervals


def _seconds(config, key, default):
    value = config.get(key, default)
    if type(value) not in (int, float) or not math.isfinite(value) or value <= 0:
        raise ValueError(f"{key} must be positive and finite")
    return value


def run(transport, data_path, config):
    # No override can turn this interface diagnostic into an official evaluation.
    if config.get("validation_mode") is not True or config.get("predictions_only") is not True:
        require_evaluation()
    if type(config.get("concurrency", 1)) is not int or config.get("concurrency", 1) != 1:
        raise ValueError("Sequence interface checks require concurrency=1")
    budget = _seconds(config, "inference_timeout_seconds", 3600)
    request_budget = _seconds(config, "request_timeout_seconds", 300)
    response_format = config.get("response_format", "png")
    if response_format not in ("png", "rgb8"):
        raise ValueError("response_format must be png or rgb8")
    if not config.get("predictions_dir"):
        raise ValueError("--interface-only requires --predictions-dir (a new directory)")
    samples = select_intervals([normalize_sample(sample) for sample in load_samples(data_path)],
                               config.get("evaluation_selection"))
    for sample in samples:
        validate_sample(sample)
    root = Path(data_path).resolve().parent
    output = Path(config["predictions_dir"]).resolve()
    output.mkdir(parents=True, exist_ok=False)
    manifest = output / "predictions.jsonl"
    status_path = output / "run.json"
    status = {"status": "running", "score": None, "scoring": "not_configured",
              "samples_expected": len(samples), "samples_completed": 0,
              "predictions_file": "predictions.jsonl"}
    status_path.write_text(json.dumps(status, indent=2) + "\n")
    has_response_limit = hasattr(transport, "max_response_bytes")
    previous_limit = getattr(transport, "max_response_bytes", None)
    if has_response_limit:
        transport.max_response_bytes = MAX_JSON_BYTES
    started = time.monotonic()
    deadline = started + budget
    request_seconds, generated_frames = 0.0, 0
    factors = {}

    def remaining():
        seconds = deadline - time.monotonic()
        if seconds <= 0:
            raise TimeoutError("Interface check exceeded its whole-run deadline")
        return seconds

    try:
        with manifest.open("x", encoding="utf-8") as stream:
            # Read, send, and export one clip at a time; never accumulate a dataset
            # of decoded frames or base64 request/response bodies in memory.
            for ordinal, sample in enumerate(samples, 1):
                remaining()
                path, payload = build_request(sample, root)
                timeout = min(request_budget, remaining())
                request_started = time.monotonic()
                if response_format == "rgb8":
                    if has_response_limit:
                        transport.max_response_bytes = sample["width"] * sample["height"] * 3 * len(sample["target_indices"])
                    status_code, response = transport.raw_request(
                        "POST", path, body=json.dumps(payload, allow_nan=False).encode(),
                        headers={"Content-Type": "application/json", "Accept": RGB8_CONTENT_TYPE}, timeout=timeout)
                    if status_code != 200:
                        raise ValueError(f"Submission returned HTTP {status_code}")
                else:
                    response = transport.request("POST", path, payload, timeout=timeout)
                elapsed = time.monotonic() - request_started
                if elapsed > timeout:
                    raise TimeoutError("Sequence request exceeded its deadline")
                prediction = (parse_rgb8_response(sample, response) if response_format == "rgb8"
                              else parse_response(sample, response))
                del response
                remaining()
                request_seconds += elapsed
                generated_frames += len(prediction["frames"])
                sequence = output / f"{ordinal:06d}"
                sequence.mkdir()
                frames = {frame["index"]: frame[response_format] for frame in prediction["frames"]}
                for frame in payload["input_frames"]:
                    frames[frame["index"]] = base64.b64decode(frame["png"], validate=True)
                target_indices = set(sample["target_indices"])
                factor = sample["frame_count"] - 1
                record = {key: sample[key] for key in ("id", "schema_version", "width", "height", "frame_count", "target_indices")}
                record.update(input_indices=[frame["index"] for frame in sample["input_frames"]],
                              interpolation_factor=factor, request_seconds=elapsed, frames=[])
                for index in range(sample["frame_count"]):
                    raw = response_format == "rgb8" and index in target_indices
                    destination = sequence / f"{index:06d}{'.rgb' if raw else '.png'}"
                    with destination.open("xb") as image:
                        image.write(frames[index])
                    record["frames"].append({"index": index, "path": destination.relative_to(output).as_posix(),
                                             "generated": index in target_indices,
                                             "sha256": hashlib.sha256(frames[index]).hexdigest()})
                remaining()
                stream.write(json.dumps(record, allow_nan=False) + "\n")
                stream.flush()
                group = factors.setdefault(str(factor), {"samples": 0, "generated_frames": 0, "request_seconds": 0.0})
                group["samples"] += 1
                group["generated_frames"] += len(sample["target_indices"])
                group["request_seconds"] += elapsed
                status["samples_completed"] = ordinal
                progress(config, "interface_check", completed=ordinal, total=len(samples))
                del payload, prediction, frames
        remaining()
        total_seconds = time.monotonic() - started
        timing = {"elapsed_seconds": total_seconds, "request_seconds": request_seconds,
                  "generated_frames": generated_frames, "concurrency": 1,
                  "response_format": response_format,
                  "generated_frames_per_request_second": generated_frames / max(request_seconds, 1e-12),
                  "note": "Diagnostic HTTP timing only; no calibrated performance gate."}
        status.update(status="complete", timing=timing, by_factor=factors)
        return {"score": None, "status": "complete", "samples": len(samples),
                "predictions_file": str(manifest), "run_file": str(status_path),
                "details": {"mode": "interface_only", "scoring": "not_configured", "timing": timing,
                            "by_factor": factors}}
    except Exception as exc:
        status.update(status="failed", error=f"{type(exc).__name__}: {exc}")
        raise
    finally:
        if has_response_limit:
            transport.max_response_bytes = previous_limit
        status_path.write_text(json.dumps(status, indent=2, allow_nan=False) + "\n")


def _arguments(parser):
    parser.set_defaults(concurrency=1)
    parser.add_argument("--interface-only", action="store_true", help="check the protocol and export predictions without scoring")
    parser.add_argument("--predictions-dir", help="new output directory for complete frame sequences and their manifest")
    parser.add_argument("--request-timeout-seconds", type=float, default=300, help="diagnostic per-sequence timeout (default 300)")
    parser.add_argument("--response-format", choices=("rgb8", "png"), default="rgb8",
                        help="raw RGB binary output (default) or legacy PNG JSON")


def _config(args):
    return {"predictions_only": args.interface_only, "predictions_dir": args.predictions_dir,
            "request_timeout_seconds": args.request_timeout_seconds, "response_format": args.response_format}


def main(argv=None):
    return public_main("video-frame-interpolation", run, argv,
                       default_data="/task/public/validation.jsonl",
                       description="Video interpolation HTTP checks/export; quality metrics run in the separate trusted scorer.",
                       extra_arguments=_arguments, extra_config=_config)


if __name__ == "__main__":
    raise SystemExit(main())
