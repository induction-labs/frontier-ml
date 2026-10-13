#!/usr/bin/env python3
"""Evaluate your music transcription server on the public example clips.

    python /task/validate.py --server-url http://127.0.0.1:8000
"""
import base64
from pathlib import Path

from frontier_ml.validation.cli import public_main
from frontier_ml.validation.evaluation import load_samples, run_http_evaluation
from amt_notes import aggregate, clip_counts, parse_notes

TASK = "music-transcription"
DEFAULT_DATA = "/task/public/validation.jsonl"
CONCURRENCY = 16
MINIMUM_REAL_TIME_FACTOR = 50.0


def build_request(sample, data_root):
    root = Path(data_root).resolve()
    path = (root / sample["audio"]).resolve()
    if root not in path.parents or path.suffix != ".flac":
        raise ValueError(f"{sample['id']}: audio must be a .flac file inside the data folder")
    return "/transcribe", {"audio_format": "flac", "audio": base64.b64encode(path.read_bytes()).decode("ascii")}


def parse_response(sample, response):
    return {"id": sample["id"], "notes": parse_notes(response)}


def failed_prediction(sample, failure):
    """A clip whose request failed or answered malformed predicts no notes: all its reference notes are missed."""
    return {"id": sample["id"], "notes": []}


def score(samples, predictions, config=None):
    if not samples or len(samples) != len(predictions):
        raise ValueError("Expected exactly one prediction per clip")
    records = []
    for sample, prediction in zip(samples, predictions, strict=True):
        if prediction.get("id") != sample["id"]:
            raise ValueError("Prediction ID/order mismatch")
        counts = clip_counts(parse_notes({"notes": sample["notes"]}), prediction["notes"])
        records.append({"id": sample["id"], **counts})
    result = aggregate(records)
    return {"score": result.pop("score"),
            "details": {"metric": "instrument_macro_onset_f1", "clips": len(records), **result, "records": records}}


def minimum_clips_per_second(samples):
    """The 50x real-time rule as a whole-batch rate: the deadline is total audio seconds / 50."""
    seconds = sum(float(sample["duration_s"]) for sample in samples)
    return MINIMUM_REAL_TIME_FACTOR * len(samples) / seconds


def run(transport, data_path, config):
    config = {"concurrency": CONCURRENCY, **dict(config or {})}
    samples = load_samples(data_path)
    result = run_http_evaluation(transport, data_path, config, build_request=build_request,
                                 parse_response=parse_response, score=score, failed_prediction=failed_prediction,
                                 minimum_per_second=minimum_clips_per_second(samples), default_concurrency=CONCURRENCY)
    timing = (result.get("details", {}).get("timing") or [None])[0]
    if isinstance(timing, dict) and timing.get("elapsed_seconds"):
        audio = sum(float(sample["duration_s"]) for sample in samples)
        timing.update({"audio_seconds": audio, "real_time_factor": audio / timing["elapsed_seconds"],
                       "required_real_time_factor": MINIMUM_REAL_TIME_FACTOR})
    return result


def main(argv=None):
    # Default to the graded concurrency so the reported real-time factor is comparable to the final run.
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__,
                       extra_arguments=lambda parser: parser.set_defaults(concurrency=CONCURRENCY))


if __name__ == "__main__":
    raise SystemExit(main())
