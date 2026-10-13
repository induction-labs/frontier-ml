#!/usr/bin/env python3
"""Validate scientific-image to TikZ reconstruction on the allowed public examples.

Use --predictions-only without metric dependencies. For full scoring, install the
public scorer-requirements.txt and TeX packages described in public/SCHEMA.md,
prepare the frozen metric weights with scitikz_models.py, then set --models-dir.
Final scoring adds the offline resource gate and training/provenance review.
"""
import json
from pathlib import Path
import subprocess
import sys
import tempfile

from frontier_ml.validation.cli import public_main
from frontier_ml.validation.evaluation import chat_answer, image_url, load_samples, run_http_evaluation
import scitikz_worker

TASK = "scitikz"
DEFAULT_DATA = "/task/public/validation.jsonl"
PROMPT = "Generate precise, well-structured TikZ/LaTeX code to faithfully recreate the image. The code must be complete and compilable."


def build_request(sample, data_root):
    return "/v1/chat/completions", {"model": "submission", "temperature": 0, "max_tokens": 16384,
        "stream": False, "messages": [{"role": "user", "content": [
            {"type": "image_url", "image_url": {"url": image_url(sample["image"], data_root)}},
            {"type": "text", "text": PROMPT}]}]}


def parse_response(sample, response):
    return {"id": sample["id"], "answer": chat_answer(response)}


def failed_prediction(sample, failure):
    """A figure whose request failed after the shared retries, timed out or returned a malformed completion
    (see CONTRACT.md) is scored as the empty answer: it does not compile, and its code metrics are the empty
    program's. The row stays in every denominator and the CrystalBLEU reference corpus is built from the whole
    split, so a failure can never score better than answering (F2, 2026-09-30)."""
    return {"id": sample["id"], "answer": "", "request_failed": {"reason": failure.reason, "error": failure.error}}


def worker(payload, config, *, check=False):
    with tempfile.TemporaryDirectory(prefix="scitikz-score-") as directory:
        job, output = Path(directory)/"input.json", Path(directory)/"output.json"
        job.write_text(json.dumps(dict(payload, config=config)))
        command = [config.get("scorer_python", sys.executable), scitikz_worker.__file__,
                   "--input", str(job), "--output", str(output)]
        if check:
            command.append("--check")
        with (Path(directory)/"log").open("w+") as log:
            process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT,
                                     timeout=300 if check else config.get("scoring_timeout_seconds", 3600))
            if process.returncode:
                log.seek(0)
                raise RuntimeError("Scorer failed: "+log.read()[-6000:])
        return json.loads(output.read_text())


def score(samples, predictions, config):
    return worker({"samples": samples, "predictions": predictions}, config)


def run(transport, data_path, config):
    config = dict(config, data_root=str(Path(data_path).resolve().parent))
    if config.get("prediction_file"):
        if not config.get("validation_mode"):
            raise ValueError("Saved predictions are for public diagnostics only")
        data = json.loads(Path(config["prediction_file"]).read_text())
        predictions = data["predictions"] if isinstance(data, dict) else data
        return score(load_samples(data_path), predictions, config)
    return run_http_evaluation(transport, data_path, config, build_request=build_request, parse_response=parse_response,
                               score=score, failed_prediction=failed_prediction, default_concurrency=16)


def arguments(parser):
    parser.set_defaults(concurrency=16, timeout_seconds=7200)
    parser.add_argument("--predictions-only", action="store_true")
    parser.add_argument("--predictions", dest="prediction_file", help="score saved public predictions")
    parser.add_argument("--models-dir", default="/opt/scitikz/models")
    parser.add_argument("--scorer-python", default=sys.executable)
    parser.add_argument("--device", default="cuda")


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__,
        extra_arguments=arguments, extra_config=lambda a: {"predictions_only": a.predictions_only,
            "prediction_file": a.prediction_file, "scorer_models_dir": a.models_dir,
            "scorer_python": a.scorer_python, "scorer_device": a.device})


if __name__ == "__main__":
    raise SystemExit(main())
