"""Metric subprocess, run with the scorer's pinned Python environment."""
import argparse
import json
import os
from pathlib import Path
import tempfile
from concurrent.futures import ThreadPoolExecutor

from scitikz_metric import CodeMetrics, ImageMetrics, aggregate
from scitikz_models import check_models
from scitikz_render import render_answers, render_tex


def check(config):
    import torch
    import crystalbleu
    import lpips
    import dreamsim
    import pytorch_msssim
    from torchmetrics.functional.text.eed import _compute_sentence_statistics
    check_models(config["scorer_models_dir"])
    if config.get("scorer_device", "cuda").startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("The trusted metric scorer requires a CUDA GPU")
    if config.get("require_isolation"):
        if os.geteuid() != 0 or not Path(config["tex_launcher"]).is_file():
            raise RuntimeError("The trusted TeX isolation launcher is unavailable")
        with tempfile.TemporaryDirectory() as directory:
            rendered = render_tex(r"\documentclass{standalone}\usepackage{tikz}\begin{document}\begin{tikzpicture}\draw (0,0)--(1,1);\end{tikzpicture}\end{document}",
                                  Path(directory)/"probe.png", 9000, config)
        if not rendered["compiled"]:
            raise RuntimeError("TeX renderer health check failed: "+str(rendered))


def evaluate(samples, predictions, config):
    import torch
    torch.set_num_threads(4)
    if not samples or len(samples) != len(predictions):
        raise ValueError("Expected one prediction per example")
    for sample, prediction in zip(samples, predictions, strict=True):
        if sample["id"] != prediction.get("id") or not isinstance(prediction.get("answer"), str):
            raise ValueError("Prediction ID/order or answer mismatch")
    check_models(config["scorer_models_dir"])
    with tempfile.TemporaryDirectory(prefix="scitikz-renders-") as directory:
        records = render_answers(predictions, directory, config)
        for prediction, record in zip(predictions, records, strict=True):
            if prediction.get("request_failed"):  # validate.failed_prediction: the empty answer, and why
                record.update(failure="request_failed", request_failed=prediction["request_failed"])
        code_metrics = CodeMetrics([row["code"] for row in samples])  # the whole split, whatever failed
        # Warm up the exact EED recurrence before running independent CPU work.
        code_metrics.compute("", "")
        with ThreadPoolExecutor(max_workers=4) as pool:
            values = pool.map(lambda pair: code_metrics.compute(pair[0]["code"], pair[1]["source"]), zip(samples, records))
            for record, metrics in zip(records, values, strict=True):
                record.update(metrics)
        successful = [(sample, record) for sample, record in zip(samples, records) if record["compiled"]]
        if successful:
            root = Path(config["data_root"]).resolve()
            pairs = []
            for sample, record in successful:
                path = (root/sample["image"]).resolve()
                if not path.is_relative_to(root):
                    raise ValueError("Reference image escapes data root")
                pairs.append((str(path), record["png"]))
            image_metrics = ImageMetrics(config["scorer_models_dir"], config.get("scorer_device", "cuda"))
            for (_, record), values in zip(successful, image_metrics.compute(pairs), strict=True):
                record.update(values)
        for record in records:
            record.pop("png", None)
            record.pop("source", None)
        result = aggregate(records)
        build = Path(config.get("tex_build", "/opt/scitikz/tex-build.json"))
        if build.is_file():
            result["details"]["tex_build"] = json.loads(build.read_text())
        return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    payload = json.loads(Path(args.input).read_text())
    if args.check:
        check(payload["config"])
        result = {"ready": True}
    else:
        result = evaluate(payload["samples"], payload["predictions"], payload["config"])
    Path(args.output).write_text(json.dumps(result, allow_nan=False)+"\n")


if __name__ == "__main__":
    main()
