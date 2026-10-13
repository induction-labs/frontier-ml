"""Free organizer sanity baselines; no model downloads, inference, training or API calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from zsc_metric import score
from zsc_protocol import prepare_samples


def measure(path):
    samples = prepare_samples([json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()])
    outputs = {}
    for name in ("uniform", "confident_first", "oracle"):
        predictions = []
        for row in samples:
            count = len(row["labels"])
            values = [1 / count] * count if name == "uniform" else [0.0] * count
            if name != "uniform":
                values[row["target"] if name == "oracle" else 0] = 1.0
            predictions.append({"id": row["id"], "probabilities": values})
        result = score(samples, predictions)
        outputs[name] = {"score": result["score"],
                         **{key: result["details"][key] for key in ("samples", "datasets", "macro_f1", "ece", "accuracy")}}
    return outputs


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=Path("data/zero-shot-classification"))
    parser.add_argument("--output", type=Path)
    args = parser.parse_args(argv)
    results = {"protocol": "zero-shot-classification-v2", "throughput_calibrated": False,
               "trained_model_baseline": None,
               "note": "Sanity baselines only. Uniform and confident-first have no trained classifier; oracle reads gold labels. All fail submission compliance.",
               "validation": measure(args.data / "public/validation.jsonl"),
               "heldout": measure(args.data / "private/test.jsonl")}
    rendered = json.dumps(results, indent=2) + "\n"
    if args.output:
        args.output.write_text(rendered)
    print(rendered)


if __name__ == "__main__":
    main()
