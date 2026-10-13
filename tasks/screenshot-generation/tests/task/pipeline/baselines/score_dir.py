"""Score a directory of generated PNGs (named <id>.png) against a split with the task metric.

python score_dir.py SPLIT_JSONL IMAGE_DIR OUT_JSON [--components T L F R] [--real-from-split] [--limit N]

F and R call gpt-6-luna through the operator's OpenRouter key with a spend ledger (pipeline/judge_local.py).
"""
import argparse
import io
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the task directory: task-local modules by bare name
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import screenshot_generation as sg  # noqa: E402

MODELS = "/scratch/frontier-ml-workspace/work/screenshot-generation/models"


def png_bytes(path):
    image = Image.open(path).convert("RGB")
    buffer = io.BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split")
    parser.add_argument("images")
    parser.add_argument("out")
    parser.add_argument("--components", nargs="+", default=list(sg.COMPONENTS))
    parser.add_argument("--real-from-split", action="store_true", help="score the split's own real screenshots")
    parser.add_argument("--limit", type=int)
    parser.add_argument("--tag", default="baseline")
    args = parser.parse_args()
    samples = [json.loads(l) for l in open(args.split) if l.strip()][:args.limit]
    root = Path(args.split).parent
    pngs = [png_bytes(root / s["image"] if args.real_from_split else Path(args.images) / f"{s['id']}.png") for s in samples]
    config = {"scorer_models_dir": MODELS, "components": args.components, "judge_concurrency": 16}
    if set(args.components) & {"F", "R"}:
        from judge_local import make_judge, spent
        before = spent()
        config["judge"] = make_judge(args.tag)
    result = sg.score_images(samples, pngs, config)
    result.pop("images")
    if "judge" in config:
        result["judge_spend_usd"] = spent() - before
    latency = Path(args.images) / "latency.json"
    if latency.exists():
        result["latency"] = json.loads(latency.read_text())
    Path(args.out).write_text(json.dumps(result, indent=1))
    print(json.dumps({"score": result["score"], "means": result["means"], "latency": result.get("latency"),
                      "judge_spend_usd": result.get("judge_spend_usd")}, indent=1))


if __name__ == "__main__":
    main()
