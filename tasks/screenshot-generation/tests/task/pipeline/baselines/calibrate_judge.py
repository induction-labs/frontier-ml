"""Calibrate the realism/faithfulness rubrics: rate several image sets for the same samples (and repeat one set
to measure run-to-run noise). Usage:

    python calibrate_judge.py SPLIT_JSONL OUT_JSON --n 30 --set real=DIR --set zimage=DIR ... [--repeat real]
"""
import argparse
import json
import sys
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import screenshot_generation as sg  # noqa: E402
from judge_local import make_judge, spent  # noqa: E402


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("split")
    parser.add_argument("out")
    parser.add_argument("--n", type=int, default=30)
    parser.add_argument("--set", action="append", required=True, help="name=image_dir (files named <id>.png)")
    parser.add_argument("--repeat", action="append", default=[])
    parser.add_argument("--real-from-split", action="store_true", help="the set named real uses the split's image field")
    args = parser.parse_args()
    rows = [json.loads(l) for l in open(args.split) if l.strip()][:args.n]
    root = Path(args.split).parent
    judge = make_judge("calibration")
    before = spent()
    results = {}
    runs = [(item.split("=", 1)[0], item.split("=", 1)[1]) for item in args.set]
    runs += [(name + "#repeat", dict(runs)[name]) for name in args.repeat]
    for name, directory in runs:
        base = name.split("#")[0]
        images = []
        for r in rows:
            path = (root / r["image"]) if (base == "real" and args.real_from_split) else Path(directory) / f"{r['id']}.png"
            images.append(Image.open(path).convert("RGB"))
        ratings = sg.judge_ratings(judge, [(im, r["description"]) for im, r in zip(images, rows)])
        R = [x["R"]["rating"] for x in ratings]
        F = [x["F"]["rating"] for x in ratings]
        results[name] = {"R_mean": sum(R) / len(R), "F_mean": sum(F) / len(F), "R_min": min(R), "F_min": min(F),
                         "R": R, "F": F}
        print(f"{name:14s} R {results[name]['R_mean']:.3f} (min {min(R):.2f})  F {results[name]['F_mean']:.3f} (min {min(F):.2f})", flush=True)
    for name in args.repeat:
        a, b = results[name], results[name + "#repeat"]
        dr = max(abs(x - y) for x, y in zip(a["R"], b["R"]))
        df = max(abs(x - y) for x, y in zip(a["F"], b["F"]))
        print(f"repeat {name}: max |dR| {dr:.3f}, max |dF| {df:.3f}; mean dR {abs(a['R_mean'] - b['R_mean']):.4f}, mean dF {abs(a['F_mean'] - b['F_mean']):.4f}")
    Path(args.out).write_text(json.dumps(results, indent=1))
    print(f"judge spend this run ${spent() - before:.4f}; total ${spent():.4f}")


if __name__ == "__main__":
    main()
