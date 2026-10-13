#!/usr/bin/env python3
"""Free renderer floor probes; does not claim a model or paid-judge baseline."""
import argparse
import importlib.util
import json
from pathlib import Path
import sys

SEED = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(SEED))
from svg_render import render_svg, renderer_binary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", required=True)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    spec = importlib.util.spec_from_file_location("svg_public", SEED / "validate.py")
    public = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(public)
    samples = [json.loads(line) for line in Path(args.data).read_text().splitlines()]
    blank = '<svg viewBox="0 0 100 100"></svg>'
    solid = '<svg viewBox="0 0 100 100"><rect width="100" height="100" fill="red"/></svg>'
    def no_judge(**payload):
        raise RuntimeError("Uniform outputs must never invoke a judge")
    results = {}
    for name, svg in (("blank_canvas", blank), ("solid_red_canvas", solid)):
        result = public.score(samples, [{"id": row["id"], "svg": svg} for row in samples], {"judge": no_judge})
        results[name] = {"score": result["score"], "samples": len(samples), "paid_calls": 0}
    report = {"protocol": public.PROTOCOL, "renderer": renderer_binary()[1], "floor_probes": results,
              "model_baseline_measured": False, "judge_calibrated": False, "h100_measured": False,
              "scope": "Mechanical rendering floors only; no semantic judge or trained model was run."}
    Path(args.output).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
