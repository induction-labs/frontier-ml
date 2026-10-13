"""Operator-only local calibration against the real simulator, without model calls."""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from pokemon_random_battle import DEVELOPMENT_SEED, evaluate


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=DEVELOPMENT_SEED)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--policy", action="append", choices=("random", "power", "type", "tactical", "stochastic"))
    args = parser.parse_args()
    results = {}
    for style in args.policy or ("random", "power", "type", "tactical", "stochastic"):
        result = evaluate(None, args.data, seed=args.seed, baseline=style)
        results[style] = result
        print(json.dumps({"policy": style, "score": result["score"],
            "seconds": result["details"]["timing"]["elapsed_seconds"],
            "capped_games": result["details"]["capped_games"]}), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
