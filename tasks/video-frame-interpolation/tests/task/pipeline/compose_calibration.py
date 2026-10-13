"""Freeze two compatible measured references as the production score calibration."""
import argparse
import json
from pathlib import Path

from vfi_evaluate import compose_calibration


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--zero", required=True, help="measured permitted starting-system receipt")
    parser.add_argument("--one", required=True, help="measured high-quality target receipt")
    parser.add_argument("--output", required=True, help="new production calibration file")
    args = parser.parse_args(argv)
    result = compose_calibration(json.loads(Path(args.zero).read_text()), json.loads(Path(args.one).read_text()))
    with Path(args.output).open("x") as stream:
        stream.write(json.dumps(result, indent=2, allow_nan=False) + "\n")


if __name__ == "__main__":
    main()
