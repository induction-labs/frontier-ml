"""python -m spin.server is spin serve."""
import sys

from ..cli import main

raise SystemExit(main(["serve", *sys.argv[1:]]))
