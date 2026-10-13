#!/usr/bin/env python3
"""Score your language-model server on the public validation documents.

    python /task/validate.py --server-url http://127.0.0.1:8000

Fetches GET /vocab, sends POST /score {"text"} for every validation document exactly as the trusted
client does, and reports bits per UTF-8 byte (bpb) overall and per genre, plus the reward, which is linear in bpb
(the same number the leaderboard records).
A document whose returned tokens do not reproduce its text, or whose log-probs are not finite and
<= 0, makes the run fail with that document named. This is the task metric only; it checks no
submission rules.
"""
import babylm
from frontier_ml.validation.cli import public_main

TASK = "babylm-replay"
DEFAULT_DATA = "/task/public/validation.jsonl"


def run(transport, data_path, config):
    """Bits per byte over every document (score = the linear reward, babylm.reward)."""
    return babylm.run(transport, data_path, config)


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
