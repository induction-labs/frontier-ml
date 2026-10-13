#!/usr/bin/env python3
"""Play Gen 9 Random Battle against your POST /act endpoint. Requires Node.js >=18.

    python /task/validate.py --server-url http://127.0.0.1:8000

Runs 32 games with fresh random teams and battle seeds on each invocation.
Never asks the server to report its own results or scores.
Public validation reports raw mean battle points, not the normalized final reward.
Final reward is max(0, min(1, (mean_battle_points - 0.5) / 0.5)).
Public validation does not certify policy provenance or information compliance.
By default, two opponent workers finish before the endpoint is called, avoiding
CPU contention on the small agent VM. --move-seconds can relax the deadline for
quality debugging; it never certifies the final 1-second H100 gate. For separate
opponent/endpoint hosts, use --opponent-workers 16 --parallel-decisions. Final
grading always uses separate hosts and times HTTP locally on the H100.
"""
import secrets

from frontier_ml.validation.cli import public_main
from pokemon_random_battle import FOUL_PLAY_WORKERS, MOVE_SECONDS, evaluate

TASK = "pokemon-random-battle"
DEFAULT_DATA = "/task/public/validation.jsonl"
VALIDATION_SECONDS = 3600


def run(transport, data_path, config):
    return evaluate(transport, data_path, seed=secrets.randbits(256),
        seconds=min(float(config.get("inference_timeout_seconds", VALIDATION_SECONDS)), VALIDATION_SECONDS),
        opponent_workers=config.get("opponent_workers", 2),
        parallel_decisions=config.get("parallel_decisions", False),
        move_seconds=config.get("move_seconds", MOVE_SECONDS))


def arguments(parser):
    parser.add_argument("--opponent-workers", type=int, choices=range(1, FOUL_PLAY_WORKERS + 1), default=2,
                        help="public opponent CPU workers (default 2; final uses 16 on its own host)")
    parser.add_argument("--parallel-decisions", action="store_true",
                        help="overlap opponent and candidate decisions; use separate CPU allocations")
    parser.add_argument("--move-seconds", type=float, default=MOVE_SECONDS,
                        help="diagnostic response deadline, at most 30 seconds (default 1; final always 1)")


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__,
                       extra_arguments=arguments, extra_config=lambda args: {
                           "opponent_workers": args.opponent_workers,
                           "parallel_decisions": args.parallel_decisions, "move_seconds": args.move_seconds})


if __name__ == "__main__":
    raise SystemExit(main())
