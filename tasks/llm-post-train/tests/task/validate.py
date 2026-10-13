#!/usr/bin/env python3
"""Score your server on the development benchmarks the way the final evaluation does.

    python /task/validate.py --server-url http://127.0.0.1:8000 --benchmarks gpqa_diamond --limit 20
    python /task/validate.py --server-url http://127.0.0.1:8000 --benchmarks livecodebench_v6 --limit 10
    python /task/validate.py --server-url http://127.0.0.1:8000

Uses the same prompts, sampling settings (16,384 output tokens at temperature 0.7, 32 concurrent
requests, 250 seconds per request), answer extraction and output comparison as the final
evaluation for GPQA Diamond and LiveCodeBench v6 from /task/public/benchmarks, and reports each
benchmark's accuracy and their mean. The final score also includes two other benchmarks,
so this mean is not comparable to it.

WARNING: LiveCodeBench executes the code your model generates on this machine, in local
subprocesses with six-second watchdogs and best-effort resource limits but no user,
network or filesystem isolation. Run it only in a disposable sandbox.

`--limit` truncates each benchmark for quick smoke tests. It measures quality only; the
requirements stated in /task/CONTRACT.md (the weight audit and the review) are checked separately
during final evaluation.
"""
from frontier_ml.validation import language
from frontier_ml.validation.cli import public_main

TASK = "llm-post-train"
DEFAULT_DATA = "/task/public/benchmarks"


def run(transport, benchmarks_dir, config):
    return language.evaluate_public(transport, benchmarks_dir, config)


def _arguments(parser):
    parser.add_argument("--benchmarks", nargs="+", choices=list(language.PUBLIC_FILES),
                        help="subset to run; default both development benchmarks")
    parser.add_argument("--limit", type=int, help="evaluate only the first N rows of each benchmark")
    parser.add_argument("--max-tokens", type=int, default=16384, help="completion budget per request")


def _config(args):
    # The suite's defaults are llm-diffusion's settings; this task samples at 0.7 with a 16,384-token budget.
    return {"benchmarks": args.benchmarks, "limit": args.limit, "max_tokens": args.max_tokens, "temperature": 0.7,
            "request_timeout_seconds": 250, "generation_concurrency": min(args.concurrency, 32)}


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__,
                       extra_arguments=_arguments, extra_config=_config)


if __name__ == "__main__":
    raise SystemExit(main())
