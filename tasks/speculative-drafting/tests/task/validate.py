#!/usr/bin/env python3
"""Evaluate your draft server on the public validation prompts.

    python /task/validate.py --server-url http://127.0.0.1:8000

For every prompt this replays greedy speculative decoding against the target's frozen greedy continuation
(`reference`), exactly as the trusted final client does. Each request is POST /draft {"tokens": [...]}, the
prompt's token ids plus the continuation accepted so far. The target accepts the longest prefix of the returned
draft that matches the continuation and then adds one token of its own. The score is (tau - 4.6) / 2.4, clamped to
[0, 1], where tau is the mean acceptance length: continuation tokens divided by verification steps over the whole set.

Every active prompt sends one request per round, so requests are concurrent across prompts. Serving time is
reported here, scaled to the held-out set's 44,314 continuation tokens (five times the public set), but not
enforced; the trusted grader enforces the serving budget and the submission size limit in /task/CONTRACT.md.
"""
from frontier_ml.validation.cli import public_main
from frontier_ml.validation.evaluation import load_samples
from frontier_ml.validation.errors import VerificationFailure
from frontier_ml.validation.progress import progress
from specdraft_walk import score, walk

TASK = "speculative-drafting"
DEFAULT_DATA = "/task/public/validation.jsonl"
CONCURRENCY = 32
HELDOUT_CONVERSATIONS = 200
HELDOUT_TOKENS = 44_314             # continuation tokens in the held-out set; requests scale with tokens / tau
SERVING_BUDGET_SECONDS = 10_800     # the trusted grader's budget for the held-out set


def heldout_projection(timing, tokens):
    """Serving time on this set scaled to the held-out continuation tokens (same drafter, same server)."""
    projected = timing["serving_seconds"] * HELDOUT_TOKENS / tokens
    return {"projected_heldout_serving_seconds": projected, "heldout_serving_budget_seconds": SERVING_BUDGET_SECONDS,
            "projected_within_heldout_budget": projected <= SERVING_BUDGET_SECONDS}


def run(transport, data_path, config):
    samples = load_samples(data_path)
    expected = config.get("expected_samples")
    if expected is not None and len(samples) != expected:
        raise VerificationFailure(f"Expected {expected} evaluation samples; found {len(samples)}")
    diagnostic = config.get("validation_mode") is True
    budget = config.get("serving_budget_seconds", config.get("inference_timeout_seconds", 3600))
    traces, timing, failures = walk(
        transport, samples, concurrency=config.get("concurrency", CONCURRENCY), serving_budget_seconds=budget,
        enforce_budget=not diagnostic,
        on_round=lambda rounds, done, total: progress(config, "rounds", completed=done, total=total,
                                                      message=f"round {rounds}"))
    result = score(samples, traces)
    if diagnostic:
        timing.update(heldout_projection(timing, result["details"]["tokens"]))
        if not timing["projected_within_heldout_budget"]:
            result["details"]["warnings"] = [
                f"Projected held-out serving time is {timing['projected_heldout_serving_seconds']:.0f} s, over the "
                f"{SERVING_BUDGET_SECONDS} s budget: the final evaluation would score zero. Serve faster (batch "
                "concurrent requests, merge adapters) or use a smaller drafter."]
    result["details"]["timing"] = [timing]
    result["details"]["failed_requests"] = failures
    return result


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
