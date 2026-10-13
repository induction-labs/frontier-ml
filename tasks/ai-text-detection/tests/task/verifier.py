"""Trusted verifier. All task gates live here (validate.py computes the metric and shows speed only):
preflight: trusted review receipt present; verify: receipt bound to the snapshot, then the public request
template and metric over a fixed stratified subset as one batch under the throughput deadline.
The measure is macro FNR at 1% FPR (details["macro_fnr_at_1pct_fpr"]); the Harbor reward ("score") is 1 - macro FNR.
selfcheck: the same requests and throughput deadline on resampled public passages, without the reviewer or a score."""
from collections import defaultdict
import hashlib
import json
from pathlib import Path
import random
import tempfile
import time

from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError
from frontier_ml.validation.evaluation import load_samples, run_http_evaluation

public = sibling(__file__, "validate.py", require=("run", "score"))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
MINIMUM_MESSAGES_PER_SECOND = public.REQUIRED_MESSAGES_PER_SECOND  # whole selected batch at the fixed concurrency
SAMPLING_VERSION = "ai-detection-evaluation-v1"
SELFCHECK_PUBLIC = ("validation.jsonl",)


def select_samples(samples, count):
    """One AI row per language/generator/condition/domain, with balanced human negatives."""
    if type(count) is not int or not 1 <= count <= len(samples):
        raise VerificationFailure(f"evaluation_samples must be an integer from 1 to {len(samples)}")
    if len({sample["id"] for sample in samples}) != len(samples):
        raise VerificationFailure("Evaluation sample IDs must be unique")

    def rank(value):
        return hashlib.sha256(f"{SAMPLING_VERSION}:{value}".encode()).digest(), value

    ai, humans = defaultdict(list), defaultdict(lambda: defaultdict(list))
    for sample in samples:
        if type(sample["label"]) is not int or sample["label"] not in (0, 1):
            raise VerificationFailure("Labels must be 0 (human) or 1 (AI)")
        if sample["label"]:
            ai[tuple(sample[key] for key in ("language", "family", "condition", "domain"))].append(sample)
        else:
            humans[sample["language"]][sample["domain"]].append(sample)
    if not ai or set(humans) != {key[0] for key in ai}:
        raise VerificationFailure("Each evaluation language needs both human and AI rows")
    selected = [min(rows, key=lambda row: rank(row["id"])) for rows in ai.values()]
    remaining = count - len(selected)
    languages = sorted(humans, key=rank)
    for index, language in enumerate(languages):
        quota = remaining // len(languages) + (index < remaining % len(languages))
        domains = sorted(humans[language], key=lambda domain: rank(f"{language}:{domain}"))
        for domain_index, domain in enumerate(domains):
            n = quota // len(domains) + (domain_index < quota % len(domains))
            rows = humans[language][domain]
            if not 1 <= n <= len(rows):
                raise VerificationFailure("Subset size cannot preserve balanced language/domain coverage")
            selected.extend(sorted(rows, key=lambda row: rank(row["id"]))[:n])
    return sorted(selected, key=lambda row: rank(row["id"]))


def preflight(config, data_path):
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")


def graded_batch(transport, data_path, config, score=None):
    """Every row of data_path as one batch at the task concurrency under the throughput deadline; grading and the
    self-check send their requests through this one path."""
    return run_http_evaluation(transport, data_path, config, build_request=public.build_request,
                               parse_response=public.parse_response, score=score or public.score,
                               failed_prediction=public.failed_prediction,
                               minimum_per_second=MINIMUM_MESSAGES_PER_SECOND)


def evaluate_with_gates(transport, data_path, config):
    """The public metric over the fixed subset, with the trusted throughput deadline."""
    config = dict(config, validation_mode=False)
    sampling = None
    with tempfile.TemporaryDirectory(prefix="ai-detection-evaluation-") as directory:
        if "evaluation_samples" in config:
            samples = load_samples(data_path)
            expected = config.get("expected_samples")
            if expected is not None and len(samples) != expected:
                raise VerificationFailure(f"Expected {expected} evaluation samples; found {len(samples)}")
            selected = select_samples(samples, config["evaluation_samples"])
            sampling = {"version": SAMPLING_VERSION, "available_samples": len(samples), "samples": len(selected),
                        "selected_ids_sha256": hashlib.sha256(json.dumps([r["id"] for r in selected]).encode()).hexdigest()}
            data_path = Path(directory) / "selected.jsonl"
            data_path.write_text("".join(json.dumps(row) + "\n" for row in selected))
            config["expected_samples"] = len(selected)
        result = graded_batch(transport, data_path, config)
    result["details"]["gates"] = {"throughput": result["details"]["timing"][0]}
    if sampling is not None:
        result["details"]["sampling"] = sampling
    return result


def verify(server, data_path, config):
    review = verify_review(server["snapshot"], config["architecture_review_path"], "ai-text-detection", REVIEW_CHECKS)
    result = evaluate_with_gates(server["transport"], data_path, config)
    result["details"]["gates"].update(architecture={"reviewer": review.get("reviewer"), "checks": review.get("checks")})
    return result


# Measured once: the subset select_samples grades is 9.18 MB of UTF-8 over eight languages, the resample below of the
# public examples 8.43 MB over seven. A byte-level or untruncated model does more work in grading than here.
HELDOUT_NOTE = ("Held-out passages also include Russian, which the public examples lack, and are about 9% longer in "
                "UTF-8 bytes, so keep more than that to spare.")


def public_rows(path):
    try:
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
    except (OSError, ValueError) as error:
        raise EvaluationError(f"The public examples could not be read: {error}") from error
    if not rows:
        raise EvaluationError("The public examples are empty")
    return rows


def resampled(rows, count):
    """`count` public rows for the throughput batch: shuffled passes over every example, IDs kept unique."""
    if type(count) is not int or count < 1:
        raise EvaluationError("evaluation_samples must be a positive integer")
    rng, picked = random.Random(SAMPLING_VERSION), []
    while len(picked) < count:
        deck = list(rows)
        rng.shuffle(deck)
        picked.extend(deck)
    return [dict(row, id=f"{row['id']}#{index}") for index, row in enumerate(picked[:count])]


def ignore_scores(samples, predictions, config):
    return {"score": None, "details": {}}


class FirstRequest:
    """The transport for the smoke request, timed as grading times its batch: on the trusted HTTP client's clock in
    the GPU sandbox when the transport has one, so relay and control-plane delays are not charged to the server;
    on the host clock otherwise (local runs). check.request retries a transport failure as grading does, and grading
    charges a retry's backoff to the batch, so `seconds` also counts everything before the answering attempt
    (earlier attempts and the backoff) on the host clock."""

    def __init__(self, transport, deadline):
        self.transport, self.deadline, self.started, self.seconds = transport, deadline, None, None

    def request(self, method, path, payload=None, *, timeout):
        started = time.monotonic()
        if self.started is None:
            self.started = started
        try:
            if callable(timed := getattr(self.transport, "request_timed", None)):
                response, seconds = timed(method, path, payload, timeout=timeout)
            else:
                response = self.transport.request(method, path, payload, timeout=timeout)
                seconds = time.monotonic() - started
        except TimeoutError as error:
            raise TimeoutError(f"no answer within the {self.deadline:g} s grading allows from the first request to "
                               f"the last of the batch ({error})") from error
        self.seconds = started - self.started + seconds
        return response


def selfcheck(check, public_dir, config):
    """One public passage through /detect, then the throughput gate: a batch as large as the graded subset (5,000
    passages in 500 s, within the default self-check budget, so it is not sampled), built from the public passages
    and sent through the graded path at its concurrency. Grading sends its batch to a cold server right after
    /health, so the passage and the batch share grading's one deadline (evaluation_samples / 10 s, capped by the
    inference timeout). Nothing is scored. selfcheck.run already ran the review's free architecture.json checks;
    prescreen and preflight add nothing static: preflight only checks the review receipt."""
    rows = public_rows(Path(public_dir) / "validation.jsonl")
    batch = resampled(rows, config.get("evaluation_samples"))
    deadline = min(config.get("inference_timeout_seconds", 3600), len(batch) / MINIMUM_MESSAGES_PER_SECOND)
    allowed = f"grading allows {deadline:g} s from the first request for {len(batch)} held-out passages"
    with check.launch() as server:
        path, payload = public.build_request(rows[0], public_dir)
        first = FirstRequest(server["transport"], deadline)
        check.request(dict(server, transport=first), path, payload, parse=public.parse_response, sample=rows[0],
                      timeout=deadline)
        check.stage("gates")
        check.note(HELDOUT_NOTE)
        remaining = deadline - first.seconds
        if remaining <= 0:  # a host clock can overrun the socket timeout slightly
            check.fail("throughput", f"the first request took {first.seconds:.1f} s; {allowed}",
                       required=MINIMUM_MESSAGES_PER_SECOND, unit="passages/s")
        with tempfile.TemporaryDirectory(prefix="ai-detection-selfcheck-") as directory:
            data = Path(directory) / "batch.jsonl"
            data.write_text("".join(json.dumps(row) + "\n" for row in batch))
            try:
                result = graded_batch(server["transport"], data, dict(
                    config, validation_mode=False, expected_samples=len(batch), inference_timeout_seconds=remaining),
                    ignore_scores)
            except EvaluationError:
                raise
            except VerificationFailure as error:  # the deadline, or more than 5% of the requests failed
                if str(error).startswith("Batch exceeded"):
                    check.fail("throughput", f"{len(batch)} passages at concurrency {config['concurrency']} did not "
                               f"finish in the {remaining:.1f} s left after a first request of {first.seconds:.1f} s; "
                               f"{allowed}", required=MINIMUM_MESSAGES_PER_SECOND, unit="passages/s")
                check.fail("batch responses", str(error), kind="smoke")
            except RecursionError as error:  # a response nested too deeply to decode on the batch path
                check.fail("batch responses", f"a response could not be decoded: {error}", kind="smoke")
        timing, failed = result["details"]["timing"][0], result["details"]["failed_requests"]
        elapsed = first.seconds + timing["elapsed_seconds"]
        check.gate("throughput", len(batch) / elapsed, MINIMUM_MESSAGES_PER_SECOND, higher_is_better=True,
                   unit="passages/s", detail=f"{len(batch)} public passages (the {len(rows)} examples resampled) at "
                   f"concurrency {timing['concurrency']}: {first.seconds:.1f} s for the first request, then "
                   f"{timing['elapsed_seconds']:.1f} s for the batch; {allowed}")
        if failed["total"]:
            check.record("batch responses", False, kind="smoke",
                         detail=f"{failed['total']} of {len(batch)} requests failed, timed out or were rejected "
                                f"({failed['reasons']}); grading scores each as a wrong answer and fails the run "
                                f"above {failed['limit_percent']}%. First errors: {failed['errors']}")
        else:
            check.record("batch responses", True, kind="smoke", detail=f"all {len(batch)} responses parsed")
