"""Trusted verifier: the snapshot-bound LLM review (review.py POLICY: data provenance and no reference lookup), then
the public template and metric over the held-out pairs with the operator's judge."""
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import time

from frontier_ml.runtime.judge import judge_from_config
from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError

public = sibling(__file__, "validate.py", require=("run",))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
SLUG = "screen-difference"
# Self-check (frontier_ml/selfcheck.py): the public pairs and their screenshots, nothing else.
# The pinned public folder keeps its images under media/; a re-prepared copy uses images/.
SELFCHECK_PUBLIC = ("validation.jsonl", "media/*", "images/*")
SMOKE_TIMEOUT_SECONDS = 60  # one request and its retries; bounded so a hung server cannot hold the self-check
BATCH_WINDOW = 4  # the timed batch may run 4x the grading deadline, so a slow server still learns its rate


def preflight(config, data_path):
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")
    # The judge credential must exist before a GPU is provisioned.
    config.setdefault("judge_model", public.JUDGE_MODEL)
    config["judge"] = judge_from_config(config)


def verify(server, data_path, config):
    review = verify_review(server["snapshot"], config["architecture_review_path"], SLUG, REVIEW_CHECKS)
    result = public.run(server["transport"], data_path, config)
    result["details"]["gates"] = {"architecture": {"reviewer": review.get("reviewer"), "checks": review.get("checks")}}
    return result


def selfcheck(check, public_dir, config):
    """The graded batch on public pairs, then one public pair through the parser; no judge, so only shapes are checked.

    The batch goes first because in grading it is the first traffic after /health: a server that compiles on its
    first requests pays for that inside the deadline. preflight checks only the review receipt and the judge credential,
    so beyond the review prerequisites selfcheck.run checks first there is no static check to run. The 70-request batch
    has a 14 s deadline, so it runs at full size."""
    check.note("No judge read the descriptions: they were checked only for parsing and being non-empty.")
    pairs = _public_pairs(public_dir)
    with tempfile.TemporaryDirectory(prefix="screen-selfcheck-") as directory:
        count = config.get("expected_samples") or len(pairs)
        data = _graded_shape(pairs, Path(public_dir), Path(directory), count)
        with check.launch() as server:
            check.stage("gates")
            _batch(check, server, data, count, len(pairs), config)
            check.stage("smoke")
            path, payload = public.build_request(pairs[0], public_dir)
            check.request(server, path, payload, parse=_described, sample=pairs[0], timeout=SMOKE_TIMEOUT_SECONDS)


def _public_pairs(public_dir):
    try:
        pairs = [json.loads(line) for line in (Path(public_dir) / "validation.jsonl").read_text().splitlines()
                 if line.strip()]
    except (OSError, ValueError) as error:
        raise EvaluationError(f"The public pairs are unreadable: {type(error).__name__}: {error}") from error
    if not pairs:
        raise EvaluationError("The public split has no pairs")
    return pairs


def _graded_shape(pairs, public_dir, directory, count):
    """`count` pairs (the held-out size) cycled from the public ones under unique ids, beside copies of their
    screenshots: image references resolve inside the JSONL's own folder. Like the held-out slots, no two slots send
    the same pixels, so a server that caches by image or prefix misses as often in the first round as in grading."""
    rows = []
    try:
        for index in range(count):
            pair, repeat = pairs[index % len(pairs)], index // len(pairs)
            rows.append({**pair, "id": f"{pair['id']}~{index}",
                         **{key: _screenshot(public_dir, directory, pair[key], repeat)
                            for key in ("image_before", "image_after")}})
    except (OSError, KeyError, TypeError, ValueError) as error:
        raise EvaluationError(f"The public pairs are incomplete: {type(error).__name__}: {error}") from error
    path = directory / "pairs.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    return path


def _screenshot(public_dir, directory, reference, repeat):
    """The public screenshot for a pair's first slot; for its `repeat`-th repeat, a PNG copy with the lowest red bit
    of one pixel flipped, which changes the bytes and nothing a description could mention. Returns its reference."""
    name = reference
    if repeat:
        source = PurePosixPath(reference)
        name = str(source.with_name(f"{source.stem}~{repeat}.png"))
    target = directory / name
    if target.exists():
        return name
    target.parent.mkdir(parents=True, exist_ok=True)
    if not repeat:
        shutil.copyfile(public_dir / reference, target)
        return name
    from PIL import Image  # the verifier image's Pillow; validate.py stays stdlib-only
    with Image.open(public_dir / reference) as image:
        image = image.convert("RGBA" if image.mode.endswith("A") or "transparency" in image.info else "RGB")
    row, column = divmod(repeat, image.width)
    position = (column, row % image.height)
    pixel = list(image.getpixel(position))
    pixel[0] ^= 1
    image.putpixel(position, tuple(pixel))
    image.save(target, format="PNG")
    return name


def _batch(check, server, data, count, distinct, config):
    """public.run as grading calls it, rounds and concurrency included, in its diagnostic mode: no judge, and the
    deadline widened to BATCH_WINDOW times grading's so the rate is measured rather than cut off. Results are
    recorded without ending the check, so the single request after it still reports the parser's verdict."""
    total, required = count * public.ROUNDS, public.MINIMUM_PAIRS_PER_SECOND
    timeout = config.get("inference_timeout_seconds", 3600)
    deadline = min(timeout, total / required)
    window = min(timeout, BATCH_WINDOW * deadline)
    options = {key: value for key, value in config.items() if key != "judge"}
    options.update(validation_mode=True, expected_samples=count, inference_timeout_seconds=window)
    shape = (f"{total} requests ({count} pairs cycled from {distinct} public, {public.ROUNDS} rounds) "
             f"at concurrency {options['concurrency']}")
    started = time.monotonic()
    try:
        result = public.run(server["transport"], data, options)
    except EvaluationError:
        raise
    except Exception as error:  # noqa: BLE001 - grading counts anything but its own infrastructure against the server
        elapsed = time.monotonic() - started
        # Out of time, whether the window or the requests' own timeouts (past the failure limit) said so first. No rate
        # is recorded: from here a uniformly slow server and a few requests that never answer look the same.
        if elapsed >= window or str(error).startswith("Batch exceeded"):
            check.record("throughput", False, required=required, unit="pairs/s",
                         detail=f"{shape} had not finished after {elapsed:.1f} s (grading allows {deadline:g} s): "
                                f"either the server is far slower than {required} pairs/s or some requests never "
                                "answered (lost, or generating to the max_tokens limit); grading scores either zero")
        else:
            check.record("batch responses", False, kind="smoke", detail=f"{type(error).__name__}: {error}")
        return
    timing, failed = result["details"]["timing"][0], result["details"]["failed_requests"]
    # Grading's own test, the elapsed time against the deadline; the rate is reported for the margin.
    check.record("throughput", timing["elapsed_seconds"] <= deadline, measured=timing["items_per_second"],
                 required=required, unit="pairs/s", higher_is_better=True,
                 detail=f"{shape} took {timing['elapsed_seconds']:.2f} s; grading allows {deadline:g} s")
    empty = sum(1 for prediction in result["predictions"] if not prediction.get("request_failed")
                and not (isinstance(prediction.get("description"), str) and prediction["description"].strip()))
    problems = []
    if failed["total"]:
        problems.append(f"{failed['total']} of {total} requests failed, timed out or were malformed "
                        f"({', '.join(f'{reason}: {n}' for reason, n in failed['reasons'].items())}); grading scores "
                        f"each such pair-round zero, and the whole run zero past {failed['limit_percent']}%. "
                        f"First errors: {failed['errors']}")
    if empty:
        problems.append(f"{empty} of {total} descriptions were empty, which score zero")
    # Within the failure limit grading zeroes only those pair-rounds; it scores the run zero when none is left to judge.
    judged = total - failed["total"] - empty
    check.record("batch responses", judged > 0, kind="smoke", measured=judged, required=total, unit="descriptions",
                 detail="; ".join(problems) or f"all {total} responses parsed with a non-empty description")


def _described(sample, response):
    """The grader's parser, plus the scorer's rule that an empty description scores zero."""
    prediction = public.parse_response(sample, response)
    if not prediction["description"].strip():
        raise VerificationFailure("the description is empty (no text content), which scores zero")
    return prediction
