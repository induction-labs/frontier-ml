"""Trusted verifier: the public validate.py request template and metric over the held-out documents.

selfcheck: grading's timed batch on resampled public documents (no score), then one document through the parser."""
import json
from pathlib import Path
import tempfile
import time

from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError, VerificationFailure
from frontier_ml.validation.evaluation import load_samples

public = sibling(__file__, "validate.py", require=("run",))
SELFCHECK_PUBLIC = ("validation.jsonl",)
SELFCHECK_REQUEST_SECONDS = 60  # the one smoke request, retries included; grading's batch gets expected_samples / 5 s


def verify(server, data_path, config):
    return public.run(server["transport"], data_path, config)


def selfcheck(check, public_dir, config):
    """Grading has no prescreen or preflight here; the review's free prerequisites (architecture.json, serve.sh, the
    declared base) already ran in selfcheck.run, so the hook adds no static check.

    The timed batch goes first because grading sends its batch as soon as /health answers: a server that loads its
    model on the first request must miss the gate here as it would there, not after a warm-up.
    """
    try:
        documents = load_samples(Path(public_dir) / "validation.jsonl")
    except (OSError, ValueError, VerificationFailure) as error:
        raise EvaluationError(f"The public documents are unavailable: {error}") from error
    with check.launch() as server:
        check.stage("gates")
        throughput(check, server, documents, config)
        check.stage("smoke")
        sample = documents[0]
        path, payload = public.build_request(sample, public_dir)
        check.request(server, path, payload, parse=public.parse_response, sample=sample,
                      timeout=SELFCHECK_REQUEST_SECONDS)


def throughput(check, server, documents, config):
    """validate.run, the grading path, over the held-out count of public documents resampled with unique ids, at the
    task concurrency. Diagnostic mode skips the score; the batch keeps grading's own deadline because the shared
    retry policy only sends a retry whose backoff fits the remaining budget, so any longer budget would retry where
    grading gives up. A miss therefore has no measured rate. The whole batch takes at most 32.2 s (161 documents at
    5 docs/s), so it runs at full size rather than as a projected sample."""
    required = public.MINIMUM_DOCUMENTS_PER_SECOND
    count, distinct = config.get("expected_samples") or len(documents), len(documents)
    deadline = min(config.get("inference_timeout_seconds", 3600), count / required)
    rows = [dict(documents[i % distinct], id=f"{documents[i % distinct]['id']}.{i // distinct}") for i in range(count)]
    characters = sum(len(row["text"]) for row in rows)
    if count > distinct:
        check.note(f"The throughput batch sent each of the {distinct} public documents up to {-(-count // distinct)} "
                   f"times to reach grading's {count}; a server that caches repeated text can be faster here than on "
                   f"{count} distinct held-out documents.")
    options = {**config, "validation_mode": True, "predictions_only": True, "expected_samples": count,
               "inference_timeout_seconds": deadline}
    # Public documents are not the held-out ones and some repeat here: the character count shows how much text was sent.
    batch = (f"{count} public documents ({distinct} distinct, {characters:,} characters) at concurrency "
             f"{options.get('concurrency', 32)}")
    with tempfile.TemporaryDirectory(prefix="pii-redaction-selfcheck-") as directory:
        path = Path(directory) / "batch.jsonl"
        path.write_text("".join(json.dumps(row) + "\n" for row in rows))
        started = time.monotonic()
        try:
            result = public.run(server["transport"], path, options)
        except EvaluationError:
            raise
        except VerificationFailure as error:
            # Requests still open at the deadline time out there, which can trip the failed-request limit before the
            # batch's own deadline error: either way the batch could not finish in time.
            expired = str(error).startswith("Batch exceeded")
            if expired or time.monotonic() - started >= deadline:
                check.record("throughput", False, required=required, unit="docs/s",
                             detail=f"{batch} did not finish within grading's {deadline:g} s deadline for its {count} "
                                    f"held-out documents: under {required} docs/s" + ("" if expired else f" ({error})"))
            else:  # more than the tolerated share of requests failed: grading fails the run the same way
                check.record("batch responses", False, kind="smoke", detail=str(error))
            return
        except RecursionError as error:  # a response nested too deeply to decode on the relay's batch path
            check.record("batch responses", False, kind="smoke", detail=f"a response could not be decoded: {error}")
            return
    timing, failed = result["details"]["timing"][0], result["details"]["failed_requests"]
    detail = (f"{failed['total']} of {count} requests failed, timed out or were malformed {failed['reasons']}; grading "
              f"scores each such document zero and fails the run above {failed['limit_percent']}%: {failed['errors']}"
              if failed["total"] else f"all {count} responses parsed")
    check.record("batch responses", not failed["total"], kind="smoke", detail=detail)
    elapsed = timing["elapsed_seconds"]
    check.gate("throughput", timing["items_per_second"], required, higher_is_better=True, unit="docs/s",
               detail=f"{batch} in {elapsed:.1f} s ({characters / max(elapsed, 1e-9):,.0f} characters/s); grading "
                      f"allows {deadline:g} s for its {count} held-out documents")
