"""Trusted verifier: provenance metadata and method review, then two isolated server phases.

Phase A starts a fresh instance of the frozen submission, encodes every held-out image into
codes, and is torn down. Phase B starts another fresh instance (a new sandbox, no network) and
decodes only the code lists, in a private random order with no image identifiers, plus hidden
repeats for the determinism check. Both instances extract the submission from the same
per-trial submission volume, which stays mounted in both sandboxes; nothing here checks which
sandbox a phase ran in. The split therefore relies on each `launch()` provisioning a fresh
sandbox and on the review policy, which forbids any per-image state outside the code list.
Scoring uses the public metric in validate.py. `selfcheck` is the agent's `frontier selfcheck`
on public images (frontier_ml/selfcheck.py).
"""
from pathlib import Path
import random
import threading
import time

from frontier_ml.runtime.server import VerificationFailure
from frontier_ml.scoring.architecture import verify_review
from ocr_latent import static_checks
from frontier_ml.taskspec import sibling
from frontier_ml.validation.errors import EvaluationError
from frontier_ml.validation.evaluation import load_samples
from frontier_ml.validation.progress import progress

SLUG = "ocr-latent-compression"
public = sibling(__file__, "validate.py", require=("encode_phase", "decode_phase", "score"))
REVIEW_CHECKS = list(sibling(__file__, "review.py").POLICY["checks"])
FIXED = {"bit_budget": 16384, "determinism_fraction": 0.05, "max_text_chars": 32768, "max_codebook_size": 2 ** 24}
MINIMUM_IMAGES_PER_SECOND = public.MINIMUM_IMAGES_PER_SECOND  # per phase; enforced here as deadline n / rate


def default_config():
    return {"minimum_images_per_second": MINIMUM_IMAGES_PER_SECOND, **FIXED}


def _checked(config):
    for key, value in FIXED.items():
        if config.get(key) != value:
            raise VerificationFailure(f"{key} is fixed at {value} for this task")
    rate = config.get("minimum_images_per_second")
    if type(rate) not in (int, float) or not rate > 0:
        raise VerificationFailure("minimum_images_per_second must be configured before provisioning")
    return float(rate)


def preflight(config, data_path):
    _checked(config)
    review = config.get("architecture_review_path")
    if not review or not Path(review).is_file():
        raise VerificationFailure("Trusted architecture_review_path is required before provisioning")


def verify(server, data_path, config):
    """Single-instance entry point. This task needs the isolated two-phase launcher."""
    raise VerificationFailure("ocr-latent-compression requires isolated two-phase grading (verify_isolated)")


def verify_isolated(launch, data_path, config):
    """`launch()` is a context manager yielding a fresh server instance of the frozen snapshot."""
    rate = _checked(config)
    config = {**config, "validation_mode": False}  # predictions are never echoed into trusted results
    snapshot = Path(config["submission_path"])
    checks = static_checks(snapshot)
    review = verify_review(snapshot, config["architecture_review_path"], SLUG, REVIEW_CHECKS)
    samples = load_samples(data_path)
    expected = config.get("expected_samples")
    if expected is not None and len(samples) != int(expected):
        raise VerificationFailure(f"Expected {expected} evaluation samples; found {len(samples)}")
    instances = []
    with launch() as server:
        encodings, encode_timing = public.encode_phase(server["transport"], samples, Path(data_path).parent, config,
                                                       minimum_per_second=rate)
        instances.append({"phase": "encode", "startup_seconds": server["startup_elapsed"],
                          "backend": server["backend"], "sandbox_id": server.get("sandbox_id")})
    observed = {encoding[1] for encoding in encodings if encoding is not None}
    if observed and observed != {checks["codebook_size"]}:
        raise VerificationFailure(f"codebook_size in /encode responses {sorted(observed)} differs from "
                                  f"architecture.json ({checks['codebook_size']})")
    # The encoder instance is gone; only the code lists cross to phase B.
    with launch() as server:
        first, repeat, decode_timing = public.decode_phase(server["transport"], encodings, config, random.SystemRandom(),
                                                           minimum_per_second=rate)
        instances.append({"phase": "decode", "startup_seconds": server["startup_elapsed"],
                          "backend": server["backend"], "sandbox_id": server.get("sandbox_id")})
    result = public.score(samples, encodings, first, repeat, config)
    result["details"]["timing"] = {"encode": public.throughput(encode_timing, rate, enforced=True),
                                   "decode": public.throughput(decode_timing, rate, enforced=True)}
    result["details"]["instances"] = instances
    result["details"]["gates"] = {"static": checks,
                                  "architecture": {"reviewer": review.get("reviewer"), "checks": review.get("checks")}}
    return result


# --------------------------------------------------------------------------- self-check

SELFCHECK_PUBLIC = ("validation.jsonl", "images/*")  # the 100 public images (about 23 MB) and their index
SMOKE_TIMEOUT_SECONDS = 120  # per smoke request, retries included; the two batches are bounded by their own deadlines


def selfcheck(check, public_dir, config):
    """verify_isolated on public images: grading's static checks of architecture.json; a fresh instance timed from
    readiness on a grading-sized encode batch, then one public image through the encode parser and the bit budget; a
    second fresh instance timed on decoding only the accepted code lists, then that image's code list through the
    decode parser. Both batches run at full size: 400 and 420 requests at 1 image/s are 400 s and 420 s deadlines.
    Neither the review nor the held-out split is involved."""
    try:
        rate = _checked(config)
    except VerificationFailure as error:  # the trusted config, not the submission
        raise EvaluationError(f"Grading config: {error}") from error
    declared = check.static("architecture.json", static_checks, check.snapshot)["codebook_size"]
    samples = _public_samples(public_dir)
    rng = random.SystemRandom()
    batch = _resampled(samples, int(config.get("expected_samples") or len(samples)), rng)
    budget, max_chars = int(config["bit_budget"]), int(config["max_text_chars"])
    if len(batch) > len(samples):
        check.note(f"The encode batch sends each of the {len(samples)} public images about "
                   f"{len(batch) / len(samples):.3g} times and the decode batch their code lists, so a server that "
                   "caches repeated images or code lists measures faster here than on grading's distinct held-out "
                   "images.")
    sample = samples[0]

    def parse_encode(sample, response):
        codes, size = public.parse_encode(response, int(config["max_codebook_size"]))
        if size != declared:
            raise VerificationFailure(f"codebook_size {size} differs from architecture.json ({declared}); grading "
                                      "fails the run on that")
        return {"codebook_size": size, "codes": codes}  # size first, so the report's clipped excerpt shows it

    def parse_decode(sample, response):
        text = public.parse_decode(response)
        if len(text) > max_chars:
            raise VerificationFailure(f"the text has {len(text)} characters; over {max_chars} scores 0")
        return text

    # Phase A as verify_isolated runs it: the batch starts at readiness, so lazy loading and first-call compiles
    # count against its deadline. The smoke request follows it and costs the deadline nothing.
    with check.launch() as server:
        check.stage("gates")
        encodings, timing = _phase(check, server, config, "encode", len(batch), rate, lambda transport:
                                   public.encode_phase(transport, batch, public_dir, config, minimum_per_second=rate))
        accepted = public.accepted_indices(encodings, config)
        missing = len(batch) - len(accepted)
        check.gate("encode throughput", timing["items_per_second"], rate, higher_is_better=True, unit="images/s",
                   detail=_detail(timing, missing and f"{missing} images got no valid code list within the bit "
                                  "budget and would score 0"))
        check.stage("smoke")  # why grading rejects a response, which the batch only counts
        encoded = check.request(server, "/encode", public.image_request(sample, public_dir),
                                parse=parse_encode, sample=sample, timeout=SMOKE_TIMEOUT_SECONDS)
        codes, size = encoded["codes"], encoded["codebook_size"]
        bits = public.code_bits(codes, size)
        detail = f"{len(codes)} codes x {public.bits_per_code(size)} bits for one public image; every image must fit"
        if bits > budget:
            check.fail("bit budget", detail + " (an over-budget code list scores 0 and is never decoded)",
                       kind="smoke", measured=bits, required=budget, unit="bits")
        check.record("bit budget", True, kind="smoke", measured=bits, required=budget, unit="bits", detail=detail)
    observed = {encoding[1] for encoding in encodings if encoding is not None}
    if observed and observed != {declared}:  # verify_isolated's cross-check, between the two instances
        check.fail("codebook_size", f"/encode responses declared {sorted(observed)}, architecture.json "
                   f"{declared}; grading fails the run on that")
    if not accepted:
        check.fail("decode throughput", "no /encode response in the batch was a valid code list within the bit "
                   "budget, so nothing was left to decode")
    # Phase B: a fresh instance that only ever sees code lists, so a decoder relying on encode-side state fails here.
    queue = len(public.decode_plan(encodings, config, random.Random(0)))  # the draw does not change its length
    with check.launch() as server:
        check.stage("gates")
        first, repeat, timing = _phase(check, server, config, "decode", queue, rate, lambda transport:
                                       public.decode_phase(transport, encodings, config, rng, minimum_per_second=rate))
        check.gate("decode throughput", timing["items_per_second"], rate, higher_is_better=True, unit="images/s",
                   detail=_detail(timing, *_decode_zeros(encodings, first, repeat, max_chars)))
        check.stage("smoke")
        # The evaluator re-serializes the parsed codes, exactly as decode_phase sends them.
        check.request(server, "/decode", {"codes": codes, "codebook_size": size}, parse=parse_decode,
                      sample=sample, timeout=SMOKE_TIMEOUT_SECONDS)


def _public_samples(directory):
    """The public rows, each image read as the encode phase will read it; a broken fetch is not the submission's."""
    try:
        samples = load_samples(Path(directory) / "validation.jsonl")
        for sample in samples:
            public.image_request(sample, directory)
    except (OSError, ValueError, TypeError, KeyError, VerificationFailure) as error:
        raise EvaluationError(f"The public validation images are unusable: {type(error).__name__}: {error}") from error
    return samples


def _resampled(samples, count, rng):
    """A grading-sized batch from the smaller public split: every image about equally often, IDs kept unique."""
    picks = samples * (count // len(samples)) + rng.sample(samples, count % len(samples))
    rng.shuffle(picks)
    return [{**sample, "id": f"{sample['id']}#{index}"} for index, sample in enumerate(picks)]


class _Counted:
    """The instance's transport, counting answers: the agent sees progress through a long batch, and a missed
    deadline still reports the rate the server managed."""

    def __init__(self, transport, config, stage, total):
        self.transport, self.config, self.stage, self.total = transport, config, stage, total
        self.answered = 0
        self.lock = threading.Lock()

    def request(self, *args, **kwargs):
        response = self.transport.request(*args, **kwargs)
        with self.lock:
            self.answered += 1
            progress(self.config, self.stage, completed=self.answered, total=self.total)
        return response


def _phase(check, server, config, name, total, rate, run):
    """One of validate.py's phases under the verifier's deadline; a miss or a failing phase ends the self-check."""
    transport = _Counted(server["transport"], config, f"measuring {name} throughput", total)
    started = time.monotonic()
    try:
        return run(transport)
    except EvaluationError:
        raise
    except VerificationFailure as error:
        elapsed = time.monotonic() - started
        # Requests cut off at the deadline can also surface as more than 5% of them timing out.
        if not (isinstance(error.__cause__, TimeoutError) or elapsed >= total / rate):
            check.fail(f"{name} phase", str(error))  # over 5% of requests failed, or mixed codebook sizes
        check.fail(f"{name} throughput", f"{error} ({transport.answered} of {total} requests answered in "
                   f"{elapsed:.1f} s at concurrency {config.get('concurrency', 16)})",
                   measured=transport.answered / elapsed, required=rate, unit="images/s", higher_is_better=True)


def _decode_zeros(encodings, first, repeat, max_chars):
    """What validate.score zeros or lowers in the decode batch without failing the run: failed or malformed decodes,
    texts over the limit, and code lists whose normalized text changes between decodes. The latter also compares the
    identical code lists of a repeated public image, besides grading's hidden repeats."""
    texts = [*first.values(), *repeat.values()]
    zeros = [f"{count} {what}" for count, what in (
        (sum(text is None for text in texts), "decode requests failed or were malformed"),
        (sum(text is not None and len(text) > max_chars for text in texts), f"texts were over {max_chars} characters"),
    ) if count]
    decoded = {}
    for index, text in [*first.items(), *repeat.items()]:
        if text is not None:
            codes, size = encodings[index]
            decoded.setdefault((size, tuple(codes)), set()).add(public.normalize(text))
    unstable = sum(len(variants) > 1 for variants in decoded.values())
    return ["; ".join(zeros) + "; their images would score 0" if zeros else None,
            unstable and f"{unstable} code lists decoded to different texts when sent more than once (decoding must be "
            "deterministic; grading keeps a repeated code list's lower score)"]


def _detail(timing, *notes):
    text = (f"{timing['items']} requests at concurrency {timing['concurrency']}: {timing['elapsed_seconds']:.1f} s of "
            f"a {timing['deadline_seconds']:g} s deadline, timed from a fresh instance's readiness as in grading")
    return "; ".join([text, *filter(None, notes)])
