#!/usr/bin/env python3
"""Score your encode/decode server(s) on the public validation images with the task metric.

    python /task/validate.py --server-url http://127.0.0.1:8000
    python /task/validate.py --server-url http://127.0.0.1:8000 --decode-server-url http://127.0.0.1:8001

Phase A posts every image to POST /encode as {"image": "<base64 PNG>"} and requires exactly
{"codes": [<int>, ...], "codebook_size": <int>}: every code is an integer in [0, codebook_size),
codebook_size is the same in every response, and len(codes) * ceil(log2(codebook_size)) is at
most 16384 bits (2 KB). codebook_size describes the serialized integer alphabet, which may
represent scalar levels, learned codebook indices or codes from multiple codebooks.
Phase B posts each accepted code list to POST /decode as
{"codes": [...], "codebook_size": K} in shuffled order with no image identifier, plus a hidden
~5% sample of repeated code lists, and requires exactly {"text": "..."}. --decode-server-url
sends phase B to a second instance of your server (recommended: start it fresh, as the final
evaluation does).

Per-image score: 1 - word-level normalized edit distance to the reference page text after the
pinned ocr-text-v1 normalization (Levenshtein over whitespace tokens divided by the longer token
count). Malformed responses and code lists over the bit budget score 0 for that image; a text
over 32768 characters scores 0; a repeated code list whose normalized text differs from its
first decode keeps the lower of its two scores. The report gives the mean, per image kind and
density bucket, plus character-level 1 - NED and word F1 as diagnostics.

Throughput: the final evaluation requires at least 1 image/s in each phase, measured over the
whole batch at up to 16 concurrent requests (the phase fails if it takes longer than
n / 1 seconds). This script reports each phase's observed images/s with required_items_per_second
and throughput_target_met, but does not enforce it (throughput_gate_enforced: false); a small
validation batch does not certify the full held-out batch. The final evaluation also runs static
checks and an architecture review, which this script does not.
"""
import base64
import math
from pathlib import Path
import random

from frontier_ml.validation.cli import public_main
from frontier_ml.validation.errors import VerificationFailure
from frontier_ml.validation.evaluation import FailedRequests, RequestFailure, attempt_request, load_samples, timed_batch
from frontier_ml.validation.http import HTTPTransport
from ocr_text import VERSION as NORMALIZATION, normalize, ned, word_f1, word_ned

TASK = "ocr-latent-compression"
DEFAULT_DATA = "/task/public/validation.jsonl"
BIT_BUDGET = 16384                 # 2 KB of codes per image
MAX_CODEBOOK = 2 ** 24
MAX_TEXT_CHARS = 32768
DETERMINISM_FRACTION = 0.05
MINIMUM_IMAGES_PER_SECOND = 1.0    # per phase; reported here, enforced only by the final evaluation
BUCKETS = ("sparse", "medium", "dense", "very_dense")
KINDS = ("document", "screenshot")


def image_request(sample, data_root):
    root = Path(data_root).resolve()
    path = (root / sample["image"]).resolve()
    if not path.is_relative_to(root) or path.suffix.lower() != ".png":
        raise VerificationFailure("Image reference must be a PNG inside the dataset directory")
    return {"image": base64.b64encode(path.read_bytes()).decode("ascii")}


def bits_per_code(codebook_size):
    return max(1, math.ceil(math.log2(codebook_size)))


def code_bits(codes, codebook_size):
    return len(codes) * bits_per_code(codebook_size)


def parse_encode(response, max_codebook=MAX_CODEBOOK):
    """Return (codes, codebook_size); require integer codes in the declared serialization range."""
    if not isinstance(response, dict) or set(response) != {"codes", "codebook_size"}:
        raise VerificationFailure('/encode must return exactly {"codes": [...], "codebook_size": K}')
    size, codes = response["codebook_size"], response["codes"]
    if type(size) is not int or not 2 <= size <= max_codebook:
        raise VerificationFailure(f"codebook_size must be an integer between 2 and {max_codebook}")
    if not isinstance(codes, list) or len(codes) > BIT_BUDGET * 4:
        raise VerificationFailure("codes must be a list of integers")
    for code in codes:
        if type(code) is not int or not 0 <= code < size:
            raise VerificationFailure("every code must be an integer in [0, codebook_size)")
    return list(codes), size


def parse_decode(response):
    if not isinstance(response, dict) or set(response) != {"text"} or not isinstance(response["text"], str):
        raise VerificationFailure('/decode must return exactly {"text": "..."}')
    return response["text"]


def encode_phase(transport, samples, data_root, config, *, minimum_per_second=None):
    """Phase A: one /encode request per image. Malformed responses are represented by None.

    ``minimum_per_second`` turns on the whole-batch deadline; only the trusted verifier passes it.
    """
    failures = FailedRequests(len(samples))
    max_codebook = int(config.get("max_codebook_size", MAX_CODEBOOK))

    def call(sample, timeout):
        # A request that fails at the transport level is retried, then scores 0 for its image like a malformed
        # response (CONTRACT.md); more than 5% of the phase's requests failing that way fails the phase.
        response = attempt_request(transport, "POST", "/encode", image_request(sample, data_root), timeout=timeout,
                                   failures=failures)
        if isinstance(response, RequestFailure):
            failures.record(sample["id"], response)
            return None
        try:
            return parse_encode(response, max_codebook)
        except VerificationFailure:
            return None

    encodings, timing = timed_batch(
        samples, call, concurrency=int(config.get("concurrency", 16)),
        minimum_per_second=minimum_per_second,
        timeout_seconds=float(config.get("inference_timeout_seconds", 3600)))
    timing["failed_requests"] = failures.summary()
    sizes = {encoding[1] for encoding in encodings if encoding is not None}
    if len(sizes) > 1:
        raise VerificationFailure(f"codebook_size must be identical in every /encode response; saw {sorted(sizes)}")
    return encodings, timing


def accepted_indices(encodings, config):
    budget = int(config.get("bit_budget", BIT_BUDGET))
    return [i for i, encoding in enumerate(encodings) if encoding is not None and code_bits(*encoding) <= budget]


def decode_plan(encodings, config, rng):
    """Shuffled decode queue of (image index, repeated?) with a hidden sample of repeats."""
    accepted = accepted_indices(encodings, config)
    fraction = float(config.get("determinism_fraction", DETERMINISM_FRACTION))
    repeats = rng.sample(accepted, min(len(accepted), max(1, math.ceil(fraction * len(accepted))))) if accepted else []
    queue = [(i, False) for i in accepted] + [(i, True) for i in repeats]
    rng.shuffle(queue)
    return queue


def decode_phase(transport, encodings, config, rng, *, minimum_per_second=None):
    """Phase B: code lists only, in shuffled order without IDs. Results are texts or None."""
    queue = decode_plan(encodings, config, rng)
    if not queue:
        return {}, {}, {"elapsed_seconds": 0.0, "items": 0, "skipped": "no valid code list within the bit budget"}
    failures = FailedRequests(len(queue))

    def call(entry, timeout):
        index, _ = entry
        codes, size = encodings[index]
        response = attempt_request(transport, "POST", "/decode", {"codes": codes, "codebook_size": size},
                                   timeout=timeout, failures=failures)
        if isinstance(response, RequestFailure):
            failures.record(f"image {index}", response)
            return None
        try:
            return parse_decode(response)
        except VerificationFailure:
            return None

    texts, timing = timed_batch(
        queue, call, concurrency=int(config.get("concurrency", 16)),
        minimum_per_second=minimum_per_second,
        timeout_seconds=float(config.get("inference_timeout_seconds", 3600)))
    timing["failed_requests"] = failures.summary()
    first, repeat = {}, {}
    for (index, repeated), text in zip(queue, texts, strict=True):
        (repeat if repeated else first)[index] = text
    return first, repeat, timing


def score(samples, encodings, first, repeat, config=None):
    config = config or {}
    budget = int(config.get("bit_budget", BIT_BUDGET))
    max_chars = int(config.get("max_text_chars", MAX_TEXT_CHARS))
    if not samples or len(samples) != len(encodings):
        raise VerificationFailure("Expected exactly one encoding per image")
    records = []
    for index, (sample, encoding) in enumerate(zip(samples, encodings, strict=True)):
        record = {"id": sample["id"], "kind": sample.get("kind"), "bucket": sample.get("bucket"),
                  "codes": None, "bits": None, "value": 0.0}
        if encoding is None:
            record["failure"] = "invalid_encode_response"
        else:
            codes, size = encoding
            record.update(codes=len(codes), codebook_size=size, bits=code_bits(codes, size))
            if record["bits"] > budget:
                record["failure"] = "codes_over_budget"
            elif index not in first:
                raise VerificationFailure("Missing decode result for an accepted code list")
            elif first[index] is None or (index in repeat and repeat[index] is None):
                record["failure"] = "invalid_decode_response"
            else:
                text = first[index]
                if len(text) > max_chars or (index in repeat and len(repeat[index]) > max_chars):
                    record["failure"] = "text_over_limit"
                else:
                    reference = normalize(sample["text"])
                    prediction = normalize(text)
                    record["word_f1"] = word_f1(text, sample["text"])
                    record["value"] = 1.0 - word_ned(prediction, reference)
                    record["char_similarity"] = 1.0 - ned(prediction, reference)
                    if index in repeat and normalize(repeat[index]) != prediction:
                        # Nondeterminism can never help: a repeated code list keeps its worse decode.
                        other = normalize(repeat[index])
                        record["value"] = min(record["value"], 1.0 - word_ned(other, reference))
                        record["char_similarity"] = min(record["char_similarity"], 1.0 - ned(other, reference))
                        record["nondeterministic"] = True
                if config.get("validation_mode"):
                    record["prediction"] = text
        records.append(record)

    def mean(rows):
        return sum(r["value"] for r in rows) / len(rows) if rows else None

    by_kind = {k: mean([r for r in records if r["kind"] == k]) for k in KINDS}
    by_bucket = {b: mean([r for r in records if r["bucket"] == b]) for b in BUCKETS}
    by_cell = {f"{k}/{b}": mean([r for r in records if r["kind"] == k and r["bucket"] == b]) for k in KINDS for b in BUCKETS}
    bits = [r["bits"] for r in records if r["bits"] is not None]
    f1 = [r["word_f1"] for r in records if "word_f1" in r]
    chars = [r.get("char_similarity", 0.0) for r in records]
    details = {
        "metric": "page_word_similarity", "normalization": NORMALIZATION, "samples": len(records),
        "bit_budget": budget,
        "codebook_size": next((r["codebook_size"] for r in records if "codebook_size" in r), None),
        "by_kind": by_kind, "by_bucket": by_bucket, "by_kind_bucket": by_cell,
        "bits": {"mean": sum(bits) / len(bits) if bits else None, "max": max(bits, default=None),
                 "over_budget": sum(b > budget for b in bits)},
        "determinism": {"rechecked": len(repeat), "mismatches": sum(bool(r.get("nondeterministic")) for r in records)},
        "diagnostic_word_f1": sum(f1) / len(f1) if f1 else None,
        "diagnostic_char_similarity": sum(chars) / len(chars),
        "failures": {name: sum(r.get("failure") == name for r in records)
                     for name in ("invalid_encode_response", "invalid_decode_response", "codes_over_budget", "text_over_limit")},
        "records": records,
    }
    return {"score": mean(records), "details": details}


def throughput(timing, required=MINIMUM_IMAGES_PER_SECOND, *, enforced=False):
    """Annotate a phase's timing with the 1 image/s requirement (as run_http_evaluation does)."""
    timing = dict(timing)
    rate = timing.get("items_per_second")
    timing["required_items_per_second"] = required
    timing["throughput_target_met"] = None if rate is None else rate >= required
    timing["throughput_gate_enforced"] = enforced
    return timing


def run(transport, data_path, config):
    """Both phases against your server(s), then the metric."""
    config = dict(config or {})
    samples = load_samples(data_path)
    encodings, encode_timing = encode_phase(transport, samples, Path(data_path).parent, config)
    decoder = config.get("decode_transport") or transport
    rng = random.Random(config["shuffle_seed"]) if "shuffle_seed" in config else random.SystemRandom()
    first, repeat, decode_timing = decode_phase(decoder, encodings, config, rng)
    result = score(samples, encodings, first, repeat, config)
    # Diagnostic only: observed throughput against the final requirement, never enforced here.
    result["details"]["timing"] = {"encode": throughput(encode_timing), "decode": throughput(decode_timing)}
    return result


def _arguments(parser):
    parser.add_argument("--decode-server-url", help="run phase B against this (fresh) instance; default: --server-url")


def _config(args):
    return {"decode_transport": HTTPTransport(args.decode_server_url)} if args.decode_server_url else {}


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__,
                       extra_arguments=_arguments, extra_config=_config)


if __name__ == "__main__":
    raise SystemExit(main())
