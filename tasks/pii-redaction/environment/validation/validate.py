#!/usr/bin/env python3
"""Evaluate your redaction server on the public validation documents.

    python /task/validate.py --server-url http://127.0.0.1:8000

Sends POST /redact {"id", "text"} for every document, exactly as the trusted final
client does, and scores the returned spans with the mean document R-score. It reports
the observed throughput against the five-documents-per-second requirement but does not
enforce it; the trusted grader enforces the requirements in /task/CONTRACT.md.
"""
from frontier_ml.validation.evaluation import run_http_evaluation
from frontier_ml.validation.errors import VerificationFailure
from spans import score_rows
from frontier_ml.validation.cli import public_main

TASK = "pii-redaction"
DEFAULT_DATA = "/task/public/validation.jsonl"
MINIMUM_DOCUMENTS_PER_SECOND = 5


def build_request(sample, data_root):
    return "/redact", {"id": sample["id"], "text": sample["text"]}


def check_spans(text, spans, error=ValueError):
    """The contract's span interface: a list of {"start", "end"} objects with integer, zero-based, end-exclusive
    Unicode code point offsets, sorted, nonempty, disjoint and within the text. Raises `error` on the first breach."""
    if not isinstance(spans, list):
        raise error("spans must be a list")
    previous_end = 0
    for span in spans:
        if not isinstance(span, dict) or set(span) != {"start", "end"}:
            raise error("Each span must contain only start and end")
        start, end = span["start"], span["end"]
        if type(start) is not int or type(end) is not int or not 0 <= previous_end <= start < end <= len(text):
            raise error("Spans must be sorted, disjoint, nonempty Unicode code point offsets within the text")
        previous_end = end


def parse_response(sample, response):
    """One document's prediction, or VerificationFailure. An undocumented shape or an invalid span (see check_spans;
    for example an end past len(text) from byte or UTF-16 offsets) is a malformed response and costs only this
    document, under the shared per-item policy in CONTRACT.md."""
    if not isinstance(response, dict) or response.get("id") != sample["id"]:
        raise VerificationFailure("Response must echo id and provide spans")
    check_spans(sample["text"], response.get("spans"), VerificationFailure)
    return response


def score(samples, predictions, config=None):
    """Strict span interface: sorted, disjoint Unicode code point offsets, one prediction per document."""
    if not samples or len(samples) != len(predictions):
        raise ValueError("Expected exactly one prediction per document")
    for row, prediction in zip(samples, predictions, strict=True):
        if not isinstance(prediction, dict) or prediction.get("id") != row["id"]:
            raise ValueError("Prediction ID/order mismatch")
        check_spans(row["text"], prediction.get("spans"))
    details = score_rows(samples, predictions)
    for key in ("protocol", "source"):
        details.pop(key, None)
    details["metric"] = "mean_document_r_score"
    return {"score": details["document_mean"], "details": details}


def run(transport, data_path, config):
    """One timed batch over every document; the final gate is five documents per second.

    A document whose request failed after the shared retries, or whose response was malformed, including any
    invalid span (parse_response; see CONTRACT.md), is left out of score() and the mean is scaled by scored/total
    documents (per_item_mean): exactly a zero R-score for that document under the equal-weight document mean. No
    failed_prediction hook is declared because no span set scores zero on every document (no spans scores 1 on a
    document without mandatory PII; redacting everything still earns mandatory coverage). details.failed_items and
    details.failed_requests name the documents.
    """
    return run_http_evaluation(transport, data_path, config, build_request=build_request,
                               parse_response=parse_response, score=score, per_item_mean=True,
                               minimum_per_second=MINIMUM_DOCUMENTS_PER_SECOND)


def main(argv=None):
    return public_main(TASK, run, argv, default_data=DEFAULT_DATA, description=__doc__)


if __name__ == "__main__":
    raise SystemExit(main())
