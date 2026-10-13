"""Trusted, root-owned loopback timing client. Run with python -I; no candidate imports."""
import http.client
import json
import os
from pathlib import Path
import signal
import sys
import time

# The grader installs this file and its two pure-Python dependencies in a private,
# root-owned directory. -I excludes the candidate's cwd and PYTHONPATH.
if __name__ == "__main__":
    sys.path.insert(0, str(Path(__file__).resolve().parent))
from zsc_protocol import (MINIMUM_DOCS_PER_SECOND, WARMUP_REQUESTS, TransportFailure,
                          evaluate_batch, timed_request)


class LoopbackTransport:
    def __init__(self, port=8000):
        self.port = port

    def raw_request(self, method, path, *, body, headers, timeout):
        connection = http.client.HTTPConnection("127.0.0.1", self.port, timeout=timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            return response.status, response.read()
        finally:
            connection.close()


def alarm(signum, frame):
    raise TimeoutError("Trusted client deadline exceeded")


def evaluate_requests(requests, transport, *, total_seconds):
    if not requests:
        raise ValueError("No requests")
    started = time.monotonic()
    # Reserve the full scored-batch budget. Warm-up shares the remaining overall
    # evaluation time; first-use compilation has no separate per-request gate.
    scored_budget = len(requests) / MINIMUM_DOCS_PER_SECOND
    warmup_deadline = started + total_seconds - scored_budget
    warmup_failures = 0
    for payload in WARMUP_REQUESTS:
        remaining = warmup_deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Warm-up deadline exceeded")
        signal.setitimer(signal.ITIMER_REAL, remaining)
        try:
            timed_request(transport, payload, timeout=remaining)
        except TransportFailure as exc:
            if exc.reason == "timeout":
                raise TimeoutError("Warm-up deadline exceeded") from exc
            warmup_failures += 1  # unscored synthetic request: noted, never fatal; the scored batch decides
        except ValueError:
            warmup_failures += 1
        finally:
            signal.setitimer(signal.ITIMER_REAL, 0)
    budget = min(scored_budget, total_seconds - (time.monotonic() - started))
    if budget <= 0:
        raise TimeoutError("Whole-run deadline exceeded")
    # The main-process alarm also bounds malicious slow-drip responses in workers.
    # evaluate_batch owns the exact gate. A small watchdog allowance avoids
    # subtracting setup overhead from its measured budget while bounding drip attacks.
    # Retries, backoff and failed documents all run inside that budget (zsc_protocol.attempt_request); a record
    # whose request failed carries request_failed and no response, and the verifier scores it as wrong.
    signal.setitimer(signal.ITIMER_REAL, budget + 0.25)
    try:
        records, timing = evaluate_batch(transport, requests, timeout_seconds=budget)
        return {"status": "ok", "records": records, "timing": timing, "warmup_failures": warmup_failures}
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)


def main(argv=None):
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("requests")
    parser.add_argument("output")
    parser.add_argument("--port", type=int, default=8000)
    parser.add_argument("--total-seconds", type=float, default=600)
    args = parser.parse_args(argv)
    signal.signal(signal.SIGALRM, alarm)
    try:
        requests = [json.loads(line) for line in Path(args.requests).read_text().splitlines() if line.strip()]
        if any(not isinstance(row, dict) or set(row) != {"text", "labels"} for row in requests):
            raise ValueError("Runner accepts input-only requests")
        result = evaluate_requests(requests, LoopbackTransport(args.port), total_seconds=args.total_seconds)
    except Exception as exc:
        result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}"}
    Path(args.output).write_text(json.dumps(result, allow_nan=False) + "\n")
    return 0 if result["status"] == "ok" else 1


if __name__ == "__main__":
    # The report file is closed by main(). Avoid Python's implicit thread-pool
    # joins on failure: a candidate must not prolong grading by trickling bytes.
    os._exit(main())
