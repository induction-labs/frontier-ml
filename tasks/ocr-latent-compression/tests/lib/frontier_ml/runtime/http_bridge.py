"""Trusted root HTTP client copied into a candidate-only inference sandbox."""
import base64
from concurrent.futures import ThreadPoolExecutor, as_completed
import http.client
import json
from pathlib import Path
import re
import sys
import time


INLINE_RESPONSE_BYTES = 64 * 1024  # Bound stdout; larger image responses still use a file.


def request_one(request):
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=request["timeout"])
    try:
        connection.request(request["method"], request["path"], body=base64.b64decode(request["body"]), headers=request["headers"])
        response = connection.getresponse()
        limit = request.get("max_response_bytes", 16 * 1024**2)
        data = response.read(None if limit is None else limit + 1)
        if limit is not None and len(data) > limit:
            raise ValueError("Response exceeds size limit")
        result = {"status": response.status, "body": base64.b64encode(data).decode()}
    except Exception as exc:
        # The verifier classifies the failure from the class name; the message is a bounded diagnostic.
        result = {"error": type(exc).__name__, "detail": str(exc)[:200]}
    finally:
        connection.close()
    return result


def request_batch(request):
    """Time HTTP serving with the trusted root client's clock, excluding provider RPCs."""
    started = time.monotonic()
    deadline = started + request["timeout"]
    jobs = request["batch"]
    results = [None] * len(jobs)

    def call(job):
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise TimeoutError("Batch deadline exceeded")
        # A job may carry its own ceiling (the caller's per-request timeout); the batch deadline still caps it.
        timeout = remaining if job.get("timeout") is None else min(remaining, job["timeout"])
        result = request_one({**job, "timeout": timeout})
        if time.monotonic() > deadline:
            raise TimeoutError("Batch deadline exceeded")
        return result

    pool = ThreadPoolExecutor(max_workers=request["concurrency"])
    try:
        futures = {pool.submit(call, job): i for i, job in enumerate(jobs)}
        for future in as_completed(futures, timeout=max(0, deadline - time.monotonic())):
            results[futures[future]] = future.result()
        elapsed = time.monotonic() - started
        if elapsed > request["timeout"]:
            raise TimeoutError("Batch deadline exceeded")
        return {"batch": results, "elapsed_seconds": elapsed}
    except TimeoutError:
        return {"error": "BatchDeadlineExceeded"}
    finally:
        pool.shutdown(wait=False, cancel_futures=True)


def main(name=None):
    name = name or sys.argv[1]
    if not re.fullmatch(r"/opt/frontier-grading/requests/[0-9a-f]{32}", name):
        raise ValueError("Invalid request path")
    request = json.loads(Path(name + ".request").read_bytes())
    Path(name + ".request").unlink()
    result = request_batch(request) if "batch" in request else request_one(request)
    encoded = json.dumps(result)
    if len(encoded.encode()) <= INLINE_RESPONSE_BYTES:
        print(encoded)
    else:
        Path(name + ".response").write_text(encoded)
        print(json.dumps({"response_file": True}))


if __name__ == "__main__":
    main()
