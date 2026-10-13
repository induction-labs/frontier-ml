"""Trusted H100-local HTTP client enforcing the complete move-response deadline.

Installed under /opt/frontier-grading, inaccessible to the candidate UID. Timings
include sending the observation and reading the entire response, including any
feature processing/search/inference, but exclude Modal file-transfer/exec RPCs.
A late response is recorded too (marked late), so the evaluator can match every
request to a timing and forfeit the games in a late batch.
"""
import base64
import http.client
import json
from pathlib import Path
import re
import signal
import sys
import time

MOVE_SECONDS = 1.0
TIMINGS = "/opt/frontier-grading/move-latency.jsonl"


def expired(signum, frame):
    raise TimeoutError("Move response deadline exceeded")


def perform(request):
    if request["method"] != "POST" or request["path"] != "/act":
        raise ValueError("Move client accepts only POST /act")
    body = base64.b64decode(request["body"], validate=True)
    connection = http.client.HTTPConnection("127.0.0.1", 8000, timeout=MOVE_SECONDS)
    previous = signal.signal(signal.SIGALRM, expired)
    started = time.monotonic()
    try:
        # Socket inactivity timeouts alone permit indefinitely trickled bodies.
        signal.setitimer(signal.ITIMER_REAL, MOVE_SECONDS)
        connection.request("POST", "/act", body=body, headers=request["headers"])
        response = connection.getresponse()
        data = response.read(16 * 1024**2 + 1)
        elapsed = time.monotonic() - started
        if elapsed > MOVE_SECONDS:
            raise TimeoutError("Move response deadline exceeded")
        if len(data) > 16 * 1024**2:
            raise ValueError("Response exceeds size limit")
        return {"status": response.status, "body": base64.b64encode(data).decode(),
                "elapsed_seconds": elapsed}
    except TimeoutError as exc:  # the alarm, the socket timeout or the measured interval
        signal.setitimer(signal.ITIMER_REAL, 0)
        exc.elapsed_seconds = time.monotonic() - started
        raise
    finally:
        signal.setitimer(signal.ITIMER_REAL, 0)
        signal.signal(signal.SIGALRM, previous)
        connection.close()


def record(timing):
    with open(TIMINGS, "a") as stream:
        stream.write(json.dumps(timing) + "\n")


def main():
    name = sys.argv[1]
    if not re.fullmatch(r"/opt/frontier-grading/requests/[0-9a-f]{32}", name):
        raise ValueError("Invalid request path")
    try:
        request = json.loads(Path(name + ".request").read_bytes())
        Path(name + ".request").unlink()
        result = perform(request)
        record({"elapsed_seconds": result["elapsed_seconds"]})
    except Exception as exc:
        if isinstance(exc, TimeoutError):  # a late move: its batch's games are forfeited, its time still counts
            record({"elapsed_seconds": getattr(exc, "elapsed_seconds", MOVE_SECONDS), "late": True})
        result = {"error": type(exc).__name__}
    # Match the shared Harbor bridge protocol: small responses travel in stdout,
    # large responses advertise their file. The move deadline stays task-local.
    encoded = json.dumps(result)
    if len(encoded.encode()) <= 64 * 1024:
        print(encoded)
    else:
        Path(name + ".response").write_text(encoded)
        print(json.dumps({"response_file": True}))


if __name__ == "__main__":
    main()
