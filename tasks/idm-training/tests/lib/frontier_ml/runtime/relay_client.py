"""Verifier side of the in-sandbox relay (relay.py): one HTTPS channel per grading sandbox instead of an upload, an
exec, a download and a cleanup command per HTTP exchange.

The relay serves HTTP on a high port that the grading environment exposes (modal_grading.py: Modal's encrypted
tunnel); every request carries the relay's own random token. Environments without `expose_port` keep the
per-exchange bridge (resources.HarborHTTPTransport's original path).
"""
from __future__ import annotations

import json
import logging
import secrets
import shlex
import time
import uuid
from pathlib import Path

from ..validation.errors import EvaluationError
from . import relay

log = logging.getLogger(__name__)
RELAY_PATH = "/opt/frontier-grading/relay.py"
PORT_FILE = "/opt/frontier-grading/relay.port"


def stock_sha256():
    import hashlib
    return hashlib.sha256(Path(__file__).with_name("http_bridge.py").read_bytes()).hexdigest()
WAIT_SECONDS = 20  # HTTP long-poll per round trip; an exchange that takes longer is polled again
RESEND_LIMIT = 3  # times an exchange the relay never received is sent again (see call)


class RelayUnavailable(EvaluationError):
    """The trusted relay could not run or answer; the submission was not consulted."""


def trusted_interpreter(resource, python):
    """`python` resolved to its real file if that file and every directory above it are root-owned and writable by
    root only; else None. The relay runs as root next to the candidate, which owns /opt/venv and /opt/conda, so an
    interpreter the candidate could replace must never be started there."""
    script = (f'p=$(readlink -f {shlex.quote(python)}) || exit 1; d="$p"; '
              'while :; do s=$(stat -Lc "%u %a" "$d") || exit 1; set -- $s; [ "$1" = 0 ] || exit 3; '
              'case "$2" in *[2367]?|*?[2367]) exit 4;; esac; [ "$d" = / ] && break; d=$(dirname "$d"); done; '
              'echo "$p"')
    result = resource.exec(script, timeout=60, check=False)
    lines = (result.stdout or "").strip().splitlines()
    if result.return_code or not lines or not lines[-1].startswith("/"):
        return None
    return lines[-1]


class HTTPRelay:
    """The relay's HTTP API at the URL the grading environment exposes for it.

    Calls block in the caller's thread instead of running on the resource's event loop: with any other busy Python
    thread in the verifier, every event-loop step waits for the GIL, and a 334 KiB pokemon exchange measured 0.37-0.75 s
    there against 0.07-0.15 s as blocking calls (work/relay-tunnel-bench, 2026-10-02). Bodies are deflated when that
    pays (relay.deflate), because TCP restarts slow after the idle time between a grading loop's requests.
    """
    kind = "https-tunnel"

    def __init__(self, url, edge_headers, token):
        import httpx
        self.url, self.token = url.rstrip("/"), token
        # httpx's default Accept-Encoding includes deflate, which the relay uses for answers.
        self.client = httpx.Client(timeout=httpx.Timeout(WAIT_SECONDS + 40), follow_redirects=False,
                                   headers={**edge_headers, "X-FML-Relay-Token": token})

    def wait_ready(self, timeout=120):
        deadline = time.monotonic() + timeout
        last = None
        while time.monotonic() < deadline:
            try:
                response = self.client.get(self.url + "/ready")
                if response.status_code == 200:
                    self.hello = response.json()
                    return self.hello
                last = f"HTTP {response.status_code}"
            except Exception as exc:  # noqa: BLE001 - not up yet
                last = f"{type(exc).__name__}: {exc}"
            time.sleep(1)
        raise RelayUnavailable(f"The trusted relay did not answer at its exposed port ({last})")

    def call(self, payload, timeout):
        deadline = time.monotonic() + timeout + 60
        message_id = uuid.uuid4().hex
        message = {"id": message_id, "payload": payload, "timeout": timeout}
        answer, resent = self._send("POST", "/exchange", message), 0
        while not answer.get("done"):
            if answer.get("relay_error") == "unknown exchange" and resent < RESEND_LIMIT:
                # A POST lost before it reached the relay leaves nothing to poll (_send then polls by id): send it
                # again. The relay runs an id at most once (relay.Jobs), so this never repeats the candidate's requests.
                resent += 1
                answer = self._send("POST", "/exchange", message)
                continue
            if "relay_error" in answer:
                break
            if time.monotonic() > deadline:
                raise RelayUnavailable("The trusted relay did not finish an exchange in time")
            answer = self._send("GET", f"/exchange/{message_id}", None)
        return answer

    def _send(self, method, path, body):
        error = None
        for attempt in range(4):
            headers, content = {}, None
            if body is not None:
                content = json.dumps(body).encode()
                headers["Content-Type"] = "application/json"
                packed = relay.deflate(content)
                if packed is not None:
                    content, headers["Content-Encoding"] = packed, "deflate"
            try:
                response = self.client.request(method, f"{self.url}{path}?wait={WAIT_SECONDS}", content=content,
                                               headers=headers)
                if response.status_code == 200:
                    return response.json()
                error = f"HTTP {response.status_code}: {response.text[:200]}"
            except Exception as exc:  # noqa: BLE001
                error = f"{type(exc).__name__}: {exc}"
            if method == "POST" and attempt == 0 and "HTTP" not in error:
                # The exchange may have been accepted: poll it by id instead of sending it twice.
                method, path, body = "GET", f"{path}/{(body or {}).get('id')}", None
            time.sleep(1 + attempt)
        raise RelayUnavailable("The trusted relay's exposed port stopped answering: " + error)

    def diagnose(self):
        return {"url_host": self.url.split("/")[2]}

    def violations(self):
        response = self.client.get(self.url + "/violations")
        response.raise_for_status()
        return response.json()

    def close(self):
        self.client.close()


def start(resource, python):
    """A relay for this grading resource, or None when its environment exposes no port (the per-exchange bridge stays)."""
    environment = getattr(resource, "environment", None)
    if environment is None or not callable(getattr(environment, "expose_port", None)):
        return None
    interpreter = trusted_interpreter(resource, python)
    if interpreter is None:
        log.warning("No root-owned interpreter for the relay (%s); keeping the per-exchange bridge", python)
        return None
    resource.upload(Path(__file__).with_name("relay.py"), RELAY_PATH)
    token = secrets.token_hex(32)
    with __import__("tempfile").NamedTemporaryFile("w", delete=False) as handle:
        handle.write(token)
    try:
        resource.upload(Path(handle.name), "/opt/frontier-grading/relay.token")
    finally:
        Path(handle.name).unlink(missing_ok=True)
    monitor = getattr(environment, "monitor_uid", None)
    # The port was opened at sandbox creation, clear of the ports servers use (the relay shares the candidate's
    # network namespace); 0 lets the relay pick a free one when the environment exposes ports later.
    fixed = getattr(environment, "relay_port", None)
    resource.exec(f"chmod 600 /opt/frontier-grading/relay.token && rm -f {PORT_FILE} && cd /opt/frontier-grading && "
                  f"nohup {shlex.quote(interpreter)} -I {RELAY_PATH} --http {int(fixed or 0)} --port-file {PORT_FILE} "
                  f"--token-file /opt/frontier-grading/relay.token --stock-sha256 {stock_sha256()}"
                  + (f" --monitor-uid {int(monitor)}" if monitor is not None else "")
                  + " </dev/null >/opt/frontier-grading/relay.log 2>&1 &", timeout=60)
    found = resource.exec(f"for i in $(seq 1 60); do [ -s {PORT_FILE} ] && cat {PORT_FILE} && exit 0; sleep 0.5; done; "
                          "tail -c 1500 /opt/frontier-grading/relay.log >&2; exit 1", timeout=60, check=False)
    if found.return_code or not (found.stdout or "").strip().isdigit():
        raise RelayUnavailable("The trusted relay did not bind a port: " + (found.stderr or "")[-800:])
    port = int(found.stdout.strip())
    url, headers = resource.run(environment.expose_port(port), 120, "expose relay port")
    channel = HTTPRelay(url, headers, token)
    channel.wait_ready(timeout=120)
    log.info("Grading relay up (%s, %s)", channel.kind, interpreter)
    return channel
