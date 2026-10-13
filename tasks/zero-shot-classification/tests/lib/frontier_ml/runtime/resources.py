"""Private grading resources using Harbor's environment factory and public API.

One shared event loop services synchronous scorer threads, including concurrent
LCB resources. Harbor and provider SDK caches must stay on the same event loop.
"""
import asyncio
import atexit
from concurrent.futures import TimeoutError as FutureTimeout
from contextlib import contextmanager
import json
import logging
import math
import os
from pathlib import Path
import shlex
import shutil
import tempfile
import threading
import time
import uuid

from harbor.environments.factory import EnvironmentFactory
from harbor.models.task.config import EnvironmentConfig, NetworkPolicy
from harbor.models.trial.config import EnvironmentConfig as RuntimeConfig, ResourceMode
from harbor.models.trial.paths import TrialPaths

from ..validation.errors import EvaluationError
from ..validation.evaluation import REQUEST_ATTEMPTS, RETRY_BACKOFF_SECONDS, RequestFailure, failure_reason
from .environments import use_recoverable_modal


_loop_lock = threading.Lock()
_loop = None
_active = set()
logger = logging.getLogger(__name__)


def harbor_loop():
    global _loop
    with _loop_lock:
        if _loop is None:
            _loop = asyncio.new_event_loop()
            threading.Thread(target=_loop.run_forever, name="grading-harbor", daemon=True).start()
        return _loop


@atexit.register
def close_active_resources():
    for resource in tuple(_active):
        try:
            resource.__exit__(None, None, None)
        except Exception:
            pass  # cleanup_failed remains in the resource's durable record.


def resource_settings(config, role):
    """Resolve operator settings, never submission files or agent configuration."""
    credentials = json.loads(os.environ.get("FML_GRADING_CREDENTIALS_SECRET") or "{}")
    if not isinstance(credentials, dict) or any(not isinstance(k, str) or not isinstance(v, str) for k, v in credentials.items()):
        raise ValueError("FML_GRADING_CREDENTIALS_SECRET must map environment names to strings")
    os.environ.update(credentials)  # Only this trusted verifier process; never child env.
    resources = config.get("grading_resources")
    if resources is None:
        resources = json.loads(os.environ.get("FML_GRADING_RESOURCES") or "{}")
    settings = resources.get(role, {})
    if not isinstance(settings, dict):
        raise ValueError(f"grading_resources.{role} must be an object (type, optional image and kwargs)")
    if set(settings) - {"type", "image", "kwargs"}:
        raise ValueError(f"Unknown {role} resource settings")
    # Default: Modal builds the task's own Dockerfile for this role (rendered into /tests/images/<role>).
    settings = {"type": "modal", **settings}
    if settings.get("image") is None:
        build = (config.get("grading_images") or {}).get(role)
        if not build:
            raise ValueError(f"Configure grading_resources.{role}.image or FML_GRADING_RESOURCES; "
                             f"this task carries no {role} Dockerfile")
        settings["build"] = build
    elif not isinstance(settings["image"], str) or settings["image"].startswith("im-"):
        raise ValueError("Grading images must be prebuilt container registry references, not Modal image IDs")
    kwargs = settings.get("kwargs", {})
    if not isinstance(kwargs, dict):
        raise ValueError("Resource kwargs must be an object")
    # Do not let generic environment kwargs weaken the grading boundary or
    # mount private verifier files into a candidate environment.
    forbidden = {"secrets", "volumes", "mounts", "persistent_env", "network_policy",
                 "phase_network_policies", "task_env_config", "extra_docker_compose",
                 "cpu_enforcement_policy", "memory_enforcement_policy", "keep_containers",
                 "override_cpus", "override_memory_mb", "override_storage_mb", "override_gpus",
                 "override_tpu", "environment_dir", "trial_paths", "session_id", "environment_name"}
    if forbidden.intersection(kwargs):
        raise ValueError("Grading resource kwargs cannot override isolation or resource limits")
    runtime = runtime_config({**settings, "kwargs": kwargs})
    EnvironmentFactory.validate_resource_policies(runtime)
    return {**settings, "kwargs": dict(kwargs)}


# Modal inference sandboxes: an empty egress allowlist plus an encrypted tunnel to the grading relay
# (modal_grading.py), unless FML_MODAL_GRADING_NETWORK=block keeps Harbor's block_network and the per-exchange bridge.
MODAL_GRADING = "frontier_ml.runtime.modal_grading:GradingModalEnvironment"


def guard_candidate_network(resource):
    """Before any candidate code runs: open the relay channel and, where the provider leaves the sandbox a network
    device (Modal's egress allowlist mode), install the userspace guard (netguard.py)."""
    start = getattr(resource, "start_relay", None)
    if callable(start):
        start()
    if getattr(resource, "network_isolation", "provider") != "provider" and hasattr(resource, "annotate"):
        from . import netguard
        resource.annotate(netguard=netguard.install(resource))


def runtime_config(settings):
    kind = settings["type"]
    runtime = RuntimeConfig(type=None if ":" in kind else kind,
                            import_path=kind if ":" in kind else None, kwargs=settings["kwargs"])
    runtime = use_recoverable_modal(runtime)
    capabilities = EnvironmentFactory.resource_capabilities(runtime.type, runtime.import_path)
    if capabilities is None:
        raise ValueError("Grading backends must declare CPU and memory enforcement capabilities")
    runtime.cpu_enforcement_policy = ResourceMode.GUARANTEE if capabilities.cpu_request else ResourceMode.LIMIT
    runtime.memory_enforcement_policy = ResourceMode.GUARANTEE if capabilities.memory_request else ResourceMode.LIMIT
    return runtime


# Modal may silently upgrade a plain "H100" request to another card (its docs: H200 at H100 pricing); the "!"
# suffix pins the exact type, which is what the contract promises. FML_INFERENCE_GPU_TYPE is an operator override.
INFERENCE_GPU_TYPE = {"modal": "H100!"}
SLEEP = time.sleep  # retry backoff inside request_batch; tests replace it


def inference_gpu_type(kind):
    return os.environ.get("FML_INFERENCE_GPU_TYPE") or INFERENCE_GPU_TYPE.get(kind, "H100")


START_TIMEOUT_SECONDS = 3600  # sandbox creation + image pull + start; see GradingResource._start


class GradingResource:
    def __init__(self, settings, *, role, max_seconds, logs_dir=None, volumes=None):
        if type(max_seconds) is not int or not 1 <= max_seconds <= 86400:
            raise ValueError("Resource lifetime must be 1..86400 seconds")
        self.settings, self.role, self.max_seconds = settings, role, max_seconds
        # Trusted Modal volume mounts ({path: name}); settings kwargs may not add any.
        self.volumes = dict(volumes) if volumes and settings["type"] == "modal" else {}
        self.name = "fml-" + role + "-" + uuid.uuid4().hex
        self.directory = Path(logs_dir or os.environ.get("FML_GRADING_RESOURCE_LOG_DIR", "/logs/verifier/resources")) / self.name
        self.loop = None
        self.environment = None
        self.deadline = None
        self.relay, self.relay_attempted = None, False
        self.watchdog = None
        self.stop_lock = None
        self.stopped = False
        self.expired = False
        self.state = "created"
        self.notes = {}  # operator diagnostics written into resource.json (e.g. the GPU nvidia-smi reported)

    def _record(self, state=None):
        self.state = state or self.state
        self.directory.mkdir(parents=True, exist_ok=True)
        (self.directory / "resource.json").write_text(json.dumps({
            "name": self.name, "role": self.role, "backend": self.settings["type"],
            "image": self.settings.get("image") or "built from " + self.settings["build"],
            "state": self.state, "max_seconds": self.max_seconds,
            "provider_lifetime": self.settings["type"] == "modal", **self.notes,
        }, indent=2) + "\n")

    def annotate(self, **notes):
        """Add diagnostic fields to the durable resource record."""
        self.notes.update(notes)
        self._record()

    @property
    def network_isolation(self):
        """How outbound networking is cut off: "provider" (block_network), or the environment's own description."""
        return getattr(self.environment, "network_isolation", "provider")

    def start_relay(self):
        """Open the trusted relay channel to this sandbox (relay_client.py), once; None keeps the per-exchange bridge."""
        if self.relay_attempted:
            return self.relay
        self.relay_attempted = True
        if os.environ.get("FML_GRADING_RELAY", "1").strip().lower() in {"0", "false", "no", "off"}:
            self.annotate(relay="disabled (FML_GRADING_RELAY=0)")
            return None
        from . import relay_client
        try:
            self.relay = relay_client.start(self, sandbox_python(self))
        except Exception as exc:  # noqa: BLE001 - the bridge still works; record why the relay did not
            self.relay = None
            self.annotate(relay=f"unavailable: {type(exc).__name__}: {str(exc)[:300]}")
            return None
        self.annotate(relay=self.relay.kind if self.relay else "none (per-exchange bridge)",
                      network_isolation=self.network_isolation)
        return self.relay

    def relay_violations(self):
        """Outbound connections the relay's monitor saw the candidate open, else []."""
        relay = self.relay
        if relay is None or not callable(getattr(relay, "violations", None)):
            return []
        report = relay.violations()
        self.annotate(network_monitor={"scans": report.get("scans"), "violations": report.get("violations")})
        return report.get("violations") or []

    def run(self, coroutine, timeout, what="operation"):
        future = asyncio.run_coroutine_threadsafe(coroutine, self.loop)
        try:
            return future.result(timeout=timeout)
        except FutureTimeout:
            future.cancel()
            # The prefix is matched by callers that classify transport faults; the suffix names the hung step (on
            # 2026-09-30 two grades died this way after their sandbox had started, and the record could not say where).
            raise TimeoutError(f"Grading resource operation timed out: {what} did not return within {timeout} s") from None

    async def _start(self):
        self.stop_lock = asyncio.Lock()
        self.directory.mkdir(parents=True, exist_ok=True)
        context = self.directory / "environment"
        context.mkdir()
        if self.settings.get("build"):
            shutil.copyfile(Path(self.settings["build"]) / "Dockerfile", context / "Dockerfile")
        kind = self.settings["type"]
        kwargs = dict(self.settings["kwargs"])
        inference = self.role == "inference"
        runtime_type = kind
        if kind == "modal":
            # Harbor exposes Modal's external lifetime as a constructor option.
            kwargs["sandbox_timeout_secs"] = self.max_seconds
            if self.volumes:
                kwargs["volumes"] = self.volumes
            if inference and os.environ.get("FML_MODAL_GRADING_NETWORK", "allowlist").strip().lower() != "block":
                import random
                runtime_type = MODAL_GRADING
                kwargs["relay_port"] = random.randint(40000, 59999)  # clear of the ports servers use
        runtime = runtime_config({**self.settings, "type": runtime_type, "kwargs": kwargs})
        task = EnvironmentConfig(docker_image=self.settings.get("image"),
                                 cpus=8 if inference else 2, memory_mb=81920 if inference else 4096,
                                 gpus=1 if inference else 0, gpu_types=[inference_gpu_type(kind)] if inference else None,
                                 network_mode="no-network", workdir="/tmp", build_timeout_sec=1800)
        self.environment = EnvironmentFactory.create_environment_from_config(
            runtime, environment_dir=context, environment_name=self.name,
            session_id=self.name, trial_paths=TrialPaths(self.directory),
            task_env_config=task, network_policy=NetworkPolicy(network_mode="no-network"), mounts=[])
        if not self.environment.capabilities.disable_internet:
            raise ValueError("Grading backend must enforce disabled outbound networking")
        self._record("starting")
        # The provider may queue the sandbox: on 2026-09-30, with ~30 GPU grading sandboxes in flight, Modal took 14 min to
        # schedule one H100 and grades died at a 1,800 s allowance while the submission sat intact. One hour still fits
        # every task's verifier budget except the shortest (Harbor's own limit ends those first).
        await asyncio.wait_for(self.environment.start(force_build=False), timeout=START_TIMEOUT_SECONDS)
        self.deadline = time.monotonic() + self.max_seconds
        self.watchdog = asyncio.create_task(self._expire())
        self._record("running")

    async def _expire(self):
        await asyncio.sleep(self.max_seconds)
        self.expired = True
        await self._stop()

    async def _stop(self):
        if self.environment is None:
            return
        async with self.stop_lock:
            if self.stopped:
                return
            try:
                await asyncio.wait_for(self.environment.stop(delete=True), timeout=120)
            except BaseException:
                self._record("cleanup_failed")
                raise
            self.stopped = True
            self._record("expired" if self.expired else "stopped")

    def __enter__(self):
        self.loop = harbor_loop()
        _active.add(self)
        try:
            self.run(self._start(), START_TIMEOUT_SECONDS + 30, f"{self.role} sandbox start")
            return self
        except BaseException:
            self.__exit__(None, None, None)
            raise

    def __exit__(self, *exc):
        violations = []
        if exc[0] is None and self.network_isolation != "provider" and not (self.expired or self.stopped):
            violations = self.relay_violations()

        async def close():
            if self.watchdog is not None:
                self.watchdog.cancel()
                await asyncio.gather(self.watchdog, return_exceptions=True)
            if self.relay is not None:
                self.relay.close()
            await self._stop()
        try:
            self.run(close(), 150, f"{self.role} sandbox stop")
        finally:
            _active.discard(self)
        if violations:
            # Recorded (resource.json network_monitor) and logged, never a grade failure: Modal's empty egress
            # allowlist already drops these packets, and an honest program that skips libc (a Go or static binary
            # phoning home) shows up here exactly like a deliberate bypass of the userspace guard would.
            shown = ", ".join(f"{v['protocol']} {v['host']}:{v['port']}" for v in violations[:10])
            logger.warning("The submission's UID attempted outbound connections during grading (%s); recorded only", shown)

    def _timeout(self, seconds):
        remaining = self.deadline - time.monotonic()
        if self.expired or self.stopped or remaining <= 0:
            raise TimeoutError("Grading resource lifetime expired")
        return max(1, math.ceil(min(seconds, remaining)))

    def exec(self, command, *, timeout=60, env=None, check=True):
        seconds = self._timeout(timeout)
        result = self.run(self.environment.exec(command, timeout_sec=seconds, user="root", env=env), seconds + 10,
                          f"exec {command[:120]!r}")
        if check and result.return_code:
            # Keep the "Grading sandbox command failed" prefix: pokemon's transport_fault classifies on it. Name the
            # command, since a -1 with empty stderr (the sandbox or exec went away) is otherwise undiagnosable.
            raise RuntimeError(f"Grading sandbox command failed ({result.return_code}) running {command[:120]!r}: "
                               f"{(result.stderr or '')[-1000:]}")
        return result

    def upload(self, source, target):
        self.run(self.environment.upload_file(Path(source), target), self._timeout(600), f"upload {target}")

    def upload_dir(self, source, target):
        self.run(self.environment.upload_dir(Path(source), target), self._timeout(1800), f"upload_dir {target}")

    def download(self, source, target):
        self.run(self.environment.download_file(source, Path(target)), self._timeout(600), f"download {source}")


@contextmanager
def grading_resource(config, role, max_seconds, volumes=None):
    with GradingResource(resource_settings(config, role), role=role, max_seconds=max_seconds, volumes=volumes) as resource:
        yield resource


BRIDGE_PYTHON = ("/usr/local/bin/python3", "/usr/bin/python3", "/opt/venv/bin/python3", "/opt/conda/bin/python3")


class _ChannelFailure(Exception):
    """The relay channel itself failed; the exchange falls back to the per-exchange bridge."""


# The relay saves the per-exchange bridge's upload/exec/download round trips, which matters for many small requests
# (pokemon's ~334 KiB moves). An exchange bigger than this goes by the bridge's file upload instead: end-of-turn sends
# 81-130 MB of FLAC per conversation, base64 twice over by the time it is a relay POST, and on long verifier-to-sandbox
# routes those POSTs never reached the relay ("unknown exchange", 2026-10-09: Kimi end-of-turn x2 and
# music-transcription, Fable end-of-turn), while the bridge graded the same submissions.
RELAY_MAX_BYTES = 32 * 1024 * 1024


def _payload_bytes(payload):
    """The request bodies an exchange carries (base64 text), the bulk of its size."""
    if isinstance(payload.get("batch"), list):
        return sum(len(item.get("body") or "") for item in payload["batch"] if isinstance(item, dict))
    return len(payload.get("body") or "")


class BridgeUnavailable(EvaluationError):
    """The trusted HTTP bridge itself could not run in the grading sandbox; the submission was not consulted."""


def sandbox_python(resource):
    """The interpreter that runs the trusted bridge as root inside the candidate sandbox, resolved once.

    Harbor's exec does not necessarily see the image's Dockerfile PATH (on Modal the command runs under
    `su`, and Ubuntu-based images then reset PATH to the system default), so a bare `python3` exits 127
    in conda-based images such as the babylm-replay image. Prefer the exec PATH, then the standard
    locations; every candidate must import the bridge's standard-library modules.
    """
    cached = getattr(resource, "_bridge_python", None)
    if cached:
        return cached
    script = ('for p in "$(command -v python3 2>/dev/null)" ' + " ".join(BRIDGE_PYTHON) + '; do '
              'if [ -n "$p" ] && [ -x "$p" ] && "$p" -I -c "import base64, http.client, json, concurrent.futures" >/dev/null 2>&1; '
              'then echo "$p"; exit 0; fi; done; echo "PATH=$PATH" >&2; exit 127')
    result = resource.exec(script, timeout=120, check=False)
    lines = (result.stdout or "").strip().splitlines()
    if result.return_code or not lines or not lines[-1].startswith("/"):
        raise BridgeUnavailable("The grading image has no python3 for the trusted HTTP bridge (checked the exec PATH and "
                                + ", ".join(BRIDGE_PYTHON) + "); fix the inference image, the submission was not consulted. "
                                + (result.stderr or "").strip()[-300:])
    resource._bridge_python = lines[-1]
    return lines[-1]


class HarborHTTPTransport:
    """HTTP over Harbor file transfer/exec; no provider-specific public tunnel.

    Requests use files to support image payloads larger than shell argument limits.
    Small responses return in exec stdout; large responses use files to stay below
    provider stdout limits. Scalar requests include control-plane wall time;
    batched throughput and `request_timed` use the trusted HTTP client's clock inside the sandbox.
    """
    def __init__(self, resource):
        self.resource = resource
        self.max_response_bytes = 16 * 1024 * 1024
        # Per-request phase timings on the verifier's clock (upload, exec, fetch, cleanup), for transport_breakdown().
        self.exchanges = []

    def start_relay(self):
        """The resource's relay channel (started once on first use); None means the per-exchange bridge below."""
        start = getattr(self.resource, "start_relay", None)
        return start() if callable(start) else None

    def _relay_exchange(self, relay, payload, timeout):
        """One exchange through the open relay channel: the same bridge payload and answer, without the provider's
        upload, exec, download and cleanup round trips. A failure of the channel itself raises _ChannelFailure."""
        started = time.monotonic()
        record = {"bytes": len(json.dumps(payload)), "channel": relay.kind}
        try:
            try:
                answer = relay.call(payload, timeout)  # blocking, in this thread (see relay_client.HTTPRelay)
            except Exception as exc:  # noqa: BLE001 - the channel, never the bridge or the candidate's server
                record["channel_error"] = f"{type(exc).__name__}: {str(exc)[:300]}"
                raise _ChannelFailure(record["channel_error"]) from exc
            record["exec"] = time.monotonic() - started
            if answer.get("relay_error") == "unknown exchange":
                # The relay never received this POST, even after relay_client's re-sends, so nothing ran: the
                # channel lost it, and the per-exchange bridge runs it instead (end-of-turn, 2026-10-09).
                record["channel_error"] = "unknown exchange after re-sends"
                raise _ChannelFailure("The relay never received the exchange (unknown exchange after re-sends)")
            if "relay_error" in answer:
                raise BridgeUnavailable("The trusted relay could not run the HTTP bridge; the submission was not consulted: "
                                        + answer["relay_error"])
            value = answer["value"]
            if not isinstance(value, dict):
                raise BridgeUnavailable("The trusted relay returned an invalid bridge answer")
            record["relay"] = answer.get("relay_seconds")
            if isinstance(value.get("elapsed_seconds"), (int, float)):
                record["server"] = value["elapsed_seconds"]
            return value
        finally:
            record["total"] = time.monotonic() - started
            record["verifier_load"] = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
            self.exchanges.append(record)

    def _exchange(self, payload, timeout):
        relay = None if getattr(self, "_relay_broken", False) else self.start_relay()
        if relay is not None and _payload_bytes(payload) > RELAY_MAX_BYTES:
            relay = None  # one large request: the bridge's file upload, not a long POST through the tunnel
        if relay is not None:
            try:
                return self._relay_exchange(relay, payload, timeout)
            except _ChannelFailure as failure:
                # The relay channel broke (nothing answered): this exchange and the rest of the grade use the
                # per-exchange bridge below, which runs the same bridge on the same payload.
                self._relay_broken = True
                if hasattr(self.resource, "annotate"):
                    self.resource.annotate(relay_fallback=str(failure))
        name = uuid.uuid4().hex
        remote = "/opt/frontier-grading/requests/" + name
        python = sandbox_python(self.resource)
        with tempfile.TemporaryDirectory(prefix="grading-http-") as directory:
            request, response = Path(directory) / "request.json", Path(directory) / "response.json"
            started = time.monotonic()
            request.write_text(json.dumps(payload))
            record = {"bytes": request.stat().st_size, "write": time.monotonic() - started}
            cleanup = True
            try:
                mark = time.monotonic()
                self.resource.upload(request, remote + ".request")
                record["upload"] = time.monotonic() - mark
                mark = time.monotonic()
                result = self.resource.exec(f"{python} -I /opt/frontier-grading/http_bridge.py " + shlex.quote(remote),
                                            timeout=timeout + 5, check=False)
                record["exec"] = time.monotonic() - mark
                # The bridge reports every HTTP-level problem inside its JSON and exits 0, so a nonzero exit or
                # non-JSON output means the trusted client did not run: a verifier fault, never the server's.
                try:
                    if getattr(result, "return_code", 0):
                        raise ValueError(f"exit {result.return_code}")
                    value = json.loads(result.stdout)
                except ValueError as exc:
                    raise BridgeUnavailable(f"The trusted HTTP bridge did not run ({exc}); the submission was not consulted: "
                                            + ((getattr(result, "stderr", "") or "").strip()[-500:] or "(no stderr)")) from None
                if value.get("response_file") is True:
                    mark = time.monotonic()
                    self.resource.download(remote + ".response", response)
                    value = json.loads(response.read_bytes())
                    record["fetch"] = time.monotonic() - mark
                else:
                    cleanup = False  # The bridge removed the request and created no response file.
                if isinstance(value.get("elapsed_seconds"), (int, float)):
                    record["server"] = value["elapsed_seconds"]  # the trusted client's in-sandbox HTTP time
                return value
            finally:
                if cleanup:
                    mark = time.monotonic()
                    self._cleanup(remote, record)
                    record["cleanup"] = time.monotonic() - mark
                record["total"] = time.monotonic() - started
                record["verifier_load"] = os.getloadavg()[0] if hasattr(os, "getloadavg") else None
                self.exchanges.append(record)

    def _cleanup(self, remote, record):
        # This exact rm is idempotent. Never retry the bridge or model request.
        # A cleanup outage must neither discard a fetched response nor mask the
        # exception that caused cleanup; the sandbox is destroyed on exit anyway.
        command = "rm -f " + shlex.quote(remote + ".request") + " " + shlex.quote(remote + ".response")
        for attempt in range(3):
            record["cleanup_attempts"] = attempt + 1
            try:
                result = self.resource.exec(command, timeout=10, check=False)
                if not result.return_code:
                    return
                error = f"exit {result.return_code}"
            except Exception as exc:
                error = type(exc).__name__  # Exception text may contain request data.
            record.setdefault("cleanup_errors", []).append(error)
            if attempt < 2:
                SLEEP(0.25 * (attempt + 1))
        record["cleanup_failed"] = True
        logger.warning("HTTP bridge temporary-file cleanup failed after 3 attempts; deferred to sandbox teardown")

    def transport_breakdown(self):
        """Where each scalar request's wall time went, on the verifier's clock: medians, p90 and max per phase, the
        in-sandbox server time and the remainder (Modal control plane), verifier load, and both sandboxes' regions."""
        def summary(values):
            values = sorted(v for v in values if isinstance(v, (int, float)))
            if not values:
                return None
            return {"median": values[len(values) // 2], "p90": values[int(0.9 * (len(values) - 1))],
                    "max": values[-1], "sum": sum(values)}
        rows = self.exchanges
        phases = {key: summary(r.get(key) for r in rows)
                  for key in ("bytes", "write", "upload", "exec", "fetch", "cleanup", "relay", "server", "total", "verifier_load")}
        phases["exec_minus_server"] = summary(r["exec"] - r["server"] for r in rows if "exec" in r and "server" in r)
        return {"requests": len(rows), "phases": phases, "verifier_cpus": os.cpu_count(),
                "cleanup_retries": sum(max(0, r.get("cleanup_attempts", 0) - 1) for r in rows),
                "cleanup_failures": sum(bool(r.get("cleanup_failed")) for r in rows),
                "verifier_region": os.environ.get("MODAL_REGION"), "verifier_cloud": os.environ.get("MODAL_CLOUD_PROVIDER"),
                "inference_region": getattr(self.resource, "region", None),
                "inference_cloud": getattr(self.resource, "cloud", None)}

    @staticmethod
    def _error_text(value):
        """The bridge's per-request failure: the exception class name and, when it recorded one, its message."""
        detail = value.get("detail")
        return "Submission HTTP request failed: " + value["error"] + (f": {detail}" if detail else "")

    @staticmethod
    def _decode(value):
        import base64
        if "error" in value:
            raise RuntimeError(HarborHTTPTransport._error_text(value))
        return value["status"], base64.b64decode(value["body"], validate=True)

    @classmethod
    def _settle(cls, value, raise_on_failure):
        """One batch item: the parsed JSON body, or (lenient mode) a RequestFailure describing why there is none."""
        if "error" in value:
            error = RuntimeError(cls._error_text(value))
            if raise_on_failure:
                raise error
            return RequestFailure(failure_reason(error), str(error))
        import base64
        status, data = value["status"], base64.b64decode(value["body"], validate=True)
        if status != 200:
            error = RuntimeError(f"Submission returned HTTP {status}: {data[:500]!r}")
            if raise_on_failure:
                raise error
            return RequestFailure("http_status", f"Submission returned HTTP {status}: {data[:200]!r}")
        try:
            return json.loads(data)
        except ValueError:
            if raise_on_failure:
                raise
            return RequestFailure("invalid_json", "Submission did not return valid JSON")

    def raw_request(self, method, path, *, body=b"", headers=None, timeout=30):
        import base64
        if timeout <= 0 or not path.startswith("/") or path.startswith("//"):
            raise ValueError("Invalid HTTP request")
        return self._decode(self._exchange({"method": method, "path": path, "body": base64.b64encode(body).decode(),
                                           "headers": headers or {}, "timeout": timeout,
                                           "max_response_bytes": self.max_response_bytes}, timeout))

    def request_batch(self, requests, *, concurrency, minimum_per_second=None, timeout_seconds=3600, on_complete=None,
                      raise_on_failure=True, request_timeout_seconds=None):
        """Bounded chunks keep labels off the GPU and exclude RPC latency from serving throughput.

        With raise_on_failure=False (the shared evaluator), an item whose request fails at the transport level
        (bridge error, non-200 status, non-JSON body) is sent again on the identical payload after
        RETRY_BACKOFF_SECONDS, up to REQUEST_ATTEMPTS times, as long as the backoff fits in the budget; items that
        still fail, or timed out, come back as RequestFailure entries. Retry waves and their backoff are added
        to the timed elapsed seconds: a retry is the server's problem, so it counts against the deadline.
        With request_timeout_seconds, the trusted client also caps each request at that many seconds (the batch
        deadline still applies): a request that has not answered by then is a timeout for its item alone, never
        retried, instead of holding the whole batch until its deadline.
        """
        import base64
        from ..validation.errors import VerificationFailure
        if not requests or type(concurrency) is not int or not 1 <= concurrency <= 64:
            raise VerificationFailure("Batch must be nonempty; concurrency must be between 1 and 64")
        if request_timeout_seconds is not None and (isinstance(request_timeout_seconds, bool)
                                                    or not isinstance(request_timeout_seconds, (int, float))
                                                    or not math.isfinite(request_timeout_seconds)
                                                    or request_timeout_seconds <= 0):
            raise ValueError("request_timeout_seconds must be positive and finite")
        budget = timeout_seconds
        if minimum_per_second is not None:
            if minimum_per_second <= 0:
                raise ValueError("Throughput must be positive")
            budget = min(budget, len(requests) / minimum_per_second)
        if budget <= 0 or not math.isfinite(budget):
            raise ValueError("Batch timeout must be positive and finite")
        jobs = []
        for path, payload in requests:
            if not path.startswith("/") or path.startswith("//"):
                raise ValueError("Invalid HTTP request")
            jobs.append({"method": "POST", "path": path, "headers": {"Content-Type": "application/json"},
                         "max_response_bytes": self.max_response_bytes,
                         "body": base64.b64encode(json.dumps(payload, allow_nan=False).encode()).decode()})
            if request_timeout_seconds is not None:
                jobs[-1]["timeout"] = request_timeout_seconds  # the bridge caps it by the remaining batch time
        responses, elapsed, retries = [None] * len(jobs), 0.0, 0
        errors = {}  # index -> error texts of every failed attempt so far
        pending = list(range(len(jobs)))
        for attempt in range(1 if raise_on_failure else REQUEST_ATTEMPTS):
            if attempt:
                backoff = RETRY_BACKOFF_SECONDS[attempt - 1]
                if budget - elapsed - backoff <= 0:
                    break  # no room for a retry wave; the pending items stay failed
                SLEEP(backoff)
                elapsed += backoff
                retries += len(pending)
            retry = []
            for offset in range(0, len(pending), 64):
                chunk = pending[offset:offset + 64]
                remaining = budget - elapsed
                if remaining <= 0:
                    raise VerificationFailure(f"Batch exceeded {budget:g}s deadline; score is zero")
                value = self._exchange({"batch": [jobs[i] for i in chunk], "concurrency": concurrency, "timeout": remaining},
                                       remaining)
                if value.get("error") == "BatchDeadlineExceeded":
                    raise VerificationFailure(f"Batch exceeded {budget:g}s deadline; score is zero")
                if len(value["batch"]) != len(chunk) or not 0 <= value["elapsed_seconds"] <= remaining:
                    raise VerificationFailure("Invalid trusted HTTP batch result")
                decode_started = time.monotonic()
                for index, response in zip(chunk, value["batch"]):
                    outcome = self._settle(response, raise_on_failure)
                    if isinstance(outcome, RequestFailure):
                        errors.setdefault(index, []).append(outcome.error)
                        outcome = RequestFailure(outcome.reason, f"failed {len(errors[index])} attempts: "
                                                 + " | ".join(errors[index]), len(errors[index]))
                        if outcome.reason != "timeout" and attempt + 1 < REQUEST_ATTEMPTS:
                            retry.append(index)
                    responses[index] = outcome
                elapsed += value["elapsed_seconds"] + time.monotonic() - decode_started
                if on_complete is not None:
                    on_complete(sum(response is not None for response in responses))
            pending = retry
            if not pending:
                break
        if elapsed > budget:
            raise VerificationFailure(f"Batch exceeded {budget:g}s deadline; score is zero")
        timing = {"elapsed_seconds": elapsed, "items": len(requests),
                  "items_per_second": len(requests) / max(elapsed, 1e-12),
                  "deadline_seconds": budget, "concurrency": concurrency, "retried_requests": retries,
                  "timing_scope": "trusted HTTP client; provider transfer/exec latency excluded; retries included"}
        if request_timeout_seconds is not None:
            timing["request_timeout_seconds"] = request_timeout_seconds
        if getattr(self.resource, "gpu", None):
            timing["gpu"] = self.resource.gpu  # the exact device the throughput was measured on
        if getattr(self.resource, "relay", None) is not None:
            timing["channel"] = self.resource.relay.kind
            timing["network_isolation"] = self.resource.network_isolation
        return responses, timing

    def request_timed(self, method, path, payload, *, timeout=30):
        """One JSON request timed by the trusted HTTP client inside the sandbox: (response, elapsed_seconds).

        `request` charges an upload RPC, an exec RPC and an interpreter start to every call (0.35-0.5 s of
        grader control plane on Modal), which a per-request speed measurement must not count against the
        server: the bridge's batch path times the HTTP exchange itself, so the request rides it alone with
        concurrency 1. Failures raise as `request` does; a request the bridge gave up on at the deadline raises
        TimeoutError.
        """
        import base64
        if timeout <= 0 or not path.startswith("/") or path.startswith("//"):
            raise ValueError("Invalid HTTP request")
        job = {"method": method, "path": path, "headers": {"Content-Type": "application/json"},
               "max_response_bytes": self.max_response_bytes,
               "body": base64.b64encode(json.dumps(payload, allow_nan=False).encode()).decode()}
        value = self._exchange({"batch": [job], "concurrency": 1, "timeout": timeout}, timeout)
        if value.get("error") == "BatchDeadlineExceeded":
            raise TimeoutError(f"Submission HTTP request timed out after {timeout:g}s")
        if len(value.get("batch") or ()) != 1 or not 0 <= value.get("elapsed_seconds", -1.0) <= timeout:
            raise BridgeUnavailable("The trusted HTTP bridge returned an invalid timed result; the submission was not consulted")
        return self._settle(value["batch"][0], True), value["elapsed_seconds"]

    def request(self, method, path, payload=None, *, body=None, headers=None, timeout=30):
        # Same contract as HTTPTransport.request: a JSON payload, or a raw body (e.g. multipart video).
        headers = dict(headers or {})
        if payload is not None:
            if body is not None:
                raise ValueError("Provide payload or body, not both")
            body = json.dumps(payload).encode()
            headers["Content-Type"] = "application/json"
        status, data = self.raw_request(method, path, body=body or b"", headers=headers, timeout=timeout)
        if status != 200:
            raise RuntimeError(f"Submission returned HTTP {status}: {data[:500]!r}")
        return json.loads(data)
