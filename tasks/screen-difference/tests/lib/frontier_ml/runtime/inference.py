"""Run the submitted HTTP server in a private Harbor-managed H100 environment."""
from contextlib import contextmanager
import math
import os
from pathlib import Path
import re
import time

from . import submission_volume
from .resources import grading_resource, guard_candidate_network, HarborHTTPTransport, sandbox_python
from .server import VerificationFailure
from ..validation.errors import EvaluationError

# The sandbox clock starts before the submission copy, digest check and server startup, so the lifetime must
# leave the whole batch budget available after them; the batch deadline, not the sandbox, is meant to bind.
TRANSFER_MARGIN_SECONDS = 1800   # tar extraction + digest (each bounded by 1800 s, typically minutes) and teardown
MAX_LIFETIME_SECONDS = 86400
DEFAULT_STARTUP_SECONDS = 300
DEFAULT_INFERENCE_SECONDS = 3600

GPU_QUERY = "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits"
MINIMUM_GPU_MIB = 75000
# The contract promises exactly one H100. resources.py pins Modal's exact type ("H100!"); any other card, including
# the H200 Modal substitutes for plain "H100" requests, is a hardware mismatch: one replacement sandbox is tried,
# then the run is an evaluation error. The exact device is recorded with every result.
GPU_FAMILIES = ("H100",)
GPU_ATTEMPTS = 2  # a provider that hands out the wrong card, or no answer, gets one replacement sandbox


class HardwareMismatch(EvaluationError):
    """The provisioned sandbox is not the promised single H100: the provider's fault, not the submission's."""


def default_lifetime(config):
    """inference budget + startup deadline + transfer margin, capped: what a compliant run can legitimately use."""
    inference = config.get("inference_timeout_seconds", DEFAULT_INFERENCE_SECONDS)
    startup = config.get("startup_timeout_seconds", DEFAULT_STARTUP_SECONDS)
    if type(inference) not in (int, float) or not math.isfinite(inference) or inference <= 0:
        inference = DEFAULT_INFERENCE_SECONDS
    if type(startup) is not int or startup <= 0:
        startup = DEFAULT_STARTUP_SECONDS
    return int(min(MAX_LIFETIME_SECONDS, math.ceil(inference) + startup + TRANSFER_MARGIN_SECONDS))


def parse_gpus(stdout):
    """(name, MiB) per device line of GPU_QUERY; driver warnings and blank lines are skipped."""
    devices = []
    for line in (stdout or "").splitlines():
        name, separator, memory = line.strip().rpartition(",")
        if separator and memory.strip().isdigit():
            devices.append((name.strip(), int(memory.strip())))
    return devices


def compliant_gpu(result):
    """The single H100 device (any H100 variant with at least MINIMUM_GPU_MIB), or raise HardwareMismatch
    quoting what nvidia-smi actually printed so the operator can tell a wrong card from a driver hiccup."""
    stdout, stderr = (getattr(result, "stdout", "") or ""), (getattr(result, "stderr", "") or "")
    code = getattr(result, "return_code", 0) or 0
    devices = parse_gpus(stdout)
    if (len(devices) == 1 and any(family in devices[0][0].upper() for family in GPU_FAMILIES)
            and devices[0][1] >= MINIMUM_GPU_MIB):
        return {"name": devices[0][0], "memory_mib": devices[0][1]}
    raise HardwareMismatch(f"Inference requires exactly one H100 with at least {MINIMUM_GPU_MIB:,} MiB memory; "
                           f"the sandbox reported {devices or 'no device'} (nvidia-smi exit {code}, "
                           f"stdout {stdout.strip()[:500]!r}, stderr {stderr.strip()[:500]!r})")


@contextmanager
def compliant_resource(config, lifetime, volumes):
    """A fresh inference sandbox whose GPU is the promised H100. A mismatch (or an empty nvidia-smi) is an
    infrastructure fault: one replacement sandbox is tried, then the run is an evaluation error, not a failure."""
    last = None
    for attempt in range(GPU_ATTEMPTS):
        with grading_resource(config, "inference", lifetime, volumes=volumes) as resource:
            result = resource.exec(GPU_QUERY, check=False)
            try:
                gpu = compliant_gpu(result)
            except HardwareMismatch as error:
                last = error
                if hasattr(resource, "annotate"):
                    resource.annotate(gpu=None, hardware_mismatch=str(error), gpu_attempt=attempt + 1)
                continue
            resource.gpu = gpu  # the exact device, for the result record and the batch timing
            # Where the provider placed the sandbox (Modal sets these); the verifier records its own the same way.
            where = resource.exec('printf "%s %s" "${MODAL_REGION:-}" "${MODAL_CLOUD_PROVIDER:-}"', timeout=30, check=False)
            region, _, cloud = ((getattr(where, "stdout", "") or "").strip() + " ").partition(" ")
            resource.region, resource.cloud = region or None, cloud.strip() or None
            if hasattr(resource, "annotate"):
                resource.annotate(gpu=gpu, gpu_attempt=attempt + 1, region=resource.region, cloud=resource.cloud,
                                  verifier_region=os.environ.get("MODAL_REGION"),
                                  verifier_cloud=os.environ.get("MODAL_CLOUD_PROVIDER"))
            yield resource
            return
    raise HardwareMismatch(f"{GPU_ATTEMPTS} sandboxes in a row did not provide the promised H100; the submission was not "
                           f"consulted. Last: {last}")


@contextmanager
def serve_submission(snapshot, config):
    lifetime = config.get("verifier_lifetime_seconds", default_lifetime(config))
    if type(lifetime) is not int or not 300 <= lifetime <= MAX_LIFETIME_SECONDS:
        raise VerificationFailure("Inference lifetime must be 300..86400 seconds")
    startup = config.get("startup_timeout_seconds", DEFAULT_STARTUP_SECONDS)
    if type(startup) is not int or not 0 < startup <= lifetime:
        raise VerificationFailure("startup_timeout_seconds must be a positive integer within the inference lifetime")
    snapshot = Path(snapshot)
    # A task that serves submitted weights itself (config["trusted_serve_dir"], a directory of the trusted task seed
    # holding serve.sh and any files it reads) runs that serve.sh instead of the submission's, with the same user,
    # environment and deadline. The directory is uploaded to /opt/frontier-serve, root-owned and read-only.
    trusted = config.get("trusted_serve_dir")
    if trusted is not None and not (Path(trusted) / "serve.sh").is_file():
        raise VerificationFailure("trusted_serve_dir must be a task seed directory containing serve.sh")
    if trusted is None and not (snapshot / "serve.sh").is_file():
        raise VerificationFailure("Submission has no serve.sh")
    for path in snapshot.rglob("*"):
        if path.is_symlink() or not (path.is_file() or path.is_dir()):
            raise VerificationFailure("Submission must contain only regular files and directories")
    volume = config.get("submission_volume")
    volumes = {submission_volume.MOUNT: volume} if volume else {}
    model_volume = config.get("grading_model_volume")
    if model_volume is not None:
        if not isinstance(model_volume, str) or not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9._-]{0,127}", model_volume):
            raise VerificationFailure("grading_model_volume must name an operator-owned Modal volume")
        volumes["/models"] = model_volume
    with compliant_resource(config, lifetime, volumes or None) as resource:
        resource.exec("mkdir -p /submission /workspace /opt/frontier-grading/requests; chmod 700 /opt/frontier-grading; chmod 777 /workspace")
        if volume:
            copy_from_volume(resource, config["submission_digest"],
                             submission_volume.safe_archive(config.get("submission_archive", submission_volume.ARCHIVE)))
        else:
            resource.upload_dir(snapshot, "/submission")
        resource.upload(Path(__file__).with_name("http_bridge.py"), "/opt/frontier-grading/http_bridge.py")
        if trusted is not None:
            # Root-owned and read-only; /opt/frontier-grading itself stays 0700, so the server lives beside it.
            resource.upload_dir(Path(trusted), "/opt/frontier-serve")
            resource.exec("chown -R root:root /opt/frontier-serve && chmod -R a+rX,go-w /opt/frontier-serve", timeout=30)
        transport = HarborHTTPTransport(resource)
        # Before any candidate code runs: the relay (its root interpreter vetted untouched) and the network guard.
        guard_candidate_network(resource)
        # The digest is verified; hand the server its own writable copy so read-only modes cannot break startup.
        # The preinstalled Python environment is handed over too: the contract advertises it with pip on PATH,
        # and a serve.sh that installs its vendored wheels there must not die on a root-owned site-packages.
        # The sandbox is discarded after grading and the trusted bridge runs the system interpreter, so this
        # changes nothing the submission could not already do through its own server.
        resource.exec("chown -R 65534:65534 /submission && chmod -R u+rwX /submission; "
                      "for d in /opt/venv /opt/conda; do [ -d $d ] && chown -R 65534:65534 $d; done; true")
        started = time.monotonic()
        # serve.sh starts in /submission, so relative paths that worked from /workspace/output keep working;
        # HOME stays on the writable scratch directory.
        resource.exec("cd /submission && nohup setpriv --reuid=65534 --regid=65534 --clear-groups --no-new-privs "
                      "env -i PATH=/opt/venv/bin:/usr/local/bin:/usr/bin:/bin HOME=/workspace HOST=0.0.0.0 PORT=8000 MODEL_NAME=submission "
                      "LD_LIBRARY_PATH=/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64 HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 "
                      f"bash {'/opt/frontier-serve/serve.sh' if trusted is not None else '/submission/serve.sh'} "
                      "</dev/null >/opt/frontier-grading/server.log 2>&1 &", timeout=30)
        if callable(launched := config.get("on_server_launch")):
            launched()  # in-process observers only (selfcheck.py reports the provisioning/startup boundary)
        deadline = started + startup
        last_error = "no probe completed"
        while time.monotonic() < deadline:
            if resource.expired or resource.stopped:
                raise VerificationFailure("Inference resource expired before readiness")
            try:
                status, body = transport.raw_request("GET", "/health", timeout=max(0.1, min(2, deadline-time.monotonic())))
                if status == 200 and time.monotonic() < deadline:
                    break
                last_error = f"HTTP {status} {body[:200]!r}"
            except EvaluationError:
                raise  # the verifier's own bridge failed: report that, do not wait out the deadline and blame the server
            except Exception as exc:  # noqa: BLE001 - not ready yet
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(0.1, max(0, deadline-time.monotonic())))
        else:
            raise VerificationFailure(f"Server did not become healthy in {startup}s (last /health probe: {last_error}); "
                                      "end of its log:\n" + server_log_tail(resource))
        yield {"transport": transport, "snapshot": snapshot,
               "startup_elapsed": time.monotonic() - started, "gpu": getattr(resource, "gpu", None),
               "backend": "harbor:" + resource.settings["type"], "sandbox_id": resource.name,
               "network_isolation": getattr(resource, "network_isolation", "provider"),
               "channel": getattr(getattr(resource, "relay", None), "kind", "per-exchange bridge"),
               "server_log_tail": lambda size=2000: server_log_tail(resource, size)}


def copy_from_volume(resource, digest, archive=submission_volume.ARCHIVE):
    """Extract the submission from the GPU sandbox's own volume mount; it must match the verified snapshot."""
    resource.exec(f"tar -xf {submission_volume.MOUNT}/{archive} -C /submission "
                  "--no-same-owner --no-same-permissions", timeout=1800)
    from ..scoring import architecture
    resource.upload(Path(architecture.__file__), "/opt/frontier-grading/digest.py")
    result = resource.exec(digest_command(sandbox_python(resource)), timeout=1800)
    if (result.stdout or "").strip() != digest:
        raise VerificationFailure("Submission in the GPU sandbox differs from the verified snapshot")


def digest_command(python, grading_dir="/opt/frontier-grading", submission="/submission"):
    """Materialize internal symlinks exactly as the host snapshot did, then print the submission digest.

    `-I` drops the working directory from sys.path, so digest.py's directory is added explicitly."""
    return (f"cd {grading_dir} && {python} -I -c 'import sys; sys.path.insert(0, \"{grading_dir}\"); "
            "from digest import materialize_internal_symlinks, submission_digest; "
            f"materialize_internal_symlinks(\"{submission}\"); print(submission_digest(\"{submission}\"))'")


def server_log_tail(resource, size=2000):
    """The end of the candidate server's output, for the operator's diagnosis of a failed startup."""
    try:
        return resource.exec(f"tail -c {size} /opt/frontier-grading/server.log", check=False).stdout or "(empty)"
    except Exception as exc:
        return f"(unavailable: {type(exc).__name__})"
