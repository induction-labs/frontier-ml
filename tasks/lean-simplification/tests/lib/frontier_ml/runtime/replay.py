"""Training replay: run a submission's train.sh offline on a pinned corpus, then serve its outputs.

Used by tasks whose artifact is training code (babylm-replay). Additive to inference.py:
the same candidate-only H100 resource and HTTP bridge are used, with this protocol:

1. Upload the frozen submission to /submission and the pinned corpus to /corpus (root-owned,
   read-only). Verify corpus hashes inside the sandbox.
2. Run `bash /submission/train.sh` as UID 65534 with CORPUS_DIR=/corpus, OUTPUT_DIR=/replay/output,
   HOME/TMPDIR under /replay, no network, under a hard wall-clock limit enforced inside the sandbox.
   A nonzero exit or exceeding the limit fails the submission.
3. Kill every UID-65534 process, make /replay/output root-owned and read-only, delete all other
   candidate-writable state and hide /corpus.
4. Launch `bash /submission/serve.sh` with WEIGHTS_DIR=/replay/output; /health within 300 s.
5. `restart()` kills the server, wipes candidate-writable state again and relaunches it, so a
   later audit phase cannot see requests from an earlier phase.

The agent's self-check (frontier_ml/selfcheck.py) runs the same steps with probe_training in place of
run_training: train.sh gets a shorter TRAIN_BUDGET_SECONDS under the same limit, is watched for a short
window only and stopped if it is still running, and serve_trained then starts serve.sh only if it exited 0
within that window.

LocalDockerResource implements the same resource interface with a local Docker daemon. It is an
operator tool for pilots and integration tests (explicit trusted configuration only).
"""
from contextlib import contextmanager
import hashlib
from pathlib import Path
import shlex
import subprocess
import time
import uuid

from .server import VerificationFailure
from ..validation.errors import EvaluationError

UID = 65534
CONTROL = "/opt/frontier-grading"
PATH = "/opt/conda/bin:/opt/venv/bin:/usr/local/bin:/usr/bin:/bin"
LIBS = "/usr/local/nvidia/lib:/usr/local/nvidia/lib64:/usr/local/cuda/lib64"
OFFLINE = "HF_HUB_OFFLINE=1 HF_DATASETS_OFFLINE=1 TRANSFORMERS_OFFLINE=1 PYTHONDONTWRITEBYTECODE=1"
DROP = f"setpriv --reuid={UID} --regid={UID} --clear-groups --no-new-privs"
CANDIDATE_STATE = "/tmp /var/tmp /dev/shm /workspace /replay/home /replay/tmp /replay/work"
STARTUP_SECONDS = 300  # serve.sh's /health deadline, at launch and after each restart


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as stream:
        while chunk := stream.read(1 << 20):
            digest.update(chunk)
    return digest.hexdigest()


def check_corpus(corpus, expected):
    """Trusted-side check that the corpus directory holds exactly the pinned files."""
    corpus = Path(corpus)
    if not isinstance(expected, dict) or not expected:
        raise VerificationFailure("Trusted config must pin corpus_sha256")
    present = sorted(p.name for p in corpus.iterdir()) if corpus.is_dir() else []
    if present != sorted(expected):
        raise VerificationFailure("Replay corpus files differ from the pinned manifest")
    for name, digest in expected.items():
        path = corpus / name
        if path.is_symlink() or not path.is_file() or file_sha256(path) != digest:
            raise VerificationFailure(f"Replay corpus file {name} does not match its pinned hash")


class LocalDockerResource:
    """Candidate sandbox on a local Docker daemon: no network, one GPU, 8 CPUs, 80 GiB RAM."""

    def __init__(self, image, *, gpu="0", name=None, max_seconds=10800):
        self.image, self.gpu, self.max_seconds = image, str(gpu), max_seconds
        self.name = name or "fml-replay-" + uuid.uuid4().hex[:12]
        self.settings = {"type": "local-docker", "image": image}
        self.expired = self.stopped = False
        self.deadline = None

    def __enter__(self):
        subprocess.run(["docker", "run", "--detach", "--name", self.name, "--network", "none",
                        "--gpus", f"device={self.gpu}", "--cpus", "8", "--memory", "80g", "--shm-size", "16g",
                        "--pids-limit", "4096", "--pull", "never", "--entrypoint", "sleep", self.image, "infinity"],
                       check=True, capture_output=True, text=True, timeout=120)
        self.deadline = time.monotonic() + self.max_seconds
        return self

    def __exit__(self, *exc):
        subprocess.run(["docker", "rm", "--force", self.name], capture_output=True, timeout=120)
        self.stopped = True

    def _remaining(self, timeout):
        remaining = self.deadline - time.monotonic()
        if remaining <= 0 or self.stopped:
            self.expired = True
            raise TimeoutError("Replay resource lifetime expired")
        return max(1, min(timeout, remaining))

    def exec(self, command, *, timeout=60, env=None, check=True):
        result = subprocess.run(["docker", "exec", "-u", "0", self.name, "bash", "-c", command],
                                capture_output=True, text=True, timeout=self._remaining(timeout))

        class Result:
            return_code, stdout, stderr = result.returncode, result.stdout, result.stderr
        if check and result.returncode:
            raise RuntimeError(f"Sandbox command failed ({result.returncode}): {result.stderr[-1000:]}")
        return Result

    def upload(self, source, target):
        subprocess.run(["docker", "cp", str(source), f"{self.name}:{target}"], check=True,
                       capture_output=True, timeout=self._remaining(600))

    def upload_dir(self, source, target):
        self.exec(f"mkdir -p {shlex.quote(target)}")
        subprocess.run(["docker", "cp", f"{source}/.", f"{self.name}:{target}"], check=True,
                       capture_output=True, timeout=self._remaining(1800))

    def download(self, source, target):
        subprocess.run(["docker", "cp", f"{self.name}:{source}", str(target)], check=True,
                       capture_output=True, timeout=self._remaining(600))


def _kill_candidate(resource):
    # Up to 60 s: a killed CUDA process can sit in the driver's teardown for several seconds.
    resource.exec(f"pkill -KILL -u {UID} || true; for i in $(seq 600); do pgrep -u {UID} >/dev/null || exit 0; "
                  f"sleep 0.1; done; pgrep -u {UID} >/dev/null && exit 1 || exit 0", timeout=120)


def _wipe_candidate_state(resource):
    """Delete everything the candidate UID owns (its outputs are root-owned by now), then prove nothing is left.

    Files in a world-writable image directory outside CANDIDATE_STATE (a cache, a stray .pyc) are removed
    rather than failing the grade: the requirement is that no candidate state survives, not where it was.
    """
    resource.exec(f"for d in {CANDIDATE_STATE}; do [ -e $d ] && find $d -mindepth 1 -user {UID} -delete 2>/dev/null; done; "
                  f"rm -rf /replay/home /replay/tmp /replay/work; "
                  f"find / -xdev -user {UID} -not -path '/proc/*' -delete 2>/dev/null; "
                  f"! find / -xdev -user {UID} -not -path '/proc/*' -print -quit 2>/dev/null | grep -q .", timeout=300)


def _tail(resource, path, limit=4000):
    result = resource.exec(f"tail -c {limit} {path} 2>/dev/null || true", timeout=30, check=False)
    return result.stdout or ""


class ReplayServer:
    def __init__(self, resource, transport_factory, startup_seconds=STARTUP_SECONDS):
        self.resource = resource
        self.transport = transport_factory(resource)
        self.startup_seconds = startup_seconds
        self.launches = []

    def start(self):
        started = time.monotonic()
        self.resource.exec(
            f"cd /submission && nohup {DROP} env -i PATH={PATH} HOME=/workspace USER=nobody HOST=0.0.0.0 PORT=8000 "
            f"MODEL_NAME=submission WEIGHTS_DIR=/replay/output LD_LIBRARY_PATH={LIBS} {OFFLINE} "
            f"bash /submission/serve.sh </dev/null >>{CONTROL}/server.log 2>&1 &", timeout=30)
        deadline = started + self.startup_seconds
        last_error = "no probe completed"
        while time.monotonic() < deadline:
            if self.resource.expired or self.resource.stopped:
                raise VerificationFailure("Replay resource expired before readiness")
            try:
                status, body = self.transport.raw_request("GET", "/health", timeout=max(0.1, min(2, deadline - time.monotonic())))
                if status == 200 and time.monotonic() < deadline:
                    elapsed = time.monotonic() - started
                    self.launches.append(elapsed)
                    return elapsed
                last_error = f"HTTP {status} {body[:200]!r}"
            except EvaluationError:
                raise  # the verifier's own bridge failed: report that, do not wait out the deadline and blame the server
            except Exception as exc:  # noqa: BLE001 - not ready yet
                last_error = f"{type(exc).__name__}: {exc}"
            time.sleep(min(0.5, max(0, deadline - time.monotonic())))
        raise VerificationFailure(f"Server did not become healthy within {self.startup_seconds}s "
                                  f"(last /health probe: {last_error}); end of its log:\n"
                                  + _tail(self.resource, f"{CONTROL}/server.log", 1500))

    def restart(self):
        _kill_candidate(self.resource)
        _wipe_candidate_state(self.resource)
        return {"startup_seconds": self.start()}


def _prepare_training(resource, snapshot, corpus, config):
    """Steps 1-2 up to train.sh: the trusted checks, uploads and network guard. Returns (limit, GPU lines)."""
    limit = config.get("train_seconds", 3600)
    if type(limit) is not int or not 60 <= limit <= 14400:
        raise VerificationFailure("train_seconds must be an integer in 60..14400")
    check_corpus(corpus, config.get("corpus_sha256"))
    result = resource.exec("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader,nounits", timeout=60)
    devices = (result.stdout or "").strip().splitlines()
    if config.get("require_h100", True) and (len(devices) != 1 or "H100" not in devices[0]
                                            or int(devices[0].rsplit(",", 1)[1]) < 75000):
        raise VerificationFailure(f"Replay requires exactly one H100 with >= 75,000 MiB; got {devices!r}")
    resource.exec(f"mkdir -p /submission /corpus /replay/output /replay/home /replay/tmp /replay/work /workspace {CONTROL}/requests; "
                  f"chmod 700 {CONTROL}; chmod 777 /workspace", timeout=60)
    resource.upload_dir(Path(snapshot), "/submission")
    resource.upload_dir(Path(corpus), "/corpus")
    resource.upload(Path(__file__).with_name("http_bridge.py"), f"{CONTROL}/http_bridge.py")
    manifest = " ".join(shlex.quote(f"{digest}  /corpus/{name}") for name, digest in sorted(config["corpus_sha256"].items()))
    resource.exec(f"chown -R 0:0 /submission /corpus {CONTROL} && chmod -R a+rX,a-w /submission /corpus && "
                  f"chown {UID}:{UID} /replay/output /replay/home /replay/tmp /replay/work && "
                  "chmod 700 /replay/output /replay/home /replay/tmp /replay/work && "
                  f"test $(ls /corpus | wc -l) -eq {len(config['corpus_sha256'])} && "
                  f"printf '%s\\n' {manifest} | sha256sum --check --quiet --strict -", timeout=600)
    from .resources import guard_candidate_network
    guard_candidate_network(resource)  # train.sh is candidate code: relay and network guard first
    return limit, devices


def _start_training(resource, limit, budget=None):
    script = (f"s=$(date +%s.%N); echo $s > {CONTROL}/train.start; "
              f"cd /replay/work && timeout --kill-after=5 -s KILL {limit} {DROP} env -i PATH={PATH} HOME=/replay/home "
              f"TMPDIR=/replay/tmp USER=nobody CORPUS_DIR=/corpus OUTPUT_DIR=/replay/output REPLAY_START=${{s%.*}} "  # integer epoch seconds, as date +%s; train.start keeps the fraction
              f"TRAIN_BUDGET_SECONDS={limit if budget is None else budget} LD_LIBRARY_PATH={LIBS} {OFFLINE} "
              "bash /submission/train.sh "
              f"</dev/null >{CONTROL}/train.log 2>&1; c=$?; date +%s.%N > {CONTROL}/train.end; "
              f"echo $c > {CONTROL}/train.status.tmp; mv {CONTROL}/train.status.tmp {CONTROL}/train.status")
    resource.exec(f"nohup setsid bash -c {shlex.quote(script)} </dev/null >/dev/null 2>&1 &", timeout=30)


def _training_status(resource):
    out = resource.exec(f"cat {CONTROL}/train.status 2>/dev/null || true", timeout=30, check=False).stdout.strip()
    return int(out) if out else None


def _wait_for_training(resource, seconds, poll):
    """train.sh's exit status, or None if it has not reported one within `seconds`."""
    guard = time.monotonic() + seconds
    while time.monotonic() < guard:
        status = _training_status(resource)
        if status is not None:
            return status
        time.sleep(poll)
    return None


def run_training(resource, snapshot, corpus, config):
    """Steps 1-3. Returns a JSON-serializable training record; raises VerificationFailure on failure."""
    limit, _ = _prepare_training(resource, snapshot, corpus, config)
    _start_training(resource, limit)
    status = _wait_for_training(resource, limit + 300, config.get("train_poll_seconds", 15))
    return _finish_training(resource, status, limit)


def _finish_training(resource, status, limit):
    """Step 3 once train.sh has exited (status None: it never reported): the limit and exit checks, then the
    outputs made read-only, candidate state wiped and the corpus hidden."""
    times = resource.exec(f"cat {CONTROL}/train.start {CONTROL}/train.end 2>/dev/null || true", timeout=30, check=False).stdout.split()
    log_tail = _tail(resource, f"{CONTROL}/train.log")
    _kill_candidate(resource)
    record = {"exit_status": status, "limit_seconds": limit, "log_tail": log_tail}
    if len(times) == 2:
        record["train_seconds"] = float(times[1]) - float(times[0])
    if status is None:
        raise VerificationFailure("Training replay did not report completion")
    if status in (124, 137) or record.get("train_seconds", limit + 1) > limit:
        raise VerificationFailure(f"train.sh exceeded the {limit}s wall-clock limit")
    if status != 0:
        raise VerificationFailure(f"train.sh exited with status {status}: {log_tail[-800:]}")
    inventory = resource.exec("chown -R 0:0 /replay/output && chmod -R a+rX,a-w /replay/output && "
                              "find /replay/output -type f -printf '%s %P\\n' | sort -k2 | head -n 1000", timeout=600).stdout
    files = [line.split(" ", 1) for line in inventory.splitlines() if line]
    record["output_files"] = {name: int(size) for size, name in files}
    record["output_bytes"] = sum(record["output_files"].values())
    _wipe_candidate_state(resource)
    resource.exec("chmod 700 /corpus", timeout=30)
    return record


def probe_training(resource, snapshot, corpus, config, seconds, budget=None):
    """The self-check's steps 1-3: the same replay (image, user, environment, wall-clock limit), with
    TRAIN_BUDGET_SECONDS=`budget` when given (else the limit), watched for at most `seconds`. Returns run_training's
    record with `running` False when train.sh exited 0 within the window (the sandbox is then ready for
    serve_trained), or {"running": True, ...} when it was still running at the end of the window and was stopped
    there. Raises VerificationFailure as run_training does."""
    if type(seconds) is not int or seconds < 1:
        raise EvaluationError("The self-check's training window must be a positive integer number of seconds")
    try:
        limit, devices = _prepare_training(resource, snapshot, corpus, config)
    except EvaluationError:
        raise
    except VerificationFailure as error:  # before train.sh runs: the trusted config, corpus or GPU, not the submission
        raise EvaluationError(f"The training replay could not start: {error}") from error
    if budget is not None and (type(budget) is not int or not 1 <= budget <= limit):
        raise EvaluationError(f"The self-check's training budget must be an integer in 1..{limit}")
    name, _, memory = devices[0].rpartition(",") if len(devices) == 1 else ("", "", "")
    gpu = {"name": name.strip(), "memory_mib": int(memory)} if memory.strip().isdigit() else None
    _start_training(resource, limit, budget)
    started = time.monotonic()
    poll = config.get("train_poll_seconds", 15)
    status = _wait_for_training(resource, seconds if seconds < limit else limit + 300, poll)
    if status is None and seconds < limit:
        status = _training_status(resource)  # it may have exited during the last poll interval
        if status is None:
            elapsed = time.monotonic() - started
            log_tail = _tail(resource, f"{CONTROL}/train.log")
            # Best effort: nothing is served after this, and the caller discards the sandbox next.
            resource.exec(f"pkill -KILL -u {UID} || true", timeout=60, check=False)
            return {"running": True, "probe_seconds": elapsed, "limit_seconds": limit, "log_tail": log_tail, "gpu": gpu}
    try:
        record = _finish_training(resource, status, limit)
    except VerificationFailure as error:
        if status in (124, 137) and seconds < limit:  # long before the limit's timeout: the kernel or train.sh killed it
            raise VerificationFailure(f"train.sh was killed (exit status {status}; out of memory?) within the self-check's "
                                      f"{seconds}s window: {_tail(resource, f'{CONTROL}/train.log')[-800:]}") from error
        raise
    except ValueError as error:  # the output inventory parses file names train.sh chose; grading fails this run too
        raise VerificationFailure(f"train.sh's outputs could not be listed (a file name with a newline?): {error}") from error
    except (RuntimeError, TimeoutError) as error:
        # The kill, wipe and inventory are root execs over what train.sh left behind (processes, files): after it ran,
        # their failing or timing out is the submission's, or it could end every self-check in a refunded error.
        raise VerificationFailure(f"Stopping train.sh or listing its outputs failed: {error}") from error
    record.update(running=False, gpu=gpu)
    return record


def _session(resource, snapshot, server, startup, record):
    return {"transport": server.transport, "snapshot": Path(snapshot), "startup_elapsed": startup,
            "backend": "harbor:" + resource.settings["type"] if resource.settings["type"] != "local-docker" else "local-docker",
            "sandbox_id": resource.name, "replay": record, "restart": server.restart,
            "server_log_tail": lambda size=4000: _tail(resource, f"{CONTROL}/server.log", size)}


def replay_session(resource, snapshot, corpus, config, transport_factory):
    record = run_training(resource, snapshot, corpus, config)
    server = ReplayServer(resource, transport_factory)
    try:
        startup = server.start()
    except VerificationFailure as exc:
        # The training record is otherwise lost with the exception; it says whether the replay itself was sound.
        raise type(exc)(f"{exc}\n(before this launch: train.sh exited 0 in {record.get('train_seconds', float('nan')):.0f}s, "
                        f"{len(record['output_files'])} output files, {record['output_bytes']} bytes)") from exc
    return _session(resource, snapshot, server, startup, record)


@contextmanager
def serve_trained(resource, snapshot, config, *, record, transport_factory=None):
    """Step 4 alone, as a launcher(snapshot, config) for selfcheck.py once probe_training's train.sh exited 0 on
    `resource`: the session replay_session returns, plus the GPU. config's on_server_launch hears serve.sh start."""
    if transport_factory is None:
        from .resources import HarborHTTPTransport as transport_factory
    server = ReplayServer(resource, transport_factory)
    if callable(launched := config.get("on_server_launch")):
        launched()
    startup = server.start()
    yield {**_session(resource, snapshot, server, startup, record), "gpu": record.get("gpu"),
           "network_isolation": getattr(resource, "network_isolation", "provider"),
           "channel": getattr(getattr(resource, "relay", None), "kind", "per-exchange bridge")}


def keep_outputs(resource, target):
    """Operator pilots: copy the replay outputs and training log out of a local sandbox."""
    resource.exec(f"rm -rf /keep && cp -r /replay/output /keep && cp {CONTROL}/train.log /keep/replay-train.log && "
                  "chmod -R u+w /keep", timeout=600)
    resource.download("/keep", target)
    resource.exec("rm -rf /keep", timeout=120)


def serve_existing_outputs(resource, snapshot, outputs, transport_factory):
    """Operator re-scoring of weights kept from an earlier replay (NOT an official grade: no training)."""
    resource.exec(f"mkdir -p /submission /replay/output /workspace {CONTROL}/requests; chmod 700 {CONTROL}; chmod 777 /workspace", timeout=60)
    resource.upload_dir(Path(snapshot), "/submission")
    resource.upload_dir(Path(outputs), "/replay/output")
    resource.upload(Path(__file__).with_name("http_bridge.py"), f"{CONTROL}/http_bridge.py")
    resource.exec("rm -f /replay/output/replay-train.log; chown -R 0:0 /submission /replay/output && "
                  "chmod -R a+rX,a-w /submission /replay/output", timeout=600)
    server = ReplayServer(resource, transport_factory)
    startup = server.start()
    return {"transport": server.transport, "startup_elapsed": startup, "restart": server.restart}


@contextmanager
def replay_resource(config):
    """The candidate sandbox of one replay: Harbor's inference resource, or local Docker for operator pilots."""
    lifetime = config.get("verifier_lifetime_seconds", 10800)
    if type(lifetime) is not int or not 600 <= lifetime <= 86400:
        raise VerificationFailure("Replay lifetime must be 600..86400 seconds")
    local = config.get("replay_backend")
    if local is not None:
        if not isinstance(local, dict) or local.get("type") != "local-docker" or not local.get("image"):
            raise VerificationFailure("replay_backend supports only an explicit local-docker operator configuration")
        with LocalDockerResource(local["image"], gpu=local.get("gpu", "0"), max_seconds=lifetime) as resource:
            yield resource
        return
    from .resources import grading_resource
    with grading_resource(config, "inference", lifetime) as resource:
        yield resource


@contextmanager
def replay_submission(snapshot, corpus, config):
    """Trusted entry point for the grading library."""
    from .resources import HarborHTTPTransport
    with replay_resource(config) as resource:
        session = replay_session(resource, snapshot, corpus, config, HarborHTTPTransport)
        if isinstance(resource, LocalDockerResource):
            session["keep_outputs"] = lambda target: keep_outputs(resource, target)
        yield session
