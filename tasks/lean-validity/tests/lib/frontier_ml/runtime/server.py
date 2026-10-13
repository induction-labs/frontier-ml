"""Submission snapshots and fail-closed offline server supervision."""

import os
from pathlib import Path
import signal
import stat
import subprocess
import sys
import tempfile
import time
import uuid

from .transport import HTTPTransport


from ..validation.errors import VerificationFailure


def snapshot_submission(source, destination, *, require_serve_sh=True):
    """Copy the submission as regular files using no-follow descriptors; return a content digest.

    The destination must not exist and must be on verifier-controlled storage. Relative symlinks
    whose target stays inside the submission (a venv's `bin/python3 -> python3.12`, a toolchain's
    `libfoo.so -> libfoo.so.1`) become the regular files or directories they point to, exactly as the
    GPU sandbox materializes them before checking the same digest. Absolute, escaping, dangling or
    cyclic symlinks, sockets, devices and concurrent modifications fail closed. Hashes include
    relative paths and executable bits as well as every file's contents.
    """
    source, destination = Path(source), Path(destination)
    destination.mkdir(mode=0o755)

    def copy_tree(source_fd, target, relative):
        for name in sorted(os.listdir(source_fd)):
            rel = str(relative / name)
            output = target / name
            if stat.S_ISLNK(os.stat(name, dir_fd=source_fd, follow_symlinks=False).st_mode):
                os.symlink(os.readlink(name, dir_fd=source_fd), output)  # resolved below, inside the copy
                continue
            fd = os.open(name, os.O_RDONLY | os.O_NOFOLLOW | os.O_NONBLOCK, dir_fd=source_fd)
            try:
                before = os.fstat(fd)
                if stat.S_ISDIR(before.st_mode):
                    output.mkdir(mode=0o755)
                    copy_tree(fd, output, relative / name)
                elif stat.S_ISREG(before.st_mode):
                    executable = bool(before.st_mode & 0o111)
                    with output.open("xb") as out:
                        while chunk := os.read(fd, 1024 * 1024):
                            out.write(chunk)
                    after = os.fstat(fd)
                    if (before.st_size, before.st_mtime_ns, before.st_ctime_ns) != (
                        after.st_size, after.st_mtime_ns, after.st_ctime_ns
                    ):
                        raise VerificationFailure("Submission changed during snapshot")
                    output.chmod(0o755 if executable else 0o644)
                else:
                    raise VerificationFailure(f"Submission contains a special file: {rel}")
            finally:
                os.close(fd)

    try:
        root_fd = os.open(source, os.O_RDONLY | os.O_DIRECTORY | os.O_NOFOLLOW)
        try:
            copy_tree(root_fd, destination, Path())
        finally:
            os.close(root_fd)
        from frontier_ml.scoring.architecture import materialize_internal_symlinks, submission_digest
        try:
            materialize_internal_symlinks(destination)
        except ValueError as exc:
            raise VerificationFailure(str(exc)) from exc
        # Tasks that serve submitted weights with their own trusted server (taskspec.served_by_verifier) take none.
        if require_serve_sh and not (destination / "serve.sh").is_file():
            raise VerificationFailure("Submission must contain serve.sh")
        for path in sorted(destination.rglob("*"), reverse=True):
            path.chmod(0o555 if path.is_dir() or path.stat().st_mode & 0o111 else 0o444)
        destination.chmod(0o555)
        return submission_digest(destination)
    except OSError as exc:
        raise VerificationFailure(f"Unsafe or unreadable submission: {exc}") from exc


def check_hardware():
    try:
        result = subprocess.run(["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"],
                                check=True, capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError) as exc:
        raise VerificationFailure("Cannot establish verifier GPU hardware") from exc
    names = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if len(names) != 1 or "H100" not in names[0]:
        raise VerificationFailure(f"Expected exactly one H100, got {names!r}")


class OfflineServer:
    """Launch only inside a trusted, single-GPU verifier environment.

    Namespace backend needs root + CAP_SYS_ADMIN; Docker backend needs a daemon.
    Neither silently falls back to executing a submission on the host network.
    """

    def __init__(self, submission, *, backend="namespace", startup_seconds=300,
                 image=None, require_h100=True, log_path=None):
        if not 0 < startup_seconds <= 300:
            raise ValueError("Startup limit must be in (0, 300]")
        if backend not in ("namespace", "docker"):
            raise ValueError("Unknown offline backend")
        self.submission = Path(submission).absolute()
        self.backend = backend
        self.startup_seconds = startup_seconds
        self.image = image
        self.require_h100 = require_h100
        self.log_path = log_path
        self.process = None
        self.container = None
        self.directory = None
        self.log = None
        self.transport = None

    def __enter__(self):
        try:
            self.start()
            return self
        except BaseException:
            self.close()
            raise

    def start(self):
        if self.require_h100:
            check_hardware()
        self.directory = tempfile.TemporaryDirectory(prefix="frontier-verifier-")
        root = Path(self.directory.name)
        root.chmod(0o755)
        snapshot = root / "submission"
        self.submission_digest = snapshot_submission(self.submission, snapshot)
        self.snapshot = snapshot
        work = root / "work"
        work.mkdir(mode=0o700)
        self.log = open(self.log_path or root / "server.log", "wb")
        self.started_at = time.monotonic()
        deadline = self.started_at + self.startup_seconds
        if self.backend == "namespace":
            if os.environ.get("FRONTIER_MLE_VERIFIER") != "1":
                raise VerificationFailure("Namespace backend is restricted to a dedicated Harbor verifier")
            if os.geteuid() != 0:
                raise VerificationFailure("Namespace verifier requires root and CAP_SYS_ADMIN")
            os.chown(work, 65534, 65534)
            launch = Path(__file__).with_name("_launch.py")
            command = ["unshare", "--mount", "--net", "--pid", "--fork", "--kill-child=KILL",
                       "--mount-proc", sys.executable, "-I", str(launch), str(snapshot), str(work)]
            self.process = subprocess.Popen(command, stdout=self.log, stderr=subprocess.STDOUT,
                                            start_new_session=True)
            namespace_pid = self.process.pid
        else:
            if not self.image:
                raise VerificationFailure("Docker verifier requires a preloaded pinned image")
            self.container = "frontier-verifier-" + uuid.uuid4().hex
            command = ["docker", "run", "--detach", "--name", self.container,
                       "--network", "none", "--read-only", "--cap-drop", "ALL",
                       "--security-opt", "no-new-privileges", "--user", "65534:65534",
                       "--pids-limit", "1024", "--cpus", "8", "--memory", "80g",
                       "--tmpfs", "/tmp:rw,nosuid,nodev,size=16g", "--gpus", "device=0",
                       "--mount", f"type=bind,src={snapshot},dst=/submission,readonly",
                       "--workdir", "/submission", "--env", "HOST=0.0.0.0", "--env", "PORT=8000",
                       "--env", "MODEL_NAME=submission", "--env", "HOME=/tmp",
                       "--env", "HF_HUB_OFFLINE=1", "--env", "TRANSFORMERS_OFFLINE=1",
                       "--pull", "never", self.image, "bash", "/submission/serve.sh"]
            subprocess.run(command, check=True, stdout=self.log, stderr=subprocess.STDOUT,
                           timeout=max(0.01, deadline - time.monotonic()))
            result = subprocess.run(["docker", "inspect", "--format", "{{.State.Pid}}", self.container],
                                    capture_output=True, text=True, check=True,
                                    timeout=max(0.01, deadline - time.monotonic()))
            namespace_pid = int(result.stdout.strip())
            if namespace_pid <= 0:
                raise VerificationFailure("Submission container exited before readiness")
        self.transport = HTTPTransport(namespace_pid=namespace_pid)
        last_error = None
        while time.monotonic() < deadline:
            if self.process and self.process.poll() is not None:
                raise VerificationFailure("Offline namespace setup or submission startup failed; inspect server log")
            if self.process:
                try:
                    if os.readlink(f"/proc/{namespace_pid}/ns/net") == os.readlink("/proc/self/ns/net"):
                        time.sleep(min(0.01, max(0, deadline - time.monotonic())))
                        continue
                except FileNotFoundError:
                    raise VerificationFailure("Candidate namespace exited during startup")
            try:
                status, _ = self.transport.raw_request("GET", "/health",
                                                      timeout=min(0.5, deadline - time.monotonic()))
                if status == 200 and time.monotonic() <= deadline:
                    self.startup_elapsed = time.monotonic() - self.started_at
                    return
            except Exception as exc:
                last_error = exc
            time.sleep(min(0.05, max(0, deadline - time.monotonic())))
        raise VerificationFailure(f"Server did not become healthy within {self.startup_seconds}s: {last_error}")

    def close(self):
        if self.process and self.process.poll() is None:
            try:
                os.killpg(self.process.pid, signal.SIGKILL)
            except ProcessLookupError:
                pass
            self.process.wait(timeout=10)
        if self.container:
            subprocess.run(["docker", "rm", "--force", self.container], capture_output=True, timeout=30)
        if self.log:
            self.log.close()
        if self.directory:
            # Snapshots are read-only, restore owner permissions for cleanup.
            for path, dirs, _ in os.walk(self.directory.name):
                os.chmod(path, 0o700)
            self.directory.cleanup()

    def __exit__(self, *_):
        self.close()
