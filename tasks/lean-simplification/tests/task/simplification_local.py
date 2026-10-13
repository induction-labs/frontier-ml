"""Trusted Lean checker on the CPU verifier; no external service or Docker daemon."""
import hashlib
import importlib.util
import json
import os
from pathlib import Path
from queue import Queue
import re
import shutil
import stat
import subprocess
import tempfile
import time

from frontier_ml.validation.errors import VerificationFailure
from simplification_client import LEAN_VERSION, MATHLIB_REVISION, PROTOCOL

spec = importlib.util.spec_from_file_location("frontier_lean_worker", Path(__file__).parent / "checker/worker.py")
worker = importlib.util.module_from_spec(spec)
spec.loader.exec_module(worker)


def private_memory(pid):
    """Writable private mappings, excluding Mathlib's read-only file mappings.

    Some runtimes omit RssAnon. VmData exists on both Linux and Modal; inspect
    live threads too because Lean's main thread may exit before its worker.
    """
    process = Path("/proc") / str(pid)
    for status in [process / "status", *sorted((process / "task").glob("*/status"))]:
        try:
            for line in status.read_text().splitlines():
                if line.startswith("VmData:"):
                    return int(line.split()[1]) * 1024
        except FileNotFoundError:
            continue
    return None


class LocalChecker:
    def __init__(self, directory="/opt/frontier-lean"):
        self.directory = Path(directory)
        self.root = self.directory / "rootfs"
        self.launcher = self.directory / "sandbox"
        if os.geteuid() != 0:
            raise VerificationFailure("Local Lean checker requires the trusted root verifier")
        try:
            self.health = json.loads((self.directory / "build.json").read_text())
            expected = {"protocol": PROTOCOL, "lean": LEAN_VERSION, "mathlib": MATHLIB_REVISION}
            if any(self.health.get(k) != v for k, v in expected.items()):
                raise ValueError("toolchain/protocol mismatch")
            if not re.fullmatch(r"[0-9a-f]{64}", self.health.get("fingerprint", "")):
                raise ValueError("missing build fingerprint")
            source = Path(__file__).parent / "checker/ProofCheck.lean"
            for path, key in [(source, "checker_source_sha256"), (self.launcher, "sandbox_sha256")]:
                if hashlib.sha256(path.read_bytes()).hexdigest() != self.health[key]:
                    raise ValueError("checker build/source mismatch")
            if not (self.root / "opt/lean/bin/lean").is_file():
                raise ValueError("missing Lean compiler")
        except (OSError, ValueError, KeyError, TypeError) as exc:
            raise VerificationFailure(f"Local Lean checker image is unavailable or inconsistent: {exc}") from exc
        libraries = [self.root / "checker", self.root / "opt/mathlib/.lake/build/lib"]
        libraries.extend(sorted((self.root / "opt/mathlib/.lake/packages").glob("*/.lake/build/lib")))
        self.lean_path = ":".join("/" + str(p.relative_to(self.root)) for p in libraries)
        cpus = sorted(os.sched_getaffinity(0))
        # Lean reads its random seed through this device. It exposes entropy only;
        # no /dev mount, host devices, or real /proc filesystem enter the chroot.
        entropy = self.root / "dev/urandom"
        try:
            if not entropy.exists():
                os.mknod(entropy, stat.S_IFCHR | 0o444, os.makedev(1, 9))
            info = entropy.lstat()
            if not stat.S_ISCHR(info.st_mode) or info.st_rdev != os.makedev(1, 9):
                raise ValueError("unexpected entropy device")
        except (OSError, ValueError) as exc:
            raise VerificationFailure(f"Lean entropy device is unavailable: {exc}") from exc
        self.slots = Queue()
        for slot in range(8):
            self.slots.put((10000 + slot, cpus[slot % len(cpus)]))
        self.health = {k: self.health[k] for k in ("protocol", "lean", "mathlib", "fingerprint")}
        self.health.update(backend="local-isolated-process", concurrency=8, timeout_seconds=60,
                           private_memory_bytes=4 * 1024**3, cpus_per_proof=1)

    def _run(self, source=None, *, probe=None):
        uid, cpu = self.slots.get()
        directory = None
        process = None
        try:
            directory = Path(tempfile.mkdtemp(prefix="proof-", dir=self.root / "jobs"))
            if source is not None:
                path = directory / "Attempt.lean"
                path.write_text(source)
                path.chmod(0o444)
            os.chown(directory, uid, uid)
            directory.chmod(0o500)
            cwd = "/" + str(directory.relative_to(self.root))
            program = ["/checker/isolation-probe"] if probe else ["/opt/lean/bin/lean", "-j1", "-M4096", "--json", cwd + "/Attempt.lean"]
            if probe == "memory":
                program.append("--memory")
            command = [str(self.launcher), str(self.root), str(uid), str(cpu), cwd, self.lean_path, *program]
            # Log lives outside the chroot and is opened by the trusted parent.
            with tempfile.TemporaryFile() as output:
                process = subprocess.Popen(command, stdin=subprocess.DEVNULL, stdout=output, stderr=subprocess.STDOUT,
                                           env={"FRONTIER_CHECKER_SECRET_CANARY": "must-not-reach-lean"},
                                           start_new_session=True)
                deadline = time.monotonic() + (10 if probe else 60)
                unavailable_since = None
                while True:
                    remaining = deadline - time.monotonic()
                    if remaining <= 0:
                        process.kill()
                        process.wait()
                        return None, b"timeout"
                    try:
                        process.wait(timeout=min(0.05, remaining))
                        break
                    except subprocess.TimeoutExpired:
                        pass
                    memory = private_memory(process.pid)
                    if memory is None:
                        if process.poll() is not None:
                            break
                        # /proc can disappear during process/thread teardown.
                        # A still-running but unobservable checker fails closed.
                        now = time.monotonic()
                        unavailable_since = now if unavailable_since is None else unavailable_since
                        if now - unavailable_since > 1:
                            raise VerificationFailure("Cannot monitor Lean subprocess memory")
                    else:
                        unavailable_since = None
                        if memory > 4 * 1024**3:
                            process.kill()
                            process.wait()
                            return None, b"memory_limit"
                output.seek(0)
                raw = output.read(1_048_577)
            if process.returncode == 125:
                raise VerificationFailure("Lean subprocess isolation could not be established: " + raw[:1000].decode(errors="replace"))
            return process.returncode, raw
        finally:
            try:
                if process is not None:
                    if process.poll() is None:
                        process.kill()
                        process.wait()
                    procdir = self.root / "proc" / str(process.pid)
                    if procdir.exists():
                        shutil.rmtree(procdir)
                if directory is not None:
                    directory.chmod(0o700)
                    shutil.rmtree(directory)
            finally:
                self.slots.put((uid, cpu))

    def __call__(self, sample, proof):
        source, early = worker.source_for({"statement": sample["statement"], "proof": proof})
        if early:
            return early
        code, raw = self._run(source)
        if code is None:
            return {"valid": False, "status": raw.decode("ascii")}
        return worker.verdict(code, raw)

    def selftest(self):
        code, raw = self._run(probe="isolation")
        if code != 0 or raw.strip() != b"FRONTIER_SANDBOX_OK":
            raise VerificationFailure("Local Lean isolation self-test failed: " + raw[:1000].decode(errors="replace"))
        code, raw = self._run(probe="memory")
        if not ((code == 0 and raw.strip() == b"FRONTIER_MEMORY_OK") or (code is None and raw == b"memory_limit")):
            raise VerificationFailure("Local Lean memory enforcement self-test failed")
        positive = self({"statement": "(n : ℕ) : n = n"}, "by rfl")
        negative = self({"statement": ": False"}, "by exact True.intro")
        if not positive["valid"] or negative["valid"]:
            raise VerificationFailure(f"Local Lean kernel self-test failed: {positive}, {negative}")
