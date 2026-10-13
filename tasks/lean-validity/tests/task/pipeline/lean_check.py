"""Organizer-side Lean labeling. Two independent compile paths for the double-compile rule.

* ``repl``: a persistent leanprover-community REPL per worker. The pinned header is
  elaborated once (env 0); each attempt body is elaborated against that environment.
* ``cli``: a fresh ``lean --json`` process on the complete file.

Both append ``#print axioms <theorem>`` (never part of the item text) and apply the
same pass rule: no error messages, no ``sorry`` warning, and axioms limited to
propext / Classical.choice / Quot.sound. Wall-clock expiry is reported as ``timeout``
(the item is dropped); heartbeat exhaustion is an ordinary deterministic error.

Environment: LEAN_VALIDITY_WORK (default /scratch/frontier-ml-workspace/work/lean-validity)
must contain elan/, mathlib4/ (tag v4.15.0 with cached oleans) and repl/ (tag v4.15.0).
"""
from __future__ import annotations

import json
import os
import re
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

WORK = Path(os.environ.get("LEAN_VALIDITY_WORK", "/scratch/frontier-ml-workspace/work/lean-validity"))
MATHLIB = WORK / "mathlib4"
REPL_BIN = WORK / "repl/.lake/build/bin/repl"
LEAN_BIN = WORK / "elan/toolchains/leanprover--lean4---v4.15.0/bin/lean"
TOOLCHAIN = "v4.15.0"

HEADER = ("import Mathlib\nimport Aesop\n\nset_option maxHeartbeats 200000\n\n"
          "open BigOperators Real Nat Topology Rat\n\n")
ALLOWED_AXIOMS = {"propext", "Classical.choice", "Quot.sound"}
AXIOM_RE = re.compile(r"'(?P<name>[^']+)' depends on axioms: \[(?P<axioms>[^\]]*)\]")
NO_AXIOM_RE = re.compile(r"'(?P<name>[^']+)' does not depend on any axioms")


def lean_env() -> dict:
    env = dict(os.environ)
    env["ELAN_HOME"] = str(WORK / "elan")
    env["PATH"] = f"{WORK / 'elan/bin'}:{env.get('PATH', '')}"
    # LEAN_PATH for Mathlib and dependencies (what `lake env` would set).
    packages = MATHLIB / ".lake/packages"
    paths = [MATHLIB / ".lake/build/lib"] + [p / ".lake/build/lib" for p in sorted(packages.iterdir())]
    env["LEAN_PATH"] = ":".join(str(p) for p in paths)
    return env


def verdict(messages: list[dict], theorem: str) -> dict:
    """Apply the pass rule to Lean messages (dicts with severity and data)."""
    errors = [m for m in messages if m.get("severity") == "error"]
    sorry = any("declaration uses 'sorry'" in m.get("data", "") for m in messages)
    axioms = None
    for m in messages:
        data = m.get("data", "")
        match = AXIOM_RE.search(data)
        if match and match.group("name").split(".")[-1] == theorem.split(".")[-1]:
            axioms = [a.strip() for a in match.group("axioms").split(",") if a.strip()]
        elif NO_AXIOM_RE.search(data):
            axioms = []
    ok = not errors and not sorry and axioms is not None and set(axioms) <= ALLOWED_AXIOMS
    return {"valid": ok, "errors": [m.get("data", "") for m in errors][:8], "sorry": sorry, "axioms": axioms}


def split_header(text: str) -> str:
    if not text.startswith(HEADER):
        raise ValueError("Item text must start with the pinned header")
    return text[len(HEADER):]


class ReplWorker:
    def __init__(self, restart_every: int = 150, memory_limit_gb: int = 24):
        self.restart_every = restart_every
        self.memory_limit_gb = memory_limit_gb
        self.proc = None
        self.used = 0

    def _start(self):
        self.close()
        limit = self.memory_limit_gb * 1024 * 1024
        self.proc = subprocess.Popen(["bash", "-c", f"ulimit -v {limit}; exec {REPL_BIN}"], cwd=MATHLIB,
                                     stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
                                     env=lean_env(), text=True, bufsize=1)
        import queue
        import threading
        self.lines = queue.Queue()
        stream, sink = self.proc.stdout, self.lines

        def pump():
            for line in stream:
                sink.put(line)
            sink.put(None)
        threading.Thread(target=pump, daemon=True).start()
        response = self._send({"cmd": HEADER.rstrip()}, 120)
        if response.get("env") != 0:
            raise RuntimeError(f"REPL header failed: {response}")
        self.used = 0

    def _send(self, payload: dict, timeout: float) -> dict:
        self.proc.stdin.write(json.dumps(payload, ensure_ascii=False) + "\n\n")
        self.proc.stdin.flush()
        deadline = time.monotonic() + timeout
        lines = []
        while True:
            remaining = deadline - time.monotonic()
            if remaining <= 0:
                raise TimeoutError
            try:
                line = self.lines.get(timeout=remaining)
            except Exception:
                raise TimeoutError from None
            if line is None:
                raise RuntimeError("REPL exited")
            if line.strip() == "" and lines:
                return json.loads("".join(lines))
            if line.strip():
                lines.append(line)

    def check(self, text: str, theorem: str, timeout: float = 60) -> dict:
        body = split_header(text)
        if self.proc is None or self.proc.poll() is not None or self.used >= self.restart_every:
            try:
                self._start()
            except (TimeoutError, RuntimeError, ValueError) as error:
                self.close()
                return {"status": "crash", "detail": f"start: {type(error).__name__}: {error}"}
        self.used += 1
        started = time.monotonic()
        try:
            response = self._send({"cmd": body + f"\n\n#print axioms {theorem}\n", "env": 0}, timeout)
        except (TimeoutError, RuntimeError, ValueError) as error:
            self.close()
            return {"status": "timeout" if isinstance(error, TimeoutError) else "crash", "seconds": time.monotonic() - started}
        if "message" in response and "messages" not in response:  # REPL-level error
            return {"status": "crash", "detail": response.get("message"), "seconds": time.monotonic() - started}
        result = verdict(response.get("messages", []), theorem)
        result.update(status="ok", seconds=time.monotonic() - started)
        return result

    def close(self):
        if self.proc is not None:
            try:
                self.proc.kill()
                self.proc.wait(5)
            except Exception:
                pass
        self.proc = None


def check_cli(text: str, theorem: str, timeout: float = 90) -> dict:
    split_header(text)
    with tempfile.TemporaryDirectory(prefix="leanval-") as directory:
        path = Path(directory) / "Attempt.lean"
        path.write_text(text + f"\n\n#print axioms {theorem}\n")
        started = time.monotonic()
        try:
            proc = subprocess.run(["bash", "-c", f"ulimit -v {24 * 1024 * 1024}; exec {LEAN_BIN} --json {path}"],
                                  cwd=MATHLIB, env=lean_env(), capture_output=True, text=True, timeout=timeout)
        except subprocess.TimeoutExpired:
            return {"status": "timeout", "seconds": time.monotonic() - started}
        messages = []
        for line in proc.stdout.splitlines():
            line = line.strip()
            if line.startswith("{"):
                try:
                    messages.append(json.loads(line))
                except ValueError:
                    pass
        if proc.returncode not in (0, 1) or (proc.returncode == 1 and not any(m.get("severity") == "error" for m in messages)):
            return {"status": "crash", "detail": proc.stderr[-500:], "seconds": time.monotonic() - started}
        result = verdict(messages, theorem)
        result.update(status="ok", seconds=time.monotonic() - started)
        return result


def run_pool(jobs: list[dict], mode: str, workers: int, timeout: float, progress_every: int = 500) -> list[dict]:
    """jobs: [{key, lean, theorem}] -> results in the same order (each with key)."""
    import queue
    import threading

    results: list = [None] * len(jobs)
    pending: queue.Queue = queue.Queue()
    for index in range(len(jobs)):
        pending.put(index)
    done = [0]
    lock = threading.Lock()
    started = time.monotonic()

    def worker():
        repl = ReplWorker() if mode == "repl" else None
        try:
            while True:
                try:
                    index = pending.get_nowait()
                except queue.Empty:
                    return
                job = jobs[index]
                try:
                    if repl:
                        result = repl.check(job["lean"], job["theorem"], timeout)
                    else:
                        result = check_cli(job["lean"], job["theorem"], timeout)
                except Exception as error:  # keep the pool alive; item gets dropped
                    result = {"status": "crash", "detail": f"{type(error).__name__}: {error}"}
                result["key"] = job["key"]
                results[index] = result
                with lock:
                    done[0] += 1
                    if done[0] % progress_every == 0:
                        rate = done[0] / (time.monotonic() - started)
                        print(f"[{mode}] {done[0]}/{len(jobs)} ({rate:.1f}/s)", flush=True)
        finally:
            if repl:
                repl.close()

    with ThreadPoolExecutor(workers) as pool:
        for future in [pool.submit(worker) for _ in range(workers)]:
            future.result()
    return results


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("jobs", help="JSONL with key, lean, theorem")
    parser.add_argument("output")
    parser.add_argument("--mode", choices=["repl", "cli"], default="repl")
    parser.add_argument("--workers", type=int, default=64)
    parser.add_argument("--timeout", type=float, default=60)
    args = parser.parse_args()
    jobs = [json.loads(line) for line in open(args.jobs) if line.strip()]
    results = run_pool(jobs, args.mode, args.workers, args.timeout)
    with open(args.output, "w") as stream:
        for result in results:
            stream.write(json.dumps(result, ensure_ascii=False) + "\n")
