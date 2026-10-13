"""LiveCodeBench execution worker, watchdog, and the privileged remote supervisor.

The trusted verifier never launches this on its own host: the Harbor adapter transfers
this file into a fresh offline CPU sandbox containing only candidate code and testcase
inputs, where `supervise()` runs as root and drops privileges for the worker. Expected
outputs and scoring logic are never transferred.

The public development runner (`frontier_ml.validation.livecodebench.LocalExecutor`)
reuses the same `worker`, `exchange` and `run_inputs` on the developer's own machine,
without root, so both paths share one execution protocol.

AST wrapping/import prelude/input mocking adapt LiveCodeBench testing_util.py,
MIT Copyright (c) 2024 LiveCodeBench; see LICENSE-livecodebench.txt
(scoring/vendor/livecodebench.LICENSE in the operator repository).
"""

import ast
import ctypes
import io
import json
import os
import resource
import secrets
import selectors
import signal
import struct
import subprocess
import sys
import time
from contextlib import redirect_stdout
from types import ModuleType
from unittest.mock import mock_open, patch


IMPORTS = (
    "from string import *\nfrom re import *\nfrom datetime import *\nfrom collections import *\n"
    "from heapq import *\nfrom bisect import *\nfrom copy import *\nfrom math import *\n"
    "from random import *\nfrom statistics import *\nfrom itertools import *\nfrom functools import *\n"
    "from operator import *\nfrom io import *\nfrom sys import *\nfrom json import *\n"
    "from builtins import *\nfrom typing import *\n"
    "import string,re,datetime,collections,heapq,bisect,copy,math,random,statistics,itertools,functools,operator,io,sys,json\n"
    "sys.setrecursionlimit(50000)\n"
)


def stdio_code(code):
    # Match official clean_if_name + make_function transformations.
    try:
        tree = ast.parse(code)
        last = tree.body[-1]
        if isinstance(last, ast.If) and ast.unparse(last.test).strip() == "__name__ == '__main__'":
            code = ast.unparse(tree.body[:-1]) + "\n" + ast.unparse(last.body)
    except Exception:
        pass
    try:
        tree = ast.parse(code)
        imports = [v for v in tree.body if isinstance(v, (ast.Import, ast.ImportFrom))]
        rest = [v for v in tree.body if not isinstance(v, (ast.Import, ast.ImportFrom))]
        function = ast.FunctionDef(name="wrapped_function", args=ast.arguments(
            posonlyargs=[], args=[], kwonlyargs=[], kw_defaults=[], defaults=[]),
            body=rest, decorator_list=[], lineno=-1)
        return IMPORTS + "\n" + ast.unparse(imports) + "\n" + ast.unparse(function)
    except Exception:
        return code


def encode(value):
    # Preserve tuple/dict semantics across the JSON boundary. Only the outer
    # returned tuple is converted to a list, as in the upstream comparator.
    if value is None or type(value) in (bool, int, float, str):
        return ["scalar", value]
    if isinstance(value, (list, tuple)):
        return ["tuple" if isinstance(value, tuple) else "list", [encode(v) for v in value]]
    if isinstance(value, dict):
        return ["dict", [[encode(k), encode(v)] for k, v in value.items()]]
    raise TypeError("Unsupported return type")


class Capture(io.StringIO):
    def close(self):
        pass


class MockStdin:
    def __init__(self, text):
        self.inputs = text
        self._stream = io.StringIO(text)
        self.buffer = MockBuffer(text)
    def __getattr__(self, name):
        return getattr(self._stream, name)
    def read(self, *args):
        return self.inputs
    def readlines(self, *args):
        return self.inputs.split("\n")


class MockBuffer:
    def __init__(self, text):
        self.inputs = text.encode()
    def read(self, *args):
        return self.inputs
    def readline(self, *args):
        return self.inputs.split(b"\n")[0] + b"\n"


def worker():
    incoming, outgoing = sys.stdin, sys.stdout
    fn_name = json.loads(sys.argv[2])
    source = open(sys.argv[3] if len(sys.argv) > 3 else "/candidate.py").read()
    module = ModuleType("tmp_sol", "")
    with redirect_stdout(Capture()):
        exec(IMPORTS + "\n" + source if fn_name else stdio_code(source), module.__dict__)
        owner = module.Solution() if "class Solution" in source and fn_name else module
        method = getattr(owner, fn_name or "wrapped_function")
    outgoing.write('{"ready":true}\n')
    outgoing.flush()
    for line in incoming:
        request = json.loads(line)
        text = request["input"]
        captured = Capture()
        try:
            if fn_name:
                args = [json.loads(item) for item in text.split("\n")]
                with redirect_stdout(captured):
                    value = method(*args)
                if isinstance(value, tuple):
                    value = list(value)
                response = {"ok": True, "value": encode(value)}
            else:
                if isinstance(text, list):
                    text = "\n".join(text)
                stdin = MockStdin(text)
                lines = iter(text.split("\n"))
                with patch("builtins.open", mock_open(read_data=text)), patch("sys.stdin", stdin), \
                     patch("sys.stdin.readline", lambda *a: next(lines)), redirect_stdout(captured):
                    try:
                        method()
                    except SystemExit:
                        pass
                response = {"ok": True, "stdout": captured.getvalue()}
        except BaseException as exc:
            response = {"ok": False, "error": type(exc).__name__}
        outgoing.write(json.dumps(response, allow_nan=False) + "\n")
        outgoing.flush()


def drop_privileges(uid):
    os.setsid()
    resource.setrlimit(resource.RLIMIT_AS, (2 * 1024**3, 2 * 1024**3))
    resource.setrlimit(resource.RLIMIT_FSIZE, (64 * 1024**2, 64 * 1024**2))
    resource.setrlimit(resource.RLIMIT_NPROC, (64, 64))
    resource.setrlimit(resource.RLIMIT_NOFILE, (128, 128))
    os.setgroups([])
    os.setgid(uid)
    os.setuid(uid)
    if ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0):
        raise OSError("Cannot enforce no-new-privileges")


def limit_local():
    """Best-effort limits for the unprivileged local development runner (no root, no UID change)."""
    for name, value in (("RLIMIT_AS", 2 * 1024**3), ("RLIMIT_FSIZE", 64 * 1024**2), ("RLIMIT_NOFILE", 128),
                        ("RLIMIT_CORE", 0)):
        limit = getattr(resource, name, None)
        if limit is None:
            continue
        try:
            hard = resource.getrlimit(limit)[1]
            value = value if hard == resource.RLIM_INFINITY else min(value, hard)
            resource.setrlimit(limit, (value, value))
        except (OSError, ValueError):
            pass
    if sys.platform.startswith("linux"):
        try:
            ctypes.CDLL(None, use_errno=True).prctl(38, 1, 0, 0, 0)  # PR_SET_NO_NEW_PRIVS
        except Exception:
            pass


def exchange(process, request, seconds, *, max_bytes=64 * 1024**2):
    """Nonblocking pipe write/read with a parent-owned absolute deadline."""
    deadline = time.monotonic() + seconds
    outgoing = memoryview(b"" if request is None else (json.dumps(request) + "\n").encode())
    received = bytearray()
    with selectors.DefaultSelector() as poller:
        poller.register(process.stdout, selectors.EVENT_READ, "read")
        if outgoing:
            poller.register(process.stdin, selectors.EVENT_WRITE, "write")
        while time.monotonic() < deadline:
            for key, event in poller.select(max(0, deadline-time.monotonic())):
                if key.data == "write":
                    count = os.write(process.stdin.fileno(), outgoing[:65536])
                    outgoing = outgoing[count:]
                    if not outgoing:
                        poller.unregister(process.stdin)
                else:
                    block = os.read(process.stdout.fileno(), 65536)
                    if not block:
                        raise RuntimeError("Candidate worker exited")
                    received.extend(block)
                    if len(received) > max_bytes:
                        raise ValueError("Candidate response exceeded output limit")
                    if b"\n" in received:
                        line, extra = received.split(b"\n", 1)
                        if extra:
                            raise ValueError("Candidate emitted unsolicited protocol output")
                        return json.loads(line)
        raise TimeoutError("Candidate testcase exceeded time limit")


def run_inputs(process, inputs, timeout_seconds, *, max_total_bytes=256 * 1024**2):
    """Yield one JSON record line per testcase; stop after the first failing testcase.

    Compilation (readiness) and each testcase get independent watchdogs owned by the
    caller's process. Raises if the worker never becomes ready or output is excessive.
    """
    total = 0
    if exchange(process, None, timeout_seconds) != {"ready": True}:
        raise RuntimeError("Invalid candidate readiness")
    for index, text in enumerate(inputs):
        try:
            result = exchange(process, {"input": text}, timeout_seconds)
        except Exception as exc:
            result = {"ok": False, "error": type(exc).__name__}
        record = json.dumps({"index": index, "result": result}, allow_nan=False)
        total += len(record)
        if total > max_total_bytes:
            raise ValueError("Problem output exceeded total limit")
        yield record
        if not isinstance(result, dict) or result.get("ok") is not True:
            break


def supervise():
    if os.geteuid() != 0 or os.environ.get("FRONTIER_LCB_REMOTE") != "1":
        raise RuntimeError("Execution is restricted to the dedicated remote LCB image")
    config = json.load(open("/inputs.json"))
    # RLIMIT_NPROC counts real UIDs across containers on some backends. Sharing
    # nobody with unrelated jobs can make exec fail with EAGAIN before code runs.
    # A fresh unprivileged identity keeps the 64-process cap useful per problem.
    uid = 100000 + secrets.randbelow(2**31 - 100001)
    os.makedirs("/tmp/candidate", mode=0o700, exist_ok=True)
    os.chown("/tmp/candidate", uid, uid)
    process = subprocess.Popen(
        [sys.executable, "-I", "/runner.py", "worker", json.dumps(config["fn_name"])],
        stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL,
        cwd="/tmp/candidate", preexec_fn=lambda: drop_privileges(uid),
        env={"PATH": "/usr/local/bin:/usr/bin:/bin", "HOME": "/tmp/candidate",
             "PYTHONDONTWRITEBYTECODE": "1"},
    )
    os.set_blocking(process.stdin.fileno(), False)
    os.set_blocking(process.stdout.fileno(), False)
    try:
        for record in run_inputs(process, config["inputs"], config["timeout_seconds"]):
            print(record, flush=True)
    except Exception as exc:
        print(json.dumps({"error": type(exc).__name__}), flush=True)
    finally:
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        process.wait(timeout=10)


if __name__ == "__main__":
    mode = sys.argv[1] if len(sys.argv) > 1 else None
    if mode == "local-worker":
        limit_local()
        worker()
    elif mode == "worker":
        worker()
    else:
        supervise()
