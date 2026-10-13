"""Trusted relay inside a grading sandbox: runs the installed HTTP bridge for each exchange the verifier sends.

The verifier used to reach the candidate server through one provider round trip per step of every request (upload
the request file, exec the bridge, download the response, remove the files). The relay keeps one channel open
instead and executes the same bridge protocol locally, so the timing semantics do not change: the bridge's own clock
inside the sandbox still times every request, and the provider's latency stays outside it.

    python3 -I relay.py --http PORT|0 --token-file FILE [--port-file FILE] [--monitor-uid UID]
        Exchanges over HTTP at a port the provider exposes (Modal's encrypted tunnel).
        Every request must carry the X-FML-Relay-Token from FILE. With --monitor-uid, a thread watches the kernel
        socket tables for that UID's non-loopback connections and reports them at GET /violations.

Message: {"id": str, "payload": <bridge request>, "timeout": seconds}. Answer: {"id", "value": <bridge answer>,
"relay_seconds"} or {"id", "relay_error": str}. Messages may arrive deflate-compressed (Content-Encoding: deflate);
answers are compressed (deflate()) for clients that accept it. When /opt/frontier-grading/http_bridge.py is the stock
bridge (its sha256 is --stock-sha256), exchanges run in-process; any other installed bridge (pokemon's move client) runs
exactly as before, as `python -I http_bridge.py <request file>`. Stdlib only, Python 3.8+.
"""
import argparse
import hashlib
import hmac
import importlib.util
import json
import os
import socket
import struct
import sys
import threading
import time
import uuid
import zlib
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import subprocess

ROOT = "/opt/frontier-grading"
BRIDGE = ROOT + "/http_bridge.py"
REQUESTS = ROOT + "/requests"
MARGIN_SECONDS = 30  # beyond the exchange's own deadline, for interpreter start-up and file handling
# Bodies travel deflate-compressed: after the idle time a grading loop spends between requests, TCP restarts slow, and
# a pokemon batch (334 KiB, 65 KiB deflated) took 0.45 s instead of 0.27 s from us-east to eu-west.
COMPRESS_ABOVE = 1024
MAX_MESSAGE_BYTES = 1024 ** 3


def deflate(data):
    """`data` deflate-compressed when that pays, else None. JSON (pokemon observations, prompts) shrinks 4-6x for
    ~1-3 ms; base64-encoded media shrinks ~23% at ~30 ms per MB, so a 64 KiB sample decides first."""
    if len(data) <= COMPRESS_ABOVE:
        return None
    sample = data[:65536]
    if len(zlib.compress(sample, 1)) > 0.6 * len(sample):
        return None
    return zlib.compress(data, 6 if len(data) <= 4 * 1024 * 1024 else 1)


class Executor:
    def __init__(self, root=ROOT, python=None, stock_sha256=None):
        self.root, self.bridge, self.requests = root, root + "/http_bridge.py", root + "/requests"
        self.python = python or sys.executable
        self.stock_sha256 = stock_sha256
        self._module, self._stamp, self._lock = None, None, threading.Lock()

    def _stock_bridge(self):
        """The installed bridge as a module when its bytes are the stock bridge's (then it runs in-process), else
        None (run as its own process, exactly as before). Nothing of an unknown bridge runs inside the relay."""
        try:
            stat = os.stat(self.bridge)
        except FileNotFoundError:
            raise RuntimeError("No HTTP bridge is installed at " + self.bridge)
        stamp = (stat.st_mtime_ns, stat.st_size, stat.st_ino)
        with self._lock:
            if stamp != self._stamp:
                with open(self.bridge, "rb") as stream:
                    source = stream.read()
                module = None
                if self.stock_sha256 and hmac.compare_digest(hashlib.sha256(source).hexdigest(), self.stock_sha256):
                    spec = importlib.util.spec_from_file_location("fml_stock_bridge_%d" % stat.st_mtime_ns, self.bridge)
                    module = importlib.util.module_from_spec(spec)
                    exec(compile(source, self.bridge, "exec"), module.__dict__)  # the verified bytes, not a re-read
                self._module, self._stamp = module, stamp
            return self._module

    def run(self, payload, timeout):
        module = self._stock_bridge()
        if module is not None:
            return module.request_batch(payload) if "batch" in payload else module.request_one(payload)
        name = self.requests + "/" + uuid.uuid4().hex
        os.makedirs(self.requests, mode=0o700, exist_ok=True)
        with open(name + ".request", "w") as stream:
            json.dump(payload, stream)
        try:
            done = subprocess.run([self.python, "-I", self.bridge, name], capture_output=True, text=True,
                                  timeout=timeout + MARGIN_SECONDS)
            if done.returncode:
                raise RuntimeError("bridge exited %d: %s" % (done.returncode, done.stderr[-500:]))
            value = json.loads(done.stdout)
            if value.get("response_file") is True:
                with open(name + ".response") as stream:
                    value = json.load(stream)
            return value
        finally:
            for suffix in (".request", ".response"):
                try:
                    os.unlink(name + suffix)
                except FileNotFoundError:
                    pass

    def exchange(self, message):
        started = time.monotonic()
        answer = {"id": message.get("id")}
        try:
            answer["value"] = self.run(message["payload"], float(message.get("timeout") or 3600))
        except Exception as exc:  # the verifier treats this as its own fault, never the candidate's
            answer["relay_error"] = "%s: %s" % (type(exc).__name__, str(exc)[:500])
        answer["relay_seconds"] = time.monotonic() - started
        return answer


# --- connection monitor -----------------------------------------------------------------------------------------

def _address(text, family):
    host, port = text.split(":")
    raw = bytes.fromhex(host)
    if family == socket.AF_INET:
        return socket.inet_ntop(family, struct.pack("<I", struct.unpack(">I", raw)[0])), int(port, 16)
    words = [raw[i:i + 4][::-1] for i in range(0, 16, 4)]  # four little-endian 32-bit words
    return socket.inet_ntop(family, b"".join(words)), int(port, 16)


def local_addresses(proc="/proc/net"):
    """This sandbox's own interface addresses: frameworks dial their own container IP (NCCL, ZMQ), which is local."""
    found = set()
    try:
        last = None
        with open(os.path.join(proc, "fib_trie")) as stream:
            for line in stream:
                token = line.strip().lstrip("|-+` ").strip()
                if token and token[0].isdigit() and "/" not in token:
                    last = token
                elif "host LOCAL" in line and last:
                    found.add(last)
    except OSError:
        pass
    try:
        with open(os.path.join(proc, "if_inet6")) as stream:
            for line in stream:
                raw = bytes.fromhex(line.split()[0])
                found.add(socket.inet_ntop(socket.AF_INET6, raw))
    except (OSError, ValueError, IndexError):
        pass
    try:
        found.update(socket.gethostbyname_ex(socket.gethostname())[2])
    except OSError:
        pass
    return found


def _local(host, own=()):
    return (host.startswith("127.") or host in {"::1", "0.0.0.0", "::"} or host.startswith("::ffff:127.")
            or host == "::ffff:0.0.0.0" or host in own or host.removeprefix("::ffff:") in own)


def remote_sockets(uid, proc="/proc/net", own=None):
    """(protocol, remote host, remote port) of every socket of `uid` whose peer is not this machine."""
    own = local_addresses(proc) if own is None else own
    found = []
    for name, family in (("tcp", socket.AF_INET), ("tcp6", socket.AF_INET6), ("udp", socket.AF_INET), ("udp6", socket.AF_INET6)):
        try:
            with open(os.path.join(proc, name)) as stream:
                next(stream)
                for line in stream:
                    fields = line.split()
                    if len(fields) < 8 or int(fields[7]) != uid:
                        continue
                    host, port = _address(fields[2], family)
                    if port and not _local(host, own):
                        found.append((name, host, port))
        except (OSError, StopIteration, ValueError):
            continue
    return found


class Monitor(threading.Thread):
    def __init__(self, uid, interval=0.1, proc="/proc/net"):
        super().__init__(daemon=True)
        self.uid, self.interval, self.proc = uid, interval, proc
        self.violations, self.scans, self._seen = [], 0, set()

    def run(self):
        own, refreshed = set(), 0.0
        while True:
            if time.monotonic() - refreshed > 10:
                own, refreshed = local_addresses(self.proc), time.monotonic()
            for item in remote_sockets(self.uid, self.proc, own):
                if item not in self._seen:
                    self._seen.add(item)
                    self.violations.append({"protocol": item[0], "host": item[1], "port": item[2], "at": time.time()})
            self.scans += 1
            time.sleep(self.interval)


# --- HTTP mode --------------------------------------------------------------------------------------------------

# How long a delivered answer is kept for a client whose copy of the reply was lost: relay_client._send tries 4 times,
# each up to its 60 s client timeout plus a 1-4 s pause.
KEEP_ANSWER_SECONDS = 300


class Jobs:
    """Exchanges by id. Both calls are idempotent, because the client retries over a tunnel that can drop a request
    or a reply: a repeated submit of a known id runs nothing again, and a delivered answer is kept for
    KEEP_ANSWER_SECONDS, so a retried wait whose first reply was lost gets the same answer. "unknown exchange" now
    means the POST never arrived, and the client sends it again (relay_client.HTTPRelay.call)."""

    def __init__(self, executor, workers=64, keep_seconds=KEEP_ANSWER_SECONDS):
        self.executor, self.pool = executor, ThreadPoolExecutor(max_workers=workers)
        self.futures, self.lock = {}, threading.Lock()
        self.delivered, self.keep_seconds = {}, keep_seconds   # id -> (answer, monotonic time of first delivery)
        self.seen = set()   # every id ever submitted: one is never run twice, even after its answer is forgotten

    def submit(self, message):
        job = message.get("id") or uuid.uuid4().hex
        with self.lock:
            if job in self.seen:
                return job  # a retried or re-sent POST: the exchange is running or done, never run twice
            self.seen.add(job)
            self.futures[job] = self.pool.submit(self.executor.exchange, {**message, "id": job})
        return job

    def wait(self, job, seconds):
        with self.lock:
            self._forget_old()
            if job in self.delivered:
                return self.delivered[job][0]
            future = self.futures.get(job)
        if future is None:
            return {"id": job, "relay_error": "unknown exchange"}
        try:
            answer = future.result(timeout=max(0.0, seconds))
        except Exception:  # not finished yet (futures.TimeoutError)
            return {"id": job, "done": False}
        result = {**answer, "done": True}
        with self.lock:
            self.futures.pop(job, None)
            self.delivered.setdefault(job, (result, time.monotonic()))
        return result

    def _forget_old(self):
        cutoff = time.monotonic() - self.keep_seconds
        for job in [job for job, (_, at) in self.delivered.items() if at < cutoff]:
            del self.delivered[job]


def bind_server(handler, port, low=40000, high=59999, port_file=None):
    """A ThreadingHTTPServer on `port`, or (port 0) on a random free port in [low, high], written to `port_file`.
    The relay shares the candidate's network namespace and starts first, so it keeps clear of the ports servers
    commonly use (8000-8999 and friends): a submission that runs vLLM on 8001 behind a proxy on 8000 must work."""
    import random
    candidates = [port] if port else random.sample(range(low, high + 1), 50)
    for candidate in candidates:
        try:
            server = ThreadingHTTPServer(("0.0.0.0", candidate), handler)
        except OSError:
            continue
        server.daemon_threads = True
        if port_file:
            with open(port_file + ".tmp", "w") as stream:
                stream.write(str(candidate))
            os.replace(port_file + ".tmp", port_file)
        return server
    raise SystemExit("no free port for the relay")


def serve_http(executor, port, token, monitor=None, port_file=None):
    jobs = Jobs(executor)

    class Handler(BaseHTTPRequestHandler):
        protocol_version = "HTTP/1.1"
        # TCP_NODELAY: the headers and the body are two writes, and Nagle held the second one back for the peer's
        # delayed ACK. Same region, 360 KB: 85 -> 44 ms per exchange (blocking client), 45 -> 19 ms (asyncio client).
        disable_nagle_algorithm = True

        def log_message(self, *args):
            pass

        def _reply(self, status, body):
            data = json.dumps(body).encode()
            compressed = deflate(data) if "deflate" in self.headers.get("Accept-Encoding", "") else None
            if compressed is not None:
                data = compressed
            self.send_response(status)
            self.send_header("Content-Type", "application/json")
            if compressed is not None:
                self.send_header("Content-Encoding", "deflate")
            self.send_header("Content-Length", str(len(data)))
            self.end_headers()
            self.wfile.write(data)

        def _authorized(self):
            given = self.headers.get("X-FML-Relay-Token", "")
            if hmac.compare_digest(given.encode(), token.encode()):
                return True
            self._reply(403, {"error": "forbidden"})
            return False

        def _wait(self, query):
            for part in query.split("&"):
                if part.startswith("wait="):
                    return min(50.0, float(part[5:]))
            return 20.0

        def do_GET(self):
            if not self._authorized():
                return
            path, _, query = self.path.partition("?")
            if path == "/ready":
                return self._reply(200, {"ready": True, "python": executor.python, "monitor": monitor is not None})
            if path == "/violations":
                return self._reply(200, {"violations": monitor.violations if monitor else None,
                                         "scans": monitor.scans if monitor else 0})
            if path.startswith("/exchange/"):
                return self._reply(200, jobs.wait(path[len("/exchange/"):], self._wait(query)))
            self._reply(404, {"error": "not found"})

        def do_POST(self):
            if not self._authorized():
                return
            path, _, query = self.path.partition("?")
            if path != "/exchange":
                return self._reply(404, {"error": "not found"})
            length = int(self.headers.get("Content-Length") or 0)
            raw = self.rfile.read(length)
            if self.headers.get("Content-Encoding", "").strip() == "deflate":
                inflater = zlib.decompressobj()
                raw = inflater.decompress(raw, MAX_MESSAGE_BYTES)
                if inflater.unconsumed_tail:
                    return self._reply(413, {"error": "message too large"})
            message = json.loads(raw)
            job = jobs.submit(message)
            self._reply(200, jobs.wait(job, self._wait(query)))

    bind_server(Handler, port, port_file=port_file).serve_forever()


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--http", type=int, required=True)
    parser.add_argument("--token-file")
    parser.add_argument("--monitor-uid", type=int)
    parser.add_argument("--root", default=ROOT)
    parser.add_argument("--stock-sha256", help="sha256 of the stock http_bridge.py, which then runs in-process")
    parser.add_argument("--port-file", help="with --http 0: where to write the random port the relay bound")
    args = parser.parse_args(argv)
    executor = Executor(root=args.root, stock_sha256=args.stock_sha256)
    with open(args.token_file) as stream:
        token = stream.read().strip()
    if len(token) < 32:
        raise SystemExit("relay token too short")
    monitor = None
    if args.monitor_uid is not None:
        monitor = Monitor(args.monitor_uid)
        monitor.start()
    serve_http(executor, args.http, token, monitor, port_file=args.port_file)


if __name__ == "__main__":
    main()
