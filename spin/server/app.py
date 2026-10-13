"""spin serve: one process with the account and agent APIs, the reaper, and an optional Cloudflare tunnel."""
from __future__ import annotations

from decimal import Decimal
import hashlib
import hmac
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
import json
import logging
import os
from pathlib import Path
import queue
import re
import secrets
import shutil
import signal
import subprocess
import threading
import time
from urllib.parse import unquote, urlsplit
import uuid

from .. import __version__
from .broker import ApiError, Broker, cleanup_storage, error_response, market_offers, present
from .ledger import Ledger
from .openrouter import OpenRouterProxy, Stream
from .pricing import load_prices, parse_table
from .providers import load_providers
from .storage import R2TransferStore

MAX_BODY = 16 * 1024 * 1024
REAP_SECONDS = 10
KEEPALIVE_AFTER, KEEPALIVE_EVERY = 60, 15  # seconds; Cloudflare gives up on a response that has not started in 100 s
MODEL_PATHS = {"/v1/models", "/v1/chat/completions"}
AGENT_ROUTES = {("GET", "/budget"), ("GET", "/compute/offers"), ("GET", "/compute/list"), ("GET", "/v1/models"),
                *(("POST", "/compute/" + action) for action in ("create", "extend", "stop", "exec", "ssh", "status", "upload", "download")),
                ("POST", "/transfer/credentials"), ("POST", "/openrouter/key"), ("POST", "/v1/chat/completions"), ("POST", "/run/close")}
WALLET_ROUTE = re.compile(r"/v1/wallets(?:/([^/]+)(?:/(token|close|resources))?)?")
ENDPOINTS_ROUTE = re.compile(r"/v1/models/.+")
WALLET_FIELDS = {"name", "budget_cents", "expires_in_seconds", "static_pricing", "unlimited"}
DEFAULT_EXPIRY, MAX_EXPIRY = 86400, 30 * 86400


def model_path(path):
    """Routes for model calls and model information, which OpenRouter-scoped keys may use."""
    return path in MODEL_PATHS or ENDPOINTS_ROUTE.fullmatch(path) is not None


def agent_path(path):
    """The broker path of an agent route: /v1/budget and /budget are the same; model routes keep /v1."""
    return path[3:] if path.startswith("/v1/") and not model_path(path) else path


class Spin:
    """The installation: one operator account, its wallets, and every wallet's agent API."""

    def __init__(self, ledger, providers, *, secret, operator_key, prices, openrouter=None, storage=None,
                 public_url="http://127.0.0.1:8080"):
        self.ledger, self.providers, self.prices = ledger, providers, prices
        self.secret, self.operator_key = secret, operator_key
        self.openrouter, self.storage, self.public_url = openrouter, storage, public_url.rstrip("/")

    def agent_token(self, wallet_id):
        """Derived, so the operator can fetch it again; only its hash is stored."""
        mac = hmac.new(self.secret.encode(), ("wallet:" + wallet_id).encode(), hashlib.sha256).hexdigest()
        return "spin_agent_" + wallet_id + "_" + mac

    def broker(self, wallet):
        return Broker(self.ledger, self.providers, wallet, openrouter=self.openrouter, storage=self.storage,
                      public_url=self.public_url)

    def handle(self, method, path, authorization, body, key):
        """(HTTP status, JSON-able result or a Stream) for one request."""
        if method == "GET" and path == "/healthz":
            return 200, {"status": "ok"}
        token = authorization[7:] if authorization.startswith("Bearer ") else ""
        account = WALLET_ROUTE.fullmatch(path)
        if account or path in {"/v1/offers", "/v1/static-pricing"}:
            self._operator(token)
            return self._account(method, path, account, body, key)
        path = agent_path(path)
        if (method, path) not in AGENT_ROUTES and not (method == "GET" and ENDPOINTS_ROUTE.fullmatch(path)):
            raise ApiError(404, "not_found", "Unknown endpoint or method")
        wallet = self._wallet_for(token, method, path)
        if path == "/run/close":
            self.ledger.mark_closing(wallet["id"])
        result = self.broker(wallet).dispatch(method, path, body, key)
        if path == "/run/close":
            self._finish_close(wallet["id"], result)
        return 200, result

    def _operator(self, token):
        if hmac.compare_digest(token.encode(), self.operator_key.encode()):
            return
        if token.startswith("spin_agent_"):
            raise ApiError(403, "wrong_token_scope", "Use an account API key for this endpoint")
        raise ApiError(401, "unauthorized", "Invalid API key" if token else "Bearer token required")

    def _wallet_for(self, token, method, path):
        found = self.ledger.authenticate(token) if token else None
        if found is None:
            raise ApiError(401, "unauthorized", "Invalid wallet token")
        if found["scope"] == "openrouter" and not model_path(path):
            raise PermissionError("token is scoped to OpenRouter")
        if method == "POST" and path == "/run/close" and found["scope"] == "agent":
            # Closing again is allowed after the close revoked the token.
            if found["revoked"] and not found["frozen"]:
                raise ApiError(403, "wallet_closed", "Wallet is expired or closed")
            return found
        expired = found["expires_at"] is not None and found["expires_at"] <= time.time()
        if found["revoked"] or found["status"] != "active" or expired:
            raise ApiError(403, "wallet_closed", "Wallet is expired or closed")
        return found

    # Account API

    def _account(self, method, path, match, body, key):
        wallet_id, action = (unquote(match.group(1)) if match.group(1) else None, match.group(2)) if match else (None, None)
        if method == "GET" and path == "/v1/offers":
            offers, providers = market_offers(self.providers)
            return 200, present({"offers": offers, "providers": providers, "updated_at": time.time()})
        if method == "GET" and path == "/v1/static-pricing":
            return 200, self.prices["static"]
        if method == "POST" and wallet_id is None:
            return self.create_wallet(body, key)
        if method == "GET" and wallet_id is None:
            return 200, {"data": [self.wallet_view(wallet) for wallet in self.ledger.wallets()]}
        if method == "GET" and action is None and wallet_id:
            return 200, self.wallet_view(self.ledger.wallet(wallet_id))
        if method == "GET" and action == "token":
            return 200, self.wallet_token(wallet_id)
        if method == "POST" and action == "close":
            return self.close_wallet(wallet_id)
        if method == "GET" and action == "resources":
            self.ledger.wallet(wallet_id)
            return 200, present({"resources": [Broker._public_resource(r) for r in self.ledger.reservations(wallet_id)
                                               if r["kind"] == "compute"]})
        raise ApiError(404, "not_found", "Unknown endpoint or method")

    def wallet_view(self, wallet):
        """A wallet as the operator sees it: budget and spend in its own prices, cost_usd in real provider cost."""
        status = self.ledger.status(wallet["id"])
        shown = status.get("static") or status
        return present({"id": wallet["id"], "name": wallet["name"], "status": wallet["status"],
                        "budget_usd": shown["budget_usd"], "budget_cents": int(Decimal(shown["budget_usd"]) * 100),
                        "spent_usd": shown["spent_usd"], "reserved_usd": shown["reserved_usd"],
                        "available_usd": shown["available_usd"], "cost_usd": status["spent_usd"],
                        "static_pricing": wallet["static_pricing"] is not None,
                        "static_pricing_version": (wallet["static_pricing"] or {}).get("version"),
                        "created_at": wallet["created_at"], "expires_at": wallet["expires_at"], "closed_at": wallet["closed_at"]})

    def create_wallet(self, body, key):
        if key is None:
            key = "generated-" + uuid.uuid4().hex
        elif not 1 <= len(key) <= 128:
            raise ApiError(400, "idempotency_key_required", "Supply a stable Idempotency-Key (1-128 characters)")
        if set(body) - WALLET_FIELDS:
            raise ApiError(400, "invalid_request", "Unknown fields: " + ", ".join(sorted(set(body) - WALLET_FIELDS)))
        if body.get("unlimited"):
            raise ApiError(400, "invalid_wallet", "Unlimited wallets are not supported here; give budget_cents")
        name, budget = body.get("name"), body.get("budget_cents")
        expires = body.get("expires_in_seconds", DEFAULT_EXPIRY)
        if (not isinstance(name, str) or not 1 <= len(name) <= 80 or type(budget) is not int or not 1 <= budget <= 1_000_000
                or type(expires) is not int or not (expires == 0 or 60 <= expires <= MAX_EXPIRY)):
            raise ApiError(400, "invalid_wallet", "Name required (1-80 characters); budget_cents 1-1000000; "
                                                  "expires_in_seconds 60-2592000, or 0 for no expiry")
        pricing = body.get("static_pricing")
        try:
            table = None if pricing is None or pricing is False else self.prices["static"] if pricing is True else parse_table(pricing)
        except ValueError as exc:
            raise ApiError(400, "invalid_static_pricing", str(exc)) from None
        fingerprint = hashlib.sha256(json.dumps([name, budget, expires, pricing], sort_keys=True).encode()).hexdigest()
        wallet_id, now = "wal_" + secrets.token_hex(12), time.time()
        wallet, created = self.ledger.create_wallet(
            wallet_id, name=name, request_key=key, fingerprint=fingerprint, budget_usd=Decimal(budget) / 100,
            static_pricing=table, created_at=now, expires_at=None if expires == 0 else now + expires, token=self.agent_token(wallet_id))
        response = {"id": wallet["id"], "status": wallet["status"]}
        if wallet["status"] == "active":
            token = self.agent_token(wallet["id"])
            response.update(agent_token=token, env={"SPIN_BROKER_URL": self.public_url + "/v1", "SPIN_BROKER_TOKEN": token})
        return (201 if created else 200), response

    def wallet_token(self, wallet_id):
        wallet = self.ledger.wallet(wallet_id)
        token = self.agent_token(wallet_id)
        if wallet["status"] != "active" or (wallet["expires_at"] is not None and wallet["expires_at"] <= time.time()):
            raise ApiError(409, "wallet_closed", "Wallet is expired or closed")
        if not self.ledger.has_token(wallet_id, token):
            raise ApiError(409, "token_unavailable", "Wallet token is unavailable; create a new wallet")
        return {"agent_token": token}

    def close_wallet(self, wallet_id):
        """200 once every resource is confirmed stopped and nothing is held; 202 "closing" while cleanup continues."""
        wallet = self.ledger.wallet(wallet_id)
        if wallet["status"] != "closed":
            self.ledger.mark_closing(wallet_id)
            result = self.broker(wallet).close()
            if not self._finish_close(wallet_id, result):
                return 202, present({"id": wallet_id, "status": "closing", "reserved_usd": result["settlement"]["reserved_usd"],
                                     "message": "A resource is still being cleaned up; its reservation returns once the provider confirms"})
        status = self.ledger.status(wallet_id)
        static = status.pop("static", None)
        return 200, present({"id": wallet_id, "status": "closed", "cleanup_complete": True,
                             "budget": {**status, **static, "pricing": "static"} if static else status,
                             "settlement": {"spent_usd": status["spent_usd"], "reserved_usd": status["reserved_usd"]}})

    def _finish_close(self, wallet_id, result):
        if result.get("cleanup_complete"):
            self.ledger.mark_closed(wallet_id)
            return True
        return False

    # Maintenance

    def reap(self):
        """One reaper tick: every open wallet's leases and model calls, then due closes, then storage."""
        for wallet in self.ledger.open_wallets():
            try:
                self.broker(wallet).reap()
                expired = wallet["expires_at"] is not None and wallet["expires_at"] <= time.time()
                if wallet["status"] == "closing" or expired:
                    self.close_wallet(wallet["id"])
            except Exception as exc:
                logging.warning("Maintenance of wallet %s failed: %s", wallet["id"], type(exc).__name__)
        try:
            cleanup_storage(self.ledger, self.storage)
        except Exception as exc:
            logging.warning("Storage maintenance failed: %s", type(exc).__name__)


class Handler(BaseHTTPRequestHandler):
    protocol_version = "HTTP/1.1"
    server_version = "spin/" + __version__
    sys_version = ""
    timeout = 120  # Idle keep-alive connections and stalled socket operations.

    def log_message(self, *_):
        pass  # Never log authorization headers or request bodies.

    def do_GET(self):
        self._handle("GET")

    def do_POST(self):
        self._handle("POST")

    def _body(self):
        if self.headers.get("Transfer-Encoding"):
            self.close_connection = True
            raise ValueError("chunked request bodies are not supported; send Content-Length")
        try:
            length = int(self.headers.get("Content-Length") or 0)
        except ValueError:
            length = -1
        if not 0 <= length <= MAX_BODY:
            self.close_connection = True
            raise ValueError("Content-Length must be at most 16 MiB")
        raw = self.rfile.read(length) if length else b""
        if self.command == "GET" or not raw:
            return {}
        body = json.loads(raw)
        if not isinstance(body, dict):
            raise ValueError("JSON body must be an object")
        return body

    def _handle(self, method):
        path = urlsplit(self.path).path
        try:
            body = self._body()
        except Exception as exc:
            return self._send(*self._failure(exc, method, path))

        def call():
            try:
                return self.server.spin.handle(method, path, self.headers.get("Authorization", ""), body,
                                               self.headers.get("Idempotency-Key"))
            except Exception as exc:
                return self._failure(exc, method, path)
        if self.headers.get("Cf-Ray") and method == "POST" and body.get("stream") is not True:
            return self._keepalive(call)
        self._send(*call())

    @staticmethod
    def _failure(exc, method, path):
        status, code, message = error_response(exc, agent_path(path))
        if status in {500, 502}:
            logging.warning("%s %s failed: %s", method, path, type(exc).__name__)
        return status, {"error": {"code": code, "message": message}}

    def _send(self, status, result):
        try:
            if isinstance(result, Stream):
                self._stream(result)
            else:
                self._json(status, result)
        except OSError:
            self.close_connection = True  # The client went away.

    def _keepalive(self, call):
        """Cloudflare drops a request whose response has not started within 100 s. Behind it, a call still
        running after KEEPALIVE_AFTER commits to 200 and sends whitespace (valid before JSON) until its
        result or error envelope is ready; the spin CLI treats an error envelope as a failure at any status."""
        outcome, done = [], threading.Event()
        threading.Thread(target=lambda: (outcome.append(call()), done.set()), daemon=True).start()
        if done.wait(KEEPALIVE_AFTER):
            return self._send(*outcome[0])
        try:
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
            while not done.wait(KEEPALIVE_EVERY):
                self._chunk(b"\n")
            self._chunk(json.dumps(outcome[0][1]).encode())
            self.wfile.write(b"0\r\n\r\n")
        except OSError:
            self.close_connection = True  # The call still finishes; its outcome is in the ledger.

    def _chunk(self, data):
        self.wfile.write(b"%x\r\n%s\r\n" % (len(data), data))
        self.wfile.flush()

    def _json(self, status, result):
        data = json.dumps(result).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(data)

    def _stream(self, live):
        """Server-sent events in chunked encoding; the relay keeps draining upstream if the client leaves."""
        try:
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Cache-Control", "no-store")
            self.send_header("X-Accel-Buffering", "no")
            self.send_header("Transfer-Encoding", "chunked")
            self.end_headers()
        except OSError:
            live.relay(None)
            raise
        live.relay(self._chunk)
        self.wfile.write(b"0\r\n\r\n")


class Server(ThreadingHTTPServer):
    """A thread per connection: exec calls may block for an hour and creates for minutes."""
    daemon_threads = True
    request_queue_size = 128

    def __init__(self, spin, host, port):
        self.spin = spin
        super().__init__((host, port), Handler)


def _secret_file(path, make):
    """A secret kept in the data directory (mode 0600), made on first start."""
    if not path.exists():
        fd = os.open(path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        with os.fdopen(fd, "w") as handle:
            handle.write(make() + "\n")
    return path.read_text().strip()


def start_tunnel(port, *, timeout=60):
    """A Cloudflare quick tunnel to the local server: (process, https URL)."""
    executable = shutil.which("cloudflared")
    if not executable:
        raise SystemExit("--tunnel needs cloudflared; install it (https://developers.cloudflare.com/cloudflare-one/"
                         "connections/connect-networks/downloads/, e.g. `brew install cloudflared`) or pass --public-url")
    process = subprocess.Popen([executable, "tunnel", "--no-autoupdate", "--url", f"http://127.0.0.1:{port}"],
                               stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    found = queue.Queue()

    def read():
        for line in process.stdout:  # Keep draining so cloudflared never blocks on a full pipe.
            match = re.search(r"https://[a-z0-9-]+\.trycloudflare\.com", line)
            if match:
                found.put(match.group(0))
        found.put(None)
    threading.Thread(target=read, daemon=True).start()
    try:
        url = found.get(timeout=timeout)
    except queue.Empty:
        url = None
    if not url:
        process.kill()
        raise SystemExit("cloudflared did not report a tunnel URL; check its installation and network access")
    return process, url


def _reaper(spin, stop):
    while True:
        try:
            spin.reap()
        except Exception as exc:  # Held reservations still block spending; the next tick retries.
            logging.warning("Reaper tick failed: %s", type(exc).__name__)
        if stop.wait(REAP_SECONDS):
            return


def serve(args):
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)  # Request lines would add noise, never value.
    env = os.environ
    data = Path(args.data_dir).resolve()
    data.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(data, 0o700)
    secret = _secret_file(data / "server_secret", lambda: secrets.token_hex(32))
    operator_key = env.get("SPIN_API_KEY") or _secret_file(data / "operator_key", lambda: "spin_sk_" + secrets.token_urlsafe(32))
    try:
        prices = load_prices(args.prices)
        ssh_secret = hmac.new(secret.encode(), b"vast-ssh", hashlib.sha256).hexdigest()
        providers = load_providers(env, prices=prices, ssh_secret=ssh_secret)
        storage = R2TransferStore.from_env(env)
    except Exception as exc:
        raise SystemExit(f"spin serve: {type(exc).__name__}: {str(exc)[:300]}") from None
    ledger = Ledger(data / "spin.db")
    openrouter = None
    if env.get("OPENROUTER_API_KEY"):
        allowed = {m.strip() for m in env.get("SPIN_OPENROUTER_MODELS", "").split(",") if m.strip()} or None
        openrouter = OpenRouterProxy(ledger, env["OPENROUTER_API_KEY"], allowed=allowed)
    spin = Spin(ledger, providers, secret=secret, operator_key=operator_key, prices=prices, openrouter=openrouter, storage=storage)
    try:
        server = Server(spin, args.host, args.port)
    except OSError as exc:
        raise SystemExit(f"spin serve: cannot listen on {args.host}:{args.port}: {exc.strerror}") from None
    port = server.server_address[1]
    local = "127.0.0.1" if args.host in {"0.0.0.0", "::", ""} else args.host
    tunnel = None
    if args.tunnel:
        tunnel, spin.public_url = start_tunnel(port)
    else:
        spin.public_url = (args.public_url or f"http://{local}:{port}").rstrip("/")
    stop = threading.Event()
    for sig in (signal.SIGINT, signal.SIGTERM):
        signal.signal(sig, lambda *_: stop.set())
    threading.Thread(target=server.serve_forever, kwargs={"poll_interval": 0.5}, daemon=True).start()
    threading.Thread(target=_reaper, args=(spin, stop), daemon=True).start()
    print(f"Spin {__version__} listening on http://{local}:{port} (data in {data})")
    print(f"Agents reach it at {spin.public_url}")
    if urlsplit(spin.public_url).scheme != "https":
        print("  (plain HTTP: agents outside this network need --tunnel or --public-url behind TLS)")
    print("Providers: " + (", ".join(providers) or "none (set MODAL_TOKEN_ID/MODAL_TOKEN_SECRET or VAST_API_KEY)"))
    print("OpenRouter: " + ("on" if openrouter else "off (set OPENROUTER_API_KEY)"))
    print("Storage: " + (f"on (R2 bucket {storage.bucket})" if storage else "off (set SPIN_R2_*)"))
    print(f"Operator key: {operator_key}")
    print(f"  export SPIN_BASE_URL=http://{local}:{port}/v1")
    print(f"  export SPIN_API_KEY={operator_key}", flush=True)
    try:
        while not stop.wait(1):
            if tunnel and tunnel.poll() is not None:
                logging.error("cloudflared exited: agents can no longer reach %s; restart spin serve", spin.public_url)
                tunnel = None
    finally:
        print("Shutting down", flush=True)
        server.shutdown()
        server.server_close()
        if tunnel:
            tunnel.terminate()
            try:
                tunnel.wait(10)
            except subprocess.TimeoutExpired:
                tunnel.kill()
