"""Settings, run tokens, the pin cache, the access log and upstream forwarding shared by the cutoff proxy's services."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import json
from pathlib import Path
import re
import sqlite3
import threading
import time

from fastapi.responses import JSONResponse, StreamingResponse

from .tokens import agent_env, mint, verify  # noqa: F401 (the proxy and its tests import them from here)

# Hop-by-hop headers, and the ones the proxy sets itself for the upstream request.
DROP_REQUEST = {"host", "connection", "keep-alive", "proxy-authorization", "proxy-connection", "te", "trailer",
                "transfer-encoding", "upgrade", "content-length", "x-forwarded-for", "x-forwarded-proto",
                "x-forwarded-host", "forwarded"}
DROP_RESPONSE = {"connection", "keep-alive", "transfer-encoding", "upgrade", "trailer", "set-cookie", "alt-svc",
                 "strict-transport-security", "content-security-policy", "report-to", "nel"}


def parse_cutoff(value) -> datetime:
    """A date (end of that day is NOT implied: 2026-10-01 means 2026-10-01T00:00:00Z) or an ISO timestamp, as UTC."""
    if isinstance(value, datetime):
        result = value
    else:
        result = datetime.fromisoformat(str(value).strip().replace("Z", "+00:00"))
    return (result if result.tzinfo else result.replace(tzinfo=timezone.utc)).astimezone(timezone.utc)


def parse_time(value) -> datetime | None:
    """Upstream timestamps: ISO strings (Hugging Face, GitHub) or Unix seconds (OpenRouter)."""
    if value is None or value == "":
        return None
    if isinstance(value, (int, float)):
        return datetime.fromtimestamp(value, timezone.utc)
    try:
        return parse_cutoff(value)
    except ValueError:
        return None


@dataclass
class Settings:
    cutoff: datetime
    state_dir: Path
    # Signs run tokens. None accepts unauthenticated requests (local testing only: the proxy is an open relay then).
    secret: str | None = None
    hf_upstream: str = "https://huggingface.co"
    hf_token: str | None = None  # used when the client sends no token of its own, and for the proxy's own lookups
    github_upstream: str = "https://github.com"
    github_raw: str = "https://raw.githubusercontent.com"
    github_api: str | None = "https://api.github.com"  # None skips creation-date checks and disables releases
    github_token: str | None = None
    github_max_kb: int = 8 * 1024 * 1024  # refuse to mirror repositories GitHub reports as larger than this
    # How long a request waits for a first-time mirror before answering 503 "retry shortly" (the fetch carries on
    # in the background). Tunnels and load balancers in front of the proxy give up on a silent request at ~100 s.
    mirror_wait_seconds: float = 75
    # Repositories a task hands the agent as starting weights, which stay reachable whatever their date:
    # {task: {"<kind>/<repo>": commit}}, the task being the run label up to "__" (Harbor names trials <task>__<id>).
    # Only that commit is served, and only to that task's runs.
    hf_allow: dict | None = None
    openrouter_upstream: str = "https://api.spin.inductionlabs.com/v1"
    # Where model listing dates come from: a URL polled every `catalog_seconds`, or a saved copy of that response.
    openrouter_catalog: str = "https://openrouter.ai/api/v1/models"
    catalog_seconds: int = 900
    # Behind a load balancer (scripts/cutoff_proxy_modal.py): the URL clients reach the proxy at, when the Host
    # header cannot be trusted to say so; another instance that serves the GitHub mounts, so that the mirrors live
    # on one machine while the rest scales out; and where log lines go instead of this machine's access.jsonl.
    public_url: str | None = None
    github_forward: str | None = None
    log_sink: object = None

    def __post_init__(self):
        self.cutoff = parse_cutoff(self.cutoff)
        self.state_dir = Path(self.state_dir)
        for name in ("hf_upstream", "github_upstream", "github_api", "openrouter_upstream"):
            if getattr(self, name):
                setattr(self, name, getattr(self, name).rstrip("/"))

    def allowed_commit(self, run: str, kind: str, repo: str) -> str | None:
        return ((self.hf_allow or {}).get(run.split("__")[0]) or {}).get(f"{kind}/{repo}")

    @property
    def cutoff_text(self) -> str:
        return self.cutoff.strftime("%Y-%m-%dT%H:%M:%SZ")

    def before(self, value) -> bool:
        """Whether an upstream timestamp is at or before the cutoff; an unreadable or missing one is not."""
        when = parse_time(value)
        return when is not None and when <= self.cutoff

    @property
    def settled(self) -> bool:
        """Past cutoffs have fixed answers, which may be cached for good; a cutoff still ahead does not."""
        return self.cutoff < datetime.now(timezone.utc)


class Cache:
    """Answers that cannot change once the cutoff has passed (creation dates, pinned commits), kept across restarts."""

    def __init__(self, path):
        self.lock = threading.Lock()
        self.db = sqlite3.connect(str(path), check_same_thread=False)
        self.db.execute("CREATE TABLE IF NOT EXISTS pins (key TEXT PRIMARY KEY, value TEXT NOT NULL)")
        self.db.commit()

    def get(self, key):
        with self.lock:
            row = self.db.execute("SELECT value FROM pins WHERE key=?", (key,)).fetchone()
        return None if row is None else json.loads(row[0])

    def set(self, key, value):
        with self.lock, self.db:
            self.db.execute("INSERT OR REPLACE INTO pins(key,value) VALUES(?,?)", (key, json.dumps(value)))


class AccessLog:
    """One JSON line per decision: which run asked for what, what it was pinned to, and why anything was refused."""

    def __init__(self, path, sink=None):
        self.path, self.lock, self.sink = Path(path), threading.Lock(), sink

    def write(self, run, service, action, target, **fields):
        record = {"ts": round(time.time(), 3), "run": run, "service": service, "action": action, "target": target,
                  **{key: value for key, value in fields.items() if value is not None}}
        if self.sink is not None:
            return self.sink(json.dumps(record))
        with self.lock, self.path.open("a") as stream:
            stream.write(json.dumps(record) + "\n")


class Refused(Exception):
    """A request the proxy answers itself: the thing is newer than the cutoff, or outside what the proxy serves."""

    def __init__(self, status, message, *, code=None, reason=None):
        super().__init__(message)
        self.status, self.message, self.code, self.reason = status, message, code, reason or code or "refused"


def refusal(exc: Refused, *, openai: bool = False) -> JSONResponse:
    if openai:
        return JSONResponse({"error": {"message": exc.message, "code": exc.status, "type": exc.reason}}, exc.status)
    # huggingface_hub maps X-Error-Code to RepositoryNotFoundError / RevisionNotFoundError / EntryNotFoundError.
    headers = {"X-Error-Message": exc.message, **({"X-Error-Code": exc.code} if exc.code else {})}
    return JSONResponse({"error": exc.message}, exc.status, headers=headers)


def request_headers(request, **replace) -> dict[str, str]:
    """The client's headers for the upstream request, minus hop-by-hop ones; `replace` sets or (None) removes."""
    headers = {key: value for key, value in request.headers.items()
               if key.lower() not in DROP_REQUEST and not key.lower().startswith("modal-")}
    for key, value in replace.items():
        key = key.replace("_", "-")
        headers.pop(key, None)
        if value is not None:
            headers[key] = value
    return headers


async def forward(client, method, url, *, headers, content=None, rewrite=None, follow=False):
    """Stream an upstream response back unchanged (compressed bytes included). `rewrite(name, value)` may edit
    Location and Link, the two headers that carry upstream URLs."""
    # httpx would otherwise ask for compression on its own, and the client would get bytes it never offered to decode.
    headers = {"accept-encoding": "identity", **{key.lower(): value for key, value in headers.items()}}
    upstream = await client.send(client.build_request(method, url, headers=headers, content=content),
                                 stream=True, follow_redirects=follow)
    headers = []
    for key, value in upstream.headers.multi_items():
        if key.lower() in DROP_RESPONSE or key.lower().startswith("modal-"):
            continue  # modal-*: set by Modal's ingress on a proxy-to-proxy hop, and refused by it on the way out
        if rewrite and key.lower() in ("location", "link"):
            value = rewrite(key.lower(), value)
        headers.append((key, value))

    async def body():
        try:
            async for chunk in upstream.aiter_raw():
                yield chunk
        finally:
            await upstream.aclose()

    response = StreamingResponse(body(), status_code=upstream.status_code)
    response.raw_headers = [(key.lower().encode("latin-1"), value.encode("latin-1")) for key, value in headers]
    return response
