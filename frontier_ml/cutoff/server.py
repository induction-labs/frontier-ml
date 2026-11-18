"""The cutoff proxy's HTTP service and command line.

URL layout, below an optional run token (`/t/<token>/...`, required once a secret is configured):

    /hf/...            huggingface.co           (HF_ENDPOINT)
    /gh/...            github.com               (git clone/fetch, /archive/, /raw/, /releases/download/)
    /gh-raw/...        raw.githubusercontent.com
    /gh-api/...        api.github.com           (releases only)
    /openrouter/v1/... OpenAI-compatible inference

    frontier-cutoff-proxy serve --cutoff 2026-10-01 --state-dir cutoff-proxy-state --port 8799
    frontier-cutoff-proxy env --url https://proxy.example --run my-run     # exports for an agent's shell
    frontier-cutoff-proxy allowlist --out allow.json                       # tasks' starting weights, for --hf-allow
    frontier-cutoff-proxy catalog --out openrouter-2026-10-01.json         # freeze today's OpenRouter listing
"""
from __future__ import annotations

import argparse
from contextlib import asynccontextmanager
import json
import os
from pathlib import Path
import shlex
import sys

from fastapi import FastAPI, Request
from fastapi.responses import JSONResponse
import httpx

from .core import AccessLog, Cache, Refused, Settings, agent_env, forward, mint, refusal, request_headers, verify
from .github import GitHub
from .hf import HuggingFace
from .openrouter import OpenRouter

SECRET_ENV = "FML_CUTOFF_PROXY_SECRET"
METHODS = ["GET", "HEAD", "POST", "PUT", "PATCH", "DELETE"]


def create_app(settings: Settings, *, client: httpx.AsyncClient | None = None) -> FastAPI:
    settings.state_dir.mkdir(parents=True, exist_ok=True)
    cache, log = Cache(settings.state_dir / "pins.sqlite"), AccessLog(settings.state_dir / "access.jsonl", settings.log_sink)
    # Long read timeout: inference responses and first-time mirror fetches are slow; nothing here follows redirects
    # unless a call asks to.
    http = client or httpx.AsyncClient(timeout=httpx.Timeout(900, connect=20), follow_redirects=False,
                                       limits=httpx.Limits(max_connections=512, max_keepalive_connections=64))
    hf, github, openrouter = (cls(settings, http, cache, log) for cls in (HuggingFace, GitHub, OpenRouter))

    @asynccontextmanager
    async def lifespan(app):
        yield
        if client is None:
            await http.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)
    app.state.settings, app.state.log = settings, log

    @app.get("/healthz")
    async def health():
        return {"cutoff": settings.cutoff_text, "services": ["hf", "gh", "gh-raw", "gh-api", "openrouter"]}

    @app.api_route("/{path:path}", methods=METHODS)
    async def route(request: Request):
        # The raw path keeps percent-encoding (a Hub revision such as refs%2Fpr%2F1 is one path segment).
        raw = request.scope.get("raw_path", request.url.path.encode()).decode("latin-1").split("?")[0]
        parts = raw.lstrip("/").split("/", 3)
        prefix, run = "", "anonymous"
        if parts[0] == "t" and len(parts) >= 3:
            token, parts = parts[1], "/".join(parts[2:]).split("/", 1)
            prefix = f"/t/{token}"
            if settings.secret:
                run = verify(settings.secret, token)
                if run is None:
                    return JSONResponse({"error": "invalid or expired cutoff proxy token"}, 401)
            else:
                run = token.split(".")[0]
        elif settings.secret:
            return JSONResponse({"error": "a cutoff proxy token is required: /t/<token>/..."}, 401)
        else:
            parts = raw.lstrip("/").split("/", 1)
        service, rest = parts[0], parts[1] if len(parts) > 1 else ""
        origin = settings.public_url or request.headers.get("x-forwarded-proto", request.url.scheme) + "://" + \
            request.headers.get("x-forwarded-host", request.headers.get("host", request.url.netloc))
        mount = f"{origin}{prefix}/{service}"
        try:
            if service == "hf":
                return await hf.handle(request, rest, run, mount)
            if service in ("gh", "gh-raw", "gh-api") and settings.github_forward:
                # The instance that holds the mirrors answers; it checks the token again and writes the log line.
                url = settings.github_forward.rstrip("/") + raw + (f"?{request.url.query}" if request.url.query else "")
                body = await request.body() if request.method in ("POST", "PUT", "PATCH") else None
                return await forward(http, request.method, url, content=body, headers=request_headers(request))
            if service in ("gh", "gh-raw", "gh-api"):
                host = {"gh": "github", "gh-raw": "raw", "gh-api": "api"}[service]
                return await github.handle(request, rest, run, mount, host=host)
            if service == "openrouter":
                return await openrouter.handle(request, rest, run, mount)
        except Refused as exc:
            return refusal(exc, openai=service == "openrouter")
        except httpx.HTTPStatusError as exc:
            # One of the proxy's own lookups was refused upstream. A rate limit is passed on as one, with the wait
            # the upstream asked for, so clients back off and retry instead of seeing a broken proxy.
            status = exc.response.status_code
            log.write(run, service, "error", rest[:300], error=f"upstream {status}", status=status)
            headers = {"Retry-After": exc.response.headers.get("retry-after", "30")} if status == 429 else {}
            return JSONResponse({"error": f"upstream answered {status}"}, 429 if status == 429 else 502, headers=headers)
        except httpx.HTTPError as exc:
            log.write(run, service, "error", rest[:300], error=f"{type(exc).__name__}: {exc}"[:300])
            return JSONResponse({"error": f"upstream request failed: {type(exc).__name__}"}, 502)
        return JSONResponse({"error": "unknown service; see /healthz"}, 404)

    return app


def settings_from(args) -> Settings:
    env = os.environ
    return Settings(
        cutoff=args.cutoff, state_dir=args.state_dir, secret=env.get(SECRET_ENV) or None,
        hf_token=env.get("FML_CUTOFF_HF_TOKEN") or env.get("HF_TOKEN") or None,
        github_token=env.get("FML_CUTOFF_GITHUB_TOKEN") or env.get("GITHUB_TOKEN") or None,
        github_api=None if args.no_github_api else "https://api.github.com",
        github_max_kb=int(args.github_max_gb * 1024 * 1024),
        hf_allow=json.loads(args.hf_allow.read_text()) if args.hf_allow else None,
        openrouter_upstream=args.openrouter_upstream,
        openrouter_catalog=str(args.openrouter_catalog) if args.openrouter_catalog else Settings.openrouter_catalog)


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(prog="frontier-cutoff-proxy", description=__doc__.split("\n\n")[0])
    commands = parser.add_subparsers(dest="command", required=True)
    p = commands.add_parser("serve", help="run the proxy")
    p.add_argument("--cutoff", required=True, help="date or ISO time (UTC); nothing newer is visible")
    p.add_argument("--state-dir", type=Path, required=True, help="mirrors, pin cache and access.jsonl")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8799)
    p.add_argument("--openrouter-upstream", default=Settings.openrouter_upstream,
                   help="OpenAI-compatible API that serves inference (default: Spin)")
    p.add_argument("--openrouter-catalog", help="saved OpenRouter /models response to use instead of the live catalog")
    p.add_argument("--no-github-api", action="store_true",
                   help="skip repository creation-date checks and releases (commit dates alone decide)")
    p.add_argument("--hf-allow", type=Path, help="JSON from `frontier-cutoff-proxy allowlist`: each task's starting weights, "
                                                 "served to that task's runs whatever their date")
    p.add_argument("--github-max-gb", type=float, default=8, help="largest repository to mirror")
    p.add_argument("--insecure-no-token", action="store_true", help=f"serve without {SECRET_ENV} (open relay; local use)")
    p = commands.add_parser("env", help="print shell exports that point an agent at the proxy")
    p.add_argument("--url", required=True, help="the proxy's public URL")
    p.add_argument("--run", help=f"run label for a token signed with {SECRET_ENV}; omit for a proxy without tokens")
    p.add_argument("--days", type=float, default=7, help="token lifetime")
    p = commands.add_parser("allowlist", help="collect the tasks' starting_artifacts for --hf-allow")
    p.add_argument("--tasks", type=Path, default=Path(__file__).resolve().parents[2] / "tasks")
    p.add_argument("--out", type=Path, required=True)
    p = commands.add_parser("catalog", help="save OpenRouter's current model listing for --openrouter-catalog")
    p.add_argument("--out", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "env":
        secret = os.environ.get(SECRET_ENV)
        if args.run and not secret:
            parser.error(f"--run needs {SECRET_ENV}")
        token = mint(secret, args.run, seconds=int(args.days * 86400)) if args.run else None
        for key, value in agent_env(args.url, token).items():
            print(f"export {key}={shlex.quote(value)}")
        return 0
    if args.command == "allowlist":
        allow = {}
        for task in sorted(path for path in args.tasks.iterdir() if path.is_dir()):
            if (task / "environment/setup.json").is_file():  # a rendered Harbor package
                artifacts = json.loads((task / "environment/setup.json").read_text()).get("artifacts") or []
            elif (task / "task.yaml").is_file():  # a task seed
                import yaml
                artifacts = yaml.safe_load((task / "task.yaml").read_text()).get("starting_artifacts") or []
            else:
                continue
            for artifact in artifacts:
                if artifact.get("optional"):
                    continue  # the task's own files in the benchmark's dataset, not an upstream model
                kind = artifact.get("repo_type", "model") + "s"
                allow.setdefault(task.name, {})[f"{kind}/{artifact['repo']}"] = artifact["revision"]
        args.out.write_text(json.dumps(allow, indent=1) + "\n")
        print(f"{sum(map(len, allow.values()))} repositories for {len(allow)} tasks -> {args.out}")
        return 0
    if args.command == "catalog":
        response = httpx.get(Settings.openrouter_catalog, timeout=60, follow_redirects=True)
        response.raise_for_status()
        args.out.write_text(json.dumps(response.json()))
        print(f"{len(response.json()['data'])} models -> {args.out}")
        return 0
    settings = settings_from(args)
    if not settings.secret and not args.insecure_no_token:
        print(f"Set {SECRET_ENV} (run tokens are signed with it), or pass --insecure-no-token for local use.",
              file=sys.stderr)
        return 2
    import uvicorn
    # Without a limit a restart waits on every open inference stream while the listener is already closed.
    uvicorn.run(create_app(settings), host=args.host, port=args.port, log_level="info", timeout_graceful_shutdown=5)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
