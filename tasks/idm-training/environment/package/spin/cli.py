"""The agent's single entrypoint to shared-budget compute and model APIs."""
from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
from pathlib import Path
import sys
import urllib.error
import urllib.parse
import urllib.request

from . import __version__
from .limits import DEFAULT_TTL, INLINE_MAX_BYTES, MAX_TTL
from .render import emit


class BrokerClient:
    def __init__(self, url=None, token=None):
        self.url = (url or os.environ.get("SPIN_BROKER_URL") or "https://api.spin.inductionlabs.com/v1").rstrip("/")
        self.token = token or os.environ.get("SPIN_BROKER_TOKEN")
        if not self.token:
            raise ValueError("Set SPIN_BROKER_TOKEN to the wallet agent token (from spin wallet create)")

    def request(self, method, path, body=None, *, key=None):
        headers = {"Authorization": "Bearer " + self.token, "Content-Type": "application/json"}
        if key:
            headers["Idempotency-Key"] = key
        # Accept either a versioned public base or the legacy broker origin.
        if self.url.endswith("/v1") and path.startswith("/v1/"):
            path = path[3:]
        request = urllib.request.Request(self.url + path, method=method, headers=headers,
            data=json.dumps(body).encode() if body is not None else None)
        with urllib.request.urlopen(request, timeout=660) as response:
            result = json.load(response)
        # A slow call behind a proxy can fail after the server already answered 200.
        if isinstance(result, dict) and set(result) == {"error"} and isinstance(result["error"], dict):
            raise ValueError(json.dumps(result))
        return result


def pack_context(directory):
    """A build context directory as a base64 .tar.gz, skipping .git and .dockerignore'd paths."""
    import fnmatch
    import io
    import tarfile
    if not directory.is_dir():
        raise ValueError(f"{directory} is not a directory")
    ignore = [".git", ".git/*"]
    if (directory / ".dockerignore").is_file():
        ignore += [line.strip().rstrip("/") for line in (directory / ".dockerignore").read_text().splitlines()
                   if line.strip() and not line.startswith("#")]
    buffer = io.BytesIO()
    with tarfile.open(fileobj=buffer, mode="w:gz") as archive:
        for path in sorted(directory.rglob("*")):
            relative = path.relative_to(directory).as_posix()
            if any(fnmatch.fnmatch(relative, p) or fnmatch.fnmatch(relative, p + "/*") for p in ignore):
                continue
            if path.is_file() and not path.is_symlink():
                archive.add(path, arcname=relative, recursive=False)
    if buffer.tell() > 10 * 1024 * 1024:
        raise ValueError("context exceeds 10 MiB compressed; publish the image and use \"image\" instead")
    return base64.b64encode(buffer.getvalue()).decode()


SERVE_EPILOG = """environment:
  SPIN_API_KEY                operator key (default: generated once, kept in DATA_DIR/operator_key)
  SPIN_PROVIDERS              comma list of providers to enable: modal, vastai
                              (default: every provider whose credentials are present)
  MODAL_TOKEN_ID/SECRET       Modal credentials (or ~/.modal.toml from `modal token new`)
  SPIN_MODAL_APP              Modal app that owns lease sandboxes (default: spin-local)
  VAST_API_KEY                Vast.ai API key
  SPIN_VAST_MAX_HOURLY_USD    most a Vast rental may cost per hour (default: 30)
  SPIN_VAST_IMAGE             default Vast image (default: vastai/pytorch:cuda-12.8.1-auto)
  OPENROUTER_API_KEY          enables the OpenRouter proxy
  SPIN_OPENROUTER_MODELS      comma list restricting the models agents may call (default: all text models)
  SPIN_R2_ACCOUNT_ID          Cloudflare account ID: with the three below, enables wallet storage
  SPIN_R2_BUCKET              R2 bucket for wallet storage (objects live under SPIN_R2_PREFIX/<wallet>/)
  SPIN_R2_ACCESS_KEY_ID       parent R2 API token's access key ID (never leaves the server)
  SPIN_R2_SECRET_ACCESS_KEY   parent R2 API token's secret access key
  SPIN_R2_PREFIX              key prefix inside the bucket (default: spin-transfers)
  SPIN_R2_ENDPOINT_URL        R2 S3 endpoint for a jurisdiction bucket (default from SPIN_R2_ACCOUNT_ID)

Agents reach the server at --public-url; with --tunnel, a free Cloudflare quick tunnel provides an
HTTPS URL (install cloudflared first). The operator pays providers directly; wallets bound what agents
may spend."""


def parser():
    result = argparse.ArgumentParser(prog="spin", description="Spin CLI: budgeted compute and model access for agents. Prices and caps are set by the trusted broker, never by the sandbox.",
        epilog="Run `spin docs` for the workflow and API examples, and `spin <command> --help` for details. No cloud credentials are needed in the sandbox.")
    result.add_argument("--version", action="version", version="spin " + __version__)
    # Commands with a table view print it on a terminal; --json (or a pipe) gets JSON.
    fmt = argparse.ArgumentParser(add_help=False)
    fmt.add_argument("--json", action="store_true", help="print JSON instead of a table")
    groups = result.add_subparsers(dest="group", required=True)
    from .wallet import add_parser
    add_parser(groups, fmt)
    groups.add_parser("docs", help="show the agent workflow and programmatic access")
    budget = groups.add_parser("budget", help="inspect money spent, reserved and available").add_subparsers(dest="action", required=True)
    budget.add_parser("status", parents=[fmt])
    compute = groups.add_parser("compute", help="create, access and terminate leased compute").add_subparsers(dest="action", required=True)
    compute.add_parser("offers", help="show every available GPU/CPU offer and its hourly price", parents=[fmt])
    compute.add_parser("list", help="list only this run's resources", parents=[fmt])
    create = compute.add_parser("create", help="prepay a fixed duration and provision compute")
    create.add_argument("--provider", required=True)
    create.add_argument("--spec", required=True, help='JSON object or @path: an offer\'s spec, optionally with "image": "registry/ref"')
    create.add_argument("--request-id", required=True, help="stable unique key; reuse only for retrying the identical request")
    create.add_argument("--dockerfile", help="build the image from this Dockerfile (where the offer has dockerfile)")
    create.add_argument("--context", help="build context directory for --dockerfile (up to 10 MiB compressed)")
    create.add_argument("--no-network", action="store_true", help="switch networking off (where the offer has network_off)")
    extend = compute.add_parser("extend", help="prepay additional time on an active resource whose offer has supports_extension")
    extend.add_argument("id", help="broker resource ID")
    extend.add_argument("--seconds", type=int, required=True, help="additional seconds, added to the current deadline")
    extend.add_argument("--request-id", required=True, help="stable unique key; reuse for identical retries")
    for action, help_text in (("ssh", "get per-rental SSH details where the provider issues them; otherwise use exec"),
                              ("stop", "stop and release a resource; prepaid time is not refunded"),
                              ("status", "show a resource's provider status and deadline")):
        command = compute.add_parser(action, help=help_text, parents=[fmt] if action == "status" else [])
        command.add_argument("id", help="broker resource ID from compute create/list")
    execute = compute.add_parser("exec", help="execute a command on owned compute")
    execute.add_argument("id")
    execute.add_argument("--timeout", type=int, default=300)
    execute.add_argument("command", help="shell command, quoted as one argument")
    download = compute.add_parser("download", help="download a small file (<=512 KiB); use spin s5cmd for larger files")
    download.add_argument("id")
    download.add_argument("remote_path")
    download.add_argument("local_path")
    upload = compute.add_parser("upload", help="upload a small file (<=512 KiB); use spin s5cmd for larger files")
    upload.add_argument("id")
    upload.add_argument("local_path")
    upload.add_argument("remote_path")
    groups.add_parser("s5cmd", help="copy/list files directly in wallet storage; see spin s5cmd --help", add_help=False)
    transfer = groups.add_parser("transfer", help="export temporary S3 credentials for a VM or another client").add_subparsers(dest="action", required=True)
    credentials = transfer.add_parser("credentials", help="get a wallet-scoped storage session",
        description="Export temporary S3 credentials. For normal copies, spin s5cmd handles this automatically.",
        epilog="Use --out session.json, then spin s5cmd --session session.json cp FILE s3://spin/. Keep session files private.")
    credentials.add_argument("--prefix", default="", help="relative folder inside your wallet (default: entire wallet)")
    credentials.add_argument("--read-only", action="store_true", help="allow reading/listing only")
    credentials.add_argument("--expires-seconds", type=int, default=DEFAULT_TTL, help=f"session lifetime: 1..{MAX_TTL} (default: {DEFAULT_TTL}, 8 hours); capped by wallet expiry")
    credentials.add_argument("--out", help="write a private session file instead of printing credentials")
    router = groups.add_parser("openrouter", help="OpenAI-compatible inference billed to this wallet").add_subparsers(dest="action", required=True)
    router.add_parser("models", help="list the models this installation allows", parents=[fmt])
    endpoints = router.add_parser("endpoints", help="list a model's endpoints: provider, quantization, prices and context",
                                  parents=[fmt])
    endpoints.add_argument("model", help="model ID from spin openrouter models, e.g. deepseek/deepseek-chat-v3.1")
    key = router.add_parser("key").add_subparsers(dest="key_action", required=True)
    key.add_parser("create", help="print a scoped API key and compatible base_url")
    run = groups.add_parser("run", help="end this agent's run").add_subparsers(dest="action", required=True)
    run.add_parser("close", help="freeze spending, terminate compute and revoke wallet keys")
    serve = groups.add_parser("serve", help="run a self-hosted Spin server (Modal, Vast.ai, OpenRouter, R2)",
        description="Run a single-process Spin server. Wallets, leases and model calls are kept in DATA_DIR/spin.db.",
        epilog=SERVE_EPILOG, formatter_class=argparse.RawDescriptionHelpFormatter)
    serve.add_argument("--host", default="127.0.0.1", help="address to listen on (default: 127.0.0.1)")
    serve.add_argument("--port", type=int, default=8080, help="port to listen on (default: 8080)")
    serve.add_argument("--data-dir", default="spin-data", help="ledger and secrets directory (default: ./spin-data)")
    serve.add_argument("--public-url", help="URL agent sandboxes use to reach this server (default: http://HOST:PORT)")
    serve.add_argument("--tunnel", action="store_true", help="expose the server through a Cloudflare quick tunnel and use its HTTPS URL")
    serve.add_argument("--prices", help="static price table (default: the packaged prices.json)")
    return result


DOCS = """Spin provides budgeted compute and model access for your agents.

On the host, set SPIN_API_KEY (and SPIN_BASE_URL for a self-hosted server, e.g.
http://127.0.0.1:8080/v1 for `spin serve`), then create a wallet:
  spin wallet create --name experiment --budget 1 --request-id experiment-1

Give the sandbox only SPIN_BROKER_URL and SPIN_BROKER_TOKEN from the wallet's env;
keep the account key outside it:
  spin budget status
  spin compute offers
  spin compute create --provider PROVIDER --spec @lease.json --request-id gpu-1
  spin compute status RESOURCE_ID
  spin compute exec RESOURCE_ID 'nvidia-smi'
  spin compute ssh RESOURCE_ID
  spin compute upload RESOURCE_ID ./script.py /workspace/script.py
  spin compute download RESOURCE_ID /workspace/result.json ./result.json
  spin compute extend RESOURCE_ID --seconds 600 --request-id gpu-1-extension-1
  spin compute stop RESOURCE_ID
  spin openrouter key create
  spin run close

The providers and machines available here are whatever spin compute offers lists.
Use an offer's spec (spin compute offers --json) and set "max_seconds"; offers
marked custom_image accept "image": "registry/ref", offers with dockerfile can build
from --dockerfile Dockerfile --context DIR, and offers with network_off run with
--no-network. Reuse a request ID only for the identical request.

Leases are prepaid: the full price of max_seconds is charged when the machine is
ready, and stopping early refunds nothing. Leases whose offer shows
supports_extension can be extended (the added time is charged upfront) within the
provider's limit, wallet expiry and remaining budget; the others cannot.
A wallet with static pricing (offers and budget show "pricing": "static") pays one
fixed price per GPU model whichever provider runs it; GPUs without a fixed price
are not offered. An offer's gpu_model names its GPU; H100s come as "H100 SXM",
"H100 PCIe" and "H100 NVL".

New leases first return "checking": poll compute status (or rerun create with the
same request ID) until the initial health check passes; each poll runs one check.
It confirms the machine answers commands and, on GPU leases, that every GPU works.
Credits stay reserved and access is blocked while checking. A machine that has not
passed within 20 minutes (or by the lease or wallet deadline), or that its provider
reports dead, is destroyed and its credits released; status then shows "rejected"
with the reason. Startup counts against the lease duration.
A 409 not_ready means the machine is still starting: poll status and retry;
resource_dead or resource_inactive means it will not come back: stop it and create
another. Creates can take minutes; never wrap spin commands in a short timeout, and
after a client-side timeout run spin compute list. Stop destroys the machine and its
disk, so save output first. Ambiguous operations keep their funds reserved until
reconciled; closing a wallet returns only credit not spent or pending.

SSH credentials are per-rental, never provider account keys. Exec runs up to 3600 s
(--timeout). Upload and download move at most 512 KiB per file, for scripts and
results; do not move large files through exec output or split them up.
Where the installation has storage configured, spin s5cmd copies large files
directly to the wallet's private folder, s3://spin/ (quote wildcards):
  spin s5cmd cp ./model.bin s3://spin/results/
  spin s5cmd cp 's3://spin/results/*' ./results/
To copy from a GPU without giving it your Spin token, export a session with
spin transfer credentials --prefix results --out session.json, upload that file to
the GPU, then run spin s5cmd --session session.json cp FILE s3://spin/ there.
Sessions last 8 hours by default, capped by wallet expiry. Closing the wallet
deletes its storage; download results first.

OpenRouter: spin openrouter key create prints a wallet-scoped key and base_url for
any OpenAI-compatible client. Every chat completion needs an Idempotency-Key header
(a new key per call); calls may stream (stream: true), and a streamed call's key
cannot be replayed. Model calls count at their real cost, static wallets included.

OpenRouter routes each call to one of the model's endpoints (providers, often with
different quantizations, speeds and context limits) under a price cap. Without a
provider object the cap is the model's listed price, which usually only its
cheapest endpoints meet. To choose, list them with spin openrouter endpoints MODEL
and send OpenRouter's provider object with the call, e.g.
  "provider": {"only": ["deepinfra/fp8", "siliconflow"], "sort": "throughput"}
Spin accepts only, order, ignore (provider slugs or tags), quantizations (e.g.
["bf16", "fp8"]), sort (price, throughput or latency), allow_fallbacks, and
max_price (prompt and completion in USD per million tokens, request and image in
USD). The cap is then the dearest endpoint your selection allows, the call holds
the worst case of every endpoint under that cap, and it is billed at its real cost.
"""


def main(argv=None):
    argv = sys.argv[1:] if argv is None else list(argv)
    if argv and argv[0] == "s5cmd":
        try:
            from .s5cmd import run
            return run(argv[1:], BrokerClient)
        except urllib.error.HTTPError as exc:
            print(exc.read().decode(), file=sys.stderr)
            raise SystemExit(1)
        except (KeyError, ValueError, OSError) as exc:
            print(str(exc), file=sys.stderr)
            raise SystemExit(1)
    args = parser().parse_args(argv)
    if args.group == "docs":
        print(DOCS)
        return
    if args.group == "serve":
        from .server.app import serve
        return serve(args)
    try:
        if args.group == "wallet":
            from .wallet import main as wallet_main
            return wallet_main(args)
        if args.group == "transfer":
            value = BrokerClient().request("POST", "/transfer/credentials", {
                "prefix": args.prefix, "permission": "read-only" if args.read_only else "read-write",
                "expires_seconds": args.expires_seconds})
            if args.out:
                from .s5cmd import write_session
                value = write_session(args.out, value)
            print(json.dumps(value, indent=2))
            return
        client = BrokerClient()
        if args.group == "compute" and args.action == "download":
            result = client.request("POST", "/compute/download", {"id": args.id, "path": args.remote_path})
            data = base64.b64decode(result["content_base64"])
            if hashlib.sha256(data).hexdigest() != result["sha256"]:
                raise ValueError("download SHA-256 mismatch")
            Path(args.local_path).write_bytes(data)
            print(json.dumps({"path": args.local_path, "size_bytes": len(data), "sha256": result["sha256"]}, indent=2))
            return
        body, key, method = None, None, "GET"
        if args.group == "budget":
            path = "/budget"
        elif args.group == "compute":
            path = "/compute/" + args.action
            if args.action == "create":
                raw = Path(args.spec[1:]).read_text() if args.spec.startswith("@") else args.spec
                spec = json.loads(raw)
                if args.dockerfile:
                    spec["dockerfile"] = Path(args.dockerfile).read_text()
                if args.context:
                    if not args.dockerfile:
                        raise ValueError("--context needs --dockerfile")
                    spec["context"] = pack_context(Path(args.context))
                if args.no_network:
                    spec["network"] = "none"
                body = {"provider": args.provider, "spec": spec}
                key, method = args.request_id, "POST"
            elif args.action not in {"offers", "list"}:
                method, body = "POST", {"id": args.id}
                if args.action == "extend":
                    body["additional_seconds"] = args.seconds
                    key = args.request_id
                elif args.action == "exec":
                    body.update(command=args.command, timeout=args.timeout)
                elif args.action == "upload":
                    local = Path(args.local_path)
                    if local.stat().st_size > INLINE_MAX_BYTES:
                        raise ValueError("file exceeds inline 512 KiB ceiling; use spin s5cmd for larger files")
                    body.update(path=args.remote_path, content_base64=base64.b64encode(local.read_bytes()).decode())
        elif args.group == "openrouter":
            if args.action == "models":
                path = "/v1/models"
            elif args.action == "endpoints":
                path = "/v1/models/" + urllib.parse.quote(args.model, safe="/:") + "/endpoints"
            else:
                method, path, body = "POST", "/openrouter/key", {}
        else:
            method, path, body = "POST", "/run/close", {}
        emit(args, client.request(method, path, body, key=key))
    except urllib.error.HTTPError as exc:
        print(exc.read().decode(), file=sys.stderr)
        raise SystemExit(1)
    except (KeyError, ValueError, OSError) as exc:
        print(str(exc), file=sys.stderr)
        raise SystemExit(1)


if __name__ == "__main__":
    raise SystemExit(main())
