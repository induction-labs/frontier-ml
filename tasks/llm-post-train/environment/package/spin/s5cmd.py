"""Managed s5cmd with scoped R2 credentials and a wallet-relative S3 alias."""
from __future__ import annotations

import argparse
from datetime import datetime
import hashlib
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys
import tarfile
import tempfile
import time
import zipfile
from urllib.parse import urlsplit

import httpx

from .limits import DEFAULT_TTL, MAX_TTL

VERSION = "2.3.0"
# SHA-256 values from peak/s5cmd's v2.3.0 s5cmd_checksums.txt.
RELEASES = {
    ("Linux", "x86_64"): ("Linux-64bit.tar.gz", "de0fdbfa3aceae55e069ba81a0fc17b2026567637603734a387b2fca06c299b4"),
    ("Linux", "arm64"): ("Linux-arm64.tar.gz", "1439f0d00ecedcd2a2f1f2c6749bbb0152b2257bf5086f29646ec8ae38798e24"),
    ("Darwin", "x86_64"): ("macOS-64bit.tar.gz", "df6f76f6d317c4d051ad083219ed36eb8d8e5ebc55ceeeb91f5e66cdc3ac71fb"),
    ("Darwin", "arm64"): ("macOS-arm64.tar.gz", "ae007bc96276f498ae3c1fb017e57630cf93ef260cfe7e97b365522e240c973f"),
    ("Windows", "x86_64"): ("Windows-64bit.zip", "9d8a98c1d9f6284c6d54aeec3daeb5f1412fd101850ae756c1e70dd8ffe52166"),
    ("Windows", "arm64"): ("Windows-arm64.zip", "0d99ce0cbf4a266e7fd9b3cd2682384fb7b432754eb52f17f9a9f7cbe6bcca55"),
}


def folder(value, *, allow_empty=True):
    """Validate a relative folder without normalizing away traversal segments."""
    if value == "" and allow_empty:
        return ""
    if (not isinstance(value, str) or not value or value.startswith("/")
            or any(c in value for c in "\\*?[]") or any(ord(c) < 32 for c in value)):
        raise ValueError("prefix must be a relative folder without wildcards or traversal")
    value = value.removesuffix("/")
    if any(part in {"", ".", ".."} for part in value.split("/")):
        raise ValueError("prefix must be a relative folder without wildcards or traversal")
    return value


def r2_endpoint(value):
    """R2 JWT audience is the S3 API host, including a bucket's jurisdiction."""
    if not isinstance(value, str):
        raise ValueError("transfer.endpoint_url must be the R2 HTTPS S3 endpoint")
    url = urlsplit(value)
    match = re.fullmatch(r"([a-f0-9]{32})(?:\.(?:eu|us|fedramp))?\.r2\.cloudflarestorage\.com", url.netloc)
    if url.scheme != "https" or not match or url.path not in {"", "/"} or url.query or url.fragment:
        raise ValueError("transfer.endpoint_url must be the R2 HTTPS S3 endpoint without a bucket path")
    return value.rstrip("/"), match.group(1), url.netloc


def _download(url, destination):
    total = 0
    with httpx.stream("GET", url, follow_redirects=True, timeout=60) as response:
        response.raise_for_status()
        if response.url.scheme != "https":
            raise ValueError("s5cmd download must use HTTPS")
        with destination.open("wb") as out:
            for chunk in response.iter_bytes(1024 * 1024):
                total += len(chunk)
                if total > 32 * 1024 * 1024:
                    raise ValueError("s5cmd release archive exceeds size limit")
                out.write(chunk)


def binary():
    override = os.environ.get("SPIN_S5CMD_BINARY")
    if override:
        path = Path(override).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError("SPIN_S5CMD_BINARY must name an executable file")
        return str(path)
    system, arch = platform.system(), platform.machine().lower()
    arch = {"amd64": "x86_64", "aarch64": "arm64"}.get(arch, arch)
    if (system, arch) not in RELEASES:
        raise ValueError(f"No packaged s5cmd release for {system}/{arch}; set SPIN_S5CMD_BINARY")
    asset, expected = RELEASES[system, arch]
    filename = "s5cmd.exe" if system == "Windows" else "s5cmd"
    cache = Path(os.environ.get("XDG_CACHE_HOME") or Path.home() / ".cache")
    directory = cache / "spin" / "s5cmd" / VERSION / f"{system}-{arch}"
    target = directory / filename
    if target.is_file() and os.access(target, os.X_OK):
        return str(target)
    directory.mkdir(parents=True, exist_ok=True, mode=0o700)
    print(f"Installing s5cmd {VERSION}…", file=sys.stderr)
    with tempfile.TemporaryDirectory(dir=directory) as temp:
        archive = Path(temp) / asset
        url = f"https://github.com/peak/s5cmd/releases/download/v{VERSION}/s5cmd_{VERSION}_{asset}"
        try:
            _download(url, archive)
        except httpx.HTTPError:
            raise ValueError("Could not download s5cmd; retry or set SPIN_S5CMD_BINARY for an offline install") from None
        if hashlib.sha256(archive.read_bytes()).hexdigest() != expected:
            raise ValueError("s5cmd release checksum mismatch")
        # Read only the executable; never extract paths or links from the archive.
        if asset.endswith(".zip"):
            with zipfile.ZipFile(archive) as bundle:
                member = bundle.getinfo(filename)
                if member.file_size > 64 * 1024 * 1024:
                    raise ValueError("s5cmd executable exceeds size limit")
                data = bundle.read(member)
        else:
            with tarfile.open(archive, "r:gz") as bundle:
                member = bundle.getmember(filename)
                if not member.isfile() or member.size > 64 * 1024 * 1024:
                    raise ValueError("invalid s5cmd executable in release")
                with bundle.extractfile(member) as stream:
                    data = stream.read()
        staged = Path(temp) / filename
        staged.write_bytes(data)
        staged.chmod(0o700)
        os.replace(staged, target)
    return str(target)


def validate_session(value):
    """Validate server/file metadata without echoing credentials in failures."""
    if not isinstance(value, dict):
        raise ValueError("Invalid transfer session")
    required = ("endpoint_url", "bucket", "prefix", "s3_uri", "access_key_id", "secret_access_key", "session_token")
    if any(not isinstance(value.get(k), str) or not value[k] for k in required):
        raise ValueError("Invalid transfer session: missing connection fields")
    r2_endpoint(value["endpoint_url"])
    prefix = folder(value["prefix"], allow_empty=False) + "/"
    if value["prefix"] != prefix or value["s3_uri"] != f"s3://{value['bucket']}/{prefix}":
        raise ValueError("Invalid transfer session: inconsistent storage prefix")
    try:
        expiry = value["expires_at"]
        if isinstance(expiry, str):
            parsed = datetime.fromisoformat(expiry.replace("Z", "+00:00"))
            if parsed.tzinfo is None:
                raise ValueError()
            expiry = parsed.timestamp()
        if type(expiry) not in {int, float} or not expiry > time.time():
            raise ValueError()
    except (KeyError, TypeError, ValueError):
        raise ValueError("Transfer session expired or invalid; request fresh credentials") from None
    return value


def write_session(path, credentials):
    """Atomic, owner-readable credential handoff, without following a target link."""
    validate_session(credentials)
    target = Path(path).expanduser()
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = None
    try:
        with tempfile.NamedTemporaryFile(mode="w", dir=target.parent, delete=False) as out:
            temporary = Path(out.name)
            os.chmod(temporary, 0o600)
            json.dump(credentials, out)
            out.write("\n")
        os.replace(temporary, target)
    finally:
        if temporary:
            temporary.unlink(missing_ok=True)
    return {"path": str(target), "s3_uri": credentials["s3_uri"], "expires_at": credentials["expires_at"]}


def _arguments(args, credentials):
    root = credentials["s3_uri"]
    result = []
    for arg in args:
        if arg.split("=", 1)[0] in {"--endpoint-url", "--profile", "--credentials-file", "--no-sign-request", "--no-verify-ssl"}:
            raise ValueError("Spin manages the s5cmd endpoint and authentication; use --session for a saved session")
        if arg == "s3://spin" or arg.startswith("s3://spin/"):
            arg = root + arg.removeprefix("s3://spin").lstrip("/")
        if arg.startswith("s3://"):
            if arg == root.rstrip("/"):
                arg = root
            if not arg.startswith(root) or ".." in arg.split("/") or "\\" in arg:
                raise ValueError("S3 paths must stay inside this session's prefix; use s3://spin/")
        result.append(arg)
    return result


def _environment(credentials):
    env = {k: v for k, v in os.environ.items() if not k.startswith("AWS_")
           and k not in {"S3_ENDPOINT_URL", "SPIN_BROKER_TOKEN", "SPIN_API_KEY"}}
    env.update(AWS_ACCESS_KEY_ID=credentials["access_key_id"], AWS_SECRET_ACCESS_KEY=credentials["secret_access_key"],
               AWS_SESSION_TOKEN=credentials["session_token"], AWS_REGION="auto", AWS_DEFAULT_REGION="auto",
               AWS_EC2_METADATA_DISABLED="true", AWS_CONFIG_FILE=os.devnull, AWS_SHARED_CREDENTIALS_FILE=os.devnull)
    return env


HELP = """usage: spin s5cmd [Spin options] [s5cmd options] COMMAND [ARGS...]

Copy directly to/from your wallet's R2 storage. Credentials are fetched for each
invocation. s3://spin/ means the session's private prefix; quote wildcard paths.

  spin s5cmd cp ./model.bin s3://spin/checkpoints/
  spin s5cmd cp 's3://spin/checkpoints/*' ./checkpoints/
  spin s5cmd ls s3://spin/
  spin s5cmd --session session.json cp ./model.bin s3://spin/

Spin options (before native s5cmd options):
  --session FILE          Use an exported session; no Spin wallet token needed
  --prefix FOLDER         Limit a new session to this folder inside your wallet
  --read-only             Request read/list access only
  --expires-seconds N     Session lifetime: 1..{MAX_TTL}, default {DEFAULT_TTL} (8 hours)
  --install               Preinstall the pinned s5cmd binary; no credentials needed

spin s5cmd cp --help shows native copy options. The first use downloads and
verifies s5cmd; later uses run from cache. Transfers go directly to storage. Session
expiry is capped by wallet expiry. Download results before closing the wallet:
closure deletes its storage. Storage exists only where the installation configured it.
""".format(MAX_TTL=MAX_TTL, DEFAULT_TTL=DEFAULT_TTL)


def run(argv, client_factory):
    if not argv or argv[0] in {"-h", "--help"}:
        print(HELP, end="")
        return 0
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--session")
    parser.add_argument("--prefix", default="")
    parser.add_argument("--read-only", action="store_true")
    parser.add_argument("--expires-seconds", type=int, default=DEFAULT_TTL)
    parser.add_argument("--install", action="store_true")
    # Consume only leading Spin flags. All native flags/operands are untouched.
    split = 0
    with_value = {"--session", "--prefix", "--expires-seconds"}
    while split < len(argv):
        flag = argv[split].split("=", 1)[0]
        if flag not in with_value | {"--read-only", "--install"}:
            break
        split += 2 if flag in with_value and "=" not in argv[split] else 1
    opts = parser.parse_args(argv[:split])
    native = argv[split:]
    if opts.session and (opts.prefix or opts.read_only or opts.expires_seconds != DEFAULT_TTL):
        raise ValueError("--session already fixes the prefix, permission and expiry")
    if opts.install and native:
        raise ValueError("--install does not take an s5cmd command")
    if not opts.install and not native:
        raise ValueError("Choose an s5cmd command, e.g. spin s5cmd ls s3://spin/")
    executable = binary()
    if opts.install:
        print(executable)
        return 0
    if native in (["--version"], ["-v"], ["version"]):
        return subprocess.run([executable, "version"]).returncode
    if any(arg in {"--help", "-h"} for arg in native):
        return subprocess.run([executable, *native]).returncode
    if opts.session:
        credentials = json.loads(Path(opts.session).expanduser().read_text())
    else:
        credentials = client_factory().request("POST", "/transfer/credentials", {
            "prefix": opts.prefix, "permission": "read-only" if opts.read_only else "read-write",
            "expires_seconds": opts.expires_seconds})
    credentials = validate_session(credentials)
    args = _arguments(native, credentials)
    try:
        return subprocess.run([executable, "--endpoint-url", credentials["endpoint_url"], *args],
                              env=_environment(credentials)).returncode
    except KeyboardInterrupt:
        return 130
