"""Provider-neutral streaming transfers and safe submission archive extraction."""
from __future__ import annotations

import fnmatch
import hashlib
import os
from pathlib import Path, PurePosixPath
import re
import tarfile
import tempfile
import time
from urllib.parse import quote, urlsplit

import httpx


def _https(url: str) -> None:
    parsed = urlsplit(url)
    if parsed.scheme != "https" or not parsed.hostname or parsed.username or parsed.password:
        raise ValueError("transfer URLs must use HTTPS without embedded credentials")


def put_file(path: str | Path, url: str, *, client=None) -> dict:
    """PUT a file to a short-lived object-scoped signed URL; never log the URL."""
    _https(url)
    source = Path(path)
    if not source.is_file() or source.is_symlink():
        raise ValueError("upload source must be a regular file")
    own = client is None
    client = client or httpx.Client(timeout=120, follow_redirects=False)
    digest = hashlib.sha256()
    size = source.stat().st_size
    def chunks():
        with source.open("rb") as stream:
            while chunk := stream.read(8 * 1024 * 1024):
                digest.update(chunk)
                yield chunk
    try:
        response = client.put(url, content=chunks(), headers={"Content-Length": str(size)})
        if not 200 <= response.status_code < 300:
            raise RuntimeError(f"object upload failed: HTTP {response.status_code}")
        return {"bytes": size, "sha256": digest.hexdigest()}
    finally:
        if own:
            client.close()


def get_file(url: str, destination: str | Path, *, sha256: str, max_bytes: int, client=None) -> dict:
    """Download atomically and require an expected digest and size ceiling."""
    _https(url)
    if len(sha256) != 64 or any(x not in "0123456789abcdef" for x in sha256):
        raise ValueError("a lowercase SHA-256 digest is required")
    if max_bytes <= 0:
        raise ValueError("max_bytes must be positive")
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    own = client is None
    client = client or httpx.Client(timeout=120, follow_redirects=False)
    digest, size = hashlib.sha256(), 0
    tmp = None
    try:
        with tempfile.NamedTemporaryFile(dir=destination.parent, delete=False) as stream:
            tmp = Path(stream.name)
            with client.stream("GET", url) as response:
                if response.status_code != 200:
                    raise RuntimeError(f"object download failed: HTTP {response.status_code}")
                for chunk in response.iter_bytes(8 * 1024 * 1024):
                    size += len(chunk)
                    if size > max_bytes:
                        raise ValueError("download exceeds size ceiling")
                    digest.update(chunk)
                    stream.write(chunk)
        if digest.hexdigest() != sha256:
            raise ValueError("download SHA-256 mismatch")
        os.replace(tmp, destination)
        return {"bytes": size, "sha256": sha256}
    finally:
        if tmp is not None:
            tmp.unlink(missing_ok=True)
        if own:
            client.close()


def extract_submission(archive: str | Path, destination: str | Path, *, max_bytes: int = 200 * 1024**3,
                       max_files: int = 200_000) -> dict:
    """Extract untrusted output as files/directories only, with a size ceiling.

    No links, devices, absolute paths, duplicate members, or paths outside output/.
    Extraction uses a fresh staging directory to avoid following preexisting links.
    """
    destination = Path(destination)
    if destination.exists():
        raise ValueError("submission destination already exists")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as stage_name:
        stage = Path(stage_name)
        with tarfile.open(archive, "r:*") as bundle:
            seen, total = set(), 0
            for member in bundle:
                name = PurePosixPath(member.name)
                parts = name.parts
                if (name.is_absolute() or ".." in parts or not parts or parts[0] != "output"
                        or not (member.isfile() or member.isdir()) or name in seen):
                    raise ValueError("unsafe submission archive member")
                seen.add(name)
                if len(seen) > max_files:
                    raise ValueError("submission exceeds file count ceiling")
                total += member.size
                if total > max_bytes:
                    raise ValueError("submission exceeds size ceiling")
                target = stage.joinpath(*parts)
                if member.isdir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with bundle.extractfile(member) as source, target.open("xb") as output:
                        while chunk := source.read(8 * 1024 * 1024):
                            output.write(chunk)
                    target.chmod(member.mode & 0o777)
        if not (stage / "output" / "serve.sh").is_file():
            raise ValueError("submission missing output/serve.sh")
        os.replace(stage / "output", destination)
    return {"bytes": total, "files": len(seen)}


class S3TransferStore:
    """Trusted host signer for cross-provider object transfer (S3-compatible).

    The actor receives an object-scoped signed URL, never S3 account credentials.
    `client` is a boto3 S3 client configured on the trusted host. Configure bucket
    lifecycle expiry and account storage/egress limits independently of GPU leases.
    """
    def __init__(self, client, *, bucket: str, prefix: str = "frontier-mle"):
        self.client, self.bucket, self.prefix = client, bucket, prefix.strip("/")

    def sign(self, run_id: str, object_name: str, *, operation: str, expires_seconds: int = 900) -> dict:
        import re
        if not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", run_id):
            raise ValueError("invalid run ID")
        path = PurePosixPath(object_name)
        if not path.parts or path.is_absolute() or ".." in path.parts or "\\" in object_name:
            raise ValueError("invalid object name")
        if operation not in {"put", "get"} or not 1 <= expires_seconds <= 3600:
            raise ValueError("invalid operation or URL lifetime")
        key = f"{self.prefix}/{run_id}/{path}"
        url = self.client.generate_presigned_url(
            "put_object" if operation == "put" else "get_object",
            Params={"Bucket": self.bucket, "Key": key}, ExpiresIn=expires_seconds,
            HttpMethod="PUT" if operation == "put" else "GET")
        _https(url)
        return {"url": url, "object": str(path), "method": operation.upper(), "expires_seconds": expires_seconds}


HF_ENDPOINT = "https://huggingface.co"
COMMIT = re.compile(r"[0-9a-f]{40}")


def hf_source(source: dict) -> dict:
    """A pinned Hugging Face folder: {"repo", "revision" (a commit sha), "repo_type" ("model" or "dataset"),
    "prefix" (folder inside the repo, "" for the root), "exclude" (glob patterns relative to prefix), and optionally
    "include" (glob patterns relative to prefix; only matching files are fetched)}."""
    repo, revision = source.get("repo"), source.get("revision")
    repo_type, prefix = source.get("repo_type", "model"), source.get("prefix", "").strip("/")
    if not isinstance(repo, str) or repo.count("/") != 1 or any(part in ("", ".", "..") for part in repo.split("/")):
        raise ValueError(f"Expected an org/name Hugging Face repository, got {repo!r}")
    if not isinstance(revision, str) or not COMMIT.fullmatch(revision):
        raise ValueError(f"{repo} must be pinned to a 40-character commit sha, got {revision!r}")
    if repo_type not in ("model", "dataset"):
        raise ValueError("repo_type must be model or dataset")
    if prefix and any(part in ("", ".", "..") for part in prefix.split("/")):
        raise ValueError(f"Unsafe repository folder: {prefix}")
    exclude = source.get("exclude", [])
    if not isinstance(exclude, list) or not all(isinstance(pattern, str) for pattern in exclude):
        raise ValueError("exclude must be a list of glob patterns")
    result = {"repo": repo, "revision": revision, "repo_type": repo_type, "prefix": prefix, "exclude": exclude}
    if "include" in source:
        include = source["include"]
        if not isinstance(include, list) or not all(isinstance(pattern, str) for pattern in include):
            raise ValueError("include must be a list of glob patterns")
        result["include"] = include
    return result


def hf_files(source: dict, *, client, endpoint: str = HF_ENDPOINT) -> list[dict]:
    """Files under the pinned folder with the digests Hugging Face records for that commit."""
    source = hf_source(source)
    url = f"{endpoint}/api/{source['repo_type']}s/{source['repo']}/tree/{source['revision']}/{source['prefix']}"
    params, files = {"recursive": "true"}, []
    while url:
        for attempt in range(HF_RETRIES + 1):
            response = client.get(url, params=params)
            if response.status_code not in RETRY_STATUSES or attempt == HF_RETRIES:
                break
            time.sleep(_retry_delay(response, attempt))
        if response.status_code == 404:
            raise FileNotFoundError(f"{source['repo']}@{source['revision'][:12]} has no folder {source['prefix']!r}")
        if response.status_code != 200:
            raise RuntimeError(f"Listing {source['repo']}/{source['prefix']} failed: HTTP {response.status_code}")
        for entry in response.json():
            if entry["type"] != "file":
                continue
            relative = entry["path"][len(source["prefix"]):].lstrip("/") if source["prefix"] else entry["path"]
            if any(fnmatch.fnmatch(relative, pattern) for pattern in source["exclude"]):
                continue
            if "include" in source and not any(fnmatch.fnmatch(relative, pattern) for pattern in source["include"]):
                continue
            lfs = entry.get("lfs")
            files.append({"path": entry["path"], "relative": relative, "size": entry["size"],
                          "sha256": lfs["oid"] if lfs else None, "blob": None if lfs else entry["oid"]})
        url, params = response.links.get("next", {}).get("url"), None
    return files


# Anonymous Hugging Face downloads are rate limited (HTTP 429); a folder of a few thousand small
# files hits the limit on its own. Retry those and transient server errors within setup's timeout.
RETRY_STATUSES = {429, 500, 502, 503, 504}
HF_RETRIES = 12


def _retry_delay(response, attempt: int) -> float:
    try:
        return min(float(response.headers["Retry-After"]), 300)
    except (KeyError, ValueError):
        return min(5 * 2 ** attempt, 120)


def _fetch_one(url: str, target: Path, entry: dict, client) -> None:
    """Download one file atomically; LFS files are checked by SHA-256, small files by their git blob id."""
    for attempt in range(HF_RETRIES + 1):
        try:
            with client.stream("GET", url) as response:
                if response.status_code == 200:
                    return _write_verified(response, target, entry)
                if response.status_code not in RETRY_STATUSES or attempt == HF_RETRIES:
                    raise RuntimeError(f"Download of {entry['path']} failed: HTTP {response.status_code}")
                delay = _retry_delay(response, attempt)
        except httpx.TransportError:
            # Dropped connections and truncated bodies; _write_verified leaves no partial file behind.
            if attempt == HF_RETRIES:
                raise
            delay = min(5 * 2 ** attempt, 120)
        time.sleep(delay)


def _write_verified(response, target: Path, entry: dict) -> None:
    digest = hashlib.sha256() if entry["sha256"] else hashlib.sha1(f"blob {entry['size']}\0".encode())
    target.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(dir=target.parent, delete=False) as stream:
        tmp = Path(stream.name)
        try:
            size = 0
            for chunk in response.iter_bytes(8 * 1024 * 1024):
                size += len(chunk)
                if size > entry["size"]:
                    raise ValueError(f"{entry['path']} is larger than recorded")
                digest.update(chunk)
                stream.write(chunk)
            if size != entry["size"] or digest.hexdigest() != (entry["sha256"] or entry["blob"]):
                raise ValueError(f"{entry['path']} does not match the pinned commit")
            stream.close()
            # NamedTemporaryFile is 0600; setup runs as root and the agent user must read public data.
            os.chmod(tmp, 0o644)
            os.replace(tmp, target)
        finally:
            tmp.unlink(missing_ok=True)


def fetch_hf(source: dict, destination: str | Path, *, token: str | None = None, client=None,
             endpoint: str = HF_ENDPOINT, workers: int = 8, optional: bool = False) -> list[str]:
    """Download every file of a pinned folder into destination (the folder prefix is stripped).

    The commit sha pins the bytes: each file is verified against the digest Hugging Face records
    for that commit. A token, when given, is sent only to Hugging Face and not to its CDN redirect.
    An empty or missing folder downloads nothing."""
    from concurrent.futures import ThreadPoolExecutor
    source = hf_source(source)
    destination = Path(destination).resolve()
    own = client is None
    # httpx drops Authorization when a redirect leaves the Hugging Face origin.
    client = client or httpx.Client(timeout=120, follow_redirects=True,
                                    headers={"Authorization": f"Bearer {token}"} if token else None)
    try:
        try:
            files = hf_files(source, client=client, endpoint=endpoint)
        except FileNotFoundError:
            if optional:
                return []
            raise
        base = f"{endpoint}/{'datasets/' if source['repo_type'] == 'dataset' else ''}{source['repo']}/resolve/{source['revision']}"
        jobs = []
        for entry in files:
            relative = PurePosixPath(entry["relative"])
            if relative.is_absolute() or not relative.parts or ".." in relative.parts:
                raise ValueError(f"Unsafe repository path: {entry['path']}")
            target = destination.joinpath(*relative.parts)
            if not target.resolve().is_relative_to(destination):
                raise ValueError(f"{entry['path']} resolves outside the destination")
            jobs.append((f"{base}/{quote(entry['path'])}", target, entry))
        with ThreadPoolExecutor(max_workers=workers) as pool:
            list(pool.map(lambda job: _fetch_one(*job, client), jobs))
        return [str(target) for _, target, _ in jobs]
    finally:
        if own:
            client.close()


def main(argv=None):
    import argparse
    import json
    parser = argparse.ArgumentParser(description="Download a pinned Hugging Face folder and verify every file")
    parser.add_argument("command", choices=["fetch"])
    parser.add_argument("source", help="JSON file holding one source: repo, revision, repo_type, prefix, exclude")
    parser.add_argument("destination")
    args = parser.parse_args(argv)
    files = fetch_hf(json.loads(Path(args.source).read_text()), args.destination, token=os.environ.get("HF_TOKEN") or None)
    print(f"{len(files)} files -> {args.destination}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
