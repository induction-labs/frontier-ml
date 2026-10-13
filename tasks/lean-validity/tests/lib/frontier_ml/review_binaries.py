"""Trusted provenance audit for vendored Python wheels in reviewed submissions.

The architecture review never reads opaque binaries. Offline submissions still have to
vendor dependencies, so a wheel is accepted exactly when its bytes are a published
upstream artifact: its SHA-256 appears in an operator allowlist, or (when lookup is
enabled) PyPI or the official PyTorch index publishes a file of that name with that
digest. This inspector only establishes wheel provenance. The evidence collector
handles general archives separately; a compressed data file is not a dependency
merely because its bytes are not plain text. Loose native executables are never approved
here; the evidence collector hands anything unapproved to the reviewer as uninspected.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import re
import urllib.parse
import urllib.request

WHEEL = re.compile(r"^(?P<name>[A-Za-z0-9_.]+)-(?P<version>[A-Za-z0-9_.!+]+)(-\d[^-]*)?-[^-]+-[^-]+-[^-]+\.whl$")
PYPI = "https://pypi.org/pypi/{name}/{version}/json"
PYTORCH_INDEX = "https://download.pytorch.org/whl/{name}/"
REVIEWER = "frontier-ml upstream wheel digest audit"


def _fetch(url: str, timeout: float = 60) -> bytes:
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310 - fixed HTTPS hosts only
        return response.read()


def sha256_file(path: Path) -> str:
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def load_allowlist(path: str | Path | None) -> dict[str, str]:
    """Operator file: {"<sha256>": "<provenance URL or note>", ...}."""
    if not path:
        return {}
    data = json.loads(Path(path).read_text())
    if not isinstance(data, dict) or not all(
            isinstance(key, str) and re.fullmatch(r"[0-9a-f]{64}", key) and isinstance(value, str) and value
            for key, value in data.items()):
        raise ValueError("Wheel allowlist must map lowercase SHA-256 digests to provenance strings")
    return data


class UpstreamWheelInspector:
    """Callable binary_inspector for frontier_ml.review.create_review."""

    def __init__(self, allowlist: dict[str, str] | None = None, *, lookup: bool = True, fetch=_fetch):
        self.allowlist = dict(allowlist or {})
        self.lookup = lookup
        self.fetch = fetch
        self._cache: dict[tuple[str, str], str | None] = {}

    def __call__(self, path) -> dict:
        path = Path(path)
        digest = sha256_file(path)
        audit = {"approved": False, "sha256": digest, "reviewer": REVIEWER, "provenance": None}
        if path.suffix != ".whl":
            return {**audit, "reason": "only published Python wheels are audited; ship source or a task inspector"}
        if digest in self.allowlist:
            return {**audit, "approved": True, "provenance": self.allowlist[digest]}
        if not self.lookup:
            return {**audit, "reason": "wheel digest is not in the operator allowlist"}
        key = (path.name, digest)
        if key not in self._cache:
            self._cache[key] = self._upstream(path.name, digest)
        provenance = self._cache[key]
        if provenance:
            return {**audit, "approved": True, "provenance": provenance}
        return {**audit, "reason": "no upstream file with this name and digest on PyPI or the PyTorch index"}

    def _upstream(self, filename: str, digest: str) -> str | None:
        match = WHEEL.match(filename)
        if not match:
            return None
        name = re.sub(r"[-_.]+", "-", match["name"]).lower()
        try:
            release = json.loads(self.fetch(PYPI.format(name=urllib.parse.quote(name), version=urllib.parse.quote(match["version"]))))
            for item in release.get("urls", []):
                if item.get("filename") == filename and item.get("digests", {}).get("sha256") == digest:
                    return item.get("url") or PYPI.format(name=name, version=match["version"])
        except Exception:  # noqa: BLE001 - unknown on PyPI; try the PyTorch index
            pass
        for index_name in dict.fromkeys((name, name.replace("-", "_"))):
            url = PYTORCH_INDEX.format(name=urllib.parse.quote(index_name))
            try:
                index = self.fetch(url).decode()
            except Exception:  # noqa: BLE001 - try the next spelling
                continue
            for candidate in dict.fromkeys((filename, filename.replace("+", "%2B"))):
                if f"{candidate}#sha256={digest}" in index:
                    return url + candidate
        return None


def default_binary_inspector(options: dict) -> UpstreamWheelInspector:
    """Built from config["architecture_review"]: wheel_allowlist (path) and wheel_lookup (bool, default true)."""
    lookup = options.get("wheel_lookup", True)
    if type(lookup) is not bool:
        raise ValueError("architecture_review.wheel_lookup must be a boolean")
    return UpstreamWheelInspector(load_allowlist(options.get("wheel_allowlist")), lookup=lookup)
