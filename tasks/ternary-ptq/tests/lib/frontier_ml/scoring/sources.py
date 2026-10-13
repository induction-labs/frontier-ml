"""Pinned starting-artifact and benchmark source locks, plus checked downloads."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
import urllib.request

SOURCES = Path(__file__).with_name('metadata') / 'sources.json'


def source_lock() -> dict:
    return json.loads(SOURCES.read_text())


def download_file(url: str, path: Path, *, size: int | None = None,
                  sha256: str | None = None, token: str | None = None) -> None:
    """Atomic download; content length and LFS SHA256 are checked when available."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists() and (size is None or path.stat().st_size == size):
        if sha256 is not None and hashlib.sha256(path.read_bytes()).hexdigest() == sha256:
            return
    request = urllib.request.Request(url, headers={'Authorization': f'Bearer {token}'} if token else {})
    temporary = path.with_suffix(path.suffix + '.partial')
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temporary.open('wb') as output:
            while chunk := response.read(1024 * 1024):
                output.write(chunk)
        if size is not None and temporary.stat().st_size != size:
            raise ValueError(f'Size mismatch for {path.name}')
        if sha256 is not None and hashlib.sha256(temporary.read_bytes()).hexdigest() != sha256:
            raise ValueError(f'SHA256 mismatch for {path.name}')
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)
