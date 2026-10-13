"""Trusted static checks: submission metadata and the pinned starting-artifact lock.

The shared architecture reviewer inventories weights and traces training and serving code;
these checks only establish that architecture.json is well formed, that its evidence files
exist, and that the declared serialization alphabet size is an integer the verifier can
cross-check against the /encode responses. `verify_artifact_lock` is an operator utility
that checks a DeepSeek-OCR download against the pinned lock.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

from frontier_ml.validation.errors import VerificationFailure

MAX_CODEBOOK = 2 ** 24


def _inside(root, relative):
    if not isinstance(relative, str) or not relative or relative.startswith("/") or "\\" in relative:
        raise VerificationFailure(f"Evidence path must be relative: {relative!r}")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file() or (root / relative).is_symlink():
        raise VerificationFailure(f"Evidence path is not a regular submission file: {relative}")
    return path


def static_checks(snapshot):
    root = Path(snapshot).resolve()
    path = _inside(root, "architecture.json")
    try:
        metadata = json.loads(path.read_text())
    except (ValueError, UnicodeDecodeError) as exc:
        raise VerificationFailure("architecture.json is not valid JSON") from exc
    if not isinstance(metadata, dict):
        raise VerificationFailure("architecture.json must be a JSON object")
    size = metadata.get("codebook_size")
    if type(size) is not int or not 2 <= size <= MAX_CODEBOOK:
        raise VerificationFailure(f"architecture.json codebook_size must be an integer between 2 and {MAX_CODEBOOK}")
    for key in ("base_model", "base_revision"):
        if not isinstance(metadata.get(key), str) or not metadata[key].strip():
            raise VerificationFailure(f'architecture.json {key} must be a nonempty string ("none"/"from-scratch" without a starting checkpoint)')
    report = {"codebook_size": size, "base_model": metadata["base_model"], "base_revision": metadata["base_revision"]}
    for field in ("training_evidence_files", "inference_evidence_files"):
        references = metadata.get(field)
        if not isinstance(references, list) or not references:
            raise VerificationFailure(f"{field} must reference submission files")
        for relative in references:
            _inside(root, relative)
        report[field] = references
    if "serve.sh" not in metadata["inference_evidence_files"]:
        raise VerificationFailure("inference_evidence_files must include serve.sh")
    return report


def verify_artifact_lock(directory, lock_path):
    """Every file of the pinned DeepSeek-OCR checkpoint is present with the locked size and SHA-256."""
    root = Path(directory)
    lock = json.loads(Path(lock_path).read_text())["model"]
    for name, expected in lock["files"].items():
        path = root / name
        if not path.is_file():
            raise VerificationFailure(f"Artifact is missing {name}")
        if path.stat().st_size != expected["bytes"]:
            raise VerificationFailure(f"Artifact file {name} has the wrong size")
        if hashlib.file_digest(path.open("rb"), "sha256").hexdigest() != expected["sha256"]:
            raise VerificationFailure(f"Artifact file {name} does not match the pinned checkpoint")
    return {"repo": lock["repo"], "revision": lock["revision"], "files": len(lock["files"])}
