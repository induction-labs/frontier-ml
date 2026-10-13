"""Trusted static evidence checks; candidate Python is never imported."""
import hashlib
import json
from pathlib import Path
import re


def inspect_submission(root):
    root = Path(root).resolve()
    problems = []
    try:
        manifest_path = root / "training_manifest.json"
        if manifest_path.is_symlink():
            raise ValueError("Invalid training_manifest.json")
        manifest = json.loads(manifest_path.read_text())
        if type(manifest.get("optimizer_steps")) is not int or manifest["optimizer_steps"] <= 0:
            raise ValueError("Positive integer optimizer_steps required")
        if not isinstance(manifest.get("base_model"), str) or not manifest["base_model"].strip() or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", manifest.get("base_revision", "")):
            raise ValueError("Base model and immutable revision required")
        sources = manifest.get("training_sources")
        if not isinstance(sources, list) or not sources or any(not isinstance(s, dict) or not isinstance(s.get("source"), str) or not s["source"].strip() or not re.fullmatch(r"[0-9a-f]{40}|[0-9a-f]{64}", s.get("revision", "")) for s in sources):
            raise ValueError("Pinned training_sources required")
        evidence = manifest.get("evidence")
        if not isinstance(evidence, dict) or not {"code", "config", "log"} <= set(evidence):
            raise ValueError("Training code/config/log evidence required")
        weights = manifest.get("weights")
        if not isinstance(weights, list) or not weights:
            raise ValueError("Packaged trained weight inventory required")
        seen, weight_bytes, empty_evidence = set(), 0, []
        # Each evidence kind names one file or a list of files; additional kinds are allowed. An empty
        # file or list is recorded for the reviewer, who sees every file anyway; it is not a verdict.
        for kind, value in evidence.items():
            names = value if isinstance(value, list) else [value]
            if not names:
                empty_evidence.append(f"{kind}: no files listed")
            for name in names:
                path = _local_file(root, name)
                if path.stat().st_size == 0:
                    empty_evidence.append(f"{kind}: {name} is empty")
        for item in weights:
            if not isinstance(item, dict) or not isinstance(item.get("path"), str) or not isinstance(item.get("sha256"), str):
                raise ValueError("Weight inventory needs path and sha256")
            path = _local_file(root, item["path"])
            if path in seen:
                raise ValueError("Duplicate weight file")
            seen.add(path)
            size = path.stat().st_size
            if size == 0:
                raise ValueError("Empty weight file")
            with path.open("rb") as stream:
                actual = hashlib.file_digest(stream, "sha256").hexdigest()
            if actual != item["sha256"].strip().lower():
                raise ValueError("Weight digest mismatch")
            weight_bytes += size
        return {"passed": True, "optimizer_steps": manifest["optimizer_steps"], "weight_files": len(seen), "weight_bytes": weight_bytes,
                "empty_evidence": empty_evidence}
    except (OSError, ValueError, TypeError, KeyError, AttributeError) as error:
        problems.append(str(error))
    return {"passed": False, "problems": problems}


def _local_file(root, name):
    if not isinstance(name, str) or not name or Path(name).is_absolute() or ".." in Path(name).parts:
        raise ValueError("Evidence paths must be relative")
    raw = root / name
    current = root
    for component in Path(name).parts:
        current = current / component
        if current.is_symlink():
            raise ValueError("Symlink evidence is forbidden")
    path = raw.resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Missing or escaping evidence path")
    return path
