"""Static training-evidence checks. Historical provenance also needs trusted review."""
import hashlib
import json
from pathlib import Path
import re

def checked_file(root, relative):
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Evidence paths must be nonempty relative paths")
    path = root / relative
    if not path.resolve().is_relative_to(root.resolve()) or not path.is_file() or path.is_symlink():
        raise ValueError("Evidence path missing, symlinked or outside submission")
    return path


def inspect_training(root):
    root = Path(root)
    manifest_path = checked_file(root, "training_manifest.json")
    manifest = json.loads(manifest_path.read_text())
    if type(manifest.get("schema_version")) is not int or manifest["schema_version"] != 1:
        raise ValueError("Expected training manifest schema_version 1")
    for field in ("base_models", "training_datasets", "teacher_models", "reused_artifacts"):
        entries = manifest.get(field)
        if not isinstance(entries, list) or (field == "training_datasets" and not entries):
            raise ValueError(f"Manifest requires {field}")
        for entry in entries:
            if not isinstance(entry, dict) or any(not isinstance(entry.get(k), str) or not entry[k].strip()
                                                   for k in ("source", "revision")):
                raise ValueError(f"Every {field} entry needs source and revision")
            if not re.fullmatch(r"(?:[a-f0-9]{40}|sha256:[a-f0-9]{64})", entry["revision"]):
                raise ValueError("Sources must use immutable commits or sha256:<content digest>")
    training = manifest.get("training")
    if not isinstance(training, dict) or type(training.get("steps")) is not int or training["steps"] <= 0:
        raise ValueError("Training evidence must declare positive parameter-update steps")
    for field in ("script", "config", "log"):
        path = checked_file(root, training.get(field))
        if not path.stat().st_size:
            raise ValueError("Training evidence files must be nonempty")
    weights = manifest.get("trained_weights")
    if not isinstance(weights, list) or not weights:
        raise ValueError("List the resulting trained weights")
    seen = set()
    for weight in weights:
        if not isinstance(weight, dict):
            raise ValueError("Invalid trained weight entry")
        path = checked_file(root, weight.get("path"))
        if path in seen:
            raise ValueError("Trained weight entries must identify distinct files")
        seen.add(path)
        if not isinstance(weight.get("sha256"), str) or not re.fullmatch(r"[a-f0-9]{64}", weight["sha256"]):
            raise ValueError("Each trained weight file needs its SHA-256")
        with path.open("rb") as stream:
            digest = hashlib.file_digest(stream, "sha256").hexdigest()
        if digest != weight["sha256"]:
            raise ValueError("Trained weight checksum mismatch")
    return {"passed": True, "steps_declared": training["steps"], "weight_files": len(weights),
            "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest()}
