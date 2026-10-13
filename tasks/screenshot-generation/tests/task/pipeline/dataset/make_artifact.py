"""Materialize artifacts/screenshot-generation/z-image-turbo (hardlinks) with a lock file, and the public
STARTING_ARTIFACT.json describing it."""
import hashlib
import json
import os
from pathlib import Path

SRC = Path("/scratch/frontier-ml-workspace/work/screenshot-generation/models/zimage-turbo")
MAIN = Path("/scratch/frontier-ml-workspace/frontier-mle-clean")
DST = MAIN / "artifacts/screenshot-generation/z-image-turbo"
REPO, REV = "Tongyi-MAI/Z-Image-Turbo", "f332072aa78be7aecdf3ee76d5c247082da564a6"

files = {}
for path in sorted(SRC.rglob("*")):
    if not path.is_file() or ".cache" in path.parts:
        continue
    rel = path.relative_to(SRC).as_posix()
    target = DST / "weights" / rel
    target.parent.mkdir(parents=True, exist_ok=True)
    if not target.exists():
        os.link(path, target)
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(1 << 22), b""):
            h.update(block)
    files[rel] = {"sha256": h.hexdigest(), "bytes": path.stat().st_size}
lock = {"schema_version": 1, "model": {"repo": REPO, "revision": REV, "license": "Apache-2.0", "files": files},
        "sandbox_path": "/artifacts/screenshot-generation/z-image-turbo/weights"}
(DST / "artifacts.lock.json").write_text(json.dumps(lock, indent=1) + "\n")
public = {"schema_version": 1, "model": {"repo": REPO, "revision": REV, "license": "Apache-2.0", "format": "diffusers (ZImagePipeline)",
                                         "sandbox_path": "/artifacts/screenshot-generation/z-image-turbo/weights",
                                         "files": files},
          "reference_settings": {"num_inference_steps": 9, "guidance_scale": 0.0, "height": 720, "width": 1280,
                                 "note": "Settings of the unmodified baseline that defines the CMMD reference."}}
out = MAIN / "data/screenshot-generation/public"
out.mkdir(parents=True, exist_ok=True)
(out / "STARTING_ARTIFACT.json").write_text(json.dumps(public, indent=1) + "\n")
print(len(files), "files", sum(v["bytes"] for v in files.values()) / 1e9, "GB")
