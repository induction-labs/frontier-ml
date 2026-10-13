"""Pinned scorer-weight preparation. Not needed for predictions-only validation."""
import hashlib
import json
from pathlib import Path
import subprocess
import zipfile
import tempfile
import shutil

MODELS = {
    "siglip": ("google/siglip-so400m-patch14-384", "9fdffc58afc957d1a03a25b10dba0329ab15c2a3"),
    "clip": ("openai/clip-vit-large-patch14", "32bd64288804d66eefd0ccbe215aa642df71cc41"),
}
DINO_REVISION = "7c446df5b9f45747937fb0d72314eb9f7b66930a"


def hashes(root):
    return {p.relative_to(root).as_posix(): hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and not {".cache", "__pycache__"}.intersection(p.parts)
            and not p.name.endswith((".pyc", ".partial")) and p.name != "weights.json"}


def prepare_models(directory, expected=None):
    import torch
    import lpips
    from dreamsim import dreamsim
    from huggingface_hub import snapshot_download
    root = Path(directory)
    root.mkdir(parents=True, exist_ok=True)
    for name, (repo, revision) in MODELS.items():
        snapshot_download(repo, revision=revision, local_dir=root/name,
            allow_patterns=["*.json", "*.txt", "*.model", "model.safetensors", "vocab.json", "merges.txt"])
    torch.hub.set_dir(str(root/"torch"))
    lpips.LPIPS(net="alex")
    cache = root/"dreamsim"
    cache.mkdir(exist_ok=True)
    required = ("dino_vitb16_pretrain.pth", "open_clip_vitb16_pretrain.pth.tar",
                "clip_vitb16_pretrain.pth.tar", "ensemble_lora")
    if not all((cache/name).exists() for name in required):
        from dreamsim.config import dreamsim_weights
        archive = cache/"pretrained.zip"
        # torch.hub's downloader accepted a truncated HTTP response during our audit.
        # curl checks transfer completeness and resumes interrupted large downloads.
        subprocess.run(["curl", "-fL", "--retry", "5", "--retry-all-errors", "--continue-at", "-",
                        dreamsim_weights["ensemble"], "-o", str(archive)], check=True)
        with zipfile.ZipFile(archive) as package:
            if package.testzip() is not None:
                raise RuntimeError("Corrupt DreamSim archive")
            package.extractall(cache)
    dino = cache/"facebookresearch_dino_main"
    if not dino.is_dir():
        with tempfile.TemporaryDirectory() as temporary:
            archive = Path(temporary)/"dino.zip"
            subprocess.run(["curl", "-fL", "--retry", "5", "--retry-all-errors",
                "https://github.com/facebookresearch/dino/archive/"+DINO_REVISION+".zip", "-o", str(archive)], check=True)
            with zipfile.ZipFile(archive) as package:
                package.extractall(temporary)
            shutil.move(str(Path(temporary)/("dino-"+DINO_REVISION)), dino)
    dreamsim(pretrained=True, device="cpu", cache_dir=str(root/"dreamsim"),
             normalize_embeds=True, dreamsim_type="ensemble")
    # The download archive is redundant after extraction and may vary in metadata.
    for archive in (root/"dreamsim").glob("*.zip"):
        archive.unlink()
    manifest = {"models": MODELS, "dino_revision": DINO_REVISION, "files": hashes(root)}
    if expected:
        pinned = json.loads(Path(expected).read_text())
        if manifest["files"] != pinned["files"]:
            raise RuntimeError("Downloaded metric weights differ from the pinned weight manifest")
    (root/"weights.json").write_text(json.dumps(manifest, indent=2, sort_keys=True)+"\n")
    return manifest


def check_models(directory):
    root = Path(directory)
    expected = json.loads((root/"weights.json").read_text())
    for name in MODELS:
        if list(MODELS[name]) != expected["models"][name]:
            raise RuntimeError("Metric model revision mismatch")
    if hashes(root) != expected["files"]:
        raise RuntimeError("Metric weight hash mismatch")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("directory")
    parser.add_argument("--expected")
    args = parser.parse_args()
    prepare_models(args.directory, args.expected)
