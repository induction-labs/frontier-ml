"""Pinned Maple checkpoint identity and artifact provisioning; MMMU scoring lives in tasks/maple-vlm/verifier.py."""

MAPLE_REPO = "deepgrove/maple-preview"
MAPLE_REVISION = "ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07"


def sha256(path):
    import hashlib
    from pathlib import Path
    with Path(path).open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def _safe_path(root, relative):
    from pathlib import Path
    root = Path(root).resolve()
    path = (root / relative).resolve()
    if not path.is_relative_to(root):
        raise ValueError(f"Path escapes split: {relative}")
    return path


def provision_artifacts(lock_path, destination, *, local_weights=None, fetch_weights=False, fetch_sources=True):
    """Provision pinned artifacts before offline evaluation; never called at startup.

    Existing weights can be hardlinked from a local snapshot. Every weight file is
    SHA-256 checked against the frozen package before use. Network model downloads
    require explicit `fetch_weights=True` to avoid accidentally downloading 40 GB.
    """
    import hashlib
    import json
    import os
    from pathlib import Path
    import shutil
    import tarfile
    import tempfile
    import urllib.request

    lock = json.loads(Path(lock_path).read_text())
    destination = Path(destination)
    destination.mkdir(parents=True, exist_ok=True)
    receipt = {"model": None, "sources": {}}
    model = lock["model"]
    if model["repo"] != MAPLE_REPO or model["revision"] != MAPLE_REVISION:
        raise ValueError("Unexpected Maple checkpoint lock")
    if local_weights is not None or fetch_weights:
        for relative, expected in model["files"].items():
            target = _safe_path(destination / "weights", relative)
            target.parent.mkdir(parents=True, exist_ok=True)
            if not target.exists():
                if local_weights is not None:
                    source = _safe_path(local_weights, relative)
                    try:
                        os.link(source, target)
                    except OSError:
                        shutil.copy2(source, target)
                else:
                    url = f"https://huggingface.co/{model['repo']}/resolve/{model['revision']}/{relative}"
                    with urllib.request.urlopen(url, timeout=120) as response, target.with_suffix(target.suffix + ".part").open("wb") as output:
                        shutil.copyfileobj(response, output)
                    target.with_suffix(target.suffix + ".part").replace(target)
            if target.stat().st_size != expected["bytes"] or sha256(target) != expected["sha256"]:
                raise ValueError(f"Maple artifact checksum mismatch: {relative}")
        receipt["model"] = {"repo": model["repo"], "revision": model["revision"], "files": len(model["files"]), "verified": True}
    if fetch_sources:
        for name, source in lock["sources"].items():
            archive = destination / (name + ".tar.gz")
            if not archive.exists():
                url = source["url"].replace("https://github.com/", "https://codeload.github.com/") + "/tar.gz/" + source["revision"]
                with urllib.request.urlopen(url, timeout=120) as response, archive.with_suffix(".part").open("wb") as output:
                    shutil.copyfileobj(response, output)
                archive.with_suffix(".part").replace(archive)
            digest = sha256(archive)
            if "archive_sha256" not in source or digest != source["archive_sha256"]:
                raise ValueError(f"Source archive hash missing or mismatched: {name}")
            output_root = destination / "sources" / name
            if not output_root.exists():
                # Tar data filter rejects device files, escaping paths and escaping links.
                with tempfile.TemporaryDirectory(dir=destination) as temporary:
                    with tarfile.open(archive) as bundle:
                        bundle.extractall(temporary, filter="data")
                    children = list(Path(temporary).iterdir())
                    if len(children) != 1 or not children[0].is_dir():
                        raise ValueError("Expected one root directory in source archive")
                    output_root.parent.mkdir(parents=True, exist_ok=True)
                    children[0].rename(output_root)
            receipt["sources"][name] = {"revision": source["revision"], "archive_sha256": digest}
    (destination / "receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    return receipt


def main():
    import argparse
    import json
    from pathlib import Path
    parser = argparse.ArgumentParser(description="Fetch or reuse pinned Maple artifacts before offline startup")
    parser.add_argument("--lock", type=Path, required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--local-weights", type=Path)
    parser.add_argument("--fetch-weights", action="store_true")
    parser.add_argument("--no-sources", action="store_true")
    args = parser.parse_args()
    print(json.dumps(provision_artifacts(args.lock, args.destination, local_weights=args.local_weights, fetch_weights=args.fetch_weights, fetch_sources=not args.no_sources), indent=2))

if __name__ == "__main__":
    main()
