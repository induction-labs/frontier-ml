"""Operator-only clean Linux build of the public native dependency; no candidate input.

    uv run --env-file .env modal run tasks/pokemon-random-battle/pipeline/build_poke_engine.py \
        --output /tmp/pokemon-engine-build

Inspect the resulting provenance and pin the wheel digest in review.py before
preparing a payload. This script never edits an allowlist or publishes anything.
"""
import hashlib
import json
import os
from pathlib import Path
import subprocess
import tarfile
import tempfile
import urllib.request

import modal

SOURCE_URL = ("https://files.pythonhosted.org/packages/cf/69/"
              "47d572f526943c519927acff95685bffa1aba7024f8433bc1924d7941ae2/"
              "poke_engine-0.0.48.tar.gz")
SOURCE_SHA256 = "070010686f2aedff11e25137e696e301ccd80fd57c805d255464067fc905ca12"
RUST_VERSION = "1.89.0"
MATURIN_VERSION = "1.9.4"
app = modal.App("frontier-pokemon-dependency-build")
image = (modal.Image.from_registry("python:3.12-slim-bookworm")
    .apt_install("curl", "build-essential", "ca-certificates")
    .env({"RUSTUP_HOME": "/opt/rustup", "CARGO_HOME": "/opt/cargo",
          "PATH": "/opt/cargo/bin:/usr/local/bin:/usr/bin:/bin", "SOURCE_DATE_EPOCH": "1784471630"})
    .run_commands("curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs -o /tmp/rustup.sh",
                  f"sh /tmp/rustup.sh -y --profile minimal --default-toolchain {RUST_VERSION}")
    .pip_install(f"maturin=={MATURIN_VERSION}"))


@app.function(image=image, cpu=4, memory=8192, timeout=900)
def build():
    with urllib.request.urlopen(SOURCE_URL, timeout=60) as response:
        source = response.read()
    if hashlib.sha256(source).hexdigest() != SOURCE_SHA256:
        raise ValueError("Upstream source digest changed")
    with tempfile.TemporaryDirectory() as directory:
        root = Path(directory)
        archive = root / "poke_engine-0.0.48.tar.gz"
        archive.write_bytes(source)
        with tarfile.open(archive) as packaged:
            packaged.extractall(root, filter="data")
        checkout = root / "poke_engine-0.0.48"
        command = ["maturin", "build", "--release", "--locked", "--compatibility", "linux",
                   "--interpreter", "python3.12", "--features", "poke-engine/terastallization",
                   "--no-default-features", "--out", str(root / "wheels")]
        subprocess.run(command, cwd=checkout, check=True,
                       env={**os.environ, "RUSTFLAGS": f"--remap-path-prefix={root}=/build"})
        wheel, = (root / "wheels").glob("*.whl")
        subprocess.run(["python3.12", "-m", "pip", "install", "--no-index", str(wheel)], check=True)
        # Exercise actual native serialization and search, including the Tera feature.
        subprocess.run(["python3.12", "-c", "from poke_engine import State, monte_carlo_tree_search; "
                        "s=State(); s.to_string(); r=monte_carlo_tree_search(s, duration_ms=1); "
                        "assert any('tera' in x.move_choice for x in r.side_one)"], check=True)
        raw = wheel.read_bytes()
        manifest = {"schema_version": 1, "source_url": SOURCE_URL, "source_sha256": SOURCE_SHA256,
                    "rust": RUST_VERSION, "maturin": MATURIN_VERSION,
                    "base_image": "python:3.12-slim-bookworm", "cargo_locked": True,
                    "rustflags": "--remap-path-prefix=<build-directory>=/build",
                    "features": ["poke-engine/terastallization"], "default_features": False,
                    "wheel": wheel.name, "wheel_sha256": hashlib.sha256(raw).hexdigest(),
                    "wheel_bytes": len(raw), "smoke_test": "native serialization and MCTS with Tera actions",
                    "cargo_lock_sha256": hashlib.sha256((checkout / "Cargo.lock").read_bytes()).hexdigest()}
        return {wheel.name: raw, archive.name: source,
                "engine-build.json": (json.dumps(manifest, indent=2) + "\n").encode()}


@app.local_entrypoint()
def main(output: str):
    destination = Path(output)
    destination.mkdir(parents=True, exist_ok=False)
    for name, raw in build.remote().items():
        (destination / name).write_bytes(raw)
    print((destination / "engine-build.json").read_text())
