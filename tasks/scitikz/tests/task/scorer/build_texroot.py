"""Assemble the TeX-only root before installing Python packages or downloading models."""
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess

root = Path("/opt/scitikz/texroot")
root.mkdir(parents=True)
for directory in ("usr/bin", "usr/lib", "usr/lib64", "usr/share", "etc", "var/lib/texmf", "var/cache/fontconfig"):
    source = Path("/")/directory
    if source.exists():
        shutil.copytree(source, root/directory, symlinks=True, dirs_exist_ok=True)
for directory in ("bin", "sbin", "lib", "lib64"):
    source = Path("/")/directory
    if source.is_symlink():
        (root/directory).symlink_to(os.readlink(source))
    elif source.exists():
        shutil.copytree(source, root/directory, symlinks=True)
(root/"jobs").mkdir(mode=0o711)
(root/"tmp").mkdir(mode=0o555)
(root/"dev").mkdir()
# Modal image builds disallow device creation; the trusted runtime creates only
# null/random devices before dropping identity and entering the jail.
subprocess.run(["chown", "-R", "0:0", str(root)], check=True)
subprocess.run(["chmod", "-R", "go-w", str(root)], check=True)
files = {}
for p in sorted(root.rglob("*")):
    if p.is_file() and not p.is_symlink():
        files[p.relative_to(root).as_posix()] = hashlib.file_digest(p.open("rb"), "sha256").hexdigest()
Path("/opt/scitikz/tex-build.json").write_text(json.dumps({
    "fingerprint": hashlib.sha256(json.dumps(files, sort_keys=True).encode()).hexdigest(),
    "files": len(files), "packages": subprocess.check_output(["dpkg-query", "-W"], text=True),
    "pdflatex": subprocess.check_output(["pdflatex", "--version"], text=True).splitlines()[0],
}, indent=2)+"\n")
