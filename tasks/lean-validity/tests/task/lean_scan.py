"""Collect leads for reviewing possible compiled Lean artifacts.

Unknown archives are reported as uninspected, not labeled as Lean. The architecture
review remains responsible for establishing that serving does not use a prover.
Byte markers are leads, never proof that an artifact contains a Lean runtime.
The scanner never imports or executes candidate code.
"""
from __future__ import annotations

import bz2
import gzip
import io
import lzma
import mmap
import stat
import shutil
import tempfile
import tarfile
import zipfile
from pathlib import Path, PurePosixPath

OLEAN_MAGIC = b"olean"
ELF_MAGIC = b"\x7fELF"
LEAN_RUNTIME_MARKERS = (b"lean_initialize_runtime_module", b"lean_io_mk_world", b"initialize_Init_Prelude",
                        b"lean_set_panic_messages", b"l_Lean_Elab_", b"libleanshared", b"\x00lean_main\x00", b"l_Lake_",
                        b"initialize_Lake")
OPAQUE_MAGIC = {b"\x28\xb5\x2f\xfd": "zstd", b"7z\xbc\xaf\x27\x1c": "7z", b"Rar!\x1a\x07": "rar",
                b"\x04\x22\x4d\x18": "lz4", b"hsqs": "squashfs", b"sqsh": "squashfs"}


class ScanResult:
    def __init__(self):
        self.violations: list[str] = []
        self.uninspected: list[str] = []
        self.review_leads: list[str] = []
        self.files = 0
        self.lean_sources = 0
        self.archives = 0

    def add(self, where: str, why: str):
        if len(self.violations) < 200:
            self.violations.append(f"{where}: {why}")

    def as_dict(self) -> dict:
        return {"passed": not self.violations, "violations": self.violations, "files_scanned": self.files,
                "archives_scanned": self.archives, "lean_source_files": self.lean_sources,
                "uninspected": self.uninspected, "review_leads": self.review_leads}


def _name_rules(result: ScanResult, where: str, relative: str):
    # Counts are informational; source examples do not establish a packaged prover.
    if PurePosixPath(relative).suffix == ".lean":
        result.lean_sources += 1


def _content_rules(result: ScanResult, where: str, head: bytes):
    if head.startswith(OLEAN_MAGIC):
        result.review_leads.append(f"{where}: starts with 'olean'; verify the actual format and contents, since ordinary text can have this prefix")
    for magic, kind in OPAQUE_MAGIC.items():
        if head.startswith(magic):
            result.uninspected.append(f"{where}: unscanned {kind} archive")


def _elf_rules(result: ScanResult, where: str, data) -> None:
    for marker in LEAN_RUNTIME_MARKERS:
        if data.find(marker) != -1:
            result.review_leads.append(f"{where}: ELF-like bytes include {marker!r}; raw strings do not establish Lean runtime code or use")
            return


def _is_archive(name: str, head: bytes) -> str | None:
    if head.startswith(b"PK\x03\x04") or head.startswith(b"PK\x05\x06"):
        return "zip"
    if head.startswith(b"\x1f\x8b"):
        return "gzip"
    if head.startswith(b"BZh"):
        return "bz2"
    if head.startswith(b"\xfd7zXZ\x00"):
        return "xz"
    if len(head) >= 262 and head[257:262] == b"ustar":
        return "tar"
    # Archive-like names with unrecognized content (e.g. zlib streams named .gz) are
    # treated as plain data; detection is by magic bytes, not by file name.
    return None


def _scan_stream(result: ScanResult, where: str, name: str, stream, size: int | None, depth: int):
    """Scan one file-like object (a submission file or an archive member).

    Only the first 512 bytes are read unless the object is an ELF binary or an archive,
    so large tensor payloads inside zip-based checkpoints are never loaded in full.
    """
    head = stream.read(512)
    _content_rules(result, where, head)
    kind = _is_archive(name, head)
    if not head.startswith(ELF_MAGIC) and not kind:
        return
    if hasattr(stream, "fileno") and depth == 0:
        try:
            with mmap.mmap(stream.fileno(), 0, access=mmap.ACCESS_READ) as data:
                if head.startswith(ELF_MAGIC):
                    _elf_rules(result, where, data)
            if kind:
                stream.seek(0)
                _scan_archive(result, where, kind, stream, depth + 1)
            return
        except (OSError, ValueError, io.UnsupportedOperation):
            stream.seek(0)
            head = stream.read(512)
    # Spool nested binaries/archives to disk rather than reject by size or load
    # their entire expanded contents into Python memory.
    with tempfile.TemporaryFile() as copied:
        copied.write(head)
        shutil.copyfileobj(stream, copied, length=1024 * 1024)
        copied.seek(0)
        if head.startswith(ELF_MAGIC):
            with mmap.mmap(copied.fileno(), 0, access=mmap.ACCESS_READ) as data:
                _elf_rules(result, where, data)
        if kind:
            _scan_archive(result, where, kind, copied, depth + 1)


def _scan_archive(result: ScanResult, where: str, kind: str, stream, depth: int):
    result.archives += 1
    try:
        if kind == "zip":
            with zipfile.ZipFile(stream) as archive:
                for info in archive.infolist():
                    member = f"{where}!{info.filename}"
                    if info.is_dir():
                        _name_rules(result, member, info.filename.rstrip("/") + "/x")
                        continue
                    result.files += 1
                    _name_rules(result, member, info.filename)
                    with archive.open(info) as inner:
                        _scan_stream(result, member, info.filename, inner, info.file_size, depth)
            return
        if kind in ("gzip", "bz2", "xz", "tar"):
            try:
                with tarfile.open(fileobj=stream, mode="r:*") as archive:
                    for info in archive:
                        member = f"{where}!{info.name}"
                        if info.isdir():
                            _name_rules(result, member, info.name.rstrip("/") + "/x")
                            continue
                        _name_rules(result, member, info.name)
                        if info.issym() or info.islnk():
                            _name_rules(result, member, info.linkname)
                            continue
                        if not info.isfile():
                            continue
                        result.files += 1
                        inner = archive.extractfile(info)
                        if inner is not None:
                            _scan_stream(result, member, info.name, inner, info.size, depth)
                return
            except tarfile.ReadError:
                if kind == "tar":
                    raise
            stream.seek(0)
            opener = {"gzip": lambda f: gzip.GzipFile(fileobj=f), "bz2": bz2.BZ2File, "xz": lzma.LZMAFile}[kind]
            with opener(stream) as inner:
                base = where.rsplit("!", 1)[-1].rsplit("/", 1)[-1]
                _scan_stream(result, where + "!", base.rsplit(".", 1)[0], inner, None, depth)
            return
    except (zipfile.BadZipFile, zipfile.LargeZipFile, tarfile.TarError, OSError, EOFError, lzma.LZMAError,
            ValueError, NotImplementedError, RuntimeError) as error:
        result.uninspected.append(f"{where}: archive cannot be scanned ({type(error).__name__})")


def scan_submission(root: str | Path) -> dict:
    root = Path(root).resolve()
    result = ScanResult()
    for path in sorted(root.rglob("*")):
        relative = path.relative_to(root).as_posix()
        mode = path.lstat().st_mode
        if stat.S_ISLNK(mode):
            result.add(relative, "symlink")
            continue
        if stat.S_ISDIR(mode):
            _name_rules(result, relative, relative + "/x")
            continue
        if not stat.S_ISREG(mode):
            result.add(relative, "special file")
            continue
        result.files += 1
        _name_rules(result, relative, relative)
        size = path.stat().st_size
        if size == 0:
            continue
        try:
            with path.open("rb") as stream:
                _scan_stream(result, relative, path.name, stream, size, 0)
        except Exception as error:
            result.uninspected.append(f"{relative}: scan error ({type(error).__name__})")
    return result.as_dict()


if __name__ == "__main__":
    import json
    import sys
    print(json.dumps(scan_submission(sys.argv[1]), indent=2))
    raise SystemExit(0)
