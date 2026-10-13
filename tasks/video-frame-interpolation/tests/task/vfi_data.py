"""Organizer-only preparation of the official SNU-FILM-arb endpoint intervals.

No metrics, model execution, automatic extraction, or publication. See pipeline/README.md.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
from io import BytesIO
import json
from pathlib import Path, PurePosixPath
import re
import shutil
import tempfile
from urllib.parse import urlparse
from urllib.request import urlopen

from PIL import Image, __version__ as pillow_version

from vfi_protocol import VERSION, media_path, validate_png, validate_sample

SOURCES_PATH = Path(__file__).with_name("pipeline") / "sources.json"
SOURCES = json.loads(SOURCES_PATH.read_text())
GIB = 1024**3
RESERVE = 512 * 1024**2


def digest(blob):
    return hashlib.sha256(blob).hexdigest()


def file_record(path):
    with Path(path).open("rb") as stream:
        sha = hashlib.file_digest(stream, "sha256").hexdigest()
    return {"bytes": Path(path).stat().st_size, "sha256": sha}


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def utc_now():
    return datetime.now(timezone.utc).isoformat()


def download(url, output, max_bytes, expected_sha256=None):
    """Bound disk use and record the bytes actually obtained; never execute a download."""
    if urlparse(url).scheme != "https" or type(max_bytes) is not int or max_bytes <= 0:
        raise ValueError("A HTTPS URL and positive download size limit are required")
    output = Path(output)
    receipt = output.with_name(output.name + ".source.json")
    partial = output.with_name(output.name + ".partial")
    if any(p.exists() for p in (output, receipt, partial)):
        raise FileExistsError("Download, receipt, or partial file already exists")
    output.parent.mkdir(parents=True, exist_ok=True)
    sha, size = hashlib.sha256(), 0
    try:
        with urlopen(url, timeout=60) as response, partial.open("xb") as target:
            if urlparse(response.geturl()).scheme != "https":
                raise ValueError("Download redirected away from HTTPS")
            content_type = response.headers.get("Content-Type", "")
            if "text/html" in content_type:
                raise ValueError("Received a webpage; provide the direct archive download URL")
            length = response.headers.get("Content-Length")
            if length is not None and int(length) > min(max_bytes, shutil.disk_usage(output.parent).free - RESERVE):
                raise ValueError("Download exceeds size limit or available disk space")
            while block := response.read(1024**2):
                size += len(block)
                if size > max_bytes or shutil.disk_usage(output.parent).free < len(block) + RESERVE:
                    raise ValueError("Download exceeds size limit or available disk space")
                sha.update(block)
                target.write(block)
            if size == 0 or (length is not None and size != int(length)):
                raise ValueError("Empty or truncated download")
            if expected_sha256 is not None and sha.hexdigest() != expected_sha256:
                raise ValueError("Download SHA-256 mismatch")
            metadata = {"url": url, "resolved_url": response.geturl(), "retrieved_at": utc_now(),
                        "bytes": size, "sha256": sha.hexdigest(),
                        "expected_sha256": expected_sha256,
                        "etag": response.headers.get("ETag")}
        partial.rename(output)
        write_json(receipt, metadata)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return metadata


def fetch_protocol(output, revision="main"):
    """Pin and snapshot upstream reference code as provenance, never import it."""
    if not re.fullmatch(r"main|[0-9a-f]{40}", revision):
        raise ValueError("Protocol revision must be main or a full commit SHA")
    output = Path(output)
    if output.exists():
        raise FileExistsError(output)
    api = f"https://api.github.com/repos/GSeanCDAT/GIMM-VFI/commits/{revision}"
    with urlopen(api, timeout=60) as response:
        commit = json.loads(response.read(1024**2))["sha"]
    if not re.fullmatch(r"[0-9a-f]{40}", commit) or (revision != "main" and commit != revision):
        raise ValueError("Could not resolve an immutable upstream revision")
    output.mkdir(parents=True)
    files = {}
    for name in SOURCES["protocol"]["files"]:
        url = f"https://raw.githubusercontent.com/GSeanCDAT/GIMM-VFI/{commit}/{name}"
        files[name] = download(url, output / name, 2 * 1024**2)
    lock = {"repo": SOURCES["protocol"]["repo"], "revision": commit, "files": files,
            "retrieved_at": utc_now(), "code_executed": False}
    write_json(output / "protocol.json", lock)
    return lock


def read_protocol_lock(path):
    path = Path(path)
    lock = json.loads(path.read_text())
    if (lock.get("repo") != SOURCES["protocol"]["repo"]
            or not re.fullmatch(r"[0-9a-f]{40}", lock.get("revision", ""))
            or set(lock.get("files", {})) != set(SOURCES["protocol"]["files"])):
        raise ValueError("Protocol snapshot needs the upstream repo, full SHA, and all reference files")
    for name, expected in lock["files"].items():
        actual = file_record(media_path(path.parent, name))
        if any(actual[key] != expected[key] for key in ("bytes", "sha256")):
            raise ValueError(f"Protocol snapshot changed: {name}")
    return lock


def source_path(value):
    """Accept the reference loader's prefix or a path relative to the SNU root."""
    if value.startswith("./"):
        value = value[2:]
    if value.startswith("data/SNU-FILM/"):
        value = value[len("data/SNU-FILM/"):]
    parts = value.split("/")
    if (len(parts) != 4 or parts[0] != "test" or parts[1] not in ("GOPRO_test", "YouTube_test")
            or any(part in ("", ".", "..") for part in parts) or "\\" in value
            or not re.fullmatch(r"[0-9]+\.png", parts[-1])):
        raise ValueError(f"Invalid source frame path in official list: {value!r}")
    return PurePosixPath(value)


def read_intervals(lists_dir):
    """Expand precisely the official triplet rows, not arbitrary video windows."""
    intervals, lists, seen = [], {}, set()
    for split, spec in SOURCES["dataset"]["lists"].items():
        name, factor = spec["file"], spec["factor"]
        path = media_path(lists_dir, name)
        rows = path.read_text().splitlines()
        if len(rows) != spec["expected_rows"]:
            raise ValueError(f"{name}: expected {spec['expected_rows']} official rows, got {len(rows)}")
        lists[name] = file_record(path)
        for line, row in enumerate(rows, 1):
            tokens = row.split()
            if len(tokens) != 3:
                raise ValueError(f"{name}:{line}: expected exactly three frame paths")
            first, middle, last = [source_path(token) for token in tokens]
            start, midpoint, end = [int(p.stem) for p in (first, middle, last)]
            if (len({p.parent for p in (first, middle, last)}) != 1
                    or end - start != factor or midpoint - start != factor // 2):
                raise ValueError(f"{name}:{line}: endpoints/midpoint do not match {factor}x spacing")
            frames = [(first.parent / f"{index:0{len(first.stem)}d}.png").as_posix()
                      for index in range(start, end + 1)]
            if frames[factor // 2] != str(middle) or frames[-1] != str(last):
                raise ValueError(f"{name}:{line}: inconsistent numeric frame formatting")
            key = (str(first), str(last))
            if key in seen:
                raise ValueError(f"Duplicate endpoint pair in {name}:{line}")
            seen.add(key)
            intervals.append({"id": f"interval-{len(intervals) + 1:06d}", "factor": factor,
                              "source_split": split, "source_list": name, "source_line": line,
                              "source_sequence": first.parent.as_posix(), "source_frames": frames})
    return intervals, lists


def recording_key(folder):
    """Conservatively co-group numbered GoPro subclips of the same camera recording."""
    path = PurePosixPath(folder)
    camera = re.match(r"GOPR[0-9]+(?:_|$)", path.name) if path.parent.name == "GOPRO_test" else None
    return f"GOPRO_test/{camera[0].rstrip('_')}" if camera else folder


def make_sample(interval):
    factor = interval["factor"]
    def frame(index):
        return {"index": index, "path": interval["frame_paths"][index]}
    return validate_sample({"id": interval["id"], "schema_version": VERSION,
                           "width": interval["width"], "height": interval["height"],
                           "frame_count": factor + 1,
                           "input_frames": [frame(0), frame(factor)],
                           "target_indices": list(range(1, factor)),
                           "reference_frames": [frame(i) for i in range(1, factor)]})


def inspect_sources(snu_root, lists_dir):
    """Read official lists and inspect only the frames their intervals reference."""
    root = Path(snu_root).resolve()
    intervals, lists = read_intervals(lists_dir)
    folders = sorted({row["source_sequence"] for row in intervals})
    frames = {}
    for ordinal, folder in enumerate(folders, 1):
        relatives = sorted({frame for row in intervals if row["source_sequence"] == folder
                            for frame in row["source_frames"]}, key=lambda p: int(PurePosixPath(p).stem))
        for index, relative in enumerate(relatives):
            source = media_path(root, relative)
            with Image.open(source) as image:
                if image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"Source must be a single PNG frame: {source}")
                width, height = image.size
            frames[relative] = {"source_sequence": folder, "width": width, "height": height,
                                "source_bytes": source.stat().st_size,
                                "path": f"media/sequence-{ordinal:06d}/{index:06d}.png"}
    for interval in intervals:
        dimensions = {(frames[p]["width"], frames[p]["height"]) for p in interval["source_frames"]}
        if len(dimensions) != 1:
            raise ValueError(f"Dimensions change inside interval {interval['id']}")
        interval["width"], interval["height"] = dimensions.pop()
        interval["frame_paths"] = [frames[p]["path"] for p in interval["source_frames"]]
        make_sample(interval)  # Never resize, drop frames, or change intervals to fit transport bounds.
    raw_bytes = sum(f["width"] * f["height"] * 3 for f in frames.values())
    return {"schema_version": 2, "benchmark": "snu-film-arb", "intervals": intervals,
            "lists": lists, "frames": frames, "samples": len(intervals), "unique_frames": len(frames),
            "raw_rgb_bytes": raw_bytes, "estimated_output_upper_bytes": raw_bytes * 102 // 100 + len(frames) * 65536,
            "source_bytes": sum(f["source_bytes"] for f in frames.values()),
            "by_factor": {str(s["factor"]): s["expected_rows"] for s in SOURCES["dataset"]["lists"].values()},
            "group_map_template": {folder: recording_key(folder) for folder in folders},
            "status": "headers_checked; review group_map_template against original recording provenance before prepare"}


def assign_splits(folders, groups, validation_fraction, seed):
    if (not isinstance(groups, dict) or set(groups) != set(folders)
            or any(not isinstance(g, str) or not g.strip() for g in groups.values())):
        raise ValueError("Group map must assign every source sequence exactly once to an original recording")
    if type(validation_fraction) not in (int, float) or not 0 < validation_fraction < 1 or type(seed) is not int:
        raise ValueError("A validation fraction between zero and one and an integer seed are required")
    # An explicit map may merge more sources; it may not split known GoPro subclips.
    known = {}
    for folder, group in groups.items():
        known.setdefault(recording_key(folder), set()).add(group)
    if any(len(values) > 1 for values in known.values()):
        raise ValueError("Group map splits subclips of the same GoPro recording")
    ordered = sorted(set(groups.values()), key=lambda g: (digest(f"{seed}\0{g}".encode()), g))
    if len(ordered) < 2:
        raise ValueError("At least two original recording groups are needed for a public/private split")
    count = max(1, min(len(ordered) - 1, int(len(ordered) * validation_fraction + 0.5)))
    public = set(ordered[:count])
    return {group: "public" if group in public else "private" for group in ordered}


def split_audit(intervals, frames, groups):
    """Check declared recording groups, exact RGB overlap, and endpoint fingerprints."""
    group_sets, image_sets, input_sets = {}, {}, {}
    for split in ("public", "private"):
        rows = [row for row in intervals if row["split"] == split]
        group_sets[split] = {groups[row["source_sequence"]] for row in rows}
        image_sets[split] = {frame["rgb_sha256"] for frame in frames.values() if frame["split"] == split}
        input_sets[split] = {tuple(frames[row["source_frames"][i]]["rgb_sha256"] for i in (0, -1)) for row in rows}
    overlap = {"group_overlap": sorted(group_sets["public"] & group_sets["private"]),
               "image_overlap": sorted(image_sets["public"] & image_sets["private"]),
               "input_fingerprint_overlap": sorted(input_sets["public"] & input_sets["private"])}
    if any(overlap.values()):
        raise ValueError("Public/private recording, RGB frame, or endpoint overlap; merge related recording groups and prepare again")
    return {**overlap, "exact_checks_passed": True, "passed": False,
            "limitations": "Recording identities are operator-supplied; semantic recording verification and near-duplicate audit remain pending. "
            "These are public benchmark videos, not secret data. Exact checks do not detect re-encoded or cropped copies."}


def prepare(snu_root, lists_dir, output, protocol_snapshot, group_map, *, validation_fraction=0.2, seed=0):
    """Create native-resolution endpoint intervals, split only by original recording."""
    root, lists_dir, output = Path(snu_root).resolve(), Path(lists_dir).resolve(), Path(output).resolve()
    if output.exists() or any(output.is_relative_to(p) for p in (root, lists_dir, Path(protocol_snapshot).resolve().parent)):
        raise ValueError("Output must be a new directory outside the source trees")
    lock = read_protocol_lock(protocol_snapshot)
    inventory = inspect_sources(root, lists_dir)
    groups = json.loads(Path(group_map).read_text())
    assignments = assign_splits(inventory["group_map_template"], groups, validation_fraction, seed)
    intervals = inventory["intervals"]
    for interval in intervals:
        interval["split"] = assignments[groups[interval["source_sequence"]]]
    if any({row["factor"] for row in intervals if row["split"] == split} != {4, 8, 16}
           for split in ("public", "private")):
        raise ValueError("Both splits must contain all three interpolation factors; revise recording groups or split seed")
    output.parent.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(output.parent).free < inventory["estimated_output_upper_bytes"] + RESERVE:
        raise ValueError("Insufficient disk space for lossless intervals; use a larger volume")
    with tempfile.TemporaryDirectory(prefix=".vfi-prepare-", dir=output.parent) as temporary:
        stage = Path(temporary) / "payload"
        (stage / "private/provenance").mkdir(parents=True)
        files, frames = {}, {}
        for relative, frame in inventory["frames"].items():
            source = media_path(root, relative)
            blob = source.read_bytes()
            with Image.open(BytesIO(blob)) as image:
                image.load()
                if image.size != (frame["width"], frame["height"]) or image.format != "PNG" or getattr(image, "n_frames", 1) != 1:
                    raise ValueError(f"Source changed after inspection: {source}")
                rgb = image.convert("RGB")  # Same values as the reference loader; no resize/EXIF orientation.
                pixels = rgb.tobytes()
                rgb_hash = digest(f"{rgb.width}x{rgb.height}:".encode() + pixels)
                encoded = BytesIO()
                Image.frombytes("RGB", rgb.size, pixels).save(encoded, format="PNG", compress_level=6)
            png = validate_png(encoded.getvalue(), frame["width"], frame["height"])
            split = assignments[groups[frame["source_sequence"]]]
            neutral = f"{split}/{frame['path']}"
            destination = stage / neutral
            destination.parent.mkdir(parents=True, exist_ok=True)
            destination.write_bytes(png)
            files[neutral] = {"bytes": len(png), "sha256": digest(png)}
            frames[relative] = {**frame, "split": split, "source_sha256": digest(blob), "source_bytes": len(blob),
                                "rgb_sha256": rgb_hash, "output_path": neutral}
        audit = split_audit(intervals, frames, groups)
        splits, by_factor = {}, {}
        for split, filename, label in (("public", "validation.jsonl", "validation"), ("private", "test.jsonl", "heldout")):
            rows = [make_sample(row) for row in intervals if row["split"] == split]
            path = stage / split / filename
            path.write_text("".join(json.dumps(row, sort_keys=True) + "\n" for row in rows))
            files[f"{split}/{filename}"] = file_record(path)
            splits[label] = len(rows)
            by_factor[label] = {str(f): sum(row["frame_count"] == f + 1 for row in rows) for f in (4, 8, 16)}
        for name, expected in inventory["lists"].items():
            path = stage / "private/provenance" / name
            path.write_bytes(media_path(lists_dir, name).read_bytes())
            if file_record(path) != expected:
                raise ValueError("Official lists changed during preparation")
            files[f"private/provenance/{name}"] = expected
        write_json(stage / "private/provenance/groups.json", groups)
        files["private/provenance/groups.json"] = file_record(stage / "private/provenance/groups.json")
        manifest = {"schema_version": 2, "task": "video-frame-interpolation", "benchmark": "snu-film-arb",
                    "status": "prepared_unreleased", "prepared_at": utc_now(),
                    "source": {"selection": SOURCES, "selection_sha256": file_record(SOURCES_PATH)["sha256"],
                               "protocol_snapshot": lock, "pillow_version": pillow_version},
                    "split_config": {"validation_fraction": validation_fraction, "seed": seed,
                                     "unit": "original recording group", "algorithm": "sha256(seed + NUL + group), rounded group count"},
                    "groups": groups, "group_assignments": assignments, "splits": splits, "by_factor": by_factor,
                    "split_audit": audit, "request_context": "independent endpoint pair; no visual state across requests",
                    "transformations": ["official medium/hard/extreme interval selection", "Pillow RGB conversion", "metadata-free lossless PNG"],
                    "time": "normalized endpoint times 0 and 1; target i at i/factor; no capture/playback FPS assertion",
                    "intervals": intervals, "frames": frames, "files": files}
        write_json(stage / "manifest.json", manifest)
        verify(stage)
        stage.rename(output)
    return manifest


def verify(output):
    """Verify prepared files, official schedules, native RGB frames, and split separation."""
    output = Path(output).resolve()
    manifest = json.loads((output / "manifest.json").read_text())
    if manifest.get("schema_version") == 1 and manifest.get("benchmark") == "SNU-FILM-arb":
        from vfi_uploaded import verify_uploaded
        return verify_uploaded(output, manifest)
    if (manifest.get("schema_version") != 2 or manifest.get("task") != "video-frame-interpolation"
            or manifest.get("benchmark") != "snu-film-arb" or manifest.get("status") != "prepared_unreleased"):
        raise ValueError("Not a prepared SNU-FILM-arb payload")
    actual_paths = set()
    for path in output.rglob("*"):
        if path.is_symlink():
            raise ValueError(f"Symlink in prepared output: {path}")
        if path.is_file() and path != output / "manifest.json":
            actual_paths.add(path.relative_to(output).as_posix())
    if actual_paths != set(manifest["files"]):
        raise ValueError("Prepared file inventory differs from manifest")
    for name, expected in manifest["files"].items():
        if file_record(media_path(output, name)) != expected:
            raise ValueError(f"Prepared file changed: {name}")
    intervals, _ = read_intervals(output / "private/provenance")
    if len(intervals) != len(manifest["intervals"]):
        raise ValueError("Prepared intervals differ from official lists")
    frames, groups = manifest["frames"], manifest["groups"]
    if groups != json.loads((output / "private/provenance/groups.json").read_text()):
        raise ValueError("Recording group map changed")
    assignments = assign_splits({row["source_sequence"] for row in intervals}, groups,
                                manifest["split_config"]["validation_fraction"], manifest["split_config"]["seed"])
    if assignments != manifest["group_assignments"]:
        raise ValueError("Recording split assignments changed")
    referenced = set()
    for original, row in zip(intervals, manifest["intervals"], strict=True):
        if any(row.get(key) != value for key, value in original.items()):
            raise ValueError("Prepared interval differs from official list")
        split = assignments[groups[row["source_sequence"]]]
        if row["split"] != split or row["frame_paths"] != [frames[p]["path"] for p in row["source_frames"]]:
            raise ValueError("Prepared interval mapping or split changed")
        for source in row["source_frames"]:
            frame = frames[source]
            if (frame["split"] != split or frame["source_sequence"] != row["source_sequence"]
                    or (frame["width"], frame["height"]) != (row["width"], row["height"])
                    or frame["output_path"] != f"{split}/{frame['path']}"):
                raise ValueError("Prepared frame mapping changed")
            referenced.add(source)
    if referenced != set(frames) or len({f["output_path"] for f in frames.values()}) != len(frames):
        raise ValueError("Unreferenced or aliased source frames")
    splits, by_factor = {}, {}
    for split, filename, label in (("public", "validation.jsonl", "validation"), ("private", "test.jsonl", "heldout")):
        rows = [validate_sample(json.loads(line)) for line in (output / split / filename).read_text().splitlines()]
        expected = [make_sample(row) for row in manifest["intervals"] if row["split"] == split]
        if rows != expected or len({row["id"] for row in rows}) != len(rows):
            raise ValueError("Prepared rows differ from official endpoint schedules")
        splits[label] = len(rows)
        by_factor[label] = {str(f): sum(row["frame_count"] == f + 1 for row in rows) for f in (4, 8, 16)}
        if not all(by_factor[label].values()):
            raise ValueError("A split is missing an interpolation factor")
    if splits != manifest["splits"] or by_factor != manifest["by_factor"]:
        raise ValueError("Prepared counts differ from manifest")
    for frame in frames.values():
        path = media_path(output, frame["output_path"])
        validate_png(path.read_bytes(), frame["width"], frame["height"])
        with Image.open(path) as image:
            if digest(f"{image.width}x{image.height}:".encode() + image.tobytes()) != frame["rgb_sha256"]:
                raise ValueError("Prepared RGB values differ from manifest")
    expected_files = {f["output_path"] for f in frames.values()} | {
        "public/validation.jsonl", "private/test.jsonl", "private/provenance/groups.json",
        *(f"private/provenance/{s['file']}" for s in SOURCES["dataset"]["lists"].values())}
    if expected_files != actual_paths:
        raise ValueError("Unreferenced prepared files")
    audit = split_audit(manifest["intervals"], frames, groups)
    if manifest["split_audit"] != audit:
        raise ValueError("Split audit differs from checked data")
    return {"status": "verified_unreleased", "samples": sum(splits.values()), "unique_frames": len(frames),
            "splits": splits, "by_factor": by_factor, "scoring": "not_configured", "publication_ready": False}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    fetch = sub.add_parser("fetch", help="download all RGB frames with a size limit and hash receipt")
    fetch.add_argument("--url", help="override the direct HTTPS archive link")
    fetch.add_argument("--output", type=Path, required=True)
    fetch.add_argument("--max-gib", type=int, required=True)
    fetch.add_argument("--sha256", help="expected archive hash for a repeat fetch")
    pin = sub.add_parser("fetch-protocol", help="snapshot reference scripts at an immutable upstream commit")
    pin.add_argument("--output", type=Path, required=True)
    pin.add_argument("--revision", default="main")
    for name in ("inspect", "prepare"):
        command = sub.add_parser(name)
        command.add_argument("--snu-root", type=Path, required=True)
        command.add_argument("--lists-dir", type=Path, required=True)
        command.add_argument("--output", type=Path, required=True)
        if name == "prepare":
            command.add_argument("--protocol-snapshot", type=Path, required=True)
            command.add_argument("--group-map", type=Path, required=True, help="JSON mapping each source sequence to its original recording")
            command.add_argument("--validation-fraction", type=float, default=0.2)
            command.add_argument("--seed", type=int, default=0)
    check = sub.add_parser("verify")
    check.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    if args.command == "fetch":
        result = download(args.url or SOURCES["dataset"]["download_url"], args.output, args.max_gib * GIB, args.sha256)
    elif args.command == "fetch-protocol":
        result = fetch_protocol(args.output, args.revision)
    elif args.command == "verify":
        result = verify(args.output)
    elif args.command == "prepare":
        result = prepare(args.snu_root, args.lists_dir, args.output, args.protocol_snapshot, args.group_map,
                         validation_fraction=args.validation_fraction, seed=args.seed)
        result = {"status": result["status"], "splits": result["splits"], "by_factor": result["by_factor"], "output": str(args.output)}
    else:
        result = inspect_sources(args.snu_root, args.lists_dir)
        if args.output.exists():
            raise FileExistsError(args.output)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        write_json(args.output, result)
        result = {k: v for k, v in result.items() if k not in ("intervals", "frames")}
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
