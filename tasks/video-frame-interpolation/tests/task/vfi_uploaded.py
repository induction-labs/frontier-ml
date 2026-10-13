"""Verify the uploaded handoff payload without rewriting its interval-level split."""
import hashlib
import json
from pathlib import Path

from PIL import Image

from frontier_ml.validation.evaluation import load_samples
from vfi_protocol import media_path, normalize_sample, validate_png


def verify_uploaded(output, manifest):
    from vfi_data import file_record, read_intervals

    root = Path(output).resolve()
    if (manifest.get("schema_version") != 1 or manifest.get("task") != "video-frame-interpolation"
            or manifest.get("benchmark") != "SNU-FILM-arb"):
        raise ValueError("Unsupported uploaded manifest")
    files = manifest["files"]
    actual = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise ValueError("Symlink in uploaded dataset")
        if path.is_file() and path != root / "manifest.json":
            actual.add(path.relative_to(root).as_posix())
    if actual != set(files):
        raise ValueError("Uploaded inventory differs from manifest")
    for name, expected in files.items():
        if file_record(media_path(root, name)) != expected:
            raise ValueError(f"Uploaded checksum/size mismatch: {name}")
    official, _ = read_intervals(root / "private/provenance")
    schedules = {(row["factor"], row["source_line"]): row for row in official}
    source_intervals, source_frames = manifest["source_intervals"], manifest["source_frames"]
    if len(source_intervals) != len(schedules):
        raise ValueError("Uploaded intervals do not cover all official rows")
    seen, counts, groups, used_frames, targets = set(), {}, {}, set(), 0
    for split, filename in (("public", "validation.jsonl"), ("private", "test.jsonl")):
        rows = [normalize_sample(row) for row in load_samples(root / split / filename)]
        counts[split] = {str(f): 0 for f in (4, 8, 16)}
        groups[split] = set()
        for row in rows:
            source = source_intervals[row["id"]]
            factor = row["frame_count"] - 1
            key = (factor, source["line"])
            reference = schedules[key]
            if (key in seen or source["id"] != row["id"] or source["factor"] != factor
                    or source["split"] != split or source["frames"] != reference["source_frames"]
                    or source["list"] != reference["source_list"]
                    or source["sequence"] != reference["source_sequence"]):
                raise ValueError("Uploaded source interval differs from official schedule")
            seen.add(key)
            mapping = {frame["index"]: f"{split}/{frame['path']}"
                       for frame in row["input_frames"] + row["reference_frames"]}
            for index, name in enumerate(source["frames"]):
                frame = source_frames[name]
                if (frame["path"] != mapping[index] or frame["sequence"] != source["sequence"]
                        or frame["group"] != source["group"]
                        or (frame["width"], frame["height"]) != (row["width"], row["height"])):
                    raise ValueError("Uploaded media mapping differs from its row")
                used_frames.add(name)
            counts[split][str(factor)] += 1
            groups[split].add(source["group"])
            targets += factor - 1
    if seen != set(schedules) or used_frames != set(source_frames):
        raise ValueError("Missing or unreferenced uploaded source frames/intervals")
    frame_paths = [frame["path"] for frame in source_frames.values()]
    if len(set(frame_paths)) != len(frame_paths) or set(frame_paths) != {p for p in actual if p.endswith(".png")}:
        raise ValueError("Aliased or unreferenced PNGs")
    if (counts != manifest["by_factor"] or any(not n for split in counts.values() for n in split.values())
            or manifest["splits"] != {"validation": sum(counts["public"].values()), "heldout": sum(counts["private"].values())}
            or targets != manifest["target_predictions"] or len(source_frames) != manifest["unique_pngs"]):
        raise ValueError("Uploaded counts differ from manifest")
    # Recompute pixel identities rather than assume the handoff's undocumented RGB-hash encoding.
    rgb = {"public": set(), "private": set()}
    for frame in source_frames.values():
        path = media_path(root, frame["path"])
        validate_png(path.read_bytes(), frame["width"], frame["height"])
        with Image.open(path) as image:
            fingerprint = hashlib.sha256(f"{image.width}x{image.height}:".encode() + image.tobytes()).hexdigest()
        rgb[frame["path"].split("/", 1)[0]].add(fingerprint)
    if rgb["public"] & rgb["private"]:
        raise ValueError("Exact RGB overlap between public and private uploads")
    return {"status": "verified_uploaded_bytes_and_schedules", "samples": len(seen),
            "splits": manifest["splits"], "by_factor": counts, "unique_frames": len(source_frames),
            "exact_rgb_overlap": 0, "recording_groups_shared": len(groups["public"] & groups["private"]),
            "recording_isolation": not bool(groups["public"] & groups["private"]),
            "publication_ready": False, "scoring": "not_configured"}
