"""Baseline quality acquisition with raw RGB outputs and retained completed intervals.

This is organizer acquisition, not a measurement of candidate serving throughput.
"""
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import time

from gimm_baseline import DATA_FOLDER, read_receipt, reuse_acquisition, sha256, write_json
from vfi_protocol import media_path, normalize_sample
from vfi_selection import select_intervals


def samples_at(root):
    path = Path(root) / "download" / DATA_FOLDER / "private/test.jsonl"
    receipt = json.loads((Path(root) / "prepared.json").read_text())
    samples = [normalize_sample(json.loads(line)) for line in path.read_text().splitlines()]
    return path, select_intervals(samples, receipt.get("dataset", {}).get("selection"))


def prepare_quality(root, previous, workers, *, selection=None):
    root, previous = Path(root), Path(previous)
    root.mkdir(parents=True, exist_ok=False)
    old_receipt = json.loads((previous / "prepared.json").read_text())
    receipt = reuse_acquisition(root, previous, old_receipt["requested_revisions"], selection=selection)
    _, samples = samples_at(root)
    source = previous / "predictions-1"
    rows = [json.loads(line) for line in (source / "predictions.jsonl").read_text().splitlines()]
    wanted = {sample["id"] for sample in samples}
    rows = [row for row in rows if row["id"] in wanted]
    if len(rows) > len(samples):
        raise ValueError("Previous export has more rows than the evaluation")
    output = root / "predictions-1"
    output.mkdir()
    jobs = []
    for ordinal, (sample, row) in enumerate(zip(samples, rows), 1):
        for key in ("id", "schema_version", "width", "height", "frame_count", "target_indices"):
            if row[key] != sample[key]:
                raise ValueError("Previous export is not a prefix of this evaluation")
        if len(row["frames"]) != sample["frame_count"]:
            raise ValueError("Previous interval is incomplete")
        for frame in row["frames"]:
            original = media_path(source, frame["path"])
            relative = Path(frame["path"])
            if relative.is_absolute() or ".." in relative.parts:
                raise ValueError("Unsafe previous frame path")
            # Retained full-set ordinals can collide with new subset ordinals.
            # Put each retained frame in its selected interval's new directory.
            relative = Path(f"{ordinal:06d}") / f"{frame['index']:06d}{relative.suffix}"
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            jobs.append((original, destination, frame["sha256"]))
            frame["path"] = relative.as_posix()

    def copy_verified(job):
        original, destination, expected = job
        shutil.copyfile(original, destination)
        if sha256(destination) != expected:
            raise ValueError("Previously completed prediction checksum changed")

    with ThreadPoolExecutor(max_workers=16) as pool:
        list(pool.map(copy_verified, jobs))
    receipt["quality_acquisition"] = {
        "gpu_workers": workers, "previous_run": previous.name, "retained_intervals": len(rows),
        "previous_prediction_manifest_sha256": sha256(source / "predictions.jsonl"),
        "output_pixels": "native RGB uint8; clip, multiply by 255, truncate",
        "new_output_storage": "raw RGB; no image encoding", "serving_timing": False,
        "code_sha256": {name: sha256(Path(__file__).parent / name)
                        for name in ("vfi_quality.py", "vfi_quality_modal.py", "gimm_server.py")}}
    # The receipt is finalized before any workers are launched.
    (root / "prepared.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
    write_json(root / "retained.json", rows)
    print(json.dumps({"retained_intervals": len(rows), "remaining_intervals": len(samples) - len(rows),
                      "gpu_workers": workers}), flush=True)
    return {"retained": len(rows), "total": len(samples)}


def predict_shard(root, index, workers, retained):
    import numpy as np
    from PIL import Image
    import torch
    from gimm_server import GIMM, quantize_prediction

    root = Path(root)
    receipt = read_receipt(root)
    source = root / "gimm"
    if subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip() != receipt["source"]["revision"]:
        raise ValueError("Upstream model revision changed")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True):
        raise ValueError("Upstream model source was modified")
    for name, expected in receipt["model"]["weights_sha256"].items():
        if sha256(source / "pretrained_ckpt" / name) != expected:
            raise ValueError("Model weights changed")
    if torch.cuda.device_count() != 1 or "H100" not in torch.cuda.get_device_name():
        raise ValueError("Each quality worker requires one H100")
    hardware = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda}
    data, samples = samples_at(root)
    assigned = list(range(retained + index, len(samples), workers))
    started = time.monotonic()
    model = GIMM(source)
    startup = time.monotonic() - started
    output = root / "predictions-1"
    records = []
    times = {"read_seconds": 0., "model_seconds": 0., "quantize_seconds": 0., "write_seconds": 0.}
    generated = 0
    for ordinal in assigned:
        sample = samples[ordinal]
        t = time.monotonic()
        input_bytes, images = {}, []
        for frame in sample["input_frames"]:
            path = media_path(data.resolve().parent, frame["path"])
            input_bytes[frame["index"]] = path.read_bytes()
            with Image.open(path) as image:
                images.append(np.array(image, dtype=np.uint8, copy=True))
        times["read_seconds"] += time.monotonic() - t
        t = time.monotonic()
        predictions = model.predict(images, sample["target_indices"], sample["frame_count"] - 1)
        elapsed = time.monotonic() - t
        times["model_seconds"] += elapsed
        t = time.monotonic()
        frames = {i: quantize_prediction(image).tobytes()
                  for i, image in zip(sample["target_indices"], predictions, strict=True)}
        times["quantize_seconds"] += time.monotonic() - t
        frames.update(input_bytes)
        t = time.monotonic()
        sequence = output / f"{ordinal + 1:06d}"
        sequence.mkdir()
        record = {key: sample[key] for key in ("id", "schema_version", "width", "height", "frame_count", "target_indices")}
        record.update(input_indices=[0, sample["frame_count"] - 1],
                      interpolation_factor=sample["frame_count"] - 1, model_seconds=elapsed, frames=[])
        for i in range(sample["frame_count"]):
            raw = i in sample["target_indices"]
            destination = sequence / f"{i:06d}{'.rgb' if raw else '.png'}"
            destination.write_bytes(frames[i])
            record["frames"].append({"index": i, "path": destination.relative_to(output).as_posix(),
                                      "generated": raw, "sha256": hashlib.sha256(frames[i]).hexdigest()})
        records.append(record)
        generated += len(sample["target_indices"])
        times["write_seconds"] += time.monotonic() - t
        (root / f"quality-progress-{index}.json").write_text(json.dumps({
            "worker": index, "completed": len(records), "total": len(assigned), "generated_frames": generated,
            "timing": times}) + "\n")
        del frames, predictions, images, input_bytes
    result = {"worker": index, "assigned_ordinals": assigned, "records": records, "hardware": hardware,
              "generated_frames": generated, "startup_seconds": startup,
              "elapsed_seconds": time.monotonic() - started, "timing": times}
    write_json(root / f"quality-shard-{index}.json", result)
    return result


def assemble(root, shards, started):
    root = Path(root)
    _, samples = samples_at(root)
    retained = json.loads((root / "retained.json").read_text())
    indexed = {i: r for i, r in enumerate(retained)}
    if {s["worker"] for s in shards} != set(range(len(shards))):
        raise ValueError("Missing or duplicate GPU shards")
    for shard in shards:
        for index, row in zip(shard["assigned_ordinals"], shard["records"], strict=True):
            if index in indexed or row["id"] != samples[index]["id"]:
                raise ValueError("Duplicate or incorrect predicted interval")
            indexed[index] = row
    if set(indexed) != set(range(len(samples))):
        raise ValueError("Incomplete parallel prediction export")
    output = root / "predictions-1"
    with (output / "predictions.jsonl").open("x") as stream:
        for i in range(len(samples)):
            stream.write(json.dumps(indexed[i], allow_nan=False) + "\n")
    write_json(output / "run.json", {"status": "complete", "samples_completed": len(samples),
                                     "samples_expected": len(samples), "predictions_file": "predictions.jsonl"})
    report = {"status": "complete", "samples": len(samples), "retained_intervals": len(retained),
              "mode": "organizer_parallel_quality", "gpu_workers": len(shards),
              "wall_seconds": time.monotonic() - started, "serving_timing_calibrated": False,
              "workers": [{k: v for k, v in shard.items() if k not in ("records", "assigned_ordinals")}
                          for shard in shards]}
    write_json(root / "pass-1.json", report)
    return report
