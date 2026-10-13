"""Stage trusted scoring inputs and run a separate, short-lived Modal H100."""
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
import json
import math
import os
from pathlib import Path
import shutil
import tarfile
import tempfile
import time

from frontier_ml.validation.errors import EvaluationError
from vfi_evaluate import digest
from vfi_protocol import media_path
from vfi_runtime import CUDA_RUNTIME
from vfi_scoring import PROTOCOL, aggregate

TASK_FILES = ("vfi_gpu_worker.py", "vfi_runtime.py", "vfi_evaluate.py", "vfi_metrics.py",
              "vfi_pwc.py", "vfi_protocol.py", "vfi_scoring.py", "vfi_selection.py")
LIBRARY_FILES = ("__init__.py", "validation/__init__.py", "validation/evaluation.py",
                 "validation/errors.py", "validation/progress.py")


def upload_archive(resource, archive, *, part_size=64 * 1024**2, workers=8, timeout_seconds=1200):
    """Bound in-flight disk/memory use and avoid one multi-GB sequential SDK stream."""
    archive = Path(archive)
    expected_hash = digest(archive)
    size = archive.stat().st_size
    if size <= part_size:
        resource.upload(archive, "/tmp/vfi-inputs.tar")
    else:
        deadline = time.monotonic() + timeout_seconds
        completed = 0
        with tempfile.TemporaryDirectory(prefix="vfi-transfer-", dir=archive.parent) as directory:
            pending = {}
            with ThreadPoolExecutor(max_workers=workers) as pool, archive.open("rb") as stream:
                def drain():
                    nonlocal completed
                    ready, _ = wait(pending, timeout=max(0, deadline - time.monotonic()),
                                    return_when=FIRST_COMPLETED)
                    if not ready:
                        raise TimeoutError("VFI scoring archive transfer exceeded its deadline")
                    for future in ready:
                        part = pending.pop(future)
                        future.result()
                        part.unlink()
                        completed += 1
                    resource.annotate(transfer_parts_completed=completed,
                                      transfer_parts_total=math.ceil(size / part_size), transfer_bytes=size)

                try:
                    index = 0
                    while chunk := stream.read(part_size):
                        if time.monotonic() >= deadline:
                            raise TimeoutError("VFI scoring archive transfer exceeded its deadline")
                        part = Path(directory) / f"part-{index:06d}.bin"
                        part.write_bytes(chunk)
                        future = pool.submit(resource.upload, part, "/tmp/vfi-parts/" + part.name)
                        pending[future] = part
                        index += 1
                        if len(pending) >= workers:
                            drain()
                    while pending:
                        drain()
                finally:
                    for future in pending:
                        future.cancel()
        resource.exec("cat /tmp/vfi-parts/part-*.bin > /tmp/vfi-inputs.tar", timeout=300)
    # Check the bytes before extraction, including the single-part path.
    resource.exec(f"printf '%s\\n' '{expected_hash}  /tmp/vfi-inputs.tar' | sha256sum -c "
                  "&& rm -rf /tmp/vfi-parts", timeout=300)


def pack_inputs(destination, data_path, predictions, config, samples):
    """One uncompressed archive; only selected media and enumerated files.

    Extraction stages everything onto GPU-local disk before any model inference,
    eliminating per-frame network filesystem reads and retaining exact pixel bytes.
    """
    import frontier_ml
    from vfi_metrics import bundle_files
    task = Path(__file__).parent
    data_path, predictions = Path(data_path).resolve(), Path(predictions).resolve()
    files = {"data/test.jsonl": data_path, "calibration.json": Path(config["calibration_path"]),
             "predictions/run.json": predictions / "run.json",
             "predictions/predictions.jsonl": predictions / "predictions.jsonl"}
    for sample in samples:
        for frame in sample["input_frames"] + sample["reference_frames"]:
            files["data/" + frame["path"]] = media_path(data_path.parent, frame["path"])
    for line in (predictions / "predictions.jsonl").read_text().splitlines():
        for frame in json.loads(line)["frames"]:
            files["predictions/" + frame["path"]] = media_path(predictions, frame["path"])
    bundle = Path(config["metric_bundle"]).resolve()
    for path in bundle_files(bundle, expected_sha256=config["metric_bundle_sha256"]).values():
        files["metrics/" + str(path.relative_to(bundle))] = path
    files["metrics/lock.json"] = bundle / "lock.json"
    files.update({"task/" + name: task / name for name in TASK_FILES})
    library = Path(frontier_ml.__file__).parent
    files.update({"task/frontier_ml/" + name: library / name for name in LIBRARY_FILES})
    with tarfile.open(destination, "w", dereference=True) as archive:
        for relative, source in files.items():
            # Never preserve filesystem links, traversal, or arbitrary candidate code.
            parts = Path(relative).parts
            if Path(relative).is_absolute() or ".." in parts or not source.is_file():
                raise ValueError("Invalid scoring transfer path")
            info = archive.gettarinfo(str(source.resolve()), arcname=relative)
            if not info.isfile():
                raise ValueError("Scoring transfer requires regular files")
            info.mode, info.uid, info.gid, info.uname, info.gname = 0o600, 0, 0, "", ""
            with source.open("rb") as stream:
                archive.addfile(info, stream)


def validate_report(report, config, samples, manifest_sha256):
    from vfi_metrics import code_sha256
    expected = {"protocol": PROTOCOL, "evaluation_sha256": config["evaluation_sha256"],
                "metric_bundle_sha256": config["metric_bundle_sha256"],
                "calibration_sha256": config["calibration_sha256"],
                "prediction_manifest_sha256": manifest_sha256,
                "metric_code_sha256": code_sha256(), "runtime": CUDA_RUNTIME}
    if any(report.get(key) != value for key, value in expected.items()):
        raise EvaluationError("H100 metric report provenance differs from the requested evaluation")
    rows = report.get("intervals", [])
    if [(r.get("id"), r.get("interpolation_factor")) for r in rows] != [
            (s["id"], s["frame_count"] - 1) for s in samples]:
        raise EvaluationError("H100 metric report has missing, reordered or incorrect intervals")
    if report.get("by_factor") != aggregate(rows):
        raise EvaluationError("H100 metric aggregates do not match interval results")


def score_on_h100(data_path, predictions, config, samples):
    # Import Harbor only inside the trusted verifier, never public validation/viewer loading.
    from frontier_ml.runtime.inference import compliant_resource
    started = time.monotonic()
    with tempfile.TemporaryDirectory(prefix="frontier-vfi-score-") as directory:
        root = Path(directory)
        manifest_hash = digest(Path(predictions) / "predictions.jsonl")
        archive = root / "inputs.tar"
        pack_inputs(archive, data_path, predictions, config, samples)
        packaging_seconds = time.monotonic() - started
        request = {key: config[key] for key in (
            "evaluation_selection", "evaluation_sha256", "calibration_sha256",
            "metric_bundle_sha256", "metric_runtime", "metric_timeout_seconds")}
        request["prediction_manifest_sha256"] = manifest_hash
        (root / "request.json").write_text(json.dumps(request, allow_nan=False))
        (root / "image").mkdir()
        shutil.copyfile(Path(__file__).parent / "scorer/Dockerfile.gpu", root / "image/Dockerfile")
        # Do not inherit the candidate image, credentials, volumes or model mount.
        own = {"grading_resources": {"inference": {"type": "modal"}},
               "grading_images": {"inference": str(root / "image")}}
        if region := os.environ.get("MODAL_REGION"):
            own["grading_resources"]["inference"]["kwargs"] = {"region": region}
        lifetime = math.ceil(config["metric_timeout_seconds"]) + 1800
        with compliant_resource(own, lifetime, None) as resource:
            if "H100" not in resource.gpu["name"].upper():
                raise EvaluationError("VFI scoring requires H100 hardware")
            resource.annotate(purpose="video-frame-interpolation metrics", scoring_stage="staging")
            staging_started = time.monotonic()
            upload_archive(resource, archive)
            resource.exec("mkdir -p /tmp/vfi && tar -xf /tmp/vfi-inputs.tar -C /tmp/vfi "
                          "--no-same-owner --no-same-permissions && rm /tmp/vfi-inputs.tar", timeout=1200)
            resource.upload(root / "request.json", "/tmp/vfi/request.json")
            staging_seconds = time.monotonic() - staging_started
            resource.annotate(scoring_stage="metrics", staging_seconds=staging_seconds)
            execution = resource.exec("cd /tmp/vfi/task && /opt/conda/bin/python vfi_gpu_worker.py /tmp/vfi",
                                      timeout=config["metric_timeout_seconds"], check=False)
            if execution.return_code:
                raise EvaluationError("H100 metric worker failed: " + (execution.stderr or "")[-2000:])
            resource.download("/tmp/vfi/result.json", root / "result.json")
            resource.annotate(scoring_stage="complete")
        report = json.loads((root / "result.json").read_text())
        validate_report(report, config, samples, manifest_hash)
        report["scoring_seconds"] = time.monotonic() - started  # packaging, staging and teardown included
        report.update(packaging_seconds=packaging_seconds, staging_seconds=staging_seconds)
        return report
