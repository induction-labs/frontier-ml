"""Organizer baseline acquisition, prediction and scoring; remote entry in vfi_modal.py."""
import hashlib
import json
import math
import os
from pathlib import Path
import re
import statistics
import subprocess
import sys
import time

TASK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TASK))
from vfi_selection import DEFAULT_SELECTION
DATA_REPO = "inductionlabs/frontier-ml-tasks"
DATA_FOLDER = "video-frame-interpolation"
MODEL_REPO = "GSean/GIMM-VFI"
GIMM_REPO = "https://github.com/GSeanCDAT/GIMM-VFI.git"
LPIPS_REPO = "https://github.com/richzhang/PerceptualSimilarity.git"
WEIGHTS = ("gimmvfi_r_arb_lpips.pt", "raft-things.pth")


def sha256(path):
    # The original model uses Python 3.10 (upstream dataclass defaults need it).
    result = hashlib.sha256()
    with Path(path).open("rb") as stream:
        while block := stream.read(1024 * 1024):
            result.update(block)
    return result.hexdigest()


def write_json(path, value):
    path = Path(path)
    with path.open("x") as stream:
        stream.write(json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def implementation_identity():
    from frontier_ml.validation import http

    paths = [TASK / "pipeline" / name for name in ("gimm_server.py", "gimm_baseline.py", "vfi_modal.py")]
    paths += [TASK / name for name in ("validate.py", "vfi_protocol.py", "vfi_selection.py")]
    return {**{path.name: sha256(path) for path in paths}, "http.py": sha256(http.__file__)}


def read_receipt(root):
    receipt = json.loads((Path(root) / "prepared.json").read_text())
    if receipt["implementation_sha256"] != implementation_identity():
        raise ValueError("Baseline runner changed since acquisition; use a new run")
    return receipt


def full_commit(value):
    if not isinstance(value, str) or not re.fullmatch("[0-9a-f]{40}", value):
        raise ValueError("Expected an immutable 40-character commit")
    return value


def git_commit(repo, revision):
    if re.fullmatch("[0-9a-f]{40}", revision):
        return revision
    if not re.fullmatch("[A-Za-z0-9_./-]+", revision) or revision.startswith("-"):
        raise ValueError("Invalid Git revision")
    lines = subprocess.check_output(["git", "ls-remote", repo, f"refs/heads/{revision}"], text=True).splitlines()
    if len(lines) != 1:
        raise ValueError(f"Cannot resolve unique branch: {repo} {revision}")
    return full_commit(lines[0].split()[0])


def reuse_acquisition(root, previous, revisions, *, selection=None):
    """Reuse acquired bytes after rehashing private media; never reuse predictions."""
    from vfi_evaluate import evaluation_identity
    from vfi_metrics import bundle_files

    previous = Path(previous).resolve()
    receipt = json.loads((previous / "prepared.json").read_text())
    if receipt["requested_revisions"] != revisions:
        raise ValueError("Reused acquisition revisions differ")
    data = previous / "download" / DATA_FOLDER / "private/test.jsonl"
    _, identity = evaluation_identity(data, selection=receipt["dataset"].get("selection"))
    if identity["evaluation_sha256"] != receipt["dataset"]["evaluation_sha256"]:
        raise ValueError("Reused private evaluation bytes changed")
    bundle_files(previous / "metrics", expected_sha256=receipt["metric_bundle_sha256"])
    for name in ("download", "gimm", "metrics"):
        (root / name).symlink_to(previous / name, target_is_directory=True)
    receipt["acquisition_reused_from"] = {"run_id": previous.name,
                                         "receipt_sha256": sha256(previous / "prepared.json")}
    receipt["implementation_sha256"] = implementation_identity()
    samples, identity = evaluation_identity(data, selection=selection)
    receipt["dataset"].update(selection=selection, evaluation_sha256=identity["evaluation_sha256"],
                              expected_samples=len(samples),
                              generated_frames=sum(len(s["target_indices"]) for s in samples))
    receipt.pop("quality_acquisition", None)
    write_json(root / "prepared.json", receipt)
    return receipt


def prepare(root, revisions, acquisition_root=None, *, selection=DEFAULT_SELECTION):
    """CPU only; verify the actual HF payload before allocating any GPU."""
    from huggingface_hub import HfApi, snapshot_download, hf_hub_download
    from vfi_data import verify
    from vfi_evaluate import evaluation_identity
    from metric_bundle import main as metric_bundle

    root = Path(root)
    root.mkdir(parents=True, exist_ok=True)
    receipt_path = root / "prepared.json"
    if receipt_path.exists():
        receipt = read_receipt(root)
        if receipt["requested_revisions"] != revisions:
            raise ValueError("Resume revisions differ from the existing run")
        if receipt["dataset"].get("selection") != selection:
            raise ValueError("Resume interval selection differs from the existing run")
        _, identity = evaluation_identity(root / "download" / DATA_FOLDER / "private/test.jsonl", selection=selection)
        if identity["evaluation_sha256"] != receipt["dataset"]["evaluation_sha256"]:
            raise ValueError("Evaluation changed since acquisition")
        from vfi_metrics import bundle_files
        bundle_files(root / "metrics", expected_sha256=receipt["metric_bundle_sha256"])
        return receipt
    if acquisition_root is not None:
        return reuse_acquisition(root, acquisition_root, revisions, selection=selection)
    token = os.environ.get("HF_TOKEN")
    api = HfApi(token=token)
    data_revision = full_commit(api.dataset_info(DATA_REPO, revision=revisions["data"]).sha)
    model_revision = full_commit(api.model_info(MODEL_REPO, revision=revisions["model"]).sha)
    source_revision = git_commit(GIMM_REPO, revisions["source"])
    lpips_revision = git_commit(LPIPS_REPO, revisions["lpips"])
    # List first: a missing/incompatible upload fails without downloading media.
    inventory = list(api.list_repo_tree(DATA_REPO, path_in_repo=DATA_FOLDER,
                                       repo_type="dataset", revision=data_revision, recursive=True))
    names = {entry.path for entry in inventory}
    required = {f"{DATA_FOLDER}/{name}" for name in
                ("manifest.json", "public/validation.jsonl", "private/test.jsonl",
                 "private/provenance/groups.json")}
    if not required <= names:
        raise ValueError(f"Incompatible HF layout; missing {sorted(required - names)}")
    print(json.dumps({"data_revision": data_revision, "files": len(names),
                      "bytes": sum(getattr(entry, "size", 0) or 0 for entry in inventory)}), flush=True)
    download = root / "download"
    snapshot_download(DATA_REPO, repo_type="dataset", revision=data_revision, token=token,
                      allow_patterns=[f"{DATA_FOLDER}/**"], local_dir=download, max_workers=8)
    data = download / DATA_FOLDER
    checked = verify(data)
    samples, identity = evaluation_identity(data / "private/test.jsonl", selection=selection)
    source = root / "gimm"
    source.mkdir()
    subprocess.run(["git", "init", str(source)], check=True)
    subprocess.run(["git", "-C", str(source), "fetch", "--depth=1", GIMM_REPO, source_revision], check=True)
    subprocess.run(["git", "-C", str(source), "checkout", "--detach", source_revision], check=True)
    weights = {}
    for filename in WEIGHTS:
        path = hf_hub_download(MODEL_REPO, filename, revision=model_revision,
                               local_dir=source / "pretrained_ckpt", token=token)
        weights[filename] = sha256(path)
    metric_bundle(["fetch", "--lpips-revision", lpips_revision, "--output", str(root / "metrics")])
    # Catch scorer checkpoint incompatibility before paying for baseline inference.
    from vfi_metrics import MetricEngine
    from PIL import Image
    engine = MetricEngine(root / "metrics", expected_sha256=sha256(root / "metrics/lock.json"))
    control_dir = root / "metric-control"
    control_dir.mkdir()
    controls = []
    for i in range(5):
        path = control_dir / f"{i}.png"
        Image.new("RGB", (64, 64), (i * 40, i * 20, 40)).save(path)
        controls.append(path)
    control = engine.interval(controls, controls, 64, 64)
    if control != {"lpips": 0.0, "flolpips": 0.0, "psnr": 100.0}:
        raise ValueError(f"Pretrained exact-reference metric control failed: {control}")
    receipt = {"baseline_id": "GIMM-VFI-R-P", "requested_revisions": revisions,
               "implementation_sha256": implementation_identity(),
               "dataset": {"repo": DATA_REPO, "revision": data_revision, "path": DATA_FOLDER,
                           "expected_samples": len(samples), "evaluation_sha256": identity["evaluation_sha256"],
                           "selection": selection,
                           "generated_frames": sum(len(s["target_indices"]) for s in samples),
                           "verification": checked},
               "model": {"repo": MODEL_REPO, "revision": model_revision, "weights_sha256": weights},
               "source": {"repo": GIMM_REPO, "revision": source_revision},
               "lpips_revision": lpips_revision,
               "metric_bundle_sha256": sha256(root / "metrics/lock.json"),
               "pretrained_exact_reference_control": control,
               "inference": {"precision": "float32", "tf32": False, "tta": False,
                             "downsampling": False, "seed": 0, "png_quantization": "clip_multiply_255_truncate"}}
    write_json(receipt_path, receipt)
    return receipt


def timing_summary(passes):
    if len(passes) not in (1, 3) or any(p.get("status") != "complete" for p in passes):
        raise ValueError("Timing needs one quality pass or three complete timing passes")
    timings = [p["details"]["timing"] for p in passes]
    counts = {t["generated_frames"] for t in timings}
    if len(counts) != 1 or next(iter(counts)) <= 0:
        raise ValueError("Timing passes must generate the same nonzero frame count")
    for row in timings:
        if any(type(row[k]) not in (int, float) or not math.isfinite(row[k]) or row[k] <= 0
               for k in ("request_seconds", "elapsed_seconds")):
            raise ValueError("Invalid measured timing")
    request = statistics.median(t["request_seconds"] for t in timings)
    elapsed = statistics.median(t["elapsed_seconds"] for t in timings)
    return {"passes": len(passes), "generated_frames": next(iter(counts)),
            "median_request_seconds": request, "median_export_pass_seconds": elapsed,
            "proposed_request_budget_seconds": 2 * request if len(passes) == 3 else None,
            "timing_scope": "sequential trusted loopback HTTP; includes endpoint decoding, inference and raw RGB transfer; "
                            "excludes dataset reads, export writes, model startup and quality metrics",
            "production_gate_activated": False}


def generate(root, passes=1):
    """One H100; one quality pass by default, optionally three for timing calibration."""
    from frontier_ml.validation.http import HTTPTransport
    # Import the intended task module explicitly, not another task's validate.py.
    import importlib.util
    spec = importlib.util.spec_from_file_location("vfi_baseline_validate", TASK / "validate.py")
    public = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(public)
    import torch

    root = Path(root)
    result_path = root / "generation.json"
    receipt = read_receipt(root)
    if result_path.exists():
        return json.loads(result_path.read_text())
    source = root / "gimm"
    actual_commit = subprocess.check_output(["git", "-C", str(source), "rev-parse", "HEAD"], text=True).strip()
    if actual_commit != receipt["source"]["revision"]:
        raise ValueError("Baseline source revision changed")
    if subprocess.check_output(["git", "-C", str(source), "status", "--porcelain", "--untracked-files=no"], text=True):
        raise ValueError("Baseline upstream source was modified")
    for filename, expected in receipt["model"]["weights_sha256"].items():
        if sha256(source / "pretrained_ckpt" / filename) != expected:
            raise ValueError("Baseline weight checksum mismatch")
    hardware = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__,
                "cuda": torch.version.cuda, "cpu_limit": 8, "memory_gib_limit": 80,
                "python": sys.version, "driver": subprocess.check_output(
                    ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], text=True).strip()}
    hardware["installed_packages"] = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True).splitlines()
    if "H100" not in hardware["gpu"] or torch.cuda.device_count() != 1:
        raise RuntimeError("Baseline requires exactly one H100")
    transport = HTTPTransport("http://127.0.0.1:8123")
    reports = []
    if type(passes) is not int or passes not in (1, 3):
        raise ValueError("Expected one quality pass or three timing passes")
    for index in range(passes):
        # A new server per pass avoids systematically warming later passes with test images.
        with (root / f"server-{index + 1}.log").open("x") as log:
            started = time.monotonic()
            process = subprocess.Popen([sys.executable, str(TASK / "pipeline/gimm_server.py"),
                                        "--source", str(source)], stdout=log, stderr=subprocess.STDOUT)
            try:
                while True:
                    if process.poll() is not None:
                        raise RuntimeError(f"Baseline server exited; see server-{index + 1}.log")
                    if time.monotonic() - started > 300:
                        raise TimeoutError("Baseline server was not ready within 300 seconds")
                    try:
                        if transport.request("GET", "/health", timeout=1).get("ready") is True:
                            break
                    except (OSError, RuntimeError):
                        pass
                    time.sleep(.25)
                ready_seconds = time.monotonic() - started
                report = public.run(transport, root / "download" / DATA_FOLDER / "private/test.jsonl",
                                    {"validation_mode": True, "predictions_only": True,
                                     "response_format": "rgb8",
                                     "evaluation_selection": receipt["dataset"].get("selection"),
                                     "concurrency": 1, "predictions_dir": str(root / f"predictions-{index + 1}"),
                                     "inference_timeout_seconds": min(3600, receipt["dataset"]["generated_frames"]),
                                     "request_timeout_seconds": 300,
                                     "progress_path": str(root / "progress.json")})
                report["startup_seconds"] = ready_seconds
                write_json(root / f"pass-{index + 1}.json", report)
                reports.append(report)
                print(json.dumps({"pass": index + 1, "timing": report["details"]["timing"]}), flush=True)
            finally:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
    result = {"hardware": hardware, "passes": reports, "timing": timing_summary(reports)}
    write_json(result_path, result)
    return result


def measure(root):
    """CPU-only scoring after H100 shutdown; emits measured calibration, never activates it."""
    from vfi_evaluate import reference_calibration as calibration, evaluation_identity, main as evaluate

    root = Path(root)
    receipt = read_receipt(root)
    generation = json.loads((root / "generation.json").read_text())
    timing_summary(generation["passes"])
    data = root / "download" / DATA_FOLDER / "private/test.jsonl"
    selection = receipt["dataset"].get("selection")
    _, identity = evaluation_identity(data, selection=selection)
    if identity["evaluation_sha256"] != receipt["dataset"]["evaluation_sha256"]:
        raise ValueError("Evaluation changed after preparation")
    if sha256(root / "metrics/lock.json") != receipt["metric_bundle_sha256"]:
        raise ValueError("Metric bundle changed after preparation")
    report_path, calibration_path = root / "metrics.json", root / "calibration.json"
    if not report_path.exists() and not calibration_path.exists():
        selection_path = root / "selection.json"
        if not selection_path.exists():
            write_json(selection_path, selection)
        evaluate(["--data", str(data), "--predictions-dir", str(root / "predictions-1"),
                  "--selection-json", str(selection_path),
                  "--metric-bundle", str(root / "metrics"), "--device", "cpu",
                  "--timeout-seconds", "36000",
                  "--output", str(report_path), "--write-calibration", str(calibration_path),
                  "--baseline-id", receipt["baseline_id"], "--baseline-revision", sha256(root / "prepared.json")])
    if not report_path.exists() or not calibration_path.exists():
        raise ValueError("Interrupted scoring left partial reports; inspect before rerunning")
    calibrated = calibration({"calibration_path": str(calibration_path),
                              "calibration_sha256": sha256(calibration_path),
                              "metric_bundle_sha256": receipt["metric_bundle_sha256"]}, identity)
    if (calibrated["baseline_revision"] != sha256(root / "prepared.json")
            or calibrated["prediction_manifest_sha256"] != sha256(root / "predictions-1/predictions.jsonl")):
        raise ValueError("Calibration model provenance or predictions changed")
    # Return exact bytes for the locks/calibration, whose formatting is hash-bound.
    files = {name: (root / name).read_text() for name in
             ("prepared.json", "generation.json", "metrics.json", "calibration.json", "metrics/lock.json")}
    return files
