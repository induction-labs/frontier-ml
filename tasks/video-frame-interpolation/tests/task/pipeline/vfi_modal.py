"""Run the organizer's frozen GIMM-VFI-R-P baseline on Modal; see BASELINE.md.

    uv run modal run tasks/video-frame-interpolation/pipeline/vfi_modal.py --out work/vfi/gimm-r-p-270-run

Only small reports return locally. Dataset, weights, logs and predictions stay on
the private Modal volume. No HF uploads, Git changes or production activation.
"""
import json
from pathlib import Path
import re
import sys
import time
import uuid

import modal

TASK = Path(__file__).resolve().parents[1]
volume = modal.Volume.from_name("frontier-vfi-baseline", create_if_missing=True)
app = modal.App("frontier-vfi-baseline")

cpu_image = (modal.Image.debian_slim(python_version="3.12")
             .apt_install("git")
             .pip_install("torch==2.8.0", index_url="https://download.pytorch.org/whl/cpu")
             .pip_install("numpy==2.2.6", "Pillow==11.3.0", "PyYAML==6.0.2", "huggingface_hub==0.35.3")
             .env({"HF_HOME": "/runs/cache/hf", "HF_HUB_DISABLE_TELEMETRY": "1",
                   "HF_HUB_DISABLE_XET": "1"})
             .add_local_dir(str(TASK), "/task", ignore=["**/__pycache__/**"])
             .add_local_python_source("frontier_ml"))

# Python 3.10 retains the upstream dataclass defaults; CuPy 11 retains
# cuda.compile_with_cache. CUDA 11.8 / the cu118 torch build support Hopper.
# This avoids modifying upstream model or CUDA kernel source.
gpu_image = (modal.Image.from_registry("nvidia/cuda:11.8.0-devel-ubuntu22.04", add_python="3.10")
             .env({"DEBIAN_FRONTEND": "noninteractive", "TZ": "Etc/UTC"})
             .apt_install("git", "libgl1", "libglib2.0-0", "tk")
             .pip_install("torch==2.5.1+cu118", "torchvision==0.20.1+cu118",
                          extra_index_url="https://download.pytorch.org/whl/cu118")
             .pip_install("numpy==1.26.4", "Pillow==11.3.0", "cupy-cuda11x==11.6.0",
                          "omegaconf==2.3.0", "einops==0.6.1", "timm==0.6.13",
                          "easydict==1.13", "yacs==0.1.8", "PyYAML==6.0.2",
                          "opencv-python-headless==4.10.0.84", "scipy==1.13.1",
                          "imageio==2.37.0", "tensorboard==2.20.0", "tqdm==4.67.1", "loguru==0.7.3")
             .env({"HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1",
                   "CUDA_HOME": "/usr/local/cuda", "OMP_NUM_THREADS": "8", "MKL_NUM_THREADS": "8"})
             .add_local_dir(str(TASK), "/task", ignore=["**/__pycache__/**"])
             .add_local_python_source("frontier_ml"))


def run_root(run_id):
    if not re.fullmatch("[0-9a-f]{32}", run_id):
        raise ValueError("run-id must be 32 lowercase hexadecimal characters")
    sys.path.insert(0, "/task/pipeline")
    return Path("/runs") / run_id


@app.function(image=cpu_image, cpu=(8, 8), memory=(81920, 81920), timeout=7200,
              volumes={"/runs": volume}, retries=0, single_use_containers=True)
def prepare_run(run_id, revisions, reuse_acquisition=""):
    root = run_root(run_id)
    from gimm_baseline import prepare
    try:
        prepared = prepare(root, revisions, run_root(reuse_acquisition) if reuse_acquisition else None)
        generation = root / "generation.json"
        return {"prepared": prepared, "generation": json.loads(generation.read_text()) if generation.exists() else None}
    finally:
        volume.commit()


@app.function(image=gpu_image, gpu="H100!", cpu=(8, 8), memory=(81920, 81920),
              timeout=36000, volumes={"/runs": volume}, retries=0,
              block_network=True, single_use_containers=True, max_containers=1)
def generate_run(run_id):
    root = run_root(run_id)
    from gimm_baseline import generate
    volume.reload()
    try:
        return generate(root)
    finally:
        volume.commit()


@app.function(image=cpu_image, cpu=(8, 8), memory=(16384, 16384), timeout=8000,
              volumes={"/runs": volume}, retries=0, block_network=True, single_use_containers=True,
              max_containers=4)
def score_interval_shard(run_id, index, count):
    root = run_root(run_id)
    from vfi_parallel import score_shard
    volume.reload()
    return score_shard(root, index, count)


@app.function(image=cpu_image, cpu=0.25, memory=2048, timeout=10000,
              volumes={"/runs": volume}, retries=0, single_use_containers=True)
def score_run(run_id):
    root = run_root(run_id)
    from vfi_parallel import finish
    volume.reload()
    started = time.monotonic()
    reports = list(score_interval_shard.starmap([(run_id, i, 4) for i in range(4)]))
    try:
        return finish(root, reports, started)
    finally:
        volume.commit()


@app.function(image=cpu_image, cpu=0.125, memory=512, timeout=82800,
              retries=0, single_use_containers=True)
def complete_run(run_id, preparation_call_id):
    """Remote orchestration survives the local CLI/chat disconnecting."""
    state = modal.FunctionCall.from_id(preparation_call_id).get()
    if state["generation"] is None:
        generate_run.remote(run_id)
    return score_run.remote(run_id)


@app.local_entrypoint()
def main(out: str = "work/vfi/gimm-r-p-270-run", run_id: str = "", data_revision: str = "main",
         model_revision: str = "main", source_revision: str = "main", lpips_revision: str = "master",
         reuse_acquisition: str = ""):
    from frontier_ml.cli import hf_token
    sys.path.insert(0, str(TASK))
    from vfi_selection import DEFAULT_SELECTION

    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    previous = destination / "modal-run.json"
    if previous.exists():
        saved = json.loads(previous.read_text())
        if run_id and run_id != saved["run_id"]:
            raise ValueError("Output directory belongs to another run")
        run_id = saved["run_id"]
    run_id = run_id or uuid.uuid4().hex
    if not re.fullmatch("[0-9a-f]{32}", run_id):
        raise ValueError("Invalid run-id")
    revisions = {"data": data_revision, "model": model_revision,
                 "source": source_revision, "lpips": lpips_revision}
    if previous.exists() and saved["requested_revisions"] != revisions:
        raise ValueError("Resume revisions differ from saved run")
    if previous.exists() and saved.get("selection") != DEFAULT_SELECTION:
        raise ValueError("Resume interval selection differs; use a new output directory")
    if previous.exists() and saved.get("reuse_acquisition", "") != reuse_acquisition:
        raise ValueError("Resume acquisition source differs from saved run")
    if reuse_acquisition and (not re.fullmatch("[0-9a-f]{32}", reuse_acquisition) or reuse_acquisition == run_id):
        raise ValueError("Reuse requires a different existing acquisition run ID")
    if not previous.exists():
        previous.write_text(json.dumps({"run_id": run_id, "volume": "frontier-vfi-baseline",
                                       "reuse_acquisition": reuse_acquisition,
                                       "selection": DEFAULT_SELECTION,
                                       "requested_revisions": revisions}, indent=2) + "\n")
    print(f"Baseline run: {run_id}; reports: {destination}", flush=True)
    saved = json.loads(previous.read_text())
    if saved.get("completion_call_id"):
        completion = modal.FunctionCall.from_id(saved["completion_call_id"])
    else:
        token = hf_token()
        if not token:
            raise ValueError("HF login or HF_TOKEN is required to read the private evaluation dataset")
        # Credentials reach only the CPU downloader as a Modal secret, never a function
        # argument, controller, report, model subprocess or offline scorer.
        preparation = prepare_run.with_options(secrets=[modal.Secret.from_dict({"HF_TOKEN": token})]).spawn(
            run_id, revisions, reuse_acquisition)
        completion = complete_run.spawn(run_id, preparation.object_id)
        saved.update(preparation_call_id=preparation.object_id, completion_call_id=completion.object_id)
        previous.write_text(json.dumps(saved, indent=2) + "\n")
    print(f"Remote completion: {completion.object_id}", flush=True)
    files = completion.get()
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    metrics = json.loads(files["metrics.json"])
    generated = json.loads(files["generation.json"])
    print(json.dumps({"run_id": run_id, "by_factor": metrics["by_factor"], "timing": generated["timing"],
                      "calibration": str(destination / "calibration.json"),
                      "production_gate_activated": False}, indent=2), flush=True)
