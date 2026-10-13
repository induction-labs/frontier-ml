"""Continue a stopped baseline on one H100 with a fixed smaller evaluation subset."""
import json
from pathlib import Path
import sys
import time
import uuid

import modal

sys.path[:0] = ["/task/pipeline", str(Path(__file__).resolve().parent)]
import vfi_modal as runner

app = modal.App("frontier-vfi-parallel-quality")


@app.function(image=runner.cpu_image, cpu=(8, 8), memory=16384, timeout=7200,
              volumes={"/runs": runner.volume}, retries=0, single_use_containers=True)
def prepare(run_id, previous, workers):
    from vfi_quality import prepare_quality
    from vfi_selection import DEFAULT_SELECTION
    runner.volume.reload()
    try:
        return prepare_quality(runner.run_root(run_id), runner.run_root(previous), workers, selection=DEFAULT_SELECTION)
    finally:
        runner.volume.commit()


@app.function(image=runner.gpu_image, gpu="H100!", cpu=(8, 8), memory=(81920, 81920), timeout=7200,
              volumes={"/runs": runner.volume}, block_network=True, retries=0, single_use_containers=True,
              max_containers=1)
def generate(run_id, index, workers, retained):
    root = runner.run_root(run_id)
    from vfi_quality import predict_shard
    runner.volume.reload()
    try:
        return predict_shard(root, index, workers, retained)
    finally:
        runner.volume.commit()


@app.function(image=runner.cpu_image, cpu=(8, 8), memory=16384, timeout=8000,
              volumes={"/runs": runner.volume}, block_network=True, retries=0, single_use_containers=True,
              max_containers=4)
def measure(run_id, index, count):
    root = runner.run_root(run_id)
    from vfi_parallel import score_shard
    runner.volume.reload()
    return score_shard(root, index, count)


@app.function(image=runner.cpu_image, cpu=2, memory=4096, timeout=24000,
              volumes={"/runs": runner.volume}, retries=0, single_use_containers=True)
def complete(run_id, previous, workers):
    root = runner.run_root(run_id)
    state = prepare.remote(run_id, previous, workers)
    started = time.monotonic()
    shards = (list(generate.starmap([(run_id, i, workers, state["retained"]) for i in range(workers)]))
              if state["retained"] < state["total"] else [])
    runner.volume.reload()
    from vfi_quality import assemble
    report = assemble(root, shards, started)
    runner.volume.commit()
    print(json.dumps(report), flush=True)
    started = time.monotonic()
    scores = list(measure.starmap([(run_id, i, 4) for i in range(4)]))
    from vfi_parallel import finish
    try:
        return finish(root, scores, started)
    finally:
        runner.volume.commit()


@app.local_entrypoint()
def main(previous: str, out: str = "work/vfi/gimm-r-p-270-quality", workers: int = 1):
    sys.path.insert(0, str(runner.TASK))
    from vfi_selection import DEFAULT_SELECTION
    if workers != 1:
        raise ValueError("Baseline acquisition is limited to one H100")
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    saved_path = destination / "quality-run.json"
    if saved_path.exists():
        saved = json.loads(saved_path.read_text())
        if (saved["previous"] != previous or saved["workers"] != workers
                or saved.get("selection") != DEFAULT_SELECTION):
            raise ValueError("Resume arguments differ from saved run")
        call = modal.FunctionCall.from_id(saved["completion_call_id"])
    else:
        run_id = uuid.uuid4().hex
        call = complete.spawn(run_id, previous, workers)
        saved = {"run_id": run_id, "previous": previous, "workers": workers,
                 "selection": DEFAULT_SELECTION, "completion_call_id": call.object_id}
        saved_path.write_text(json.dumps(saved, indent=2) + "\n")
    print(json.dumps(saved), flush=True)
    files = call.get()
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(json.dumps(json.loads(files["metrics.json"])["by_factor"], indent=2), flush=True)
