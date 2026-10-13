"""Finish an already running baseline after its first complete pass, then score in parallel."""
import json
from pathlib import Path
import sys
import time

import modal

sys.path[:0] = ["/task/pipeline", str(Path(__file__).resolve().parent)]
import vfi_modal as runner

app = modal.App("frontier-vfi-finish")


@app.function(image=runner.cpu_image, cpu=(8, 8), memory=(16384, 16384), timeout=8000,
              volumes={"/runs": runner.volume}, block_network=True, retries=0, single_use_containers=True,
              max_containers=16)
def metric_shard(run_id, index, count):
    root = runner.run_root(run_id)
    from vfi_parallel import score_shard
    runner.volume.reload()
    return score_shard(root, index, count)


@app.function(image=runner.cpu_image, cpu=0.25, memory=2048, timeout=22000,
              volumes={"/runs": runner.volume}, retries=0, single_use_containers=True)
def finish_existing(run_id, generation_call_id, controller_call_id, shards=16):
    root = runner.run_root(run_id)
    deadline = time.monotonic() + 11000
    while True:
        runner.volume.reload()
        if (root / "pass-1.json").exists():
            first = json.loads((root / "pass-1.json").read_text())
            if first.get("status") != "complete":
                raise ValueError("First prediction pass is not complete")
            break
        if time.monotonic() >= deadline:
            raise TimeoutError("No complete first prediction pass arrived")
        time.sleep(20)
    # Cancel the old pipeline so neither repeated inference nor sequential scoring runs.
    generation = modal.FunctionCall.from_id(generation_call_id)
    generation.cancel()
    modal.FunctionCall.from_id(controller_call_id).cancel()
    try:
        generation.get(timeout=60)
    except Exception as exc:
        print(f"Previous generation stopped: {type(exc).__name__}", flush=True)
    runner.volume.reload()
    print("First complete prediction pass retained; starting 16 independent CPU metric shards", flush=True)
    started = time.monotonic()
    reports = list(metric_shard.starmap([(run_id, i, shards) for i in range(shards)]))
    from vfi_parallel import finish
    try:
        return finish(root, reports, started)
    finally:
        runner.volume.commit()


@app.local_entrypoint()
def main(run_id: str, generation_call_id: str, controller_call_id: str, out: str):
    destination = Path(out)
    destination.mkdir(parents=True, exist_ok=True)
    receipt_path = destination / "parallel-run.json"
    if receipt_path.exists():
        saved = json.loads(receipt_path.read_text())
        if saved["run_id"] != run_id:
            raise ValueError("Output directory belongs to another run")
        call = modal.FunctionCall.from_id(saved["completion_call_id"])
    else:
        call = finish_existing.spawn(run_id, generation_call_id, controller_call_id)
        receipt_path.write_text(json.dumps({"run_id": run_id, "completion_call_id": call.object_id,
                                           "generation_call_id": generation_call_id,
                                           "controller_call_id": controller_call_id,
                                           "shards": 16}, indent=2) + "\n")
    print(f"Remote parallel completion: {call.object_id}", flush=True)
    files = call.get()
    for name, content in files.items():
        path = destination / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content)
    print(json.dumps(json.loads(files["metrics.json"])["by_factor"], indent=2), flush=True)
