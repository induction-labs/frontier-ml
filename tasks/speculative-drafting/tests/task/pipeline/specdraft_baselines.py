"""Operator baselines for speculative-drafting (never shipped).

    python tasks/speculative-drafting/pipeline/specdraft_baselines.py ngram --data data/speculative-drafting/public/validation.jsonl
    python tasks/speculative-drafting/pipeline/specdraft_baselines.py example --model DIR --out OUTPUT_DIR

`ngram` runs the exact public walk with prompt-lookup drafting (the training-free baseline: propose the tokens
that followed the most recent earlier occurrence of the longest matching suffix). `example` writes a complete
example submission: a Hugging Face causal LM whose 2-D weights are stored as per-row int8 with float scales,
served by pipeline/example_submission/server.py. It is the packaging fixture for end-to-end grading tests and the
"quantize a small public model and serve it" baseline, not a recommended solution.
"""
import argparse
import json
from pathlib import Path
import shutil
import sys
import time

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent))

from specdraft_walk import DRAFT_TOKENS, score, walk  # noqa: E402


def prompt_lookup(tokens, draft_tokens=DRAFT_TOKENS, max_ngram=4):
    for n in range(min(max_ngram, len(tokens) - 1), 0, -1):
        suffix = tokens[-n:]
        for start in range(len(tokens) - n - 1, -1, -1):
            if tokens[start:start + n] == suffix:
                follow = tokens[start + n:start + n + draft_tokens]
                if follow:
                    return follow
    return []


class PromptLookupTransport:
    def request_batch(self, requests, *, concurrency, timeout_seconds, raise_on_failure=False, **_):
        started = time.monotonic()
        responses = [{"draft": prompt_lookup(payload["tokens"])} for _, payload in requests]
        return responses, {"elapsed_seconds": time.monotonic() - started, "retried_requests": 0}


def run_ngram(data):
    rows = [json.loads(line) for line in Path(data).read_text().splitlines() if line.strip()]
    traces, timing, failures = walk(PromptLookupTransport(), rows, concurrency=32, serving_budget_seconds=1e9)
    result = score(rows, traces)
    details = result["details"]
    return {"baseline": "prompt_lookup_ngram<=4", "score": result["score"], "tau": details["tau"],
            "tau_by_category": details["tau_by_category"], "rows": len(rows)}


def write_example(model_dir, out):
    """Quantize every 2-D floating tensor to per-row symmetric int8; keep the rest in bfloat16."""
    import torch
    from safetensors.torch import load_file, save_file
    model_dir, out = Path(model_dir), Path(out)
    if out.exists():
        shutil.rmtree(out)
    (out / "drafter").mkdir(parents=True)
    tensors = {}
    for path in sorted(model_dir.glob("*.safetensors")):
        tensors.update(load_file(path))
    packed = {}
    for name, tensor in tensors.items():
        tensor = tensor.to(torch.float32)
        if tensor.ndim == 2:
            scale = tensor.abs().amax(dim=1, keepdim=True).clamp(min=1e-8) / 127
            packed[name + ".qweight"] = torch.round(tensor / scale).clamp(-127, 127).to(torch.int8).contiguous()
            packed[name + ".scale"] = scale.to(torch.bfloat16).contiguous()
        else:
            packed[name] = tensor.to(torch.bfloat16).contiguous()
    save_file(packed, out / "drafter/weights.safetensors")
    shutil.copy(model_dir / "config.json", out / "drafter/config.json")
    for name in ("server.py", "serve.sh"):
        shutil.copy(HERE / "example_submission" / name, out / name)
    total = sum(p.stat().st_size for p in out.rglob("*") if p.is_file())
    return {"out": str(out), "bytes": total}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    ngram = sub.add_parser("ngram")
    ngram.add_argument("--data", required=True)
    example = sub.add_parser("example")
    example.add_argument("--model", required=True, help="local Hugging Face checkpoint directory")
    example.add_argument("--out", required=True)
    args = parser.parse_args(argv)
    result = run_ngram(args.data) if args.command == "ngram" else write_example(args.model, args.out)
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
