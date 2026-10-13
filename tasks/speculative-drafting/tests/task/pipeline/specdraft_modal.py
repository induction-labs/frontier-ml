"""Operator-only Modal jobs for speculative-drafting (never shipped; costs operator GPU time).

    modal run tasks/speculative-drafting/pipeline/specdraft_modal.py::references --prompts P.jsonl --out R.jsonl
    modal run tasks/speculative-drafting/pipeline/specdraft_modal.py::baseline --data D.jsonl --drafter Qwen/Qwen3-0.6B --out B.json

`references` formats every prompt with the target's chat template (non-thinking) and records the target's greedy
continuation with vLLM, then re-checks every reference token against a teacher-forced Transformers forward pass
(argmax agreement is reported; near-ties can differ between engines). `baseline` runs the exact public walk
(specdraft_walk.walk) with an in-process transport whose drafts come from a greedy vLLM draft model, so the
reported tau is what the same drafter would score through /draft.
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
TASK = HERE.parent

TARGET = "Qwen/Qwen3-8B"
TARGET_REVISION = "b968826d9c46dd6066d109eabc6255188de91218"
MAX_NEW_TOKENS = 256
EOS = 151645  # <|im_end|>
STOP_IDS = (151645, 151643)

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]==0.35.3", "bitsandbytes==0.48.1")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf", "VLLM_USE_V1": "1"})
         .add_local_dir(str(TASK), "/task", ignore=["pipeline/__pycache__", "__pycache__"])
         .add_local_python_source("frontier_ml"))
cache = modal.Volume.from_name("specdraft-hf-cache", create_if_missing=True)
app = modal.App("specdraft-pipeline", image=image)


def _download(repo, revision):
    from huggingface_hub import snapshot_download
    return snapshot_download(repo, revision=revision, allow_patterns=["*.json", "*.safetensors", "*.txt", "*.model",
                                                                      "tokenizer*", "merges.txt", "vocab.json"])


def _finish(output):
    tokens = list(output.token_ids)
    if output.finish_reason == "stop" and (not tokens or tokens[-1] not in STOP_IDS):
        tokens.append(EOS)  # an EOS stop that the engine did not echo
    return tokens


@app.function(gpu="H100!", timeout=3600, volumes={"/cache": cache})
def generate_references(rows):
    import vllm
    from transformers import AutoTokenizer
    path = _download(TARGET, TARGET_REVISION)
    cache.commit()
    tokenizer = AutoTokenizer.from_pretrained(path)
    prompts = [tokenizer.apply_chat_template(r["messages"], add_generation_prompt=True, enable_thinking=False,
                                             tokenize=True) for r in rows]
    llm = vllm.LLM(model=path, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.85)
    params = vllm.SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS, seed=0)
    outputs = llm.generate([vllm.TokensPrompt(prompt_token_ids=p) for p in prompts], params)
    engine = {"engine": "vllm", "version": vllm.__version__, "dtype": "bfloat16", "max_model_len": 4096}
    result = []
    for row, prompt, output in zip(rows, prompts, outputs, strict=True):
        completion = output.outputs[0]
        result.append({**row, "prompt_tokens": list(prompt), "reference": _finish(completion)[:MAX_NEW_TOKENS],
                       "finish_reason": completion.finish_reason, "engine": engine})
    return result


@app.function(gpu="H100!", timeout=3600, volumes={"/cache": cache})
def check_agreement(rows):
    """Teacher-forced Transformers argmax at every reference position (a data-quality diagnostic)."""
    import torch
    from transformers import AutoModelForCausalLM
    path = _download(TARGET, TARGET_REVISION)
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    model.eval()
    checked = []
    with torch.no_grad():
        for row in rows:
            ids = row["prompt_tokens"] + row["reference"]
            logits = model(torch.tensor([ids], device="cuda")).logits[0].float()
            start = len(row["prompt_tokens"]) - 1
            predicted = logits[start:start + len(row["reference"])].argmax(-1).tolist()
            top2 = logits[start:start + len(row["reference"])].topk(2, dim=-1).values
            margins = (top2[:, 0] - top2[:, 1]).tolist()
            mismatches = [i for i, (p, r) in enumerate(zip(predicted, row["reference"])) if p != r]
            checked.append({"positions": len(row["reference"]), "mismatches": len(mismatches),
                            "mismatch_margins": [round(margins[i], 4) for i in mismatches][:10],
                            "method": "transformers bf16 sdpa teacher-forced argmax"})
    return checked


class _DraftTransport:
    """request_batch() like the trusted grading transport, answered by a greedy vLLM draft model."""

    def __init__(self, llm, draft_tokens, quantization=None):
        import vllm
        self.llm, self.vllm = llm, vllm
        self.params = vllm.SamplingParams(temperature=0.0, max_tokens=draft_tokens)

    def request_batch(self, requests, *, concurrency, timeout_seconds, raise_on_failure=False, **_):
        import time
        started = time.monotonic()
        prompts = [self.vllm.TokensPrompt(prompt_token_ids=payload["tokens"]) for _, payload in requests]
        outputs = self.llm.generate(prompts, self.params, use_tqdm=False)
        responses = [{"draft": _finish(o.outputs[0])} for o in outputs]
        return responses, {"elapsed_seconds": time.monotonic() - started, "retried_requests": 0}


@app.function(gpu="H100!", timeout=3600, volumes={"/cache": cache}, single_use_containers=True)
def drafter_walk(rows, drafter, revision=None, quantization=None):
    import sys as _sys
    _sys.path.insert(0, "/task")
    import vllm
    from specdraft_walk import DRAFT_TOKENS, score, walk
    path = _download(drafter, revision)
    cache.commit()
    llm = vllm.LLM(model=path, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.6,
                   enable_prefix_caching=True, quantization=quantization)
    traces, timing, failures = walk(_DraftTransport(llm, DRAFT_TOKENS), rows, concurrency=32,
                                    serving_budget_seconds=3600)
    result = score(rows, traces)
    result["details"]["timing"] = [timing]
    result["details"]["failed_requests"] = failures
    result["details"]["drafter"] = {"repo": drafter, "revision": revision, "quantization": quantization}
    return result


def _read(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


@app.local_entrypoint()
def references(prompts: str, out: str, check: bool = True):
    rows = _read(prompts)
    result = generate_references.remote(rows)
    if check:
        for row, agreement in zip(result, check_agreement.remote(result), strict=True):
            row["argmax_agreement"] = agreement
    Path(out).write_text("".join(json.dumps(r, ensure_ascii=False) + "\n" for r in result))
    finished = sum(r["finish_reason"] == "stop" for r in result)
    tokens = sum(len(r["reference"]) for r in result)
    mismatches = sum(r.get("argmax_agreement", {}).get("mismatches", 0) for r in result)
    print(json.dumps({"rows": len(result), "finished_within_limit": finished, "reference_tokens": tokens,
                      "argmax_mismatches": mismatches}))


@app.local_entrypoint()
def baseline(data: str, drafter: str, out: str, revision: str = "", quantization: str = ""):
    result = drafter_walk.remote(_read(data), drafter, revision or None, quantization or None)
    Path(out).write_text(json.dumps(result, indent=2) + "\n")
    details = result["details"]
    print(json.dumps({"drafter": drafter, "quantization": quantization or None, "score": result["score"],
                      "tau": details["tau"], "tau_by_category": details["tau_by_category"],
                      "serving_seconds": details["timing"][0]["serving_seconds"]}))
