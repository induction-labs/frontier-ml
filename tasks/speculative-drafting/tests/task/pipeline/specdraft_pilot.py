"""Operator admission pilot for speculative-drafting (never shipped; about one H100-hour on Modal).

    modal run tasks/speculative-drafting/pipeline/specdraft_pilot.py --eval data/speculative-drafting/public/validation.jsonl --out PILOT.json

The cheapest version of the obvious $50 route: sequence-level distillation. Qwen3-8B (the pinned target, non-thinking,
greedy, 256 tokens) answers public *training-split* prompts (UltraChat SFT, GSM8K train, CodeAlpaca); Qwen3-0.6B is
fine-tuned on those continuations; both drafters are scored with the exact public walk (specdraft_walk) on the
evaluation rows given. The held-out file is never used for training or selection.
"""
import json
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
TARGET, TARGET_REVISION = "Qwen/Qwen3-8B", "b968826d9c46dd6066d109eabc6255188de91218"
DRAFTER, DRAFTER_REVISION = "Qwen/Qwen3-0.6B", "c1899de289a04d12100db370d81485cdf75e47ca"
MAX_NEW_TOKENS, EOS, STOP = 256, 151645, (151645, 151643)

image = (modal.Image.debian_slim(python_version="3.12")
         .pip_install("vllm==0.11.0", "huggingface_hub[hf_transfer]==0.35.3", "datasets==4.1.1")
         .env({"HF_HUB_ENABLE_HF_TRANSFER": "1", "HF_HOME": "/cache/hf", "VLLM_USE_V1": "1",
               "PYTORCH_CUDA_ALLOC_CONF": "expandable_segments:True"})
         .add_local_dir(str(TASK), "/task", ignore=["pipeline/__pycache__", "__pycache__"])
         .add_local_python_source("frontier_ml"))
cache = modal.Volume.from_name("specdraft-hf-cache", create_if_missing=True)
work = modal.Volume.from_name("specdraft-pilot", create_if_missing=True)
app = modal.App("specdraft-pilot", image=image)


def _finish(output):
    tokens = list(output.token_ids)
    if output.finish_reason == "stop" and (not tokens or tokens[-1] not in STOP):
        tokens.append(EOS)
    return tokens[:MAX_NEW_TOKENS]


def _prompts(limit_per_source):
    from datasets import load_dataset
    rows = []
    chat = load_dataset("HuggingFaceH4/ultrachat_200k", split="train_sft").shuffle(seed=0).select(range(limit_per_source * 2))
    rows += [[{"role": "user", "content": r["prompt"]}] for r in chat]
    math = load_dataset("openai/gsm8k", "main", split="train").shuffle(seed=0).select(range(limit_per_source))
    rows += [[{"role": "user", "content": r["question"]}] for r in math]
    code = load_dataset("sahil2801/CodeAlpaca-20k", split="train").shuffle(seed=0).select(range(limit_per_source))
    rows += [[{"role": "user", "content": (r["instruction"] + ("\n\n" + r["input"] if r["input"] else ""))}] for r in code]
    return rows


@app.function(gpu="H100!", timeout=3600, volumes={"/cache": cache, "/work": work})
def distill_data(limit_per_source=2500):
    import vllm
    from huggingface_hub import snapshot_download
    from transformers import AutoTokenizer
    path = snapshot_download(TARGET, revision=TARGET_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    cache.commit()
    tokenizer = AutoTokenizer.from_pretrained(path)
    conversations = _prompts(limit_per_source)
    prompts = [tokenizer.apply_chat_template(c, add_generation_prompt=True, enable_thinking=False, tokenize=True)
               for c in conversations]
    prompts = [p for p in prompts if len(p) <= 1536]
    llm = vllm.LLM(model=path, dtype="bfloat16", seed=0, max_model_len=4096, gpu_memory_utilization=0.85)
    outputs = llm.generate([vllm.TokensPrompt(prompt_token_ids=p) for p in prompts],
                           vllm.SamplingParams(temperature=0.0, max_tokens=MAX_NEW_TOKENS))
    rows = [{"prompt_tokens": list(p), "reference": _finish(o.outputs[0])} for p, o in zip(prompts, outputs)]
    Path("/work/distill.jsonl").write_text("".join(json.dumps(r) + "\n" for r in rows))
    work.commit()
    return {"rows": len(rows), "tokens": sum(len(r["reference"]) for r in rows)}


@app.function(gpu="H100!", timeout=7200, volumes={"/cache": cache, "/work": work})
def finetune(epochs=2, lr=2e-5, batch_tokens=16384):
    """Full fine-tune on the target's continuations (loss on continuation tokens only), padded batches."""
    import math
    import random
    import torch
    from huggingface_hub import snapshot_download
    from transformers import AutoModelForCausalLM
    work.reload()
    path = snapshot_download(DRAFTER, revision=DRAFTER_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt"])
    cache.commit()
    model = AutoModelForCausalLM.from_pretrained(path, torch_dtype=torch.bfloat16, attn_implementation="sdpa").cuda()
    rows = [json.loads(line) for line in Path("/work/distill.jsonl").read_text().splitlines()]
    rng = random.Random(0)
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0, betas=(0.9, 0.95))
    def make_batches(order):
        batches, batch, width = [], [], 0
        for row in order:  # padded tokens per batch stay under batch_tokens
            length = len(row["prompt_tokens"]) + len(row["reference"])
            if batch and max(width, length) * (len(batch) + 1) > batch_tokens:
                batches.append(batch)
                batch, width = [], 0
            batch.append(row)
            width = max(width, length)
        return batches + ([batch] if batch else [])

    by_length = sorted(rows, key=lambda r: len(r["prompt_tokens"]) + len(r["reference"]))
    steps_total = epochs * len(make_batches(by_length))
    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lambda s: min(1.0, (s + 1) / 20) * max(0.0, 1 - s / steps_total))
    log, step = [], 0
    model.train()
    for epoch in range(epochs):
        batches = make_batches(by_length)
        rng.shuffle(batches)
        for batch in batches:
            width = max(len(r["prompt_tokens"]) + len(r["reference"]) for r in batch)
            ids = torch.full((len(batch), width), 151643, dtype=torch.long)
            labels = torch.full((len(batch), width), -100, dtype=torch.long)
            mask = torch.zeros((len(batch), width), dtype=torch.long)
            for i, r in enumerate(batch):
                seq = r["prompt_tokens"] + r["reference"]
                ids[i, :len(seq)] = torch.tensor(seq)
                mask[i, :len(seq)] = 1
                labels[i, len(r["prompt_tokens"]):len(seq)] = torch.tensor(r["reference"])
            ids, labels, mask = ids.cuda(), labels.cuda(), mask.cuda()
            logits = model(input_ids=ids, attention_mask=mask).logits[:, :-1]
            targets = labels[:, 1:]
            keep = targets != -100
            loss = torch.nn.functional.cross_entropy(logits[keep].float(), targets[keep])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                log.append({"step": step, "epoch": epoch, "loss": loss.item()})
                print(f"step {step}/{steps_total} loss {loss.item():.4f}", flush=True)
    model.save_pretrained("/work/drafter", safe_serialization=True)
    for name in ("tokenizer.json", "tokenizer_config.json", "vocab.json", "merges.txt", "generation_config.json"):
        if (Path(path) / name).exists():
            Path("/work/drafter", name).write_bytes((Path(path) / name).read_bytes())
    work.commit()
    return {"steps": step, "log": log[-10:]}


@app.function(gpu="H100!", timeout=3600, volumes={"/cache": cache, "/work": work}, single_use_containers=True)
def evaluate(rows, which):
    import sys
    sys.path.insert(0, "/task")
    sys.path.insert(0, "/task/pipeline")
    import vllm
    from huggingface_hub import snapshot_download
    from specdraft_walk import DRAFT_TOKENS, score, walk
    work.reload()
    path = ("/work/drafter" if which == "distilled" else
            snapshot_download(DRAFTER, revision=DRAFTER_REVISION, allow_patterns=["*.json", "*.safetensors", "*.txt"]))
    llm = vllm.LLM(model=path, tokenizer=path, dtype="bfloat16", seed=0, max_model_len=4096,
                   gpu_memory_utilization=0.6, enable_prefix_caching=True)
    params = vllm.SamplingParams(temperature=0.0, max_tokens=DRAFT_TOKENS)

    class Transport:
        def request_batch(self, requests, **_):
            import time
            started = time.monotonic()
            outputs = llm.generate([vllm.TokensPrompt(prompt_token_ids=p["tokens"]) for _, p in requests], params,
                                   use_tqdm=False)
            return [{"draft": _finish(o.outputs[0])[:DRAFT_TOKENS]} for o in outputs], {
                "elapsed_seconds": time.monotonic() - started, "retried_requests": 0}

    traces, timing, _ = walk(Transport(), rows, concurrency=32, serving_budget_seconds=1e9)
    result = score(rows, traces)
    return {"drafter": which, "score": result["score"], "tau": result["details"]["tau"],
            "tau_by_category": result["details"]["tau_by_category"],
            "per_prompt_tau": [r["tau"] for r in result["details"]["records"]]}


@app.local_entrypoint()
def main(eval: str, out: str, heldout: str = "", limit_per_source: int = 2500, epochs: int = 2, skip_training: bool = False,
         skip_data: bool = False):
    """Hyperparameters are fixed before any evaluation; held-out rows are measured, never used to choose anything."""
    splits = {"validation": eval, **({"heldout": heldout} if heldout else {})}
    report = {}
    if not skip_training:
        if not skip_data:
            report["distill_data"] = distill_data.remote(limit_per_source)
        report["finetune"] = finetune.remote(epochs)
    for split, path in splits.items():
        rows = [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]
        for which in ("base", "distilled"):
            report[f"{split}/{which}"] = evaluate.remote(rows, which)
    Path(out).write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({k: {x: v[x] for x in ("tau", "score")} for k, v in report.items() if "/" in k}))
