"""Training-free LLM baselines with vLLM on one GPU (organizer side).

  judge:   zero-shot Yes/No judge; score = P(Yes) / (P(Yes) + P(No)) from first-token logprobs.
  logprob: a prover's mean token log-probability of the proof given the theorem (DeepSeek-Prover-V2 prompt).

    python llm_baselines.py judge   --model Qwen/Qwen3-8B ROWS.jsonl OUT.json
    python llm_baselines.py logprob --model deepseek-ai/DeepSeek-Prover-V2-7B ROWS.jsonl OUT.json

Each run writes raw scores for one split; binarize.py turns validation + held-out score
files into yes/no answers (threshold chosen on validation) and reports held-out macro F1.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))  # repo root

JUDGE_PROMPT = """You are an expert Lean 4 and Mathlib user. Below is a complete Lean 4 file, checked with Lean v4.15.0 and the matching Mathlib (maxHeartbeats 200000).

```lean4
{lean}
```

Does this file compile with no errors (and no sorry)? Answer with a single word: Yes or No."""


def judge_scores(model, rows, thinking=False):
    from vllm import LLM, SamplingParams
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.9")), enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    kwargs = {} if thinking else {"enable_thinking": False}
    prompts = []
    for row in rows:
        messages = [{"role": "user", "content": JUDGE_PROMPT.format(lean=row["lean"][-20000:])}]
        try:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)
        except TypeError:
            prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        prompts.append(prompt)
    yes_ids = {tokenizer.encode(w, add_special_tokens=False)[0] for w in ("Yes", " Yes", "yes")}
    no_ids = {tokenizer.encode(w, add_special_tokens=False)[0] for w in ("No", " No", "no")}
    outputs = llm.generate(prompts, SamplingParams(max_tokens=1, temperature=0, logprobs=20))
    scores = []
    for output in outputs:
        top = output.outputs[0].logprobs[0]
        p_yes = sum(math.exp(lp.logprob) for tid, lp in top.items() if tid in yes_ids)
        p_no = sum(math.exp(lp.logprob) for tid, lp in top.items() if tid in no_ids)
        scores.append(p_yes / (p_yes + p_no) if p_yes + p_no > 0 else 0.5)
    return scores


def logprob_scores(model, rows):
    from vllm import LLM, SamplingParams
    from lean_check import HEADER
    llm = LLM(model=model, dtype="bfloat16", max_model_len=8192, gpu_memory_utilization=float(os.environ.get("VLLM_GPU_UTIL", "0.9")), enable_prefix_caching=True)
    tokenizer = llm.get_tokenizer()
    prompts, starts = [], []
    for row in rows:
        body = row["lean"][len(HEADER):] if row["lean"].startswith(HEADER) else row["lean"]
        match = body.find(":= by")
        statement = body[:match + len(":= by")]
        messages = [{"role": "user", "content": f"Complete the following Lean 4 code:\n\n```lean4\n{HEADER}{statement}\n```"}]
        prefix = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True) + "```lean4\n" + HEADER
        full = prefix + body + "\n```"
        prefix_ids = tokenizer.encode(prefix, add_special_tokens=False)
        full_ids = tokenizer.encode(full, add_special_tokens=False)[:8000]
        prompts.append({"prompt_token_ids": full_ids})
        starts.append(min(len(prefix_ids) + len(tokenizer.encode(statement, add_special_tokens=False)), len(full_ids) - 1))
    outputs = llm.generate(prompts, SamplingParams(max_tokens=1, prompt_logprobs=0))
    scores = []
    for output, start in zip(outputs, starts):
        values = [next(iter(d.values())).logprob if d else 0.0 for d in output.prompt_logprobs[start:]]
        values = [v for v in values if v is not None and math.isfinite(v)]
        scores.append(sum(values) / max(1, len(values)))
    # Map to [0, 1] monotonically; binarize.py thresholds it on validation.
    return [1 / (1 + math.exp(-s)) for s in scores]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["judge", "logprob"])
    parser.add_argument("rows")
    parser.add_argument("output")
    parser.add_argument("--model", required=True)
    parser.add_argument("--thinking", action="store_true")
    args = parser.parse_args()
    rows = [json.loads(l) for l in open(args.rows)]
    values = judge_scores(args.model, rows, args.thinking) if args.mode == "judge" else logprob_scores(args.model, rows)
    summary = {"mode": args.mode, "model": args.model, "rows": args.rows, "n": len(values)}
    with open(args.output, "w") as stream:
        json.dump({"summary": summary, "predictions": values}, stream)
    print(json.dumps(summary))


if __name__ == "__main__":
    main()
