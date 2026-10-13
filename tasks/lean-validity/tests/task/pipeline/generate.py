"""Sample whole-proof attempts from several open provers with vLLM (organizer side, 1 GPU).

Output JSONL rows: {sid, prover, sample, text, finish_reason}. ``text`` is the raw model
output; ``extract.py`` turns it into a canonical item or drops it.
"""
from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lean_check import HEADER  # noqa: E402

COT_SUFFIX = ("\n\nBefore producing the Lean 4 code to formally prove the given theorem, provide a detailed proof plan "
              "outlining the main proof steps and strategies.\nThe plan should highlight key ideas, intermediate lemmas, "
              "and proof structures that will guide the construction of the final formal proof.")

PROVERS = {
    # name: (hf repo, prompt style, samples per statement, max new tokens, temperature)
    "ds_v2_7b": ("deepseek-ai/DeepSeek-Prover-V2-7B", "chat_plain", 6, 4096, 1.0),
    "goedel_v2_8b": ("Goedel-LM/Goedel-Prover-V2-8B", "chat_cot", 1, 8192, 1.0),
    "kimina_8b": ("AI-MO/Kimina-Prover-Distill-8B", "kimina", 1, 8192, 0.6),
    "ds_v15_rl": ("deepseek-ai/DeepSeek-Prover-V1.5-RL", "completion", 6, 2048, 1.0),
    "goedel_sft": ("Goedel-LM/Goedel-Prover-SFT", "completion", 6, 2048, 1.0),
    "qwen3_8b": ("Qwen/Qwen3-8B", "chat_general", 2, 2048, 0.8),
}


def build_prompt(style: str, statement: str, tokenizer) -> str:
    code = HEADER + statement
    if style == "completion":
        return ("Complete the following Lean 4 code with explanatory comments preceding each line of code:\n\n"
                f"```lean4\n{code}")
    if style == "chat_plain":
        messages = [{"role": "user", "content": f"Complete the following Lean 4 code:\n\n```lean4\n{code}\n```"}]
    elif style == "chat_cot":
        messages = [{"role": "user", "content": f"Complete the following Lean 4 code:\n\n```lean4\n{code}\n```" + COT_SUFFIX}]
    elif style == "kimina":
        messages = [{"role": "system", "content": "You are an expert in mathematics and Lean 4."},
                    {"role": "user", "content": "Think about and solve the following problem step by step in Lean 4.\n"
                                                f"# Formal statement:\n```lean4\n{code}\n```\n"}]
    elif style == "chat_general":
        messages = [{"role": "user", "content": "Prove the following theorem in Lean 4 with Mathlib. Reply with the complete "
                                                "Lean 4 code (including the theorem statement exactly as given) in a single "
                                                f"```lean4 code block.\n\n```lean4\n{code}\n```"}]
    else:
        raise ValueError(style)
    kwargs = {"enable_thinking": False} if style == "chat_general" else {}
    return tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, **kwargs)


def main():
    import argparse
    from vllm import LLM, SamplingParams

    parser = argparse.ArgumentParser()
    parser.add_argument("statements")
    parser.add_argument("output")
    parser.add_argument("--prover", required=True, choices=sorted(PROVERS))
    parser.add_argument("--samples", type=int)
    parser.add_argument("--max-tokens", type=int)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    repo, style, samples, max_tokens, temperature = PROVERS[args.prover]
    samples = args.samples or samples
    max_tokens = args.max_tokens or max_tokens
    rows = [json.loads(line) for line in open(args.statements) if line.strip()]
    llm = LLM(model=repo, dtype="bfloat16", max_model_len=max_tokens + 2048, gpu_memory_utilization=0.9,
              seed=args.seed, trust_remote_code=False, enable_prefix_caching=True, max_num_seqs=512,
              max_num_batched_tokens=16384)
    tokenizer = llm.get_tokenizer()
    prompts = [build_prompt(style, row["statement"], tokenizer) for row in rows]
    stop = ["```"] if style == "completion" else None
    # Global engine seed only: per-request seeds force vLLM's slower seeded sampling path.
    params = SamplingParams(n=samples, temperature=temperature, top_p=0.95, max_tokens=max_tokens, stop=stop)
    outputs = llm.generate(prompts, params)
    with open(args.output, "w") as stream:
        for row, prompt, output in zip(rows, prompts, outputs):
            for index, completion in enumerate(output.outputs):
                text = completion.text
                if style == "completion":
                    text = "```lean4\n" + HEADER + row["statement"] + text + "\n```"
                stream.write(json.dumps({"sid": row["sid"], "prover": args.prover, "sample": index, "text": text,
                                         "finish_reason": completion.finish_reason,
                                         "tokens": len(completion.token_ids)}, ensure_ascii=False) + "\n")


if __name__ == "__main__":
    main()
