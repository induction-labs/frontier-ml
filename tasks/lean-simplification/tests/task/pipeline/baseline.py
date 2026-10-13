"""Organizer-only local H100 baseline. No downloads, API calls, or remote GPU rental.

Generate N independent simplifications using a cached vLLM-compatible checkpoint.
The report retains raw outputs, parsed proof terms, and timing. Use score mode
against the isolated checker to report red@1 and best-of-N (internal selection).
This is a baseline experiment, not a certified HTTP throughput measurement.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import re
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from simplification_client import CheckerClient
from simplification_metric import proof_length, proof_policy, reduction

PROMPT = """Shorten this Lean 4 proof while proving exactly the same theorem. Use Lean v4.15.0 and Mathlib v4.15.0. Preserve correctness; remove redundant reasoning and use concise existing lemmas or tactics where possible. The environment imports Mathlib and Aesop and opens BigOperators Real Nat Topology Rat. Do not use sorry, native_decide, metaprogramming or set_option. Return only the replacement proof term, starting with by, inside one Lean code block. Do not repeat the theorem statement or explain your answer.

```lean4
theorem target {statement} := {proof}
```
"""


def extract(text, row):
    text = text.rsplit("</think>", 1)[-1].strip()
    blocks = re.findall(r"```(?:lean4?|Lean4?)?\s*\n(.*?)```", text, re.S)
    if blocks:
        text = blocks[-1].strip()
    prefix = "theorem target " + row["statement"] + " :="
    if text.startswith(prefix):
        text = text[len(prefix):].strip()
    return text


def generate(args):
    if args.model is None or not args.model.is_dir():
        raise ValueError("--model must name an existing local checkpoint directory")
    os.environ["HF_HUB_OFFLINE"] = "1"
    os.environ["TRANSFORMERS_OFFLINE"] = "1"
    from vllm import LLM, SamplingParams
    rows = [json.loads(line) for line in args.data.read_text().splitlines()]
    llm = LLM(model=str(args.model), dtype="bfloat16", max_model_len=8192,
              gpu_memory_utilization=0.8, enable_prefix_caching=True,
              max_num_seqs=8, enforce_eager=True)
    tokenizer = llm.get_tokenizer()
    prompts = [tokenizer.apply_chat_template([{"role": "user", "content": PROMPT.format(**row)}],
                tokenize=False, add_generation_prompt=True, enable_thinking=False) for row in rows]
    started = time.monotonic()
    outputs = llm.generate(prompts, SamplingParams(n=args.samples, temperature=0.7 if args.samples > 1 else 0,
                                                  top_p=0.95, max_tokens=2048, seed=20260924))
    result = {"model": str(args.model), "data_sha256": hashlib.sha256(args.data.read_bytes()).hexdigest(),
              "samples_per_proof": args.samples, "generation_seconds": time.monotonic() - started,
              "sampling": {"temperature": 0.7 if args.samples > 1 else 0, "top_p": 0.95, "max_tokens": 2048,
                           "seed": 20260924, "enable_thinking": False}, "records": []}
    for row, output in zip(rows, outputs, strict=True):
        result["records"].append({"id": row["id"], "outputs": [
            {"proof": extract(o.text, row), "raw": o.text, "finish_reason": o.finish_reason,
             "generated_tokens": len(o.token_ids)} for o in output.outputs]})
    args.output.write_text(json.dumps(result, ensure_ascii=False, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


def score(args):
    rows = {r["id"]: r for r in (json.loads(line) for line in args.data.read_text().splitlines())}
    generated = json.loads(args.predictions.read_text())
    if generated["data_sha256"] != hashlib.sha256(args.data.read_bytes()).hexdigest():
        raise ValueError("Predictions are for a different dataset")
    checker = CheckerClient(args.checker_url)
    def measure(record):
        row = rows[record["id"]]
        original = proof_length(row["proof"])
        scores = []
        for output in record["outputs"]:
            proof = output["proof"]
            if proof_policy(proof):
                scores.append(0.0)
            else:
                verdict = checker(row, proof)
                scores.append(reduction(original, proof_length(proof), verdict["valid"]))
        return {"id": row["id"], "reductions": scores, "red_at_1": scores[0], "best_of_n": max(scores)}
    with ThreadPoolExecutor(max_workers=8) as pool:
        records = []
        for index, record in enumerate(pool.map(measure, generated["records"]), 1):
            records.append(record)
            if index % 10 == 0:
                print(f"Checked {index}/{len(generated['records'])} proofs", flush=True)
    result = {"model": generated["model"], "samples": len(records), "samples_per_proof": generated["samples_per_proof"],
              "red_at_1": sum(r["red_at_1"] for r in records) / len(records),
              "best_of_n": sum(r["best_of_n"] for r in records) / len(records), "records": records}
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({k: v for k, v in result.items() if k != "records"}))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("mode", choices=["generate", "score"])
    parser.add_argument("--data", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--predictions", type=Path)
    parser.add_argument("--checker-url")
    args = parser.parse_args()
    (generate if args.mode == "generate" else score)(args)
