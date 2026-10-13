#!/usr/bin/env python3
"""Operator tool: the text-only Maple baseline that anchors maple-vlm's reward floor (task.yaml score_normalization).

Two phases, so the GPU environment never needs the judge and the scoring environment never needs the GPU:

  generate  Load the pinned deepgrove/maple-preview checkpoint (bf16, one GPU) and answer every question of a split
            with every image removed from validate.build_request's message: the same question text, options and
            trailing instruction, through the model's chat template with thinking turned off (the template opens a
            <think> block; an empty one is closed immediately, as Qwen-style templates do for enable_thinking=False).
            Multiple choice: one forward pass, argmax over the logits of the option letters at the first answer
            position (batch 1, no rotation). Open: greedy decoding over the KV cache, at most 256 new tokens
            (validate.py's max_tokens). Writes predictions JSON: [{"id", "answer", ...diagnostics}].
            The model is called with input ids only: the checkpoint's fa3.py reads the query length from the
            head axis when an attention mask is supplied (a bug in the released code that its mask-free
            position-ids path never hits), and for one unpadded sequence the all-ones mask carries nothing.
  score     Score a predictions file with tasks/maple-vlm/validate.score through the gpt-6-luna answer judge, which
            decides every question (--judge, OPENAI_API_KEY; judge_model gpt-6-luna), and print accuracy overall
            and by question type. The judge is the only scorer, so --judge is required; 900 paid calls per split.

    CUDA_VISIBLE_DEVICES=7 python text_only_floor.py generate --data data/maple-vlm/private/test.jsonl --output work/maple-floor/heldout.json
    uv run python tasks/maple-vlm/pipeline/text_only_floor.py score --data data/maple-vlm/private/test.jsonl --predictions work/maple-floor/heldout.json --judge
"""
import argparse
import ast
import json
from pathlib import Path
import sys
import time

MAPLE_REPO = "deepgrove/maple-preview"
MAPLE_REVISION = "ac1ddd79d2b5cb4406f5d2bebdf95406ce505a07"
MAX_NEW_TOKENS = 256  # validate.build_request's max_tokens
THINKING_OFF = "\n</think>\n\n"  # closes the <think> block the chat template opens for the assistant turn
TASK_DIR = Path(__file__).resolve().parents[1]
REPO = TASK_DIR.parents[1]  # frontier_ml for validate.py's imports


def load_rows(path):
    return [json.loads(line) for line in Path(path).read_text().splitlines() if line.strip()]


def question_text(row):
    """validate.build_request's text part, verbatim, with the image parts left out."""
    for path in (str(REPO), str(TASK_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import validate
    _, payload = validate.build_request({**row, "images": []}, TASK_DIR)
    return payload["messages"][0]["content"][0]["text"]


def options_of(row):
    options = row.get("options", [])
    return ast.literal_eval(options) if isinstance(options, str) else options


def generate(args):
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from huggingface_hub import snapshot_download
    path = snapshot_download(MAPLE_REPO, revision=MAPLE_REVISION, local_files_only=True)
    tokenizer = AutoTokenizer.from_pretrained(path)
    model = AutoModelForCausalLM.from_pretrained(path, trust_remote_code=True, dtype=torch.bfloat16, device_map={"": 0}).eval()
    letters = {}
    for letter in "ABCDEFGHIJ":
        ids = tokenizer.encode(letter, add_special_tokens=False)
        assert len(ids) == 1, (letter, ids)
        letters[letter] = ids[0]
    rows = load_rows(args.data)
    if args.limit:
        rows = rows[: args.limit]
    eos = {tokenizer.convert_tokens_to_ids("<|im_end|>"), tokenizer.eos_token_id}

    def greedy(input_ids):
        """Greedy continuation over the KV cache; stops at <|im_end|>/eos or MAX_NEW_TOKENS. No attention mask."""
        out = model(input_ids=input_ids, use_cache=True)
        past, next_id, generated = out.past_key_values, out.logits[0, -1].argmax().item(), []
        while len(generated) < MAX_NEW_TOKENS and next_id not in eos:
            generated.append(next_id)
            out = model(input_ids=torch.tensor([[next_id]], device=input_ids.device), past_key_values=past, use_cache=True)
            past, next_id = out.past_key_values, out.logits[0, -1].argmax().item()
        return generated

    predictions, started = [], time.monotonic()
    for index, row in enumerate(rows):
        text = question_text(row)
        prompt = tokenizer.apply_chat_template([{"role": "user", "content": text}], tokenize=False, add_generation_prompt=True) + THINKING_OFF
        input_ids = tokenizer(prompt, return_tensors="pt", add_special_tokens=False)["input_ids"].to(model.device)
        record = {"id": row["id"], "question_type": row["question_type"], "prompt_tokens": int(input_ids.shape[1])}
        with torch.inference_mode():
            if row["question_type"] == "multiple-choice":
                logits = model(input_ids=input_ids).logits[0, -1].float()
                candidates = [chr(65 + i) for i in range(len(options_of(row)))]
                scores = {letter: logits[letters[letter]].item() for letter in candidates}
                record.update(answer=max(scores, key=scores.get), letter_logits=scores)
            else:
                new_tokens = greedy(input_ids)
                record.update(answer=tokenizer.decode(new_tokens, skip_special_tokens=True).strip(),
                              completion_tokens=len(new_tokens))
        predictions.append(record)
        if (index + 1) % 50 == 0 or index + 1 == len(rows):
            print(f"{index + 1}/{len(rows)} in {time.monotonic() - started:.0f}s", flush=True)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps({"model": MAPLE_REPO, "revision": MAPLE_REVISION, "data": str(args.data),
                                             "thinking": "off (empty <think> block)", "multiple_choice": "argmax over option-letter logits, batch 1, no rotation",
                                             "open": f"greedy, max_new_tokens {MAX_NEW_TOKENS}", "elapsed_seconds": time.monotonic() - started,
                                             "predictions": predictions}, indent=1) + "\n")
    print("wrote", args.output)


def score(args):
    if not args.judge:
        raise SystemExit("validate.score has no scorer but the gpt-6-luna judge; pass --judge (OPENAI_API_KEY)")
    for path in (str(REPO), str(TASK_DIR)):
        if path not in sys.path:
            sys.path.insert(0, path)
    import validate
    rows = load_rows(args.data)
    saved = json.loads(Path(args.predictions).read_text())
    by_id = {p["id"]: p for p in saved["predictions"]}
    rows = [row for row in rows if row["id"] in by_id]
    predictions = [{"id": row["id"], "answer": by_id[row["id"]]["answer"]} for row in rows]
    from frontier_ml.runtime.judge import judge_from_config
    config = {"judge_model": validate.JUDGE_MODEL, "judge_timeout_seconds": 120}
    config["judge"] = judge_from_config(config)
    result = validate.score(rows, predictions, config)
    documents = result["details"]["documents"]
    by_type = {}
    for row, document in zip(rows, documents, strict=True):
        counts = by_type.setdefault(row["question_type"], [0, 0])
        counts[0] += int(document["correct"])
        counts[1] += 1
    summary = {"data": str(args.data), "predictions": str(args.predictions), "samples": len(rows), "score": result["score"],
               "correct": sum(int(d["correct"]) for d in documents), "by_type": {k: {"correct": v[0], "total": v[1], "accuracy": v[0] / v[1]} for k, v in by_type.items()},
               "answer_judge": result["details"]["answer_judge"],
               "open_records": [{k: d.get(k) for k in ("id", "correct", "judged")} | {"reply": by_id[d["id"]]["answer"][:200]}
                                for d in documents if d.get("question_type") == "open"]}
    if args.output:
        Path(args.output).write_text(json.dumps(summary, indent=1) + "\n")
    print(json.dumps({k: v for k, v in summary.items() if k != "open_records"}, indent=1))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    commands = parser.add_subparsers(dest="command", required=True)
    g = commands.add_parser("generate")
    g.add_argument("--data", required=True)
    g.add_argument("--output", required=True)
    g.add_argument("--limit", type=int, default=0)
    s = commands.add_parser("score")
    s.add_argument("--data", required=True)
    s.add_argument("--predictions", required=True)
    s.add_argument("--judge", action="store_true", help="score through the gpt-6-luna answer judge (OPENAI_API_KEY); required")
    s.add_argument("--output")
    args = parser.parse_args(argv)
    (generate if args.command == "generate" else score)(args)


if __name__ == "__main__":
    main()
