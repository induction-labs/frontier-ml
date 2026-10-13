"""Organizer fine-tuned classifier baseline (the "$50 route" in miniature, one GPU, ~1 hour).

Trains a sequence classifier (Qwen3 base + linear head, full fine-tune, bf16) on every
Lean-labeled attempt of the organizer-only train statements (disjoint from validation and
held-out), then answers yes above a threshold chosen on validation and reports held-out macro F1.

    python finetune.py train RUN_DIR OUT_DIR --model Qwen/Qwen3-1.7B
    python finetune.py eval OUT_DIR VALIDATION.jsonl HELDOUT.jsonl   (threshold fit on validation)
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import sys
import time

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.dirname(HERE))
sys.path.insert(0, os.path.dirname(os.path.dirname(HERE)))
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(HERE)))))

MAX_LEN = 2048
HEADER_END = "open BigOperators Real Nat Topology Rat\n\n"


def strip_header(text):
    return text.split(HEADER_END, 1)[-1]


def balance(rows, items_prover):
    """Keep, per (statement, source prover, kind) bucket, equally many valid and invalid rows
    (the evaluation's matched design), so the model cannot learn prover/edit priors."""
    buckets = {}
    for row in rows:
        buckets.setdefault((row["sid"], row["prover"], row["kind"]), ([], []))[0 if row["valid"] else 1].append(row)
    out, rng = [], random.Random(0)
    for pos, neg in buckets.values():
        k = min(len(pos), len(neg))
        out += rng.sample(pos, k) + rng.sample(neg, k)
    return out


def training_rows(run, balanced=False):
    """All double-compiled, agreeing labels for train-split statements."""
    read = lambda p: [json.loads(l) for l in open(os.path.join(run, p))]
    statements = {r["sid"]: r for r in read("statements.jsonl")}
    items = read("originals.jsonl") + read("mutants.jsonl")
    prover_of = {i["key"]: i["origin"] for i in items if i["kind"] == "original"}
    first = {r["key"]: r for r in read("check1_originals.jsonl") + read("check1_mutants.jsonl")}
    second = {r["key"]: r for r in read("check2.jsonl")}
    rows = []
    for item in items:
        if statements[item["sid"]]["split"] != "train":
            continue
        a = first.get(item["key"])
        b = second.get(item["key"])
        if not a or a.get("status") != "ok":
            continue
        if b is not None and (b.get("status") != "ok" or b["valid"] != a["valid"]):
            continue
        rows.append({"lean": item["lean"], "valid": bool(a["valid"]), "sid": item["sid"], "kind": item["kind"],
                     "prover": item["origin"] if item["kind"] == "original" else prover_of.get(item["parent"])})
    return balance(rows, prover_of) if balanced else rows


def train(run, out, model_name, epochs, lr, batch, balanced=False):
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    rows = training_rows(run, balanced)
    random.Random(0).shuffle(rows)
    print(f"{len(rows)} training rows, {sum(r['valid'] for r in rows) / len(rows):.3f} valid", flush=True)
    tok = AutoTokenizer.from_pretrained(model_name)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=1, torch_dtype=torch.bfloat16)
    model.config.pad_token_id = tok.pad_token_id
    model.cuda().train()
    model.gradient_checkpointing_enable()
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=0.0)
    steps = epochs * math.ceil(len(rows) / batch)
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda s: min(1.0, (s + 1) / 50) * max(0.0, 1 - s / steps))
    started, step = time.time(), 0
    for epoch in range(epochs):
        random.Random(epoch).shuffle(rows)
        for i in range(0, len(rows), batch):
            chunk = rows[i:i + batch]
            enc = tok([strip_header(r["lean"]) for r in chunk], truncation=True, max_length=MAX_LEN, padding=True,
                      return_tensors="pt").to("cuda")
            labels = torch.tensor([float(r["valid"]) for r in chunk], device="cuda")
            logits = model(**enc).logits.squeeze(-1).float()
            loss = torch.nn.functional.binary_cross_entropy_with_logits(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            sched.step()
            opt.zero_grad(set_to_none=True)
            step += 1
            if step % 50 == 0:
                print(f"epoch {epoch} step {step}/{steps} loss {loss.item():.4f} {time.time() - started:.0f}s", flush=True)
    model.save_pretrained(out, safe_serialization=True)
    tok.save_pretrained(out)
    json.dump({"base_model": model_name, "balanced": balanced, "rows": len(rows), "epochs": epochs, "lr": lr, "seconds": time.time() - started},
              open(os.path.join(out, "train_info.json"), "w"))


def evaluate(out, paths):
    """paths = [VALIDATION, HELDOUT]: threshold chosen on validation, macro F1 reported on held-out."""
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer
    from binarize import evaluate as binary_eval
    tok = AutoTokenizer.from_pretrained(out)
    model = AutoModelForSequenceClassification.from_pretrained(out, torch_dtype=torch.bfloat16).cuda().eval()

    def predict(rows):
        values = []
        with torch.no_grad():
            for i in range(0, len(rows), 32):
                chunk = rows[i:i + 32]
                enc = tok([strip_header(r["lean"]) for r in chunk], truncation=True, max_length=MAX_LEN, padding=True,
                          return_tensors="pt").to("cuda")
                values += torch.sigmoid(model(**enc).logits.squeeze(-1).float()).tolist()
        return values

    val, held = ([json.loads(l) for l in open(p)] for p in paths)
    started = time.time()
    held_scores = predict(held)
    rate = len(held) / (time.time() - started)
    result = binary_eval(val, predict(val), held, held_scores)
    result["items_per_second"] = rate
    print(json.dumps(result))


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("mode", choices=["train", "eval"])
    parser.add_argument("args", nargs="+")
    parser.add_argument("--model", default="Qwen/Qwen3-1.7B")
    parser.add_argument("--epochs", type=int, default=2)
    parser.add_argument("--lr", type=float, default=1e-5)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--balanced", action="store_true", help="matched per-bucket training rows")
    a = parser.parse_args()
    if a.mode == "train":
        train(a.args[0], a.args[1], a.model, a.epochs, a.lr, a.batch, a.balanced)
    else:
        evaluate(a.args[0], a.args[1:])
