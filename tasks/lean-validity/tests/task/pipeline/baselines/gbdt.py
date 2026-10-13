"""Syntactic feature GBDT baseline (organizer side).

Features are surface statistics of the proof text only (lengths, tactic counts, hint
counts, hashed identifier bag). Train on an organizer "train" split labeled with Lean,
answer yes above a threshold chosen on validation, report held-out macro F1.

    python gbdt.py train TRAIN.jsonl MODEL.json
    python gbdt.py eval MODEL.json VALIDATION.jsonl HELDOUT.jsonl   (threshold fit on validation)
"""
from __future__ import annotations

import hashlib
import json
import math
import re
import sys

TACTICS = ["nlinarith", "linarith", "norm_num", "simp", "simp_all", "simp only", "omega", "ring_nf", "ring", "field_simp",
           "positivity", "decide", "aesop", "rcases", "obtain", "induction", "cases", "interval_cases", "have", "calc",
           "exact", "apply", "rw", "intro", "constructor", "use", "refine", "by_contra", "push_neg", "nlinarith!", "gcongr",
           "tauto", "contradiction", "exact?", "unfold", "subst", "specialize", "split_ifs", "by_cases", "norm_cast",
           "push_cast", "zify", "qify", "Finset.sum_range_succ", "div_le_iff", "sq_nonneg", "mul_pos", "sq_abs",
           "Real.sqrt", "<;>", "·", "try", "all_goals", "any_goals", "first", "repeat", "rfl", "trivial", "linarith!"]
HASH_BUCKETS = 256
HEADER_END = "open BigOperators Real Nat Topology Rat\n\n"


def split_text(text: str):
    body = text.split(HEADER_END, 1)[-1]
    match = re.search(r"theorem\s+\S+[\s\S]*?:= by", body)
    if not match:
        return "", body
    return body[:match.end()], body[match.end():]


def features(text: str) -> list[float]:
    statement, proof = split_text(text)
    lines = [l for l in proof.split("\n") if l.strip()]
    comment_lines = [l for l in lines if l.strip().startswith("--")]
    indents = [len(l) - len(l.lstrip()) for l in lines] or [0]
    hints = re.findall(r"\[([^\[\]]*)\]", proof)
    vec = [len(text), len(statement), len(proof), len(lines), len(comment_lines), len(comment_lines) / max(1, len(lines)),
           max(indents), sum(indents) / len(indents), len(hints), sum(h.count(",") + 1 for h in hints),
           len(re.findall(r"\d+", proof)), proof.count("("), proof.count("sorry"), len(re.findall(r"\bhave\b", proof)),
           len(set(re.findall(r"[A-Za-z_][\w.']*", proof))), sum(ord(c) > 127 for c in proof) / max(1, len(proof)),
           text.count("lemma ") + text.count("theorem ") - 1, len(proof) / max(1, len(statement))]
    for tactic in TACTICS:
        vec.append(len(re.findall(r"(?<![\w.])" + re.escape(tactic) + r"(?![\w'])", proof)))
    bag = [0.0] * HASH_BUCKETS
    for token in re.findall(r"[A-Za-z_][\w.']*", proof):
        bag[int(hashlib.md5(token.encode()).hexdigest(), 16) % HASH_BUCKETS] += 1
    return vec + bag


def train(train_path, model_path):
    import lightgbm as lgb
    import numpy as np
    rows = [json.loads(l) for l in open(train_path)]
    X = np.array([features(r["lean"]) for r in rows])
    y = np.array([int(r["valid"]) for r in rows])
    model = lgb.LGBMClassifier(n_estimators=400, learning_rate=0.05, num_leaves=31, min_child_samples=10,
                               subsample=0.8, subsample_freq=1, colsample_bytree=0.8, verbose=-1)
    model.fit(X, y)
    model.booster_.save_model(model_path)
    print(f"trained on {len(rows)} rows ({y.mean():.3f} valid)")


def load_predictor(model_path):
    import lightgbm as lgb
    booster = lgb.Booster(model_file=model_path)
    return lambda text: float(booster.predict([features(text)])[0])


def evaluate(model_path, val_path, held_path):
    """Threshold chosen on validation, macro F1 reported on held-out."""
    import lightgbm as lgb
    sys.path.insert(0, __file__.rsplit("/", 1)[0])
    from binarize import evaluate as binary_eval
    booster = lgb.Booster(model_file=model_path)
    val = [json.loads(l) for l in open(val_path)]
    held = [json.loads(l) for l in open(held_path)]
    predict = lambda rows: [float(p) for p in booster.predict([features(r["lean"]) for r in rows])]
    result = binary_eval(val, predict(val), held, predict(held))
    print(json.dumps(result))
    return result


if __name__ == "__main__":
    if sys.argv[1] == "train":
        train(sys.argv[2], sys.argv[3])
    else:
        evaluate(sys.argv[2], sys.argv[3], sys.argv[4])
