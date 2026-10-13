"""Tiny CPU imitation recipe. Improve the encoder, teacher and training algorithm.

python /task/public/starter/train.py --data /task/public/validation.jsonl --output /workspace/output
Requires numpy and safetensors in the development environment (pip install numpy safetensors).
The resulting files still need this script and server.py copied into output for review.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from safetensors.numpy import save_file

# /task on the agent; --task-code supports development from the repository.
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument("--task-code", default="/task")
parser.add_argument("--data", type=Path, default=Path("/task/public/validation.jsonl"))
parser.add_argument("--output", required=True, type=Path)
parser.add_argument("--seed", type=int, default=321)
parser.add_argument("--epochs", type=int, default=25)
args = parser.parse_args()
sys.path.insert(0,args.task_code)
from pokemon_random_battle import evaluate

examples = []
def observe(observations, actions):
    for observation, action in zip(observations, actions, strict=True):
        examples.append((np.asarray(observation["action_features"],dtype=np.float32),action))

# Use independent training chance/team seeds, not the fixed public validation seed.
teacher = evaluate(None,args.data,seed=args.seed,baseline="tactical",observer=observe)
rng = np.random.default_rng(args.seed)
w = rng.normal(0,0.01,20).astype(np.float32)
initial = w.copy()
losses = []
for epoch in range(args.epochs):
    total = 0.0
    for index in rng.permutation(len(examples)):
        x, target = examples[index]
        logits = x@w; logits -= logits.max()
        probability = np.exp(logits); probability /= probability.sum()
        total -= float(np.log(max(probability[target],1e-12)))
        gradient = x.T@probability - x[target]
        w -= 0.03 * np.clip(gradient,-5,5)
    losses.append(total/len(examples))
args.output.mkdir(parents=True,exist_ok=True)
save_file({"action_weight":w},args.output/"model.safetensors")
(args.output/"training.json").write_text(json.dumps({"initialization":"random normal, mean=0 std=0.01",
    "seed":args.seed,"examples":len(examples),"epochs":args.epochs,"mean_cross_entropy":losses,
    "weight_change_l2":float(np.linalg.norm(w-initial)),"teacher_points":teacher["score"],
    "method":"imitation of organizer heuristic on newly generated development battles"},indent=2)+"\n")
print(json.dumps({"examples":len(examples),"final_loss":losses[-1],"output":str(args.output)}))
