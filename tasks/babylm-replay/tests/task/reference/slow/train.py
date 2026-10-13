"""Train a byte-level BPE tokenizer and a causal LM on $CORPUS_DIR within the replay wall-clock budget.

Writes config.json, tokenizer.json, model.safetensors and train_log.json to $OUTPUT_DIR.
Recipes:
  naive: GPT-2-small (124M-style), AdamW lr 6e-4, wd 0.1, dropout 0.1, cosine over 10 epochs.
  slow : Slowrun-style port: Muon (Polar Express + MuonEq-R + cautious WD, from qlabs-eng/slowrun, MIT),
         heavy weight decay, dropout, overparameterized model, many epochs with a time-aware
         sqrt warmdown, per-epoch document shuffling and an EMA of the weights.
"""
import argparse
import glob
import json
import math
import os
import random
import sys
import time

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import lm  # noqa: E402

START = float(os.environ.get("REPLAY_START", time.time()))
BUDGET = float(os.environ.get("TRAIN_BUDGET_SECONDS", 3600))


def elapsed():
    return time.time() - START


def log(msg):
    print(f"[{elapsed():7.1f}s] {msg}", flush=True)


RECIPES = {
    "naive": dict(arch="gpt2", n_layer=12, n_head=12, n_embd=768, context=512, dropout=0.1, vocab=16384,
                  batch=64, lr=6e-4, wd=0.1, epochs=10, warmup=0.02, reserve=150),
    # Tuned port (pilot "v1"): 65K-token batch, Muon lr 0.03, cautious WD 1.0, dropout 0.1, ~42 epochs.
    "slow": dict(arch="slow", n_layer=16, n_head=8, n_embd=1024, context=1024, dropout=0.1, vocab=16384,
                 logit_cap=15.0, batch=64, accum=2, matrix_lr=0.03, embed_lr=0.1, head_lr=0.002, scalar_lr=0.02,
                 wd=1.0, max_epochs=200, warmdown=0.45, ema_per_epoch=0.15, reserve=150),
    # Pilot "v2": Slowrun tiny-track batch/LRs (262K tokens, lr 0.032, wd 0.8). Colder: fits the corpus
    # better but overfits; held-out 1.48 bpb vs 1.30 for "slow". Kept for the record.
    "slow_v2": dict(arch="slow", n_layer=16, n_head=8, n_embd=1024, context=1024, dropout=0.1, vocab=16384,
                    logit_cap=15.0, batch=256, accum=8, matrix_lr=0.032, embed_lr=0.12, head_lr=0.0008, scalar_lr=0.02,
                    wd=0.8, max_epochs=200, warmdown=0.4, ema_per_epoch=0.15, reserve=150),
}
RECIPES["slow_v1"] = RECIPES["slow"]  # name used by the pilot run directories


# --------------------------------------------------------------------------------------------
# Data
# --------------------------------------------------------------------------------------------

def read_documents(corpus, target=8192):
    """Split each corpus file into contiguous pseudo-documents of ~8 KB (Wikipedia at article headers)."""
    docs = []
    for path in sorted(glob.glob(os.path.join(corpus, "*.txt"))):
        with open(path, encoding="utf-8") as f:
            lines = f.read().splitlines(keepends=True)
        wiki = "simple_wiki" in os.path.basename(path)
        cur, size = [], 0
        for line in lines:
            header = wiki and line.startswith("= = = ")
            if cur and (size >= target or (header and size >= 1024)):
                docs.append("".join(cur))
                cur, size = [], 0
            cur.append(line)
            size += len(line.encode("utf-8"))
        if cur:
            docs.append("".join(cur))
    return docs


def train_tokenizer(docs, vocab):
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False, use_regex=True)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=vocab, special_tokens=[lm.BOS], min_frequency=2,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    tok.train_from_iterator(docs, trainer=trainer)
    return tok


# --------------------------------------------------------------------------------------------
# Muon (adapted from qlabs-eng/slowrun, MIT License)
# --------------------------------------------------------------------------------------------

POLAR = [(8.156554524902461, -22.48329292557795, 15.878769915207462),
         (4.042929935166739, -2.808917465908714, 0.5000178451051316),
         (3.8916678022926607, -2.772484153217685, 0.5060648178503393),
         (3.285753657755655, -2.3681294933425376, 0.46449024233003106),
         (2.3465413258596377, -1.7097828382687081, 0.42323551169305323)]


@torch.compile(dynamic=False, fullgraph=True)
def muon_step(grads, params, mom, second, momentum, lr, wd, beta2, red_dim: int):
    mom.lerp_(grads, 1 - momentum)
    g = grads.lerp_(mom, momentum)
    g = g / g.float().norm(dim=-1, keepdim=True).clamp_min(1e-7).to(g.dtype)
    X = g.bfloat16()
    X = X / (X.norm(dim=(-2, -1), keepdim=True) * 1.02 + 1e-6)
    if g.size(-2) > g.size(-1):
        for a, b, c in POLAR:
            A = X.mT @ X
            X = a * X + X @ (b * A + c * (A @ A))
    else:
        for a, b, c in POLAR:
            A = X @ X.mT
            X = a * X + (b * A + c * (A @ A)) @ X
    g = X.float()
    v_mean = g.square().mean(dim=red_dim, keepdim=True)
    n = g.size(red_dim)
    v_norm = (v_mean.sum(dim=(-2, -1), keepdim=True) * n).sqrt()
    second.lerp_(v_mean, 1 - beta2)
    step = second.clamp_min(1e-10).rsqrt()
    v_new = ((v_mean * n) * step.square()).sum(dim=(-2, -1), keepdim=True).sqrt()
    g = g * (step * (v_norm / v_new.clamp_min(1e-10)))
    mask = (g * params) >= 0
    params.sub_(lr * g + lr * wd * params * mask)


class Muon:
    def __init__(self, params, lr, wd, momentum=0.95, beta2=0.95):
        self.groups = {}
        for p in params:
            self.groups.setdefault(tuple(p.shape), []).append(p)
        self.state = {}
        for shape, ps in self.groups.items():
            red = -1 if shape[-2] >= shape[-1] else -2
            second_shape = (len(ps), shape[0], 1) if red == -1 else (len(ps), 1, shape[1])
            self.state[shape] = (torch.zeros(len(ps), *shape, device=ps[0].device), torch.zeros(second_shape, device=ps[0].device), red)
        self.lr, self.wd, self.momentum, self.beta2 = lr, wd, momentum, beta2
        self.scale = 1.0

    @torch.no_grad()
    def step(self):
        for shape, ps in self.groups.items():
            mom, second, red = self.state[shape]
            grads = torch.stack([p.grad.float() for p in ps])
            params = torch.stack([p.data.float() for p in ps])
            lr = torch.tensor(self.lr * self.scale * max(1.0, shape[0] / shape[1]) ** 0.5, device=params.device)
            muon_step(grads, params, mom, second, torch.tensor(self.momentum, device=params.device), lr,
                      torch.tensor(self.wd, device=params.device), torch.tensor(self.beta2, device=params.device), red)
            for p, new in zip(ps, params.unbind(0)):
                p.data.copy_(new)

    def zero_grad(self):
        for ps in self.groups.values():
            for p in ps:
                p.grad = None


# --------------------------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--recipe", choices=sorted(RECIPES), required=True)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = dict(RECIPES[args.recipe])
    corpus, out = os.environ["CORPUS_DIR"], os.environ["OUTPUT_DIR"]
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    dev = "cuda"
    torch.set_num_threads(min(8, os.cpu_count() or 1))

    docs = read_documents(corpus)
    log(f"{len(docs)} documents, {sum(len(d.encode()) for d in docs)/1e6:.1f} MB")
    tok = train_tokenizer(docs, cfg["vocab"])
    tok.save(os.path.join(out, "tokenizer.json"))
    bos = tok.token_to_id(lm.BOS)
    ids = [e.ids for e in tok.encode_batch(docs)]
    rng = random.Random(args.seed)
    order = list(range(len(docs)))
    rng.shuffle(order)
    n_val = max(20, len(docs) // 200)
    val_docs, train_docs = [ids[i] for i in order[:n_val]], [ids[i] for i in order[n_val:]]
    val_bytes = sum(len(docs[i].encode()) for i in order[:n_val])
    ntok = sum(map(len, train_docs))
    log(f"tokenizer done; {ntok/1e6:.2f}M train tokens, {sum(map(len, val_docs))/1e3:.0f}K val tokens")
    train_docs = [np.array([bos] + d, dtype=np.int32) for d in train_docs]

    T, B = cfg["context"], cfg["batch"]
    cfg["vocab"] = tok.get_vocab_size()
    model = lm.build_model(cfg).to(dev)
    nparams = sum(p.numel() for p in model.parameters())
    log(f"model {cfg['arch']} {nparams/1e6:.1f}M params")

    def epoch_batches(epoch):
        r = np.random.default_rng(args.seed * 1000 + epoch)
        perm = r.permutation(len(train_docs))
        stream = np.concatenate([train_docs[i] for i in perm])
        off = int(r.integers(0, T))
        nseq = (len(stream) - off - 1) // T
        x = torch.from_numpy(stream[off:off + nseq * T + 1].astype(np.int64)).to(dev)
        inputs, targets = x[:-1].view(nseq, T), x[1:].view(nseq, T)
        sel = torch.from_numpy(r.permutation(nseq)).to(dev)
        for i in range(0, nseq - B + 1, B):
            s = sel[i:i + B]
            yield inputs[s], targets[s]

    steps_per_epoch = (ntok // T) // B
    compiled = torch.compile(model)

    def loss_fn(x, y):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            logits = compiled(x)
        return F.cross_entropy(logits.view(-1, logits.size(-1)), y.view(-1))

    @torch.no_grad()
    def evaluate(m):
        m.eval()
        total = 0.0
        for d in val_docs:
            x = torch.tensor([bos] + d, device=dev)
            for a, j0, j1 in lm.score_windows(len(d), T, T // 2):
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    logits = m(x[a:j1 - 1][None])[0]
                lp = torch.log_softmax(logits.float(), -1)[j0 - 1 - a:]
                total -= lp.gather(1, x[j0:j1, None]).sum().item()
        m.train()
        return total / math.log(2) / val_bytes

    history = []
    if cfg["arch"] == "gpt2":
        decay = [p for n, p in model.named_parameters() if p.dim() >= 2]
        other = [p for n, p in model.named_parameters() if p.dim() < 2]
        opt = torch.optim.AdamW([dict(params=decay, weight_decay=cfg["wd"]), dict(params=other, weight_decay=0.0)],
                                lr=cfg["lr"], betas=(0.9, 0.95), fused=True)
        total = cfg["epochs"] * steps_per_epoch
        warm = int(cfg["warmup"] * total)
        step = 0
        for epoch in range(cfg["epochs"]):
            for x, y in epoch_batches(epoch):
                if elapsed() > BUDGET - cfg["reserve"]:
                    break
                lr = cfg["lr"] * (step / max(1, warm) if step < warm else 0.1 + 0.9 * 0.5 * (1 + math.cos(math.pi * (step - warm) / max(1, total - warm))))
                for g in opt.param_groups:
                    g["lr"] = lr
                loss = loss_fn(x, y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                opt.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 100 == 0:
                    log(f"step {step}/{total} epoch {epoch} loss {loss.item():.4f} lr {lr:.2e}")
            bpb = evaluate(model)
            history.append(dict(epoch=epoch, step=step, val_bpb=bpb, t=elapsed()))
            log(f"epoch {epoch} val_bpb {bpb:.4f}")
            if elapsed() > BUDGET - cfg["reserve"]:
                log("time budget reached")
                break
        final = model
    else:
        matrices = [p for n, p in model.named_parameters() if p.dim() == 2 and (n.startswith("blocks.") or n.startswith("ve_projs."))
                    and ".gate." not in n and "ve_gate" not in n]
        mids = {id(p) for p in matrices}
        gates = [p for n, p in model.named_parameters() if ("gate.weight" in n and "c_gate" not in n)]
        adam_groups = [dict(params=[model.wte.weight], lr=cfg["embed_lr"], weight_decay=cfg["wd"]),
                       dict(params=[model.lm_head.weight], lr=cfg["head_lr"], weight_decay=cfg["wd"]),
                       dict(params=gates, lr=cfg["scalar_lr"], weight_decay=0.0),
                       dict(params=[model.resid_lambdas, model.skip_weights], lr=cfg["scalar_lr"] * 0.1, weight_decay=0.0),
                       dict(params=[model.x0_lambdas], lr=cfg["scalar_lr"], weight_decay=0.0)]
        seen = mids | {id(p) for g in adam_groups for p in g["params"]}
        assert all(id(p) in seen for p in model.parameters()), "unassigned parameters"
        for g in adam_groups:
            g["initial_lr"] = g["lr"]
        adam = torch.optim.AdamW(adam_groups, betas=(0.8, 0.95), eps=1e-10, fused=True)
        muon = Muon(matrices, lr=cfg["matrix_lr"], wd=cfg["wd"])
        ema = {n: p.detach().clone().float() for n, p in model.state_dict().items()}
        ema_every = 10
        ema_beta = cfg["ema_per_epoch"] ** (ema_every / steps_per_epoch)
        step, planned, t_probe, last_mult = 0, None, None, 1.0
        epoch = 0
        while planned is None or step < planned:
            for x, y in epoch_batches(epoch):
                if step == 20:
                    torch.cuda.synchronize()
                    t_probe = time.time()
                if step == 70:
                    torch.cuda.synchronize()
                    per_step = (time.time() - t_probe) / 50
                    remaining = BUDGET - cfg["reserve"] - elapsed()
                    planned = min(cfg["max_epochs"] * steps_per_epoch, step + int(remaining / per_step * 0.97))
                    if os.environ.get("FIXED_STEPS"):  # pilot on a shared GPU: compute-equivalent step budget
                        planned = int(os.environ["FIXED_STEPS"])
                    log(f"{per_step*1000:.0f} ms/step -> planned {planned} steps = {planned/steps_per_epoch:.1f} epochs")
                    t_replan = (step, time.time())
                elif planned is not None and step % 100 == 0 and not os.environ.get("FIXED_STEPS"):
                    # Re-plan from the recent step rate so the warmdown still completes if the GPU slows down.
                    per_step = (time.time() - t_replan[1]) / max(1, step - t_replan[0])
                    t_replan = (step, time.time())
                    remaining = BUDGET - cfg["reserve"] - elapsed()
                    planned = max(step + 1, min(cfg["max_epochs"] * steps_per_epoch, step + int(remaining / per_step * 0.97)))
                progress = step / planned if planned else 0.0
                wd_frac = 1 - cfg["warmdown"]
                mult = 1.0 if progress < wd_frac else max(0.0, (1 - progress) / cfg["warmdown"]) ** 0.5
                mult = last_mult = min(mult, last_mult)  # never re-raise the LR after a re-plan
                for g in adam.param_groups:
                    g["lr"] = g["initial_lr"] * mult
                muon.scale = mult
                muon.momentum = min(0.95, 0.85 + 0.1 * step / 300)
                accum = cfg.get("accum", 1)
                for xs, ys in zip(x.chunk(accum), y.chunk(accum)):
                    loss = loss_fn(xs, ys) / accum
                    loss.backward()
                loss = loss * accum
                adam.step()
                muon.step()
                adam.zero_grad(set_to_none=True)
                muon.zero_grad()
                step += 1
                if step % ema_every == 0:
                    with torch.no_grad():
                        beta = ema_beta if progress > 0.5 else 0.0
                        for n, p in model.state_dict().items():
                            ema[n].lerp_(p.float(), 1 - beta)
                if step % 200 == 0:
                    log(f"step {step}/{planned} epoch {epoch} loss {loss.item():.4f} lr_mult {mult:.3f}")
                if planned is not None and step >= planned:
                    break
                if elapsed() > BUDGET - cfg["reserve"]:
                    planned = step
                    log("time budget reached")
                    break
            if epoch % 5 == 4 or (planned is not None and step >= planned):
                bpb = evaluate(model)
                history.append(dict(epoch=epoch, step=step, val_bpb=bpb, t=elapsed()))
                log(f"epoch {epoch} val_bpb {bpb:.4f}")
            epoch += 1
        model.load_state_dict({n: v.to(p.dtype) for (n, v), p in zip(ema.items(), model.state_dict().values())})
        bpb = evaluate(model)
        history.append(dict(epoch="ema", step=step, val_bpb=bpb, t=elapsed()))
        log(f"EMA val_bpb {bpb:.4f}")
        final = model

    from safetensors.torch import save_file
    state = {k: v.detach().float().contiguous().cpu() for k, v in final.state_dict().items()}
    save_file(state, os.path.join(out, "model.safetensors"))
    with open(os.path.join(out, "config.json"), "w") as f:
        json.dump(cfg, f, indent=1)
    with open(os.path.join(out, "train_log.json"), "w") as f:
        json.dump(dict(recipe=args.recipe, seed=args.seed, params=nparams, train_tokens=ntok,
                       history=history, elapsed=elapsed()), f, indent=1)
    log("saved")


if __name__ == "__main__":
    main()
