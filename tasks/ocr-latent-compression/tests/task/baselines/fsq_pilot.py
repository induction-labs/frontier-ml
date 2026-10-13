"""Organizer FSQ-bottleneck pilot for the kill criterion (>= 0.2 on the sparsest bucket at 2 KB).

Frozen DeepEncoder in Base mode (1024 pad, 256 x 1280 projector outputs) -> transformer
encoder -> FSQ with 16 levels on 16 dims per token (256 x 16 x 4 bits = 2048 bytes, fixed
rate, no entropy coder) -> transformer decoder -> 256 embeddings in the vision-token
positions of the DeepSeek-OCR decoder (frozen, optional LoRA on attention projections).
Trained with next-token cross-entropy on rendered page text plus an auxiliary feature MSE.

    python fsq_pilot.py prep  --train-dir TRAIN --cache CACHE
    python fsq_pilot.py train --train-dir TRAIN --cache CACHE --out RUN --steps 3000 [--lora 16]
    python fsq_pilot.py score --run RUN --data VALIDATION_JSONL --out RUN/score.json
"""
from __future__ import annotations

import argparse
import copy
import json
import math
from pathlib import Path
import random
import sys
import time

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))
sys.path.insert(0, str(HERE.parent))
from deepseek_ocr_ref import EOS_ID, VisionDecoder, VisionEncoder, load_model, preprocess  # noqa: E402

TOKENS, DIM = 256, 1280
BUCKET_WEIGHTS = {"sparse": 0.4, "medium": 0.3, "dense": 0.2, "very_dense": 0.1}


class FSQBottleneck(nn.Module):
    def __init__(self, width=512, depth=4, code_dim=16, levels=16):
        super().__init__()
        self.code_dim, self.levels = code_dim, levels
        layer = nn.TransformerEncoderLayer(width, 8, 4 * width, dropout=0.0, batch_first=True, norm_first=True)
        self.inp = nn.Sequential(nn.LayerNorm(DIM), nn.Linear(DIM, width))
        self.pos_in = nn.Parameter(torch.zeros(TOKENS, width))
        self.enc = nn.TransformerEncoder(layer, depth)
        self.to_code = nn.Linear(width, code_dim)
        self.from_code = nn.Linear(code_dim, width)
        self.pos_out = nn.Parameter(torch.zeros(TOKENS, width))
        self.dec = nn.TransformerEncoder(copy.deepcopy(layer), depth)
        self.out = nn.Sequential(nn.LayerNorm(width), nn.Linear(width, DIM))

    def quantize(self, z):
        half = (self.levels - 1) / 2
        z = torch.tanh(z) * half
        q = (torch.round(z - 0.5) + 0.5) if self.levels % 2 == 0 else torch.round(z)
        q = q.clamp(-half, half)
        return z + (q - z).detach(), (q + half).round().long()

    def encode(self, feats):
        h = self.enc(self.inp(feats) + self.pos_in)
        return self.quantize(self.to_code(h))

    def decode_codes(self, q):
        return self.out(self.dec(self.from_code(q) + self.pos_out))

    def forward(self, feats):
        q, codes = self.encode(feats)
        return self.decode_codes(q), codes

    def codes_to_values(self, codes):
        return codes.float() - (self.levels - 1) / 2


def payload_bytes(codes, levels):
    bits = int(math.log2(levels))
    flat = codes.reshape(-1).cpu().numpy().astype(np.uint32)
    matrix = ((flat[:, None] >> np.arange(bits - 1, -1, -1)) & 1).astype(np.uint8)
    return np.packbits(matrix.reshape(-1)).tobytes()


class LoRALinear(nn.Module):
    def __init__(self, base, rank):
        super().__init__()
        self.base = base
        self.a = nn.Parameter(torch.randn(rank, base.in_features, device=base.weight.device) / math.sqrt(base.in_features))
        self.b = nn.Parameter(torch.zeros(base.out_features, rank, device=base.weight.device))

    def forward(self, x):
        return self.base(x) + (x.to(self.a.dtype) @ self.a.T @ self.b.T).to(x.dtype)


def add_lora(model, rank):
    params = []
    for layer in model.model.layers:
        for name in ("q_proj", "k_proj", "v_proj", "o_proj"):
            wrapped = LoRALinear(getattr(layer.self_attn, name), rank)
            setattr(layer.self_attn, name, wrapped)
            params += [wrapped.a, wrapped.b]
    return params


def prep(args):
    rows = [json.loads(l) for l in (args.train_dir / "train.jsonl").read_text().splitlines()]
    tokenizer, model = load_model()
    encoder = VisionEncoder(model)
    args.cache.mkdir(parents=True, exist_ok=True)
    feats = np.lib.format.open_memmap(args.cache / "features.npy", mode="w+", dtype=np.float16, shape=(len(rows), TOKENS, DIM))
    for start in range(0, len(rows), 8):
        batch = rows[start:start + 8]
        pixels = torch.stack([preprocess(Image.open(args.train_dir / r["image"]), "base") for r in batch])
        feats[start:start + len(batch)] = encoder.features(pixels).float().cpu().numpy().astype(np.float16)
        if start % 2000 == 0:
            print("features", start, flush=True)
    feats.flush()
    tokens = [tokenizer.encode(r["text"], add_special_tokens=False)[:args.max_text_tokens - 1] + [EOS_ID] for r in rows]
    (args.cache / "meta.json").write_text(json.dumps({"rows": [{"bucket": r["bucket"], "kind": r["kind"]} for r in rows],
                                                      "tokens": tokens}))
    print("prepared", len(rows))


def lm_loss(model, decoder_ref, grids, token_lists):
    """Teacher-forced CE on the text given [prefix, vision grid layout, suffix]."""
    embed = model.model.embed_tokens
    device = grids.device
    prefix = embed(torch.tensor(decoder_ref.prefix_ids, device=device))
    suffix = embed(torch.tensor(decoder_ref.suffix_ids, device=device))
    sequences, labels = [], []
    for grid, tokens in zip(grids, token_lists):
        ids = torch.tensor(tokens, device=device)
        context = torch.cat([prefix, decoder_ref.layout(grid).to(prefix.dtype), suffix])
        sequences.append(torch.cat([context, embed(ids)]))
        labels.append(torch.cat([torch.full((context.shape[0] - 1,), -100, device=device), ids, torch.tensor([-100], device=device)]))
    length = max(s.shape[0] for s in sequences)
    batch = torch.zeros(len(sequences), length, DIM, dtype=prefix.dtype, device=device)
    mask = torch.zeros(len(sequences), length, dtype=torch.long, device=device)
    target = torch.full((len(sequences), length), -100, dtype=torch.long, device=device)
    for i, (s, l) in enumerate(zip(sequences, labels)):
        batch[i, :s.shape[0]], mask[i, :s.shape[0]], target[i, :l.shape[0]] = s, 1, l
    out = decoder_ref.decoder_forward(model.model, inputs_embeds=batch, attention_mask=mask, use_cache=False, return_dict=True)
    hidden = out.last_hidden_state
    keep = target != -100
    logits = model.lm_head(hidden[keep]).float()
    return F.cross_entropy(logits, target[keep])


def train(args):
    torch.manual_seed(0)
    random.seed(0)
    meta = json.loads((args.cache / "meta.json").read_text())
    feats = np.load(args.cache / "features.npy", mmap_mode="r")
    tokenizer, model = load_model()
    for p in model.parameters():
        p.requires_grad_(False)
    model.model.sam_model = None
    model.model.vision_model = None
    model.train()  # the MoE inference path is wrapped in no_grad; the train path is differentiable
    decoder_ref = VisionDecoder(model, tokenizer)
    bottleneck = FSQBottleneck(code_dim=args.code_dim, levels=args.levels).cuda()
    groups = [{"params": bottleneck.parameters(), "lr": args.lr}]
    lora = add_lora(model, args.lora) if args.lora else []
    if lora:
        groups.append({"params": lora, "lr": args.lora_lr})
    optimizer = torch.optim.AdamW(groups, weight_decay=0.0)
    scale = float(np.sqrt(np.mean(np.asarray(feats[:256], dtype=np.float32) ** 2)))
    by_bucket = {}
    for index, row in enumerate(meta["rows"]):
        by_bucket.setdefault(row["bucket"], []).append(index)
    buckets, weights = zip(*BUCKET_WEIGHTS.items())
    log, started = [], time.time()
    for step in range(1, args.steps + 1):
        picks = [random.choice(by_bucket[random.choices(buckets, weights)[0]]) for _ in range(args.batch)]
        target = torch.from_numpy(np.stack([feats[i] for i in picks]).astype(np.float32)).cuda()
        warm = min(1.0, step / 200)
        for group, base in zip(optimizer.param_groups, (args.lr, args.lora_lr)):
            group["lr"] = base * warm * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        with torch.autocast("cuda", dtype=torch.bfloat16):
            grid, _ = bottleneck(target / scale)
            grid = grid * scale
            mse = F.mse_loss(grid.float(), target) / scale ** 2
            ce = lm_loss(model, decoder_ref, grid, [meta["tokens"][i] for i in picks])
            loss = ce + args.mse_weight * mse
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for g in groups for p in g["params"]], 1.0)
        optimizer.step()
        if step % 25 == 0:
            entry = {"step": step, "ce": round(ce.item(), 4), "mse": round(mse.item(), 4),
                     "elapsed": round(time.time() - started, 1)}
            log.append(entry)
            print(json.dumps(entry), flush=True)
    args.out.mkdir(parents=True, exist_ok=True)
    state = {"bottleneck": bottleneck.state_dict(), "scale": scale, "code_dim": args.code_dim, "levels": args.levels,
             "lora_rank": args.lora, "lora": {}}
    if lora:
        state["lora"] = {f"{i}.{n}.{w}": getattr(getattr(layer.self_attn, n), w).detach().cpu()
                         for i, layer in enumerate(model.model.layers) for n in ("q_proj", "k_proj", "v_proj", "o_proj")
                         for w in ("a", "b")}
    torch.save(state, args.out / "pilot.pt")
    (args.out / "train_log.json").write_text(json.dumps({"args": {k: str(v) for k, v in vars(args).items()}, "log": log}, indent=2))


@torch.no_grad()
def evaluate(args):
    from ocr_text import ned, normalize
    state = torch.load(args.run / "pilot.pt", map_location="cuda")
    tokenizer, model = load_model()
    encoder = VisionEncoder(model)
    if state["lora_rank"]:
        add_lora(model, state["lora_rank"])
        for key, value in state["lora"].items():
            i, name, w = key.split(".")
            getattr(getattr(model.model.layers[int(i)].self_attn, name), w).data.copy_(value.cuda())
    model.eval()
    decoder = VisionDecoder(model, tokenizer)
    bottleneck = FSQBottleneck(code_dim=state["code_dim"], levels=state["levels"]).cuda().eval()
    bottleneck.load_state_dict(state["bottleneck"])
    rows = [json.loads(l) for l in args.data.read_text().splitlines() if l.strip()]
    records, sizes = [], []
    for start in range(0, len(rows), 16):
        batch = rows[start:start + 16]
        pixels = torch.stack([preprocess(Image.open(args.data.parent / r["image"]), "base") for r in batch])
        feats = encoder.features(pixels).float()
        _, codes = bottleneck.encode(feats / state["scale"])
        payloads = [payload_bytes(c, state["levels"]) for c in codes]
        sizes += [len(p) for p in payloads]
        # Decode strictly from the payload codes, as a phase-B server would.
        values = torch.stack([bottleneck.codes_to_values(c) for c in codes])
        grids = bottleneck.decode_codes(values) * state["scale"]
        texts = decoder.generate(list(grids), max_new_tokens=1800)
        for row, text in zip(batch, texts):
            records.append({"id": row["id"], "kind": row["kind"], "bucket": row["bucket"],
                            "value": 1 - ned(normalize(text), normalize(row["text"])), "prediction": text[:300]})
    mean = lambda rs: round(sum(r["value"] for r in rs) / len(rs), 4) if rs else None
    result = {"score": mean(records), "payload_bytes_max": max(sizes),
              "by_bucket": {b: mean([r for r in records if r["bucket"] == b]) for b in BUCKET_WEIGHTS},
              "by_kind_bucket": {f"{k}/{b}": mean([r for r in records if r["kind"] == k and r["bucket"] == b])
                                 for k in ("document", "screenshot") for b in BUCKET_WEIGHTS},
              "examples": records[:6]}
    args.out.write_text(json.dumps(result, indent=2))
    print(json.dumps({k: v for k, v in result.items() if k != "examples"}, indent=2))


def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prep")
    p.add_argument("--train-dir", type=Path, required=True)
    p.add_argument("--cache", type=Path, required=True)
    p.add_argument("--max-text-tokens", type=int, default=1024)
    t = sub.add_parser("train")
    t.add_argument("--train-dir", type=Path, required=True)
    t.add_argument("--cache", type=Path, required=True)
    t.add_argument("--out", type=Path, required=True)
    t.add_argument("--steps", type=int, default=3000)
    t.add_argument("--batch", type=int, default=8)
    t.add_argument("--lr", type=float, default=5e-4)
    t.add_argument("--lora", type=int, default=0)
    t.add_argument("--lora-lr", type=float, default=1e-4)
    t.add_argument("--code-dim", type=int, default=16)
    t.add_argument("--levels", type=int, default=16)
    t.add_argument("--mse-weight", type=float, default=1.0)
    e = sub.add_parser("score")
    e.add_argument("--run", type=Path, required=True)
    e.add_argument("--data", type=Path, required=True)
    e.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    {"prep": prep, "train": train, "score": evaluate}[args.command](args)


if __name__ == "__main__":
    main()
