"""Shared model, tokenizer and context-window code for the babylm-replay reference submissions.

Two architectures:
  * "gpt2": GPT-2-small style (learned positions, LayerNorm, GELU, tied embeddings).
  * "slow": a scaled-down port of the NanoGPT Slowrun recipe (RoPE, QK-norm, ReLU^2/SwiGLU,
    U-Net skips, per-head attention gates, value embeddings from x0, logit soft-cap).
    Architecture ideas and the Muon step are adapted from qlabs-eng/slowrun (MIT License).
"""
import json
import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

BOS = "<|bos|>"


# --------------------------------------------------------------------------------------------
# Tokenizer helpers (byte-level BPE trained on the corpus inside train.sh)
# --------------------------------------------------------------------------------------------

def bytes_to_unicode():
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def vocab_bytes(tokenizer):
    """Token id -> raw bytes (None for special tokens)."""
    inverse = {v: k for k, v in bytes_to_unicode().items()}
    size = tokenizer.get_vocab_size()
    out = [None] * size
    for token, index in tokenizer.get_vocab().items():
        if token == BOS:
            continue
        out[index] = bytes(inverse[c] for c in token)
    return out


# --------------------------------------------------------------------------------------------
# Context windows: /score and /dist use the same deterministic window per target position.
# x = [BOS, t0, ..., t_{n-1}]; target index j (1..n) predicts x[j] from x[a:j].
# --------------------------------------------------------------------------------------------

def window_start(j, context, stride):
    if j < context:
        return 0
    return ((j - context + stride) // stride) * stride


def score_windows(n, context, stride):
    """Yield (a, first_target, last_target_exclusive) covering targets 1..n exactly once."""
    j = 1
    while j <= n:
        a = window_start(j, context, stride)
        end = min(n + 1, a + context)
        yield a, j, end
        j = end


# --------------------------------------------------------------------------------------------
# Models
# --------------------------------------------------------------------------------------------

def rms(x):
    return F.rms_norm(x, (x.size(-1),))


class GPT2Block(nn.Module):
    def __init__(self, d, heads, dropout):
        super().__init__()
        self.ln1, self.ln2 = nn.LayerNorm(d), nn.LayerNorm(d)
        self.qkv, self.proj = nn.Linear(d, 3 * d), nn.Linear(d, d)
        self.fc, self.out = nn.Linear(d, 4 * d), nn.Linear(4 * d, d)
        self.heads, self.dropout = heads, dropout
        self.drop = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(self.ln1(x)).view(B, T, 3, self.heads, C // self.heads).permute(2, 0, 3, 1, 4)
        y = F.scaled_dot_product_attention(q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0)
        x = x + self.drop(self.proj(y.transpose(1, 2).reshape(B, T, C)))
        return x + self.drop(self.out(F.gelu(self.fc(self.ln2(x)), approximate="tanh")))


class GPT2(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d = cfg["n_embd"]
        self.wte = nn.Embedding(cfg["vocab"], d)
        self.wpe = nn.Embedding(cfg["context"], d)
        self.drop = nn.Dropout(cfg["dropout"])
        self.blocks = nn.ModuleList(GPT2Block(d, cfg["n_head"], cfg["dropout"]) for _ in range(cfg["n_layer"]))
        self.ln_f = nn.LayerNorm(d)
        self.apply(self._init)
        for name, p in self.named_parameters():
            if name.endswith("proj.weight") or name.endswith("out.weight"):
                nn.init.normal_(p, 0.0, 0.02 / math.sqrt(2 * cfg["n_layer"]))

    @staticmethod
    def _init(m):
        if isinstance(m, nn.Linear):
            nn.init.normal_(m.weight, 0.0, 0.02)
            if m.bias is not None:
                nn.init.zeros_(m.bias)
        elif isinstance(m, nn.Embedding):
            nn.init.normal_(m.weight, 0.0, 0.02)

    def forward(self, idx):
        T = idx.size(1)
        x = self.drop(self.wte(idx) + self.wpe(torch.arange(T, device=idx.device)))
        for block in self.blocks:
            x = block(x)
        return F.linear(self.ln_f(x), self.wte.weight).float()


class SlowAttention(nn.Module):
    def __init__(self, cfg, has_ve):
        super().__init__()
        d, self.h = cfg["n_embd"], cfg["n_head"]
        self.hd = d // self.h
        self.c_q, self.c_k, self.c_v = (nn.Linear(d, d, bias=False) for _ in range(3))
        self.c_proj = nn.Linear(d, d, bias=False)
        self.gate = nn.Linear(12, self.h, bias=False)
        self.ve_gate = nn.Linear(32, self.h, bias=False) if has_ve else None
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x, ve, cos, sin):
        B, T, C = x.shape
        q = self.c_q(x).view(B, T, self.h, self.hd)
        k = self.c_k(x).view(B, T, self.h, self.hd)
        v = self.c_v(x).view(B, T, self.h, self.hd)
        if ve is not None:
            v = v + 2 * torch.sigmoid(self.ve_gate(x[..., :32])).unsqueeze(-1) * ve.view(B, T, self.h, self.hd)
        q, k = rms(rope(q, cos, sin)), rms(rope(k, cos, sin))
        y = F.scaled_dot_product_attention(q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2), is_causal=True).transpose(1, 2)
        y = y * torch.sigmoid(self.gate(x[..., :12])).unsqueeze(-1)
        return self.drop(self.c_proj(y.reshape(B, T, C)))


class SlowMLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        d = cfg["n_embd"]
        hidden = 256 * ((8 * d // 3 + 255) // 256)
        self.c_gate, self.c_fc = nn.Linear(d, hidden, bias=False), nn.Linear(d, hidden, bias=False)
        self.c_proj = nn.Linear(hidden, d, bias=False)
        self.drop = nn.Dropout(cfg["dropout"])

    def forward(self, x):
        return self.drop(self.c_proj(F.silu(self.c_gate(x)) * self.c_fc(x)))


class SlowBlock(nn.Module):
    def __init__(self, cfg, has_ve):
        super().__init__()
        self.attn, self.mlp = SlowAttention(cfg, has_ve), SlowMLP(cfg)

    def forward(self, x, ve, cos, sin):
        x = x + self.attn(rms(x), ve, cos, sin)
        return x + self.mlp(rms(x))


def rope(x, cos, sin):
    d = x.shape[-1] // 2
    cos, sin = cos.to(x.dtype), sin.to(x.dtype)
    x1, x2 = x[..., :d], x[..., d:]
    return torch.cat([x1 * cos + x2 * sin, -x1 * sin + x2 * cos], dim=-1)


class SlowGPT(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.cfg = cfg
        d, L = cfg["n_embd"], cfg["n_layer"]
        self.ve_layers = [i for i in range(L) if i % 2 == (L - 1) % 2]  # alternate layers, incl. last
        self.wte = nn.Embedding(cfg["vocab"], d)
        self.blocks = nn.ModuleList(SlowBlock(cfg, i in self.ve_layers) for i in range(L))
        self.ve_projs = nn.ModuleDict({str(i): nn.Linear(d, d, bias=False) for i in self.ve_layers})
        self.lm_head = nn.Linear(d, cfg["vocab"], bias=False)
        self.resid_lambdas = nn.Parameter(torch.ones(L))
        self.x0_lambdas = nn.Parameter(torch.full((L,), 0.1))
        self.enc = L // 2
        self.skip_weights = nn.Parameter(torch.ones(self.enc))
        hd = d // cfg["n_head"]
        inv = 1.0 / (10000 ** (torch.arange(0, hd, 2, dtype=torch.float32) / hd))
        f = torch.outer(torch.arange(cfg["context"], dtype=torch.float32), inv)
        self.register_buffer("cos", f.cos()[None, :, None, :], persistent=False)
        self.register_buffer("sin", f.sin()[None, :, None, :], persistent=False)
        self.reset()

    @torch.no_grad()
    def reset(self):
        d = self.cfg["n_embd"]
        s, n = 3 ** 0.5 * d ** -0.5, d ** -0.5
        nn.init.normal_(self.wte.weight, 0.0, 1.0)
        nn.init.normal_(self.lm_head.weight, 0.0, 0.001)
        for b in self.blocks:
            for w in (b.attn.c_q, b.attn.c_k, b.attn.c_v, b.mlp.c_gate, b.mlp.c_fc):
                nn.init.uniform_(w.weight, -s, s)
            nn.init.normal_(b.attn.c_proj.weight, 0.0, n)
            nn.init.normal_(b.mlp.c_proj.weight, 0.0, n)
            nn.init.zeros_(b.attn.gate.weight)
            if b.attn.ve_gate is not None:
                nn.init.zeros_(b.attn.ve_gate.weight)
        for p in self.ve_projs.values():
            nn.init.uniform_(p.weight, -s, s)

    def forward(self, idx):
        T = idx.size(1)
        cos, sin = self.cos[:, :T], self.sin[:, :T]
        x = x0 = rms(self.wte(idx))
        skips = []
        L = self.cfg["n_layer"]
        for i, block in enumerate(self.blocks):
            if i >= self.enc:
                j = L - 1 - i
                if 0 <= j < self.enc:
                    x = x + self.skip_weights[i - self.enc] * skips[j]
            x = self.resid_lambdas[i] * x + self.x0_lambdas[i] * x0
            ve = self.ve_projs[str(i)](x0) if str(i) in self.ve_projs else None
            x = block(x, ve, cos, sin)
            if i < self.enc:
                skips.append(x)
        logits = self.lm_head(rms(x)).float()
        cap = self.cfg.get("logit_cap", 0)
        return cap * torch.tanh(logits / cap) if cap else logits


def build_model(cfg):
    return {"gpt2": GPT2, "slow": SlowGPT}[cfg["arch"]](cfg)


def load_model(directory, device="cuda"):
    from safetensors.torch import load_file
    with open(os.path.join(directory, "config.json")) as f:
        cfg = json.load(f)
    model = build_model(cfg)
    state = load_file(os.path.join(directory, "model.safetensors"))
    model.load_state_dict(state, strict=True)
    return model.to(device).float().eval(), cfg
