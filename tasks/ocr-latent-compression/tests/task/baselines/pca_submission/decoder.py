"""/decode path of the organizer PCA baseline (frozen DeepSeek-OCR language decoder).

Payload bytes -> PCA coefficients -> 64 x 1280 vision-embedding grid, laid out exactly as
upstream Tiny mode (row newlines + view separator) in the vision-token positions of the
"<image>\\nFree OCR." prompt, followed by batched greedy decoding.
"""
import math
from pathlib import Path

import numpy as np
import torch
from safetensors.torch import load_file

from codec import dequantize, unpack

HERE = Path(__file__).resolve().parent
WEIGHTS = HERE / "weights"
BOS_ID, EOS_ID = 0, 1


class Decoder:
    def __init__(self, device="cuda", max_new_tokens=2048):
        from transformers import AutoModel, AutoTokenizer
        from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor
        self.tokenizer = AutoTokenizer.from_pretrained(WEIGHTS, trust_remote_code=True)
        model = AutoModel.from_pretrained(WEIGHTS, trust_remote_code=True, use_safetensors=True,
                                          _attn_implementation="eager", torch_dtype=torch.bfloat16)
        inner = model.model
        del inner.sam_model, inner.vision_model, inner.projector  # the decoder never sees pixels
        self.model = model.eval().to(device)
        self.inner = self.model.model
        self.forward = type(self.inner).__mro__[1].forward  # plain DeepSeek-V2 decoder forward
        self.device, self.max_new_tokens = device, max_new_tokens
        self.no_repeat = NoRepeatNGramLogitsProcessor(20)
        prefix, suffix = "<image>\nFree OCR.".split("<image>")
        self.prefix_ids = [BOS_ID] + self.tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = self.tokenizer.encode(suffix, add_special_tokens=False)
        pca = load_file(WEIGHTS / "pca_tiny.safetensors")
        self.mean, self.basis = pca["mean"].float().to(device), pca["basis"].float().to(device)
        self.low, self.high, self.bits = pca["low"].numpy(), pca["high"].numpy(), int(pca["bits"].item())
        self.tokens, self.k = 64, self.basis.shape[0]

    def grid(self, payload):
        codes = unpack(payload, self.tokens * self.k, self.bits).reshape(self.tokens, self.k)
        coefficients = torch.from_numpy(dequantize(codes, self.low, self.high, self.bits)).to(self.device)
        return coefficients @ self.basis + self.mean

    def layout(self, grid):
        side = math.isqrt(grid.shape[0])
        rows = grid.view(side, side, -1).to(torch.bfloat16)
        newline = self.inner.image_newline[None, None, :].expand(side, 1, -1)
        rows = torch.cat([rows, newline], dim=1).reshape(-1, grid.shape[1])
        return torch.cat([rows, self.inner.view_seperator[None, :]], dim=0)

    @torch.no_grad()
    def decode(self, payloads):
        from transformers.cache_utils import DynamicCache
        embed = self.inner.embed_tokens
        prefix = embed(torch.tensor(self.prefix_ids, device=self.device))
        suffix = embed(torch.tensor(self.suffix_ids, device=self.device))
        sequences = [torch.cat([prefix, self.layout(self.grid(p)), suffix]) for p in payloads]
        batch, length = len(sequences), max(s.shape[0] for s in sequences)
        embeds = torch.zeros(batch, length, prefix.shape[1], dtype=prefix.dtype, device=self.device)
        mask = torch.zeros(batch, length, dtype=torch.long, device=self.device)
        for i, s in enumerate(sequences):
            embeds[i, length - s.shape[0]:] = s
            mask[i, length - s.shape[0]:] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        cache, generated = DynamicCache(), torch.zeros(batch, 0, dtype=torch.long, device=self.device)
        done = torch.zeros(batch, dtype=torch.bool, device=self.device)
        step, step_positions = embeds, positions
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(self.max_new_tokens):
                out = self.forward(self.inner, inputs_embeds=step, attention_mask=mask, position_ids=step_positions,
                                   past_key_values=cache, use_cache=True, return_dict=True)
                cache = out.past_key_values
                logits = self.model.lm_head(out.last_hidden_state[:, -1]).float()
                if generated.shape[1]:
                    logits = self.no_repeat(generated, logits)
                token = torch.where(done, torch.full_like(logits[:, 0], EOS_ID, dtype=torch.long), logits.argmax(-1))
                generated = torch.cat([generated, token[:, None]], dim=1)
                done |= token == EOS_ID
                if bool(done.all()):
                    break
                step = embed(token)[:, None]
                mask = torch.cat([mask, torch.ones(batch, 1, dtype=mask.dtype, device=self.device)], dim=1)
                step_positions = step_positions[:, -1:] + 1
        texts = []
        for row in generated.tolist():
            row = row[:row.index(EOS_ID)] if EOS_ID in row else row
            texts.append(self.tokenizer.decode(row, skip_special_tokens=True).strip())
        return texts
