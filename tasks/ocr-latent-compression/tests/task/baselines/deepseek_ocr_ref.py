"""Reference DeepSeek-OCR pieces for organizer baselines (never run by the verifier).

Splits the upstream model into the two halves the task cares about:

* ``VisionEncoder``: DeepEncoder (SAM-B + CLIP-L + linear projector) producing the
  per-patch vision embeddings the language decoder normally receives.
* ``VisionDecoder``: the DeepSeek-V2 MoE language decoder, fed an arbitrary
  ``[tokens, 1280]`` embedding grid in the vision-token positions of the upstream
  "Free OCR." prompt, followed by batched greedy decoding with a KV cache.

Upstream layout of one global view with an ``h x w`` grid: every row of ``w`` patch
embeddings is followed by ``image_newline``; one ``view_seperator`` closes the view.
Tiny mode is a 512x512 resize (64 patches), Small 640x640 resize (100), Base a
1024x1024 pad (256), Large a 1280x1280 pad (400).
"""
from __future__ import annotations

import math
import os
from pathlib import Path

import torch
from PIL import Image, ImageOps

IMAGE_TOKEN_ID = 128815
BOS_ID, EOS_ID = 0, 1
PROMPT = "<image>\nFree OCR."
MODES = {"tiny": (512, "resize"), "small": (640, "resize"), "base": (1024, "pad"), "large": (1280, "pad")}
DEFAULT_WEIGHTS = os.environ.get(
    "DEEPSEEK_OCR_WEIGHTS", "/scratch/frontier-ml-workspace/frontier-mle-clean/artifacts/deepseek-ocr/weights")


def load_model(weights=DEFAULT_WEIGHTS, device="cuda"):
    from transformers import AutoModel, AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(weights, trust_remote_code=True)
    model = AutoModel.from_pretrained(weights, trust_remote_code=True, use_safetensors=True,
                                      _attn_implementation="eager", torch_dtype=torch.bfloat16)
    return tokenizer, model.eval().to(device)


def preprocess(image: Image.Image, mode="tiny") -> torch.Tensor:
    size, how = MODES[mode]
    image = image.convert("RGB")
    if how == "resize":
        image = image.resize((size, size), Image.BICUBIC)
    else:
        image = ImageOps.pad(image, (size, size), color=(127, 127, 127))
    import numpy as np
    array = torch.from_numpy(np.asarray(image, dtype="float32") / 255.0).permute(2, 0, 1)
    return (array - 0.5) / 0.5


class VisionEncoder:
    def __init__(self, model):
        inner = model.model
        self.sam, self.clip, self.projector = inner.sam_model, inner.vision_model, inner.projector
        self.device = next(self.sam.parameters()).device

    @torch.no_grad()
    def features(self, pixels: torch.Tensor, project=True) -> torch.Tensor:
        """pixels [B,3,S,S] in [-1,1] -> [B, h*w, 1280] (or pre-projector [B, h*w, 2048])."""
        pixels = pixels.to(self.device, torch.bfloat16)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            sam = self.sam(pixels)
            clip = self.clip(pixels, sam)
            feats = torch.cat((clip[:, 1:], sam.flatten(2).permute(0, 2, 1)), dim=-1)
            return self.projector(feats) if project else feats


class VisionDecoder:
    def __init__(self, model, tokenizer):
        from transformers.generation.logits_process import NoRepeatNGramLogitsProcessor
        self.model, self.tokenizer = model, tokenizer
        inner = model.model
        self.newline, self.separator = inner.image_newline, inner.view_seperator
        self.embed = inner.embed_tokens
        # Skip DeepseekOCRModel.forward's vision branch: run the plain V2 decoder.
        self.decoder_forward = type(inner).__mro__[1].forward
        self.inner = inner
        self.device = self.embed.weight.device
        prefix, suffix = PROMPT.split("<image>")
        self.prefix_ids = [BOS_ID] + tokenizer.encode(prefix, add_special_tokens=False)
        self.suffix_ids = tokenizer.encode(suffix, add_special_tokens=False)
        self.no_repeat = NoRepeatNGramLogitsProcessor(20)

    def layout(self, grid: torch.Tensor) -> torch.Tensor:
        """[h*w,1280] square grid -> upstream vision positions with newlines and separator."""
        n, dim = grid.shape
        side = int(math.isqrt(n))
        if side * side != n:
            raise ValueError("Vision grid must be square")
        rows = grid.view(side, side, dim).to(self.newline.dtype)
        rows = torch.cat([rows, self.newline[None, None, :].expand(side, 1, dim)], dim=1)
        return torch.cat([rows.reshape(-1, dim), self.separator[None, :]], dim=0)

    @torch.no_grad()
    def generate(self, grids: list[torch.Tensor], max_new_tokens=2048) -> list[str]:
        """Batched greedy decoding; every grid is an [h*w, 1280] vision embedding grid."""
        prefix = self.embed(torch.tensor(self.prefix_ids, device=self.device))
        suffix = self.embed(torch.tensor(self.suffix_ids, device=self.device))
        sequences = [torch.cat([prefix, self.layout(g.to(self.device)), suffix]) for g in grids]
        length = max(s.shape[0] for s in sequences)
        batch = len(sequences)
        dim = sequences[0].shape[1]
        embeds = torch.zeros(batch, length, dim, dtype=prefix.dtype, device=self.device)
        mask = torch.zeros(batch, length, dtype=torch.long, device=self.device)
        for i, s in enumerate(sequences):  # left padding
            embeds[i, length - s.shape[0]:] = s
            mask[i, length - s.shape[0]:] = 1
        positions = (mask.cumsum(-1) - 1).clamp(min=0)
        from transformers.cache_utils import DynamicCache
        cache = DynamicCache()
        generated = torch.zeros(batch, 0, dtype=torch.long, device=self.device)
        done = torch.zeros(batch, dtype=torch.bool, device=self.device)
        step_input, step_pos = embeds, positions
        with torch.autocast("cuda", dtype=torch.bfloat16):
            for _ in range(max_new_tokens):
                out = self.decoder_forward(self.inner, inputs_embeds=step_input, attention_mask=mask,
                                           position_ids=step_pos, past_key_values=cache, use_cache=True,
                                           return_dict=True)
                cache = out.past_key_values
                logits = self.model.lm_head(out.last_hidden_state[:, -1]).float()
                if generated.shape[1]:
                    logits = self.no_repeat(generated, logits)
                token = logits.argmax(-1)
                token = torch.where(done, torch.full_like(token, EOS_ID), token)
                generated = torch.cat([generated, token[:, None]], dim=1)
                done |= token == EOS_ID
                if bool(done.all()):
                    break
                step_input = self.embed(token)[:, None]
                mask = torch.cat([mask, torch.ones(batch, 1, dtype=mask.dtype, device=self.device)], dim=1)
                step_pos = step_pos[:, -1:] + 1
        texts = []
        for row in generated.tolist():
            if EOS_ID in row:
                row = row[:row.index(EOS_ID)]
            texts.append(self.tokenizer.decode(row, skip_special_tokens=False).strip())
        return texts


def load_image(path: str | Path) -> Image.Image:
    with Image.open(path) as image:
        return image.convert("RGB")
