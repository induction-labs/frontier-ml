"""Baseline generators for grounded screenshot generation.

Generators: zimage (Z-Image-Turbo, 8 NFE), flux2klein (FLUX.2-klein-4B, 4 steps), both with the
description and boxes written into the prompt; compositor (PIL text drawn into the boxes); html
(Chromium renders absolutely positioned spans). Metric sanity checks: real (reference screenshot),
blank (white), noise.
"""
import io
import json
import time
from pathlib import Path

WIDTH, HEIGHT = 1280, 720


def position_words(bbox):
    x = (bbox[0] + bbox[2]) / 2 / WIDTH
    y = (bbox[1] + bbox[3]) / 2 / HEIGHT
    v = "top" if y < 0.2 else "upper" if y < 0.4 else "middle" if y < 0.6 else "lower" if y < 0.8 else "bottom"
    h = "left" if x < 0.33 else "center" if x < 0.67 else "right"
    return f"{v} {h}"


PROMPT_KIND = __import__("os").environ.get("SCREENSHOT_PROMPT_KIND", "screen")  # "screen" (internal data) or "web" (crawl)


def build_prompt(description, texts, max_texts=80, max_chars=2400):
    ordered = sorted(texts, key=lambda t: (t["bbox"][1], t["bbox"][0]))
    lines, used = [], 0
    for t in ordered[:max_texts]:
        x1, y1, x2, y2 = t["bbox"]
        h = y2 - y1
        size = "huge" if h >= 48 else "large" if h >= 28 else "medium" if h >= 18 else "small"
        line = f'- "{t["text"]}" ({size}, {position_words(t["bbox"])})'
        if used + len(line) > max_chars:
            break
        lines.append(line)
        used += len(line)
    if PROMPT_KIND == "web":
        return ("Full-bleed screenshot of a desktop website's page content, 16:9, flat UI, sharp text, no window frame, "
                "no address bar, no tabs. " + description.strip()
                + "\nThe page shows exactly the following text, top to bottom, and no other text:\n" + "\n".join(lines))
    return ("A realistic 1280x720 screenshot of a computer screen, 16:9, sharp crisp UI text. " + description.strip()
            + "\nThe screen shows exactly the following text, top to bottom, and no other text:\n" + "\n".join(lines))


class Diffusion:
    def __init__(self, kind, model_dir, device="cuda", seed=0):
        import torch
        self.torch, self.kind, self.seed = torch, kind, seed
        if kind == "zimage":
            from diffusers import ZImagePipeline
            self.pipe = ZImagePipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device)
            self.kwargs = dict(num_inference_steps=9, guidance_scale=0.0, max_sequence_length=1024)
        elif kind == "flux2klein":
            from diffusers import Flux2KleinPipeline
            self.pipe = Flux2KleinPipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to(device)
            self.kwargs = dict(num_inference_steps=4, guidance_scale=1.0, max_sequence_length=1024)
        else:
            raise ValueError(kind)
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(self, description, texts):
        generator = self.torch.Generator("cuda").manual_seed(self.seed)
        prompt = build_prompt(description, texts)
        image = self.pipe(prompt=prompt, height=HEIGHT, width=WIDTH, generator=generator, **self.kwargs).images[0]
        return image.convert("RGB")


def _font(size, bold=False):
    from PIL import ImageFont
    base = "/scratch/frontier-ml-workspace/work/screenshot-generation/fonts/liberation-fonts-ttf-2.1.5/"
    return ImageFont.truetype(base + ("LiberationSans-Bold.ttf" if bold else "LiberationSans-Regular.ttf"), size)


class Compositor:
    """NON-COMPLIANT sanity check: rasterize each text with a font into its box on a white page."""

    def __call__(self, description, texts):
        from PIL import Image, ImageDraw
        image = Image.new("RGB", (WIDTH, HEIGHT), "white")
        draw = ImageDraw.Draw(image)
        for t in texts:
            x1, y1, x2, y2 = t["bbox"]
            size = max(6, int((y2 - y1) * 0.85))
            font = _font(size)
            while size > 6 and draw.textlength(t["text"], font=font) > (x2 - x1) * 1.02:
                size -= 1
                font = _font(size)
            draw.text((x1, (y1 + y2) / 2), t["text"], fill=(20, 20, 20), font=font, anchor="lm")
        return image


class HTMLRender:
    """Chromium baseline with absolutely positioned spans."""

    def __init__(self):
        from playwright.sync_api import sync_playwright
        self.p = sync_playwright().start()
        self.browser = self.p.chromium.launch()
        self.page = self.browser.new_page(viewport={"width": WIDTH, "height": HEIGHT}, device_scale_factor=1)

    def __call__(self, description, texts):
        from PIL import Image
        import html
        spans = []
        for t in texts:
            x1, y1, x2, y2 = t["bbox"]
            h = y2 - y1
            spans.append(f'<span style="position:absolute;left:{x1}px;top:{y1}px;width:{x2 - x1}px;height:{h}px;'
                         f'font:{max(6, round(h * 0.82))}px/{h}px Liberation Sans, sans-serif;white-space:nowrap;'
                         f'overflow:visible;color:#222" data-w="{x2 - x1}">{html.escape(t["text"])}</span>')
        doc = ("<html><body style='margin:0;background:#fff'>" + "".join(spans) +
               "<script>for(const s of document.querySelectorAll('span')){let f=parseFloat(s.style.fontSize);"
               "const w=+s.dataset.w;while(s.scrollWidth>w*1.02&&f>6){f-=0.5;s.style.fontSize=f+'px';}}</script></body></html>")
        self.page.set_content(doc)
        return Image.open(io.BytesIO(self.page.screenshot(type="png"))).convert("RGB")


class Real:
    def __init__(self, root):
        self.root = Path(root)

    def __call__(self, description, texts, sample=None):
        from PIL import Image
        return Image.open(self.root / sample["image"]).convert("RGB")


class Blank:
    def __call__(self, description, texts):
        from PIL import Image
        return Image.new("RGB", (WIDTH, HEIGHT), "white")


class Noise:
    def __call__(self, description, texts):
        import numpy as np
        from PIL import Image
        return Image.fromarray(np.random.default_rng(0).integers(0, 256, (HEIGHT, WIDTH, 3), dtype=np.uint8))


def make(kind, **kwargs):
    models = "/scratch/frontier-ml-workspace/work/screenshot-generation/models/"
    if kind == "zimage":
        return Diffusion("zimage", models + "zimage-turbo")
    if kind == "flux2klein":
        return Diffusion("flux2klein", models + "flux2-klein-4b")
    return {"compositor": Compositor, "html": HTMLRender, "blank": Blank, "noise": Noise}[kind]()
