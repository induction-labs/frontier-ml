"""Training-free Z-Image-Turbo baseline: description and boxed texts written into the prompt."""
WIDTH, HEIGHT = 1280, 720


def position_words(bbox):
    x = (bbox[0] + bbox[2]) / 2 / WIDTH
    y = (bbox[1] + bbox[3]) / 2 / HEIGHT
    v = "top" if y < 0.2 else "upper" if y < 0.4 else "middle" if y < 0.6 else "lower" if y < 0.8 else "bottom"
    h = "left" if x < 0.33 else "center" if x < 0.67 else "right"
    return f"{v} {h}"


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
    return ("Full-bleed screenshot of a desktop website's page content, 16:9, flat UI, sharp text, no window frame, "
            "no address bar, no tabs. " + description.strip()
            + "\nThe page shows exactly the following text, top to bottom, and no other text:\n" + "\n".join(lines))


class Diffusion:
    def __init__(self, model_dir, seed=0):
        import torch
        from diffusers import ZImagePipeline
        self.torch, self.seed = torch, seed
        self.pipe = ZImagePipeline.from_pretrained(model_dir, torch_dtype=torch.bfloat16).to("cuda")
        self.pipe.set_progress_bar_config(disable=True)

    def __call__(self, description, texts):
        generator = self.torch.Generator("cuda").manual_seed(self.seed)
        return self.pipe(prompt=build_prompt(description, texts), height=HEIGHT, width=WIDTH, generator=generator,
                         num_inference_steps=9, guidance_scale=0.0, max_sequence_length=1024).images[0].convert("RGB")


def make(kind):
    assert kind == "zimage"
    return Diffusion("/scratch/frontier-ml-workspace/work/screenshot-generation/models/zimage-turbo")
