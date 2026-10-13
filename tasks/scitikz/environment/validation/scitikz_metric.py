"""SciTikZ paper metrics and the task's additional single-score aggregation.

Independent implementation of the protocol in JackieForest/SciTikZ commit
6a2be537886ed005633c275ec0c9d29d2246bbb9. Heavy dependencies are imported lazily.
"""
from collections import Counter
import math
import re
from statistics import mean

METRICS = ("siglip", "clip", "lpips", "ssim", "dreamsim", "ted", "crystalbleu")


def normalize_tex(text):
    lines = []
    for line in text.replace("\r\n", "\n").replace("\r", "\n").splitlines():
        lines.append(re.split(r"(?<!\\)%", line, maxsplit=1)[0])
    return re.sub(r"\n{3,}", "\n\n", re.sub(r"[ \t]+", " ", "\n".join(lines))).strip()


def document_body(text):
    match = re.search(r"\\begin\{document\}(.*)\\end\{document\}", text, re.I | re.S)
    return normalize_tex(match.group(1) if match else text)


def aggregate(records):
    if not records:
        raise ValueError("No evaluation records")
    successful = [r for r in records if r["compiled"]]
    for record in records:
        for key in (METRICS if record["compiled"] else ("ted", "crystalbleu")):
            if not isinstance(record[key], (int, float)) or not math.isfinite(record[key]):
                raise ValueError(f"Missing or nonfinite {key} metric")
    rate = len(successful) / len(records)
    averages = {key: mean(r[key] for r in successful) if successful else None for key in METRICS}
    all_code = {key: mean(r[key] for r in records) for key in ("ted", "crystalbleu")}
    components = None
    score = 0.0
    if successful:
        clip = lambda x: min(1.0, max(0.0, x))
        components = {"siglip": clip(averages["siglip"]), "clip": clip(averages["clip"]),
                      "lpips": math.exp(-max(0, averages["lpips"]) / .5),
                      "ssim": clip(averages["ssim"]), "dreamsim": clip(1 - averages["dreamsim"]),
                      "ted": math.exp(-max(0, all_code["ted"]) / .4),
                      "crystalbleu": clip(all_code["crystalbleu"])}
        values = list(components.values())
        score = rate * math.exp(mean(math.log(x) for x in values)) if min(values) > 0 else 0.0
    all_visual = {key: sum(r[key] if r["compiled"] else (1.0 if key in ("lpips", "dreamsim") else 0.0)
                           for r in records) / len(records)
                  for key in ("siglip", "clip", "lpips", "ssim", "dreamsim")}
    return {"score": score, "details": {"metric": "compilation_times_geomean_paper_metrics",
            "samples": len(records), "compiled": len(successful), "compilation_rate": rate,
            "compiled_only": averages, "all_example_visual": all_visual, "all_example_code": all_code,
            "normalized_components": components, "records": records}}


class CodeMetrics:
    def __init__(self, reference_corpus):
        from pygments.lexers.markup import TexLexer
        from sacremoses import MosesTokenizer
        self.lexer, self.moses = TexLexer(), MosesTokenizer()
        counts = Counter()
        for reference in reference_corpus:
            tokens = self.tokens(reference, bleu=True)
            for n in range(1, 5):
                counts.update(tuple(tokens[i:i+n]) for i in range(len(tokens)-n+1))
        self.ignoring = dict(counts.most_common(500))

    def tokens(self, text, *, bleu=False):
        from pygments.token import Comment, Name, Text
        from torchmetrics.functional.text.eed import _preprocess_en
        result = []
        for kind, value in self.lexer.get_tokens(normalize_tex(text)):
            value = value.strip()
            if not value or kind in Comment:
                continue
            if bleu:
                result.extend(self.moses.tokenize(value) if kind in Text or kind in Name.Attribute or
                              kind in Name.Builtin else [value])
            else:
                result.extend((_preprocess_en(value) if kind in Text else value).split())
        return result

    def compute(self, reference, prediction):
        from crystalbleu import corpus_bleu
        from scitikz_eed import eed
        import torch
        ref, pred = document_body(reference), document_body(prediction)
        rt, pt = self.tokens(ref), self.tokens(pred)
        # ExtendedEditDistance.compute() in the release reduces through float32.
        distance = float(torch.tensor(eed(" " + " ".join(pt) + " ", " " + " ".join(rt) + " ")))
        rb, pb = self.tokens(ref, bleu=True), self.tokens(pred, bleu=True)
        bleu = float(corpus_bleu([[rb]], [pb], ignoring=self.ignoring)) if rb and pb else 0.0
        return {"ted": distance, "crystalbleu": bleu}


def trim_white(image):
    from PIL import Image, ImageChops
    box = ImageChops.difference(image, Image.new("RGB", image.size, "white")).convert("L").getbbox()
    return image.crop((max(0, box[0]-2), max(0, box[1]-2), min(image.width, box[2]+2),
                       min(image.height, box[3]+2))) if box else image


def center_pad(image, width, height):
    from PIL import Image
    out = Image.new("RGB", (width, height), "white")
    out.paste(image, ((width-image.width)//2, (height-image.height)//2))
    return out


class ImageMetrics:
    def __init__(self, models_dir, device="cuda"):
        from pathlib import Path
        import torch
        import lpips
        from dreamsim import dreamsim
        from transformers import CLIPModel, CLIPProcessor, SiglipModel, SiglipImageProcessor
        from scitikz_models import check_models
        root = Path(models_dir)
        check_models(root)
        torch.hub.set_dir(str(root / "torch"))
        self.device, self.torch = device, torch
        self.embeddings = []
        for name, model_cls, processor_cls in (("siglip", SiglipModel, SiglipImageProcessor),
                                               ("clip", CLIPModel, CLIPProcessor)):
            self.embeddings.append((name, model_cls.from_pretrained(root/name, local_files_only=True).to(device).eval(),
                                    processor_cls.from_pretrained(root/name, local_files_only=True, use_fast=False)))
        self.lpips = lpips.LPIPS(net="alex").to(device).eval()
        self.dream, self.dream_preprocess = dreamsim(pretrained=True, device=device,
            cache_dir=str(root/"dreamsim"), normalize_embeds=True, dreamsim_type="ensemble")
        self.dtype = (torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float16) if device.startswith("cuda") else torch.float32
        for extractor in self.dream.extractor_list:
            extractor.model = extractor.model.to(self.dtype)
            if hasattr(extractor, "proj"):
                extractor.proj = extractor.proj.to(self.dtype)
        self.dream = self.dream.to(self.dtype).eval()

    def compute(self, pairs, batch_size=16):
        from PIL import Image
        from torchvision.transforms.functional import to_tensor
        from pytorch_msssim import ssim
        torch = self.torch
        results = []
        with torch.inference_mode():
            for offset in range(0, len(pairs), batch_size):
                batch = pairs[offset:offset+batch_size]
                refs = [Image.open(a).convert("RGB") for a, _ in batch]
                preds = [Image.open(b).convert("RGB") for _, b in batch]
                records = [{} for _ in batch]
                for name, model, processor in self.embeddings:
                    inputs = processor(images=refs+preds, return_tensors="pt").to(self.device)
                    features = model.get_image_features(**inputs).float()
                    features = torch.nn.functional.normalize(features, dim=-1)
                    cosines = (features[:len(batch)] * features[len(batch):]).sum(-1)
                    for row, value in zip(records, cosines.clamp(0, 1).cpu().tolist()):
                        row[name] = value
                def stack(images):
                    return torch.stack([to_tensor(im.resize((384, 384), Image.Resampling.BICUBIC)) for im in images]).to(self.device)
                with torch.autocast("cuda", dtype=torch.float16, enabled=self.device.startswith("cuda")):
                    distances = self.lpips(stack(refs)*2-1, stack(preds)*2-1).float().flatten().cpu().tolist()
                for row, value in zip(records, distances):
                    row["lpips"] = value
                aligned_refs, aligned_preds = [], []
                dream_refs, dream_preds = [], []
                for ref, pred in zip(refs, preds):
                    trimmed = [trim_white(im) for im in (ref, pred)]
                    squared = [center_pad(im, max(im.size), max(im.size)) for im in trimmed]
                    dream_refs.append(self.dream_preprocess(squared[0]))
                    dream_preds.append(self.dream_preprocess(squared[1]))
                    sized = [im.resize(tuple(max(1, round(x*1024/max(im.size))) for x in im.size), Image.Resampling.BICUBIC)
                             if max(im.size)>1024 else im for im in trimmed]
                    width, height = max(im.width for im in sized), max(im.height for im in sized)
                    aligned_refs.append(center_pad(sized[0], width, height))
                    aligned_preds.append(center_pad(sized[1], width, height))
                values = ssim(stack(aligned_refs), stack(aligned_preds), data_range=1.0, size_average=False).cpu().tolist()
                for row, value in zip(records, values):
                    row["ssim"] = value
                a = torch.cat(dream_refs).to(self.device, dtype=self.dtype)
                b = torch.cat(dream_preds).to(self.device, dtype=self.dtype)
                values = self.dream(a, b).float().flatten().cpu().tolist()
                for row, value in zip(records, values):
                    row["dreamsim"] = value
                results.extend(records)
        return results
