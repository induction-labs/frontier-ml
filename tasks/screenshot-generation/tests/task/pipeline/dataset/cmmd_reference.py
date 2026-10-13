"""Embed a split's real screenshots and write its CMMD reference from the unmodified base-model outputs.

python cmmd_reference.py SPLIT_JSONL BASELINE_IMAGE_DIR   -> writes real_clip_embeddings.npy + cmmd_reference.json next to SPLIT_JSONL
"""
import json
import sys
from pathlib import Path

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # the task directory: task-local modules by bare name
import screenshot_generation as sg  # noqa: E402

MODELS = "/scratch/frontier-ml-workspace/work/screenshot-generation/models"
split, baseline = Path(sys.argv[1]), Path(sys.argv[2])
rows = [json.loads(l) for l in open(split) if l.strip()]
clip = sg.PinnedCLIP(MODELS)
real = clip.embed([Image.open(split.parent / r["image"]).convert("RGB") for r in rows])
generated = clip.embed([Image.open(baseline / f"{r['id']}.png").convert("RGB") for r in rows])
np.save(split.parent / "real_clip_embeddings.npy", real)
value = sg.cmmd(real, generated)
half = len(rows) // 2
reference = {"embeddings": "real_clip_embeddings.npy", "reference_cmmd": value, "samples": len(rows),
             "reference_model": "Tongyi-MAI/Z-Image-Turbo@f332072aa78be7aecdf3ee76d5c247082da564a6, 9 steps, guidance 0, "
                                f"seed 0 per request, prompt = description + boxed texts (pipeline/baselines/methods.py build_prompt, kind={__import__('os').environ.get('SCREENSHOT_PROMPT_KIND', 'screen')})",
             "diagnostics": {"real_split_half_cmmd": sg.cmmd(real[:half], real[half:])}}
(split.parent / "cmmd_reference.json").write_text(json.dumps(reference, indent=1) + "\n")
print(json.dumps(reference, indent=1))
