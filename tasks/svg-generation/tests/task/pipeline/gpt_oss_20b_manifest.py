#!/usr/bin/env python3
"""Write ../gpt_oss_20b_bf16.json: name, dtype, shape and sha256 of every tensor of openai/gpt-oss-20b in its
unquantized bf16 layout, the reference the weight audit (svg_weights.py) compares submissions against.

    python pipeline/gpt_oss_20b_manifest.py --bf16-dir DIR

DIR is the base revision dequantized to bf16, e.g. with transformers' Mxfp4Config(dequantize=True) and the state dict
written with safetensors (transformers 5.18 save_pretrained() drops the expert tensors after dequantization). The
MXFP4-to-bf16 conversion is exact, so the digests do not depend on how the conversion was done.
"""
import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from svg_weights import read_header, tensor_digest  # noqa: E402

BASE = {"model": "openai/gpt-oss-20b", "revision": "6cee5e81ee83917806bbde320786a8fb61efebee"}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bf16-dir", required=True)
    args = parser.parse_args()
    tensors = {}
    for shard in sorted(Path(args.bf16_dir).glob("*.safetensors")):
        data_start, header = read_header(shard)
        for name, entry in header.items():
            tensors[name] = {"dtype": entry["dtype"], "shape": entry["shape"], "sha256": tensor_digest(shard, data_start, entry)}
    if len(tensors) != 411 or any(t["dtype"] != "BF16" for t in tensors.values()):
        raise SystemExit(f"expected 411 bf16 tensors, found {len(tensors)}")
    out = Path(__file__).resolve().parents[1] / "gpt_oss_20b_bf16.json"
    out.write_text(json.dumps({"base": BASE, "layout": "transformers GptOssForCausalLM, unquantized bf16",
                               "tensors": dict(sorted(tensors.items()))}, indent=0) + "\n")
    print(f"{len(tensors)} tensors -> {out}")


if __name__ == "__main__":
    main()
