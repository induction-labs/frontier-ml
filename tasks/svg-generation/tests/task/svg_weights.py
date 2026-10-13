"""Trusted static audit of the submitted gpt-oss-20b finetune (output/model/); candidate code is never imported.

The trusted server (trusted_server/serve.sh) reads only the safetensors shards, their index and tokenizer.json from
model/, so this audit pins exactly those: every tensor of openai/gpt-oss-20b in its unquantized bf16 layout (names,
shapes and dtype from gpt_oss_20b_bf16.json), nothing else, tokenizer.json byte-identical to the base revision's,
and at least one tensor that differs from the base (an unchanged checkpoint is not a finetune). Dequantizing the
base's MXFP4 experts to bf16 is exact, so an unchanged base hashes identically however it was converted.
"""
import hashlib
import json
from pathlib import Path
import struct

SEED = Path(__file__).resolve().parent
MANIFEST = SEED / "gpt_oss_20b_bf16.json"
BASE_FILES = SEED / "trusted_server/base_files.json"
MAX_HEADER_BYTES = 100 * 1024 * 1024
DTYPE_BYTES = {"BF16": 2}


class WeightAuditError(ValueError):
    pass


def read_header(path):
    """The safetensors header of one shard, with every data range checked against the file size."""
    size = path.stat().st_size
    with path.open("rb") as stream:
        raw = stream.read(8)
        if len(raw) != 8:
            raise WeightAuditError(f"{path.name}: not a safetensors file")
        length = struct.unpack("<Q", raw)[0]
        if not 2 <= length <= min(MAX_HEADER_BYTES, size - 8):
            raise WeightAuditError(f"{path.name}: invalid safetensors header length")
        try:
            header = json.loads(stream.read(length))
        except ValueError as error:
            raise WeightAuditError(f"{path.name}: unreadable safetensors header") from error
    if not isinstance(header, dict):
        raise WeightAuditError(f"{path.name}: invalid safetensors header")
    header.pop("__metadata__", None)
    spans = []
    for name, entry in header.items():
        if not isinstance(entry, dict) or set(entry) != {"dtype", "shape", "data_offsets"}:
            raise WeightAuditError(f"{path.name}: malformed entry for {name}")
        start, end = entry["data_offsets"]
        if type(start) is not int or type(end) is not int or not 0 <= start <= end or 8 + length + end > size:
            raise WeightAuditError(f"{path.name}: {name} lies outside the file")
        spans.append((start, end, name))
    spans.sort()
    for (_, end, name), (start, _, other) in zip(spans, spans[1:]):
        if start < end:
            raise WeightAuditError(f"{path.name}: {name} and {other} overlap")
    return 8 + length, header


def tensor_digest(path, data_start, entry):
    start, end = entry["data_offsets"]
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        stream.seek(data_start + start)
        remaining = end - start
        while remaining:
            chunk = stream.read(min(remaining, 64 * 1024 * 1024))
            if not chunk:
                raise WeightAuditError(f"{path.name}: truncated tensor data")
            digest.update(chunk)
            remaining -= len(chunk)
    return digest.hexdigest()


def inspect_weights(submission):
    """{"passed": True, ...} or {"passed": False, "problems": [...]}; reads only model/ under the submission."""
    model = Path(submission) / "model"
    try:
        expected = json.loads(MANIFEST.read_text())["tensors"]
        base_files = json.loads(BASE_FILES.read_text())
        if not model.is_dir() or model.is_symlink():
            raise WeightAuditError("Submit the finetuned weights in output/model/")
        shards = sorted(p for p in model.glob("*.safetensors"))
        if not shards:
            raise WeightAuditError("model/ holds no .safetensors shards")
        tokenizer = model / "tokenizer.json"
        if not tokenizer.is_file() or tokenizer.is_symlink():
            raise WeightAuditError("model/tokenizer.json is required (copy it unchanged from the base checkpoint)")
        data = tokenizer.read_bytes()
        if (len(data), hashlib.sha256(data).hexdigest()) != (base_files["tokenizer_json"]["bytes"],
                                                            base_files["tokenizer_json"]["sha256"]):
            raise WeightAuditError("model/tokenizer.json differs from openai/gpt-oss-20b's; copy it unchanged")
        found, located = {}, {}
        for shard in shards:
            if shard.is_symlink() or not shard.is_file():
                raise WeightAuditError(f"{shard.name}: weights must be regular files")
            data_start, header = read_header(shard)
            for name, entry in header.items():
                if name in found:
                    raise WeightAuditError(f"tensor {name} appears in more than one shard")
                found[name] = (shard, data_start, entry)
        missing, extra = sorted(set(expected) - set(found)), sorted(set(found) - set(expected))
        if missing or extra:
            raise WeightAuditError(f"tensor set differs from gpt-oss-20b's bf16 layout: {len(missing)} missing "
                                   f"(first {missing[:3]}), {len(extra)} unexpected (first {extra[:3]})")
        index_path = model / "model.safetensors.index.json"
        if not index_path.is_file() or index_path.is_symlink():
            raise WeightAuditError("model/model.safetensors.index.json is required")
        try:
            weight_map = json.loads(index_path.read_text()).get("weight_map")
        except ValueError as error:
            raise WeightAuditError("model.safetensors.index.json is not valid JSON") from error
        if not isinstance(weight_map, dict) or weight_map != {n: found[n][0].name for n in found}:
            raise WeightAuditError("model.safetensors.index.json does not map exactly the tensors in the shards")
        changed, total_bytes = [], 0
        for name, spec in expected.items():
            shard, data_start, entry = found[name]
            if entry["dtype"] != spec["dtype"] or entry["shape"] != spec["shape"]:
                raise WeightAuditError(f"{name}: expected {spec['dtype']} {spec['shape']}, found {entry['dtype']} "
                                       f"{entry['shape']} (submit unquantized bf16 weights)")
            count = 1
            for dim in entry["shape"]:
                count *= dim
            start, end = entry["data_offsets"]
            if end - start != count * DTYPE_BYTES[entry["dtype"]]:
                raise WeightAuditError(f"{name}: data size does not match its shape")
            total_bytes += end - start
            if tensor_digest(shard, data_start, entry) != spec["sha256"]:
                changed.append(name)
        if not changed:
            raise WeightAuditError("every tensor equals openai/gpt-oss-20b's: the checkpoint is not finetuned")
        ignored = sorted(p.name for p in model.iterdir() if p.is_file() and p.suffix != ".safetensors"
                         and p.name not in ("tokenizer.json", "model.safetensors.index.json"))
        return {"passed": True, "shards": len(shards), "tensors": len(found), "tensor_bytes": total_bytes,
                "changed_tensors": len(changed), "changed_examples": changed[:8], "ignored_files": ignored}
    except (OSError, KeyError, TypeError, WeightAuditError) as error:
        return {"passed": False, "problems": [str(error)]}
