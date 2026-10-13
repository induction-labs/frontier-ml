"""Trusted checkpoint-layout audit: output/model/*.safetensors is Qwen3-4B-Base's architecture. Never shipped to the
candidate.

The audit is static and exact (nothing is imported or executed from the submission): the files must hold exactly the
pinned base tensors (names and shapes from ``qwen3_4b_inventory.json``), each in a floating-point dtype (BF16, F16 or
F32), each stored once, with stored bytes matching shape and dtype; the only extra tensor allowed is ``lm_head.weight``
shaped like the embedding (an untied output head); nonfinite values in a sampled set of tensors fail. A checkpoint of
another size or architecture, or a quantized/packed checkpoint presented as the trained model, fails here.

Whether these weights were actually trained from the pinned base, rather than copied, averaged or adapted from a
post-trained sibling, is for the trusted LLM review to establish from the training evidence and the participant's
trajectory (review.py: pinned_base_start). An earlier version of this module also compared the
checkpoint's weight update with the official siblings' updates (cosine/projection on sampled tensors); the operator
dropped it on 2026-10-01 as brittle, in favour of the reviewer.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np

REFERENCE = json.loads(Path(__file__).with_name('qwen3_4b_inventory.json').read_text())
RULE = REFERENCE['rule']
CHECKPOINT_DIR = 'model'
UNTIED_HEAD = 'lm_head.weight'
EMBEDDING = 'model.embed_tokens.weight'
FLOAT_DTYPES = {'BF16': 2, 'F16': 2, 'F32': 4}
# Tensors whose values are read (not only their headers) to reject NaN/Inf storage: one small norm per block end.
SAMPLED = ('model.layers.0.input_layernorm.weight', 'model.layers.17.post_attention_layernorm.weight', 'model.norm.weight')


def read_header(path: Path) -> tuple[int, dict]:
    with path.open('rb') as stream:
        size = int.from_bytes(stream.read(8), 'little')
        if not 0 < size <= 100_000_000:
            raise ValueError(f'{path.name}: implausible safetensors header size {size}')
        header = json.loads(stream.read(size))
    header.pop('__metadata__', None)
    return size, header


def _decode(raw: np.ndarray, dtype: str, shape: list[int]) -> np.ndarray:
    if dtype == 'BF16':
        array = (raw.view('<u2').astype(np.uint32) << 16).view(np.float32)  # IEEE high bits, as inspect_ternary does
    elif dtype == 'F16':
        array = raw.view('<f2').astype(np.float32)
    elif dtype == 'F32':
        array = raw.view('<f4')
    else:
        raise ValueError(f'Unsupported dtype {dtype}')
    return array.reshape(shape)


def read_tensor(path: Path, header_size: int, spec: dict) -> np.ndarray:
    start, end = spec['data_offsets']
    raw = np.memmap(path, dtype=np.uint8, mode='r', offset=8 + header_size + start, shape=(end - start,))
    return _decode(np.asarray(raw), spec['dtype'], spec['shape'])


def checkpoint_files(snapshot: str | Path) -> list[Path]:
    root = Path(snapshot).resolve() / CHECKPOINT_DIR
    if not root.is_dir():
        raise ValueError(f'The trained checkpoint must be in output/{CHECKPOINT_DIR}/ (directory missing)')
    files = sorted(path for path in root.glob('*.safetensors') if path.is_file() and not path.is_symlink())
    if not files:
        raise ValueError(f'output/{CHECKPOINT_DIR}/ holds no .safetensors files')
    return files


def inspect_inventory(snapshot: str | Path) -> dict:
    """Names, shapes, dtypes and storage of output/model/*.safetensors against the pinned base inventory."""
    inventory = REFERENCE['inventory']
    files = checkpoint_files(snapshot)
    found: dict[str, dict] = {}
    problems: list[str] = []
    for path in files:
        size, header = read_header(path)
        for name, spec in header.items():
            if name in found:
                problems.append(f'{name} stored twice ({found[name]["file"]} and {path.name})')
                continue
            found[name] = {'file': path.name, 'header_size': size, 'dtype': spec['dtype'], 'shape': list(spec['shape']),
                           'data_offsets': list(spec['data_offsets'])}
    expected = dict(inventory)
    untied = UNTIED_HEAD in found and UNTIED_HEAD not in inventory
    if untied:
        expected[UNTIED_HEAD] = {'shape': inventory[EMBEDDING]['shape'], 'dtype': inventory[EMBEDDING]['dtype']}
    missing = sorted(set(expected) - set(found))
    unexpected = sorted(set(found) - set(expected))
    if missing:
        problems.append(f'missing {len(missing)} pinned tensors: {", ".join(missing[:5])}{", ..." if len(missing) > 5 else ""}')
    if unexpected:
        problems.append(f'{len(unexpected)} tensors outside the pinned inventory: {", ".join(unexpected[:5])}{", ..." if len(unexpected) > 5 else ""}')
    stored = 0
    for name, spec in found.items():
        if name not in expected:
            continue
        if spec['shape'] != list(expected[name]['shape']):
            problems.append(f'{name}: shape {spec["shape"]} differs from the pinned {expected[name]["shape"]}')
        width = FLOAT_DTYPES.get(spec['dtype'])
        if width is None:
            problems.append(f'{name}: dtype {spec["dtype"]} is not a floating-point dtype (BF16, F16, F32)')
            continue
        elements = math.prod(spec['shape'])
        if spec['data_offsets'][1] - spec['data_offsets'][0] != elements * width:
            problems.append(f'{name}: stored bytes do not match shape and dtype')
        stored += elements
    return {'passed': not problems, 'problems': problems, 'files': [path.name for path in files],
            'tensors': len(found), 'stored_parameters': stored, 'base_parameters': REFERENCE['base']['parameters'],
            'untied_head': untied, 'dtypes': sorted({spec['dtype'] for spec in found.values()}), 'found': found}


def audit(snapshot: str | Path) -> dict:
    """The complete static audit: layout, then finite values in the sampled tensors. Never raises for a candidate's
    layout mistake; the result says what is wrong."""
    try:
        inventory = inspect_inventory(snapshot)
    except ValueError as error:
        return {'rule': RULE, 'passed': False, 'failure': 'inventory', 'inventory': {'passed': False, 'problems': [str(error)]}}
    result = {'rule': RULE, 'passed': inventory['passed'], 'inventory': {k: v for k, v in inventory.items() if k != 'found'},
              'base': dict(REFERENCE['base'])}
    if not inventory['passed']:
        result['failure'] = 'inventory'
        return result
    root = Path(snapshot).resolve() / CHECKPOINT_DIR
    for name in SAMPLED:
        spec = inventory['found'][name]
        if not np.isfinite(read_tensor(root / spec['file'], spec['header_size'], spec)).all():
            result['inventory']['problems'].append(f'{name}: nonfinite values')
    if result['inventory']['problems']:
        result['inventory']['passed'] = result['passed'] = False
        result['failure'] = 'inventory'
    return result
