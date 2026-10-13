"""Fail-closed architecture evidence and actual checkpoint inspection.

Reviews must be created by the trusted verifier/operator, never read from output/.
Numerical inspection does not prove the server uses the inspected checkpoint; the
independent code review is also mandatory and bound to all submission bytes.
"""
from __future__ import annotations

import hashlib
import json
import os
import math
from pathlib import Path


def materialize_internal_symlinks(submission: str | Path, max_rounds: int = 32) -> int:
    """Replace symlinks that stay inside the submission with the regular files or directories they point to.

    A venv's `bin/python3 -> python3.12` or an untarred toolchain's `libfoo.so -> libfoo.so.1` is an
    ordinary way to package a server, so it must not cost the whole score. The host snapshot and the
    GPU sandbox both call this on the unpacked tree before hashing, so the digest is computed over
    identical regular files. Links with absolute targets, links that leave the submission, dangling
    links and directory cycles raise ValueError; nothing outside the submission is ever read.
    Standard library only: this file is uploaded on its own into the inference sandbox.
    """
    import os
    import shutil
    root = Path(submission).resolve()
    replaced = 0
    for _ in range(max_rounds):
        links = []
        for dirpath, dirnames, filenames in os.walk(root, followlinks=False):
            for name in dirnames + filenames:
                path = Path(dirpath) / name
                if path.is_symlink():
                    links.append(path)
        if not links:
            return replaced
        for link in sorted(links):
            if not link.is_symlink():
                continue  # already replaced through a copied parent directory in this round
            rel = link.relative_to(root).as_posix()
            raw = os.readlink(link)
            if os.path.isabs(raw):
                raise ValueError(f'Submission symlink has an absolute target: {rel} -> {raw}')
            try:
                target = link.resolve(strict=True)
            except (OSError, RuntimeError) as exc:
                raise ValueError(f'Submission contains a dangling or looping symlink: {rel} -> {raw}') from exc
            if not target.is_relative_to(root):
                raise ValueError(f'Submission symlink leaves the submission: {rel} -> {raw}')
            if target.is_dir():
                parent = link.parent.resolve()
                if parent == target or parent.is_relative_to(target):
                    raise ValueError(f'Submission contains a symlink cycle: {rel} -> {raw}')
                link.unlink()
                shutil.copytree(target, link, symlinks=True)  # nested links are resolved in the next round
            elif target.is_file():
                link.unlink()
                shutil.copy2(target, link, follow_symlinks=True)
            else:
                raise ValueError(f'Submission symlink points at a special file: {rel} -> {raw}')
            replaced += 1
    raise ValueError('Submission symlinks did not resolve to regular files within the allowed depth')


def submission_digest(submission: str | Path) -> str:
    root = Path(submission).resolve()
    digest = hashlib.sha256()
    for path in sorted(root.rglob('*')):
        if path.is_symlink():
            raise ValueError('Submission symlinks are not accepted for architecture review')
        if path.is_file():
            relative = path.relative_to(root).as_posix().encode()
            digest.update(len(relative).to_bytes(8, 'big'))
            digest.update(relative)
            digest.update(path.stat().st_size.to_bytes(8, 'big'))
            digest.update(bytes([bool(path.stat().st_mode & 0o111)]))
            with path.open('rb') as stream:
                while chunk := stream.read(1024 * 1024):
                    digest.update(chunk)
    return digest.hexdigest()


def verify_review(submission: str | Path, review_path: str | Path, task_slug: str,
                  required_checks: list[str]) -> dict:
    root = Path(submission).resolve()
    path = Path(review_path).resolve()
    if path.is_relative_to(root):
        raise ValueError('Architecture review must come from trusted verifier storage')
    record = json.loads(path.read_text())
    if (record.get('decision') == 'skipped' and record.get('task') == task_slug
            and os.environ.get('FML_REVIEW_SKIP', '').strip().lower() in ('1', 'true', 'yes')):
        # Operator switch for a reviewer outage (frontier_ml.grading.write_skipped_review): the score is provisional
        # until a real review of the same submission digest passes; the digest binding still holds.
        if record.get('submission_digest') != submission_digest(root):
            raise ValueError('Skipped-review receipt does not match submitted code and weights')
        return record
    if record.get('task') != task_slug or record.get('decision') != 'pass':
        raise ValueError('Architecture review missing or rejected')
    if record.get('submission_digest') != submission_digest(root):
        raise ValueError('Architecture review does not match submitted code and weights')
    if not record.get('reviewer') or not record.get('evidence'):
        raise ValueError('Architecture review needs reviewer identity and cited code/weight evidence')
    from frontier_ml.review_rules import required_checks as required_with_session_checks
    if not all(record.get('checks', {}).get(check) is True for check in required_with_session_checks(record, required_checks)):
        raise ValueError('Architecture review did not establish every required constraint')
    return record


def _inside(root: Path, relative: str) -> Path:
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError('Checkpoint path must be a file inside submission')
    return path


# The input embedding matrix and the output (LM) head are exempt from the ternary fraction. A base
# tensor is exempt when the last two components of its name (module.leaf), compared case-insensitively,
# are one of these modules with a weight or bias leaf: HF `embed_tokens`/`lm_head`, GPT-2 `wte`,
# BLOOM/Falcon `word_embeddings`, Llama-original `tok_embeddings`/`output`, GPT-NeoX `embed_out`.
# Tied heads share the embedding's storage and are exempt through the embedding entry. The optional
# vision tower and MTP head are handled separately below. Main language model norms, biases and every
# attention/MLP/MoE tensor are not exempt: their module component is not in this set.
EMBEDDING_HEAD_MODULES = frozenset({'embed_tokens', 'lm_head', 'wte', 'word_embeddings', 'tok_embeddings',
                                    'embed_out', 'output'})
EMBEDDING_HEAD_LEAVES = frozenset({'weight', 'bias'})
EMBEDDING_HEAD_RULE = ('The input embedding matrix and the output head (module component embed_tokens, lm_head, '
                       'wte, word_embeddings, tok_embeddings, embed_out or output with a weight/bias leaf, matched '
                       'case-insensitively on the last two name components) and their scales are excluded from '
                       'both the numerator and the denominator of the ternary fraction; they may stay floating point.')
OPTIONAL_COMPONENTS_RULE = ('The pinned Qwen vision tower (model.visual.*), MTP head (mtp.*), and their scales '
                            'are also excluded from both the numerator and denominator. These components '
                            'may be omitted or kept unchanged in full precision; they are not used or evaluated.')
# The pinned Qwen ties its output head to the input embedding. A submission may untie it by adding this one
# tensor, shaped like the embedding; it is an output head, so it is exempt like the embedding.
UNTIED_HEAD = 'lm_head.weight'
UNTIED_HEAD_RULE = (f'The output head may be untied: {UNTIED_HEAD}, shaped like the input embedding, is the only '
                    'tensor allowed beyond the pinned inventory.')
TERNARY_EXCLUSION_RULE = EMBEDDING_HEAD_RULE + ' ' + OPTIONAL_COMPONENTS_RULE + ' ' + UNTIED_HEAD_RULE


def is_embedding_or_head(name: str) -> bool:
    """True for base tensor names that the ternary fraction exempts: the input embedding and output head."""
    parts = name.lower().split('.')
    return len(parts) >= 2 and parts[-2] in EMBEDDING_HEAD_MODULES and parts[-1] in EMBEDDING_HEAD_LEAVES


def is_optional_ternary_parameter(name: str) -> bool:
    """Pinned Qwen vision tower (including its merger) or MTP head; neither is evaluated."""
    return name.startswith(('model.visual.', 'mtp.'))


# Stored parameters (ternary codes, scales, retained floats and any safetensors tensor the manifest does not reference)
# may exceed the original text-model parameter count by at most 10%.
PARAMETER_CAP = 1.10


def inspect_ternary(submission: str | Path, inventory: dict | None = None) -> dict:
    """Count actual {-1,0,1} codes against the pinned text-model parameter inventory.

    quantization_manifest.json maps every main language model base tensor to a safetensors file/tensor.
    Vision-tower (model.visual.*) and MTP (mtp.*) tensors may be omitted or remain floating point; they
    and their scales do not affect the fraction. Omitted tensors are recorded under omitted_parameters.
    Supported encodings: literal (any numeric dtype with exact ternary values),
    packed_2bit (uint8, four low-to-high 2-bit codes, 0=-1, 1=0, 2=+1, 3 invalid),
    and float (unquantized residual parameters). Optional scales are explicit
    finite tensors (zero and signed scales are valid); they count against the 5% nonternary allowance.
    The input embedding and output head (is_embedding_or_head) may stay floating point: they and
    their scales leave both the numerator and the denominator. All supplied tensors, including exempt
    embedding/head, vision and MTP tensors, are read and validated, listed in audited_storage, and
    exemptions are reported under excluded_parameters.
    Custom formats need a separately implemented trusted inspector, never pickle.
    """
    import numpy as np
    from safetensors import safe_open

    root = Path(submission).resolve()
    manifest = json.loads((root / 'quantization_manifest.json').read_text())
    if inventory is None:
        inventory = json.loads((Path(__file__).with_name('metadata') / 'qwen3.5-4b.inventory.json').read_text())
        if manifest.get('base_model') != 'Qwen/Qwen3.5-4B' or manifest.get('base_revision') != '851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a':
            raise ValueError('PTQ must identify the pinned Qwen3.5-4B source')
    tensors = manifest.get('tensors', {})
    required = {name for name in inventory if not is_optional_ternary_parameter(name)}
    embedding = next((name for name in inventory if name.endswith('embed_tokens.weight')
                      and not is_optional_ternary_parameter(name)), None)
    untied = {UNTIED_HEAD: inventory[embedding]} if embedding and UNTIED_HEAD not in inventory else {}
    allowed = set(inventory) | set(untied)
    if not isinstance(tensors, dict) or not required <= set(tensors) <= allowed:
        missing = sorted(required - set(tensors)) if isinstance(tensors, dict) else []
        unexpected = sorted(set(tensors) - allowed) if isinstance(tensors, dict) else []
        detail = ''.join(f'; {label} {len(names)}: {", ".join(names[:5])}{", ..." if len(names) > 5 else ""}'
                         for label, names in (('missing', missing), ('not in the pinned inventory', unexpected)) if names)
        raise ValueError('Manifest must map every pinned text-model parameter exactly once; '
                         f'only model.visual.* and mtp.* parameters may be omitted, and only {UNTIED_HEAD} added'
                         + detail)
    used: set[tuple[str, str]] = set()
    audited_scales: dict[tuple[str, str], int] = {}   # scale storage of audited tensors -> element count
    exempt_scales: dict[tuple[str, str], int] = {}    # scale storage referenced only by exempt tensors
    ternary_count = total = 0
    excluded: list[dict] = []
    omitted: list[dict] = []
    def read(reference):
        key = (reference['file'], reference['tensor'])
        path = _inside(root, key[0])
        with safe_open(path, framework='numpy') as checkpoint:
            if checkpoint.get_slice(key[1]).get_dtype() == 'BF16':
                # NumPy has no native BF16 dtype. Decode its exact IEEE high bits.
                with path.open('rb') as stream:
                    header_size = int.from_bytes(stream.read(8), 'little')
                    header = json.loads(stream.read(header_size))[key[1]]
                raw = np.memmap(path, dtype='<u2', mode='r', offset=8 + header_size + header['data_offsets'][0],
                                shape=tuple(header['shape']))
                array = (raw.astype(np.uint32) << 16).view(np.float32)
            else:
                array = checkpoint.get_tensor(key[1])
        return key, array
    for name, spec in {**inventory, **{k: v for k, v in untied.items() if k in tensors}}.items():
        count = math.prod(spec['shape'])
        if name not in tensors:
            omitted.append({'name': name, 'elements': count})
            continue
        item = tensors[name]
        exempt = is_embedding_or_head(name) or is_optional_ternary_parameter(name)
        key, array = read(item)
        used.add(key)
        encoding = item['encoding']
        ternary = False
        if encoding == 'packed_2bit':
            if array.dtype != np.uint8 or array.size != (count + 3) // 4:
                raise ValueError(f'Invalid packed tensor storage: {name}')
            # Bounded temporary memory, including large embedding tensors.
            flat = array.reshape(-1)
            for start in range(0, flat.size, 1_000_000):
                block = flat[start:start + 1_000_000]
                codes = ((block[:, None] >> np.array([0, 2, 4, 6], dtype=np.uint8)) & 3).reshape(-1)
                relevant = min(codes.size, count - start * 4)
                if np.any(codes[:relevant] == 3):
                    raise ValueError(f'Nonternary packed code in {name}')
            ternary = True
        elif encoding in {'literal', 'float'}:
            if list(array.shape) != spec['shape'] or not np.isfinite(array).all():
                raise ValueError(f'Nonfinite tensor or base shape mismatch: {name}')
            if encoding == 'literal':
                if not np.isin(array, [-1, 0, 1]).all():
                    raise ValueError(f'Nonternary literal code in {name}')
                ternary = True
        else:
            raise ValueError(f'Unsupported checkpoint encoding: {encoding}')
        scale_elements = 0
        if 'scale' in item:
            scale_key, scale = read(item['scale'])
            if not np.isfinite(scale).all():
                raise ValueError('Quantization scales must be finite')
            scale_elements = int(scale.size)
            (exempt_scales if exempt else audited_scales)[scale_key] = scale_elements
        if exempt:
            excluded.append({'name': name, 'elements': count, 'encoding': encoding, 'ternary': ternary,
                             'scale_elements': scale_elements})
            continue
        total += count
        if ternary:
            ternary_count += count
    # Scale storage shared between an exempt tensor and an audited one is audited once (as before).
    scales_count = sum(audited_scales.values())
    excluded_scale_elements = sum(size for key, size in exempt_scales.items() if key not in audited_scales)
    audited = total + scales_count
    if audited == 0:
        raise ValueError('No text-model weight elements outside the input embedding and output head to audit')
    # The architecture review traces the active loading path against these
    # references. Training/optimizer artifacts are not inference parameters, and
    # logical parameters may share storage; neither establishes an alternate model.
    ratio = ternary_count / audited
    # Roughly one ternary code per original weight: every tensor in every safetensors file of the submission that the
    # manifest does not reference counts too (a uint8/int8 tensor as four parameters, since it can pack four 2-bit
    # codes), and codes + scales + retained floats + unreferenced tensors may not exceed PARAMETER_CAP times the
    # original text-model parameters outside the exempt embedding/head/vision/MTP tensors. Stacking residual ternary
    # stages, or keeping weights outside the manifest, exceeds it. The exempt tensors themselves (vision tower, MTP head,
    # embedding, head) may sit outside the manifest, since it lists vision and MTP tensors only optionally: an
    # unreferenced tensor with an exempt pinned name and its pinned shape is not counted, once per name.
    referenced = used | set(audited_scales) | set(exempt_scales)
    exempt_specs = {name: list(spec['shape']) for name, spec in {**inventory, **untied}.items()
                    if is_embedding_or_head(name) or is_optional_ternary_parameter(name)}
    exempt_seen, exempt_unreferenced = set(), 0
    unreferenced, unreferenced_count = [], 0
    for path in sorted(root.rglob('*.safetensors')):
        if path.is_symlink() or not path.is_file():
            continue
        relative = path.relative_to(root).as_posix()
        with safe_open(path, framework='numpy') as checkpoint:
            for tensor in checkpoint.keys():
                if (relative, tensor) in referenced:
                    continue
                shape = checkpoint.get_slice(tensor).get_shape()
                dtype = checkpoint.get_slice(tensor).get_dtype()
                if exempt_specs.get(tensor) == list(shape) and tensor not in exempt_seen:
                    exempt_seen.add(tensor)
                    exempt_unreferenced += math.prod(shape)
                    continue
                elements = math.prod(shape) * (4 if dtype in ('U8', 'I8') else 1)
                unreferenced.append({'file': relative, 'tensor': tensor, 'dtype': dtype, 'parameters': elements})
                unreferenced_count += elements
    stored = audited + unreferenced_count
    parameter_limit = PARAMETER_CAP * total
    excluded.sort(key=lambda entry: entry['name'])
    omitted.sort(key=lambda entry: entry['name'])
    return {'passed': ratio > 0.95 and stored <= parameter_limit, 'ternary_fraction': ratio,
            'stored_parameters': stored, 'parameter_limit': parameter_limit, 'parameter_cap': PARAMETER_CAP,
            'parameter_cap_passed': stored <= parameter_limit,
            'unreferenced_parameters': unreferenced_count, 'unreferenced_tensors': unreferenced[:50],
            'unreferenced_exempt_parameters': exempt_unreferenced,
            'ternary_parameters': ternary_count, 'base_parameters': total,
            'scale_parameters': scales_count, 'threshold_exclusive': 0.95,
            'inventory_parameters': sum(math.prod(spec['shape']) for spec in inventory.values()),
            'exclusion_rule': TERNARY_EXCLUSION_RULE,
            'excluded_parameters': excluded,
            'excluded_elements': sum(entry['elements'] for entry in excluded),
            'excluded_scale_elements': excluded_scale_elements,
            'omitted_parameters': omitted,
            'omitted_elements': sum(entry['elements'] for entry in omitted),
            'audited_storage': [{'file': file, 'tensor': tensor}
                                for file, tensor in sorted(used | set(audited_scales) | set(exempt_scales))]}
