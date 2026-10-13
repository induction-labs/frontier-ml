"""Operator-only admission baseline for music-transcription: YourMT3+ on one H100 (never shipped; costs GPU time).

    set -a; source .env; set +a
    # Space-faithful decoding (the Space's own model.inference), segments of many clips batched together:
    uv run modal run tasks/music-transcription/pipeline/amt_yourmt3.py --split validation --out V.json --batch 128
    # Fast path (same greedy outputs up to batch-shape TF32 noise), used for the throughput numbers:
    uv run modal run tasks/music-transcription/pipeline/amt_yourmt3.py --split test --out T.json \
        --batch 1500 --compact --fast-moe --workers 8 --latency-clips 20
"""
import hashlib
import json
import sys
import time
from pathlib import Path

import modal

HERE = Path(__file__).resolve().parent
TASK = HERE.parent
REPO = TASK.parent.parent

SPACE = "mimbres/YourMT3"
SPACE_REVISION = "5e66c1ea173a8186e0d20432b841d3180cc015b5"
EXP = "mc13_256_g4_all_v7_mt3f_sqr_rms_moe_wf4_n8k2_silu_rope_rp_b36_nops"
CHECKPOINT = f"{EXP}@last.ckpt"
# The Space's argument list for "YPTF.MoE+Multi (noPS)" (app.py); '-pr' only configures a Lightning trainer.
ARGS = [CHECKPOINT, "-p", "2024", "-tk", "mc13_full_plus_256", "-dec", "multi-t5", "-nl", "26", "-enc", "perceiver-tf",
        "-sqr", "1", "-ff", "moe", "-wf", "4", "-nmoe", "8", "-kmoe", "2", "-act", "silu", "-epe", "rope", "-rp", "1",
        "-ac", "spec", "-hop", "300", "-atc", "1", "-pr", "32"]
SPACE_DIR = "/yourmt3"


def _fetch_space():
    from huggingface_hub import snapshot_download
    snapshot_download(SPACE, repo_type="space", revision=SPACE_REVISION, local_dir=SPACE_DIR,
                      allow_patterns=["model_helper.py", "amt/src/**", f"amt/logs/2024/{EXP}/checkpoints/last.ckpt"])


image = (modal.Image.debian_slim(python_version="3.11")
         .apt_install("libsndfile1")
         .pip_install("torch==2.4.1", "torchaudio==2.4.1", "transformers==4.45.1", "pytorch-lightning==2.4.0",
                      "torchmetrics==1.4.2", "numpy==1.26.4", "scipy==1.13.1", "scikit-learn==1.5.2", "einops==0.8.0",
                      "mido==1.3.2", "mir_eval==0.7", "deprecated==1.2.14", "wandb==0.18.3", "soundfile==0.12.1",
                      "requests==2.32.3", "tqdm==4.66.5", "huggingface_hub==0.25.2")
         .run_function(_fetch_space))
app = modal.App("amt-yourmt3", image=image)

# Raw decoded programs are the representative program of each YourMT3 (MT3 "full plus") class, e.g. 0 Acoustic Piano,
# 2 Electric Piano, 8 Chromatic Percussion, 16 Organ, 24/26/29 guitars, 32/33 basses, 40-43 violin..contrabass,
# 46 Orchestral Harp, 47 Timpani, 48 String Ensemble, 50 Synth Strings, 52 Choir, 55 Orchestra Hit, 56-63 brass,
# 64-71 reeds, 73 Pipe, 80 Synth Lead, 88 Synth Pad, 100/101 singing. Harp, timpani, choir/voice, orchestra hit, synth
# lead/pad, singing and drums are dropped; synth strings count as strings.
FAMILIES = (("piano", 0, 7), ("mallets", 8, 15), ("organ", 16, 23), ("guitar", 24, 31), ("bass", 32, 39),
            ("strings", 40, 45), ("strings", 48, 51), ("brass", 56, 63), ("reed", 64, 71), ("pipe", 72, 79))


def program_class(program, is_drum):
    if is_drum:
        return None
    for name, low, high in FAMILIES:
        if low <= program <= high:
            return name
    return None


def _moe_forward_tensor_index(self, hidden_states):
    """MixtralSparseMoeBlock.forward (amt/src/model/ff_layer.py) with tensor instead of Python-list indexing.

    The Space's version converts the token indices of every expert to Python lists (`.tolist()`), which for a batch
    of B segments x 26 latents x 110 frames tokens dominates encoder time; the arithmetic here is identical."""
    import torch
    import torch.nn.functional as F
    batch_size, sequence_length, hidden_dim = hidden_states.shape
    hidden_states = hidden_states.view(-1, hidden_dim)
    router_logits = self.gate(hidden_states)
    routing_weights = F.softmax(router_logits, dim=1, dtype=torch.float)
    routing_weights, selected_experts = torch.topk(routing_weights, self.top_k, dim=-1)
    routing_weights /= routing_weights.sum(dim=-1, keepdim=True)
    routing_weights = routing_weights.to(hidden_states.dtype)
    final_hidden_states = torch.zeros((batch_size * sequence_length, hidden_dim), dtype=hidden_states.dtype,
                                      device=hidden_states.device)
    expert_mask = torch.nn.functional.one_hot(selected_experts, num_classes=self.num_experts).permute(2, 1, 0)
    for expert_idx in range(self.num_experts):
        idx, top_x = torch.where(expert_mask[expert_idx])
        if top_x.shape[0] == 0:
            continue
        current_state = hidden_states[top_x].reshape(-1, hidden_dim)
        current_hidden_states = self.experts[expert_idx](current_state) * routing_weights[top_x, idx, None]
        final_hidden_states.index_add_(0, top_x, current_hidden_states.to(hidden_states.dtype))
    return final_hidden_states.reshape(batch_size, sequence_length, hidden_dim), router_logits


def _load_model(fast_moe=False):
    import os
    os.chdir(SPACE_DIR)
    sys.path.insert(0, f"{SPACE_DIR}/amt/src")
    sys.path.insert(0, SPACE_DIR)
    import torch
    from model_helper import load_model_checkpoint
    if fast_moe:
        from model.ff_layer import MixtralSparseMoeBlock
        MixtralSparseMoeBlock.forward = _moe_forward_tensor_index
    model = load_model_checkpoint(args=list(ARGS), device="cpu")
    return model.to("cuda").eval(), torch


def _segments(model, flac, torch):
    import io

    import soundfile as sf
    import torchaudio
    from utils.audio import slice_padded_array
    data, sr = sf.read(io.BytesIO(flac), dtype="float32", always_2d=True)
    audio = torch.from_numpy(data.T.copy()).mean(dim=0, keepdim=True)
    audio = torchaudio.functional.resample(audio, sr, model.audio_cfg["sample_rate"])
    frames = model.audio_cfg["input_frames"]
    return torch.from_numpy(slice_padded_array(audio.numpy(), frames, frames).astype("float32")).unsqueeze(1)


def _decode(model, token_batches, n_segments):
    """Token arrays of one clip ([(b, C, L)], b summing to n_segments) -> raw notes (Space post-processing)."""
    from utils.event2note import merge_zipped_note_events_and_ties_to_notes
    from utils.note2event import mix_notes
    frames, rate = model.audio_cfg["input_frames"], model.audio_cfg["sample_rate"]
    starts = [frames * i / rate for i in range(n_segments)]
    channels = []
    for ch in range(model.task_manager.num_decoding_channels):
        zipped, _, _ = model.task_manager.detokenize_list_batches([a[:, ch, :] for a in token_batches], starts,
                                                                  return_events=True)
        notes, _ = merge_zipped_note_events_and_ties_to_notes(zipped)
        channels.append(notes)
    return [{"onset": float(n.onset), "offset": float(n.offset), "pitch": int(n.pitch), "program": int(n.program),
             "is_drum": bool(n.is_drum)} for n in mix_notes(channels)]


def _generate_compact(model, torch, x, encoder_chunk=128):
    """model.inference(x) for the multi-T5 decoder, with finished rows dropped from the step batch.

    Same greedy decoding as t5mod_helper.task_cond_dec_generate (no task prefix; a row is finished once a decoding
    step emits <eos>=1, later positions are <pad>=0; the loop ends when every row is finished or at the maximum
    length), but each of the B x 13 channel rows is independent, so rows that finished are removed from the batch
    (and from the KV cache) instead of being decoded to the end of the longest row."""
    enc = []
    for i in range(0, len(x), encoder_chunk):  # spectrogram -> pre-encoder -> encoder -> pre-decoder (as inference())
        h = model.pre_encoder(model.spectrogram(x[i:i + encoder_chunk]))
        enc.append(model.pre_decoder(model.encoder(inputs_embeds=h)["last_hidden_state"]))
    enc = torch.cat(enc)  # (B, C, T, D)
    b, c, t, d = enc.shape
    n, max_length = b * c, model.max_total_token_length
    enc = enc.reshape(n, 1, t, d)
    tok = torch.zeros((n, 1, 1), dtype=torch.long, device=enc.device)  # <bos> = decoder_start_token_id 0
    hs, past = model.decoder(inputs_embeds=model.embed_tokens(tok), encoder_hidden_states=enc, return_dict=False)
    tok = model.lm_head(hs).argmax(-1)
    out = torch.zeros((n, max_length), dtype=torch.long, device=enc.device)
    out[:, 0] = tok.view(-1)
    active = torch.arange(n, device=enc.device)
    length = 1
    for _ in range(max_length - 1):
        hs, past = model.decoder(inputs_embeds=model.embed_tokens(tok), encoder_hidden_states=enc,
                                 past_key_values=past, return_dict=False)
        tok = model.lm_head(hs).argmax(-1)  # (n_active, 1, 1)
        out[active, length] = tok.view(-1)
        length += 1
        keep = tok.view(-1) != 1
        n_keep = int(keep.sum())
        if n_keep == 0:
            break
        if n_keep < len(keep):
            idx = keep.nonzero().squeeze(1)
            active, tok, enc = active[idx], tok[idx], enc[idx]
            past = tuple(tuple(p[idx] for p in layer) for layer in past)
    return out[:, :length].reshape(b, c, length)


_POST = None  # (audio_cfg, task_manager) for forked detokenization workers


def _decode_job(job):
    token_batches, n_segments = job
    return _decode(_POST, token_batches, n_segments) if n_segments else []


def _infer(model, torch, x, dtype, compact=False):
    x = x.cuda(non_blocking=True)
    run = (lambda: _generate_compact(model, torch, x)) if compact else (lambda: model.inference(x, None))
    with torch.inference_mode():
        if dtype == "fp32":
            return run().cpu().numpy()
        with torch.autocast("cuda", dtype={"bf16": torch.bfloat16, "fp16": torch.float16}[dtype]):
            return run().cpu().numpy()


@app.function(gpu="H100!", cpu=8, memory=32768, timeout=3600, single_use_containers=True)
def transcribe(clips, batch=128, dtype="fp32", latency_clips=0, compact=False, fast_moe=False, workers=0):
    """clips: [(id, flac bytes)] -> {"notes": {id: [raw notes]}, "timing": {...}}.

    workers > 0 decodes/resamples FLACs in a thread pool and detokenizes clips in a forked process pool."""
    started = time.monotonic()
    model, torch = _load_model(fast_moe)
    load_seconds = time.monotonic() - started
    info = {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "load_seconds": load_seconds}
    print(f"Loaded baseline on {info['gpu']} in {load_seconds:.2f}s; warming up", flush=True)

    warm = _segments(model, clips[0][1], torch)
    _infer(model, torch, warm.repeat(max(1, batch // len(warm) + 1), 1, 1)[:batch], dtype, compact)
    torch.cuda.synchronize()
    print(f"Warmup complete; transcribing {len(clips)} clips", flush=True)
    threads = processes = None
    if workers:
        import multiprocessing
        import types
        from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
        global _POST
        _POST = types.SimpleNamespace(audio_cfg=model.audio_cfg, task_manager=model.task_manager)
        threads = ThreadPoolExecutor(workers)
        processes = ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("fork"))
        list(processes.map(_decode_job, [([], 0)] * workers))  # start the workers before timing

    t0 = time.monotonic()
    if threads:
        segs = list(threads.map(lambda c: _segments(model, c[1], torch), clips))
    else:
        segs = [_segments(model, flac, torch) for _, flac in clips]
    t1 = time.monotonic()
    stream = torch.cat(segs).pin_memory()
    owner = [i for i, s in enumerate(segs) for _ in range(len(s))]
    per_clip = [[] for _ in clips]
    for start in range(0, len(stream), batch):
        preds = _infer(model, torch, stream[start:start + batch], dtype, compact)
        print(f"Inferred {min(start + batch, len(stream))}/{len(stream)} segments", flush=True)
        lo = start
        while lo < start + len(preds):  # split the batch back into per-clip pieces
            clip = owner[lo]
            hi = lo
            while hi < start + len(preds) and owner[hi] == clip:
                hi += 1
            per_clip[clip].append(preds[lo - start:hi - start])
            lo = hi
    t2 = time.monotonic()
    if processes:
        jobs = [(per_clip[i], len(segs[i])) for i in range(len(clips))]
        notes = dict(zip([cid for cid, _ in clips], processes.map(_decode_job, jobs, chunksize=4)))
    else:
        notes = {cid: _decode(model, per_clip[i], len(segs[i])) for i, (cid, _) in enumerate(clips)}
    t3 = time.monotonic()
    print(f"Decoded all {len(clips)} clips in {t3 - t0:.2f}s", flush=True)
    info.update({"batch_segments": batch, "dtype": dtype, "compact": compact, "fast_moe": fast_moe, "workers": workers,
                 "clips": len(clips),
                 "segments": len(stream),
                 "decode_resample_seconds": t1 - t0, "inference_seconds": t2 - t1, "detokenize_seconds": t3 - t2,
                 "wall_seconds": t3 - t0, "peak_memory_gb": torch.cuda.max_memory_allocated() / 2**30})

    latencies = []
    for cid, flac in clips[:latency_clips]:  # one clip at a time: all its segments in one batch
        torch.cuda.synchronize()
        a = time.monotonic()
        s = _segments(model, flac, torch)
        _decode(model, [_infer(model, torch, s, dtype, compact)], len(s))
        latencies.append(time.monotonic() - a)
    info["latency_seconds"] = latencies
    if processes:
        processes.shutdown()
    if threads:
        threads.shutdown()
    return {"notes": notes, "info": info}


def _load_rows(split):
    path = {"validation": REPO / "data/music-transcription/public/validation.jsonl",
            "test": REPO / "data/music-transcription/private/test.jsonl"}[split]
    rows = [json.loads(line) for line in path.read_text().splitlines() if line.strip()]
    return rows, path.parent


def _score(rows, raw):
    sys.path.insert(0, str(TASK))
    import amt_notes
    records, programs, dropped = [], {}, 0
    for row in rows:
        predicted = []
        for n in raw[row["id"]]:
            key = "drum" if n["is_drum"] else str(n["program"])
            programs[key] = programs.get(key, 0) + 1
            name = program_class(n["program"], n["is_drum"])
            if name is None or n["offset"] <= n["onset"]:
                dropped += 1
                continue
            predicted.append({"onset": n["onset"], "offset": n["offset"], "pitch": n["pitch"], "instrument": name})
        counts = amt_notes.clip_counts(row["notes"], predicted)
        records.append({"id": row["id"], **counts})
    return amt_notes.aggregate(records), records, programs, dropped


@app.local_entrypoint()
def main(split: str = "validation", out: str = "", batch: int = 128, dtype: str = "fp32", latency_clips: int = 0,
         compact: bool = False, fast_moe: bool = False, workers: int = 0):
    rows, root = _load_rows(split)
    row_path = root / ("validation.jsonl" if split == "validation" else "test.jsonl")
    manifest_path = root.parent / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    dataset_sha256 = hashlib.sha256(row_path.read_bytes()).hexdigest()
    if dataset_sha256 != manifest["files"][str(row_path.relative_to(root.parent))]["sha256"]:
        raise ValueError("Evaluation rows differ from the dataset manifest")
    clips = [(row["id"], (root / row["audio"]).read_bytes()) for row in rows]
    for row, (_, flac) in zip(rows, clips, strict=True):
        key = str((root / row["audio"]).relative_to(root.parent))
        if hashlib.sha256(flac).hexdigest() != manifest["files"][key]["sha256"]:
            raise ValueError(f"Audio differs from manifest: {row['id']}")
    output = Path(out) if out else None
    if output:
        output.parent.mkdir(parents=True, exist_ok=True)
    result = transcribe.remote(clips, batch=batch, dtype=dtype, latency_clips=latency_clips, compact=compact,
                               fast_moe=fast_moe, workers=workers)
    # Retain raw predictions before scoring, so a scoring/serialization failure never requires another GPU pass.
    if output:
        output.with_suffix(".raw.json").write_text(json.dumps(result, separators=(",", ":")))
    info = result["info"]
    info["audio_seconds"] = sum(row["duration_s"] for row in rows)
    info["rtf_audio_per_wall"] = info["audio_seconds"] / info["wall_seconds"]
    metrics, records, programs, dropped = _score(rows, result["notes"])
    summary = {"split": split, "space": SPACE, "space_revision": SPACE_REVISION, "checkpoint": CHECKPOINT,
               "args": ARGS, "timing": info, "metrics": metrics, "raw_program_counts": programs,
               "dropped_notes": dropped, "per_clip_counts": records,
               "dataset_version": manifest["dataset_version"], "evaluation_sha256": dataset_sha256,
               "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
               "baseline_script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
               "timing_scope": "Direct batched inference after warmup; excludes transfer, startup and HTTP serving. "
                               "Not a formal serving, size or architecture gate evaluation."}
    print(json.dumps({"split": split, "timing": info, "score": metrics["score"],
                      "agnostic": metrics["instrument_agnostic_onset_f1"], "onset_offset": metrics["onset_offset_f1"],
                      "per_class_f1": {c: v["f1"] for c, v in metrics["per_instrument"].items()}}, indent=1))
    if output:
        output.write_text(json.dumps(summary, indent=1) + "\n")
