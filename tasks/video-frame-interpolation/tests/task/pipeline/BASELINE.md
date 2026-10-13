# GIMM-VFI-R-P quality reference and baseline history

The measured specialist quality target is the released **GIMM-VFI-R-P** checkpoint from
[GSean/GIMM-VFI](https://huggingface.co/GSean/GIMM-VFI), using
[the authors' implementation](https://github.com/GSeanCDAT/GIMM-VFI).
A full private-set quality result completed on 2026-09-30. Data verification and
the pretrained exact-reference scorer control passed. The current subset is
calibrated for an admission pilot. Release still requires
pretrained scorer parity, corruption controls, and validation of the serving-speed gate.
The source license allows non-commercial use; commercial use needs
separate permission. See the [upstream license](https://github.com/GSeanCDAT/GIMM-VFI/blob/main/LICENSE).

## Current two-anchor scoring

The production zero reference is the permitted RAFT-plus-deterministic-synthesis
system in [flow_baseline_results.json](flow_baseline_results.json). The GIMM errors
below are now the score-1 target. GIMM weights remain prohibited for candidates;
using its measured quality as a target does not make it a permitted initialization.
The score averages `clip((baseline - candidate) / (baseline - target), 0, 1)` across
LPIPS/FloLPIPS and 4x/8x/16x. Reaching the target under the new initialization rules
within $50 has not yet been demonstrated. See [METRICS.md](../METRICS.md).

Both saved prediction exports were rescored with current metric code and the pinned
CUDA runtime on a single H100; no new model predictions were generated. The active
[calibration.json](../calibration.json) binds both references, their model provenance
and prediction hashes. [anchor_calibration_results.json](anchor_calibration_results.json)
records the remote run, timings and 0/0.5/1 score controls.

## Historical single-anchor H100 calibration

Saved predictions from run `2cb05727f37f43efbbe4a2943d5831ea` were rescored
with the current production metric implementation and pinned CUDA runtime on
2026-10-01 UTC. This covers all 270 selected intervals and 2,250 generated frames.
The task originally pinned the resulting single-anchor `calibration.json`.
GIMM scored exactly 0 on that historical scale; it scores 1 on the current scale. Exact-reference controls
for all three factors produce LPIPS = FloLPIPS = 0 and PSNR = 100 dB.

| Factor | LPIPS ↓ | Sequence FloLPIPS ↓ | RGB PSNR ↑ |
| --- | ---: | ---: | ---: |
| 4× | 0.01946847 | 0.02447412 | 36.0795 |
| 8× | 0.03089009 | 0.04003089 | 32.7400 |
| 16× | 0.05186808 | 0.07500480 | 28.2371 |

Staging 7.63 GB from the saved volume to local disk took 134.62
seconds. Total calibration and control time was 454.51 seconds.
Reports are in `work/vfi/cuda-calibration/` and the source run's
`cuda-calibration-20261001/` directory on the Modal volume. This validates the
normalization values; full production-transport timing and the admission pilot
remain separate checks. Upstream pretrained parity and corruption rankings also
remain pending.

## Historical full-set result

Run `cd4fb831e28c43a2a31857be21fe1cf2` on the `frontier-vfi-baseline` Modal volume
scored all 927 held-out intervals (309 per factor, 7,725 target frames):

| Factor | LPIPS ↓ | Sequence FloLPIPS ↓ | RGB PSNR ↑ |
| --- | ---: | ---: | ---: |
| 4× | 0.01705562 | 0.01892058 | 36.6923 |
| 8× | 0.03077762 | 0.03952479 | 32.3685 |
| 16× | 0.05889690 | 0.08438056 | 27.7046 |

It retained 339 complete intervals from run `1a1f2023c78e4ff1b8d0a654f3e4c1ce`.
Four H100s generated the remaining 6,588 frames in 834.52 seconds (13.91 minutes,
7.89 frames/second combined, including worker launch and storage overhead). Worker
rates including local startup and I/O ranged from 2.13 to 2.54 frames/second.
These are organizer acquisition measurements, not one-H100 HTTP serving rates.
Sixteen CPU metric workers scored the complete set in 2,411.80 seconds (40.20 minutes).
All containers have finished; the private volume retains predictions and receipts.

The local reports are in `work/vfi/gimm-r-p-parallel/`; they include the exact
calibration values, per-interval metrics, model/data/metric identities and worker
timings. The evaluation hash is
`d570cbb70a2fb4d8a458fd657d6b7d1387096bc5861b5544a97d146b11b625d2`.
These custom-split results are not directly comparable to published full-benchmark
scores. This historical full-set calibration is not the current task calibration.

## Launch

Current acquisition and grading use **one H100 and 270 complete intervals**: 90 per
factor, 2,250 generated frames. The generation budget is **2,250 seconds (37m30s)**
including HTTP transport and export, with startup and quality scoring separate.
`vfi_selection.py` fixes membership using a seeded
SHA-256 ordering of interval IDs within each factor (seed 0). It never consults
baseline errors and never drops intermediate frames within a selected interval.
The HF payload retains the full set; selection happens before inference and scoring.
Calibration binds both the selection and selected media. Full-set calibration
cannot be used for the smaller set.

The saved full-run interval metrics give the following current-subset baseline without
another GPU or metric run:

| Factor | LPIPS ↓ | Sequence FloLPIPS ↓ | RGB PSNR ↑ |
| --- | ---: | ---: | ---: |
| 4× | 0.01946850 | 0.02447426 | 36.0795 |
| 8× | 0.03089008 | 0.04003008 | 32.7400 |
| 16× | 0.05186809 | 0.07500480 | 28.2371 |

`work/vfi/gimm-r-p-270/subset-metrics.json` records the selected IDs and hashes of
the source report/data. This is a reaggregation of measured values, not a newly
activated calibration. The smaller sample has greater statistical uncertainty;
keep the fixed subset rather than trying seeds for favorable baseline numbers.

A fresh **270-interval** run, `2cb05727f37f43efbbe4a2943d5831ea`, completed on
2026-09-30. One H100 generated all 2,250 frames over the public loopback HTTP path
in **941.69 seconds (15m42s)** including input reads and output writes: **2.389
frames/second**, inside the 2,250-second budget. Requests took 882.11 seconds;
startup took 4.52 seconds. The slowest interval's request time divided by its
generated frame count was 0.634 seconds/frame.

Its CPU scoring controller failed and replacement workers restarted, so that
phase was stopped. A separate H100 then scored the saved predictions, after the
generation GPU had exited. The GPU metric pass took **851.23 seconds**, or
**976.13 seconds (16m16s)** including dataset verification and setup. These times
exclude the interrupted CPU work. Parallel file prefetching was introduced near
interval 210; this is not a clean measurement of prefetch enabled from startup.

| Factor | GPU LPIPS ↓ | GPU sequence FloLPIPS ↓ | RGB PSNR ↑ |
| --- | ---: | ---: | ---: |
| 4× | 0.01946847 | 0.02447412 | 36.0795 |
| 8× | 0.03089009 | 0.04003089 | 32.7400 |
| 16× | 0.05186808 | 0.07500480 | 28.2371 |

LPIPS/FloLPIPS factor means differ from the earlier CPU subset by at most
0.00000082. That comparison uses earlier predictions of the same intervals, so
it is not strict CPU/CUDA parity on identical new pixels. This historical diagnostic
was superseded by the activated calibration above, bound to the current metric code
and CUDA runtime. The historical CPU runner below remains useful only for diagnostics. Reports and recovery
details are in `work/vfi/gimm-r-p-270-run/`, including `run-summary.json` and
`gpu-metrics.json`. The metric report was saved before Modal's return upload
failed, and was recovered directly from the volume. Both apps have stopped.

The earlier **90-interval** one-H100 HTTP run `77d48cb7ccbf4da190c728a83eec696c` completed generation
on 2026-09-30. It generated all 750 frames in **323.15 seconds (5 minutes 23 seconds)**
including input reads and output writes. Requests took 296.75 seconds, or **2.53
generated frames/second**; model-server startup took another 8.04 seconds. The
response format was raw RGB, with sequential requests and the same FP32 model
settings. This is one measured pass over loopback HTTP, not a production Harbor
transport calibration. Exact receipts and reports are in `work/vfi/gimm-r-p-small-run/`.
All 90 intervals also completed fresh quality scoring in 551.21 seconds (9 minutes
11 seconds) on four CPU workers after the H100 exited. LPIPS and FloLPIPS factor
means agree with the saved 90-interval values within 0.0000006. The run produced a
90-interval `calibration.json`, which cannot calibrate the current 270-interval set.
This historical 90-interval calibration has been superseded by the 270-interval CUDA calibration above. Both inference and scoring containers have finished.

| Factor | Generated frames | HTTP request seconds | Generated frames/second |
| --- | ---: | ---: | ---: |
| 4× | 90 | 40.74 | 2.21 |
| 8× | 210 | 77.09 | 2.72 |
| 16× | 450 | 178.91 | 2.52 |

For the current acquired dataset, quality-only continuation retains completed
selected intervals and uses at most one H100 for any remaining ones:

```bash
uv run modal run --detach tasks/video-frame-interpolation/pipeline/vfi_quality_modal.py \
  --previous <stopped-run-id> --workers 1 --out work/vfi/gimm-r-p-270-quality
```

It validates and copies the completed prefix, predicts only the remaining intervals,
and stores generated frames as raw RGB uint8 `.rgb` files. There is no PNG codec,
HTTP, or base64 on this organizer path. The quantization, model weights, FP32 settings,
native resolution, interval boundaries, and quality metrics stay the same. Four
CPU workers score the selected intervals once. A completely retained subset needs
no GPU allocation. The resulting `pass-1.json` records worker
timings and explicitly does **not** claim single-H100 serving throughput; use the
public HTTP path separately when calibrating a serving budget.

From a checkout with working network access, an authenticated Modal account and
HF access to `inductionlabs/frontier-ml-tasks`:

```bash
uv run modal run --detach tasks/video-frame-interpolation/pipeline/vfi_modal.py \
  --out work/vfi/gimm-r-p-270-run
```

The command is a paid organizer run. It makes no HF uploads and does not change
the task configuration. All large downloads and outputs stay on the private
`frontier-vfi-baseline` Modal volume, under the printed run ID. Only JSON reports
return to the local output directory. Modal tears down single-use containers when
their calls finish; the volume persists for inspection and reproducibility.

The phases run sequentially:

1. **CPU preparation.** Resolve the HF dataset and model revisions, GIMM source,
   and LPIPS source to immutable commits. Inspect the remote folder structure,
   download only `video-frame-interpolation/`, run `vfi_data.verify`, and hash all
   held-out rows and reference/endpoint PNGs. Fetch `gimmvfi_r_arb_lpips.pt` plus
   the required `raft-things.pth`; fetch/hash the three scorer weight files.
2. **H100 inference.** One H100, exactly 8 CPUs and 80 GiB host RAM, network blocked.
   Run one complete held-out pass by default. The server loads
   strict checkpoint state, warms up with synthetic images, and must become ready
   within 300 seconds. The public HTTP client sends each interval once,
   sequentially, with 300 seconds per request and a 2,250-second whole-pass budget.
   Predictions include every intermediate frame and exact supplied endpoints.
3. **Parallel CPU scoring.** After inference, four workers split the 270 intervals,
   each using the exact trusted CPU runtime (torch 2.8.0, NumPy 2.2.6, Pillow 11.3.0,
   8 threads). They evaluate the same complete first-pass export. The collector
   requires every interval exactly once, matching data/weight/code identities,
   and combines interval results before factor averaging. Each worker has a
   two-hour metric watchdog. This preserves sequential metric calculations while
   reducing wall time; no best-of-several prediction selection occurs.

Inference is float32 without TF32, autocast or test-time augmentation. It follows
the authors' 32-pixel symmetric replicate padding and evaluates all requested
timestamps together, with no downsampling. PNG export clips to [0,1], multiplies
by 255 and truncates to uint8, matching the reference export. Raw output preserves
exactly those same pixels. The HTTP server supports raw RGB binary responses and
uses compression level 1 for legacy PNG responses. No training occurs.
Python 3.10 and CuPy 11 retain APIs used by the unmodified upstream code; the
separate CPU scorer uses Python 3.12. The GPU image has been built and real private intervals have been processed
successfully.

## Reports and resume

The local directory receives:

- `modal-run.json` (or `quality-run.json` for parallel continuation): run/call IDs
  and launch parameters.
- `prepared.json`: resolved commits, checkpoint hashes, data verification/counts,
  complete-media evaluation hash and inference settings.
- `generation.json` (or `pass-1.json` for parallel continuation): hardware, pass
  reports, startup times and diagnostic timing.
- `metrics.json`: raw quality values and every interval's metrics.
- `calibration.json`: the measured normalization baseline.
- `metrics/lock.json`: exact scorer weight URLs, sizes and hashes.

Use `modal run --detach` for unattended runs. A remote controller starts prediction
and scoring after preparation, so all phases continue if the local CLI disconnects.
Rerun the same command with the same output directory to retrieve the existing
controller's result without launching duplicate GPU work. The saved call IDs also
allow inspection of failures; clear them only after the stopped run is diagnosed.
To recover another machine's run, add `--run-id <printed-32-character-id>` and use
the same revision arguments. A partially completed generation pass is retained
for diagnosis and deliberately fails on reuse rather than overwriting evidence;
start with a new output directory/run ID after fixing it. Failed runs are not
automatically retried. Volume paths also contain server logs, progress,
prediction exports and acquired data/weights. Do not put them in the public task.

For a new run after changing the runner, `--reuse-acquisition <old-run-id>` reuses
its downloaded dataset, upstream checkout and metric weights. The runner rehashes
all private media and metric weights and records the source receipt digest. It
creates fresh prediction exports; incomplete old predictions remain available
for diagnosis. Keep both run directories because the acquired assets are linked.

Acquisition watchdogs are operational limits, not a production speed gate. Initial
native 720p measurements took about 2.45 seconds per scored adjacent frame pair on
eight CPUs; the completed parallel run above establishes actual scoring capacity.

To repeat an acquisition exactly in a new run, pass `--data-revision`,
`--model-revision`, `--source-revision`, and `--lpips-revision` using the full
commits recorded in `prepared.json`. Branch names are resolved only on initial
acquisition; resumes use the persisted commits and bytes.

## Timing and release

`generation.json` reports both complete export-pass time and trusted loopback HTTP
request time. A single quality pass does not propose a production budget. If an
organizer explicitly runs three complete timing passes, the helper proposes
**twice their median request time** as a generation budget. That includes endpoint
JSON/PNG decoding, model execution and raw RGB response transfer; it excludes
dataset reads, export writes, startup and quality scoring.
This is a measurement/proposal, not an activated production gate. The current
production verifier times host transport and export too; its timing must be aligned
with the proposal before installing this request-time threshold.

Neither the runner nor an arbitrary measured report automatically declares the
task ready. The data, weights and H100 calibration are now pinned for the admission
pilot. Before release, complete pretrained scorer parity and corruption controls,
validate production transport, and confirm the disclosed runtime gate and improvement
headroom. Official grading rejects missing or changed calibration inputs.
See [METRICS.md](../METRICS.md) for the exact scoring and release checks.

The uploaded payload at HF commit `6049d6652b1cdaee80bf0ec290e366d60743f529`
contains 3 public intervals (one per factor) and 927 held-out intervals (309 per
factor, 7,725 target frames). Its handoff manifest uses schema 1 and its JSONL rows
use `factor`, `inputs`, and timestamped `targets`. The public client and scorer
normalize these rows to the same version-2 wire requests in memory; no media or
uploaded files are rewritten. The uploaded split is per interval, and recording
groups are shared across the split. Verification checks every file checksum,
official schedule, frame mapping and exact cross-split RGB overlap, and explicitly
reports the recording overlap instead of claiming recording independence.

The legacy `vfi_finish.py` can attach to an existing three-pass run, retain its first complete
export, cancel the remaining inference/controller calls, and launch metric workers
for that original workload. It records its completion call in `parallel-run.json`; its output
includes `pass-1.json` instead of inventing a three-pass generation report.
