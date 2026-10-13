# Quality scoring and calibration

The raw metric protocol is `vfi-lpips-alex01-flolpips-sequence-v1`; the reward
protocol is `vfi-lpips-flolpips-two-anchor-v2`.
It is specific to independent SNU-FILM-arb intervals and the uploaded interval split;
its aggregate reward is not a published leaderboard score.

Evaluation selects 90 complete private intervals per factor (270 intervals and 2,250
generated frames) using `vfi_selection.py`, seed 0. Both inference and scoring apply
the same fixed, score-independent selection. Its method, seed, selected IDs and
media hashes are bound into the evaluation identity. A full-set calibration is
rejected for this subset; public interface checks continue to use all public rows.

## Definition

| Measurement | Inputs and averaging | Reward weight |
| --- | --- | --- |
| LPIPS | AlexNet ImageNet features, LPIPS v0.1 linear weights; generated frames only | 50% |
| Sequence FloLPIPS | The same feature errors, spatially weighted by PWC-Net flow differences on adjacent frame pairs | 50% |
| RGB PSNR | Generated frames only; 8-bit RGB, peak 255; per-frame dB capped at 100 | Diagnostic |

All metrics use native-resolution RGB uint8 pixels from PNG or headerless `.rgb` files.
Raw frames have exactly `height*width*3` bytes in row-major RGB order. This storage
change preserves the decoded pixels and avoids an image codec. No cropping, border removal,
YUV conversion, or resizing is applied to LPIPS/PSNR. Output quantization has already
fixed the submitted values to 8-bit RGB. Both dimensions must be at least 32 for
AlexNet. Models run in eval mode, float32, without gradients, autocast or TF32.
Production scoring uses one H100 with torch 2.5.1+cu118, CUDA 11.8, NumPy 1.26.4
and Pillow 11.3.0. Runtime versions, including the CUDA build, are recorded in the
baseline and must match during grading. CPU scoring remains an organizer diagnostic.

For factor `f`, the complete interval contains frames `0,...,f`; the endpoints
are the exact supplied PNGs. Frame LPIPS/PSNR average over `1,...,f-1`.
For every adjacent pair `i -> i+1`, sequence FloLPIPS computes forward PWC-Net
flow on reference and reconstructed frames, takes their vector difference,
bilinearly resizes that difference to each AlexNet layer's spatial resolution,
and normalizes its Euclidean magnitude into spatial weights. Those weights
average the corresponding learned LPIPS error map. Sum the five layer values,
then average over the `f` pairs. This is the **sequence** formulation in the
[FloLPIPS reference implementation](https://github.com/danier97/flolpips/blob/main/flolpips.py),
not its alternate middle-frame-to-endpoints triplet wrapper. Different intervals
are never joined. The first pair contributes zero appearance error because its
current frame is a supplied endpoint; this convention matches that sequence formulation.

PWC-Net internally resizes each pair to the next multiple of 64 and rescales flow
back to native coordinates. It receives RGB in [0,1], as in the FloLPIPS sequence
implementation. Its correlation volume is computed with PyTorch operations on
CPU or CUDA, replacing the upstream CUDA-only kernel. The checkpoint structure,
flow scale, feature ordering and backwarping conventions are retained. We do not
substitute another flow model. This implementation has structural/numerical unit
tests; parity against the upstream pretrained implementation still needs to be
measured on the machine acquiring the weights before freezing the benchmark.

Two explicit differences from the upstream video-file demo are part of our versioned
protocol: we read the original RGB PNGs directly instead of an FFmpeg YUV420
round-trip, and **exactly zero flow-difference mass uses uniform spatial weights**.
The latter avoids an undefined 0/0 for identical flow, including perfect references
and static intervals. Nonfinite flow or metric outputs fail evaluation.
FloLPIPS is motion-weighted perceptual error, not a separate endpoint-error or
pure flicker loss; it does not guarantee that every temporal corruption increases
its score. Inspect control rankings before release.

Average interval values within each factor. For each factor and quality metric:

```text
improvement[f, m] = clip((baseline[f, m] - mean_metric[f, m]) /
                         (baseline[f, m] - target[f, m]), 0, 1)
reward = sum(improvement[f, m] for f in [4,8,16] for m in [LPIPS,FloLPIPS]) / 6
```

Thus each factor has weight 1/3 regardless of its number of frames or intervals.
Every factor must be present. Both measured anchors must be finite and strictly
positive, with the target strictly lower than the starting baseline in every
component. Matching the permitted RAFT-plus-splatting starting system earns 0;
closing half the gap earns 0.5; matching or beating the measured GIMM quality
target in all six components earns 1. Perfect reconstruction is not required.
GIMM remains prohibited for candidate initialization and use; it is only the
quality target. Reaching that quality with a permitted $50 training run has not
yet been demonstrated. Clipping applies separately to each component; PSNR has
no effect on reward. Reports retain all raw factor values, six normalized components,
calibration/weight/data hashes, review checks and timing. Incomplete runs have no
partial quality score.

## Offline weights

`vfi_metrics.py` and `vfi_pwc.py` implement the trusted scorer. No submitted code is
imported, and no metric downloads occur during grading. The standard public
`validate.py` stays dependency-free and exports raw RGB targets without a quality score;
organizers can score those exports using the CLI below. The score formula in
`vfi_scoring.py` is shipped with that public client.

On a networked machine, resolve and inspect a full commit of
[PerceptualSimilarity](https://github.com/richzhang/PerceptualSimilarity), then run:

```bash
python3 tasks/video-frame-interpolation/pipeline/metric_bundle.py fetch \
  --lpips-revision <full-40-character-commit> --output work/vfi-metrics
```

This fetches the official AlexNet ImageNet checkpoint, LPIPS v0.1 AlexNet linear
weights, and the default PWC-Net checkpoint used by FloLPIPS. It records every URL,
byte count and SHA-256 in `lock.json`. Review the acquired files and pin the
resulting lock in `tasks/video-frame-interpolation/scorer/metric-lock.json`.
No weights are committed to Git or put in the public task data. The seed's
`scorer/Dockerfile` restores exactly the locked bytes into `/opt/vfi-metrics` when
building the trusted image. A pending lock or changed download fails the build.

```bash
python3 tasks/video-frame-interpolation/pipeline/metric_bundle.py restore \
  --lock tasks/video-frame-interpolation/scorer/metric-lock.json \
  --output work/vfi-metrics-restored
```

The downloader records hashes on first acquisition; it does not pretend to have
independently known upstream full SHA-256 values. Subsequent restoration and grading
require exact hashes. Restore keeps `lock.json` byte-identical to the frozen file.

## Measure and freeze both anchors

1. Finish the existing data preparation and immutable HF publication workflow.
2. Choose and record the permitted starting system and the quality target, with
   immutable revisions. Run each reference through the same `/interpolate` request/export client on the
   final held-out set. Do not copy numbers from a different split or paper.
3. Use the pinned `scorer/Dockerfile.gpu` runtime on one H100, then run:

```bash
PYTHONPATH="$PWD" python tasks/video-frame-interpolation/vfi_evaluate.py \
  --data work/vfi/prepared/private/test.jsonl \
  --predictions-dir work/vfi/baseline-predictions \
  --metric-bundle work/vfi-metrics --device cuda \
  --selection-json work/vfi/selection.json \
  --output work/vfi/baseline-metrics.json \
  --write-calibration work/vfi/zero-reference.json \
  --baseline-id <chosen-model> --baseline-revision <immutable-revision-or-submission-digest>
```

Write `{"method":"sha256-id-v1","per_factor":90,"seed":0}` to the selection file.
The CLI can also compute raw metrics for public exports or corruption controls;
omit the three calibration arguments and use the selection appropriate to that export.
CPU calibration is accepted only by organizer diagnostics; production requires the
pinned CUDA runtime and checks the H100 hardware. No training or
candidate code runs in this command. Keep sufficient free disk for all prediction
PNGs as well as the prepared data and metric weights. Both report destinations
must be new files.

Repeat for the target's saved predictions, writing `target-reference.json` with
its own model provenance. Each `--write-calibration` output is a raw reference
receipt; it cannot enable production grading alone. Combine the receipts:

```bash
PYTHONPATH="$PWD/tasks/video-frame-interpolation:$PWD" python \
  tasks/video-frame-interpolation/pipeline/compose_calibration.py \
  --zero work/vfi/zero-reference.json --one work/vfi/target-reference.json \
  --output work/vfi/calibration.json
```

The composer requires matching data, weights, metric code and runtime, and a
strictly better target in every component. Copy the generated file into the seed
as `calibration.json`; pin its exact hash and the new scoring protocol. Production
rejects single-anchor or old-protocol calibration even if its file hash is supplied.

Calibration binds both measured anchors to the complete JSONL **and all referenced
PNG bytes**, the metric bundle, the metric implementation source, CUDA runtime,
checkpoint provenance and prediction manifest. Changing any bound input requires
remeasurement. Freeze these values in `task.yaml` under `verifier.config`:

- `scoring_protocol`: `vfi-lpips-flolpips-two-anchor-v2`.
- `evaluation_sha256`: the report's complete-media evaluation digest (not merely JSONL SHA).
- `metric_bundle_sha256`: SHA-256 of the exact metric `lock.json` bytes.
- `calibration_sha256`: SHA-256 of the generated `calibration.json` bytes.
- `minimum_generated_frames_per_second`: the chosen generation/export throughput floor, now 1.0.

The one-frame-per-second requirement tightens generation to **2,250 seconds
(37 minutes 30 seconds)** for the 270-interval evaluation. This is an average over
all generated frames, not a one-second deadline for each HTTP request. The generic
generation ceiling remains 3600 seconds, each request allows at most 300 seconds,
and the H100 metric worker allows 3600 seconds, with 1800 seconds of separate
resource lifetime for transfer, extraction and teardown. The actual generation deadline is
`min(3600, total_generated_frames / minimum_generated_frames_per_second)`.
It includes input reads, request building, transport, parsing and raw RGB export;
it excludes startup and quality computation. The floor is published in the candidate
contract. Baseline and media/weight pins are activated for the admission pilot. The outer task timeout allows the two
phases, with the repository's usual extra review/setup allowance.

The earlier CPU scorer reserved eight cores for its eight PyTorch threads. A 720p probe measured
2.32 seconds per adjacent pair at eight threads, 3.85 at four, 6.69 at two and 11.17
at one; lowering threads did not improve this workload. Optical flow accounted for
about 93% of CPU time. Organizer runs also distribute intervals across four CPU
workers; these remain available for historical CPU diagnostics.

A separate one-H100 probe took 0.092 seconds per pair for metric computation and
2.33 seconds for one complete 8x interval including file I/O. Its LPIPS/FloLPIPS
interval values differed from the CPU result by less than 0.000000002. This is a
small capacity/parity probe, not a full CUDA calibration. Its runtime was torch
2.5.1+cu118, compared with CPU torch 2.8.0. Production now uses a separate
trusted H100 phase after candidate teardown. Only one GPU is active at a time,
and calibration must match the selected CUDA runtime.
Reports are in `work/vfi/cpu-profile.json` and `work/vfi/gpu-metric-profile.json`.

A full 270-interval CUDA diagnostic subsequently completed in 851.23 seconds for
metrics, or 976.13 seconds including verification/setup, after CPU orchestration
failed and was stopped. Cold volume reads materially affected this run: reading
and hashing one interval's 43 MB of predictions took 7.61 seconds, versus 0.056
seconds from cache. Eight parallel readers prefetched the final 61 intervals
(2.05 GB) in 58.55 seconds while scoring continued. This prefetch was introduced
mid-pass, so the result does not establish a clean end-to-end speedup over CPU.
Production now stages all selected media, predictions and frozen weights onto
ephemeral local disk in one uncompressed archive before scoring. Packaging,
transfer, extraction and teardown are included in `scoring_seconds`;
`metric_seconds` covers the metric pass. It never reads individual frames from
a network volume. A full performance measurement of this transport is pending. The full report is `work/vfi/gimm-r-p-270-run/gpu-metrics.json`;
this remains a CUDA diagnostic, not production calibration.

The separate production H100 worker has also passed a real Harbor/Modal smoke
test with the locked pretrained weights and three synthetic 32x32 intervals
(one per factor). It verified the pinned runtime, transferred and extracted the
input archive, computed metrics offline, downloaded its report and destroyed the
GPU resource. This checks orchestration, not benchmark quality or full-set speed.
The report is `work/vfi/h100-smoke/result.json`; its synthetic normalization
fixture must never be used as the benchmark calibration.

The 270-interval production CUDA calibration uses the permitted starting system
and measured quality target; see [the reference history](pipeline/BASELINE.md). Exact-reference
controls pass for all factors. Before release, run upstream parity comparisons and static,
endpoint-copy, linear-blend, shuffled, shifted, blurred and flickering controls.
Check finiteness, reproducibility, temporal pairing and plausible rankings; retain
raw reports. Current unit tests use deterministic lightweight feature/flow fixtures
for corruption controls and randomly initialized topology tests, explicitly **not**
measured pretrained quality or baseline calibration. These pretrained release comparisons remain pending; earlier GPU timing probes
and baseline reports do not substitute for them.

Prescreen checks calibration, data and weights before a paid architecture review or
candidate GPU launch. Normal grading collects outputs, destroys the candidate
server, then provisions the separate offline H100 metric worker. The parent
verifier stays on CPU. Only selected media, exported pixels, frozen weights and
enumerated organizer modules enter the worker; no submission code, credentials
or candidate volumes are copied. Results return through file download, avoiding
offline result-blob uploads. The parent rechecks report provenance and coverage. An architecture review
receipt is checked against the exact submission snapshot. There is no additional
trusted-data-audit subsystem.

## Training-data restriction

The candidate contract excludes task-time training, fine-tuning, calibration and
model selection on **any public validation/test split**, including mirrors, renamed
copies, extracted frames and derivatives. That covers SNU-FILM, SNU-FILM-arb,
SNU-FILM-entire and the corresponding GoPro/YouTube test footage. The trusted review
policy names these sources; the candidate contract does not reveal the held-out
source. Public training data, synthetic data and this task's shipped examples are
allowed. Incidental exposure in an existing public checkpoint's upstream training
is not itself disqualifying.

Review examines available manifests, loaders, download commands, training/tuning
code and logs. When a recorded trajectory is available, the policy asks the reviewer
to inspect acquisition, training and model-selection steps. Mere name mentions or
reading protocol documentation are not evidence of prohibited training. The shared
harness does not require a trajectory to exist or be fully read, and the API reviewer
backend does not receive one. Review is evidence-based, not a deterministic proof
of training provenance; its trajectory status is retained in the grading result.
