# Video interpolation data preparation

For a new machine and a complete Hugging Face handoff, use
[Process and upload SNU-FILM-arb](PROCESS_AND_UPLOAD.md).

The primary benchmark is **SNU-FILM-arb**, using the
[GIMM-VFI reference evaluator](https://github.com/GSeanCDAT/GIMM-VFI/blob/main/src/SNU_FILM_arb.py).
The medium/hard/extreme lists define 4×/8×/16× endpoint intervals, with 3/7/15
missing frames. There are 310 official list rows per factor, 930 intervals total
before the public/private split. X-TEST is deferred as a possible secondary test.

The organizer pipeline is [`../vfi_data.py`](../vfi_data.py); source URLs and rights
notes are in [`sources.json`](sources.json). Real source media have not been
prepared in this workspace. Scoring is implemented; frozen metric weights and measured GPU baselines remain pending.

## Fetch and inspect

Run from the repository root on a networked machine with space for the archive,
extracted originals, prepared PNGs, and predictions. The preparer estimates output
space before decoding. The archive size is not pinned yet; `--max-gib` is a download
cap, not a claim about its actual size.

```bash
uv run python tasks/video-frame-interpolation/vfi_data.py fetch-protocol \
  --output work/vfi/protocol

uv run python tasks/video-frame-interpolation/vfi_data.py fetch \
  --max-gib 64 --output work/vfi/downloads/snufilm-test.zip
```

`fetch-protocol` resolves `main` to an immutable commit, snapshots the two reference
scripts, and records hashes without executing either file. Inspect the snapshot
against the sampling rules below before a release. Repeat with
`--revision <full-commit-sha>` to reuse the same protocol.

The media link is CAIN's **all RGB frames** download. Selected triplets are
insufficient. Extract the frames and retain the original SNU-FILM
`test-medium.txt`, `test-hard.txt`, and `test-extreme.txt` lists used by the reference
evaluator. Pass their containing directory as `--lists-dir` (it may be the dataset
root or an `eval_modes` directory). Do not substitute another project's reformatted
`test-arb-*` lists: this loader expects the original three-column lists.

Expected layout:

```text
work/vfi/snu/test/GOPRO_test/<sequence>/<number>.png
work/vfi/snu/test/YouTube_test/<sequence>/<number>.png
work/vfi/snu/test-medium.txt
work/vfi/snu/test-hard.txt
work/vfi/snu/test-extreme.txt
```

A row contains three paths, with the first and third identifying the endpoints.
The preparer verifies their distance (4, 8, or 16 frames), the original midpoint,
numeric filename padding, native dimensions, and every intermediate PNG. It accepts
`data/SNU-FILM/test/...`, `./data/SNU-FILM/test/...`, or root-relative `test/...` paths.
It rejects partial lists, duplicate endpoint pairs, path traversal, and missing
frames. Frames outside the official intervals are not needed or prepared.

```bash
uv run python tasks/video-frame-interpolation/vfi_data.py inspect \
  --snu-root work/vfi/snu --lists-dir work/vfi/snu \
  --output work/vfi/inventory.json
```

The inventory includes file counts, storage estimates, and `group_map_template`.
Copy that object's contents into `work/vfi/groups.json` and verify its recording
identities against the source provenance. Each key is a source sequence directory;
each value identifies its original recording. Merge subclips from the same video,
including differently named or re-encoded copies. Example shape:

```json
{
  "test/GOPRO_test/GOPR0384_11_00": "recording-a",
  "test/GOPRO_test/GOPR0384_11_05": "recording-a",
  "test/YouTube_test/0000": "recording-b"
}
```

These are illustrative entries, not a complete verified map. An explicit map is
required: folder names alone do not establish original-video independence.
Preparation also prevents splitting known `GOPR<number>` subclips even if the map
assigns them different groups. All intervals and factors of a group stay together.

## Prepare and verify

```bash
uv run python tasks/video-frame-interpolation/vfi_data.py prepare \
  --snu-root work/vfi/snu --lists-dir work/vfi/snu \
  --protocol-snapshot work/vfi/protocol/protocol.json \
  --group-map work/vfi/groups.json --validation-fraction 0.2 --seed 0 \
  --output work/vfi/prepared

uv run python tasks/video-frame-interpolation/vfi_data.py verify \
  --output work/vfi/prepared
```

**Assumed default:** 20% of original recording groups for public validation, the
remainder held out. Selection sorts groups by SHA-256 of `seed + NUL + group`, then
rounds the requested public group count, keeping both partitions nonempty. Row
counts depend on recording sizes; they are reported separately for each factor.
Both partitions must include all three factors. This creates a custom split of
the official examples, so held-out-only results are not directly comparable with
papers that evaluate all 310 rows per factor. Rerun baselines on the frozen split.

Each request has two endpoints and all 3/7/15 targets. Times are normalized to
`[0,1]`; there is no assumed capture or playback FPS. A model may use only the
current interval's visual context. Related benchmark intervals can overlap, so
request ordering does not replace that independence rule. The contract states it
and submission review checks it.

Preparation converts images to RGB as the reference loader does, removes metadata,
and writes lossless PNGs at native dimensions. There is no resize, spatial crop,
EXIF orientation, or video re-encoding. Each selected source frame is stored once
and shared by overlapping intervals within its partition.

Output is staged atomically into a new directory:

```text
prepared/manifest.json          # operator-only source/protocol/frame hashes and split audit
prepared/private/provenance/    # exact source lists and original-recording group map
prepared/public/validation.jsonl
prepared/public/media/sequence-000001/000000.png
prepared/private/test.jsonl
prepared/private/media/sequence-000002/000000.png
```

The manifest records the full protocol commit, source/list/group hashes, Pillow
version, original paths and byte hashes, native RGB hashes, interval mappings,
split seed, counts by factor, and prepared file hashes. Download `.source.json`
receipts record the archive URL, retrieval time, bytes, and SHA-256; retain these
alongside the raw archive. Use `fetch --sha256 <recorded-hash>` for repeat downloads.
No actual archive hash or immutable upstream commit is claimed before a fetch.

Preparation rejects cross-partition recording-group, exact RGB-frame, or endpoint
fingerprint overlap. If found, merge related source groups and prepare again.
Verification rechecks every file, RGB hash, official interval, row, mapping, count,
and split. It reports `publication_ready: false`: source-identity verification,
near-duplicate review, media rights, metric parity checks and calibration remain release work.
It does not manufacture a passing semantic audit from exact hash checks.

Keep preparation in ignored `work/` until the data freeze. Source names and
provenance must stay outside the agent's public payload. The source benchmark is
publicly available; neutral filenames do not make its answers secret. Dataset-media
rights remain unresolved. These commands neither publish data nor run a model.

## Local interface check

The smoke generator makes nine 32×24 moving-square frames, supplying only frames
0 and 8. It is an interface fixture, not benchmark or training data.

```bash
uv run python tasks/video-frame-interpolation/pipeline/vfi_smoke.py \
  --output /tmp/frontier-vfi-fixture
```

In another terminal:

```bash
HOST=127.0.0.1 PORT=8123 uv run python frontier_ml/smoke_server.py
```

Then check the interface and export the complete interval:

```bash
PYTHONPATH=. uv run python tasks/video-frame-interpolation/validate.py \
  --server-url http://127.0.0.1:8123 \
  --data /tmp/frontier-vfi-fixture/samples.jsonl \
  --interface-only --predictions-dir /tmp/frontier-vfi-predictions \
  --output /tmp/frontier-vfi-report.json
```

Use new output directories for each run. To exercise prepared media, replace
`--data` with `work/vfi/prepared/public/validation.jsonl` and use a learned model
server. The report has `diagnostic_score: null` and `official_score: false`, with
counts and request time by interpolation factor. `run.json` must say `complete`
before consuming predictions. See [PROTOCOL.md](../PROTOCOL.md) for schemas and bounds.

The smoke server copies the nearest endpoint; it cannot satisfy the learned-model
requirement. Production builds remain blocked by `revision: unpublished`;
prescreen/preflight reject missing weight pins, measured baseline calibration and
performance settings. See [METRICS.md](../METRICS.md) for the implemented offline
scorer and calibration workflow.
