# Process SNU-FILM-arb and publish the task data

This is the handoff for a Codex session on a machine with network access and ample
storage. Run from the `frontier-ml` repository root. This is CPU data preparation;
no GPU, pretrained weights, training run, or paid evaluation is needed.

## Paste this into the other Codex session

> Prepare SNU-FILM-arb for the `video-frame-interpolation` task and upload the
> verified payload to the Hugging Face dataset configured in its `task.yaml`
> (currently `inductionlabs/frontier-ml-tasks`, prefix `video-frame-interpolation`).
> Follow `tasks/video-frame-interpolation/pipeline/PROCESS_AND_UPLOAD.md` end to end.
> Use all RGB frames, the original medium/hard/extreme three-column lists, native
> resolution, two endpoints per interval, and all 3/7/15 targets at 4×/8×/16×.
> Verify original-recording groups and use the implemented 20% public-group split,
> seed 0. Keep all intervals and factors from a recording together. Use the existing
> pipeline, fix demonstrated preparation/packaging bugs if necessary, and rerun its
> tests. Upload only this task's folder with `frontier upload`; retain the returned
> immutable commit in `task.yaml`, set the actual held-out sample count, then fetch
> the pinned payload into a fresh directory and verify it. Report counts by factor,
> provenance, hashes, split limitations, the HF commit/link, and modified code files.
> Preserve the implemented metrics; leave measured calibration and speed thresholds unset. Do not launch
> evaluations, add X-TEST, change other tasks, or push GitHub changes as part of this
> handoff. The upload is part of this request; do not ask again merely because it
> writes to Hugging Face. Never fabricate missing frames, source identities, license
> clearance, or passing audits.

## 1. Obtain the current implementation

The task was added in a local working tree. A checkout of `main` alone may not
contain it. Transfer the current task, tests, and documentation before starting:

- `tasks/video-frame-interpolation/`
- `tests/test_video_frame_interpolation.py`
- `tests/test_video_frame_interpolation_data.py`
- `tests/test_video_frame_interpolation_scoring.py`
- `docs/tasks/video-frame-interpolation.md`
- The `/interpolate` fixture addition in `frontier_ml/smoke_server.py` and the
  accompanying task entries in `README.md` and `docs/DATA.md`.

If using the supplied handoff archive, unpack it outside the checkout and follow
its `INSTALL.md`. It contains source files and a patch, without datasets or tokens.
Merge with existing work normally; do not reset the destination checkout or blindly
overwrite another session's edits.

Check the implementation before any large download:

```bash
git status --short
uv sync --frozen --extra data --extra test
uv run --no-sync python tasks/video-frame-interpolation/vfi_data.py --help
uv run --no-sync pytest -q \
  tests/test_video_frame_interpolation.py \
  tests/test_video_frame_interpolation_data.py \
  tests/test_agent_surface.py tests/test_build.py
```

Python 3.12+ and the repository's locked dependencies are required. The organizer
uses Pillow; the public inference client remains stdlib-only. A missing-real-data
skip is expected before preparation. On a normal networked machine, the local HTTP
smoke test should run; restricted sandboxes may skip listening sockets.

Choose a work directory on the large disk. Paths below are examples; keep it
outside the raw dataset tree and keep payload output separate from input sources.
Use a persistent terminal/session for long transfers.

```bash
export VFI_WORK="$PWD/work/vfi-snu-arb"
mkdir -p "$VFI_WORK"
df -h "$VFI_WORK"
```

Budget space for the archive, extraction, prepared PNGs, and an independent HF
round-trip copy. The true archive size is not pinned yet. The preparer reports a
conservative output-size estimate after inspecting the source frames.

## 2. Fetch the protocol and all RGB frames

The benchmark definition follows the [GIMM-VFI evaluator](https://github.com/GSeanCDAT/GIMM-VFI/blob/main/src/SNU_FILM_arb.py).
The [CAIN project page](https://myungsub.github.io/CAIN/) distinguishes the all-RGB
archive from the selected-triplet archive. This task needs **all RGB frames**.

```bash
uv run --no-sync python tasks/video-frame-interpolation/vfi_data.py fetch-protocol \
  --output "$VFI_WORK/protocol"

uv run --no-sync python tasks/video-frame-interpolation/vfi_data.py fetch \
  --max-gib 64 --output "$VFI_WORK/snufilm-test.zip"
```

The 64-GiB argument is a maximum download size, not an expected size. The downloader
also reserves disk headroom, hashes the received bytes, rejects HTML/truncation,
and writes a `.source.json` receipt. The reference snapshot resolves `main` to a
full Git commit and downloads two scripts without executing them. Inspect its
loader to confirm the list names, endpoint selection, and intermediate-frame rule.

Retain the archive receipt and `protocol/protocol.json`. Repeat downloads should
use `fetch --sha256 <receipt-sha256>` and
`fetch-protocol --revision <protocol-commit>`, with new output paths. These commands
refuse to overwrite existing or partial outputs. After an interrupted fetch, inspect
and move aside the incomplete artifact before retrying; the downloader does not resume.

If Dropbox changes the URL, follow the **all RGB frames** link from the CAIN page
and use `fetch --url '<direct HTTPS archive URL>' ...`. Preserve the actual resolved
URL and receipt. Do not replace the benchmark with a third-party subset to avoid a
failed download.

Inspect the ZIP layout and its expanded size, then extract to a new directory:

```bash
uv run --no-sync python - <<'PY'
import os, shutil, stat, zipfile
from pathlib import Path, PurePosixPath
work = Path(os.environ["VFI_WORK"])
target = work / "extracted"
if target.exists():
    raise SystemExit(f"Use a new extraction directory: {target}")
with zipfile.ZipFile(work / "snufilm-test.zip") as archive:
    entries = archive.infolist()
    expanded = sum(entry.file_size for entry in entries)
    print({"entries": len(entries), "expanded_bytes": expanded,
           "top_level": sorted({entry.filename.split('/')[0] for entry in entries})})
    if shutil.disk_usage(work).free < expanded + 512 * 1024**2:
        raise SystemExit("Insufficient extraction space; use a larger volume")
    for entry in entries:
        path = PurePosixPath(entry.filename)
        if path.is_absolute() or '..' in path.parts or '\\' in entry.filename:
            raise SystemExit(f"Unsafe ZIP path: {entry.filename}")
        if stat.S_ISLNK(entry.external_attr >> 16):
            raise SystemExit(f"Unexpected archive symlink: {entry.filename}")
    target.mkdir()
    archive.extractall(target)
PY
```

Locate the actual directory containing `test/GOPRO_test` and `test/YouTube_test`,
and the directory containing all three original evaluation lists:

```bash
rg --files "$VFI_WORK/extracted" -g 'test-medium.txt' -g 'test-hard.txt' -g 'test-extreme.txt'
```

Set these to the discovered locations; the following assumes no extra archive wrapper:

```bash
export VFI_RAW="$VFI_WORK/extracted"
export VFI_LISTS="$VFI_RAW"
```

The [upstream layout](https://github.com/GSeanCDAT/GIMM-VFI#dataset-preparation)
places `test/` and the lists under `SNU-FILM/`. If extraction creates such a wrapper,
set `VFI_RAW` to that wrapper. `VFI_LISTS` may instead be an `eval_modes` directory.
If the archive does not include lists, obtain the original lists from the authors'
SNU-FILM distribution and retain their source URL/hash. The pipeline checks their
bytes into the payload; it does not generate new test lists or download them separately.

Required files and semantics:

| List | Rows | Endpoint gap | Targets per request |
| --- | ---: | ---: | ---: |
| `test-medium.txt` | 310 | 4 frames | 3 |
| `test-hard.txt` | 310 | 8 frames | 7 |
| `test-extreme.txt` | 310 | 16 frames | 15 |

Each row has three paths. Only the first and third images are inputs; every integer
frame between them is a target, including the original midpoint. Use the original
three-column lists, not reformatted `test-arb-*` lists. The sample grid is normalized
to `[0,1]`; source filenames retain their original numbering until preparation.

## 3. Inspect sources and establish recording groups

```bash
uv run --no-sync python tasks/video-frame-interpolation/vfi_data.py inspect \
  --snu-root "$VFI_RAW" --lists-dir "$VFI_LISTS" \
  --output "$VFI_WORK/inventory.json"

uv run --no-sync python - <<'PY'
import json, os
from pathlib import Path
work = Path(os.environ["VFI_WORK"])
inventory = json.loads((work / "inventory.json").read_text())
with (work / "groups.json").open("x") as stream:
    json.dump(inventory["group_map_template"], stream, indent=2, sort_keys=True)
    stream.write("\n")
print({key: inventory[key] for key in
       ("samples", "unique_frames", "by_factor", "estimated_output_upper_bytes")})
PY
```

The template is a starting point, not evidence that differently named folders are
independent videos. Review the source recording identities before preparing. Group
all clips from one original recording, including any renamed or re-encoded copies.
The pipeline conservatively groups `GOPR<number>` prefixes and prevents splitting
those subclips. Check YouTube identities against available source provenance; if
identity is unresolved, record that fact rather than calling it verified. Inspect
representative frames/near-duplicate candidates as needed; a manual inspection of
every frame is not a requirement.

Keep evidence and any unresolved grouping/rights findings in
`$VFI_WORK/processing-notes.json`. This is an operator note, not an extra trusted
model audit. Media redistribution rights are currently recorded as unresolved;
a code license is not a dataset-media license. Preserve that status unless actual
source terms establish otherwise. The current verifier deliberately reports
`publication_ready: false` and `split_audit.passed: false`; do not change those values
by hand to make the upload appear like a completed benchmark release.

The preparation default is 20% of **recording groups** for public validation, seed 0,
with the rest held out. All factors and intervals from a group stay together.
Actual row proportions depend on group sizes. Do not tune the seed using model scores.

## 4. Prepare and verify the payload

```bash
uv run --no-sync python tasks/video-frame-interpolation/vfi_data.py prepare \
  --snu-root "$VFI_RAW" --lists-dir "$VFI_LISTS" \
  --protocol-snapshot "$VFI_WORK/protocol/protocol.json" \
  --group-map "$VFI_WORK/groups.json" --validation-fraction 0.2 --seed 0 \
  --output "$VFI_WORK/prepared"

uv run --no-sync python tasks/video-frame-interpolation/vfi_data.py verify \
  --output "$VFI_WORK/prepared"
```

Preparation uses native dimensions and RGB values, strips metadata, and writes
lossless PNGs. It copies only frames referenced by the official intervals, storing
a source frame once across overlapping intervals/factors. It verifies every missing
frame exists, both splits contain every factor, and no declared recording group,
exact RGB frame, or endpoint fingerprint overlaps between splits.

Expected result: 930 total intervals and 7,750 target predictions across both splits
(`310 * (3 + 7 + 15)`). Unique stored frames can be fewer than total frame appearances.
The output split counts and their per-factor counts are authoritative. The validation
and test subsets together retain all official rows; private-only results will be on
a custom subset, requiring baselines measured on that same split.

```text
prepared/
  manifest.json
  public/validation.jsonl
  public/media/sequence-.../*.png
  private/test.jsonl
  private/media/sequence-.../*.png
  private/provenance/groups.json
  private/provenance/test-medium.txt
  private/provenance/test-hard.txt
  private/provenance/test-extreme.txt
```

`private/provenance/` is intentional: `frontier upload` includes `public/**`,
`private/**`, `artifacts/**`, and `manifest.json`, but skips other top-level folders.
Older drafts used top-level `provenance/`; use the updated implementation and
regenerate that draft payload. Do not drop verification dependencies from its manifest.
The protocol commit/file hashes and source-frame mappings are in `manifest.json`;
retain the actual reference snapshot and download receipts in the work archive too.

If preparation fails, fix the demonstrated cause and rerun into a new output directory:

- **Missing intermediates:** check for a selected-triplet download or wrong raw root.
- **List count/spacing mismatch:** inspect the original file; do not weaken checks or
  synthesize intervals to reach 310 rows.
- **Overlap across splits:** identify the matching frames/recordings and merge related
  groups, then reprepare. Do not discard official rows or silence the overlap check.
- **Unresolved identities or ubiquitous duplicate frames:** document the actual
  obstruction if a defensible nonempty split cannot be constructed; do not invent one.
- **Disk limit:** move to a larger volume; do not downscale, shorten, or JPEG-encode data.
- **Published-source or semantic audit still pending:** keep that limitation explicit.
  It is distinct from passing byte/schedule checks and from having a configured scorer.

Rerun the focused tests from step 1 after any code change. Optionally run the CPU
HTTP interface fixture from [README.md](README.md#local-interface-check). Its copied
endpoint predictions have no quality score and are not an ML baseline.

## 5. Set the actual sample count and inspect the upload

Keep the existing `data.repo` and `data.path` unless the user specifies another
repository. Current destination:

```text
https://huggingface.co/datasets/inductionlabs/frontier-ml-tasks
video-frame-interpolation/
```

This uploader uses a **public dataset**. `private/` describes isolation from the
agent workspace, not Hugging Face access controls; its files and provenance will
also be downloadable from HF. Source names belong in the manifest/private provenance,
not in public examples or the task's candidate-facing instructions.

Set `data.expected_samples` to the actual held-out row count, preserving YAML prose:

```bash
uv run --no-sync python - <<'PY'
import json, os, re
from pathlib import Path
payload = Path(os.environ["VFI_WORK"]) / "prepared"
manifest = json.loads((payload / "manifest.json").read_text())
count = len((payload / "private/test.jsonl").read_text().splitlines())
assert count == manifest["splits"]["heldout"] > 0
path = Path("tasks/video-frame-interpolation/task.yaml")
text, replacements = re.subn(
    r"(\ndata:\n(?:  .*\n|#.*\n)*?  expected_samples: )\S+",
    lambda match: match[1] + str(count), path.read_text(), count=1)
assert replacements == 1, "Expected exactly one data.expected_samples field"
path.write_text(text)
print({"expected_samples": count, "by_factor": manifest["by_factor"]})
PY
```

The task remains a draft: preserve the selected scoring protocol, but leave the
unmeasured baseline, metric weight pins and performance threshold unset. The
[separate metric calibration workflow](../METRICS.md) is implemented; this data-only
handoff does not require acquiring model weights or running a baseline. Update operator documentation with observed sizes, source hashes,
counts, and remaining limitations after processing. Do not put those source details
into candidate-facing `task.yaml` prose. The upload command itself sets `data.revision`.

List exactly what the repo uploader will send, and check completeness:

```bash
uv run --no-sync python - <<'PY'
import json, os
from pathlib import Path
from frontier_ml.cli import local_files
from frontier_ml.taskspec import load_task
work = Path(os.environ["VFI_WORK"])
payload = work / "prepared"
manifest = json.loads((payload / "manifest.json").read_text())
files = local_files(payload)
assert set(files) == set(manifest["files"]) | {"manifest.json"}
report = {"destination": load_task("video-frame-interpolation").spec["data"],
          "file_count": len(files), "bytes": sum(p.stat().st_size for p in files.values()),
          "files": sorted(files), "splits": manifest["splits"], "by_factor": manifest["by_factor"]}
(work / "upload-inventory.json").write_text(json.dumps(report, indent=2) + "\n")
print({k: v for k, v in report.items() if k != "files"})
PY
```

Keep the output directory clean: extra files are verification errors. Processing
notes, logs, raw downloads, credentials, code snapshots, and predictions stay outside
`prepared/`. No files from other task folders are part of this upload.

## 6. Authenticate and upload

Use an existing `HF_TOKEN` with write access or the cached Hugging Face login. If
neither is present, the machine's operator can run the interactive
[HF login command](https://huggingface.co/docs/huggingface_hub/guides/cli#hf-auth-login):

```bash
uv run --no-sync hf auth login
uv run --no-sync hf auth whoami
```

Never paste tokens into the guide, source files, terminal logs, or chat. Missing
credentials need to be supplied locally. Do not create a different repository or
change visibility to work around denied access to the requested destination.

```bash
uv run --no-sync frontier upload video-frame-interpolation \
  --from "$VFI_WORK/prepared"
```

This command uploads the selected task prefix and writes the returned immutable
HF commit into `tasks/video-frame-interpolation/task.yaml`. It replaces the prefix's
`public/`, `private/`, and `artifacts/` contents with the local copies, so verify the
destination printed in step 5 first. Other task prefixes are outside its scope.
Do not upload the whole `work/` directory or replace the HF repository root.

If a network failure happens, inspect whether `task.yaml` was repinned and whether
a remote commit exists before retrying. Do not use `frontier run` to upload: this
task is not calibrated, and that command is for starting a paid trial. An HF upload is
data publication, not evidence of benchmark admission, media-license clearance,
a completed semantic audit, or scoring readiness.

## 7. Download the pinned version and verify it independently

Use a **fresh** output path; `frontier fetch` replaces an existing mirror there.
This checks the exact commit in `task.yaml`, rather than the moving HF `main` branch.

```bash
uv run --no-sync frontier fetch video-frame-interpolation \
  --out "$VFI_WORK/hf-roundtrip"

uv run --no-sync python tasks/video-frame-interpolation/vfi_data.py verify \
  --output "$VFI_WORK/hf-roundtrip/video-frame-interpolation"

uv run --no-sync python - <<'PY'
import json, os
from pathlib import Path
from frontier_ml.cli import data_changes
from frontier_ml.taskspec import load_task
work = Path(os.environ["VFI_WORK"])
task = load_task("video-frame-interpolation")
assert data_changes(task.spec, work / "prepared") == [], "HF bytes differ from prepared payload"
original = work / "prepared/manifest.json"
restored = work / "hf-roundtrip/video-frame-interpolation/manifest.json"
assert original.read_bytes() == restored.read_bytes(), "Manifest changed in transit"
revision = task.spec["data"]["revision"]
assert len(revision) == 40 and revision != "unpublished"
summary = {"repo": task.spec["data"]["repo"], "revision": revision,
           "path": task.spec["data"]["path"], "expected_samples": task.spec["data"]["expected_samples"],
           "url": f"https://huggingface.co/datasets/{task.spec['data']['repo']}/tree/{revision}/{task.spec['data']['path']}",
           "verification": "fresh pinned download verified; source and HF file digests match",
           "scoring": "not_configured"}
(work / "publication-result.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY

uv run --no-sync frontier build video-frame-interpolation --out "$VFI_WORK/rendered"
uv run --no-sync frontier check video-frame-interpolation
```

`frontier check` may report missing model/Modal/Spin/reviewer credentials on a
CPU-only processing machine. Report that result accurately; those credentials are
not needed for data preparation or HF upload. It does not call the task scorer or
certify that the draft can be graded. A successful data pin allows packaging while
the verifier still refuses scored evaluation.

Do not add a symlink to `data/` merely to rerun the conditional real-data unit test:
the explicit `vfi_data.py verify` above checks the chosen path. Avoid overwriting an
existing local `data/video-frame-interpolation` mirror during this handoff.

## 8. Return the completion record

Report the HF URL and full immutable commit; actual public/private counts by factor;
unique PNG count and payload bytes; archive SHA-256 and resolved URL; GIMM protocol
commit; recording grouping method and remaining uncertainties; exact-overlap results;
test, round-trip, build, and check outcomes; and any code fixes made during processing.
Keep `publication-result.json`, `upload-inventory.json`, `processing-notes.json`,
archive receipts, the group evidence, and the protocol snapshot in the work archive.

Return the changed `task.yaml` with its actual `expected_samples` and HF `revision`,
plus documentation/code changes. Data remain out of Git. No Git push is included in
this handoff; the original session needs the updated code or patch to use the same
pinned payload. Scoring and GPU baseline work remain for a later decision.
