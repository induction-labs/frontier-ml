# Catalog build

`amt_catalog.py` imports the supplied XLSX and companion ZIP, checks every MIDI against both, and uses
`amt_render.render_audio` for every clip. `amt_render.py::prepare` uses the same function on Modal CPU workers.
Neither entry point generates procedural music. The 200 catalog selections contain 281 MIDI files; every
file contributes one 30-second excerpt. Composer and related-work groups cannot cross the split.
The current split has 10 public validation clips and 271 test clips. `VALIDATION_SELECTIONS` specifies ten
catalog selections by eight composers, covering each scored class in at least two clips and drums in four.
The styles include ragtime, jazz, rock, electronic/game music, a recorder country dance and historical/
contemporary ensembles, based on catalog descriptions and titles rather than listening-verified labels.
`reduce_validation` reconstructs the first published 10/271 split; `diversify_validation` then swaps eight
clips in each direction. These steps preserve all excerpts, soundfonts, mixes and labels. The 44 currently
held-out clips that were previously public retain their original public IDs in the manifest.

Excerpts are uniformly random over eligible millisecond start positions throughout the file. The sampler
maximizes pitched instrument classes first, then distinct GM programs, counting a program only when at least
100 ms of one note lies in the window. Sustained notes count. Selection is independent of note density and
is not limited to note-onset timestamps. When all programs cannot fit within 30 seconds, the chosen maximum
coverage and missing programs/classes are recorded in `excerpt_sampling`. This affects two of the 281 files;
the other 279 excerpts contain every source program.

Rebuild locally, using the source snapshot and soundfonts beside the payload:

```sh
uv run --with mido==1.3.3 --with pretty-midi==0.2.10 --with setuptools==75.8.0 \
  --with scipy==1.16.2 --with soundfile==0.13.1 python \
  tasks/music-transcription/pipeline/amt_catalog.py \
  --catalog data/music-transcription-sources/niche-midi-catalog.xlsx \
  --midi-zip data/music-transcription-sources/niche-midi-files.zip \
  --seed-file data/music-transcription.private-seed \
  --soundfont-dir data/music-transcription-sources/soundfonts \
  --out data/music-transcription --validation-count 10 --workers 4
```

Install FluidSynth before running. The manifest records its version and Python package versions; use the same
versions/platform for waveform reproduction. Build into an empty output folder. Source timing, labels, split assignment and mix choices are
deterministic from the frozen inputs and private seed. The seed and source directory are outside the upload.

Required soundfonts (hashes are enforced in `amt_render.py`):

| Name | File | Original asset |
| --- | --- | --- |
| fluidr3 | FluidR3_GM.sf2 | Debian `fluid-soundfont-gm_3.1-5.3_all.deb` |
| musescore | MuseScore_General.sf2 | `https://ftp.osuosl.org/pub/musescore/soundfont/MuseScore_General/MuseScore_General.sf2` |
| generaluser | GeneralUser-GS.sf2 | `mrbumpy409/GeneralUser-GS@684543d5e5efaef08d02be50dcda8d552478fa60` |
| timgm6mb | TimGM6mb.sf2 | Debian `timgm6mb-soundfont_1.3-5_all.deb` |

The exact zero-based pitched-program mapping is `amt_catalog.PROGRAMS`, checked against Coverage!A8:E16.
Programs outside it fail; there is no nearest-family remapping. MIDI port and channel determine program state
at attack. Nonstandard percussion kits 2 and 27 use kit 0; kit 48 remains orchestral percussion. Every used
bank/program pair must exist in the chosen soundfont. Default-program parts remain actual piano and are audited.

Rendering normalizes playback controllers and pitch bends, preserves note events and program assignments,
rearticulates notes crossing the crop start, and deduplicates exact class/pitch/onset unisons in labels only.
It does not apply the historical composer's 60 ms repetition filter or playable-pitch clamps to source music.
Credits, original hashes, program audits and transformations live in `manifest.json`; private rows additionally
retain track mappings and mixes. Public rows contain only neutral IDs, audio paths, durations and labels.

`manifest.json` is the single source for attribution: `source.provenance` records each clip's creator,
source links, license, supplied attribution, original MIDI member/hash and `crop_start_s`. Every excerpt
covers a 30-second window; its end is `crop_start_s + 30`. The manifest stays at the dataset root, outside
the agent's `public/` examples, and the Hugging Face dataset README links directly to it.

Verify with:

```sh
uv run --with mido==1.3.3 pytest -q tests/test_music_transcription.py
uv run frontier check music-transcription
```

The catalog payload is published in `inductionlabs/frontier-ml-tasks` at revision
`925401fb8425ec9da5d198fa975b4ced6563c706`, pinned in `task.yaml`. It contains 285 data/manifest files.
That revision (2026-09-30) differs from `62de91975b37b4a45f0e8fc668644d9cdd93e08a` only in `public/SCHEMA.md`, whose last
sentence now says the examples may be used however the agent wants (matching the contract and the review
policy), and in the manifest's `files` record for that file; audio, labels and `private/` are byte-identical.
The Hugging Face dataset README links directly to the manifest for credits. Audio and labels are unchanged
from the first catalog publication, `7531cc031caa35054c101fb601bec32584cf72be`; the manifest and row files
reflect the new split and IDs. Both splits now contain all four soundfonts, preserving each clip's original rendering.
For a future rebuild, publish and pin the changed payload before a remote run; do not reuse this
revision with a different evaluation digest. Previous procedural-data scores and onset-lag measurements
do not apply to the new repertoire and expanded programs.

## YourMT3+ baseline

The September 29, 2026 private-set run measured 0.552948 macro onset F1 and 365.81x real-time direct batched
inference on one H100, on the current 271-clip split with the pinned checkpoint and the existing output mapping.
It is the task's normalization floor (`score_normalization` in `task.yaml`). See the
[task card](../../../docs/tasks/music-transcription.md#calibration-and-limitations) for per-class scores and
timing limitations. This does not constitute a formal HTTP serving, submission size or architecture gate run.

With GPU execution authorized and Modal credentials configured, reproduce with:

```sh
uv run modal run --detach tasks/music-transcription/pipeline/amt_yourmt3.py \
  --split test \
  --out data/music-transcription-baselines/yourmt3-niche-catalog-v4-private-<date>.json \
  --batch 1500 --compact --fast-moe --workers 8 --latency-clips 20
```

The wrapper verifies evaluation-row and audio hashes before GPU execution, saves raw predictions before
scoring, and records the dataset, manifest and baseline-script digests in the summary. Keep output artifacts
outside `data/music-transcription/` so predictions are not included in a dataset upload. Choose a new output
filename for a subsequent run to preserve the original measurements.
