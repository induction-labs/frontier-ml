# Video frame interpolation protocol, version 2

This draft implements independent two-endpoint inference and prediction export.
The quality scorer is implemented; measured normalization, metric weight pins and
the performance threshold remain unconfigured. See [METRICS.md](METRICS.md).
Version 2 replaces the earlier whole-sequence draft; version 1 requests are rejected.

## Server

`serve.sh` starts an HTTP server using `HOST` and `PORT`; `GET /health` returns HTTP
200 when ready. Package active weights and dependencies for offline execution on
one H100, 8 CPUs, and 80 GiB RAM, with a 300-second startup deadline. The generated
`/task/CONTRACT.md` specifies the image and submission details.

`POST /interpolate` accepts one independent interval:

```json
{
  "schema_version": 2,
  "width": 640,
  "height": 360,
  "frame_count": 9,
  "input_frames": [
    {"index": 0, "png": "<base64 PNG>"},
    {"index": 8, "png": "<base64 PNG>"}
  ],
  "target_indices": [1, 2, 3, 4, 5, 6, 7]
}
```

The interpolation factor is `frame_count - 1`: 4, 8, or 16. The two supplied
endpoints have normalized times **0 and 1**; target `i` occurs at `i/(frame_count-1)`.
Thus 8× interpolation generates seven frames at 1/8 through 7/8. No capture or
playback frame rate is asserted. There is no `fps` field in version 2.

Exactly two input frames, at indices 0 and `frame_count-1`, are required. Targets
are every integer index between them, strictly increasing. Intermediate anchors,
extrapolation tails, duplicate indices, and omitted targets are rejected. Images
must retain the requested dimensions. Internal model padding is allowed if removed
before returning the result. Requests contain no sample ID, source metadata, or
reference images.

Each interval uses only its own endpoints as visual context. Do not carry images,
predictions, image features, or model updates between evaluation requests. Fixed
weights, compiled kernels, and image-independent caches may persist. Recursive
interpolation and reuse of features **within the current interval** are allowed.
The submission review checks this rule; transport validation alone does not prove
that a server is independent of previous requests.

The default client sends `Accept: application/vnd.frontier.rgb8`. Return HTTP 200
with that content type and a binary body containing the requested frames in order:
each frame is `H*W*3` unsigned bytes, row-major RGB, with no header, compression,
padding or base64. The exact body length is `len(target_indices)*H*W*3`. This avoids
image encoding and decoding while preserving the same 8-bit pixels.

For legacy clients (`--response-format png`, without that Accept header), return
HTTP 200 and exactly this JSON object, with only requested missing frames:

```json
{
  "frames": [
    {"index": 1, "png": "<base64 PNG>"},
    {"index": 2, "png": "<base64 PNG>"},
    {"index": 3, "png": "<base64 PNG>"},
    {"index": 4, "png": "<base64 PNG>"},
    {"index": 5, "png": "<base64 PNG>"},
    {"index": 6, "png": "<base64 PNG>"},
    {"index": 7, "png": "<base64 PNG>"}
  ]
}
```

Response URLs, paths, extra fields, missing frames, and reordered frames are
rejected in PNG mode. PNGs use plain base64 without a data-URL prefix. PNG inputs and outputs
must be non-interlaced 8-bit RGB PNGs of exactly the specified dimensions, without
transparency or animation. The validator checks chunk CRCs, compressed pixel
lengths, and scanline filters. Quality metrics use decoded native-resolution RGB. LPIPS requires both dimensions
at least 32; smaller images remain useful only for unscored protocol diagnostics.

Bounds: 5, 9, or 17 total frames; dimensions 1–4096;
`width * height * frame_count * 3 <= 512 MiB`; 32 MiB per PNG; 768 MiB per JSON body.
These are transport bounds, not calibrated speed targets. Oversized intervals are
rejected without cropping or dropping frames. Requests run sequentially.

## Local samples and diagnostics

The client reads JSONL. Each row uses the request fields, adds a unique string `id`,
and replaces each input's `png` with a `path` relative to the JSONL directory:

```json
{"id":"interval-000001","schema_version":2,"width":640,"height":360,"frame_count":5,"input_frames":[{"index":0,"path":"media/000000.png"},{"index":4,"path":"media/000004.png"}],"target_indices":[1,2,3],"reference_frames":[{"index":1,"path":"media/000001.png"},{"index":2,"path":"media/000002.png"},{"index":3,"path":"media/000003.png"}]}
```

Optional `reference_frames` must match all target indices in order. The interface
check validates their metadata but never reads or sends those images. Only supplied
endpoints are sent; IDs and other local metadata are stripped. Paths cannot escape
the data root, including through symlinks.

Run `validate.py --server-url <url> --data <samples.jsonl> --interface-only
--predictions-dir <new-directory>`. Raw RGB output is the default; add
`--response-format png` for the older PNG JSON interface. The predictions directory must not exist.
The default per-request deadline is 300 seconds, and the whole-pass deadline is
3600 seconds; diagnostic flags can adjust both. A failed request fails the check;
there is no retry or partial score. Without `--interface-only`, the client refuses
to run; quality computation is provided by the separate trusted `vfi_evaluate.py`
CLI. The official verifier refuses to run until frozen metric weights, both measured score anchors,
calibration, data identity and a performance floor are supplied.

Exports contain `predictions.jsonl`, ordinal interval directories with every frame
(including the exact supplied endpoints), and `run.json`. Each prediction record
contains dimensions, frame count, interpolation factor, indices, generated flags,
SHA-256 hashes, relative output paths, and request time. Supplied endpoints remain
their original PNG files; generated raw frames use `.rgb` files of exactly `H*W*3`
bytes. Both formats decode to the same RGB uint8 arrays for scoring. Normalized timestamps
follow directly from the indices. `run.json` must say `complete` before consuming a
run; failures leave an explicit failed status and may leave partial outputs.

Reports have no quality score. Sample counts, generated-frame counts, and HTTP
request times are grouped by interpolation factor. Timing includes transport and
is diagnostic, not a calibrated performance measurement. The trusted scorer computes
frame quality and motion-weighted perceptual error **within each interval**; unrelated
rows must never be concatenated into a single temporal sequence.
