# Scientific figure to TikZ

`validation.jsonl` contains 50 development examples from the allowed
`JackieLin0123/SciTikZ-230K` dataset, revision
`3061cc331aa6f71cc7430e0480929d0ccddade67`. Each row has `id`, an `image` path
relative to this directory, reference `code`, `category` and `figure_type`.
These development examples may be used for training. The corpus is declared
Apache-2.0 by its publisher; see LICENSE-datasets.txt.

SciTikZ-Bench is forbidden for training, development testing, validation, reward
tuning, retrieval and model selection, including copies/mirrors. Only the trusted
final evaluator uses it. Retain training/data provenance for compliance review.

## Requests and metrics

`/task/validate.py` uses the paper's image reconstruction prompt, temperature 0,
16384 output tokens and concurrency 16. Return a complete document in a closed
`latex` fence. The first fence is scored, with a 64,000-character response limit.

The automatic metrics are compilation/rendering rate, SigLIP SO400M patch14/384,
CLIP ViT-L/14, AlexNet LPIPS, SSIM, DreamSim ensemble, normalized Token Edit
Distance (Extended Edit Distance on TeX tokens), and sentence CrystalBLEU.
Visual means are reported for successful renders and for all examples with the
paper's failure defaults. Code metrics cover all examples, including failures;
truncated open fences are retained for code scoring, as in the released renderer.
No usable code is scored as the empty program. CrystalBLEU ignores the 500 most
frequent 1–4-grams in the full reference corpus for the split being evaluated.

The task adds one scalar score:

`compilation_rate * geometric_mean(SigLIP, CLIP, exp(-LPIPS/.5), SSIM,
                                  1-DreamSim, exp(-TED/.4), CrystalBLEU)`

Visual components use successful-render means; TED and CrystalBLEU use all-output
means. Similarities are clipped to [0,1]. No successful render gives zero. This
aggregate is our addition; the component metrics are from arXiv:2604.06079.

Render with pdfLaTeX (30s), shell escape disabled, first page at 300 DPI. Preserve
standalone page size; otherwise suppress page numbers and pdfcrop with a 10pt
margin. Rendering is bounded to 8192 pixels per side and 32 megapixels. The trusted
renderer has no input raster or external files. Missing scorer dependencies fail
evaluation instead of skipping metrics. Final generation has a 7200s deadline on
one offline H100, separate from rendering and metric computation.

## Public validation

Collect predictions without installing the metric models:

```
python /task/validate.py --server-url http://localhost:8000 \
  --predictions-only --output predictions.json
```

For full metrics in a development compute environment, install the pinned Python
dependencies in a separate scorer environment, plus the TeX toolchain. These pins
reproduce the metrics; they are not a dependency specification for your model
server. Metric models require approximately 7 GB on disk.

```
python -m venv /workspace/scitikz-metrics
/workspace/scitikz-metrics/bin/pip install -r /task/public/scorer-requirements.txt
apt-get update
apt-get install -y texlive-latex-extra texlive-pictures texlive-science \
  texlive-fonts-recommended texlive-fonts-extra texlive-extra-utils imagemagick ghostscript curl
/workspace/scitikz-metrics/bin/python /task/scitikz_models.py /workspace/metric-models \
  --expected /task/public/scorer-weights.json
/workspace/scitikz-metrics/bin/python /task/validate.py --server-url http://localhost:8000 \
  --models-dir /workspace/metric-models --predictions predictions.json
```

ImageMagick must allow reading PDF; some distributions disable it in policy.xml.
`--device cpu` is supported for diagnostics, but the trusted scorer runs on a GPU
with the released mixed-precision settings. Public diagnostic rendering executes
your generated TeX under your own account with shell escape disabled; use an
isolated development environment. The official renderer uses a restricted chroot.

Public validation is diagnostic, does not certify the training/provenance review,
and never grants permission to test on SciTikZ-Bench. Final scoring uses all 611
benchmark examples; no benchmark subset is exposed here.

## Offline serving dependencies

Bundle and pin the complete serving environment you tested, then check both
`/health` and an image request with network access disabled. A successful package
installation alone does not establish runtime compatibility. For example, vLLM
0.9.2 accepts newer Transformers through a broad dependency constraint, but fails
to start with Transformers 5.17.0 (`aimv2` registration collision). Its
[published test environment](https://github.com/vllm-project/vllm/blob/v0.9.2/requirements/test.txt)
pins Transformers 4.52.4, tokenizers 0.21.1 and huggingface-hub 0.33.0.
Choose compatible versions for the serving engine you use; no engine is required.
