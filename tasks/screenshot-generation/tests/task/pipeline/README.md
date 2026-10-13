# screenshot-generation data and baseline pipeline (operator only; never shipped to the agent)

Paths assume the work directory `/scratch/frontier-ml-workspace/work/screenshot-generation` (`W`) with
`venv` (vllm 0.17.1, torch 2.10 cu128, rapidocr 3.9.2, onnxruntime-gpu 1.23.2, playwright), `venv-gen`
(diffusers main, transformers 5) and `py.sh` / `py-gen.sh` wrappers that pin caches and `CUDA_VISIBLE_DEVICES`.

1. Fonts and browser: `bash fetch.sh` (Liberation + Noto, writes `fonts.conf`); Playwright chromium headless
   shell build 1243 under `$W/ms-playwright`.
2. Domains: `crawl/exclusion_domains.py` (ScreenParse v2 url column → `exclusion_domains.txt`),
   `crawl/make_candidates.py` (Tranco 8P4GV ranks > 10k, English-leaning TLDs, salted shuffle).
3. Crawl: `crawl/crawl.py --domains candidates.txt --out raw --workers 16` (robots.txt, consent dismissal,
   stability check, `extract_boxes.js` DOM text boxes and imagery rectangles).
4. Pages: `dataset/process_pages.py raw processed.jsonl --require-imagery` (pinned OCR completeness filter,
   ignore regions, real-page T/L).
5. Text: `dataset/draft_descriptions.py` (Qwen3.6-27B-FP8) then `dataset/validate_questions.py` (pinned judge on
   the real image and on the description alone; leak filter).
6. Splits: `dataset/assemble.py --processed processed.jsonl --validated validated.jsonl --out data/screenshot-generation`
   (one page per domain, salted order; salt in `$W/dataset/split_salt.txt`, keep it private).
7. Human pass (PENDING): `dataset/review_tool.py build|apply`.
8. Baselines: `baselines/sweep.sh DATA_DIR` generates and scores real, zimage, flux2klein, compositor, html
   and blank with the OCR, faithfulness and realism metric.
9. Manifest: `.claude/skills/frontier-task/scripts/write_manifest.py data/screenshot-generation --source dataset/source.json
   --media-fields image --input-fields texts description`.
10. Starting artifact: `dataset/make_artifact.py` (hardlinks Z-Image-Turbo into `artifacts/screenshot-generation/z-image-turbo`
    with a lock file and writes `public/STARTING_ARTIFACT.json`).
11. Local end-to-end check: `e2e/run_verifier.py e2e/zimage_submission private/test.jsonl out.json`
    (scorer preflight, HTTP generation, pinned scorers and quality gates).

## Internal screen-recording data (v2; local only, never upload)

Output: `/scratch/frontier-ml-workspace/frontier-mle-clean/data/screenshot-generation-internal/`. Work dir
`$W/internal` (salt in `$W/internal/salt.txt`; keep it private).

1. `internal/extract.py raw --videos 4000`: one salted frame per video from
   `/scratch/data/ocr_screens_v2_described_with_images`, PNG + Cloud Vision text lines.
2. `internal/verify.py raw verified.jsonl --workers 3`: pinned PP-OCRv5 exact-match cross-check -> requested
   boxes and neutral ignore regions; page filters. Resumable; error rows are retried.
3. `internal/prepare_candidates.py verified.jsonl candidates.jsonl` (real-page L >= 0.85).
4. `dataset/draft_descriptions.py candidates.jsonl drafts.jsonl --raw raw --kind screen`, then
   `dataset/validate_questions.py candidates.jsonl drafts.jsonl validated.jsonl --raw raw`
   (`dataset/reclassify_leaks.py` re-applies the leak rule without judge calls).
5. `internal/assemble.py --candidates ... --drafts ... --validated ... --raw raw --out DATA_DIR`.
6. `internal/sweep.sh`: generate and score baselines with the screen prompt.
7. Manifest: `.agents/skills/frontier-task/scripts/write_manifest.py DATA_DIR --task screenshot-generation
   --source internal/source.json --media-fields image --input-fields texts description`.
