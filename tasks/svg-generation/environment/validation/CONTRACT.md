# Finetune gpt-oss-20b to draw SVGs

The submitted weights must descend from openai/gpt-oss-20b at the provided revision and from no other model's weights. Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want. Also don't train on any public text-to-SVG evaluation suite.

## Server interface

I will serve model/ with SGLang 0.5.9 (the lmsysorg/sglang image at commit bbe9c7eeb520b0a67e92d133dfc137a3688dc7f2) on one H100, with --quantization fp8 --reasoning-parser gpt-oss --sampling-defaults openai  and a 20,480-token context, with gpt-oss-20b's own config.json, generation_config.json, tokenizer config and chat template. Only your safetensors shards, their index and tokenizer.json are read from model/. /task/public/serving/ holds the exact server script and files. Each request is POST /v1/chat/completions with model=submission and one user message, "Generate a self-contained SVG for this instruction. Return only the SVG." followed by a blank line and the instruction, with reasoning_effort=low, temperature=0.5, max_tokens=16384 and stream=false. Only choices[0].message.content is used and its outermost <svg ...>...</svg> must be at most 32,768 UTF-8 bytes, have a finite viewBox and use only the elements and attributes allowed in /task/public/SCHEMA.md.

## Evaluation

Each SVG is rendered at 512x512 and gpt-6-sol grades it against the instruction on six aspects from 1 to 10: fidelity, detail, proportions, composition, aesthetics and overall; a reference drawing is shown to it only to identify the subjects. An example scores (fidelity - 1)/9 times (the mean of the other five aspects - 1)/9, so a drawing that misses the instruction earns nothing for polish. The raw score is the mean over complexity groups, and the reward rescales it so that untouched gpt-oss-20b served this way scores 0 and the reference drawings score 1, clamped to [0, 1]. The exact rubric is in /task/validate.py. An invalid SVG (including a missing or over-budget one) scores 0, and so does a request that fails after the shared retries, times out or answers in an undocumented shape: it stays in its complexity group at 0. All 254 held-out requests are sent at 8 in flight as one timed batch that must finish within 10,800 seconds, retries included; a slower batch scores zero. Answers are sampled, so grading the same weights twice can differ by about 0.01 in the raw score.

## Artifacts

openai/gpt-oss-20b at revision 6cee5e81ee83917806bbde320786a8fb61efebee is in /artifacts/gpt-oss-20b/; its experts are stored in MXFP4, so dequantize to bf16 before training. Check that model/ ends up with all 411 tensors: for example, transformers 5.x save_pretrained() after loading with Mxfp4Config(dequantize=True) silently drops the expert weights. Include training_manifest.json (format in /task/public/SCHEMA.md) and every file it lists.

Include architecture.json with base_model "openai/gpt-oss-20b", base_revision "6cee5e81ee83917806bbde320786a8fb61efebee" and training_evidence_files (paths relative to output/, including the training source and the code that wrote model/). Keep training logs with the submission as readable text or JSON. Name the source of every training, calibration and selection corpus (in architecture.json or a data manifest) and keep its download or generation commands and logs. I will check that the weights were finetuned from that checkpoint during your session, and the training-data provenance.

## Submission

Put the finetuned checkpoint in /workspace/output/model/ as regular files: bf16 safetensors shards holding exactly gpt-oss-20b's 411 tensors in the unquantized layout (names, shapes and dtypes in /task/public/serving/tensors.json), model.safetensors.index.json mapping them, and tokenizer.json copied unchanged from /artifacts/gpt-oss-20b/. Other files in model/ are ignored. At least one tensor must differ from the base checkpoint. Do not submit a serve.sh: the verifier copies /workspace/output to /submission and starts its own server (/task/public/serving/serve.sh) offline on one H100 SXM 80GB, and GET /health must answer within 900 seconds. Also put training_manifest.json, architecture.json and your training code, configuration and logs in /workspace/output/. An answer that holds no SVG document (for example when the model spends all 16,384 tokens reasoning) or whose SVG exceeds 32,768 bytes is an invalid SVG: it gets no credit for that item, and grading goes on however many there are. A request that times out or still fails after two retries on the same payload also gets no credit for that item; more than one in twenty failing that way stops grading as a fault of the grading server.

## Development

The development machine is CPU-only. Use `spin --help` to rent compute and access
model APIs, and `spin budget` to check how much allowance you have left.
The API key your coding harness was started with is for the harness alone. Do not use the API key
in your own code, scripts, curl or Spin machines, copy it anywhere, or start extra harness sessions as workers;
AI model calls you make yourself must go through `spin` and spend your allowance. I will check your session transcript for this.

To check your solution locally, run `python /task/validate.py --server-url URL`. See /task/public for examples 
and supporting files.

You can check your solution on the same infrastructure I will evaluate it on using `frontier selfcheck`,
which copies /workspace/output/, starts it as the evaluation does, sends public examples through the same 
response checks and measures the speed requirements on public inputs. Its report gives the stage reached, 
the digest of the files it checked, the GPU, the startup time and each requirement's measured value and 
margin, and the end of the server log. You can check your solution four times.

A check usually takes 10-40 minutes (`frontier selfcheck status` shows this task's limit) and runs while you keep
working. See `frontier selfcheck --help` for more details.

When your solution is final, hand it in with `frontier submit`: it shows what /workspace/output/ holds and asks
you to confirm with `frontier submit --confirm`, which submits the folder as it is and ends your session. Without
a submission, /workspace/output/ is graded as it is when your session ends. See `frontier submit --help`.

Your session ends the moment you finish a turn without a tool call: background commands, scheduled
wake-ups and reminders will not bring you back, and anything not yet in /workspace/output/ is lost.
Wait for long jobs with foreground commands under the tool time limit (for example
`sleep 600; tail -n 3 train.log`, repeated) and stop only when the final files are in place.
