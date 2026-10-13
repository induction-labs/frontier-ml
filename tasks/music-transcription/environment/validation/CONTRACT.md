# Transcribe multi-instrument music into notes labelled with pitch, timing and instrument

There are nine instrument classes: piano (acoustic and electric pianos), mallets (celesta, glockenspiel, music box, vibraphone, marimba, xylophone), organ (organs and accordion), guitar (nylon, steel, jazz, clean, muted, overdriven and distorted guitars), bass (acoustic, electric, fretless, slap and synth bass), strings (violin, viola, cello, string ensemble, including tremolo and pizzicato), brass (trumpet, including muted trumpet, trombone, tuba, French horn, brass section), reed (saxophones, oboe, English horn, bassoon, clarinet) and pipe (piccolo, flute, recorder, pan flute, ocarina). Clips may contain multiple parts of the same class, sometimes with an unpitched drum kit that is not transcribed. The held-out clips use different compositions and mixes from the examples, with a range of instrument sounds. Simultaneous notes of the same class and pitch count as one note. Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want.

## Server interface

POST /transcribe with {"audio_format": "flac", "audio": "<base64>"}: a 30-second mono 44.1 kHz 16-bit FLAC clip. Return {"notes": [{"onset": <seconds>, "offset": <seconds>, "pitch": <int>, "instrument": "<class>"}, ...]}: times in seconds from the start of the clip with 0 <= onset < offset, pitch an integer MIDI note number 0-127 (60 is middle C), instrument one of piano, mallets, organ, guitar, bass, strings, brass, reed, pipe; at most 10,000 notes per clip, in any order. Any other shape is a failed request.

## Evaluation

A predicted note matches a reference note when both have the same instrument class and pitch and their onsets differ by at most 50 ms; each note matches at most once, and the number of matches is a maximum matching (the onset-only note metric of mir_eval, applied per class). For each class, matches, predicted notes and reference notes are summed over all 271 held-out clips; the class F1 is 2 x matches / (predicted + reference), and the score is the mean of the nine class F1s. Offsets are not scored. Reference onsets are the times the notes are played; rapid repeated notes are retained. A failed request counts as a clip with no notes. The 271 clips (135.5 minutes of audio) are sent 16 at a time, and the whole pass must finish within 162.6 seconds (50x real time), or the score is zero. A submission over 1,000,000,000 bytes (every file counted: weights, code, vendored wheels) scores zero before it is started.

## Artifacts

No starting model is prescribed: start from any public checkpoint, or from scratch (base_model "none", base_revision "from-scratch").

Include architecture.json with base_model, base_revision, training_evidence_files and inference_evidence_files. The file lists use paths relative to output/ and include training source, serve.sh and model-loading code. Keep initialization/training logs with the submission as readable text or JSON. Name the source of every training, calibration and selection corpus (in architecture.json or a data manifest) and keep its download commands and logs with the submission. I will check for training-data provenance. Ship compiled dependencies as unmodified PyPI or PyTorch wheels, installed into /tmp at startup. Prewarmed JIT kernel caches (Triton, TorchInductor) may be included.

## Submission

Put serve.sh, code, weights and dependencies in /workspace/output/ as regular files. Relative
symlinks that resolve inside output/ are copied as the regular files they point to. However, absolute, escaping,
dangling or cyclic symlinks fail the submission (for example, use `python -m venv --copies`, or vendor wheels).
The /workspace/output folder is copied to /submission when being evaluated. serve.sh runs as an unprivileged 
user that can write to /submission, /tmp and the preinstalled Python environment (/opt/venv) and nowhere else; 
install vendored wheels there or under /submission. You should package dependencies for offline installation,
since the evaluation server will not have internet access.
The command `bash /submission/serve.sh` receives HOST=0.0.0.0, PORT=8000 and
MODEL_NAME=submission. GET /health must return 200 when ready, within 300 seconds.
Startup and inference run offline on one H100 SXM 80GB, with 8 CPU cores and 80 GiB RAM.
The serving image is Ubuntu 24.04 with the CUDA 12.8 runtime libraries (no CUDA compiler), Python 3.12 in /opt/venv first on PATH (pip, numpy, pillow, safetensors, httpx preinstalled), gcc and Python headers. Also preinstalled: torch==2.8.0, torchaudio==2.8.0, soundfile==0.13.1, requests==2.34.2, urllib3==2.8.0, idna==3.20, certifi==2026.7.22, charset-normalizer==3.5.1. Vendor cp312 manylinux wheels for anything else.
Every request must answer HTTP 200 with the documented JSON. A request that fails (non-200, connection
error, invalid JSON) is retried twice on the same payload. If it still fails, or answers in an undocumented shape, 
you get no credit for that item. Up to one evaluation request in twenty may fail that way, but more than that gives 
you no credit for the entire run.

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
