Train a model that transcribes multi-instrument music. Given a 30-second audio clip, it lists every pitched note with its onset, offset, MIDI pitch and instrument class. I have provided 10 labeled example clips.

Write a script `serve.sh` which serves the model. It should have a `POST /transcribe` endpoint which takes a FLAC clip and returns its notes. It needs to run offline on one H100, be ready within 300 seconds, and transcribe at least 50x faster than real time. The whole submission, including weights, code and dependencies, must be at most 1,000,000,000 bytes.

Optimize note-onset F1, averaged over the nine instrument classes.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
