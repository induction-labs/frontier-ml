Train a draft model for speculative decoding with Qwen3-8B as the target, run in non-thinking mode with greedy decoding. The target's tokenizer and configuration are in `/artifacts/qwen3-8b/`, and `/task/public/TARGET.json` names its revision. I have provided 40 example conversations with the target's greedy continuations.

You can start from any open-weights model that isn't a specialized speculative drafting model.

Write a script `serve.sh` which serves the drafter. It should have a `POST /draft` endpoint which takes the token IDs of a conversation so far and returns up to 16 token IDs that it predicts the target will generate next. Run offline on one H100, ready within 300 seconds. The whole submission, including weights, code and dependencies, must be at most 1,000,000,000 bytes. The held-out set is 200 conversations, five times the public examples; each request must answer within 60 seconds, and answering all of its requests may take at most 10,800 seconds in total.

Optimize the mean acceptance length.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
