Train a model that describes what visibly changed between two consecutive screenshots, like edited text, cursor moves, selections, menus, layout shifts or status indicators. I have provided one example pair with reference observations.

Write a script `serve.sh` which serves the model through an OpenAI-compatible chat endpoint that takes the before and after screenshots as two images and returns the description. Run offline on one H100, ready within 500 seconds, handling at least 5 screenshot pairs per second with 14 requests in flight. You can start from any open-weights model that isn't a specialized screen-difference model.

Optimize for describing every real change without claiming anything the screenshots don't show.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
