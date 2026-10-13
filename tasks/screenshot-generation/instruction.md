Build a system that generates realistic 1280x720 computer screenshots. Each request has a description of the screen and a list of text lines with pixel boxes. Render every line exactly, inside its box, and don't add any other text. I have provided 25 example screenshots.

Write a script `serve.sh` which serves the generator. It should have a `POST /v1/images/generations` endpoint which returns the screenshot as a PNG. Run offline on one H100, ready within 300 seconds, with p95 latency of at most 30 seconds per image and every request answered within 120 seconds. You can use any open-weights model that isn't a specialized screenshot generation model.

Optimize OCR text accuracy and layout, plus gpt-6-luna's ratings of realism and faithfulness to the description.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
