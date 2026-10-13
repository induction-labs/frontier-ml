Build a learned video frame interpolation system that generates intermediate frames from two endpoints. You may use public pretrained models that are not specialist VFI models, including general image/video models and standalone optical-flow models, or train from scratch.

Do not use specialist VFI checkpoints such as GIMM-VFI, RIFE or FILM, their VFI-trained components or descendants, including as teachers, ensemble members or fallbacks. Public architectures and source code are allowed. Declare all pretrained components and their origins.

Achieve the highest possible final evaluation score within your budget. After obtaining a working submission, continue testing promising quality improvements.

Serve POST /interpolate offline on one H100, ready within 300 seconds. Generate 2,250 frames within 2,250 seconds, including transport and export.

Both measured anchors are frozen: matching the starting baseline earns 0; matching the quality target earns 1. Perfect reconstruction is not required. The reference errors, equal LPIPS/FloLPIPS and 4x/8x/16x weights, and exact gap-closing formula are in /task/CONTRACT.md. The quality target does not authorize prohibited weights.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine. However, you are free to use the validation data samples I have provided however you want.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
