Build a Pokémon battle bot for Generation 9 Random Battle singles. Pilot a randomly generated team of six, choosing moves, switches and Terastallization with incomplete information about the opponent. Research and choose your own approach. You may use existing bots, including unchanged Foul Play, improve them, or train your own policy.

Serve batched decisions offline through POST /act. Maximize battle wins across fixed held-out matchups against the pinned Foul Play opponent. Draws count as half a win.

Do not train, fine-tune, calibrate or select on the validation or test split of any public dataset or benchmark; training splits and other public corpora are fine.

Submit in `/workspace/output/`. Interface and runtime details: `/task/CONTRACT.md`.
Supporting files: `/task/public/`. Evaluate with `python /task/validate.py --server-url URL`.
Test the finished submission the way it will be run with `frontier selfcheck`.
Hand it in with `frontier submit` when it is final.
Use `spin --help` for compute and model APIs.
