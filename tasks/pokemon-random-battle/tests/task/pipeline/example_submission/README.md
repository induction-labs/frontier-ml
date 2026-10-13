# Development starter

The 20-feature linear imitation policy is intentionally small. Use it to verify
the protocol and training/export loop, then replace it with a richer learned
encoder, recurrent/attention policy, population training, or policy-guided search.

Install `numpy` and `safetensors` in development, run `train.py`, copy `server.py`
next to its weights, and start it with `python server.py`. Evaluate using the
standard public validator after installing the separate Foul Play dependencies
with `public/setup_foul_play.sh`. Demonstration collection also plays against
Foul Play and takes longer than the former heuristic-only setup. The training script uses fresh development seeds.
Do not describe these toy imitation weights as PPO-trained or human-level.

For a final submission, add `serve.sh`, vendor compatible upstream dependency
wheels for offline installation into `/tmp`, and include training source/logs and
the standard `architecture.json`. The example is not a pre-approved submission.
It must pass the same trusted review as any other policy.
