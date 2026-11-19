# FrontierML

[GitHub](https://github.com/induction-labs/frontier-ml) | [Website](https://frontierml.org) | [Artifacts](https://huggingface.co/datasets/inductionlabs/frontier-ml-tasks) | [Trajectories](https://huggingface.co/datasets/inductionlabs/frontier-ml-trajectories)

FrontierML measures how well agents turn capital into model performance on challenging, end-to-end machine learning tasks. Each agent gets a 12-hour CPU sandbox and a fixed budget to spend on GPUs and model APIs through [Spin](https://spin.inductionlabs.com). It submits a standalone inference server, which is graded on held-out data on an offline H100. The 21 tasks span language, vision, audio, and games.

## Running

The tasks need the `frontier` Harbor plugin, which opens one spending wallet per trial, and Harbor 0.22.0. Install both from the GitHub repository, then download the tasks and run them from the local copy; the plugin does not accept `harbor run -d`.

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install "frontier-ml @ git+https://github.com/induction-labs/frontier-ml"

harbor download {{org}}/frontier-ml@v{{version}} -o .
harbor run -y -p frontier-ml/pii-redaction -e modal -a claude-code -m anthropic/claude-opus-5-5 \
  --plugin frontier --pk budget_usd=50
```

Pass `-p frontier-ml -n 21` to run every task, each with its own wallet. A trial allows up to 12 hours for the agent and up to 22 more for grading.

The run needs `MODAL_TOKEN_ID`, `MODAL_TOKEN_SECRET`, `HF_TOKEN`, `ANTHROPIC_API_KEY`, `OPENAI_API_KEY` and `SPIN_API_KEY` in the environment. The [GitHub README](https://github.com/induction-labs/frontier-ml#quickstart) explains each one, the costs, and how to view and regrade results.

Tasks downloaded from the Hub are named `{{org}}/<task>` in results; the copies in the GitHub repository are named `<task>`. The files are otherwise identical.
