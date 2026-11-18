"""Cutoff proxy: Hugging Face, GitHub and OpenRouter as they stood at a fixed date.

An evaluation run that reaches the open internet sees whatever was published since the last run, so scores drift
with the world as well as with the agent. This service answers for three of those sources as of one cutoff date:

- hf.py: the Hub API and file downloads (`HF_ENDPOINT`), every revision pinned to its last commit before the cutoff.
- github.py: git clone/fetch, archives, raw files and releases from mirrors whose refs stop at the cutoff.
- openrouter.py: the OpenAI-compatible inference API in front of Spin, limited to models listed before the cutoff.

server.py is the HTTP service and its command line; core.py holds what the three share; tokens.py signs the per-run
tokens and builds the agent's environment; modal_app.py deploys the proxy on Modal behind its load balancer.
The agent is pointed at the proxy with environment variables (core.agent_env); nothing here blocks direct access to
the real hosts, so the per-request log (access.jsonl) is the record of what a run actually fetched.
"""
