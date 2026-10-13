"""Trusted LLM judge clients; credentials stay in the verifier's environment."""
import json
import os
import urllib.request

from .server import VerificationFailure

# A reasoning judge answers only after spending its completion budget, so the read timeout grows with the budget a
# request allows: a fixed 120 s cut off every 32,768-token retry of MATH-500's escalating budgets.
MIN_TOKENS_PER_SECOND = 50


def judge_from_config(config):
    env_name = config.get("judge_api_key_env", "OPENAI_API_KEY")
    key = os.environ.get(env_name)
    if not key:
        raise VerificationFailure(f"Judge requires trusted {env_name}")
    endpoint = config.get("judge_endpoint", "https://api.openai.com/v1/chat/completions")
    if not endpoint.startswith("https://"):
        raise VerificationFailure("Judge endpoint must use HTTPS")

    def judge(**payload):
        request = urllib.request.Request(endpoint, json.dumps(payload).encode(),
                                         {"Authorization": "Bearer " + key, "Content-Type": "application/json"})
        budget = payload.get("max_completion_tokens") or payload.get("max_tokens") or 0
        timeout = max(config.get("judge_timeout_seconds", 120), budget / MIN_TOKENS_PER_SECOND)
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.load(response)
    return judge
