"""Run tokens and the agent-side environment for the cutoff proxy. Standard library only: the Harbor plugin imports
this to point a trial at a proxy without needing the proxy's own dependencies."""
from __future__ import annotations

import hashlib
import hmac
import re
import time

RUN = re.compile(r"[A-Za-z0-9_-]{1,96}")


def mint(secret: str, run: str, *, seconds: int = 7 * 86400, now: float | None = None) -> str:
    """A run token `<run>.<expiry>.<signature>`: stateless, so the launcher and the proxy only share the secret."""
    if not RUN.fullmatch(run):
        raise ValueError("run labels are 1-96 characters of A-Z a-z 0-9 _ -")
    expiry = int((time.time() if now is None else now) + seconds)
    return f"{run}.{expiry}.{_sign(secret, run, expiry)}"


def verify(secret: str, token: str, *, now: float | None = None) -> str | None:
    """The run label of a valid, unexpired token."""
    parts = token.split(".")
    if len(parts) != 3 or not RUN.fullmatch(parts[0]) or not parts[1].isdecimal():
        return None
    if not hmac.compare_digest(parts[2], _sign(secret, parts[0], int(parts[1]))):
        return None
    return parts[0] if int(parts[1]) > (time.time() if now is None else now) else None


def _sign(secret: str, run: str, expiry: int) -> str:
    return hmac.new(secret.encode(), f"{run}.{expiry}".encode(), hashlib.sha256).hexdigest()[:32]


def agent_env(base_url: str, token: str | None = None) -> dict[str, str]:
    """Environment that points an agent's Hugging Face, git and OpenRouter clients at the proxy.

    git reads the GIT_CONFIG_* triple as extra configuration, so clones of github.com URLs (https, scp-style and
    ssh, including submodules and `pip install git+https://github.com/...`) go to the mirror without a config file.
    """
    root = base_url.rstrip("/") + (f"/t/{token}" if token else "")
    rewrite = f"url.{root}/gh/.insteadOf"
    sources = ("https://github.com/", "git@github.com:", "ssh://git@github.com/")
    env = {"FML_CUTOFF_PROXY": root, "HF_ENDPOINT": root + "/hf", "SPIN_OPENROUTER_BASE_URL": root + "/openrouter/v1",
           "GIT_CONFIG_COUNT": str(len(sources))}
    for index, source in enumerate(sources):
        env[f"GIT_CONFIG_KEY_{index}"] = rewrite
        env[f"GIT_CONFIG_VALUE_{index}"] = source
    return env
