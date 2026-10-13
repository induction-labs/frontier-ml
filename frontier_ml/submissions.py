"""Agent sessions that ended with `frontier submit --confirm`, as the Harbor process knows them.

selfcheck_service.TrialSelfChecks marks the agent environment's session when it accepts a submission, just before it
stops every agent process. The harness then sees its command fail (killed); harnesses that resume a session after a
failed command (codex_agent.ResilientCodex, harness/muse_code.py, harness/antigravity_cli.py) ask `ended()` first, so
that a submitted session is never resumed. Keys are Harbor environment session IDs (`<trial name>__env`); a new trial
attempt reuses its trial's name, so the plugin forgets the key when an attempt's agent starts.
"""

_ended: set[str] = set()


def mark(session_id: str) -> None:
    _ended.add(session_id)


def forget(session_id: str) -> None:
    _ended.discard(session_id)


def ended(environment) -> bool:
    """Whether the agent in this Harbor environment has submitted, which ends its session."""
    return getattr(environment, "session_id", None) in _ended
