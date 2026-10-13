"""Environment selection without importing optional provider SDKs."""
import os
from pathlib import Path
import tomllib

MODAL_IMPORT_PATH = "frontier_ml.runtime.modal_environment:RecoverableModalEnvironment"


def use_recoverable_modal(config):
    """Detach a built-in Modal config before selecting our compatible adapter."""
    if getattr(config.type, "value", config.type) == "modal" and not config.import_path:
        config = config.model_copy(deep=True)
        config.type = None
        config.import_path = MODAL_IMPORT_PATH
    return config


def is_modal(config):
    return (getattr(config.type, "value", config.type) == "modal"
            or config.import_path == MODAL_IMPORT_PATH)


def modal_profile() -> dict:
    """MODAL_TOKEN_ID/SECRET from the environment, else the active profile written by `modal token new`."""
    path = Path.home() / ".modal.toml"
    if os.environ.get("MODAL_TOKEN_ID") or not path.is_file():
        return {}
    profiles = tomllib.loads(path.read_text())
    active = next((p for p in profiles.values() if isinstance(p, dict) and p.get("active")), None)
    active = active or next((p for p in profiles.values() if isinstance(p, dict)), {})
    if active.get("token_id") and active.get("token_secret"):
        return {"MODAL_TOKEN_ID": active["token_id"], "MODAL_TOKEN_SECRET": active["token_secret"]}
    return {}
