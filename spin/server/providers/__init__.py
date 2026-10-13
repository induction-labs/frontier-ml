"""Compute providers: Modal (CPU and GPU sandboxes) and Vast.ai (GPU VMs). Credentials never leave the server."""
import importlib.util

NAMES = ("modal", "vastai")


def modal_configured():
    """True when the Modal SDK is installed and resolves a token (MODAL_TOKEN_ID/SECRET or ~/.modal.toml)."""
    if importlib.util.find_spec("modal") is None:
        return False
    from modal.config import config
    return bool(config.get("token_id") and config.get("token_secret"))


def enabled(env):
    """SPIN_PROVIDERS when set, else every provider whose credentials are present."""
    if env.get("SPIN_PROVIDERS", "").strip():
        names = [name.strip() for name in env["SPIN_PROVIDERS"].split(",") if name.strip()]
        unknown = sorted(set(names) - set(NAMES))
        if unknown:
            raise ValueError(f"Unknown providers in SPIN_PROVIDERS: {', '.join(unknown)}; choose from {', '.join(NAMES)}")
        return list(dict.fromkeys(names))
    return [name for name, present in (("modal", modal_configured()), ("vastai", bool(env.get("VAST_API_KEY")))) if present]


def require(*modules):
    missing = [module for module in modules if importlib.util.find_spec(module) is None]
    if missing:
        raise ValueError(f"Install {', '.join(missing)} to enable this provider (pip install {' '.join(missing)})")


def load_providers(env, *, prices, ssh_secret):
    result = {}
    for name in enabled(env):
        if name == "modal":
            require("modal")
            import modal
            from .modal import ModalProvider
            if not prices.get("modal"):
                raise ValueError("The price table has no Modal rates (a \"modal\" section)")
            app = modal.App.lookup(env.get("SPIN_MODAL_APP") or "spin-local", create_if_missing=True)
            result["modal"] = ModalProvider(modal, app=app, rates=prices["modal"])
        else:
            require("paramiko", "cryptography")
            import httpx
            from .vastai import DEFAULT_IMAGE, VastAIProvider
            if not env.get("VAST_API_KEY"):
                raise ValueError("Set VAST_API_KEY to enable Vast.ai")
            client = httpx.Client(base_url="https://console.vast.ai", follow_redirects=False, timeout=30,
                                  headers={"Authorization": "Bearer " + env["VAST_API_KEY"]})
            config = {"image": env.get("SPIN_VAST_IMAGE") or DEFAULT_IMAGE, "operator_covers_overrun": True,
                      "max_hourly_usd": env.get("SPIN_VAST_MAX_HOURLY_USD") or "30"}
            result["vastai"] = VastAIProvider(client, config=config, ssh_secret=ssh_secret)
    return result
