"""Account-side Spin wallet client. Account keys never enter agent sandboxes."""
from decimal import Decimal, InvalidOperation
import json
import os
from pathlib import Path
import sys
from urllib.parse import quote, urlsplit
import uuid

from .cli import BrokerClient
from .render import emit


class SpinClient(BrokerClient):
    def __init__(self, url=None, token=None):
        url = url or os.environ.get("SPIN_BASE_URL") or "https://api.spin.inductionlabs.com/v1"
        token = token or os.environ.get("SPIN_API_KEY")
        if not token:
            raise ValueError("Set SPIN_API_KEY to an account API key (spin serve prints the operator key)")
        parts = urlsplit(url)
        local = parts.scheme == "http" and parts.hostname in {"localhost", "127.0.0.1", "::1"}
        allowed = parts.scheme == "http" and os.environ.get("SPIN_ALLOW_HTTP") == "1"
        if parts.scheme != "https" and not (local or allowed):
            raise ValueError("Spin requires HTTPS, except on localhost or with SPIN_ALLOW_HTTP=1")
        super().__init__(url, token)


def add_parser(groups, fmt):
    group = groups.add_parser("wallet", help="manage budgeted Spin wallets using an account API key")
    sub = group.add_subparsers(dest="action", required=True)
    sub.add_parser("list", help="list your account's wallets", parents=[fmt])
    create = sub.add_parser("create", help="reserve credit and create credentials for an agent sandbox")
    create.add_argument("--name", required=True)
    create.add_argument("--budget", required=True, help="resource allowance in USD")
    create.add_argument("--request-id", help="stable ID for safe retries; generated when omitted")
    create.add_argument("--expires-seconds", type=int, default=86400, help="seconds until auto-close, up to 30 days; 0 never expires")
    create.add_argument("--static-pricing", nargs="?", const="index", metavar="TABLE",
                        help="show and spend the budget in fixed GPU prices (the account still pays real cost): alone, "
                             'the installation\'s price table; or a table as JSON or @path, e.g. {"gpu_hour_usd": {"H100": 3}}, '
                             "or a gpu-prices index file, priced at each model's median per GPU")
    for action, help_text in (("close", "stop resources and return unused credit"), ("resources", "list a wallet's allocated resources")):
        command = sub.add_parser(action, help=help_text, parents=[fmt] if action == "resources" else [])
        command.add_argument("id", help="wallet ID")


def cents(value):
    try:
        amount = Decimal(value)
        if not amount.is_finite() or amount * 100 != (amount * 100).to_integral_value():
            raise ValueError("Budget must be USD with at most two decimal places")
        if not Decimal("0.01") <= amount <= 10_000:
            raise ValueError("Budget must be between $0.01 and $10,000")
        return int(amount * 100)
    except InvalidOperation as exc:
        raise ValueError("Invalid USD budget") from exc


def static_pricing(value):
    """--static-pricing as the API takes it: true for the installation's table, else a price table."""
    if value == "index":
        return True
    raw = Path(value[1:]).read_text() if value.startswith("@") else value
    table = json.loads(raw, parse_float=str)  # Prices travel as exact decimal strings.
    if isinstance(table, dict) and isinstance(table.get("models"), dict):
        # An index made before form factors were priced apart has only the median.
        return {"gpu_hour_usd": {model: entry.get("static_per_gpu_hourly_usd", entry["median_per_gpu_hourly_usd"])
                                 for model, entry in table["models"].items()},
                **({"version": "gpu-prices " + table["generated_at"]} if table.get("generated_at") else {})}
    return table


def main(args, *, client=None):
    client = client or SpinClient()
    if args.action == "list":
        result = client.request("GET", "/v1/wallets")
    elif args.action == "create":
        body = {"name": args.name, "budget_cents": cents(args.budget), "expires_in_seconds": args.expires_seconds}
        if args.static_pricing:
            body["static_pricing"] = static_pricing(args.static_pricing)
        key = args.request_id or str(uuid.uuid4())
        try:
            result = client.request("POST", "/v1/wallets", body, key=key)
        except OSError as exc:  # includes HTTPError
            # Only an unknown outcome needs the key: retrying with it cannot create a second wallet.
            if not args.request_id and getattr(exc, "code", 500) >= 500:
                print("Outcome unknown; retry with --request-id " + key, file=sys.stderr)
            raise
    else:
        path = "/v1/wallets/" + quote(args.id, safe="")
        if args.action == "close":
            result = client.request("POST", path + "/close", {})
        else:
            result = client.request("GET", path + "/resources")
    emit(args, result)
