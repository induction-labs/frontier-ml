"""Host-only provider contracts. Provider credentials never enter agent sandboxes."""
from __future__ import annotations

from decimal import Decimal, ROUND_CEILING
import math
import re
import time


class ProviderError(RuntimeError):
    pass


class ProviderRejectedError(ProviderError):
    """The provider definitively rejected provisioning without creating a resource."""


class ProviderUncertainError(ProviderError):
    """A mutation may have succeeded. Keep its full budget reservation."""


class ProviderNotReadyError(ProviderError):
    """The resource exists but is not ready for this operation; clients should poll."""


class ResourceDeadError(ProviderError):
    """The provider resource has stopped or died and will not resume by itself; polling cannot help.

    The lease is not refunded: the client should stop it and create another. Deliberately not a
    ResourceGoneError, which makes the broker persist a terminal status and skip provider.stop; a
    dead rental on a provider without a billing cap (Vast) keeps charging until it is destroyed.
    """


class CapabilityUnavailable(ProviderError):
    pass


class ResourceGoneError(ProviderError):
    """The provider resource no longer exists or has exited; a new resource is required."""
    status = "stopped"  # What the broker records so later calls fail fast without a provider call.


def duration(spec: dict, maximum: int = 86400) -> int:
    n = spec.get("max_seconds")
    if type(n) is not int or not 1 <= n <= maximum:
        raise ValueError(f"max_seconds must be an integer in [1, {maximum}]")
    return n


def lease_name(value: str) -> str:
    """The provider-side name of a lease, which recovery looks it up by."""
    if not re.fullmatch(r"[a-zA-Z0-9_-]{1,64}", value):
        raise ValueError("Invalid host-generated lease id")
    return "spin-" + value.lower().replace("_", "-")


def quote(provider: str, spec: dict, hourly_usd, enforcement: str) -> dict:
    seconds = duration(spec)
    rate = Decimal(str(hourly_usd))
    if not rate.is_finite() or rate <= 0:
        raise ValueError("A positive, finite host-configured billing rate is required")
    cost = (rate * Decimal(seconds) / 3600).quantize(Decimal("0.000001"), rounding=ROUND_CEILING)
    return {"provider": provider, "hourly_usd": str(rate), "max_seconds": seconds,
            "max_cost_usd": str(cost), "enforcement": enforcement, "spec": dict(spec)}


class OwnedProvider:
    """The broker persists ownership; adapters reattach only from that trusted ledger."""
    provider: str
    supports_extension = False
    # Initial health gate: the broker holds a new lease's credits until the trusted probe passes through
    # this adapter's exec, and destroys the machine (releasing them) if it has not passed by the deadline.
    initial_health_checks = True
    health_check_deadline_seconds = 1200

    def __init__(self, *, clock=time.time):
        self.clock = clock
        self.records: dict[str, dict] = {}

    def requires_initial_health_check(self, spec) -> bool:
        return self.initial_health_checks

    def initial_health_check(self, resource_id: str, command: str, timeout: int) -> dict:
        """Run the broker's probe before agent access. A not-ready or transport error means "try again";
        ResourceDeadError means the machine will never start and is cleaned up at once."""
        return self.exec(resource_id, command, timeout)

    def restore(self, record: dict) -> None:
        if record.get("provider") != self.provider:
            raise ValueError("Wrong provider")
        self.records[record["id"]] = dict(record)

    def _record(self, resource_id: str, *, active=False) -> dict:
        try:
            record = self.records[resource_id]
        except KeyError:
            raise ProviderError("Resource does not belong to this wallet") from None
        if active and (record["expires_at"] <= self.clock() or record.get("status") in {"stopped", "deleted", "terminated", "completed"}):
            raise ProviderError("Lease expired or stopped; a new reservation is required")
        return record

    def _timeout(self, resource_id: str, timeout: int) -> int:
        record = self._record(resource_id, active=True)
        if type(timeout) is not int or timeout <= 0:
            raise ValueError("timeout must be a positive integer")
        return max(1, min(timeout, math.ceil(record["expires_at"] - self.clock())))

    def quote_extension(self, resource_id, additional_seconds):
        raise CapabilityUnavailable("This provider cannot extend an existing lease; create a new resource")

    def _extension_quote(self, resource_id, additional_seconds, maximum=86400):
        r = self._record(resource_id, active=True)
        if r.get("billing_policy") != "prepaid_time":
            raise CapabilityUnavailable("Only prepaid leases can be extended; create a new resource")
        seconds = additional_seconds
        remaining = maximum - r["purchased_seconds"]
        if remaining < 1:
            raise ValueError(f"Lease already has this provider's maximum of {maximum} seconds; create a new resource")
        if type(seconds) is not int or not 1 <= seconds <= remaining:
            raise ValueError(f"Extension must be 1 to {remaining} seconds; leases on this provider "
                             f"last at most {maximum} seconds in total")
        total = r["purchased_seconds"] + seconds
        rate = Decimal(r["price_hourly_usd"])
        if not rate.is_finite() or rate < 0:
            raise ValueError("Invalid stored lease price")
        cost = (rate * seconds / 3600).quantize(Decimal("0.000001"), rounding=ROUND_CEILING)
        return {"additional_seconds": seconds, "total_seconds": total,
                "previous_expires_at": r["expires_at"], "expires_at": r["expires_at"] + seconds,
                "max_cost_usd": str(cost), "total_cost_usd": str(Decimal(r["charged_usd"]) + cost)}

    def _extended(self, resource_id, plan):
        r = self._record(resource_id)
        r.update(expires_at=plan["expires_at"], broker_expires_at=plan["expires_at"],
                 purchased_seconds=plan["total_seconds"], charged_usd=plan["total_cost_usd"],
                 max_cost_usd=plan["total_cost_usd"])
        return dict(r)

    def _created(self, resource_id: str, lease_id: str, q: dict, started: float, **fields) -> dict:
        record = {"id": resource_id, "lease_id": lease_id, "provider": self.provider,
                  "status": "running", "created_at": started,
                  "expires_at": started + q["max_seconds"],
                  "max_cost_usd": q["max_cost_usd"], **fields}
        self.records[resource_id] = record
        return dict(record)

    def _stopped(self, record: dict, status: str) -> dict:
        """A confirmed stop. Leases are prepaid, so the settlement is the full price; it also marks the stop done."""
        record.update(status=status, settled_cost_usd=record["max_cost_usd"], settlement_basis="prepaid")
        return dict(record)
