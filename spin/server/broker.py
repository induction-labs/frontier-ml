"""Agent API semantics for one wallet: prepaid leases, the initial health gate, cleanup and model calls."""
from __future__ import annotations

import base64
from contextlib import contextmanager
from datetime import datetime, timezone
from decimal import Decimal, InvalidOperation
import io
import json
import logging
import secrets
import time

import httpx

from ..limits import DEFAULT_TTL, INLINE_MAX_BYTES
from .ledger import BudgetExceeded, Conflict, IdempotencyConflict, NotFound
from .pricing import StaticPricing, cost as static_cost, market_price_field, with_gpu_model
from .providers import files
from .providers.base import (CapabilityUnavailable, ProviderError, ProviderNotReadyError, ProviderRejectedError,
                             ProviderUncertainError, ResourceDeadError, ResourceGoneError)
from .providers.build import record_spec
from .providers.health import PROBE_TIMEOUT, passed, probe_command
from .storage import CLEANUP_GRACE, CLEANUP_INTERVAL

RECOVERY_GRACE_SECONDS = 15 * 60  # before an unconfirmed create found nowhere is released
CREATE_IN_FLIGHT_SECONDS = 60      # a fresh unresolved lease is a create still talking to the provider
TERMINAL = {"stopped", "deleted", "terminated", "destroyed", "completed", "rejected"}
# Spec fields that name an offer; everything else in a spec (duration, disk, image) is adjustable.
OFFER_IDENTITY = ("offer_id",)
# Operations that take a request ID (Idempotency-Key) and are safe to retry with it.
KEYED = {"/compute/create", "/compute/extend", "/v1/chat/completions"}


class ApiError(Exception):
    def __init__(self, status, code, message):
        super().__init__(message)
        self.status, self.code = status, code


class NotConfigured(Exception):
    """The operator has not enabled this capability on the installation."""


class ResourceInactive(Exception):
    pass


class NotSupported(Exception):
    pass


def present(value, key=""):
    """Public JSON: ISO 8601 UTC timestamps and plain decimal USD strings."""
    if isinstance(value, dict):
        return {k: present(v, k) for k, v in value.items()}
    if isinstance(value, list):
        return [present(v, key) for v in value]
    if isinstance(value, bool) or value is None:
        return value
    if key.endswith("_at") and isinstance(value, (int, float)):
        # Same shape as Go's RFC 3339 output: microseconds, trailing zeros trimmed.
        return datetime.fromtimestamp(value, timezone.utc).strftime("%Y-%m-%dT%H:%M:%S.%f").rstrip("0").rstrip(".") + "Z"
    if key.endswith("_usd") and isinstance(value, (str, int, float, Decimal)):
        try:
            amount = Decimal(str(value))
        except InvalidOperation:
            return value
        if amount.is_finite():
            return "0" if amount == 0 else format(amount.normalize(), "f")
    return value


def error_response(exc, path=None):
    """Map an exception to (HTTP status, error code, client-facing message)."""
    if isinstance(exc, ApiError):
        return exc.status, exc.code, str(exc)
    if isinstance(exc, IdempotencyConflict):
        return 409, "idempotency_conflict", str(exc)
    if isinstance(exc, Conflict):
        return 409, "operation_pending", str(exc)
    if isinstance(exc, BudgetExceeded):
        return 402, "budget_exceeded", str(exc)
    if isinstance(exc, NotConfigured):
        return 501, "not_configured", str(exc)
    if isinstance(exc, (ResourceInactive, ResourceGoneError)):
        return 409, "resource_inactive", str(exc)
    if isinstance(exc, CapabilityUnavailable):
        return 400, "extension_not_supported", str(exc)
    if isinstance(exc, NotSupported):
        return 400, "not_supported", str(exc)
    if isinstance(exc, ProviderRejectedError):
        return 409, "provider_rejected", str(exc)
    if isinstance(exc, NotFound):
        return 404, "not_found", exc.args[0] if exc.args else "Not found"
    if isinstance(exc, PermissionError):
        return 403, "forbidden", str(exc) or "Token is not allowed to perform this operation"
    if isinstance(exc, KeyError):
        return 400, "invalid_request", "Missing required field: " + str(exc.args[0] if exc.args else "")
    if isinstance(exc, TypeError):
        return 400, "invalid_request", "Request fields have the wrong type"
    if isinstance(exc, ValueError):
        return 400, "invalid_request", str(exc) or "Invalid request"
    retry = "Retry with the same request ID." if path in KEYED else "Retry shortly."
    if isinstance(exc, ProviderNotReadyError):
        return 409, "not_ready", str(exc)  # A normal transient state, not a failure: poll and retry.
    if isinstance(exc, ResourceDeadError):
        # A final state: polling would only run down the prepaid clock. Not persisted as terminal, so
        # compute stop still destroys the rental and ends its billing.
        return 409, "resource_dead", str(exc)
    if isinstance(exc, ProviderError):
        # Adapters write these messages themselves; they carry no credentials.
        held = " Any reservation is held until it is reconciled." if isinstance(exc, ProviderUncertainError) else ""
        return 502, "provider_error", f"{exc}.{held} {retry}".replace("..", ".")
    if isinstance(exc, httpx.HTTPError):
        # Name only the class; the text can carry the provider URL.
        return 502, "provider_error", f"Provider API request failed ({type(exc).__name__}). {retry}"
    return 500, "internal_error", f"Internal error ({type(exc).__name__}); {retry[0].lower()}{retry[1:]}"


def market_offers(providers):
    """Every provider's current offers at market prices, and each provider's listing status.

    A provider whose listing fails is logged and left out so the others stay available.
    """
    offers, statuses = [], []
    for name, provider in providers.items():
        try:
            found = provider.offers()
        except Exception as exc:
            logging.warning("Provider %s offers unavailable: %s", name, type(exc).__name__)
            statuses.append({"provider": name, "status": "error", "message": error_response(exc)[2]})
            continue
        offers += [with_gpu_model({**offer, "supports_extension": getattr(provider, "supports_extension", False)}) for offer in found]
        statuses.append({"provider": name, "status": "ok", "offers": len(found)})
    return offers, statuses


def cleanup_storage(ledger, storage, wallet_id=None, *, force=False):
    """One cleanup pass over closed wallets' storage; due scopes only unless forced."""
    results = []
    for scope in ledger.transfer_scopes(wallet_id, due_at=None if force else time.time()):
        started, complete = time.time(), False
        with ledger.try_resource_lock("transfer:" + scope["wallet_id"]) as locked:
            if locked:
                try:
                    if not storage:
                        raise NotConfigured("Restore the R2 configuration to finish wallet storage cleanup")
                    complete = storage.cleanup(scope["wallet_id"], scope)
                except Exception as exc:
                    logging.warning("Storage cleanup for %s failed: %s", scope["wallet_id"], type(exc).__name__)
                # Sessions issued before the close may still write until they expire, so sweeping continues until then.
                ledger.record_transfer_cleanup(scope["wallet_id"], next_cleanup_at=started + CLEANUP_INTERVAL,
                                               cleaned=complete and started >= scope["expires_at"] + CLEANUP_GRACE)
        results.append({"id": scope["wallet_id"], "kind": "storage",
                        "status": "storage_cleaned" if complete else "storage_cleanup_pending"})
    return results


class Broker:
    """The agent API of one wallet. Built per request; all state lives in the ledger and the providers."""

    def __init__(self, ledger, providers, wallet, *, openrouter=None, storage=None, public_url="http://127.0.0.1:8080"):
        self.ledger, self.providers, self.openrouter, self.storage = ledger, providers, openrouter, storage
        self.wallet_id, self.wallet_expires_at = wallet["id"], wallet["expires_at"]
        # The wallet's price table, when it pays static prices rather than the providers'.
        self.static = StaticPricing(wallet["static_pricing"]) if wallet.get("static_pricing") else None
        self.public_url = public_url.rstrip("/")
        self.lease_lock_wait = 10.0  # seconds a due lease is retried for its lock in one reap tick

    def _provider(self, name):
        if name not in self.providers:
            enabled = ", ".join(sorted(self.providers)) or "none"
            raise ValueError(f"Unknown provider {name!r}; enabled providers: {enabled}")
        return self.providers[name]

    @staticmethod
    def _public_resource(record):
        result = dict(record["result"] or {})
        provider_id = result.pop("id", None)
        result.pop("settled_cost_usd", None)
        result.pop("settlement_basis", None)
        # Adapters never return account credentials. Only scoped resource access
        # (a per-lease SSH key) may be disclosed to the wallet that owns it.
        public = {**result, "id": record["id"], "provider_id": provider_id,
                  "provider": record["metadata"]["provider"], "billing_state": record["state"],
                  "reserved_usd": "0" if record["state"] == "settled" else record["maximum_usd"],
                  "charged_usd": result.get("charged_usd", record["spent_usd"])}
        static = record["metadata"].get("static")
        if not static:
            return public
        # A static wallet sees its own prices. The stored result keeps the provider's, which
        # extensions and recovery are computed from.
        extensions = Decimal(result.get("static_extensions_usd", "0"))
        return {**{k: v for k, v in public.items() if not market_price_field(k)}, "pricing": "static",
                "priced_as": {k: v for k, v in static.items() if k != "hourly_usd"}, "price_hourly_usd": static["hourly_usd"],
                "reserved_usd": "0" if record["state"] == "settled" else record["static_maximum_usd"],
                "charged_usd": str(Decimal(record["static_spent_usd"] or "0") + extensions)}

    def _static_price(self, provider, quote, spec):
        """The wallet's static price for a quote, or None when the lease counts at its market price."""
        if not self.static:
            return None
        gpu, gpus, cpus = self._machine(provider, quote, quote.get("spec", spec))
        price = self.static.price(gpu, gpus, cpus)
        if price is None:
            raise ValueError(f"This wallet uses static prices and has none for {gpu or 'this machine'}; "
                             "choose an offer listed by compute offers")
        if price == "market":
            return None
        return {**price, "max_cost_usd": static_cost(price["hourly_usd"], quote["max_seconds"])}

    @staticmethod
    def _machine(provider, quote, spec):
        """(gpu, gpus, cpus) a quote leases: from the quote or spec when they say, else from the offer the spec names."""
        if "gpus" in quote:
            return quote.get("gpu"), quote["gpus"], quote.get("cpus")
        if "gpus" in spec and ("gpu" in spec or spec["gpus"] == 0):
            return spec.get("gpu"), spec["gpus"], spec.get("cpus")
        named = [k for k in OFFER_IDENTITY if k in spec]
        for offer in provider.offers() if named else []:
            listed = offer.get("spec") or {}
            if all(str(listed.get(k)) == str(spec[k]) for k in named):
                return offer.get("gpu"), spec.get("gpus", offer.get("gpus")), spec.get("cpus", offer.get("cpus"))
        raise ValueError("That offer is no longer listed; choose one from compute offers")

    @staticmethod
    def _static_extension(record, payment, result):
        """A static lease's extensions add their static price to the resource beside the provider's real total."""
        added = payment.get("static_maximum_usd")
        if not added:
            return result
        return {**result, "static_extensions_usd": str(Decimal(record["result"].get("static_extensions_usd", "0")) + Decimal(added))}

    def budget(self):
        """The budget as the wallet sees it: in its static prices when it has them."""
        status = self.ledger.status(self.wallet_id)
        static = status.pop("static", None)
        return {**status, **static, "pricing": "static"} if static else status

    def _prepaid(self, record, result, quote):
        created = result.get("created_at")
        start = max(record["created"], created) if isinstance(created, (int, float)) else record["created"]
        expiry = start + quote["max_seconds"]
        return {**result, "provider": record["metadata"]["provider"],
                "broker_expires_at": expiry, "expires_at": expiry,
                "billing_policy": "prepaid_time", "charged_usd": record["maximum_usd"],
                "supports_extension": getattr(self._provider(record["metadata"]["provider"]), "supports_extension", False),
                "price_hourly_usd": quote.get("hourly_usd", str(Decimal(record["maximum_usd"]) * 3600 / quote["max_seconds"])),
                "purchased_seconds": quote["max_seconds"], "spec": quote.get("spec", record["metadata"]["spec"])}

    def _initial_check(self, record, result, quote):
        """Keep the reservation until a trusted, pre-access health check succeeds."""
        provider = self._provider(record["metadata"]["provider"])
        pending = self._prepaid(record, result, quote)
        try:
            gpus = self._machine(provider, quote, pending["spec"])[1]
        except Exception:
            gpus = None  # Unknown hardware gets the GPU probe; a lease that cannot pass it is released, not charged.
        now = time.time()
        # The earliest of the provider's boot allowance, the lease's own expiry and the wallet's.
        limits = {"provider": now + getattr(provider, "health_check_deadline_seconds", 1200),
                  "lease_expiry": pending["expires_at"], "wallet_expiry": self.wallet_expires_at or float("inf")}
        source = min(limits, key=limits.get)
        pending.update(status="checking", charged_usd="0", health_check={
            "state": "pending", "gpus": gpus, "started_at": now, "attempts": 0,
            "deadline_at": limits[source], "deadline_source": source, "next_check_at": 0})
        return pending

    def _health_event(self, record, check, event):
        """One structured log line per gate outcome: how long the machine was given against which limit."""
        logging.info(json.dumps({"event": event, "provider": record["metadata"]["provider"], "lease_id": record["id"],
            "wallet_id": record["wallet_id"], **{k: check.get(k) for k in ("waited_seconds", "deadline_source", "attempts",
            "gpus", "last_outcome", "last_error", "last_exit_code", "reason")}}, default=str))

    def _reject_initial_check(self, record, provider, reason):
        now = time.time()
        previous = record["result"].get("health_check") or {}
        check = {k: v for k, v in previous.items() if k != "next_check_at"}
        check.update(state="failed", reason=reason, failed_at=now,
                     waited_seconds=round(now - previous.get("started_at", record["created"])))
        if previous.get("state") != "failed":
            self._health_event(record, check, "initial_health_check_failed")
        # Persist the cleanup intent first: a restart must never hand this machine out.
        result = {**record["result"], "health_check": check}
        self.ledger.set_result(record["id"], result)
        provider.restore(result)
        provider.stop(result["id"])  # An uncertain stop leaves the full reservation held.
        self.ledger.reject_compute(record["id"])
        self.ledger.set_result(record["id"], {**result, "id": None, "status": "rejected", "charged_usd": "0",
            "reason": reason + "; resource cleaned up and reserved credits released",
            "released_usd": record["maximum_usd"]})
        return self.ledger.get(record["id"])

    def _check_initial_health(self, record, provider, *, probe=True):
        """Polls probe readiness; the reaper enforces cleanup without waiting on a probe."""
        record = self.ledger.get(record["id"])
        result = record["result"]
        if result.get("status") != "checking":
            return record
        check = result["health_check"]
        now = time.time()
        if check["state"] == "failed":
            return self._reject_initial_check(record, provider, check["reason"])
        if self.ledger.frozen(self.wallet_id):
            return self._reject_initial_check(record, provider, "Wallet closed before initial health check passed")
        if now >= check["deadline_at"]:
            return self._reject_initial_check(record, provider, "Initial health check did not pass before its deadline")
        if not probe or now < check["next_check_at"]:
            return record
        provider.restore(result)
        # What this attempt saw, kept with the lease so a rejection can be read back: adapters' own
        # messages are credential-free; other exceptions are recorded by class only.
        attempt = {"attempts": check.get("attempts", 0) + 1, "first_check_at": check.get("first_check_at", now), "last_check_at": now}
        try:
            outcome = provider.initial_health_check(result["id"], probe_command(check.get("gpus")), PROBE_TIMEOUT)
            healthy = passed(outcome)
            if not healthy:
                attempt["last_outcome"] = "probe_failed"
                if isinstance(outcome, dict):
                    tail = str(outcome.get("stderr") or outcome.get("stdout") or "").strip().splitlines()[-1:]
                    attempt.update(last_exit_code=outcome.get("exit_code"), last_output=(tail[0] if tail else "")[:160])
        except ResourceDeadError as exc:
            # The provider says this machine will never start: destroy it now, not at the deadline.
            record = {**record, "result": {**result, "health_check": {**check, **attempt, "last_outcome": "dead"}}}
            return self._reject_initial_check(record, provider, f"Machine failed before its initial health check passed ({exc})")
        except Exception as exc:
            healthy = False  # Provider/SSH failures may be transient.
            attempt.update(last_outcome="not_ready" if isinstance(exc, ProviderError) else "error", last_error=(
                str(exc)[:160] if isinstance(exc, ProviderError) else type(exc).__name__))
        check = {**check, **attempt}
        record = {**record, "result": {**result, "health_check": check}}
        now = time.time()
        if now >= check["deadline_at"]:
            return self._reject_initial_check(record, provider, "Initial health check did not pass before its deadline")
        if healthy:
            passed_check = {**{k: v for k, v in check.items() if k != "next_check_at"}, "state": "passed", "checked_at": now,
                            "waited_seconds": round(now - check.get("started_at", record["created"]))}
            try:
                self.ledger.charge_compute(record["id"], {**result, "status": "running",
                    "charged_usd": record["maximum_usd"], "health_check": passed_check})
            except BudgetExceeded:
                return self._reject_initial_check(record, provider, "Wallet closed before initial health check passed")
            self._health_event(record, passed_check, "initial_health_check_passed")
        else:
            self.ledger.set_result(record["id"], {**result, "health_check": {**check, "next_check_at": now + 15}})
        return self.ledger.get(record["id"])

    def create(self, key, body):
        if set(body) - {"provider", "spec"}:
            raise ValueError("expected provider and spec only; prices belong to the host")
        name = body["provider"]
        provider = self._provider(name)
        spec = body["spec"]
        if not isinstance(spec, dict):
            raise ValueError("spec must be a JSON object")
        recorded = record_spec(spec)  # Build contexts are stored as digests, not archives.
        previous = self.ledger.find(self.wallet_id, "compute:" + key)
        if previous:
            if (previous["metadata"]["provider"], previous["metadata"]["spec"]) != (name, recorded):
                raise IdempotencyConflict("Request ID already used for a different reservation; use a new request ID")
            if not previous["result"]:
                raise Conflict("Provisioning unresolved; retry with the same request ID")
            return self._replayed(previous, provider)
        quote = provider.quote(spec)
        if self.wallet_expires_at and time.time() + quote["max_seconds"] > self.wallet_expires_at:
            raise ValueError("Reservation exceeds wallet expiry")
        if not quote.get("enforcement"):
            raise ValueError("provider lacks a configured enforceable spending/lifetime cap")
        static = self._static_price(provider, quote, spec)
        metadata = {"provider": name, "spec": recorded}
        if static:
            metadata["static"] = {k: v for k, v in static.items() if k != "max_cost_usd"}
        # The provider's price is the lease's real cost; a static wallet's budget holds its static price instead.
        record, created = self.ledger.reserve(self.wallet_id, "compute:" + key, quote["max_cost_usd"], "compute", metadata,
                                              static_usd=static and static["max_cost_usd"])
        if not created:
            if record["result"]:
                return self._replayed(record, provider)
            raise Conflict("Provisioning unresolved; retry with the same request ID")
        # Recovery must not race an in-flight create or overwrite its result.
        lock = self.ledger.resource_lock(record["id"])
        try:
            lock.__enter__()
        except BaseException:
            # The provider was never asked, so nothing can exist: release the hold now.
            self.ledger.reject_compute(record["id"])
            self.ledger.set_result(record["id"], {"id": None, "status": "rejected",
                                                  "reason": "Lease lock unavailable before provisioning; use a new request ID"})
            raise
        try:
            result = provider.create(spec, lease_id=record["id"])
            if provider.requires_initial_health_check(spec):
                self.ledger.set_result(record["id"], self._initial_check(record, result, quote))
            else:
                # The deadline remains host-owned even if the provider omits expires_at.
                self.ledger.charge_compute(record["id"], self._prepaid(record, result, quote))
            return self._public_resource(self.ledger.get(record["id"]))
        except ProviderRejectedError:
            self.ledger.reject_compute(record["id"])
            raise
        except BaseException:
            self.ledger.mark_uncertain(record["id"])
            raise
        finally:
            lock.__exit__(None, None, None)

    def _replayed(self, record, provider):
        """A same-key create retry returns the lease as it stands; a checking lease runs one probe first."""
        if record["result"].get("status") == "checking":
            with self.ledger.resource_lock(record["id"]):
                record = self._check_initial_health(record, provider)
        if record["result"].get("status") == "rejected":
            raise ProviderRejectedError(record["result"].get("reason") or "Provisioning was rejected; use a new request ID")
        return self._public_resource(record)

    def _owned(self, resource_id):
        record = self.ledger.get(resource_id, self.wallet_id)
        if record["kind"] != "compute":
            raise NotFound("No resource with that ID in this wallet")
        if not record["result"]:
            raise Conflict("Provisioning is unresolved for this resource")
        provider = self._provider(record["metadata"]["provider"])
        if record["result"].get("id"):
            provider.restore(record["result"])
        return record, provider

    def _persist_terminal(self, record, result):
        """Record a resource the provider confirmed has ended, so later calls fail fast without a provider call."""
        if record["state"] == "settled":
            self.ledger.set_result(record["id"], result)

    @staticmethod
    def _require_placed(record):
        if record["result"].get("status") == "checking":
            raise ProviderNotReadyError("Initial health check/cleanup is pending; poll compute status. Credits are reserved, not charged")

    @staticmethod
    def _require_active(record):
        result = record["result"]
        Broker._require_placed(record)
        if result.get("status") in TERMINAL or result.get("broker_expires_at", float("inf")) <= time.time():
            raise ResourceInactive("Resource is " + ("expired" if result.get("status") not in TERMINAL else result["status"])
                                   + "; create a new resource")

    def stop(self, resource_id):
        with self.ledger.resource_lock(resource_id):
            record, provider = self._owned(resource_id)
            try:
                self._recover_extensions(record, provider)
            except Exception:
                pass
            return self._stop_unlocked(resource_id)

    def _stop_unlocked(self, resource_id):
        record, provider = self._owned(resource_id)
        if record["result"].get("status") == "checking":
            return self._public_resource(self._reject_initial_check(record, provider, "Cancelled before initial health check passed"))
        if record["result"].get("status") in TERMINAL:
            return self._public_resource(record)
        try:
            merged = {**record["result"], **provider.stop(record["result"]["id"])}
            if record["state"] == "settled":
                self.ledger.set_result(resource_id, merged)
            else:
                # A recovered create that was never charged captures its full price too.
                self.ledger.charge_compute(resource_id, merged)
            return self._public_resource(self.ledger.get(resource_id))
        except BaseException:
            self.ledger.mark_uncertain(resource_id)
            raise

    def _pending_extensions(self, resource_id):
        return [r for r in self.ledger.reservations(self.wallet_id) if r["kind"] == "compute_extension"
                and r["metadata"]["resource_id"] == resource_id and r["state"] != "settled"]

    def _recover_extensions(self, record, provider):
        for payment in self._pending_extensions(record["id"]):
            recover = getattr(provider, "recover_extension", None)
            if recover:
                result = recover(record["result"]["id"], payment["metadata"]["plan"])
                if result:
                    self.ledger.charge_compute(payment["id"], self._static_extension(record, payment, result),
                                               resource_id=record["id"])

    def extend(self, key, body):
        if set(body) != {"id", "additional_seconds"} or type(body["additional_seconds"]) is not int or body["additional_seconds"] <= 0:
            raise ValueError("Expected id and positive integer additional_seconds")
        with self.ledger.resource_lock(body["id"]):
            record, provider = self._owned(body["id"])
            self._require_placed(record)
            previous = self.ledger.find(self.wallet_id, "compute_extend:" + key)
            if previous:
                if previous["metadata"]["request"] != body:
                    raise IdempotencyConflict("Request ID already used for a different extension; use a new request ID")
                if previous["state"] != "settled":
                    self._recover_extensions(record, provider)
                    previous = self.ledger.get(previous["id"])
                if previous["state"] != "settled":
                    raise Conflict("Extension unresolved; retry with the same request ID")
                if previous["result"].get("status") == "rejected":
                    raise ProviderRejectedError("Extension was rejected; use a new request ID")
                return self._public_resource(self.ledger.get(record["id"]))
            if self.ledger.frozen(self.wallet_id):
                raise BudgetExceeded("Wallet is closed")
            if self._pending_extensions(record["id"]):
                raise Conflict("An earlier extension is unresolved; retry its request ID")
            self._require_active(record)
            plan = provider.quote_extension(record["result"]["id"], body["additional_seconds"])
            if self.wallet_expires_at and plan["expires_at"] > self.wallet_expires_at:
                raise ValueError("Extension exceeds wallet expiry")
            static = record["metadata"].get("static")
            payment, _ = self.ledger.reserve(self.wallet_id, "compute_extend:" + key, plan["max_cost_usd"],
                "compute_extension", {"resource_id": record["id"], "request": body, "plan": plan},
                static_usd=static and static_cost(static["hourly_usd"], body["additional_seconds"]))
            try:
                result = provider.extend(record["result"]["id"], plan)
                self.ledger.charge_compute(payment["id"], self._static_extension(record, payment, result), resource_id=record["id"])
            except ProviderRejectedError:
                self.ledger.reject_compute(payment["id"])
                raise
            except BaseException:
                self.ledger.mark_uncertain(payment["id"])
                raise
            return self._public_resource(self.ledger.get(record["id"]))

    def reap(self):
        """One maintenance pass: health deadlines, lease deadlines, lost creates and unconfirmed model calls.

        Deadlines are re-read under the lease lock; a busy lease is revisited next tick.
        """
        results = []
        for snapshot in self.ledger.reservations(self.wallet_id):
            if snapshot["kind"] == "openrouter" and snapshot["state"] != "settled" and self.openrouter:
                try:
                    cost = self.openrouter.reconcile(snapshot)
                    if cost is not None:
                        results.append({"id": snapshot["id"], "status": "settled", "cost_usd": cost})
                except Exception as exc:
                    logging.warning("Reconciling model call %s failed: %s", snapshot["id"], type(exc).__name__)
                continue
            if snapshot["kind"] != "compute":
                continue
            result, now = snapshot["result"], time.time()
            if not result and snapshot["state"] == "reserved" and now - snapshot["created"] < CREATE_IN_FLIGHT_SECONDS:
                continue  # Its create is still talking to the provider under the lease lock.
            if result and result.get("status") in TERMINAL:
                continue
            try:
                frozen = self.ledger.frozen(self.wallet_id)
                health = (result or {}).get("health_check", {})
                health_due = result and result.get("status") == "checking" and (
                    health.get("state") == "failed" or health.get("deadline_at", float("inf")) <= now)
                due = (not result or health_due or frozen
                       or min(result.get("broker_expires_at", float("inf")), self.wallet_expires_at or float("inf")) <= now)
                with self._lease_lock(snapshot["id"], wait=due) as locked:
                    if not locked:
                        results.append({"id": snapshot["id"], "status": "busy"})
                        continue
                    results.extend(self._reap_lease(snapshot["id"]))
            except Exception as exc:
                logging.warning("Reconciling lease %s failed: %s", snapshot["id"], type(exc).__name__)
                results.append({"id": snapshot["id"], "status": "termination_uncertain" if snapshot["result"]
                                else "needs_provisioning_reconciliation"})
        return results

    def _reap_lease(self, lease_id):
        """The reaper's turn at one lease, under its lock."""
        record = self.ledger.get(lease_id)
        resource = record["result"]
        provider = self._provider(record["metadata"]["provider"])
        if not resource:
            resource = provider.recover(lease_id=record["id"], spec=record["metadata"]["spec"],
                                        created_at=record["created"], maximum_usd=record["maximum_usd"])
            if not resource:
                if getattr(provider, "absence_is_final", False) and time.time() - record["created"] > RECOVERY_GRACE_SECONDS:
                    # Long enough for any in-flight create to have shown up: it never happened.
                    self.ledger.reject_compute(record["id"])
                    return [{"id": record["id"], "status": "rejected", "reason": "create never took effect"}]
                return [{"id": record["id"], "status": "needs_provisioning_reconciliation"}]
            # The machine as the provider found it, so a recovered GPU lease still gets the GPU probe.
            q = {"max_seconds": record["metadata"]["spec"]["max_seconds"],
                 **{k: resource[k] for k in ("gpu", "gpus", "cpus") if k in resource}}
            if provider.requires_initial_health_check(record["metadata"]["spec"]):
                self.ledger.set_result(record["id"], self._initial_check(record, resource, q))
            else:
                self.ledger.charge_compute(record["id"], self._prepaid(record, resource, q))
            # A same-key retry now returns the recovered resource; a gated one still holds its funds.
            return [{"id": record["id"], "status": "recovered"}]
        if resource.get("status") == "checking":
            checked = self._check_initial_health(record, provider, probe=False)
            return [{"id": record["id"], "status": checked["result"]["status"]}]
        if resource.get("status") in TERMINAL:
            return []
        record, provider = self._owned(record["id"])
        # Resolve a lost extension response before evaluating the deadline; its failure must not block cleanup.
        try:
            self._recover_extensions(record, provider)
        except Exception:
            pass
        resource = self.ledger.get(record["id"])["result"]
        deadline = min(resource.get("broker_expires_at", float("inf")), self.wallet_expires_at or float("inf"))
        if deadline <= time.time() or self.ledger.frozen(self.wallet_id):
            return [self._stop_unlocked(record["id"])]
        return []

    @contextmanager
    def _lease_lock(self, resource_id, *, wait):
        """Never blocking for long: a lease with work due keeps trying for lease_lock_wait seconds."""
        deadline = time.monotonic() + (self.lease_lock_wait if wait else 0)
        while True:
            with self.ledger.try_resource_lock(resource_id) as locked:
                if locked or time.monotonic() >= deadline:
                    yield locked
                    return
            time.sleep(min(0.25, self.lease_lock_wait))

    def close(self):
        """Freeze spending, stop every resource, revoke tokens and clean storage; reports whether cleanup is complete."""
        self.ledger.freeze(self.wallet_id)
        results = []
        for record in self.ledger.reservations(self.wallet_id):
            if record["kind"] == "compute" and (not record["result"] or record["result"].get("status") != "rejected"):
                try:
                    with self.ledger.resource_lock(record["id"]):
                        current, provider = self._owned(record["id"])
                        try:
                            self._recover_extensions(current, provider)
                        except Exception:
                            pass
                        results.append(self._stop_unlocked(record["id"]))
                except Exception as exc:
                    if not isinstance(exc, Conflict):  # An unresolved create waits for the reaper's recovery.
                        logging.warning("Stopping lease %s failed: %s", record["id"], type(exc).__name__)
                    results.append({"id": record["id"], "status": "requires_operator_reconciliation"})
        self.ledger.revoke_tokens(self.wallet_id)
        errors = [r for r in results if r.get("status") not in TERMINAL]
        storage = cleanup_storage(self.ledger, self.storage, self.wallet_id, force=True)
        errors.extend(r for r in storage if r["status"] != "storage_cleaned")
        status = self.ledger.status(self.wallet_id)
        static = status.pop("static", None)
        # The operator pays the real cost; the wallet's own view may be in static prices.
        return {"resources": results, "storage": storage, "budget": {**status, **static, "pricing": "static"} if static else status,
                "settlement": {"spent_usd": status["spent_usd"], "reserved_usd": status["reserved_usd"]},
                "cleanup_complete": not errors and Decimal(status["reserved_usd"]) == 0, "errors": errors}

    def offers(self):
        offers = market_offers(self.providers)[0]
        return self.static.offers(offers) if self.static else offers

    def dispatch(self, method, path, body=None, key=None):
        result = self._dispatch(method, path, body or {}, key)
        # Model routes keep OpenAI's response format untouched.
        return result if path.startswith("/v1/") else present(result)

    def _dispatch(self, method, path, body, key):
        if method == "GET" and path == "/budget":
            return self.budget()
        if method == "GET" and path == "/compute/offers":
            return {"offers": self.offers()}
        if method == "GET" and path == "/compute/list":
            return {"resources": [self._public_resource(r) for r in self.ledger.reservations(self.wallet_id) if r["kind"] == "compute"]}
        if method == "POST" and path in {"/compute/create", "/compute/extend"}:
            if not key:
                raise ValueError("Idempotency-Key required")
            return self.create(key, body) if path == "/compute/create" else self.extend(key, body)
        if method == "POST" and path in {"/compute/stop", "/compute/exec", "/compute/ssh", "/compute/status",
                                         "/compute/upload", "/compute/download"}:
            if "id" not in body:
                raise KeyError("id")
            if path == "/compute/stop":
                return self.stop(body["id"])
            record, provider = self._owned(body["id"])
            if self.ledger.frozen(self.wallet_id):
                raise BudgetExceeded("Wallet is closed")
            if path in {"/compute/upload", "/compute/download"}:
                return self._transfer(path, record, provider, body)
            if record["result"].get("status") == "checking":
                if path != "/compute/status":
                    self._require_placed(record)
                with self.ledger.resource_lock(record["id"]):
                    return self._public_resource(self._check_initial_health(record, provider))
            provider_id = record["result"]["id"]
            if path != "/compute/status":
                self._require_active(record)
            elif not provider_id:
                return self._public_resource(record)  # Rejected before a machine existed: nothing to ask the provider.
            # Commands and reads run without the lease lock; a stop or reap landing mid-command only makes it fail.
            if path == "/compute/exec":
                timeout = body.get("timeout", 300)
                if not isinstance(timeout, int) or isinstance(timeout, bool) or not 1 <= timeout <= 3600:
                    raise ValueError("timeout must be 1..3600 seconds")
                if not isinstance(body.get("command"), str):
                    raise KeyError("command")
                try:
                    return provider.exec(provider_id, body["command"], timeout=timeout)
                except ResourceGoneError as exc:
                    self._persist_terminal(record, {**record["result"], "status": exc.status})
                    raise
            if path == "/compute/ssh":
                access = provider.ssh(provider_id)
                if access.get("supported") is False:
                    raise NotSupported(access.get("reason") or "SSH is not available for this resource")
                return access
            merged = {**record["result"], **provider.status(provider_id)}
            # Adapters whose lookups cannot mistake a live resource for a missing one opt in; a persisted
            # terminal status also stops the reaper.
            if merged.get("status") in TERMINAL and getattr(provider, "status_is_final", False):
                self._persist_terminal(record, merged)
            return self._public_resource({**record, "result": merged})
        if method == "POST" and path == "/transfer/credentials":
            return self._credentials(body)
        if method == "POST" and path == "/openrouter/key":
            self._require_openrouter()
            if self.ledger.frozen(self.wallet_id):
                raise BudgetExceeded("Wallet is closed")
            token = "spin_or_" + secrets.token_urlsafe(32)
            self.ledger.issue_token(self.wallet_id, token, "openrouter")
            return {"key": token, "base_url": self.public_url + "/v1",
                    "note": "Wallet-scoped key; upstream OpenRouter credentials are never disclosed."}
        if method == "GET" and path == "/v1/models":
            self._require_openrouter()
            return {"object": "list", "data": [{"id": model, "object": "model"} for model in self.openrouter.models]}
        if method == "GET" and path.startswith("/v1/models/"):
            self._require_openrouter()
            return self.openrouter.endpoint_listing(path[len("/v1/models/"):])
        if method == "POST" and path == "/v1/chat/completions":
            self._require_openrouter()
            if not key:
                raise ValueError("Idempotency-Key required for paid requests")
            return self.openrouter.complete(self.wallet_id, key, body)
        if method == "POST" and path == "/run/close":
            return self.close()
        raise NotFound("Unknown endpoint")

    def _transfer(self, path, record, provider, body):
        self._require_active(record)
        provider_id = record["result"]["id"]
        try:
            if path == "/compute/upload":
                if not isinstance(body.get("path"), str):
                    raise KeyError("path")
                data = base64.b64decode(body["content_base64"], validate=True)
                if len(data) > INLINE_MAX_BYTES:
                    raise ValueError("inline upload exceeds 512 KiB; use spin s5cmd with temporary S3 credentials for larger files")
                return provider.write_stream(provider_id, body["path"], io.BytesIO(data))
            remote = body.get("path")
            if not isinstance(remote, str) or not remote.startswith("/"):
                raise ValueError("path must be an absolute file path in the sandbox")
            with provider.read_stream(provider_id, remote) as stream:
                counted = files.Counting(stream, INLINE_MAX_BYTES)
                try:
                    data = b"".join(iter(lambda: counted.read(files.CHUNK), b""))
                except ValueError:
                    raise ValueError("File is larger than 512 KiB; use spin s5cmd with temporary S3 credentials for larger files") from None
            return {"path": remote, "size_bytes": len(data), "sha256": counted.sha256.hexdigest(),
                    "content_base64": base64.b64encode(data).decode()}
        except ResourceGoneError as exc:
            self._persist_terminal(record, {**record["result"], "status": exc.status})
            raise

    def _credentials(self, body):
        if not self.storage:
            raise NotConfigured("Object transfers are not configured on this Spin installation")
        if self.ledger.frozen(self.wallet_id):
            raise BudgetExceeded("Wallet is closed")
        if set(body) - {"prefix", "permission", "expires_seconds"}:
            raise ValueError("unsupported transfer fields")
        session = self.storage.credentials(self.wallet_id, prefix=body.get("prefix", ""),
            permission=body.get("permission", "read-write"), expires_seconds=body.get("expires_seconds", DEFAULT_TTL),
            wallet_expires_at=self.wallet_expires_at)
        self.ledger.register_transfer_scope(self.wallet_id, {"endpoint_url": session["endpoint_url"], "bucket": session["bucket"],
            "key_prefix": self.storage.wallet_prefix(self.wallet_id), "expires_at": session["expires_at"]})
        return session

    def _require_openrouter(self):
        if not self.openrouter:
            raise NotConfigured("Inference (OpenRouter) is not configured on this Spin installation")
