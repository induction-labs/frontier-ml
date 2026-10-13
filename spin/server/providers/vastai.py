"""Vast.ai rentals behind the broker; provider keys stay on the trusted host.

Vast has no documented per-instance TTL or spending cap. The broker's deadline destroys rentals (the reaper
retries until Vast confirms), and only verified on-demand offers are rented. Leases prepay the hourly price
plus a bandwidth allowance at the host's per-GB rate. A stopped rental still incurs storage charges: stop()
always destroys it. Wallet charges never exceed the quote; the operator covers provider overruns.
"""
from decimal import Decimal, InvalidOperation, ROUND_CEILING
import hashlib
import hmac
import io
import ipaddress
import json
import math
import shlex
import socket
import threading
import time
from contextlib import contextmanager

import httpx

from . import files
from .base import (OwnedProvider, ProviderError, ProviderNotReadyError, ProviderRejectedError,
                   ProviderUncertainError, ResourceDeadError, duration, lease_name, quote)
from .modal import IMAGE_REF

# Vast's SSH launch mode adds sshd to images that lack it; this one also has CUDA and Python.
DEFAULT_IMAGE = "vastai/pytorch:cuda-12.8.1-auto"
# Same cap as every other provider; on-demand rentals run until Spin destroys them.
MAX_SECONDS = 86400
# actual_status values a rental never leaves on its own. Everything else that is not "running"
# (None, created, scheduling, loading, starting, unknown) is a transient state worth polling.
DEAD_STATES = frozenset({"exited", "stopped", "offline"})


def amount(value, *, positive=False):
    try:
        result = Decimal(str(value))
    except InvalidOperation as exc:
        raise ValueError("Invalid Vast price") from exc
    if not result.is_finite() or result < 0 or (positive and result == 0):
        raise ValueError("Vast prices must be finite and nonnegative")
    return result


def epoch(value):
    """A Vast epoch-seconds field, or None when it is missing or malformed."""
    if isinstance(value, bool) or not isinstance(value, (int, float, str)):
        return None
    try:
        result = float(value)
    except ValueError:
        return None
    return result if math.isfinite(result) and result > 0 else None


def validate_config(config):
    if config.get("operator_covers_overrun") is not True:
        raise ValueError("Vast requires operator_covers_overrun=true: broker expiry is not a provider billing cap")
    image = config.get("image", DEFAULT_IMAGE)
    if not isinstance(image, str) or not IMAGE_REF.fullmatch(image) or image.startswith("YOUR_"):
        raise ValueError("Vast image must be a container registry reference")
    if config.get("max_hourly_usd") is not None:  # Optional ceiling; offers are priced individually.
        amount(config["max_hourly_usd"], positive=True)
    for key, default, limit in (("max_disk_gb", 100, 10000), ("default_disk_gb", 20, 10000),
                                ("max_seconds", MAX_SECONDS, MAX_SECONDS), ("offer_limit", 1000, 5000),
                                ("offers_per_gpu", 3, 100), ("bandwidth_gb_per_hour", 10, 10000)):
        value = config.get(key, default)
        if type(value) is not int or not 1 <= value <= limit:
            raise ValueError("Invalid Vast " + key)
    if config.get("default_disk_gb", 20) > config.get("max_disk_gb", 100):
        raise ValueError("Vast default disk exceeds its limit")
    reliability = amount(config.get("min_reliability", "0.99"))
    if reliability > 1:
        raise ValueError("Vast reliability must be in [0, 1]")


class VastAIProvider(OwnedProvider):
    provider = "vastai"
    # The label is sent in the create request, so a lookup that finds no instance
    # with it (after a grace period) proves the create did not happen.
    absence_is_final = True
    supports_extension = True

    def __init__(self, client, *, config, ssh_secret, ssh_transport=None, clock=None, sleep=time.sleep):
        """config: image, max_hourly_usd, operator_covers_overrun and optional disk/time/offer limits."""
        super().__init__(**({"clock": clock} if clock else {}))
        validate_config(config)
        if len(ssh_secret) < 32:
            raise ValueError("Vast needs a stable, private 32+ character SSH derivation secret")
        self.client, self.config, self.ssh_secret, self.sleep = client, config, ssh_secret, sleep
        self.ssh_transport = ssh_transport or VastSSH()
        self._listing, self._seen = None, {}
        self._key_attached_at = {}  # In memory; unknown after a restart.

    def _request(self, method, path, *, body=None, params=None, retries=3):
        # Paths are under /api/v0 unless given in full (e.g. /api/v1/instances/).
        url = path if path.startswith("/api/") else "/api/v0" + path
        for attempt in range(retries + 1):
            response = self.client.request(method, url, json=body, params=params, timeout=30)
            if response.status_code != 429 or attempt == retries:
                break
            time.sleep(2 ** attempt * 3)  # Marketplace searches are rate-limited to a few per minute.
        if response.status_code == 429:
            raise ProviderRejectedError("Vast is rate-limiting requests; retry in a minute")
        response.raise_for_status()
        result = response.json()
        if not isinstance(result, dict) or result.get("success") is False or result.get("error"):
            raise ProviderError("Vast returned an invalid/error response")
        return result

    @staticmethod
    def _label(lease_id):
        return lease_name(lease_id)

    def _spec(self, spec):
        if not isinstance(spec, dict) or set(spec) - {"offer_id", "disk_gb", "max_seconds", "image"}:
            raise ValueError("Vast spec accepts offer_id, disk_gb, max_seconds, image")
        offer = spec.get("offer_id")
        # Offers list IDs as strings like every provider; Vast's API wants integers.
        if isinstance(offer, str) and offer.isascii() and offer.isdigit():
            offer = int(offer)
        if type(offer) is not int or offer < 1:
            raise ValueError("Vast offer_id must be an offer ID from compute offers")
        disk = spec.get("disk_gb", self.config.get("default_disk_gb", 20))
        if type(disk) is not int or not 1 <= disk <= self.config.get("max_disk_gb", 100):
            raise ValueError("Vast disk_gb exceeds the configured limit")
        image = spec.get("image")
        if image is not None and not (isinstance(image, str) and IMAGE_REF.fullmatch(image)):
            raise ValueError("image must be a container registry reference, e.g. pytorch/pytorch:latest")
        return {"offer_id": offer, "disk_gb": disk,
                "max_seconds": duration(spec, self.config.get("max_seconds", MAX_SECONDS)),
                **({"image": image} if image else {})}

    def quote(self, spec):
        # Priced from the offer's current listing. Retries of a created lease never
        # re-quote, and the operator covers any price move before the rental starts.
        s = self._spec(spec)
        seen = self._seen.get(s["offer_id"])
        # An offer listed in the last minute is quoted from that listing, saving a rate-limited search.
        offers = ([seen[1]] if seen and self.clock() - seen[0] < 60
                  else self._search(s["disk_gb"], offer_id=s["offer_id"], seconds=s["max_seconds"]))
        try:
            if len(offers) != 1 or offers[0].get("id") != s["offer_id"]:
                raise ValueError("offer unavailable")
            rate = self._offer(offers[0], s["disk_gb"], s["max_seconds"])
        except (ValueError, TypeError):
            raise ValueError("Vast offer is unavailable or too short for max_seconds; choose another from compute offers") from None
        q = quote(self.provider, s, rate, "broker_deadline_operator_backstop")
        # Bandwidth (including the image pull) bills per GB, outside the hourly price, and
        # Vast reports no per-instance usage to settle against: prepay a pro-rated allowance.
        per_gb, allowance = self._bandwidth(offers[0]), self._allowance_gb(s["max_seconds"])
        q["max_cost_usd"] = str((Decimal(q["max_cost_usd"]) + per_gb * allowance).quantize(Decimal("0.000001"), rounding=ROUND_CEILING))
        q.update(bandwidth_usd_per_gb=str(per_gb), bandwidth_allowance_gb=str(allowance), provider_billing_cap=False,
                 # The listing keeps a few hosts per GPU, so say what this offer is for static wallet prices.
                 gpu=offers[0].get("gpu_name"), gpus=offers[0].get("num_gpus"))
        return q

    def _allowance_gb(self, seconds):
        return max(Decimal(1), Decimal(self.config.get("bandwidth_gb_per_hour", 10)) * seconds / 3600).quantize(Decimal("0.01"))

    @staticmethod
    def _bandwidth(offer):
        return max(amount(offer.get("inet_up_cost") or 0), amount(offer.get("inet_down_cost") or 0))

    def _search(self, disk, *, offer_id=None, seconds=1):
        query = {"type": "on-demand", "verified": {"eq": True},
                 "rentable": {"eq": True}, "rented": {"eq": False},
                 "reliability": {"gte": float(self.config.get("min_reliability", "0.99"))},
                 "disk_space": {"gte": disk}, "duration": {"gte": seconds},
                 # Hosts whose driver runs the default image's CUDA.
                 "cuda_max_good": {"gte": float(self.config.get("min_cuda", 12.8))},
                 "allocated_storage": disk, "order": [["dph_total", "asc"]],
                 "limit": 1 if offer_id else self.config.get("offer_limit", 1000)}
        if self.config.get("max_hourly_usd") is not None:
            query["dph_total"] = {"lte": float(self.config["max_hourly_usd"])}
        if offer_id:
            query["ask_contract_id"] = {"eq": offer_id}  # Vast ignores filters on "id".
        data = self._request("POST", "/bundles/", body=query).get("offers")
        if not isinstance(data, list):
            raise ProviderError("Vast offer response is not a list")
        return data

    def _offer(self, offer, disk, seconds):
        # Check the response too; filtering alone is not a billing guarantee.
        if not isinstance(offer, dict) or type(offer.get("id")) is not int or offer["id"] < 1:
            raise ValueError("Invalid Vast offer")
        verified = offer.get("verified") is True or offer.get("verification") == "verified"
        if not verified or offer.get("rentable") is not True or offer.get("rented") is not False or offer.get("is_bid") is not False:
            raise ValueError("Vast offer must be verified, available, and on-demand")
        if self.config.get("max_bandwidth_usd_per_gb") is not None and self._bandwidth(offer) > amount(self.config["max_bandwidth_usd_per_gb"]):
            raise ValueError("Vast offer bandwidth price exceeds policy")
        if amount(offer.get("reliability", offer.get("reliability2"))) < amount(self.config.get("min_reliability", "0.99")):
            raise ValueError("Vast offer reliability below policy")
        if amount(offer.get("disk_space")) < disk or amount(offer.get("duration")) < seconds:
            raise ValueError("Vast offer is too small or expires too soon")
        # dph_total includes allocated storage. Also check the storage component
        # explicitly using a conservative 28-day month conversion.
        total = amount(offer.get("dph_total"), positive=True)
        storage = amount(offer.get("storage_cost")) * disk / (28 * 24)
        hourly = max(total, amount(offer.get("dph_base"), positive=True) + storage).quantize(
            Decimal("0.000001"), rounding=ROUND_CEILING)
        if self.config.get("max_hourly_usd") is not None and hourly > amount(self.config["max_hourly_usd"], positive=True):
            raise ValueError("Vast offer exceeds the hourly ceiling")
        return hourly

    def offers(self):
        # Listing is cached briefly: Vast rate-limits marketplace searches.
        if self._listing and self.clock() - self._listing[0] < 60:
            return self._listing[1]
        self._listing = (self.clock(), self._list_offers())
        return self._listing[1]

    def _list_offers(self):
        disk = self.config.get("default_disk_gb", 20)
        # Listings are qualified for a short rental so that few hosts are excluded on
        # duration; a lease may ask for more, up to the offer's available_seconds.
        seconds = min(600, self.config.get("max_seconds", MAX_SECONDS))
        result, per_gpu = [], {}
        now = self.clock()
        for offer in self._search(disk, seconds=seconds):  # Cheapest first.
            if isinstance(offer, dict) and type(offer.get("id")) is int:
                self._seen[offer["id"]] = (now, offer)
            try:
                rate = self._offer(offer, disk, seconds)
            except (ValueError, TypeError):
                continue
            # Every GPU model and count, a few of the cheapest hosts each.
            shape = (offer.get("gpu_name"), offer.get("num_gpus"))
            per_gpu[shape] = per_gpu.get(shape, 0) + 1
            if per_gpu[shape] > self.config.get("offers_per_gpu", 3):
                continue
            result.append({"id": str(offer["id"]), "provider": self.provider,
                           "gpu": offer.get("gpu_name"), "gpus": offer.get("num_gpus"), "hourly_usd": str(rate),
                           "spec": {"offer_id": str(offer["id"]), "disk_gb": disk, "max_seconds": seconds},
                           "limits": {"max_seconds": self.config.get("max_seconds", MAX_SECONDS),
                                      "disk_gb": self.config.get("max_disk_gb", 100), "listed_for_seconds": seconds},
                           "gpu_ram_mb": offer.get("gpu_ram"), "location": (offer.get("geolocation") or "").strip(" ,") or None,
                           "cpus": offer.get("cpu_cores_effective") and int(offer["cpu_cores_effective"]),
                           "memory_gib": offer.get("cpu_ram") and round(offer["cpu_ram"] / 1024),
                           "bandwidth_usd_per_gb": str(self._bandwidth(offer)),
                           "bandwidth_gb_per_hour": self.config.get("bandwidth_gb_per_hour", 10),
                           "available_seconds": int(amount(offer.get("duration", 0))), "custom_image": True,
                           "supports_extension": True, "enforcement": "broker_deadline_operator_backstop"})
        return result

    def create(self, spec, *, lease_id):
        self._spec(spec)  # Invalid input stays a ValueError; only a vanished offer is a rejection.
        try:
            q = self.quote(spec)
        except ValueError as exc:
            raise ProviderRejectedError(str(exc) + "; no instance was created") from exc
        s = q["spec"]
        rate = q["hourly_usd"]
        label, started = self._label(lease_id), self.clock()
        try:
            result = self._request("PUT", f'/asks/{s["offer_id"]}/', body={
                "client_id": "me", "image": s.get("image") or self.config.get("image", DEFAULT_IMAGE), "disk": s["disk_gb"],
                "label": label, "runtype": "ssh", "target_state": "running",
                "cancel_unavail": True, "env": {}})
            instance_id = result.get("new_contract")
            if result.get("success") is not True or type(instance_id) is not int or instance_id < 1:
                raise ProviderError("Vast did not confirm the instance ID")
            return self._created(str(instance_id), lease_id, q, started, label=label,
                                 hourly_usd=str(rate), status="provisioning", disk_gb=s["disk_gb"])
        except ProviderRejectedError:
            raise  # Vast refused the request (e.g. rate-limited after retries): nothing was created.
        except httpx.HTTPStatusError as exc:
            code = exc.response.status_code
            if 400 <= code < 500 and code != 408:
                # A refused request creates nothing; Vast reports a vanished offer as 400/404 too.
                try:
                    error = exc.response.json()
                except ValueError:
                    error = {}
                if error.get("error") in {"no_such_ask", "invalid_args"}:
                    raise ProviderRejectedError("Vast offer expired; no instance was created") from exc
                raise ProviderRejectedError(f"Vast refused the rental (HTTP {code}); no instance was created") from exc
            raise ProviderUncertainError("Vast create uncertain; reconcile lease label " + label) from exc
        except Exception as exc:
            raise ProviderUncertainError("Vast create uncertain; reconcile lease label " + label) from exc

    def recover(self, *, lease_id, spec, created_at, maximum_usd):
        """Recover a lost create response by exact persisted lease label, never re-rent."""
        label = self._label(lease_id)
        # Match the label here: Vast silently ignores some list filters.
        instances = [i for i in self._listed_instances() if isinstance(i, dict) and i.get("label") == label]
        if len(instances) > 1:
            raise ProviderError("Ambiguous Vast lease lookup; operator reconciliation required")
        if not instances:
            return None  # The broker releases the hold once absence has lasted past a grace period.
        found = instances[0]
        if found.get("label") != label or type(found.get("id")) is not int or found["id"] < 1:
            raise ProviderError("Vast lookup did not match the lease")
        q = {"max_seconds": spec["max_seconds"], "max_cost_usd": maximum_usd}
        machine = {"gpu": found.get("gpu_name"), "gpus": found["num_gpus"]} if type(found.get("num_gpus")) is int else {}
        return self._created(str(found["id"]), lease_id, q, created_at, label=label,
                             hourly_usd=str(amount(maximum_usd) * 3600 / spec["max_seconds"]),
                             status="provisioning", recovered=True, **machine)

    def _listed_instances(self):
        """The account's instances, shared for 30 s: one lookup serves every lease being
        reconciled, instead of one rate-limited request each."""
        cached = getattr(self, "_instances_cache", None)
        if cached and self.clock() - cached[0] < 30:
            return cached[1]
        # Vast retired GET /api/v0/instances/ (410 deprecated_endpoint) in favor of v1.
        # v1 paginates even with no filters. Absence is billing evidence, so never
        # treat a partial or failed listing as proof that a rental does not exist.
        params = {"limit": 25, "order_by": json.dumps([{"col": "id", "dir": "asc"}])}
        listed, seen_tokens = [], set()
        for _ in range(250):
            page = self._request("GET", "/api/v1/instances/", params=params)
            rows = page.get("instances")
            if not isinstance(rows, list) or any(not isinstance(row, dict) or type(row.get("id")) is not int
                                                or row["id"] < 1 for row in rows):
                raise ProviderError("Vast instance list is malformed")
            listed.extend(rows)
            token = page.get("next_token")
            if token is None or token == "":
                break
            if not isinstance(token, str) or token in seen_tokens:
                raise ProviderError("Vast instance pagination is malformed")
            seen_tokens.add(token)
            params["after_token"] = token
        else:
            raise ProviderError("Vast instance pagination did not finish; retry reconciliation")
        self._instances_cache = (self.clock(), listed)
        return listed

    def _instance(self, resource_id, *, active=False):
        record = self._record(resource_id, active=active)
        result = self._request("GET", f"/instances/{int(resource_id)}/").get("instances")
        if not isinstance(result, dict) or str(result.get("id")) != resource_id or result.get("label") != record["label"]:
            raise ProviderError("Vast instance identity does not match the owned lease")
        return record, result

    def status(self, resource_id):
        r, instance = self._instance(resource_id)
        r["status"] = str(instance.get("actual_status") or "unknown")
        return dict(r)  # Do not expose provider account metadata or environment.

    def _check_extension(self, resource_id, plan):
        r, instance = self._instance(resource_id)
        if instance.get("actual_status") != "running":
            raise ProviderRejectedError("Only running Vast rentals can be extended")
        end = amount(instance.get("end_date"), positive=True)
        if end < Decimal(str(plan["expires_at"])):
            raise ProviderRejectedError("Vast host availability ends before the requested deadline")
        if plan["expires_at"] <= self.clock():
            raise ProviderRejectedError("The requested extension has expired")

    def quote_extension(self, resource_id, additional_seconds):
        plan = self._extension_quote(resource_id, additional_seconds, self.config.get("max_seconds", MAX_SECONDS))
        self._check_extension(resource_id, plan)
        return plan

    def extend(self, resource_id, plan):
        # On-demand rentals keep running until destroyed. Only our persisted
        # deadline changes; no new rental or provider mutation is needed.
        return self._extended(resource_id, plan)

    def recover_extension(self, resource_id, plan):
        self._check_extension(resource_id, plan)
        return self._extended(resource_id, plan)

    def stop(self, resource_id):
        r = self._record(resource_id)
        if "settled_cost_usd" in r:
            return dict(r)
        try:
            result = self._request("DELETE", f"/instances/{int(resource_id)}/")
            if result.get("success") is not True:
                raise ProviderError("Vast did not confirm destruction")
        except httpx.HTTPStatusError as exc:
            try:
                absent = exc.response.status_code == 404 and exc.response.json().get("error") == "not_found"
            except ValueError:
                absent = False
            if not absent:
                raise ProviderUncertainError("Vast destruction uncertain; retain reservation") from exc
        except Exception as exc:
            raise ProviderUncertainError("Vast destruction uncertain; retain reservation") from exc
        return self._stopped(r, "deleted")

    def _keys(self, lease_id):
        """Deterministic per-lease Ed25519 key: survives restarts without being stored."""
        from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
        from cryptography.hazmat.primitives import serialization
        seed = hmac.new(self.ssh_secret.encode(), ("vast-ssh:" + lease_id).encode(), hashlib.sha256).digest()
        key = Ed25519PrivateKey.from_private_bytes(seed)
        public = key.public_key().public_bytes(serialization.Encoding.OpenSSH, serialization.PublicFormat.OpenSSH).decode()
        private = key.private_bytes(serialization.Encoding.PEM, serialization.PrivateFormat.OpenSSH,
                                    serialization.NoEncryption()).decode()
        return public, private

    def ssh(self, resource_id):
        r, instance = self._instance(resource_id, active=True)
        status = instance.get("actual_status")
        if status in DEAD_STATES:
            # Polling a dead rental only runs down its prepaid hour; the record stays non-terminal so
            # that compute stop still destroys the rental (a stopped one keeps billing storage).
            raise ResourceDeadError(f"Vast instance is {status} and will not resume; run compute stop "
                                    "(destroys it and ends storage billing) and create a new resource")
        if status != "running":
            # Image pulls take minutes; a loading rental is a normal state, not a failure.
            raise ProviderNotReadyError(f"Vast instance is {status or 'not running yet'}; poll compute status")
        host, port = instance.get("ssh_host"), instance.get("ssh_port")
        if not isinstance(host, str) or not host or any(c not in "abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-:" for c in host):
            raise ProviderError("Invalid Vast SSH host")
        if type(port) is not int or not 1 <= port <= 65535:
            raise ProviderNotReadyError("Vast SSH connection is not ready; poll compute status")
        public, private = self._keys(r["lease_id"])
        # Every ssh call returns the same key; attach it once, and treat a refused
        # repeat attach as success when the key is already on the instance.
        attached_now = False
        if not self._key_attached(resource_id, public):
            attached_now = self._attach_key(resource_id, public)
            if not attached_now:
                # Vast occasionally refuses the first attach; look again, then try once more.
                self.sleep(3)
                attached_now = self._key_attached(resource_id, public) or self._attach_key(resource_id, public)
                if not attached_now:
                    raise ProviderError("Vast did not attach the lease SSH key; retry shortly")
        if attached_now:
            self._key_attached_at[resource_id] = self.clock()
        started_at = epoch(instance.get("start_date"))
        filename = "spin-vast-" + resource_id + ".key"
        # key_attached_now / instance_started_at / key_attached_at anchor the SSH retry
        # window to readiness: the key reaches the container 20-60 s after it is attached.
        return {"supported": True, "host": host, "port": port, "user": "root",
                "provider": self.provider, "created_at": r["created_at"], "connection_status": "unverified",
                "key_attached_now": attached_now,
                **({"instance_started_at": started_at} if started_at is not None else {}),
                **({"key_attached_at": self._key_attached_at[resource_id]} if resource_id in self._key_attached_at else {}),
                "private_key": private, "identity_file": filename, "expires_at": r["expires_at"],
                "ssh_command": f"ssh -o IdentitiesOnly=yes -i {shlex.quote(filename)} -p {port} root@{shlex.quote(host)}",
                "note": "Save private_key to identity_file with mode 0600. This key grants access only to this rental. "
                        "Running status and SSH credentials do not confirm SSH readiness; verify with compute exec."}

    def _key_attached(self, resource_id, public):
        try:
            keys = self._request("GET", f"/instances/{int(resource_id)}/ssh/")
        except Exception:
            return False
        return public.split()[1] in json.dumps(keys)  # Compare key material, whatever the field names.

    def _attach_key(self, resource_id, public):
        try:
            result = self._request("POST", f"/instances/{int(resource_id)}/ssh/", body={"ssh_key": public})
            return result.get("success") is True
        except (httpx.HTTPStatusError, ProviderError):
            return False

    def exec(self, resource_id, command, timeout=60):
        timeout = self._timeout(resource_id, timeout)
        if not isinstance(command, str) or not command or len(command) > 65536:
            raise ValueError("Command must be 1..65536 characters")
        # Vast's images keep Python and PyTorch in /venv/main, which only interactive shells activate.
        activate = "[ -f /venv/main/bin/activate ] && . /venv/main/bin/activate >/dev/null 2>&1; "
        return self.ssh_transport.exec(self.ssh(resource_id), activate + command, timeout)

    @staticmethod
    def _absolute(path):
        if not isinstance(path, str) or not path.startswith("/") or "\x00" in path:
            raise ValueError("Use an absolute remote path")
        return path

    def read_stream(self, resource_id, path):
        return self.ssh_transport.open(self.ssh(resource_id), self._absolute(path), "rb")

    def write_stream(self, resource_id, path, stream):
        with self.ssh_transport.open(self.ssh(resource_id), self._absolute(path), "wb") as handle:
            files.copy(stream, handle)
        return {"path": path}


def private_key(pem):
    import paramiko
    for kind in (paramiko.Ed25519Key, paramiko.RSAKey, paramiko.ECDSAKey):
        try:
            return kind.from_private_key(io.StringIO(pem))
        except paramiko.SSHException:
            continue
    raise ProviderError("Unsupported SSH private key type")


class VastSSH:
    """SSH uses a per-rental key, no SSH agent or operator account credentials."""
    connect_window = 90  # While the lease key may still be propagating to the container.
    settled_window = 30  # Afterwards: enough for transient proxy hiccups, not another long wait.
    def _connect(self, access, timeout):
        import paramiko
        # The authenticated Vast API supplies the endpoint. Refuse private-network
        # destinations and pin the resolved public address for this connection.
        try:
            addresses = socket.getaddrinfo(access["host"], access["port"], type=socket.SOCK_STREAM)
        except OSError as exc:
            raise ProviderError("SSH endpoint could not be resolved; retry after 30 seconds") from exc
        if not addresses or any(not ipaddress.ip_address(a[4][0]).is_global for a in addresses):
            raise ProviderError("Vast SSH endpoint must resolve to public addresses")
        # Vast attaches the per-rental key lazily, on the first ssh/exec after "running",
        # and the container accepts it 20-60 s later. Custom images pull for minutes, so
        # the window is anchored to that readiness (not to lease creation): a key attached
        # by this call, by a `compute ssh` under 3 minutes ago (it attaches without
        # connecting), or a rental that started under 3 minutes ago, gets the full window.
        # key_attached_at is in memory; after a restart the settled window applies.
        window = self.connect_window
        if access.get("provider") == "vastai" and not access.get("key_attached_now"):
            now = time.time()
            if not any(isinstance(at, (int, float)) and now - at < 180
                       for at in (access.get("instance_started_at"), access.get("key_attached_at"))):
                window = self.settled_window
        give_up = time.monotonic() + min(window, timeout)
        provider_name = "Vast" if access.get("provider", "vastai") == "vastai" else "Compute"
        while True:
            client = paramiko.SSHClient()
            client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                client.connect(addresses[0][4][0], port=access["port"], username=access.get("user", "root"),
                               pkey=private_key(access["private_key"]),
                               allow_agent=False, look_for_keys=False, timeout=min(timeout, 15),
                               auth_timeout=min(timeout, 15), banner_timeout=min(timeout, 15))
                return client
            except (OSError, paramiko.SSHException) as exc:
                client.close()
                if time.monotonic() >= give_up:
                    # Connection failures cannot create a billing reservation.
                    # Report a safe, actionable error instead of falling through
                    # to the broker's generic uncertain-provider response.
                    if isinstance(exc, paramiko.AuthenticationException):
                        attached_at = access.get("key_attached_at")
                        settled = isinstance(attached_at, (int, float)) and time.time() - attached_at > 600
                        if access.get("provider") == "vastai" and not settled:
                            raise ProviderError(
                                f"{provider_name} SSH rejected the lease key; the key may still be propagating to the instance. "
                                "Retry after 30 seconds; if it still fails 10 minutes after the key was attached, "
                                "contact the operator or choose another offer"
                            ) from exc
                        raise ProviderError(
                            f"{provider_name} SSH rejected the lease key; the instance SSH service or proxy may be unhealthy. "
                            "Retry after 30 seconds; if it persists, contact the operator or choose another offer"
                        ) from exc
                    raise ProviderError(
                        f"{provider_name} SSH connection failed; check compute status and the instance SSH endpoint. "
                        "Retry after 30 seconds; if it persists, contact the operator or choose another offer"
                    ) from exc
                time.sleep(5)
            except BaseException:
                client.close()
                raise

    def exec(self, access, command, timeout):
        client = self._connect(access, timeout)
        # SSH command negotiation can block before the output loop starts.
        # Close the entire transport on deadline so cleanup can acquire its lock.
        timer = threading.Timer(timeout, client.close)
        timer.daemon = True
        timer.start()
        try:
            channel = client.get_transport().open_session(timeout=min(timeout, 15))
            channel.exec_command(f"timeout --signal=KILL {timeout}s bash -lc {shlex.quote(command)}")
            deadline = time.monotonic() + timeout
            out, err = bytearray(), bytearray()
            try:
                while True:
                    if channel.recv_ready():
                        out.extend(channel.recv(65536))
                    if channel.recv_stderr_ready():
                        err.extend(channel.recv_stderr(65536))
                    if len(out) + len(err) > 8 * 1024 * 1024:
                        raise ProviderError("Command output exceeds 8 MiB; write large output to a file")
                    if channel.exit_status_ready() and not channel.recv_ready() and not channel.recv_stderr_ready():
                        code = channel.recv_exit_status()
                        break
                    if time.monotonic() >= deadline:
                        code = 124
                        break
                    time.sleep(0.01)
            finally:
                channel.close()
            return {"exit_code": code, "stdout": out.decode(errors="replace"), "stderr": err.decode(errors="replace")}
        finally:
            timer.cancel()
            client.close()

    @contextmanager
    def open(self, access, remote, mode, timeout=3600):
        """Stream a remote file over SFTP ("rb" or "wb"); writes create parent directories."""
        client = self._connect(access, min(timeout, self.connect_window))  # Same readiness window as exec.
        try:
            sftp = client.open_sftp()
            sftp.get_channel().settimeout(timeout)
            if "w" in mode:
                parent = ""
                for part in remote.strip("/").split("/")[:-1]:
                    parent += "/" + part
                    try:
                        sftp.stat(parent)
                    except FileNotFoundError:
                        sftp.mkdir(parent)
            try:
                handle = sftp.open(remote, mode)
            except FileNotFoundError:
                raise ValueError(f"No file at {remote}") from None
            with handle:
                if "r" in mode:
                    handle.prefetch()
                else:
                    handle.set_pipelined(True)
                yield handle
        finally:
            client.close()
