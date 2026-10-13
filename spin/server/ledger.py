"""Trusted spending ledger in SQLite: wallets, their tokens, and every reservation against their budgets.

Money is integer nano-dollars: costs round up, budgets round down. A market wallet's budget bounds the real
cost of its reservations. A static wallet's budget is in its static prices (a lease's static price, or the
real cost of anything the table does not price); its real cost is recorded but not capped, since the operator
pays the providers directly.
"""
from __future__ import annotations

from contextlib import closing, contextmanager
from decimal import Decimal, ROUND_CEILING, ROUND_FLOOR
import fcntl
import hashlib
import json
import os
from pathlib import Path
import sqlite3
import time
import uuid

SCALE = Decimal(1_000_000_000)

SCHEMA = """
PRAGMA journal_mode=WAL;
CREATE TABLE IF NOT EXISTS wallets (
    id TEXT PRIMARY KEY, name TEXT NOT NULL, request_key TEXT NOT NULL UNIQUE, fingerprint TEXT NOT NULL,
    budget INTEGER, static_budget INTEGER, static_pricing TEXT,
    status TEXT NOT NULL DEFAULT 'active', frozen INTEGER NOT NULL DEFAULT 0,
    created_at REAL NOT NULL, expires_at REAL, closed_at REAL);
CREATE TABLE IF NOT EXISTS reservations (
    id TEXT PRIMARY KEY, wallet_id TEXT NOT NULL REFERENCES wallets(id),
    request_key TEXT NOT NULL, maximum INTEGER NOT NULL,
    spent INTEGER NOT NULL DEFAULT 0, state TEXT NOT NULL,
    kind TEXT NOT NULL, metadata TEXT NOT NULL, result TEXT,
    created REAL NOT NULL, updated REAL NOT NULL,
    static_maximum INTEGER, static_spent INTEGER,
    UNIQUE(wallet_id, request_key));
CREATE TABLE IF NOT EXISTS tokens (
    hash TEXT PRIMARY KEY, wallet_id TEXT NOT NULL REFERENCES wallets(id),
    scope TEXT NOT NULL, revoked INTEGER NOT NULL DEFAULT 0);
CREATE TABLE IF NOT EXISTS transfer_scopes (
    wallet_id TEXT PRIMARY KEY REFERENCES wallets(id), endpoint_url TEXT NOT NULL,
    bucket TEXT NOT NULL, key_prefix TEXT NOT NULL, expires_at REAL NOT NULL,
    next_cleanup_at REAL NOT NULL DEFAULT 0, cleaned INTEGER NOT NULL DEFAULT 0);
"""


class BudgetError(ValueError):
    pass


class BudgetExceeded(BudgetError):
    pass


class Conflict(BudgetError):
    pass


class IdempotencyConflict(Conflict):
    """A request ID was reused for a different request."""


class NotFound(KeyError):
    pass


def units(value, *, budget=False):
    value = Decimal(str(value))
    if not value.is_finite() or value < 0:
        raise BudgetError("USD must be finite and nonnegative")
    result = int((value * SCALE).to_integral_value(rounding=ROUND_FLOOR if budget else ROUND_CEILING))
    if result > 9_000_000_000_000_000_000:
        raise BudgetError("USD exceeds ledger capacity")
    return result


def dollars(value):
    return None if value is None else format(Decimal(value) / SCALE, "f")


def digest(token):
    return hashlib.sha256(token.encode()).hexdigest()


class Ledger:
    def __init__(self, path):
        self.path = str(path)
        Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        Path(self.path).touch(mode=0o600)
        os.chmod(self.path, 0o600)
        with closing(self._connect()) as db:
            db.executescript(SCHEMA)

    def _connect(self):
        db = sqlite3.connect(self.path, timeout=30, isolation_level=None, check_same_thread=False)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        db.execute("PRAGMA synchronous=FULL")
        return db

    @contextmanager
    def _transaction(self):
        db = self._connect()
        try:
            db.execute("BEGIN IMMEDIATE")
            yield db
            db.commit()
        except BaseException:
            db.rollback()
            raise
        finally:
            db.close()

    def _read(self, query, params=()):
        with closing(self._connect()) as db:
            return db.execute(query, params).fetchall()

    # Wallets

    @staticmethod
    def _wallet(row):
        result = dict(row)
        result["static_pricing"] = json.loads(result["static_pricing"]) if result["static_pricing"] else None
        return result

    def create_wallet(self, wallet_id, *, name, request_key, fingerprint, budget_usd, static_pricing, expires_at, token,
                      created_at=None):
        """Returns (wallet, created). A static wallet's budget is its static budget; its real cost is uncapped.

        A request key used again with the same fingerprint returns the first wallet; with another, it conflicts.
        """
        budget = units(budget_usd, budget=True)
        with self._transaction() as db:
            old = db.execute("SELECT * FROM wallets WHERE request_key=?", (request_key,)).fetchone()
            if old:
                if old["fingerprint"] != fingerprint:
                    raise IdempotencyConflict("Key was already used with different parameters")
                return self._wallet(old), False
            static = static_pricing is not None
            db.execute("INSERT INTO wallets(id,name,request_key,fingerprint,budget,static_budget,static_pricing,created_at,expires_at)"
                       " VALUES(?,?,?,?,?,?,?,?,?)", (wallet_id, name, request_key, fingerprint, None if static else budget,
                       budget if static else None, json.dumps(static_pricing) if static else None, created_at or time.time(), expires_at))
            db.execute("INSERT INTO tokens(hash,wallet_id,scope) VALUES(?,?,?)", (digest(token), wallet_id, "agent"))
            row = db.execute("SELECT * FROM wallets WHERE id=?", (wallet_id,)).fetchone()
        return self._wallet(row), True

    def wallet(self, wallet_id):
        rows = self._read("SELECT * FROM wallets WHERE id=?", (wallet_id,))
        if not rows:
            raise NotFound("Wallet not found")
        return self._wallet(rows[0])

    def wallets(self, *, limit=100):
        return [self._wallet(row) for row in self._read("SELECT * FROM wallets ORDER BY created_at DESC, id DESC LIMIT ?", (limit,))]

    def open_wallets(self):
        return [self._wallet(row) for row in self._read("SELECT * FROM wallets WHERE status!='closed' ORDER BY created_at")]

    def mark_closing(self, wallet_id):
        with self._transaction() as db:
            db.execute("UPDATE wallets SET status='closing' WHERE id=? AND status='active'", (wallet_id,))

    def mark_closed(self, wallet_id):
        with self._transaction() as db:
            db.execute("UPDATE wallets SET status='closed',closed_at=? WHERE id=? AND status!='closed'", (time.time(), wallet_id))

    def has_token(self, wallet_id, token):
        return bool(self._read("SELECT 1 FROM tokens WHERE hash=? AND wallet_id=? AND scope='agent' AND revoked=0",
                               (digest(token), wallet_id)))

    # Tokens

    def issue_token(self, wallet_id, token, scope):
        if scope not in {"agent", "openrouter"}:
            raise ValueError("invalid scope")
        with self._transaction() as db:
            db.execute("INSERT INTO tokens(hash,wallet_id,scope) VALUES(?,?,?)", (digest(token), wallet_id, scope))

    def authenticate(self, token):
        """The token's wallet with its scope and revocation, or None for an unknown token."""
        rows = self._read("SELECT t.wallet_id,t.scope,t.revoked,w.* FROM tokens t JOIN wallets w ON w.id=t.wallet_id WHERE t.hash=?",
                          (digest(token),))
        return self._wallet(rows[0]) if rows else None

    def revoke_tokens(self, wallet_id):
        with self._transaction() as db:
            db.execute("UPDATE tokens SET revoked=1 WHERE wallet_id=?", (wallet_id,))

    # Storage scopes

    def register_transfer_scope(self, wallet_id, scope):
        """Remember the last credential expiry before returning secrets; serialize against freeze."""
        with self._transaction() as db:
            wallet = db.execute("SELECT frozen FROM wallets WHERE id=?", (wallet_id,)).fetchone()
            if wallet is None or wallet[0]:
                raise BudgetExceeded("Wallet is closed")
            old = db.execute("SELECT * FROM transfer_scopes WHERE wallet_id=?", (wallet_id,)).fetchone()
            destination = (scope["endpoint_url"], scope["bucket"], scope["key_prefix"])
            if old and tuple(old[k] for k in ("endpoint_url", "bucket", "key_prefix")) != destination:
                raise Conflict("Wallet storage destination changed; restore the original storage configuration")
            if old:
                db.execute("UPDATE transfer_scopes SET expires_at=? WHERE wallet_id=?",
                           (max(old["expires_at"], scope["expires_at"]), wallet_id))
            else:
                db.execute("INSERT INTO transfer_scopes(wallet_id,endpoint_url,bucket,key_prefix,expires_at) VALUES(?,?,?,?,?)",
                           (wallet_id, *destination, scope["expires_at"]))

    def transfer_scopes(self, wallet_id=None, *, due_at=None):
        """Storage of closed (frozen) wallets that still needs a cleanup pass."""
        query = "SELECT t.* FROM transfer_scopes t JOIN wallets w ON w.id=t.wallet_id WHERE t.cleaned=0 AND w.frozen=1"
        params = []
        if wallet_id is not None:
            query += " AND t.wallet_id=?"
            params.append(wallet_id)
        if due_at is not None:
            query += " AND t.next_cleanup_at<=?"
            params.append(due_at)
        return [dict(row) for row in self._read(query, params)]

    def record_transfer_cleanup(self, wallet_id, *, next_cleanup_at, cleaned):
        with self._transaction() as db:
            db.execute("UPDATE transfer_scopes SET next_cleanup_at=?,cleaned=? WHERE wallet_id=?",
                       (next_cleanup_at, int(cleaned), wallet_id))

    # Reservations

    @staticmethod
    def _record(row):
        result = dict(row)
        result["maximum_usd"] = dollars(result.pop("maximum"))
        result["spent_usd"] = dollars(result.pop("spent"))
        # Static amounts exist only for leases with a static price; NULL means the real amount counts as is.
        static_maximum, static_spent = result.pop("static_maximum", None), result.pop("static_spent", None)
        result["static_maximum_usd"] = dollars(static_maximum)
        result["static_spent_usd"] = None if static_maximum is None else dollars(static_spent or 0)
        result["metadata"] = json.loads(result["metadata"])
        result["result"] = json.loads(result["result"]) if result["result"] else None
        return result

    def reserve(self, wallet_id, key, maximum_usd, kind, metadata=None, *, static_usd=None):
        """Returns (reservation, created). Only the creator may submit paid work.

        maximum_usd is the real cost held; static_usd (a lease's static price) is held against a static
        budget instead, which otherwise counts maximum_usd.
        """
        amount = units(maximum_usd)
        static = None if static_usd is None else units(static_usd)
        metadata_json = json.dumps(metadata or {}, sort_keys=True)
        if not key or len(key) > 200:
            raise BudgetError("idempotency key must contain 1..200 characters")
        with self._transaction() as db:
            old = db.execute("SELECT * FROM reservations WHERE wallet_id=? AND request_key=?", (wallet_id, key)).fetchone()
            if old:
                if (old["maximum"] != amount or old["static_maximum"] != static or old["kind"] != kind
                        or old["metadata"] != metadata_json):
                    raise IdempotencyConflict("Request ID already used for a different request; use a new request ID")
                return self._record(old), False
            wallet = db.execute("SELECT * FROM wallets WHERE id=?", (wallet_id,)).fetchone()
            if wallet is None:
                raise BudgetError("unknown wallet")
            if wallet["frozen"]:
                raise BudgetExceeded("Wallet is closed")
            committed, static_committed = db.execute(
                "SELECT COALESCE(SUM(CASE WHEN state='settled' THEN spent ELSE maximum END),0),"
                " COALESCE(SUM(CASE WHEN state='settled' THEN COALESCE(static_spent,spent) ELSE COALESCE(static_maximum,maximum) END),0)"
                " FROM reservations WHERE wallet_id=?", (wallet_id,)).fetchone()
            if wallet["static_budget"] is not None and static_committed + (amount if static is None else static) > wallet["static_budget"]:
                raise BudgetExceeded("Not enough budget left in this wallet")
            if wallet["budget"] is not None and committed + amount > wallet["budget"]:
                raise BudgetExceeded("Not enough budget left in this wallet")
            reservation_id = uuid.uuid4().hex
            now = time.time()
            db.execute("INSERT INTO reservations(id,wallet_id,request_key,maximum,state,kind,metadata,created,updated,static_maximum,static_spent)"
                       " VALUES(?,?,?,?,?,?,?,?,?,?,?)", (reservation_id, wallet_id, key, amount, "reserved", kind, metadata_json, now, now,
                                                       static, None if static is None else 0))
            row = db.execute("SELECT * FROM reservations WHERE id=?", (reservation_id,)).fetchone()
        return self._record(row), True

    def get(self, reservation_id, wallet_id=None):
        rows = self._read("SELECT * FROM reservations WHERE id=?", (reservation_id,))
        if not rows or (wallet_id is not None and rows[0]["wallet_id"] != wallet_id):
            raise NotFound("No resource with that ID in this wallet")
        return self._record(rows[0])

    def find(self, wallet_id, key):
        rows = self._read("SELECT * FROM reservations WHERE wallet_id=? AND request_key=?", (wallet_id, key))
        return self._record(rows[0]) if rows else None

    def reservations(self, wallet_id):
        return [self._record(row) for row in self._read(
            "SELECT * FROM reservations WHERE wallet_id=? ORDER BY created, id", (wallet_id,))]

    def charge_compute(self, reservation_id, result, *, resource_id=None):
        """Atomically capture the full prepaid price and persist the lease update.

        For extensions, the payment and the original resource's new deadline commit together.
        Callers hold the original resource lock throughout the provider operation.
        """
        with self._transaction() as db:
            row = db.execute("SELECT * FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            if row is None or row["kind"] not in {"compute", "compute_extension"}:
                raise KeyError(reservation_id)
            if row["state"] == "settled":
                if row["spent"] != row["maximum"] or not row["result"]:
                    raise Conflict("compute payment was already rejected or settled differently")
                return
            if row["result"] and json.loads(row["result"]).get("status") == "checking":
                wallet = db.execute("SELECT frozen FROM wallets WHERE id=?", (row["wallet_id"],)).fetchone()
                if wallet["frozen"]:
                    raise BudgetExceeded("Wallet closed before initial health check completed")
            now = time.time()
            if resource_id:
                resource = db.execute("SELECT * FROM reservations WHERE id=?", (resource_id,)).fetchone()
                if resource is None or resource["wallet_id"] != row["wallet_id"] or resource["kind"] != "compute":
                    raise KeyError(resource_id)
                db.execute("UPDATE reservations SET result=?,updated=? WHERE id=?", (json.dumps(result), now, resource_id))
                payment_result = {"status": "applied", "resource_id": resource_id, "expires_at": result["expires_at"]}
            else:
                payment_result = result
            db.execute("UPDATE reservations SET result=?,state='settled',spent=maximum,static_spent=static_maximum,updated=? WHERE id=?",
                       (json.dumps(payment_result), now, reservation_id))

    def set_result(self, reservation_id, result):
        with self._transaction() as db:
            db.execute("UPDATE reservations SET result=?, updated=? WHERE id=?", (json.dumps(result), time.time(), reservation_id))

    def reject_compute(self, reservation_id):
        """Persist a definitive rejection and release its hold in one commit."""
        with self._transaction() as db:
            row = db.execute("SELECT * FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            if row is None or row["kind"] not in {"compute", "compute_extension"}:
                raise KeyError(reservation_id)
            if row["state"] == "settled":
                if row["result"] and json.loads(row["result"]).get("status") == "rejected":
                    return
                raise Conflict("A captured compute charge cannot be refunded by rejection")
            db.execute("UPDATE reservations SET result=?,state='settled',spent=0,"
                       "static_spent=CASE WHEN static_maximum IS NULL THEN NULL ELSE 0 END,updated=? WHERE id=?",
                       (json.dumps({"id": None, "status": "rejected"}), time.time(), reservation_id))

    def mark_uncertain(self, reservation_id):
        # Provider exception text may contain credentials; nothing about the failure is persisted.
        with self._transaction() as db:
            db.execute("UPDATE reservations SET state='uncertain',updated=? WHERE id=? AND state!='settled'", (time.time(), reservation_id))

    def discard(self, reservation_id):
        """Forget an unsettled reservation whose request certainly never reached the provider; its key can be reused."""
        with self._transaction() as db:
            db.execute("DELETE FROM reservations WHERE id=? AND state!='settled'", (reservation_id,))

    def settle(self, reservation_id, cost_usd, *, result=None):
        """Record the real cost exactly once; with `result`, the provider's reply commits in the same transaction.

        A cost above the hold is still recorded, and freezes the wallet. A reservation already settled
        at this cost is left as it is, result included.
        """
        cost = units(cost_usd)
        with self._transaction() as db:
            row = db.execute("SELECT * FROM reservations WHERE id=?", (reservation_id,)).fetchone()
            if row is None:
                raise KeyError(reservation_id)
            if row["state"] == "settled":
                if row["spent"] != cost:
                    raise Conflict("reservation already settled at another cost")
                return
            if result is None:
                db.execute("UPDATE reservations SET state='settled',spent=?,updated=? WHERE id=?", (cost, time.time(), reservation_id))
            else:
                db.execute("UPDATE reservations SET result=?,state='settled',spent=?,updated=? WHERE id=?",
                           (json.dumps(result), cost, time.time(), reservation_id))
            if cost > row["maximum"]:
                db.execute("UPDATE wallets SET frozen=1 WHERE id=?", (row["wallet_id"],))
        if cost > row["maximum"]:
            raise BudgetExceeded("provider exceeded reserved maximum; wallet frozen and actual cost recorded")

    def status(self, wallet_id):
        """Real amounts (what the operator pays); a static wallet also reports its static prices under "static"."""
        with closing(self._connect()) as db:
            db.execute("BEGIN")  # One snapshot for the wallet and its totals.
            wallet = db.execute("SELECT * FROM wallets WHERE id=?", (wallet_id,)).fetchone()
            spent, reserved, static_spent, static_reserved = db.execute(
                "SELECT COALESCE(SUM(CASE WHEN state='settled' THEN spent ELSE 0 END),0),"
                " COALESCE(SUM(CASE WHEN state='settled' THEN 0 ELSE maximum END),0),"
                " COALESCE(SUM(CASE WHEN state='settled' THEN COALESCE(static_spent,spent) ELSE 0 END),0),"
                " COALESCE(SUM(CASE WHEN state='settled' THEN 0 ELSE COALESCE(static_maximum,maximum) END),0)"
                " FROM reservations WHERE wallet_id=?", (wallet_id,)).fetchone()
            db.execute("COMMIT")
        if wallet is None:
            raise NotFound("Wallet not found")
        budget = wallet["budget"]
        status = {"run_id": wallet_id, "budget_usd": dollars(budget), "spent_usd": dollars(spent),
                  "reserved_usd": dollars(reserved), "available_usd": None if budget is None else dollars(max(0, budget - spent - reserved)),
                  "frozen": bool(wallet["frozen"])}
        if wallet["static_budget"] is not None:
            static_budget = wallet["static_budget"]
            status["static"] = {"budget_usd": dollars(static_budget), "spent_usd": dollars(static_spent),
                                "reserved_usd": dollars(static_reserved),
                                "available_usd": dollars(max(0, static_budget - static_spent - static_reserved))}
        return status

    def frozen(self, wallet_id):
        rows = self._read("SELECT frozen FROM wallets WHERE id=?", (wallet_id,))
        return bool(rows and rows[0][0])

    def freeze(self, wallet_id):
        with self._transaction() as db:
            db.execute("UPDATE wallets SET frozen=1 WHERE id=?", (wallet_id,))

    # Lease locks

    def _lock_path(self, resource_id):
        directory = Path(self.path + ".locks")
        directory.mkdir(mode=0o700, exist_ok=True)
        return directory / hashlib.sha256(str(resource_id).encode()).hexdigest()

    @contextmanager
    def resource_lock(self, resource_id):
        """Exclusion between calls that change a lease (create, stop, extend, health gate)."""
        with self._lock_path(resource_id).open("a") as lock:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)

    @contextmanager
    def try_resource_lock(self, resource_id):
        """resource_lock without waiting: yields whether the lease is ours. The reaper skips a busy lease."""
        with self._lock_path(resource_id).open("a") as lock:
            try:
                fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                yield False
                return
            try:
                yield True
            finally:
                fcntl.flock(lock.fileno(), fcntl.LOCK_UN)
