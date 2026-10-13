"""Wallet-scoped R2 sessions, using Cloudflare's local JWT credential format.

The parent R2 API token never leaves the server: each session is a JWT it signs, scoped by R2 to one
operation class and the wallet's own key prefix, and expiring with the session (and the wallet).
"""
from __future__ import annotations

import base64
from functools import cached_property
import hashlib
import hmac
import importlib.util
import json
import re
import time

from ..limits import DEFAULT_TTL, MAX_TTL
from ..s5cmd import folder, r2_endpoint

CLEANUP_INTERVAL = 60
# Allow requests accepted just before credential expiry to finish before the last sweep.
CLEANUP_GRACE = 300


class R2TransferStore:
    def __init__(self, *, endpoint_url, bucket, access_key_id, secret_access_key, prefix="spin-transfers", clock=time.time):
        self.endpoint_url, self.account_id, self.audience = r2_endpoint(endpoint_url)
        if not isinstance(bucket, str) or not re.fullmatch(r"[a-z0-9][a-z0-9-]{1,61}[a-z0-9]", bucket):
            raise ValueError("SPIN_R2_BUCKET must be an R2 bucket name")
        if not access_key_id or not secret_access_key:
            raise ValueError("R2 storage needs SPIN_R2_ACCESS_KEY_ID and SPIN_R2_SECRET_ACCESS_KEY")
        self.bucket, self.prefix = bucket, folder(prefix, allow_empty=False)
        self.access_key_id, self._secret = access_key_id, secret_access_key
        self.clock = clock

    @classmethod
    def from_env(cls, env):
        """The store SPIN_R2_* configures, or None when storage is off."""
        names = ("SPIN_R2_BUCKET", "SPIN_R2_ACCESS_KEY_ID", "SPIN_R2_SECRET_ACCESS_KEY")
        endpoint = env.get("SPIN_R2_ENDPOINT_URL") or (
            f"https://{env['SPIN_R2_ACCOUNT_ID']}.r2.cloudflarestorage.com" if env.get("SPIN_R2_ACCOUNT_ID") else None)
        if not endpoint and not any(env.get(name) for name in names):
            return None
        if not endpoint or not all(env.get(name) for name in names):
            raise ValueError("R2 storage needs SPIN_R2_ACCOUNT_ID (or SPIN_R2_ENDPOINT_URL), SPIN_R2_BUCKET, "
                             "SPIN_R2_ACCESS_KEY_ID and SPIN_R2_SECRET_ACCESS_KEY")
        if importlib.util.find_spec("boto3") is None:
            raise ValueError("R2 storage cleanup needs boto3 (pip install boto3)")
        return cls(endpoint_url=endpoint, bucket=env["SPIN_R2_BUCKET"], prefix=env.get("SPIN_R2_PREFIX") or "spin-transfers",
                   access_key_id=env["SPIN_R2_ACCESS_KEY_ID"], secret_access_key=env["SPIN_R2_SECRET_ACCESS_KEY"])

    def wallet_prefix(self, wallet_id):
        if not isinstance(wallet_id, str) or not re.fullmatch(r"[A-Za-z0-9_-]{1,80}", wallet_id):
            raise ValueError("invalid wallet ID")
        return f"{self.prefix}/{wallet_id}/"

    def credentials(self, wallet_id, *, prefix="", permission="read-write", expires_seconds=DEFAULT_TTL,
                    wallet_expires_at=None):
        if permission not in ("read-only", "read-write"):
            raise ValueError("permission must be read-only or read-write")
        if type(expires_seconds) is not int or not 1 <= expires_seconds <= MAX_TTL:
            raise ValueError(f"expires_seconds must be 1..{MAX_TTL}")
        relative = folder(prefix)
        key_prefix = self.wallet_prefix(wallet_id) + (relative + "/" if relative else "")
        now = int(self.clock())
        expiry = now + expires_seconds
        if wallet_expires_at is not None:
            expiry = min(expiry, int(wallet_expires_at))
        if expiry <= now:
            raise PermissionError("Wallet has expired")
        # Cloudflare's documented local signing format. R2 enforces both the
        # object operation scope and this exact, slash-terminated wallet prefix.
        claims = {"bucket": self.bucket, "scope": "object-" + permission,
                  "paths": {"prefixPaths": [key_prefix], "objectPaths": []},
                  "sub": self.account_id, "iss": self.access_key_id, "aud": self.audience,
                  "iat": now, "exp": expiry}

        def encoded(value):
            return base64.urlsafe_b64encode(json.dumps(value, separators=(",", ":")).encode()).rstrip(b"=")
        message = encoded({"alg": "HS256", "typ": "JWT"}) + b"." + encoded(claims)
        signature = hmac.new(self._secret.encode(), message, hashlib.sha256).digest()
        jwt = message + b"." + base64.urlsafe_b64encode(signature).rstrip(b"=")
        return {"endpoint_url": self.endpoint_url, "region": "auto", "bucket": self.bucket,
                "prefix": key_prefix, "s3_uri": f"s3://{self.bucket}/{key_prefix}",
                "permission": permission, "expires_at": expiry, "expires_seconds": expiry - now,
                "access_key_id": self.access_key_id,
                "secret_access_key": hashlib.sha256(jwt).hexdigest(),
                "session_token": base64.b64encode(b"jwt/" + jwt).decode()}

    @cached_property
    def s3(self):
        import boto3
        from botocore.config import Config
        return boto3.client("s3", endpoint_url=self.endpoint_url, region_name="auto",
            aws_access_key_id=self.access_key_id, aws_secret_access_key=self._secret,
            config=Config(connect_timeout=5, read_timeout=15, retries={"max_attempts": 2, "mode": "standard"}))

    def cleanup(self, wallet_id, scope):
        """Delete only this wallet's objects and incomplete uploads, with bounded work per pass.

        Return False when another pass is needed. Never treat a partial S3 batch failure as success.
        Persisted destination checks protect against configuration changes.
        """
        prefix = self.wallet_prefix(wallet_id)
        if (scope["endpoint_url"], scope["bucket"], scope["key_prefix"]) != (self.endpoint_url, self.bucket, prefix):
            raise ValueError("Storage cleanup destination changed; restore the original storage configuration")
        deadline = time.monotonic() + 30
        for _ in range(10):
            if time.monotonic() >= deadline:
                return False
            page = self.s3.list_multipart_uploads(Bucket=self.bucket, Prefix=prefix, MaxUploads=100)
            for upload in page.get("Uploads", []):
                if not upload["Key"].startswith(prefix):
                    raise ValueError("Storage returned an upload outside the wallet prefix")
                if time.monotonic() >= deadline:
                    return False
                self.s3.abort_multipart_upload(Bucket=self.bucket, Key=upload["Key"], UploadId=upload["UploadId"])
            if not page.get("IsTruncated"):
                break
        else:
            return False
        for _ in range(10):
            if time.monotonic() >= deadline:
                return False
            page = self.s3.list_objects_v2(Bucket=self.bucket, Prefix=prefix, MaxKeys=1000)
            objects = [{"Key": obj["Key"]} for obj in page.get("Contents", [])]
            if any(not obj["Key"].startswith(prefix) for obj in objects):
                raise ValueError("Storage returned an object outside the wallet prefix")
            if not objects:
                return True
            result = self.s3.delete_objects(Bucket=self.bucket, Delete={"Objects": objects, "Quiet": True})
            if result.get("Errors"):
                raise RuntimeError("Storage reported a partial deletion failure; cleanup will retry")
        return False
