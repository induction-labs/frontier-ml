"""Client for a separately running, trusted Lean checker; no compiler in this module."""
import json
import urllib.parse
import urllib.request

from frontier_ml.validation.errors import VerificationFailure

PROTOCOL = "lean-simplification-v1"
LEAN_VERSION = "4.15.0"
MATHLIB_REVISION = "9837ca9d65d9de6fad1ef4381750ca688774e608"


class CheckerClient:
    def __init__(self, url, fingerprint=None):
        parsed = urllib.parse.urlsplit(url or "")
        if parsed.scheme != "https" and not (parsed.scheme == "http" and parsed.hostname in {"localhost", "127.0.0.1", "::1"}):
            raise VerificationFailure("Checker needs HTTPS, or HTTP on loopback")
        if parsed.query or parsed.fragment or parsed.username or parsed.password:
            raise VerificationFailure("Invalid checker URL")
        self.url = url.rstrip("/")
        self.fingerprint = fingerprint
        health = self._request("/health")
        expected = {"protocol": PROTOCOL, "lean": LEAN_VERSION, "mathlib": MATHLIB_REVISION}
        if any(health.get(k) != v for k, v in expected.items()):
            raise VerificationFailure("Checker toolchain/protocol mismatch")
        if fingerprint and health.get("fingerprint") != fingerprint:
            raise VerificationFailure("Checker fingerprint mismatch")
        self.health = health

    def _request(self, path, body=None):
        request = urllib.request.Request(self.url + path,
                                         None if body is None else json.dumps(body).encode(),
                                         {"Content-Type": "application/json"})
        try:
            with urllib.request.urlopen(request, timeout=75) as response:
                raw = response.read(1_048_577)
            if len(raw) > 1_048_576:
                raise ValueError("oversized checker response")
            result = json.loads(raw)
            if not isinstance(result, dict):
                raise ValueError("checker response must be an object")
            return result
        except Exception as exc:
            raise VerificationFailure(f"Checker infrastructure failure: {exc}") from exc

    def __call__(self, sample, proof):
        result = self._request("/check", {"statement": sample["statement"], "proof": proof})
        if type(result.get("valid")) is not bool or result.get("status") not in {"ok", "invalid", "timeout"}:
            raise VerificationFailure("Missing or invalid trusted checker verdict")
        if result["valid"] != (result["status"] == "ok"):
            raise VerificationFailure("Inconsistent checker verdict")
        return result
