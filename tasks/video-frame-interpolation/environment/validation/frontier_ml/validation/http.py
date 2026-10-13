"""Bounded HTTP requests to a development server."""
import http.client
import json
import urllib.parse

class TransportError(RuntimeError):
    pass

class HTTPTransport:
    def __init__(self, base_url="http://127.0.0.1:8000", *, max_response_bytes=16 * 1024 * 1024):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/"):
            raise ValueError("Transport requires an HTTP(S) origin without a path")
        self.scheme = parsed.scheme
        self.base_url = base_url.rstrip("/")
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.max_response_bytes = max_response_bytes

    def raw_request(self, method, path, *, body=b"", headers=None, timeout=30):
        if timeout <= 0:
            raise TimeoutError("Request deadline exceeded")
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Request path must be relative to the submission origin")
        headers = dict(headers or {})
        connection_type = http.client.HTTPSConnection if self.scheme == "https" else http.client.HTTPConnection
        connection = connection_type(self.host, self.port, timeout=timeout)
        try:
            connection.request(method, path, body=body, headers=headers)
            response = connection.getresponse()
            content = response.read(None if self.max_response_bytes is None else self.max_response_bytes + 1)
            if self.max_response_bytes is not None and len(content) > self.max_response_bytes:
                raise TransportError("Submission response exceeds size limit")
            return response.status, content
        finally:
            connection.close()

    def request(self, method, path, payload=None, *, body=None, headers=None, timeout=30):
        headers = dict(headers or {})
        if payload is not None:
            if body is not None:
                raise ValueError("Provide payload or body, not both")
            body = json.dumps(payload, allow_nan=False).encode()
            headers["Content-Type"] = "application/json"
        status, content = self.raw_request(method, path, body=body or b"", headers=headers, timeout=timeout)
        if status != 200:
            raise TransportError(f"Submission returned HTTP {status}: {content[:200]!r}")
        try:
            result = json.loads(content)
        except (ValueError, UnicodeDecodeError) as exc:
            raise TransportError("Submission did not return valid JSON") from exc
        if not isinstance(result, (dict, list)):
            raise TransportError("Submission response must be a JSON object or array")
        return result
