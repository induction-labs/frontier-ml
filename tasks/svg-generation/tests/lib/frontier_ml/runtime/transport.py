"""HTTP transport whose trusted client can enter a candidate network namespace.

Only the network namespace is entered. Python, files, credentials and the client
process remain in the verifier's namespaces, outside the candidate environment.
"""

import base64
import http.client
import json
import subprocess
import sys
import urllib.parse


class TransportError(RuntimeError):
    pass


# Run with -I: neither candidate cwd nor PYTHONPATH can supply Python modules.
_BRIDGE = r'''
import base64, http.client, json, sys
r = json.load(sys.stdin)
c = http.client.HTTPConnection(r['host'], r['port'], timeout=r['timeout'])
c.request(r['method'], r['path'], body=base64.b64decode(r['body']), headers=r['headers'])
v = c.getresponse()
b = v.read(None if r['limit'] is None else r['limit'] + 1)
if r['limit'] is not None and len(b) > r['limit']:
    raise ValueError('Response exceeds size limit')
print(json.dumps({'status':v.status, 'body':base64.b64encode(b).decode()}))
c.close()
'''


class HTTPTransport:
    def __init__(self, base_url="http://127.0.0.1:8000", *, namespace_pid=None,
                 max_response_bytes=16 * 1024 * 1024):
        parsed = urllib.parse.urlsplit(base_url)
        if parsed.scheme not in ("http", "https") or not parsed.hostname or parsed.path not in ("", "/"):
            raise ValueError("Transport requires an HTTP(S) origin without a path")
        self.scheme = parsed.scheme
        self.base_url = base_url.rstrip("/")
        self.host = parsed.hostname
        self.port = parsed.port or (443 if parsed.scheme == "https" else 80)
        self.namespace_pid = namespace_pid
        self.max_response_bytes = max_response_bytes

    def raw_request(self, method, path, *, body=b"", headers=None, timeout=30):
        if timeout <= 0:
            raise TimeoutError("Request deadline exceeded")
        if not path.startswith("/") or path.startswith("//"):
            raise ValueError("Request path must be relative to the submission origin")
        headers = dict(headers or {})
        if self.namespace_pid is not None:
            if self.scheme != "http":
                raise ValueError("Namespace transport supports local HTTP only")
            request = dict(host=self.host, port=self.port, method=method, path=path,
                           body=base64.b64encode(body).decode(), headers=headers,
                           timeout=timeout, limit=self.max_response_bytes)
            try:
                result = subprocess.run(
                    ["nsenter", "--target", str(self.namespace_pid), "--net", "--",
                     sys.executable, "-I", "-c", _BRIDGE],
                    input=json.dumps(request), capture_output=True, text=True,
                    timeout=timeout, check=True,
                )
            except subprocess.TimeoutExpired as exc:
                raise TimeoutError("Submission HTTP deadline exceeded") from exc
            except subprocess.CalledProcessError as exc:
                raise TransportError("Trusted network-namespace HTTP bridge failed: " + exc.stderr[-1500:]) from exc
            response = json.loads(result.stdout)
            return response["status"], base64.b64decode(response["body"])
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
