"""Streaming helpers for server-side file transfers (nothing is staged on disk)."""
import hashlib
import io

CHUNK = 1024 * 1024


class Counting(io.RawIOBase):
    """Wraps a readable stream: counts bytes, hashes them, and stops past a limit."""

    def __init__(self, stream, limit=None):
        self.stream, self.limit, self.bytes, self.sha256 = stream, limit, 0, hashlib.sha256()

    def readable(self):
        return True

    def read(self, size=-1):
        data = self.stream.read(CHUNK if size is None or size < 0 else size)
        self.bytes += len(data)
        if self.limit is not None and self.bytes > self.limit:
            raise ValueError(f"file exceeds {self.limit} bytes")
        self.sha256.update(data)
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


class Chunks(io.RawIOBase):
    """A readable stream over an iterator of byte chunks (e.g. httpx iter_bytes)."""

    def __init__(self, chunks):
        self.chunks, self.pending = iter(chunks), b""

    def readable(self):
        return True

    def read(self, size=-1):
        while (size is None or size < 0 or len(self.pending) < size):
            chunk = next(self.chunks, None)
            if chunk is None:
                break
            self.pending += chunk
        if size is None or size < 0:
            data, self.pending = self.pending, b""
        else:
            data, self.pending = self.pending[:size], self.pending[size:]
        return data

    def readinto(self, buffer):
        data = self.read(len(buffer))
        buffer[:len(data)] = data
        return len(data)


def copy(source, target):
    while chunk := source.read(CHUNK):
        target.write(chunk)
