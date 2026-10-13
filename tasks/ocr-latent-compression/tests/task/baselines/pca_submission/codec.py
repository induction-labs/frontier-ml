"""Fixed-rate uniform scalar quantizer for per-token PCA coefficients (no header, no text)."""
import numpy as np


def quantize(coefficients, low, high, bits):
    """coefficients [tokens, k] -> integer codes in [0, 2**bits - 1]."""
    levels = (1 << bits) - 1
    scaled = (coefficients - low) / np.maximum(high - low, 1e-6) * levels
    return np.clip(np.rint(scaled), 0, levels).astype(np.uint16)


def dequantize(codes, low, high, bits):
    levels = (1 << bits) - 1
    return low + codes.astype(np.float32) / levels * (high - low)


def pack(codes, bits):
    flat = codes.reshape(-1).astype(np.uint32)
    shifts = np.arange(bits - 1, -1, -1, dtype=np.uint32)
    bit_matrix = ((flat[:, None] >> shifts) & 1).astype(np.uint8)
    return np.packbits(bit_matrix.reshape(-1)).tobytes()


def unpack(payload, count, bits):
    """Inverse of pack; short or malformed payloads are zero-padded so decode never crashes."""
    needed = (count * bits + 7) // 8
    raw = np.frombuffer(payload[:needed].ljust(needed, b"\0"), dtype=np.uint8)
    bit_matrix = np.unpackbits(raw)[:count * bits].reshape(count, bits).astype(np.uint32)
    weights = (1 << np.arange(bits - 1, -1, -1, dtype=np.uint32))
    return (bit_matrix * weights).sum(axis=1).astype(np.uint16)
