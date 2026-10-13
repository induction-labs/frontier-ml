"""Limits shared by the lightweight CLI and the server."""

INLINE_MAX_BYTES = 512 * 1024
# Storage sessions last 8 hours by default and at most, capped by wallet expiry.
DEFAULT_TTL = MAX_TTL = 8 * 60 * 60
