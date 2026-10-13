"""Spec fields shared by providers that build images and can switch networking off.

  "network": "none"        no network access from the sandbox
  "dockerfile": "..."      build the image from this Dockerfile (instead of "image")
  "context": "<base64>"    optional build context: a .tar.gz, base64-encoded
"""
import base64
import gzip
import hashlib
import io
import tarfile

MAX_DOCKERFILE_BYTES = 64 * 1024
MAX_CONTEXT_BYTES = 10 * 1024 * 1024  # decoded .tar.gz; prebuild and publish larger images
FIELDS = {"network", "dockerfile", "context"}


def validate(spec):
    """Check the build and network fields; returns them for the quoted spec (context as a digest)."""
    fields = {}
    network = spec.get("network", "default")
    if network not in {"none", "default"}:
        raise ValueError('network must be "none" or "default"')
    if network == "none":
        fields["network"] = "none"
    dockerfile, context = spec.get("dockerfile"), spec.get("context")
    if dockerfile is not None:
        if spec.get("image") is not None:
            raise ValueError("Give either image or dockerfile, not both")
        if not isinstance(dockerfile, str) or not dockerfile.strip() or len(dockerfile.encode()) > MAX_DOCKERFILE_BYTES:
            raise ValueError("dockerfile must be the Dockerfile's contents, at most 64 KiB")
        fields["dockerfile"] = dockerfile
    if context is not None:
        if dockerfile is None:
            raise ValueError("context needs a dockerfile")
        fields["context"] = "sha256:" + hashlib.sha256(context_tar(context)).hexdigest()
    return fields


def context_tar(context):
    """The build context as an uncompressed tar, checked for size and safety."""
    if not isinstance(context, str):
        raise ValueError("context must be a base64-encoded .tar.gz")
    try:
        packed = base64.b64decode(context, validate=True)
    except ValueError:
        raise ValueError("context must be a base64-encoded .tar.gz") from None
    if len(packed) > MAX_CONTEXT_BYTES:
        raise ValueError("context exceeds 10 MiB; publish the image and use image instead")
    try:
        raw = gzip.decompress(packed)
    except (OSError, EOFError):
        raise ValueError("context must be a gzip-compressed tar") from None
    if len(raw) > 8 * MAX_CONTEXT_BYTES:
        raise ValueError("context expands to more than 80 MiB")
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        for member in archive.getmembers():
            if member.name.startswith("/") or ".." in member.name.split("/") or not (member.isfile() or member.isdir()):
                raise ValueError("context may only contain regular files and directories with relative paths")
    return raw


def extract(raw, destination):
    with tarfile.open(fileobj=io.BytesIO(raw)) as archive:
        archive.extractall(destination, filter="data")


def record_spec(spec):
    """The spec as stored in the ledger: the context archive is replaced by its digest."""
    if isinstance(spec, dict) and isinstance(spec.get("context"), str):
        return {**spec, "context": "sha256:" + hashlib.sha256(spec["context"].encode()).hexdigest()}
    return spec
