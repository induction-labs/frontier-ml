"""Versioned, stdlib-only frame protocol. No scorer or dataset-specific assumptions."""
import base64
import binascii
import struct
import zlib
from pathlib import Path

VERSION = 2
ENDPOINT = "/interpolate"
FACTORS = (4, 8, 16)
MAX_FRAMES = 17
MAX_DIMENSION = 4096
MAX_SEQUENCE_BYTES = 512 * 1024**2
MAX_PNG_BYTES = 32 * 1024**2
MAX_JSON_BYTES = 768 * 1024**2
PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"
RGB8_CONTENT_TYPE = "application/vnd.frontier.rgb8"
TIMELINE_KEYS = {"schema_version", "width", "height", "frame_count", "target_indices"}
REQUEST_KEYS = TIMELINE_KEYS | {"input_frames"}


def _integer(value, name, lower, upper):
    if type(value) is not int or not lower <= value <= upper:
        raise ValueError(f"{name} must be an integer in [{lower}, {upper}]")
    return value


def _indices(values, name, frame_count):
    if not isinstance(values, list) or not values or len(values) > frame_count:
        raise ValueError(f"{name} must be a nonempty bounded list")
    for value in values:
        _integer(value, name, 0, frame_count - 1)
    if any(a >= b for a, b in zip(values, values[1:])):
        raise ValueError(f"{name} must be strictly increasing without duplicates")
    return values


def validate_timeline(value, *, image_field):
    """Validate either a JSONL sample (path) or a wire request (png)."""
    if not isinstance(value, dict) or not REQUEST_KEYS <= value.keys():
        raise ValueError("Missing sequence fields")
    _integer(value["schema_version"], "schema_version", VERSION, VERSION)
    width = _integer(value["width"], "width", 1, MAX_DIMENSION)
    height = _integer(value["height"], "height", 1, MAX_DIMENSION)
    count = _integer(value["frame_count"], "frame_count", 3, MAX_FRAMES)
    if count - 1 not in FACTORS:
        raise ValueError("frame_count must be 5, 9, or 17 (4x, 8x, or 16x interpolation)")
    if width * height * count * 3 > MAX_SEQUENCE_BYTES:
        raise ValueError("Uncompressed sequence exceeds the 512 MiB limit")
    inputs = value["input_frames"]
    if not isinstance(inputs, list) or len(inputs) != 2:
        raise ValueError("input_frames must contain exactly two endpoint frames")
    for frame in inputs:
        if not isinstance(frame, dict) or set(frame) != {"index", image_field}:
            raise ValueError(f"Each input frame needs exactly index and {image_field}")
        if not isinstance(frame[image_field], str) or not frame[image_field]:
            raise ValueError(f"Frame {image_field} must be a nonempty string")
    indices = _indices([frame["index"] for frame in inputs], "input indices", count)
    targets = _indices(value["target_indices"], "target_indices", count)
    if indices != [0, count - 1]:
        raise ValueError("First and last frames must be the two supplied endpoints")
    if set(indices) & set(targets) or sorted(indices + targets) != list(range(count)):
        raise ValueError("Input and target indices must partition the complete frame grid")
    return value


def validate_sample(sample):
    validate_timeline(sample, image_field="path")
    if not isinstance(sample.get("id"), str) or not sample["id"] or len(sample["id"]) > 256:
        raise ValueError("Sample id must be a nonempty string of at most 256 characters")
    # References are optional for interface checks. If present they must describe
    # exactly the missing frames. Neither reference paths nor metadata go on the wire.
    if "reference_frames" in sample:
        refs = sample["reference_frames"]
        if not isinstance(refs, list) or len(refs) != len(sample["target_indices"]):
            raise ValueError("reference_frames must match target_indices")
        for frame, index in zip(refs, sample["target_indices"], strict=True):
            if (not isinstance(frame, dict) or set(frame) != {"index", "path"}
                    or type(frame["index"]) is not int or frame["index"] != index
                    or not isinstance(frame["path"], str) or not frame["path"]):
                raise ValueError("reference_frames must match target_indices in order")
    return sample


def normalize_sample(sample):
    """Accept the uploaded factor/time JSONL format; keep the wire contract at v2."""
    if not isinstance(sample, dict) or "factor" not in sample or "schema_version" in sample:
        return validate_sample(sample)
    if set(sample) != {"id", "width", "height", "factor", "input_times", "inputs", "targets"}:
        raise ValueError("Unexpected fields in factor/time sample")
    factor = sample["factor"]
    if type(factor) is not int or factor not in FACTORS:
        raise ValueError("Sample factor must be 4, 8 or 16")
    times = sample["input_times"]
    if (not isinstance(times, list) or len(times) != 2
            or any(type(t) not in (int, float) for t in times) or times != [0, 1]):
        raise ValueError("Endpoint times must be [0, 1]")
    inputs, targets = sample["inputs"], sample["targets"]
    if not isinstance(inputs, list) or len(inputs) != 2 or not isinstance(targets, list) or len(targets) != factor - 1:
        raise ValueError("Incomplete factor/time sample")
    references = []
    for index, target in enumerate(targets, 1):
        if (not isinstance(target, dict) or set(target) != {"path", "time"}
                or type(target["time"]) not in (int, float) or target["time"] != index / factor):
            raise ValueError("Target times must cover every interior frame in order")
        references.append({"index": index, "path": target["path"]})
    return validate_sample({"id": sample["id"], "schema_version": VERSION,
                            "width": sample["width"], "height": sample["height"], "frame_count": factor + 1,
                            "input_frames": [{"index": i, "path": p} for i, p in zip((0, factor), inputs)],
                            "target_indices": list(range(1, factor)), "reference_frames": references})


def media_path(root, relative):
    """Confine all reads to the dataset root, including symlink resolution."""
    root = Path(root).resolve()
    if not isinstance(relative, str) or not relative or Path(relative).is_absolute():
        raise ValueError("Frame path must be relative to the dataset root")
    if ".." in Path(relative).parts:
        raise ValueError("Frame path must not contain parent traversal")
    path = (root / relative).resolve()
    if not path.is_relative_to(root) or not path.is_file():
        raise ValueError("Frame path escapes the dataset root or is not a regular file")
    return path


def validate_png(blob, width, height):
    """Check CRCs and bounded decompression, not just the easily forged PNG header.

    RGB samples need not be unfiltered to validate structure: each scanline must
    have the prescribed byte length and a legal filter byte. Pixel decoding for a
    future metric is separate. APNG, transparency, interlacing and trailing data
    are rejected. Ancillary metadata never affects the encoded RGB samples here.
    """
    _integer(width, "width", 1, MAX_DIMENSION)
    _integer(height, "height", 1, MAX_DIMENSION)
    if not isinstance(blob, bytes) or not 45 <= len(blob) <= MAX_PNG_BYTES:
        raise ValueError("PNG is empty, truncated or exceeds 32 MiB")
    if not blob.startswith(PNG_SIGNATURE):
        raise ValueError("Frame is not a PNG")
    offset, seen_header, seen_data, closed_data, ended = 8, False, False, False, False
    compressed = bytearray()
    while offset < len(blob):
        if offset + 12 > len(blob):
            raise ValueError("Truncated PNG chunk")
        length = struct.unpack_from(">I", blob, offset)[0]
        kind = blob[offset + 4:offset + 8]
        if any(not (65 <= char <= 90 or 97 <= char <= 122) for char in kind) or kind[2] & 32:
            raise ValueError("Invalid PNG chunk type or reserved bit")
        end = offset + 12 + length
        if end > len(blob):
            raise ValueError("Truncated PNG chunk payload")
        data = blob[offset + 8:end - 4]
        crc = struct.unpack_from(">I", blob, end - 4)[0]
        if zlib.crc32(kind + data) & 0xFFFFFFFF != crc:
            raise ValueError("PNG chunk CRC mismatch")
        if not seen_header and kind != b"IHDR":
            raise ValueError("PNG must begin with IHDR")
        if kind == b"IHDR":
            if seen_header or length != 13:
                raise ValueError("Invalid or duplicate PNG IHDR")
            header = struct.unpack(">IIBBBBB", data)
            if header != (width, height, 8, 2, 0, 0, 0):
                raise ValueError("PNG must match dimensions and be non-interlaced 8-bit RGB")
            seen_header = True
        elif kind == b"IDAT":
            if closed_data:
                raise ValueError("PNG IDAT chunks must be consecutive")
            seen_data = True
            compressed.extend(data)
        elif kind == b"IEND":
            if length or not seen_data or end != len(blob):
                raise ValueError("Invalid PNG IEND or trailing data")
            ended = True
        else:
            if kind in (b"acTL", b"fcTL", b"fdAT", b"tRNS") or not (kind[0] & 32):
                raise ValueError("Unsupported PNG chunk (animation, transparency, or critical chunk)")
        if seen_data and kind != b"IDAT":
            closed_data = True
        offset = end
    if not ended:
        raise ValueError("PNG has no IEND")
    stride = width * 3 + 1
    expected = height * stride
    decoder = zlib.decompressobj()
    try:
        scanlines = decoder.decompress(bytes(compressed), expected + 1)
    except zlib.error as exc:
        raise ValueError("Invalid PNG compressed pixels") from exc
    if (len(scanlines) != expected or not decoder.eof or decoder.unconsumed_tail or decoder.unused_data
            or any(scanlines[start] > 4 for start in range(0, expected, stride))):
        raise ValueError("Invalid PNG pixel length, filter, or compressed stream")
    return blob


def decode_png(encoded, width, height):
    if not isinstance(encoded, str) or len(encoded) > 4 * ((MAX_PNG_BYTES + 2) // 3):
        raise ValueError("PNG base64 exceeds its size limit or is not text")
    try:
        blob = base64.b64decode(encoded, validate=True)
    except (ValueError, binascii.Error) as exc:
        raise ValueError("PNG must use plain base64, without a data URL") from exc
    return validate_png(blob, width, height)


def build_request(sample, data_root):
    validate_sample(sample)
    request = {key: sample[key] for key in TIMELINE_KEYS}
    frames = []
    for frame in sample["input_frames"]:
        path = media_path(data_root, frame["path"])
        with path.open("rb") as stream:
            blob = stream.read(MAX_PNG_BYTES + 1)
        validate_png(blob, sample["width"], sample["height"])
        frames.append({"index": frame["index"], "png": base64.b64encode(blob).decode("ascii")})
    request["input_frames"] = frames
    # Leave headroom for JSON punctuation even for incompressible PNGs.
    if sum(len(frame["png"]) for frame in frames) + 65536 > MAX_JSON_BYTES:
        raise ValueError("Sequence request exceeds JSON size limit")
    return ENDPOINT, request


def parse_response(sample, response):
    """Return bounded PNG bytes; never accept paths, URLs, scores or reordered frames."""
    validate_sample(sample)
    if not isinstance(response, dict) or set(response) != {"frames"}:
        raise ValueError("Response needs exactly a frames array")
    frames = response["frames"]
    targets = sample["target_indices"]
    if not isinstance(frames, list) or len(frames) != len(targets):
        raise ValueError("Response must contain exactly the requested missing frames")
    decoded, encoded_bytes = [], 0
    for frame, index in zip(frames, targets, strict=True):
        if (not isinstance(frame, dict) or set(frame) != {"index", "png"}
                or type(frame["index"]) is not int or frame["index"] != index):
            raise ValueError("Response indices must match target_indices exactly in order")
        blob = decode_png(frame["png"], sample["width"], sample["height"])
        encoded_bytes += len(frame["png"])
        if encoded_bytes + 65536 > MAX_JSON_BYTES:
            raise ValueError("Sequence response exceeds JSON size limit")
        decoded.append({"index": index, "png": blob})
    return {"id": sample["id"], "frames": decoded}


def parse_rgb8_response(sample, body):
    """Packed RGB uint8 frames in requested order; no image codec or JSON/base64."""
    validate_sample(sample)
    stride = sample["width"] * sample["height"] * 3
    if not isinstance(body, bytes) or len(body) != stride * len(sample["target_indices"]):
        raise ValueError("Raw RGB response must contain exactly W*H*3 bytes per target frame")
    return {"id": sample["id"], "frames": [
        {"index": index, "rgb8": body[i * stride:(i + 1) * stride]}
        for i, index in enumerate(sample["target_indices"])]}
