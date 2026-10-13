"""Bounded, self-contained SVG parsing and trusted rasterization (stdlib only)."""
import math
import os
from pathlib import Path
import re
import resource
import shutil
import struct
import subprocess
import tempfile
import xml.etree.ElementTree as ET
import zlib

MAX_SVG_BYTES = 32768
MAX_ELEMENTS = 2048
MAX_DEPTH = 32
MAX_COORDINATE = 1_000_000
SIZE = 512
RENDER_SECONDS = 5
MAX_PNG_BYTES = 2 * 1024 * 1024
OFFICIAL_RENDERER_VERSION = "rsvg-convert version 2.54.7"
NS = "http://www.w3.org/2000/svg"
TAGS = {"svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "tspan", "defs", "linearGradient", "radialGradient", "stop", "clipPath"}
ATTRS = set("id viewBox width height x y x1 y1 x2 y2 cx cy r rx ry d points transform fill stroke stroke-width stroke-linecap stroke-linejoin stroke-miterlimit stroke-dasharray stroke-dashoffset fill-rule clip-rule opacity fill-opacity stroke-opacity clip-path gradientUnits gradientTransform spreadMethod offset stop-color stop-opacity font-size font-family font-weight font-style text-anchor dominant-baseline dx dy rotate preserveAspectRatio".split())
NUMERIC = set("width height x y x1 y1 x2 y2 cx cy r rx ry stroke-width stroke-miterlimit stroke-dashoffset opacity fill-opacity stroke-opacity offset stop-opacity font-size".split())
NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}\Z")
FRAGMENT = re.compile(r"url\(\s*#([A-Za-z_][A-Za-z0-9_.-]{0,127})\s*\)\Z")
PAINT = re.compile(r"(?:none|currentColor|transparent|[a-zA-Z]{1,30}|#[0-9a-fA-F]{3,8}|(?:rgb|rgba)\([0-9.,%\s]+\))\Z")


class InvalidSVG(ValueError):
    pass


class SVGBudgetExceeded(InvalidSVG):
    pass


def extract_svg(answer):
    if not isinstance(answer, str):
        raise InvalidSVG("SVG answer must be text")
    # Only one optional fence; prose or hidden material outside it is rejected.
    source = answer
    if source.startswith("```svg\n") and source.endswith("\n```"):
        source = source[7:-4]
    elif source.startswith("```xml\n") and source.endswith("\n```"):
        source = source[7:-4]
    if len(source.encode("utf-8")) > MAX_SVG_BYTES:
        raise SVGBudgetExceeded("SVG exceeds 32,768 UTF-8 bytes")
    return source


def _numbers(value):
    numbers = NUMBER.findall(value)
    for raw in numbers:
        number = float(raw)
        if not math.isfinite(number) or abs(number) > MAX_COORDINATE:
            raise InvalidSVG("SVG numbers must be finite and have magnitude at most 1,000,000")
    if re.search(r"(?i)(?:nan|infinity|\binf\b)", value):
        raise InvalidSVG("Nonfinite SVG number")
    return numbers


def sanitize_svg(answer):
    source = extract_svg(answer)
    # Comments are inert but count toward the byte limit. No XML declarations,
    # DTDs, entities, processing instructions or alternate encodings are accepted.
    without_comments = re.sub(r"<!--[\s\S]*?-->", "", source)
    if "<!" in without_comments or "<?" in without_comments:
        raise InvalidSVG("XML declarations, entities and processing instructions are forbidden")
    try:
        root = ET.fromstring(source)
    except (ET.ParseError, ValueError) as error:
        raise InvalidSVG("Malformed SVG XML") from error
    if root.tag not in ("svg", "{" + NS + "}svg"):
        raise InvalidSVG("Root element must be svg")
    ids, references, element_count = {}, [], 0

    def walk(node, depth):
        nonlocal element_count
        element_count += 1
        if depth > MAX_DEPTH or element_count > MAX_ELEMENTS:
            raise InvalidSVG("SVG tree exceeds depth or element limit")
        name = node.tag
        if name.startswith("{" + NS + "}"):
            name = name[len(NS) + 2:]
        if name not in TAGS or (node is not root and name == "svg"):
            raise InvalidSVG("Unsupported SVG element")
        for attr, value in node.attrib.items():
            if attr not in ATTRS or len(value) > MAX_SVG_BYTES or any(ord(c) < 32 and c not in "\n\r\t" for c in value):
                raise InvalidSVG("Unsupported SVG attribute")
            if attr == "id":
                if not ID.fullmatch(value) or value in ids:
                    raise InvalidSVG("Invalid or duplicate SVG ID")
                ids[value] = name
            elif attr in {"fill", "stroke", "clip-path"}:
                match = FRAGMENT.fullmatch(value)
                if match:
                    references.append((attr, match[1]))
                elif attr == "clip-path" and value != "none" or attr != "clip-path" and not PAINT.fullmatch(value):
                    raise InvalidSVG("Only simple colors and internal fragment paint references are allowed")
            elif attr in {"stop-color"} and not PAINT.fullmatch(value):
                raise InvalidSVG("Invalid stop color")
            elif attr in NUMERIC:
                raw = value.removesuffix("%")
                if not NUMBER.fullmatch(raw):
                    raise InvalidSVG("Numeric attributes do not accept units or expressions")
                _numbers(raw)
            elif attr in {"viewBox", "d", "points", "transform", "gradientTransform", "stroke-dasharray", "dx", "dy", "rotate"}:
                _numbers(value)
                # Forbid escapes, XML/CSS resource syntax and uncontrolled strings.
                if re.search(r"[^A-Za-z0-9+.,()\s-]", value):
                    raise InvalidSVG("Invalid geometry syntax")
            elif any(c in value for c in "\\:;@{}<>/") or "url" in value.casefold():
                raise InvalidSVG("External resources and CSS are forbidden")
        if node.text and node.text.strip() and name not in {"text", "tspan"}:
            raise InvalidSVG("Text content is allowed only inside text and tspan")
        for child in node:
            # A child's tail is content of its parent, including mixed text/tspan.
            if child.tail and child.tail.strip() and name not in {"text", "tspan"}:
                raise InvalidSVG("Unexpected text outside SVG text elements")
            walk(child, depth + 1)

    walk(root, 0)
    try:
        viewbox = [float(v) for v in re.split(r"[\s,]+", root.attrib["viewBox"].strip())]
    except (KeyError, ValueError):
        raise InvalidSVG("A numeric viewBox is required") from None
    if len(viewbox) != 4 or not all(math.isfinite(v) and abs(v) <= MAX_COORDINATE for v in viewbox) or min(viewbox[2:]) <= 0:
        raise InvalidSVG("Invalid viewBox")
    for attr, target in references:
        allowed = {"clipPath"} if attr == "clip-path" else {"linearGradient", "radialGradient"}
        if ids.get(target) not in allowed:
            raise InvalidSVG("Unresolved or invalid internal SVG reference")
    # Fix the canvas and default font. Child font-family attributes may override
    # this default with another installed font; external fonts remain forbidden.
    root.set("width", str(SIZE))
    root.set("height", str(SIZE))
    root.set("font-family", "DejaVu Sans")
    return ET.tostring(root, encoding="utf-8"), {"svg_bytes": len(source.encode("utf-8")), "elements": element_count}


def renderer_binary(binary=None, *, official=False):
    resolved = shutil.which(binary or "rsvg-convert")
    if not resolved:
        raise RuntimeError("Install librsvg2-bin: rsvg-convert is required")
    result = subprocess.run([resolved, "--version"], capture_output=True, text=True, timeout=5, check=True)
    version = result.stdout.strip()
    if not version.startswith("rsvg-convert version ") or official and version != OFFICIAL_RENDERER_VERSION:
        raise RuntimeError(f"Official renderer requires {OFFICIAL_RENDERER_VERSION}; found {version}")
    return resolved, version


def _limits():
    resource.setrlimit(resource.RLIMIT_CPU, (3, 3))
    resource.setrlimit(resource.RLIMIT_AS, (512 * 1024**2, 512 * 1024**2))
    resource.setrlimit(resource.RLIMIT_FSIZE, (MAX_PNG_BYTES, MAX_PNG_BYTES))
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


def _png_nonuniform(png):
    """Validate a bounded PNG and detect uniform canvases without Pillow."""
    if not png.startswith(b"\x89PNG\r\n\x1a\n") or len(png) > MAX_PNG_BYTES:
        raise InvalidSVG("Renderer did not return a bounded PNG")
    offset, compressed, channels = 8, bytearray(), None
    while offset + 12 <= len(png):
        length = struct.unpack(">I", png[offset:offset+4])[0]
        kind = png[offset+4:offset+8]
        payload = png[offset+8:offset+8+length]
        if offset + 12 + length > len(png):
            raise InvalidSVG("Truncated PNG")
        if kind == b"IHDR":
            width, height, depth, color, compression, filtering, interlace = struct.unpack(">IIBBBBB", payload)
            if (width, height, depth, compression, filtering, interlace) != (SIZE, SIZE, 8, 0, 0, 0) or color not in (2, 6):
                raise InvalidSVG("Unexpected PNG format")
            channels = 3 if color == 2 else 4
        elif kind == b"IDAT":
            compressed.extend(payload)
        elif kind == b"IEND":
            break
        offset += length + 12
    if channels is None:
        raise InvalidSVG("Missing PNG header")
    row_size = SIZE * channels
    expected = (row_size + 1) * SIZE
    try:
        inflater = zlib.decompressobj()
        raw = inflater.decompress(bytes(compressed), expected + 1)
        if len(raw) != expected or not inflater.eof or inflater.unused_data:
            raise InvalidSVG("Invalid or oversized PNG pixels")
    except zlib.error as error:
        raise InvalidSVG("Invalid PNG compression") from error
    previous = bytearray(row_size)
    first_pixel = None
    nonuniform = False
    for y in range(SIZE):
        start = y * (row_size + 1)
        mode = raw[start]
        row = bytearray(raw[start+1:start+1+row_size])
        if mode > 4:
            raise InvalidSVG("Invalid PNG filter")
        for x in range(row_size):
            left = row[x-channels] if x >= channels else 0
            up = previous[x]
            ul = previous[x-channels] if x >= channels else 0
            if mode == 1: predictor = left
            elif mode == 2: predictor = up
            elif mode == 3: predictor = (left + up) // 2
            elif mode == 4:
                p = left + up - ul
                distances = (abs(p-left), abs(p-up), abs(p-ul))
                predictor = (left, up, ul)[distances.index(min(distances))]
            else: predictor = 0
            row[x] = (row[x] + predictor) & 255
        if first_pixel is None:
            first_pixel = bytes(row[:channels])
        if row != first_pixel * SIZE:
            nonuniform = True
        previous = row
    return nonuniform


def render_svg(answer, *, binary=None):
    sanitized, details = sanitize_svg(answer)
    binary = binary or renderer_binary()[0]
    with tempfile.TemporaryDirectory(prefix="svg-render-") as temp:
        svg_path, png_path = Path(temp) / "input.svg", Path(temp) / "output.png"
        svg_path.write_bytes(sanitized)
        try:
            subprocess.run([binary, "--format=png", "--width=512", "--height=512", "--keep-aspect-ratio", "--background-color=white", "--output", str(png_path), str(svg_path)],
                           stdin=subprocess.DEVNULL, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                           timeout=RENDER_SECONDS, check=True, preexec_fn=_limits, close_fds=True,
                           env={"PATH": os.defpath, "LANG": "C.UTF-8", "HOME": temp})
            png = png_path.read_bytes()
        except (subprocess.SubprocessError, OSError) as error:
            raise InvalidSVG("SVG renderer failed or exceeded its resource budget") from error
    details["nonuniform"] = _png_nonuniform(png)
    return png, details
