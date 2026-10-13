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
SIZE = 512
RENDER_SECONDS = 5
MAX_PNG_BYTES = 2 * 1024 * 1024
OFFICIAL_RENDERER_VERSION = "rsvg-convert version 2.54.7"
NS = "http://www.w3.org/2000/svg"
XML_NS = "http://www.w3.org/XML/1998/namespace"
TAGS = {"svg", "g", "path", "rect", "circle", "ellipse", "line", "polyline", "polygon", "text", "tspan", "defs", "linearGradient", "radialGradient", "stop", "clipPath"}
# Structural, geometric and presentation attributes of the allowed elements. Styling
# (`style`, CSS), references (`href`), events (`on*`) and foreign namespaces are excluded.
ATTRS = set("""id class version baseProfile lang
    viewBox width height x y x1 y1 x2 y2 cx cy r rx ry fx fy fr d points pathLength transform preserveAspectRatio
    fill fill-opacity fill-rule stroke stroke-width stroke-linecap stroke-linejoin stroke-miterlimit stroke-dasharray
    stroke-dashoffset stroke-opacity opacity color paint-order vector-effect shape-rendering color-rendering
    color-interpolation display visibility overflow enable-background clip-path clip-rule clipPathUnits
    gradientUnits gradientTransform spreadMethod offset stop-color stop-opacity
    font-family font-size font-weight font-style font-variant font-stretch font-size-adjust letter-spacing word-spacing
    text-anchor dominant-baseline alignment-baseline baseline-shift text-decoration text-rendering direction
    unicode-bidi writing-mode dx dy rotate textLength lengthAdjust""".split()) | {"{" + XML_NS + "}space", "{" + XML_NS + "}lang"}
LENGTHS = set("width height x y x1 y1 x2 y2 cx cy r rx ry fx fy fr stroke-width stroke-miterlimit stroke-dashoffset opacity fill-opacity stroke-opacity offset stop-opacity pathLength textLength".split())
LENGTH_OR_KEYWORD = {"font-size", "font-size-adjust", "letter-spacing", "word-spacing", "baseline-shift"}
LISTS = {"viewBox", "d", "points", "transform", "gradientTransform", "stroke-dasharray", "dx", "dy", "rotate"}
PAINTS = {"fill", "stroke", "stop-color", "color"}
NUMBER = re.compile(r"[-+]?(?:\d*\.\d+|\d+\.?\d*)(?:[eE][-+]?\d+)?")
LENGTH = re.compile(NUMBER.pattern + r"(?:%|px|pt|pc|mm|cm|in|em|ex)?\Z")
KEYWORD = re.compile(r"[A-Za-z-]{1,24}\Z")
ID = re.compile(r"[A-Za-z_][A-Za-z0-9_.-]{0,127}\Z")
FRAGMENT = re.compile(r"url\(\s*(['\"]?)#([A-Za-z_][A-Za-z0-9_.-]{0,127})\1\s*\)")
COLOR = r"(?:none|currentColor|transparent|inherit|[a-zA-Z]{1,30}|#[0-9a-fA-F]{3,8}|(?:rgb|rgba|hsl|hsla)\([0-9.,%/\sdeg-]+\))"
PAINT = re.compile(COLOR + r"\Z")
# SVG paint: an internal fragment reference with an optional fallback color, or a color.
PAINT_REFERENCE = re.compile(FRAGMENT.pattern + r"(?:\s+" + COLOR + r")?\Z")
FENCE = re.compile(r"```[A-Za-z0-9_+-]*[ \t]*\r?\n?([\s\S]*?)\r?\n?[ \t]*```\Z")


class InvalidSVG(ValueError):
    pass


class SVGBudgetExceeded(InvalidSVG):
    pass


def extract_svg(answer):
    """The SVG document inside a chat answer: an optional code fence (any or no language tag),
    an XML declaration or brief prose around the outermost <svg ...>...</svg> are tolerated and
    excluded from the byte budget. Only the document itself is validated, rendered and judged."""
    if not isinstance(answer, str):
        raise InvalidSVG("SVG answer must be text")
    source = answer.strip()
    fenced = FENCE.fullmatch(source)
    if fenced:
        source = fenced.group(1)
    start, end = source.find("<svg"), source.rfind("</svg>")
    if start >= 0 and end > start:
        source = source[start:end + len("</svg>")]
    if len(source.encode("utf-8")) > MAX_SVG_BYTES:
        raise SVGBudgetExceeded("SVG exceeds 32,768 UTF-8 bytes")
    return source


def _numbers(value):
    numbers = NUMBER.findall(value)
    for raw in numbers:
        number = float(raw)
        if not math.isfinite(number):
            raise InvalidSVG("SVG numbers must be finite")
    if re.search(r"(?i)(?:nan|infinity|\binf\b)", value):
        raise InvalidSVG("Nonfinite SVG number")
    return numbers


def sanitize_svg(answer):
    source = extract_svg(answer)
    # Comments and CDATA are inert but count toward the byte limit. No XML declarations,
    # DTDs, entities, processing instructions or alternate encodings are accepted.
    without_comments = re.sub(r"<!--[\s\S]*?-->", "", source)
    if re.search(r"<!(?!\[CDATA\[)", without_comments) or "<?" in without_comments:
        raise InvalidSVG("XML declarations, entities and processing instructions are forbidden")
    try:
        root = ET.fromstring(source)
    except (ET.ParseError, ValueError) as error:
        raise InvalidSVG("Malformed SVG XML") from error
    if root.tag not in ("svg", "{" + NS + "}svg"):
        raise InvalidSVG("Root element must be svg")
    ids, references, element_count = {}, [], 0

    for node in root.iter():
        element_count += 1
        name = node.tag
        if name.startswith("{" + NS + "}"):
            name = name[len(NS) + 2:]
        if name not in TAGS:
            raise InvalidSVG("Unsupported SVG element")
        for attr, value in node.attrib.items():
            if attr not in ATTRS or len(value) > MAX_SVG_BYTES or any(ord(c) < 32 and c not in "\n\r\t" for c in value):
                raise InvalidSVG("Unsupported SVG attribute")
            if attr == "id":
                if not ID.fullmatch(value) or value in ids:
                    raise InvalidSVG("Invalid or duplicate SVG ID")
                ids[value] = name
            elif attr == "clip-path":
                match = FRAGMENT.fullmatch(value)
                if match:
                    references.append((attr, match[2]))
                elif value != "none":
                    raise InvalidSVG("Only internal clip path references are allowed")
            elif attr in PAINTS:
                match = PAINT_REFERENCE.fullmatch(value)
                if match:
                    references.append((attr, match[2]))
                elif not PAINT.fullmatch(value):
                    raise InvalidSVG("Only simple colors and internal fragment paint references are allowed")
            elif attr in LENGTHS:
                if not LENGTH.fullmatch(value):
                    raise InvalidSVG("Numeric attributes accept only unitless numbers, percentages or standard length units")
                _numbers(value)
            elif attr in LENGTH_OR_KEYWORD:
                if not (LENGTH.fullmatch(value) or KEYWORD.fullmatch(value)):
                    raise InvalidSVG("Invalid length or keyword")
                _numbers(value)
            elif attr in LISTS:
                _numbers(value)
                # Forbid escapes, XML/CSS resource syntax and uncontrolled strings.
                if re.search(r"[^A-Za-z0-9+.,()%\s-]", value):
                    raise InvalidSVG("Invalid geometry syntax")
            elif any(c in value for c in "\\:;@{}<>/") or "url" in value.casefold():
                raise InvalidSVG("External resources and CSS are forbidden")
        if node.text and node.text.strip() and name not in {"text", "tspan"}:
            raise InvalidSVG("Text content is allowed only inside text and tspan")
        for child in node:
            # A child's tail is content of its parent, including mixed text/tspan.
            if child.tail and child.tail.strip() and name not in {"text", "tspan"}:
                raise InvalidSVG("Unexpected text outside SVG text elements")
    try:
        viewbox = [float(v) for v in re.split(r"[\s,]+", root.attrib["viewBox"].strip())]
    except (KeyError, ValueError):
        raise InvalidSVG("A numeric viewBox is required") from None
    if len(viewbox) != 4 or not all(math.isfinite(v) for v in viewbox) or min(viewbox[2:]) <= 0:
        raise InvalidSVG("Invalid viewBox")
    for attr, target in references:
        allowed = {"clipPath"} if attr == "clip-path" else {"linearGradient", "radialGradient"}
        if ids.get(target) not in allowed:
            raise InvalidSVG("Unresolved or invalid internal SVG reference")
    # Fix the canvas and default font. Child font-family attributes may override
    # this default with another installed font; external fonts remain forbidden.
    root.set("width", str(SIZE))
    root.set("height", str(SIZE))
    if "font-family" not in root.attrib:
        root.set("font-family", "DejaVu Sans")
    # Serialize iteratively so Python recursion depth is not another hidden
    # drawing constraint. The XML parser and isolated renderer own their budgets.
    qnames, namespaces = ET._namespaces(root, None)
    output = []
    pending = [(root, False)]
    while pending:
        node, closing = pending.pop()
        name = qnames[node.tag]
        if closing:
            output.append("</" + name + ">")
            if node.tail:
                output.append(ET._escape_cdata(node.tail))
        else:
            output.append("<" + name)
            if node is root:
                for uri, prefix in sorted(namespaces.items(), key=lambda pair: pair[1]):
                    key = "xmlns" + (":" + prefix if prefix else "")
                    output.append(' ' + key + '="' + ET._escape_attrib(uri) + '"')
            for key, value in node.attrib.items():
                output.append(' ' + qnames[key] + '="' + ET._escape_attrib(value) + '"')
            if len(node) or node.text:
                output.append(">")
                if node.text:
                    output.append(ET._escape_cdata(node.text))
                pending.append((node, True))
                pending.extend((child, False) for child in reversed(node))
            else:
                output.append(" />")
                if node.tail:
                    output.append(ET._escape_cdata(node.tail))
    return "".join(output).encode("utf-8"), {"svg_bytes": len(source.encode("utf-8")), "elements": element_count}


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
