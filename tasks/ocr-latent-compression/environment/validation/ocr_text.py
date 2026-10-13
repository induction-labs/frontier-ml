"""Pinned text normalization and normalized edit distance for page transcription.

Normalization version ``ocr-text-v1`` (applied identically to prediction and reference):

1. Unicode NFKC.
2. HTML tags (``<...>``) become a space; HTML entities are unescaped.
3. Markdown images/links ``[text](target)`` keep only ``text``.
4. Per line: leading heading/quote/bullet markers (``#``, ``>``, ``-``, ``+``, ``*``,
   ``•`` followed by whitespace) are removed; table separator rows (``|---|:-:|``) and
   horizontal rules (``---``) are dropped.
5. The characters ``* _ # | ` ~`` and LaTeX delimiters ``\\( \\) \\[ \\]`` are deleted.
6. All whitespace runs become one space; leading/trailing space is removed.

Case, digits and remaining punctuation are kept. References never contain the
deleted characters, so deleting them cannot remove reference content.

Headline per-image score (``similarity``): ``1 - word_ned``, where ``word_ned`` is the
unit-cost Levenshtein distance between the whitespace-token sequences of the two
normalized strings divided by the longer token count (0 when both are empty). Word
level is used because character-level NED rewards content-free text: an unrelated
fluent page of the same length scores about 0.24 under character 1-NED but about 0.04
under word 1-NED. ``char_similarity`` (character 1-NED, Fox-style) and ``word_f1``
(order-insensitive) are diagnostics only.
"""
import html
import re
import unicodedata

VERSION = "ocr-text-v1"
_TAG = re.compile(r"<[^<>]*>")
_LINK = re.compile(r"!?\[([^\]]*)\]\([^)\s]*\)")
_LEADING = re.compile(r"^(?:#{1,6}|>|[-+*\u2022])\s+")
_SEPARATOR = re.compile(r"^\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?$")
_DELETE = str.maketrans("", "", "*_#|`~")
_LATEX = re.compile(r"\\[()\[\]]")
_SPACE = re.compile(r"\s+")


def normalize(text):
    if not isinstance(text, str):
        raise TypeError("Text must be a string")
    text = unicodedata.normalize("NFKC", text)
    text = html.unescape(_TAG.sub(" ", text))
    text = _LINK.sub(r"\1", text)
    lines = []
    for line in text.splitlines():
        line = line.strip()
        if _SEPARATOR.match(line) and ("|" in line or line.count("-") >= 3):
            continue
        while True:
            stripped = _LEADING.sub("", line, count=1)
            if stripped == line:
                break
            line = stripped
        lines.append(line)
    text = _LATEX.sub("", "\n".join(lines)).translate(_DELETE)
    return _SPACE.sub(" ", text).strip()


def levenshtein(a, b):
    """Unit-cost edit distance over two sequences of hashable symbols (characters of a
    string or word tokens of a list); Myers/Hyyro bit-parallel algorithm, exact."""
    if len(a) < len(b):
        a, b = b, a
    m = len(b)
    if m == 0:
        return len(a)
    peq = {}
    for index, char in enumerate(b):
        peq[char] = peq.get(char, 0) | (1 << index)
    mask = (1 << m) - 1
    high = 1 << (m - 1)
    pv, mv, score = mask, 0, m
    for char in a:
        eq = peq.get(char, 0)
        xv = eq | mv
        xh = (((eq & pv) + pv) ^ pv) | eq
        ph = mv | (~(xh | pv) & mask)
        mh = pv & xh
        if ph & high:
            score += 1
        elif mh & high:
            score -= 1
        ph = ((ph << 1) | 1) & mask
        mh = (mh << 1) & mask
        pv = mh | (~(xv | ph) & mask)
        mv = ph & xv
    return score


def ned(prediction, reference):
    """Normalized edit distance of two already-normalized sequences, in [0, 1]."""
    longest = max(len(prediction), len(reference))
    return levenshtein(prediction, reference) / longest if longest else 0.0


def word_ned(prediction, reference):
    """Word-level NED of two already-normalized strings (tokens split on single spaces)."""
    return ned(prediction.split(), reference.split())


def similarity(prediction, reference):
    """Headline per-image score: 1 - word-level NED after the pinned normalization."""
    return 1.0 - word_ned(normalize(prediction), normalize(reference))


def char_similarity(prediction, reference):
    """Diagnostic: 1 - character-level NED after the pinned normalization (Fox-style)."""
    return 1.0 - ned(normalize(prediction), normalize(reference))


def word_f1(prediction, reference):
    """Order-insensitive diagnostic: multiset F1 over whitespace tokens (not the headline)."""
    from collections import Counter
    predicted, expected = Counter(normalize(prediction).split()), Counter(normalize(reference).split())
    overlap = sum((predicted & expected).values())
    if not predicted and not expected:
        return 1.0
    if not overlap:
        return 0.0
    precision, recall = overlap / sum(predicted.values()), overlap / sum(expected.values())
    return 2 * precision * recall / (precision + recall)
