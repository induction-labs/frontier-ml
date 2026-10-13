"""Exact source-token metric and proof-only input policy (standard library only).

A token is an identifier/number (including qualified names), a quoted identifier,
a literal, a listed multi-character operator, or one remaining non-space character.
Comments nest; literals protect comment markers. An initial `by` is not counted.
"""
import unicodedata

MAX_PROOF_BYTES = 65_536
MAX_STATEMENT_BYTES = 32_768
OPERATORS = tuple(sorted((":=", "!=", "&&", "-.", "->", "<-", "...", "..", "::", ":>",
                          "<;>", ";;", "==", "||", "=>", "<=", ">=", "⁻¹", "?_"), key=len, reverse=True))


def _word(c):
    return c.isalnum() or c in "_'" or unicodedata.category(c) in {"Mn", "Mc"}


def lex(text):
    """Return (kind, spelling) pairs; malformed comments/literals raise ValueError."""
    if not isinstance(text, str):
        raise ValueError("Lean source must be a string")
    out, i, n = [], 0, len(text)
    while i < n:
        c = text[i]
        if c.isspace():
            i += 1
            continue
        if unicodedata.category(c) in {"Cc", "Cf", "Cs"}:
            raise ValueError("Control, format and surrogate characters are unsupported")
        if text.startswith("--", i):
            j = text.find("\n", i + 2)
            i = n if j < 0 else j + 1
            continue
        if text.startswith("/-", i):
            i, depth = i + 2, 1
            while i < n and depth:
                if text.startswith("/-", i):
                    depth, i = depth + 1, i + 2
                elif text.startswith("-/", i):
                    depth, i = depth - 1, i + 2
                else:
                    i += 1
            if depth:
                raise ValueError("Unterminated block comment")
            continue
        start = i
        if c == "«":
            j = text.find("»", i + 1)
            if j < 0:
                raise ValueError("Unterminated quoted identifier")
            i = j + 1
            out.append(("quoted", text[start:i]))
            continue
        # Lean raw strings r###"..."###.
        if c == "r":
            j = i + 1
            while j < n and text[j] == "#":
                j += 1
            if j < n and text[j] == '"':
                ending = '"' + "#" * (j - i - 1)
                k = text.find(ending, j + 1)
                if k < 0:
                    raise ValueError("Unterminated raw string")
                i = k + len(ending)
                out.append(("literal", text[start:i]))
                continue
        if c in {'"', "'"}:
            quote, i = c, i + 1
            while i < n and text[i] != quote:
                if text[i] == "\\":
                    i += 1
                i += 1
            if i >= n:
                raise ValueError("Unterminated literal")
            i += 1
            out.append(("literal", text[start:i]))
            continue
        op = next((op for op in OPERATORS if text.startswith(op, i)), None)
        if op:
            out.append(("symbol", op))
            i += len(op)
        elif _word(c):
            i += 1
            while i < n:
                if _word(text[i]):
                    i += 1
                elif text[i] == "." and i + 1 < n and _word(text[i + 1]):
                    i += 1
                else:
                    break
            out.append(("word", text[start:i]))
        else:
            out.append(("symbol", c))
            i += 1
    return out


def proof_length(proof):
    tokens = lex(proof)
    if tokens and tokens[0] == ("word", "by"):
        tokens = tokens[1:]
    return len(tokens)


def proof_policy(proof):
    """Return a deterministic rejection reason, or None. Kernel checks are separate."""
    try:
        if len(proof.encode("utf-8")) > MAX_PROOF_BYTES:
            return "proof byte limit"
        tokens = lex(proof)
    except (AttributeError, UnicodeError, ValueError) as exc:
        return str(exc)
    if not tokens or proof_length(proof) == 0:
        return "empty proof"
    for kind, token in tokens:
        # The declared proof language excludes literals and quotation syntax.
        # Keywords are checked on Lean's parsed syntax, where binder names are
        # identifiers rather than commands. Ordinary $ application is supported.
        if kind in {"literal", "quoted"} or token in {"#", "`"}:
            return "literal, quotation or command syntax"
    return None


def reduction(original_length, candidate_length, valid):
    if type(valid) is not bool or type(original_length) is not int or original_length <= 0:
        raise ValueError("Expected positive original length and boolean validity")
    if type(candidate_length) is not int or candidate_length < 0:
        raise ValueError("Expected nonnegative candidate length")
    return max(0.0, 1.0 - candidate_length / original_length) if valid and candidate_length else 0.0
