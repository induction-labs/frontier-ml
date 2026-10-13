"""Turn raw prover outputs into canonical items, dropping disallowed attempts.

Item text = pinned HEADER + optional helper declarations + the exact statement + proof.
Dropped: no Lean block, statement not reproduced, sorry/admit/native_decide/axiom,
extra imports, set_option (including heartbeat overrides), metaprogramming/custom
elaborators, ``#`` commands, or implausibly long code.
"""
from __future__ import annotations

import re

from lean_check import HEADER



def code_blocks(raw: str) -> list[str]:
    """Fenced blocks tagged lean4/lean (or untagged), pairing fences line by line."""
    blocks, current, language = [], None, None
    for line in raw.split("\n"):
        stripped = line.strip()
        if current is None:
            if stripped.startswith("```"):
                current, language = [], stripped[3:].strip().lower()
        elif stripped.startswith("```"):
            if language in ("lean4", "lean", ""):
                blocks.append("\n".join(current) + "\n")
            current = None
        else:
            current.append(line)
    return blocks
DISALLOWED = re.compile(
    r"\b(sorry|admit|native_decide|axiom|import|set_option|elab|macro|macro_rules|syntax|run_tac|run_cmd|"
    r"implemented_by|extern|unsafe|ofReduceBool|decide\s*:=\s*true)\b|"
    r"\bopen\s+Lean\b|\bLean\.(Elab|Meta|Parser|Compiler)\b|^\s*#|@\[\s*(extern|implemented_by|csimp)", re.M)
MAX_CHARS = 12000


def _ws(text: str) -> str:
    return re.sub(r"\s+", " ", text).strip()


def canonical_item(raw: str, statement: str) -> tuple[str | None, str]:
    """Return (item text, reason). The statement ends with ':= by'."""
    blocks = code_blocks(raw)
    if not blocks:
        return None, "no_block"
    code = blocks[-1].replace("\r\n", "\n")
    # Strip a leading header (imports / opens / options before the first declaration).
    lines = code.split("\n")
    start = 0
    while start < len(lines) and (lines[start].strip() == "" or re.match(r"\s*(import|open|set_option)\b", lines[start])):
        if re.match(r"\s*(set_option|open)\b.*\bin\s*$", lines[start]):
            return None, "disallowed"  # scoped to the theorem: part of the attempt, not the header
        start += 1
    code = "\n".join(lines[start:])
    # Also drop leading doc comments on the target theorem only if they precede everything.
    index = code.find(statement)
    if index < 0:
        # Accept whitespace-only reformatting of the statement and restore the exact text.
        head = statement.split(":= by")[0]
        pattern = r"\s+".join(re.escape(tok) for tok in _ws(head).split(" ")) + r"\s*:=\s*by"
        match = re.search(pattern, code)
        if not match:
            return None, "statement_changed"
        code = code[:match.start()] + statement + code[match.end():]
        index = match.start()
    prefix, rest = code[:index], code[index + len(statement):]
    prefix = re.sub(r"/--.*?-/\s*$", "", prefix, flags=re.S)  # docstring attached to the theorem
    if statement in rest:
        return None, "duplicate_statement"
    body = (prefix.strip() + "\n\n" if prefix.strip() else "") + statement + rest.rstrip()
    if DISALLOWED.search(body.replace(statement, "")):
        return None, "disallowed"
    if len(body) > MAX_CHARS:
        return None, "too_long"
    if not rest.strip():
        return None, "empty_proof"
    return HEADER + body + "\n", "ok"


if __name__ == "__main__":
    import argparse
    import collections
    import json

    parser = argparse.ArgumentParser()
    parser.add_argument("statements")
    parser.add_argument("output")
    parser.add_argument("generations", nargs="+")
    args = parser.parse_args()
    statements = {r["sid"]: r for r in map(json.loads, open(args.statements))}
    reasons = collections.Counter()
    seen = set()
    with open(args.output, "w") as stream:
        for path in args.generations:
            for line in open(path):
                row = json.loads(line)
                if row.get("finish_reason") != "stop":
                    reasons["truncated"] += 1
                    continue
                item, reason = canonical_item(row["text"], statements[row["sid"]]["statement"])
                reasons[reason] += 1
                if item is None or (row["sid"], item) in seen:
                    reasons["duplicate"] += item is not None
                    continue
                seen.add((row["sid"], item))
                stream.write(json.dumps({"sid": row["sid"], "origin": row["prover"], "parent": None,
                                         "mutation": None, "lean": item}, ensure_ascii=False) + "\n")
    print(dict(reasons))
