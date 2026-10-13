"""Near-miss and neutral edits of prover attempts (organizer side).

Every mutant is labeled by the compiler like any other attempt; the operator names are
only generation strategies. Both kinds are applied to passing *and* failing parents so
that "this proof was edited" does not by itself predict the label.
"""
from __future__ import annotations

import random
import re

from lean_check import HEADER

IDENT = r"[A-Za-z_][\w'.₀-₉]*"
NUM = re.compile(r"(?<![\w'₀-₉.])(\d+)(?![\w'₀-₉])")
HINT_LIST = re.compile(r"\b(nlinarith|linarith|positivity|simp|simp_all|norm_num|field_simp|omega|simp only|rw)\s*\[([^\[\]]*)\]")
TACTIC_SWAP = {
    "nlinarith": ["linarith", "positivity"], "linarith": ["nlinarith", "omega"], "omega": ["linarith", "simp"],
    "norm_num": ["simp", "linarith"], "simp": ["norm_num", "simp_all"], "ring_nf": ["ring", "norm_num"],
    "ring": ["ring_nf", "linarith"], "positivity": ["nlinarith", "norm_num"], "field_simp": ["ring_nf", "simp"],
    "decide": ["norm_num", "rfl"], "simp_all": ["simp", "aesop"], "rfl": ["norm_num", "simp"],
    "aesop": ["simp_all", "tauto"], "tauto": ["aesop", "simp"], "exact": ["apply"], "nlinarith!": ["nlinarith"],
}
LEMMA_SWAP = [("_le_", "_lt_"), ("_lt_", "_le_"), ("_pos", "_nonneg"), ("_nonneg", "_pos"), ("_left", "_right"),
              ("_right", "_left"), ("add_", "mul_"), ("mul_", "add_"), ("sub_", "add_"), ("_comm", "_assoc"),
              ("Nat.", "Int."), ("sq_nonneg", "sq_abs"), ("div_", "mul_")]
REL_SWAP = {"≤": "<", "<": "≤", "≥": ">", ">": "≥", "=": "≤"}
COMMENTS = ["-- simplify the goal", "-- main estimate", "-- use the hypotheses", "-- key step", "-- close the goal",
            "-- rewrite", "-- case analysis", "-- auxiliary fact"]


def split(item: str, statement: str) -> tuple[str, str]:
    body = item[len(HEADER):]
    index = body.index(statement) + len(statement)
    return HEADER + body[:index], body[index:]


def _replace_one(text: str, pattern: re.Pattern, rng, function) -> str | None:
    matches = list(pattern.finditer(text))
    if not matches:
        return None
    m = rng.choice(matches)
    new = function(m)
    if new is None or new == m.group(0):
        return None
    return text[:m.start()] + new + text[m.end():]


# ---- near-miss operators -------------------------------------------------------------
def op_number(proof, rng):
    def change(m):
        value = int(m.group(1))
        return str(rng.choice([v for v in (value + 1, value - 1, value + 2, value * 2) if v >= 0 and v != value]))
    return _replace_one(proof, NUM, rng, change)


def op_drop_hint(proof, rng):
    def drop(m):
        items = [x for x in re.split(r",(?![^()]*\))", m.group(2)) if x.strip()]
        if len(items) < 2:
            return None
        items.pop(rng.randrange(len(items)))
        return f"{m.group(1)} [{','.join(items).strip()}]"
    return _replace_one(proof, HINT_LIST, rng, drop)


def op_swap_tactic(proof, rng):
    pattern = re.compile(r"(?<![\w.'])(" + "|".join(sorted(map(re.escape, TACTIC_SWAP), key=len, reverse=True)) + r")(?![\w'!?])")
    return _replace_one(proof, pattern, rng, lambda m: rng.choice(TACTIC_SWAP[m.group(1)]))


def op_swap_lemma(proof, rng):
    idents = [m for m in re.finditer(IDENT, proof) if "_" in m.group(0) or "." in m.group(0)]
    rng.shuffle(idents)
    for m in idents:
        name = m.group(0)
        options = [(a, b) for a, b in LEMMA_SWAP if a in name]
        if options:
            a, b = rng.choice(options)
            return proof[:m.start()] + name.replace(a, b, 1) + proof[m.end():]
    return None


def op_swap_hypothesis(proof, rng, statement=""):
    names = sorted(set(re.findall(r"(?<![\w'₀-₉.])(h[\w'₀-₉]*)(?![\w'₀-₉])", statement + proof)))
    uses = [m for m in re.finditer(r"(?<![\w'₀-₉.])(h[\w'₀-₉]*)(?![\w'₀-₉])", proof)]
    if len(names) < 2 or not uses:
        return None
    m = rng.choice(uses)
    other = rng.choice([n for n in names if n != m.group(1)])
    return proof[:m.start()] + other + proof[m.end():]


def op_drop_line(proof, rng):
    lines = proof.split("\n")
    candidates = [i for i, line in enumerate(lines) if line.strip() and not line.strip().startswith("--")]
    if len(candidates) < 2:
        return None
    lines.pop(rng.choice(candidates[1:] if len(candidates) > 2 else candidates))
    return "\n".join(lines)


def op_swap_relation(proof, rng):
    pattern = re.compile(r"(have\s+[^\n:]*:[^\n]*?)(≤|≥|<|>)")
    return _replace_one(proof, pattern, rng, lambda m: m.group(1) + REL_SWAP[m.group(2)])


NEAR_MISS = {"number": op_number, "drop_hint": op_drop_hint, "swap_tactic": op_swap_tactic,
             "swap_lemma": op_swap_lemma, "swap_hypothesis": op_swap_hypothesis, "drop_line": op_drop_line,
             "swap_relation": op_swap_relation}


# ---- neutral operators ---------------------------------------------------------------
def op_rename_have(proof, rng):
    names = list(dict.fromkeys(re.findall(r"\bhave\s+(h[\w'₀-₉]*)\s*:", proof)))
    if not names:
        return None
    old = rng.choice(names)
    new = rng.choice([old + "'", old + "_aux", "h_" + old.lstrip("h") if old != "h" else "h_main", old + "x"])
    if re.search(r"(?<![\w'₀-₉.])" + re.escape(new) + r"(?![\w'₀-₉])", proof):
        return None
    return re.sub(r"(?<![\w'₀-₉.])" + re.escape(old) + r"(?![\w'₀-₉])", new, proof)


def op_add_comment(proof, rng):
    lines = proof.split("\n")
    candidates = [i for i, line in enumerate(lines) if line.strip() and i > 0]
    if not candidates:
        return None
    i = rng.choice(candidates)
    indent = re.match(r"\s*", lines[i]).group(0)
    lines.insert(i, indent + rng.choice(COMMENTS))
    return "\n".join(lines)


def op_strip_comments(proof, rng):
    new = "\n".join(line for line in proof.split("\n") if not line.strip().startswith("--"))
    new = re.sub(r"[ \t]+--[^\n]*", "", new)
    return new if new != proof else None


def op_permute_hints(proof, rng):
    def permute(m):
        items = [x.strip() for x in re.split(r",(?![^()]*\))", m.group(2)) if x.strip()]
        if len(items) < 2 or m.group(1) in ("rw", "simp only"):
            return None
        original = list(items)
        for _ in range(5):
            rng.shuffle(items)
            if items != original:
                break
        return f"{m.group(1)} [{', '.join(items)}]"
    return _replace_one(proof, HINT_LIST, rng, permute)


def op_blank_lines(proof, rng):
    lines = proof.split("\n")
    if any(not line.strip() for line in lines[1:-1]):
        return "\n".join(line for i, line in enumerate(lines) if line.strip() or i == 0 or i == len(lines) - 1)
    candidates = [i for i in range(2, len(lines)) if lines[i].strip()]
    if not candidates:
        return None
    lines.insert(rng.choice(candidates), "")
    return "\n".join(lines)


NEUTRAL = {"rename_have": op_rename_have, "add_comment": op_add_comment, "strip_comments": op_strip_comments,
           "permute_hints": op_permute_hints, "blank_lines": op_blank_lines}


def mutate(item: str, statement: str, kind: str, rng: random.Random) -> tuple[str, str] | None:
    head, proof = split(item, statement)
    operators = NEAR_MISS if kind == "near_miss" else NEUTRAL
    names = list(operators)
    rng.shuffle(names)
    for name in names:
        function = operators[name]
        new = function(proof, rng, statement) if name == "swap_hypothesis" else function(proof, rng)
        if new and new != proof and new.strip():
            return head + new, name
    return None
