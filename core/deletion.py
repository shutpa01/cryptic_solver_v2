"""Deletion primitive — the pure letter-selection layer for the deletion family.

A deletion clue forms a piece by REMOVING letters from a source string: the source is
either a DB value (LOATH = synonym of "unwilling", minus L -> OATH) or a word's own
raw letters (BUGATTI emptied -> BI). This module is the PURE, DB-free MATH core: given a
source string it yields the candidate results of each deletion sub-type. It holds NO
indicator vocabulary — which words signal a deletion (and which letters they remove) is
read from the DB `indicators` table by core.deletion_engine. `SUBTYPE_OP` maps a DB
deletion `subtype` onto the canonical op the math here applies.

Sub-types (canonical op names), all observed in real clues:
  behead     drop the first letter      AGAIN -> GAIN     (beheaded, headless, topless)
  curtail    drop the last letter       TAUT  -> TAU      (endless, curtailed, reduced)
  outer      drop BOTH end letters       CHOPIN-> HOPI     (shaving, "stripped" of outer)
  heartless  drop the one central letter BANAL-> BAAL     (heartless, gutless)   [odd len]
  empty      keep ONLY first + last     BUGATTI-> BI      (empty, emptied, hollow)
  internal   drop one interior letter   (generic; an indicator that names no sub-type)

"remove a named substring" (LOATH-L, DELIBERATES-RATE) is NOT enumerated here — it is
answer-driven in the engine (the removed run is derived from source vs target and then
validated as a DB value of another clue word). Use `removed_runs(source, target)` for it.

ANSWER-DRIVEN / OOM-safe: `candidates` is O(L) per source (a handful of ops + L interior
deletions), L<20. The engine enumerates a fodder run's bounded DB values, never a product.
"""

# DB deletion `subtype` -> canonical op. Generic subtypes (general / removal / deletion /
# NULL) map to NO specific op: they mark a plain removal whose removed letters are NAMED
# by another clue word (Form B), handled answer-driven in the engine.
SUBTYPE_OP = {
    "head": "behead", "first": "behead",
    "tail": "curtail", "last": "curtail",
    "ends": "outer", "outer": "outer",
    "middle": "heartless",
    "empty": "empty",
}


def apply_op(op, source):
    """Result of deletion op `op` on `source`, or None."""
    f = _OP_FUNCS.get(op)
    return f((source or "").upper()) if f else None


def behead(s):
    return s[1:] if len(s) >= 2 else None


def curtail(s):
    return s[:-1] if len(s) >= 2 else None


def outer(s):
    return s[1:-1] if len(s) >= 3 else None


def heartless(s):
    # remove the single central letter; defined only for odd length >= 3
    if len(s) >= 3 and len(s) % 2 == 1:
        m = len(s) // 2
        return s[:m] + s[m + 1:]
    return None


def empty(s):
    # keep first and last only ("empty Bugatti" -> BI); >= 2 letters
    return s[0] + s[-1] if len(s) >= 2 else None


_OP_FUNCS = {
    "behead": behead,
    "curtail": curtail,
    "outer": outer,
    "heartless": heartless,
    "empty": empty,
}


def candidates(source, ops=None):
    """[(result, op)] for `source` under the requested ops (default: all). Adds the
    'internal' op — dropping each single interior letter — only when ops is None or
    explicitly includes 'internal' (it is the catch-all an unspecific indicator needs).
    Empty/None results dropped; de-duplicated on (result, op)."""
    s = (source or "").upper()
    if len(s) < 2:
        return []
    use = set(_OP_FUNCS) if ops is None else (set(ops) & set(_OP_FUNCS))
    out, seen = [], set()
    for op in use:
        r = _OP_FUNCS[op](s)
        if r and (r, op) not in seen:
            seen.add((r, op))
            out.append((r, op))
    if ops is None or "internal" in (ops or ()):
        for i in range(1, len(s) - 1):          # drop one interior letter
            r = s[:i] + s[i + 1:]
            if (r, "internal") not in seen:
                seen.add((r, "internal"))
                out.append((r, "internal"))
    return out


def removed_runs(source, target):
    """ALL distinct contiguous runs whose removal turns `source` into `target`
    (uppercased), in left-to-right order. Covers edge runs (LOATH-L) and interior runs
    (DELIBERATES-RATE). Returns every candidate because a repeated letter makes the
    removal ambiguous (DELIBERATES->DELIBES admits both 'ERAT' and 'RATE'); the engine
    picks the one that is a DB value of a clue word ('speed'->RATE) — that validation,
    not position, is what makes a substring deletion real. Empty list if none."""
    s = (source or "").upper()
    t = (target or "").upper()
    if not s or not t or len(t) >= len(s):
        return []
    rem = len(s) - len(t)
    out, seen = [], set()
    for i in range(0, len(s) - rem + 1):
        if s[:i] + s[i + rem:] == t:
            run = s[i:i + rem]
            if run not in seen:
                seen.add(run)
                out.append(run)
    return out
