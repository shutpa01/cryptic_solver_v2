"""Deletion primitive — the pure letter-selection layer for the deletion family.

A deletion clue forms a piece by REMOVING letters from a source string: the source is
either a DB value (LOATH = synonym of "unwilling", minus L -> OATH) or a word's own
raw letters (BUGATTI emptied -> BI). This module is the PURE, DB-free core: given a
source string it yields the candidate results of each deletion sub-type, and maps
indicator words to the specific sub-type they signal. The engines layer DB lookup and
answer-matching on top — this file never touches the DB and has no answer in it, so it
is trivially testable.

Sub-types (canonical op names), all observed in real clues:
  behead     drop the first letter      AGAIN -> GAIN     (beheaded, headless, topless)
  curtail    drop the last letter       TAUT  -> TAU      (endless, curtailed, reduced)
  outer      drop BOTH end letters      CHOPIN-> HOPI     (shaving, "stripped" of outer)
  heartless  drop the one central letter BANAL-> BAAL     (heartless, gutless)   [odd len]
  empty      keep ONLY first + last     BUGATTI-> BI      (empty, emptied, hollow)
  internal   drop one interior letter   (generic; an indicator that names no sub-type)

"remove a named substring" (LOATH-L, DELIBERATES-RATE) is NOT enumerated here — it is
answer-driven in the engine (the removed run is derived from source vs target and then
validated as a DB value of another clue word). Use `removed_run(source, target)` for it.

ANSWER-DRIVEN / OOM-safe: `candidates` is O(L) per source (a handful of ops + L interior
deletions), L<20. The engine enumerates a fodder run's bounded DB values, never a product.
"""

# FUSED indicators — a single word/phrase that encodes BOTH the removal and WHICH letters
# go, so it pins one op on its own. (lowercased)
FUSED_OP = {
    "behead": frozenset({
        "beheaded", "headless", "topless", "decapitated", "leaderless", "uncapped"}),
    "curtail": frozenset({
        "curtailed", "endless", "tailless", "docked", "clipped", "truncated",
        "unfinished", "incomplete", "mostly", "almost", "nearly", "largely"}),
    "outer": frozenset({
        "shaved", "shelled", "peeled", "husked", "skinned", "topped and tailed"}),
    "heartless": frozenset({
        "heartless", "gutless", "coreless", "hollow"}),
    "empty": frozenset({
        "empty", "emptied", "hollowed out", "evacuated", "gutted"}),
}

# POSITION nouns — a noun naming WHICH letters, paired with a removal word to form an
# indicator ("without LEADER" = behead, "losing HEART" = heartless). op -> nouns.
POSITION_NOUN = {
    "behead": frozenset({
        "head", "heading", "leader", "leading", "top", "topping", "start", "starting",
        "beginning", "front", "fronting", "opener", "opening", "lead", "capital",
        "introduction", "first", "initial", "header", "crown"}),
    "curtail": frozenset({
        "tail", "tailing", "end", "ending", "bottom", "rear", "back", "foot",
        "finish", "finishing", "close", "butt", "last", "rump", "extremity"}),
    "heartless": frozenset({
        "heart", "centre", "center", "core", "middle", "gut", "guts", "interior",
        "innards", "inside"}),
    "outer": frozenset({
        "ends", "extremes", "edges", "sides", "exterior", "outsides", "extremities"}),
}

# REMOVAL words — verbs/particles that signal a deletion but not which letters (they pair
# with a position noun, OR name the removed letters via another clue word = Form B).
REMOVAL_WORDS = frozenset({
    "without", "losing", "loses", "lose", "lost", "missing", "miss", "dropping",
    "drops", "drop", "dropped", "off", "leaving", "leave", "leaves", "left",
    "shedding", "sheds", "shed", "removing", "removed", "removes", "remove", "gone",
    "cut", "cutting", "cuts", "discard", "discarding", "discards", "discarded",
    "ousting", "ousts", "ousted", "shunning", "shuns", "shunned", "abandoning",
    "abandons", "omitting", "omits", "minus", "less", "lacking", "lacks", "knocked",
    "evicting", "banishing", "excluding", "rejecting", "sacrificing", "ditching",
    "no", "not", "free", "rid", "out", "unwrapped", "stripped", "skimmed",
})


def positional_op(items):
    """The deletion op signalled POSITIONALLY by the candidate indicator words, with the
    indices that form the indicator. `items` is [(idx, text)]. Returns (op, set_of_idx)
    or (None, set()). A FUSED word pins its op; otherwise a removal word PLUS a position
    noun gives the noun's op (first+last nouns together -> outer). A bare removal word
    with no position noun returns None (that is Form B's job, not a positional op)."""
    low = [(i, (t or "").lower()) for i, t in items]
    joined = " ".join(t for _, t in low)
    for op, words in FUSED_OP.items():
        for i, t in low:
            if t in words:
                return op, {i}
        for ph in words:
            if " " in ph and ph in joined:
                return op, {i for i, t in low if t in ph.split()}
    rverb = {i for i, t in low if t in REMOVAL_WORDS}
    nouns = {}
    for op, ns in POSITION_NOUN.items():
        for i, t in low:
            if t in ns:
                nouns.setdefault(op, set()).add(i)
    if rverb and nouns:
        ops = set(nouns)
        if {"behead", "curtail"} <= ops:               # "top and bottom" -> both ends
            idx = set().union(*nouns.values()) | rverb
            return "outer", idx
        op = "outer" if "outer" in ops else next(iter(ops))
        return op, nouns[op] | rverb
    return None, set()


def is_removal(items):
    """Indices of any plain removal word among `items` (for Form B, where the removed
    letters are NAMED by another clue word and a generic removal word marks the cut)."""
    return {i for i, t in items if (t or "").lower() in REMOVAL_WORDS}


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


def ops_for_indicators(indicator_texts):
    """The canonical op set signalled by the given indicator words/phrases. A specific
    indicator pins its op(s); a generic deletion indicator allows ALL ops; unknown ->
    empty set. `indicator_texts` is an iterable of raw clue strings (any case)."""
    texts = {(t or "").strip().lower() for t in indicator_texts}
    ops = set()
    for op, words in INDICATOR_OPS.items():
        if texts & words:
            ops.add(op)
    if texts & GENERIC_INDICATORS:
        ops |= set(_OP_FUNCS)            # generic: every op is admissible
    return ops


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
