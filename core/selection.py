"""Selection-letters primitive — the shared building block for letter-selection
clue types (SOLVER_REDESIGN §3.3 mechanisms first_letter / last_letter / outer).

It takes specific letters from clue words and, crucially, returns the EXACT source
atom for each taken letter, so the engine that consumes it can record the §5.5
per-letter provenance (each answer letter <- the atom it was taken from). Pure: it
operates on a wfw_atoms context + tokens, does NO DB lookup and NO solving — only
selection. Acrostic uses 'first' / 'last'; extremes ('outer') and middle extend the
MODES map later, reusing this same per-letter sourcing.
"""


def _letter_atoms(ctx, token):
    """The token's letter atoms in order (drops any apostrophe/punctuation atoms)."""
    by_id = {a.atom_id: a for a in ctx.clue_atoms}
    return [by_id[aid] for aid in token.atom_ids
            if aid in by_id and by_id[aid].kind == "letter"]


def first_letter(ctx, token):
    """(normalized_char, atom_id) of the token's FIRST letter, or None if none."""
    la = _letter_atoms(ctx, token)
    return (la[0].normalized, la[0].atom_id) if la else None


def last_letter(ctx, token):
    """(normalized_char, atom_id) of the token's LAST letter, or None if none."""
    la = _letter_atoms(ctx, token)
    return (la[-1].normalized, la[-1].atom_id) if la else None


# mode -> selector. Extended later with 'outer' (first+last) and 'middle'.
MODES = {"first": first_letter, "last": last_letter}


def selected(ctx, tokens, mode):
    """[(normalized_char, atom_id), ...] taking the `mode` letter from each token in
    order. Returns None if any token has no letter (so the caller rejects the run)."""
    fn = MODES[mode]
    out = []
    for t in tokens:
        x = fn(ctx, t)
        if x is None:
            return None
        out.append(x)
    return out


# --- span selection from ONE word (the charade/container piece source) -----------
#
# The above takes one letter per word (acrostic). A charade/container piece instead
# takes a RUN of letters from a SINGLE word by a named rule licensed by an indicator:
#   house "originally" -> H        (first)
#   bird "essentially"  -> IR       (middle)
#   count "content gone"-> CY       (outer)
#   secret "occasionally"-> ERT     (alternate)
#   creature "heartless"-> CRITER   (remove_middle)
# Each rule yields one or more CANDIDATE selections; the consumer keeps only the one
# whose letters EXACTLY match the answer span it must fill, so a loose rule cannot
# fabricate (selection is answer-driven, exactly like a synonym landing in the answer).
# Every candidate carries its source atom_ids for the §5.5 per-letter provenance.

def _middle(la):
    """Every CENTRED run of the word bar the whole word, shortest first: SAINTLY ->
    N, INT, AINTL; SHROUD -> RO, HROU; RITE -> IT. A run is centred, so an odd-length
    word yields odd-length runs and an even-length word even-length ones.

    Widened 2026-08-11 (was: the single centre letter / central two, one candidate).
    "at heart of saintly" = INT is ordinary cryptic usage and was unfilable, by hand
    or by engine. Looseness is safe here for the reason stated above: every consumer
    is ANSWER-DRIVEN — charade keeps a candidate only when answer.startswith(value,
    pos), container only when it completes the insertion to the exact answer — and the
    rule is indicator-gated, so extra candidates can only be discarded, never fabricate.
    Shortest first keeps the previous sole candidate the first one tried."""
    n = len(la)
    if n < 3:
        return []
    return [la[(n - L) // 2:(n - L) // 2 + L]
            for L in range(1 if n % 2 else 2, n - 1, 2)]


def _heartless(la):
    """The word with its central letter(s) removed, rest kept in order (critter ->
    CRITER). Drops the single centre letter (odd) or central two (even)."""
    n = len(la)
    if n < 3:
        return []
    if n % 2:
        return [la[:n // 2] + la[n // 2 + 1:]]
    return [la[:n // 2 - 1] + la[n // 2 + 1:]]


# rule -> function(letter_atoms) -> list[candidate], each candidate a list of atoms.
SPAN_RULES = {
    "first":         lambda la: [la[:1]] if la else [],
    "last":          lambda la: [la[-1:]] if la else [],
    "outer":         lambda la: [[la[0], la[-1]]] if len(la) >= 2 else [],
    "middle":        _middle,
    "alternate":     lambda la: [la[0::2], la[1::2]] if len(la) >= 2 else [],
    "remove_first":  lambda la: [la[1:]] if len(la) >= 2 else [],
    "remove_last":   lambda la: [la[:-1]] if len(la) >= 2 else [],
    "remove_outer":  lambda la: [la[1:-1]] if len(la) >= 3 else [],
    "remove_middle": _heartless,
}


def _letter_sets(ctx, tokens):
    """The letter-atom sequences a span rule may be applied to: the WHOLE run first, then
    each apostrophe-separated part when there is more than one.

    An apostrophe divides a word for selection purposes (MARIO'S -> MARIO | S). The setter
    selects from the base word, not from the possessive form, so "Mario's guts" = ARI needs
    the middle of MARIO — over MARIOS the rule yields RI and ARIO and never ARI. Five
    puzzles in a row carried a selection like this ("Symphony's finale" -> Y, "effort's
    last" -> T) and none could be filed.

    Whole run FIRST, so every candidate that existed before is still produced, and still
    tried first. The extra part-candidates are safe for the reason given above: the
    consumer keeps only the candidate that reproduces the exact answer span, so a looser
    rule can only offer more to discard, never fabricate.

    The /hs commit gate (core.wfw_web._selection_candidates) already split this way; the
    engines did not, so a possessive selection was hand-fileable but never machine-
    derivable. This closes that asymmetry.
    """
    by_id = {a.atom_id: a for a in ctx.clue_atoms}
    seq = [by_id[aid] for t in tokens for aid in t.atom_ids if aid in by_id]
    whole = [a for a in seq if a.kind == "letter"]
    parts, cur = [], []
    for a in seq:
        if a.kind == "quote":
            if cur:
                parts.append(cur)
                cur = []
        elif a.kind == "letter":
            cur.append(a)
    if cur:
        parts.append(cur)
    return [whole] + (parts if len(parts) > 1 else [])


def _spans(ctx, tokens, rule):
    """Shared body of select_span / select_span_run: every candidate the rule produces over
    every letter set, de-duplicated, order preserved (whole-word candidates first)."""
    fn = SPAN_RULES.get(rule)
    if fn is None:
        return []
    out, seen = [], set()
    for la in _letter_sets(ctx, tokens):
        for cand in fn(la):
            s = "".join(a.normalized for a in cand)
            ids = tuple(a.atom_id for a in cand)
            if s and (s, ids) not in seen:
                seen.add((s, ids))
                out.append((s, ids))
    return out


def select_span(ctx, token, rule):
    """[(selected_string, (atom_id, ...)), ...] — every candidate the `rule` produces
    from the single word `token`. Empty when the word is too short for the rule."""
    return _spans(ctx, [token], rule)


def select_span_run(ctx, tokens, rule):
    """select_span over a RUN of tokens: the rule is applied to the CONCATENATED letter
    atoms of all tokens in order ('is upset' alternate -> IUST / SPE). For a single
    token this equals select_span. Additive — select_span is untouched. Part of the
    single-word-assumption fix (2026-07-08): selection fodder, like indicators, may
    span several clue words."""
    return _spans(ctx, list(tokens), rule)


def match_span(ctx, token, rule, target):
    """The atom_ids if applying `rule` to `token` yields EXACTLY `target`, else None.
    Answer-driven: the consumer passes the answer span the piece must fill, so the
    selection is accepted only when it reproduces those exact letters."""
    t = (target or "").upper()
    for s, atom_ids in select_span(ctx, token, rule):
        if s.upper() == t:
            return atom_ids
    return None
