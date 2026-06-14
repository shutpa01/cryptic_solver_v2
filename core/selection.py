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
    """Central letter(s): the single centre letter (odd length) or the central two
    (even length). Returns a list with one candidate selection."""
    n = len(la)
    if n < 3:
        return []
    if n % 2:
        return [[la[n // 2]]]
    return [[la[n // 2 - 1], la[n // 2]]]


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


def select_span(ctx, token, rule):
    """[(selected_string, (atom_id, ...)), ...] — every candidate the `rule` produces
    from the single word `token`. Empty when the word is too short for the rule."""
    la = _letter_atoms(ctx, token)
    fn = SPAN_RULES.get(rule)
    if fn is None:
        return []
    out = []
    for cand in fn(la):
        s = "".join(a.normalized for a in cand)
        if s:
            out.append((s, tuple(a.atom_id for a in cand)))
    return out


def match_span(ctx, token, rule, target):
    """The atom_ids if applying `rule` to `token` yields EXACTLY `target`, else None.
    Answer-driven: the consumer passes the answer span the piece must fill, so the
    selection is accepted only when it reproduces those exact letters."""
    t = (target or "").upper()
    for s, atom_ids in select_span(ctx, token, rule):
        if s.upper() == t:
            return atom_ids
    return None
