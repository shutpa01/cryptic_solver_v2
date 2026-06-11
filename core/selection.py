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
