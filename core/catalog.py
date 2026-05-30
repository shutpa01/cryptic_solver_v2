"""Catalog engine — stage 4 (build in progress).

Reuses the existing signature engine and converts its parse into a core
ParseResult carrying word-for-word provenance, rather than rebuilding the big
matcher. Built one operation at a time.

FIRST SLICE: the charade (and any parse whose pieces concatenate left-to-right
to the answer). Operations that scramble letter order — container, reversal,
anagram — are handled in later slices, where the placement is not a simple
left-to-right walk.
"""

from .model import Piece, Provenance, ParseResult

# Map the engine's role tokens to plain mechanism names.
_MECH = {
    "SYN_F": "synonym",
    "ABR_F": "abbreviation",
    "RAW": "raw",
    "POS_F": "positional",
    "ANA_F": "anagram_fodder",
    "HID_F": "hidden",
}


def _pieces_in_order(word_roles):
    """The letter-contributing pieces, in clue order (value is not None)."""
    out = []
    for entry in word_roles:
        word, tok = entry[0], entry[1]
        val = entry[2] if len(entry) > 2 else None
        if val:
            out.append((word, tok, val))
    return out


def from_engine_result(sr, answer_clean: str):
    """Convert a signature SolveResult into a core ParseResult with provenance,
    for the charade (left-to-right concatenation) case. Returns None if this
    slice can't represent the parse (a scrambling operation — later slice)."""
    if sr is None or not getattr(sr, "solved", False) or sr.result is None:
        return None

    pieces_raw = _pieces_in_order(sr.result.word_roles)
    if not pieces_raw:
        return None

    # This slice only handles the case where the pieces concatenate, in clue
    # order, to the answer (a forward charade).
    if "".join(v for _, _, v in pieces_raw) != answer_clean:
        return None

    pieces, provenance, pos = [], [], 0
    for word, tok, val in pieces_raw:
        p = Piece(atoms=[], source_text=word, value=val,
                  mechanism=_MECH.get(tok, tok.lower()))
        pieces.append(p)
        provenance.append(Provenance(pos, pos + len(val), p, "charade"))
        pos += len(val)

    pr = ParseResult(
        answer=answer_clean,
        definition=getattr(sr, "definition", "") or "",
        pieces=pieces,
        provenance=provenance,
        operation="charade",
        confidence=getattr(sr, "confidence", 0),
        solved_by="catalog",
    )
    return pr if pr.covers_answer() else None
