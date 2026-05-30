"""Catalog engine — stage 4 (clean rewrite).

A small per-operation assembler per wordplay mechanism. Each gathers what
letters each wordplay word can supply (via the injected `lookup`) and tries to
assemble the answer its own way, emitting provenance natively. No template
list, no generic slot-matcher, no dependency on the legacy engine.

Injected, so core/ stays decoupled from any particular database:
    lookup(word_text)  -> list of (value, mechanism)   # value = UPPER letters
    is_link(word_text) -> bool                          # joining word, no role

Done so far: charade. Other operations (container, anagram, reversal, deletion,
homophone, acrostic) are added one assembler at a time.
"""

from .model import Piece, Provenance, ParseResult


def _candidates(atom, answer, lookup):
    """The (value, mechanism) options for this word that appear in the answer."""
    out, seen = [], set()
    for value, mech in lookup(atom.text):
        v = (value or "").upper()
        if v and v in answer and (v, mech) not in seen:
            out.append((v, mech))
            seen.add((v, mech))
    raw = "".join(c for c in atom.text.upper() if c.isalpha())
    if raw and raw in answer and (raw, "raw") not in seen:
        out.append((raw, "raw"))
    return out


def solve_charade(atoms, answer, lookup, is_link):
    """Pieces concatenate left-to-right to the answer; any non-piece word must
    be a link word. Returns a ParseResult or None. (Forward order only;
    reverse-order charades are a later addition.)"""
    n = len(atoms)
    if n < 2 or not answer:
        return None
    cand = [_candidates(a, answer, lookup) for a in atoms]

    best = [None]

    def search(ai, pos, acc):
        if best[0] is not None:
            return
        if ai == n:
            if pos == len(answer) and len(acc) >= 2:
                best[0] = list(acc)
            return
        atom = atoms[ai]
        if is_link(atom.text):                       # skip a link word
            search(ai + 1, pos, acc)
            if best[0] is not None:
                return
        for value, mech in cand[ai]:                 # or use this word as a piece
            if answer.startswith(value, pos):
                acc.append((pos, pos + len(value), atom, value, mech))
                search(ai + 1, pos + len(value), acc)
                if best[0] is not None:
                    return
                acc.pop()

    search(0, 0, [])
    if best[0] is None:
        return None

    pieces, provenance = [], []
    for start, end, atom, value, mech in best[0]:
        p = Piece(atoms=[atom.index], source_text=atom.surface,
                  value=value, mechanism=mech)
        pieces.append(p)
        provenance.append(Provenance(start, end, p, "charade"))

    pr = ParseResult(answer=answer, definition="", pieces=pieces,
                     provenance=provenance, operation="charade",
                     solved_by="catalog")
    return pr if pr.covers_answer() else None


def solve(atoms, answer, lookup, is_link):
    """Try each operation assembler in turn (charade only, for now)."""
    return solve_charade(atoms, answer, lookup, is_link)
