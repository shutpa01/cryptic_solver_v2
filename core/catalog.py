"""Catalog engine — stage 4 (clean rewrite).

A small per-operation assembler per wordplay mechanism. Each gathers what
letters each wordplay word can supply (via the injected `lookup`) and tries to
assemble the answer its own way, emitting provenance natively. No template
list, no generic slot-matcher, no dependency on the legacy engine.

Injected, so core/ stays decoupled from any particular database:
    lookup(word_text)          -> list of (value, mechanism)   # value = UPPER
    is_link(word_text)         -> bool                          # joining word
    indicator_types(word_text) -> set of wordplay-type strings  # e.g. {'container'}

Done so far: charade, container. Other operations (anagram, reversal, deletion,
homophone, acrostic) are added one assembler at a time.
"""

from .model import Piece, Provenance, ParseResult


def _candidates(atom, target, lookup):
    """(value, mechanism) options for this word that are substrings of target."""
    out, seen = [], set()
    for value, mech in lookup(atom.text):
        v = (value or "").upper()
        if v and v in target and (v, mech) not in seen:
            out.append((v, mech))
            seen.add((v, mech))
    raw = "".join(c for c in atom.text.upper() if c.isalpha())
    if raw and raw in target and (raw, "raw") not in seen:
        out.append((raw, "raw"))
    return out


def _all_values(atom, lookup):
    """All (value, mechanism) options for this word, unfiltered — used for a
    container outer, whose letters are split around the inner and so are NOT a
    substring of the answer."""
    out, seen = [], set()
    for value, mech in lookup(atom.text):
        v = (value or "").upper()
        if v and (v, mech) not in seen:
            out.append((v, mech))
            seen.add((v, mech))
    raw = "".join(c for c in atom.text.upper() if c.isalpha())
    if raw and (raw, "raw") not in seen:
        out.append((raw, "raw"))
    return out


def _assemble_concat(atoms, target, lookup, is_link, min_pieces):
    """Can `atoms` (in clue order) concatenate to `target`? Non-piece words must
    be link words. Returns [(start, end, atom, value, mech), ...] over
    [0:len(target)] (>= min_pieces pieces), or None."""
    n = len(atoms)
    if not target:
        return None
    cand = [_candidates(a, target, lookup) for a in atoms]
    best = [None]

    def search(ai, pos, acc):
        if best[0] is not None:
            return
        if ai == n:
            if pos == len(target) and len(acc) >= min_pieces:
                best[0] = list(acc)
            return
        atom = atoms[ai]
        if is_link(atom.text):
            search(ai + 1, pos, acc)
            if best[0] is not None:
                return
        for value, mech in cand[ai]:
            if target.startswith(value, pos):
                acc.append((pos, pos + len(value), atom, value, mech))
                search(ai + 1, pos + len(value), acc)
                if best[0] is not None:
                    return
                acc.pop()

    search(0, 0, [])
    return best[0]


def _finish(answer, pieces, provenance, operation):
    pr = ParseResult(answer=answer, definition="", pieces=pieces,
                     provenance=provenance, operation=operation,
                     solved_by="catalog")
    return pr if pr.covers_answer() else None


def solve_charade(atoms, answer, lookup, is_link):
    """Pieces concatenate left-to-right to the answer (forward order only)."""
    if len(atoms) < 2 or not answer:
        return None
    placed = _assemble_concat(atoms, answer, lookup, is_link, min_pieces=2)
    if placed is None:
        return None
    pieces, provenance = [], []
    for start, end, atom, value, mech in placed:
        p = Piece(atoms=[atom.index], source_text=atom.surface,
                  value=value, mechanism=mech)
        pieces.append(p)
        provenance.append(Provenance(start, end, p, "charade"))
    return _finish(answer, pieces, provenance, "charade")


def solve_container(atoms, answer, lookup, is_link, indicator_types):
    """Outer word wraps an inner (one or more joined pieces). Requires a
    container/insertion indicator word to license it."""
    n = len(atoms)
    if n < 2 or not answer:
        return None
    is_con = [bool({"container", "insertion"} & set(indicator_types(a.text) or []))
              for a in atoms]
    if not any(is_con):
        return None

    for ind in range(n):
        if not is_con[ind]:
            continue
        for oi in range(n):
            if oi == ind:
                continue
            inner_atoms = [atoms[k] for k in range(n) if k != ind and k != oi]
            for outer_val, outer_mech in _all_values(atoms[oi], lookup):
                if len(outer_val) < 2:
                    continue
                inner_len = len(answer) - len(outer_val)
                if inner_len < 1:
                    continue
                for pos in range(1, len(outer_val)):
                    if not (answer.startswith(outer_val[:pos])
                            and answer.endswith(outer_val[pos:])):
                        continue
                    inner_target = answer[pos:pos + inner_len]
                    placed = _assemble_concat(inner_atoms, inner_target,
                                              lookup, is_link, min_pieces=1)
                    if placed is None:
                        continue
                    return _build_container(atoms, oi, outer_val, outer_mech,
                                            placed, pos, inner_len, answer)
    return None


def _build_container(atoms, oi, outer_val, outer_mech, inner_placed, pos,
                     inner_len, answer):
    outer = Piece(atoms=[atoms[oi].index], source_text=atoms[oi].surface,
                  value=outer_val, mechanism=outer_mech)
    pieces = [outer]
    provenance = [Provenance(0, pos, outer, "container")]          # outer head
    for s, e, atom, value, mech in inner_placed:                   # inner pieces
        p = Piece(atoms=[atom.index], source_text=atom.surface,
                  value=value, mechanism=mech)
        pieces.append(p)
        provenance.append(Provenance(pos + s, pos + e, p, "container"))
    provenance.append(Provenance(pos + inner_len, len(answer), outer, "container"))
    return _finish(answer, pieces, provenance, "container")


def solve(atoms, answer, lookup, is_link, indicator_types=None):
    """Try each operation assembler in turn (charade, then container)."""
    pr = solve_charade(atoms, answer, lookup, is_link)
    if pr is not None:
        return pr
    if indicator_types is not None:
        pr = solve_container(atoms, answer, lookup, is_link, indicator_types)
        if pr is not None:
            return pr
    return None
