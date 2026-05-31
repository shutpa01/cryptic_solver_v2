"""Catalog engine — stage 4 (clean rewrite).

A small per-operation assembler per wordplay mechanism. Each gathers what
letters each wordplay word can supply (via the injected `lookup`) and tries to
assemble the answer its own way, emitting provenance natively. No template
list, no generic slot-matcher, no dependency on the legacy engine.

Injected, so core/ stays decoupled from any particular database:
    lookup(word_text)          -> list of (value, mechanism)   # value = UPPER
    is_link(word_text)         -> bool                          # joining word
    indicator_types(word_text) -> set of wordplay-type strings  # e.g. {'container'}

Done so far: charade, container, anagram, reversal, deletion, homophone,
acrostic. Operations are added one assembler at a time.
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


def _raw(atom):
    return "".join(c for c in atom.text.upper() if c.isalpha())


def solve_anagram(atoms, answer, lookup, is_link, indicator_types):
    """The non-indicator, non-link words' letters rearrange to the whole answer.
    Requires an anagram indicator. Provenance is span-level — the rearranged
    letters are not separately sourced, so the whole answer is one entry."""
    n = len(atoms)
    if n < 1 or not answer:
        return None
    ana_idx = {i for i in range(n)
               if "anagram" in (indicator_types(atoms[i].text) or set())}
    if not ana_idx:
        return None
    fodder_idx = [i for i in range(n)
                  if i not in ana_idx and not is_link(atoms[i].text)]
    if not fodder_idx:
        return None
    letters = "".join(_raw(atoms[i]) for i in fodder_idx)
    if sorted(letters) != sorted(answer):
        return None
    if letters[::-1] == answer:
        # An exact reversal of the fodder is a REVERSAL, never an anagram —
        # a definitional rule, not a tie-break. Refuse it here even though an
        # anagram indicator is present (many reversal words are also tagged as
        # anagram indicators); the reversal assembler gives the honest label.
        return None
    fodder_words = " ".join(atoms[i].surface for i in fodder_idx)
    piece = Piece(atoms=[atoms[i].index for i in fodder_idx],
                  source_text=fodder_words, value=answer,
                  mechanism="anagram_fodder")
    prov = [Provenance(0, len(answer), piece, "anagram", transform="anagram_of")]
    return _finish(answer, [piece], prov, "anagram")


def solve_reversal(atoms, answer, lookup, is_link, indicator_types):
    """The wordplay reads forward, then the whole thing is reversed to give the
    answer.

    Implemented by reusing the charade concatenation engine: if the fodder
    concatenates (in clue order) to the *reversed* answer, then reversing that
    concatenation yields the answer. Each forward piece occupying [s:e] of the
    reversed-answer string therefore occupies answer[n-e:n-s] (its letters now
    reversed). This covers a single reversed word and a reversed charade alike.

    Two paths:

    - Licensed: a reversal indicator is present. The fodder (everything else)
      may be a single word or a charade and may use synonyms/abbreviations.

    - Unlicensed: no reversal indicator, but an EXACT reversal is a reversal by
      definition, never an anagram — so it must still be labelled a reversal
      (the anagram assembler refuses exact reverses for the same reason). Kept
      deliberately tight to avoid asserting a reversal on a coincidence: a
      SINGLE clue word, spelled backwards using its OWN letters (raw, no
      synonym), must equal the whole answer. A palindrome is identity, not a
      reversal, so it is skipped.

    Scope: the WHOLE answer is the reversal. A reversal of just one piece inside
    a larger charade is a separate (future) slice.
    """
    n = len(atoms)
    if n < 1 or not answer:
        return None
    nans = len(answer)
    target = answer[::-1]
    rev_idx = {i for i in range(n)
               if "reversal" in (indicator_types(atoms[i].text) or set())}

    placed = None
    if rev_idx:
        fodder = [atoms[i] for i in range(n)
                  if i not in rev_idx and not is_link(atoms[i].text)]
        if fodder:
            placed = _assemble_concat(fodder, target, lookup, is_link, min_pieces=1)
    elif answer != target:                       # skip palindromes (identity)
        for atom in atoms:
            if is_link(atom.text):
                continue
            raw = _raw(atom)
            if raw and raw == target:
                placed = [(0, nans, atom, raw, "raw")]
                break

    if placed is None:
        return None

    pieces, provenance = [], []
    for start, end, atom, value, mech in placed:
        p = Piece(atoms=[atom.index], source_text=atom.surface,
                  value=value, mechanism=mech)
        pieces.append(p)
        # forward span [start:end] of reversed-answer -> answer[n-end:n-start]
        provenance.append(Provenance(nans - end, nans - start, p,
                                     "reversal", transform="reversed"))
    return _finish(answer, pieces, provenance, "reversal")


def _deletion_detail(value, f, b):
    """Human description of an end-deletion: which letters were removed."""
    parts = []
    if f:
        parts.append(f"'{value[:f]}' (start)")
    if b:
        parts.append(f"'{value[len(value) - b:]}' (end)")
    return f"{value} minus " + " and ".join(parts)


def solve_deletion(atoms, answer, lookup, is_link, indicator_types):
    """A single source word's value, with letters trimmed from an end, is the
    answer. Requires a deletion indicator.

    Scope (this slice): the source is ONE content word (after the indicator and
    any link words are set aside), and the deletion is END-anchored, removing up
    to TWO letters in total — off the front (beheadment), the back (curtailment),
    or one off each end — so the answer is a contiguous substring of the source
    value. The smallest trim that works is chosen, giving the most conservative
    account, and the two-letter cap keeps long synonyms from matching a
    coincidental deep-interior substring.

    Out of scope for now (separate future slices): removing three-plus letters,
    deletion of an interior or named segment ("heartless", "without tea"), and a
    deleted piece sitting inside a larger charade.
    """
    n = len(atoms)
    if n < 1 or not answer:
        return None
    del_idx = {i for i in range(n)
               if "deletion" in (indicator_types(atoms[i].text) or set())}
    if not del_idx:
        return None
    fodder = [atoms[i] for i in range(n)
              if i not in del_idx and not is_link(atoms[i].text)]
    if len(fodder) != 1:
        return None                 # multi-piece deletion is a future slice

    atom = fodder[0]
    alen = len(answer)
    # Try every source value, every end-trim; prefer the smallest total trim.
    best = None                      # (total_trim, value, mech, f, b)
    seen_values = set()
    for raw_value, mech in _all_values(atom, lookup):
        # An end-deletion trims the letters of ONE solid word. Reject multi-word
        # or punctuated source values ("at sea", "sea,"): otherwise a trim could
        # coincidentally strip a whole word at a space boundary, which is not an
        # honest letter-level deletion.
        token = raw_value.strip()
        if not token.isalpha():
            continue
        value = token.upper()
        if (value, mech) in seen_values:
            continue
        seen_values.add((value, mech))
        extra = len(value) - alen
        if not (1 <= extra <= 2):
            continue                 # remove 1-2 letters total; else not this slice
        for f in range(0, extra + 1):
            b = extra - f
            if f == 0 and b == 0:
                continue
            if value[f:len(value) - b] == answer:
                if best is None or extra < best[0]:
                    best = (extra, value, mech, f, b)
                break                # smallest f for this value found
    if best is None:
        return None

    _, value, mech, f, b = best
    piece = Piece(atoms=[atom.index], source_text=atom.surface,
                  value=answer, mechanism=mech)   # value = surviving letters
    prov = [Provenance(0, alen, piece, "deletion",
                       transform=_deletion_detail(value, f, b))]
    return _finish(answer, [piece], prov, "deletion")


def solve_homophone(atoms, answer, lookup, is_link, indicator_types, sounds_like):
    """A clue word SOUNDS like the whole answer. Requires a homophone indicator
    ("we hear", "reportedly", "on the radio") and an injected `sounds_like`
    lookup (word -> list of words it sounds like).

    Scope (this slice): whole-answer, single source word — one wordplay word
    whose homophone equals the answer. The wordplay reads aloud as the answer;
    the answer is the spelling that reaches the grid.

    Out of scope for now (future slices): a homophone of a SYNONYM of the source
    word, a multi-word sound-alike phrase, and a homophone of just one piece
    inside a larger charade.
    """
    n = len(atoms)
    if n < 1 or not answer or sounds_like is None:
        return None
    hom_idx = {i for i in range(n)
               if "homophone" in (indicator_types(atoms[i].text) or set())}
    if not hom_idx:
        return None
    for i in range(n):
        if i in hom_idx or is_link(atoms[i].text):
            continue
        atom = atoms[i]
        if answer in {h.upper() for h in (sounds_like(atom.text) or [])}:
            piece = Piece(atoms=[atom.index], source_text=atom.surface,
                          value=answer, mechanism="homophone")
            prov = [Provenance(0, len(answer), piece, "homophone",
                               transform=f"sounds like \"{atom.surface}\"")]
            return _finish(answer, [piece], prov, "homophone")
    return None


def solve_acrostic(atoms, answer, lookup, is_link, indicator_types):
    """The first letters of consecutive words spell the answer. Requires an
    acrostic indicator.

    One word supplies one answer letter, so the provenance is per-letter — the
    most granular account of any operation. Link words are KEPT as fodder here:
    in an acrostic every word of the phrase contributes its initial, including
    small joining words.

    First-letter only. In this reference DB acrostic indicators are the
    initial-letter kind ('first'/'initial'); the final-letter selectors
    ("finally", "endings", "tails") are tagged 'parts', not 'acrostic', so a
    last-letter reading is never licensed here. Supporting last-letter acrostics
    (and non-edge letter selections like "second letters of") is a future slice
    that needs its own indicator gate, not a coincidence fallback.
    """
    n = len(atoms)
    if n < 1 or not answer:
        return None
    acro_idx = {i for i in range(n)
                if "acrostic" in (indicator_types(atoms[i].text) or set())}
    if not acro_idx:
        return None
    fodder = [atoms[i] for i in range(n) if i not in acro_idx and _raw(atoms[i])]
    if len(fodder) != len(answer):
        return None

    if "".join(_raw(a)[0] for a in fodder) != answer:
        return None
    pieces, provenance = [], []
    for idx, a in enumerate(fodder):
        p = Piece(atoms=[a.index], source_text=a.surface,
                  value=answer[idx], mechanism="first_letter")
        pieces.append(p)
        provenance.append(Provenance(idx, idx + 1, p, "acrostic",
                                     transform="first_letter"))
    return _finish(answer, pieces, provenance, "acrostic")


def solve(atoms, answer, lookup, is_link, indicator_types=None, sounds_like=None):
    """Try each operation assembler. Indicator-gated operations run first, then
    the permissive charade.

    Order within the gated group is most-specific first. Reversal precedes
    anagram deliberately: a reversal is one particular letter permutation, so an
    anagram indicator (many reversal words are tagged as both) would otherwise
    claim a clue whose answer is the exact reverse of its fodder. Reversal only
    fires on that exact-reverse case — a genuine reversal — and returns None for
    a true rearrangement, so anagram still handles real anagrams.
    """
    if indicator_types is not None:
        for fn in (solve_reversal, solve_deletion, solve_acrostic,
                   solve_anagram, solve_container):
            pr = fn(atoms, answer, lookup, is_link, indicator_types)
            if pr is not None:
                return pr
        if sounds_like is not None:
            pr = solve_homophone(atoms, answer, lookup, is_link,
                                 indicator_types, sounds_like)
            if pr is not None:
                return pr
    return solve_charade(atoms, answer, lookup, is_link)
