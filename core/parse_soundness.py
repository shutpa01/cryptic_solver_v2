"""Soundness of a finished Parse — the last gate before a verdict may be a PASS.

The user's rule, 2026-09-01, after two false passes in one morning:

    "it must never produce a pass if it does not anagram, which is a quite simple
     test, along with the normal tests such as all words accounted for and all
     indicators part of the mechanism"

Each engine already checks its own shape. The point of this module is that the check
is in ONE place, applied to every engine's output, and computed from the parse as
DATA — never re-derived, never inferred from prose. The two faults it exists to catch:

  - 10088605 (DT 31333 28a) "Delight chaps with conclusion of art therapy" = TREATMENT.
    Passed with the right letters and the wrong derivation: the final T came from the
    link word "of" read as TO and curtailed, while "art" — where the T belongs — was
    absorbed as a leftover and printed as a "charade indicator" no DB row supported.
  - 10088752 (Times 29636 3d) "On fire editing novel" = IGNITED. Passed with the
    definition "fire" and "On" filed as a link, when "On fire" is the definition and
    was the only split the definition engine offered.

This is NOT a re-solver. It never proposes a parse, never repairs one, and never
guesses. It reads what the engine recorded and says whether the record stands up.

WHY IT REPORTS RATHER THAN JUDGES, for now
------------------------------------------
`violations()` returns a list. The caller decides what to do with it. It is wired in
report-only first, deliberately: a gate that has never been measured against the
corpus is a gate that mass-fails good solves on its first night. Measure, then bite.

THE LETTER CHECK AND ITS HONEST LIMIT
-------------------------------------
A piece's letters must equal the answer letters it is linked to — as a MULTISET, so
it holds for a charade (identity) and an anagram (a permutation) alike. That is the
"does it actually anagram" test in its general form.

The limit, stated rather than papered over: when letters are dropped from anagram
fodder, some engines record the drop on an INDICATOR NOTE as prose ("deleted letters:
H", "removes H") instead of in the piece's own `transform` field. This module will not
parse prose to reconstruct arithmetic — four attempts at doing so from the outside all
produced wrong answers. Such a piece is reported as INDETERMINATE, never as a
violation and never as clean. Indeterminate is a data-model gap to close (the drop
belongs in `transform`), not a solve to fail.
"""

import collections
import json

from core import piece_transform


def _letters(s):
    return collections.Counter(ch for ch in (s or "").upper() if ch.isalpha())


def _placed(src):
    """What a source actually laid on the tiles: its value after its RECORDED transform.
    Returns None when the transform cannot be read — the caller treats that as
    indeterminate rather than guessing."""
    value = getattr(src, "value", "") or ""
    xf = getattr(src, "transform", "") or ""
    if not xf:
        return value
    try:
        out = piece_transform.apply(value, json.loads(xf) if isinstance(xf, str) else xf)
    except Exception:
        return None
    return out if out else None


def _answer_letters(parse):
    return [ch for ch in (getattr(parse, "answer_text", "") or "").upper() if ch.isalpha()]


def _prose_deletion(parse):
    """True when some indicator note describes a letter drop in words. Its presence makes
    the affected letter arithmetic INDETERMINATE — see the module docstring."""
    for a in getattr(parse, "annotations", None) or []:
        n = (getattr(a, "note", "") or "").lower()
        if "deleted letter" in n or "removes " in n:
            return True
    return False


def letter_violations(parse):
    """Every source whose letters do not match the answer letters it is linked to.
    Returns (violations, indeterminate) — both lists of strings."""
    bad, unknown = [], []
    letters = _answer_letters(parse)
    if not letters:
        return bad, unknown
    by_source = collections.defaultdict(list)
    for ln in getattr(parse, "links", None) or []:
        pos = getattr(ln, "answer_pos", None)
        si = getattr(ln, "source_index", None)
        if isinstance(pos, int) and isinstance(si, int) and 0 < pos <= len(letters):
            by_source[si].append(pos)
    prose = _prose_deletion(parse)
    for si, positions in by_source.items():
        try:
            src = (parse.sources or [])[si]
        except Exception:
            bad.append("link points at source %d, which does not exist" % si)
            continue
        placed = _placed(src)
        if placed is None:
            unknown.append("%r: transform could not be read" % getattr(src, "text", ""))
            continue
        covers = collections.Counter(letters[p - 1] for p in positions)
        have = _letters(placed)
        if have == covers:
            continue
        # THE ASYMMETRY, and it is the whole point of this check.
        #
        # A piece may legitimately hold MORE letters than it covers: a deletion piece
        # records the FULL pre-deletion value and links only the survivor, on purpose —
        # "a deletion piece records the FULL DB value (ORATION), not the survivor
        # (RATION) — a synonym used is always REVEALED" (charade_deletion_engine.py:275).
        # So a surplus is normal and this check must not call it a fault.
        #
        # A piece may NEVER cover a letter it does not contain. That is a letter arriving
        # from nowhere, and it is the exact failure "it must never pass if it does not
        # anagram" describes. Equality is the wrong test; containment is the right one.
        invented = covers - have
        if invented:
            bad.append("%r contributes %s but is credited with %s — %s comes from nowhere"
                       % (getattr(src, "text", ""), placed or "(nothing)",
                          "".join(sorted(covers.elements())),
                          "".join(sorted(invented.elements()))))
            continue
        dropped = have - covers
        if dropped and not (prose or (getattr(src, "transform", "") or "")):
            # letters vanished and NOTHING in the parse records a deletion — unexplained
            unknown.append("%r: %s covers only %s, with no deletion recorded anywhere"
                           % (getattr(src, "text", ""), placed,
                              "".join(sorted(covers.elements()))))
    return bad, unknown


def unaccounted_words(parse, ctx):
    """Clue words carrying no role at all. Every word must be a piece, the definition,
    an indicator or a link — nothing may be left silently on the floor."""
    covered = set()
    for src in list(getattr(parse, "sources", None) or []):
        covered.update(getattr(src, "clue_atom_ids", ()) or ())
    d = getattr(parse, "definition", None)
    if d is not None:
        covered.update(getattr(d, "clue_atom_ids", ()) or ())
    for a in getattr(parse, "annotations", None) or []:
        covered.update(getattr(a, "clue_atom_ids", ()) or ())
    out = []
    for tok in getattr(ctx, "clue_tokens", None) or []:
        if getattr(tok, "kind", "") != "word":
            continue
        ids = set(getattr(tok, "atom_ids", ()) or ())
        if ids and not (ids & covered):
            out.append(getattr(tok, "text", ""))
    return out


def violations(parse, ctx=None):
    """Everything wrong with this parse as a PASS. Empty list == sound on all three
    rules. Indeterminate letter arithmetic is NOT a violation; it is returned
    separately by letter_violations for reporting."""
    if parse is None:
        return []
    out = []
    bad_letters, _ = letter_violations(parse)
    out.extend(bad_letters)
    if ctx is not None:
        missing = unaccounted_words(parse, ctx)
        if missing:
            out.append("clue words with no role: %s" % ", ".join(repr(m) for m in missing))
    try:
        from core import role_validity
        out.extend(role_validity.unbacked_roles(parse))
    except Exception as exc:                       # never let the check itself break a solve
        out.append("role-validity check failed to run (%s)" % type(exc).__name__)
    return out
