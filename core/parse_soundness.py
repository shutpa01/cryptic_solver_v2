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

2026-09-22 — HALF OF THAT GAP IS NOW CLOSED, and the module is WIRED IN.
`fodder_violations` catches the fault this whole file was written for and could not
see: letters lost on the way IN to the fodder, before any link exists to compare
against. Anagram fodder now records that cut in `transform`, read WORD -> VALUE
(anagram_deletion_engine._build), and engine_registry._finish downgrades a PASS with
an unrecorded cut to a REVIEW pending. Only that rule bites. Measured over all 7,777
stored passes first: fodder_violations 12 hits, all genuine; letter_violations' own
"letters from nowhere" list 20 hits, ALL SOUND (a homophone's and a spoonerism's
letters differ from their value by design), so it stays report-only until that is
fixed. Measure, then bite — the rule still holds.
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
    # ANAGRAM FODDER reads its transform the other way round: WORD -> VALUE, the cut
    # made before the anagram scatters the letters (see fodder_violations). Its VALUE
    # is already what it lays on the tiles, so applying the cut again would delete a
    # letter twice.
    if not xf or getattr(src, "mechanism", "") == "anagram_fodder":
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


def fodder_violations(parse):
    """Anagram fodder whose VALUE is its own WORD minus letters, with nothing anywhere
    recording the cut. Returns a list of strings.

    THE HOLE THIS CLOSES (user, 2026-09-22). letter_violations compares a piece's value
    against the answer letters it is LINKED to, so it can only see letters lost on the
    way OUT of the value. Fodder loses them on the way IN: an engine shortens the word
    and files the short value, and from then on RUL covers R, U, L perfectly and every
    check is clean. Times 29654 17d ("Civil case rule shortly to be reformed" = SECULAR)
    was served as a PASS reading `rule -> RUL` with the E named nowhere at all.

    Fodder is the one mechanism whose value IS its word's own letters, so word-against-
    value is a fair comparison here and nowhere else (a synonym's value has nothing to
    do with its word's letters). The cut is recorded in the piece's `transform`, read
    WORD -> VALUE — see anagram_deletion_engine._build.

    Two things are deliberately NOT faults here:
      * a value holding letters its word has not — that is a substitution filed as
        fodder, a different fault, and letter_violations judges what it places;
      * a trailing possessive "'s" the value drops ("Lionel's" -> LIONEL). The
        apostrophe-s is a word of its own in the clue (it reads as `is`/`has`), and the
        piece text merely spans it. No letter of the fodder went missing."""
    from core.wordplay import raw

    out = []
    for src in list(getattr(parse, "sources", None) or []):
        if getattr(src, "mechanism", "") != "anagram_fodder":
            continue
        text = getattr(src, "text", "") or ""
        word, value = raw(text), raw(getattr(src, "value", "") or "")
        if not word or not value:
            continue
        lost = _letters(word) - _letters(value)
        if not lost or (_letters(value) - _letters(word)):
            continue
        if text.rstrip().lower().endswith(("'s", "’s")) \
                and lost == collections.Counter("S"):
            continue                              # the possessive, not a cut — see above
        xf = getattr(src, "transform", "") or ""
        if xf:
            try:
                got = piece_transform.apply(
                    word, json.loads(xf) if isinstance(xf, str) else xf)
            except Exception:
                got = None
            if got is not None and _letters(got) == _letters(value):
                continue                          # the cut is RECORDED: nothing hidden
            out.append("%r: fodder %s records a cut that does not make %s"
                       % (text, word, value))
            continue
        out.append("%r: fodder %s is used as %s — %s is dropped with nothing recording "
                   "the cut" % (text, word, value,
                                "".join(sorted(lost.elements()))))
    return out


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
    out.extend(fodder_violations(parse))
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
