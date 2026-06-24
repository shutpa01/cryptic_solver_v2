"""Shared wordplay helpers for the catalog engines.

The cross-cutting bits that were duplicated across charade / anagram / anagram+
charade / anagram+container — POS sets, residue/link classification, anagram-
indicator detection, fodder letters — live here, ONE home each, so a fix lands in
one place. The TYPE-SPECIFIC assembly (how each clue type tiles or inserts) stays in
its own engine, per the isolated-engines rule; only these common helpers are shared.
"""

import unicodedata

from core import contractions

# Function-word parts of speech — never an operative indicator or fodder piece;
# peelable from an indicator phrase and eligible as link/connective words.
FUNCTION_POS = frozenset({"ADP", "PART", "AUX", "DET", "CCONJ", "SCONJ"})
# Allowed as connective glue (a link): function words plus connective verbs/adverbs.
GLUE_POS = FUNCTION_POS | frozenset({"VERB", "ADV"})


def raw(text):
    """The uppercase alphabetic letters of `text`, with diacritics FOLDED to their base
    letter (cryptic convention: gratiné -> GRATINE, Señor -> SENOR), so accented fodder
    and pieces letter-match the answer. Strict no-op for plain ASCII text: NFKD leaves
    unaccented characters unchanged, and only combining marks are dropped."""
    decomposed = unicodedata.normalize("NFKD", text or "")
    return "".join(c for c in decomposed.upper()
                   if c.isalpha() and not unicodedata.combining(c))


def is_anagram_indicator(text, indicator_types):
    """True if `text` is a DB-confirmed anagram indicator (inflection/contraction
    aware via the injected indicator_types)."""
    try:
        return "anagram" in (indicator_types(text) or set())
    except Exception:
        return False


def is_link_or_glue(text, pos_tag, is_link):
    """A residue word that is a function/connective word (link-eligible): a known
    link word, or a function / VERB / ADV part of speech."""
    return bool(is_link and is_link(text))


def anagram_indicator_source(tokens, indicator_types):
    """The provenance of a candidate anagram-indicator run: 'db' if any token is a
    DB-confirmed anagram indicator, else 'pending'.

    'pending' is the MISSING-INDICATOR FALLBACK: when the anagram is otherwise proven
    (the fodder's letters equal the answer/span exactly) and the definition is found,
    the leftover word adjacent to the fodder MUST be the indicator even if the DB does
    not know it yet — so we accept it provisionally and queue it for enrichment, the
    same loop as the definition fallback. The CALLER decides which run is the candidate
    and applies its evidence guard (exactly one leftover run / adjacency); this only
    reports whether that run is confirmed or provisional."""
    if any(is_anagram_indicator(t.text, indicator_types) for t in tokens):
        return "db"
    return "pending"


def adjacent_run(skipped, a, b):
    """The contiguous run of skipped word-indices touching a piece span [a:b] —
    immediately after it (b, b+1, ...) or, failing that, immediately before (a-1,
    a-2, ...). Used by the missing-indicator fallback to find the indicator next to
    the anagram piece."""
    sk = set(skipped)
    run, k = [], b
    while k in sk:
        run.append(k)
        k += 1
    if run:
        return run
    k = a - 1
    while k in sk:
        run.append(k)
        k -= 1
    return sorted(run)


def fodder_letter_forms(tokens):
    """Candidate letter strings for an anagram fodder run: as written, and with
    apostrophe/possessive suffixes stripped ("Lionel's" -> LIONEL). Empty strings
    dropped."""
    as_written = "".join(raw(t.text) for t in tokens)
    stripped = "".join(raw(contractions.strip_suffixes(t.text)) for t in tokens)
    return [s for s in (as_written, stripped) if s]
