"""Contraction / possessive handling for clue-word matching.

A clue word may carry an apostrophe suffix — a possessive or a contraction — that
is not part of the word a cryptic device actually uses. "Lionel's cracking ..."
anagrams LIONEL (not LIONELS); "he's" may mean "he" + "is"/"has". The exact token
hides the real form, so we offer the alternatives (memory: feedback-inflection-match
is the sibling rule for plural/verb forms).

Two operations:
- forms(word): for a token with an apostrophe suffix, the BASE (suffix dropped) and
  the standard expansions ("Lionel's" -> "Lionel", "Lionel is", "Lionel has"). Used
  to widen the indicator / definition / synonym lookups. Original NOT included
  (callers already try it).
- strip_suffixes(text): drop the apostrophe suffix from every word, for letter-level
  use such as anagram fodder ("Lionel's cracking" -> "Lionel cracking").

Conservative: only the regular apostrophe suffixes are expanded; anything else with
an apostrophe yields just the base, which is harmless (a non-word never matches a DB
row or an answer's letters).
"""

_APOS = "'’ʼ‘`"     # ' ' (right single) ' (modifier) ' (left single) `

# apostrophe suffix (lowercased) -> the words it expands to
_EXPANSIONS = {
    "s":  ("is", "has"),           # also the possessive (base form alone)
    "d":  ("had", "would"),
    "re": ("are",),
    "ve": ("have",),
    "ll": ("will",),
    "m":  ("am",),
}


def _apostrophe_index(word):
    for i, ch in enumerate(word):
        if ch in _APOS:
            return i
    return -1


def forms(text):
    """Base + expansion forms for a token carrying an apostrophe suffix; [] if none.

    "Lionel's" -> ["Lionel", "Lionel is", "Lionel has"]. The original is excluded.

    Strips only the apostrophe SUFFIX from its own word and KEEPS any words that
    follow it: "Virtuoso's vocal" -> ["Virtuoso vocal", "Virtuoso is vocal",
    "Virtuoso has vocal"], never "Virtuoso". Dropping the trailing word would let a
    multi-word phrase be confirmed off its first word alone — the bug this avoids.
    """
    i = _apostrophe_index(text)
    if i <= 0:
        return []
    head = text[:i]                                # word(s) up to the apostrophe
    if not head:
        return []
    rest = text[i + 1:]                            # the suffix, plus any trailing words
    j = rest.find(" ")
    suffix = (rest if j < 0 else rest[:j]).lower()  # the apostrophe suffix only
    trailing = "" if j < 0 else rest[j:]            # following words, with leading space
    out = [head + trailing]                        # base: drop only the suffix, keep words
    for e in _EXPANSIONS.get(suffix, ()):
        out.append(head + " " + e + trailing)
    return out


def strip_suffixes(text):
    """Drop the apostrophe suffix from every word — for letter-level use (anagram
    fodder). "Lionel's cracking" -> "Lionel cracking"."""
    out = []
    for w in text.split():
        i = _apostrophe_index(w)
        out.append(w[:i] if i > 0 else w)
    return " ".join(out)
