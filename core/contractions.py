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
    Operates on a single token; a no-op for multi-word text without an apostrophe.
    """
    i = _apostrophe_index(text)
    if i <= 0:
        return []
    head = text[:i]
    tail = text[i + 1:].lower()
    if not head:
        return []
    out = [head]                                   # base: drop the apostrophe suffix
    for e in _EXPANSIONS.get(tail, ()):
        out.append(head + " " + e)
    return out


def strip_suffixes(text):
    """Drop the apostrophe suffix from every word — for letter-level use (anagram
    fodder). "Lionel's cracking" -> "Lionel cracking"."""
    out = []
    for w in text.split():
        i = _apostrophe_index(w)
        out.append(w[:i] if i > 0 else w)
    return " ".join(out)
