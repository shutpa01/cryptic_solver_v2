"""Palindrome indicators — phrases signalling the answer reads the same both ways.

The reference indicators table carries none (palindrome overlaps reversal wording, so
adding them there would pollute the shared table), and the set is small and specific,
so it lives here — the single gate for the palindrome engine. A palindrome clue has no
clue-letter source; the indicator is the ONLY evidence that the symmetric answer is
intended, so without one the engine must abstain.

Two ways to match:
  - a known single word / contiguous phrase ("either way", "back and forth");
  - a both-directions pair conjoined in the clue ("from the east and from the west" ->
    east + west), which catches the many phrasings without enumerating them all.
"""

_SINGLE = {"palindrome", "palindromic", "palindromes", "reversible",
           "symmetrical", "symmetric"}

# contiguous multi-word phrases (lower-cased, alphabetic-only per word)
_PHRASES = [
    ("either", "way"), ("both", "ways"), ("any", "way"),
    ("back", "and", "forth"), ("backwards", "and", "forwards"),
    ("forwards", "and", "backwards"), ("to", "and", "fro"),
    ("either", "direction"), ("both", "directions"),
    ("same", "both", "ways"), ("the", "same", "either", "way"),
    ("same", "in", "reverse"), ("same", "backwards"),
]

# opposite-direction pairs: if BOTH appear, those two words are the indicator core.
_OPP_PAIRS = [
    {"east", "west"}, {"left", "right"}, {"up", "down"}, {"north", "south"},
    {"forwards", "backwards"}, {"forward", "backward"}, {"back", "forth"},
]


def _norm(text):
    return "".join(c for c in text.lower() if c.isalpha())


def find_indicator(words):
    """The palindrome indicator word positions in `words` (a list of token objects
    with .text), or None if none. The connecting function words ("from"/"the"/"and")
    are left for the engine to classify as links."""
    norms = [_norm(t.text) for t in words]
    nset = set(norms)
    pos = set()
    for i, w in enumerate(norms):
        if w in _SINGLE:
            pos.add(i)
    for phrase in _PHRASES:
        L = len(phrase)
        for i in range(len(norms) - L + 1):
            if tuple(norms[i:i + L]) == phrase:
                pos.update(range(i, i + L))
    for pair in _OPP_PAIRS:
        if pair <= nset:
            pos.update(i for i, w in enumerate(norms) if w in pair)
    return sorted(pos) if pos else None
