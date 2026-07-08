"""Palindrome indicators — phrases signalling the answer reads the same both ways.

DB-DRIVEN (was a hardcoded word list — the same anti-pattern selection/deletion shed).
The vocabulary now lives in the reference `indicators` table under wordplay_type
'palindrome', so it is curated through the clue-page add gate (never a code edit) and
cannot be silently extended to force a solve. The wiring (engine_registry) loads the rows
and installs them via set_vocab(); this module keeps only the MATCH LOGIC, no vocabulary.

The rows are bucketed by subtype:
  - 'single'   a one-word indicator ("palindrome", "reversible");
  - 'phrase'   a contiguous multi-word phrase ("either way", "back and forth");
  - 'opp_pair' a both-directions pair stored as two words ("east west"): if BOTH appear
               anywhere in the clue, those two words are the indicator core.

A palindrome clue has no clue-letter source; the indicator is the ONLY evidence that the
symmetric answer is intended, so without one the engine must abstain.
"""

# Installed by the wiring: normalised vocabulary from the indicators table. Empty until
# wired (find_indicator then returns None, exactly like an unwired selection provider).
_VOCAB = {"singles": frozenset(), "phrases": (), "pairs": ()}


def set_vocab(singles, phrases, pairs):
    """Install the palindrome vocabulary (already _norm-ed): singles = set of words,
    phrases = list of word-tuples, pairs = list of 2-word frozensets."""
    global _VOCAB
    _VOCAB = {"singles": frozenset(singles),
              "phrases": tuple(tuple(p) for p in phrases),
              "pairs": tuple(frozenset(p) for p in pairs)}


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
        if w in _VOCAB["singles"]:
            pos.add(i)
    for phrase in _VOCAB["phrases"]:
        L = len(phrase)
        for i in range(len(norms) - L + 1):
            if tuple(norms[i:i + L]) == phrase:
                pos.update(range(i, i + L))
    for pair in _VOCAB["pairs"]:
        if pair <= nset:
            pos.update(i for i, w in enumerate(norms) if w in pair)
    return sorted(pos) if pos else None
