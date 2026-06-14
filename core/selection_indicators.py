"""Selection indicators — which indicator words license which letter-selection rule.

A selection piece (core.selection.select_span) is only legitimate when an indicator
LICENSES the rule: "house originally" -> H is a first-letter selection ONLY because
"originally" is present and means "take the first letter". Without the indicator a
selection is a fabrication (any word yields a first/middle/alternate run), so the
charade/container engines must find a matching indicator before filling a SEL slot.

Kept deliberately conservative: only words/phrases that unambiguously mean a letter
selection. Ambiguous words (head, top, start, heart, inside, sides) are EXCLUDED for
now — they double as content words and other engines' indicators, and a false match
would let selection hijack a clue. The map is the single gate; widen it only on
evidence.
"""

# rule -> the single-word indicators that license it.
_SINGLE = {
    "first":         {"originally", "initially", "primarily", "firstly",
                      "first", "leading", "opening"},
    "last":          {"ultimately", "finally", "lastly", "latterly"},
    "outer":         {"outskirts", "borders", "extremes", "bounds", "limits"},
    "middle":        {"essentially", "centrally"},
    "alternate":     {"occasionally", "regularly", "alternately",
                      "periodically", "oddly", "evenly"},
    "remove_middle": {"heartless"},
}

# rule -> multi-word indicator phrases (matched as a contiguous run, lower-cased).
_PHRASES = {
    "first":     [("at", "first"), ("to", "start"), ("to", "begin")],
    "last":      [("at", "last"), ("at", "the", "end")],
    "outer":     [("outer", "limits"), ("on", "the", "edges")],
    "middle":    [("at", "heart"), ("at", "the", "centre"),
                  ("at", "the", "center")],
    "alternate": [("now", "and", "then"), ("every", "other"),
                  ("at", "intervals"), ("from", "time", "to", "time")],
}

# Reverse single-word index for O(1) lookup.
_WORD_RULE = {w: rule for rule, words in _SINGLE.items() for w in words}


def _norm(text):
    return "".join(c for c in text.lower() if c.isalpha())


def find_indicators(words):
    """[(rule, (idx, ...)), ...] — every selection indicator in `words` (a list of
    token objects with .text), each with the word indices it occupies. Multi-word
    phrases are matched as contiguous runs and reported before single words so the
    caller prefers the longer match. Empty when none."""
    norms = [_norm(t.text) for t in words]
    out = []
    # multi-word phrases first (longer, more specific)
    for rule, phrases in _PHRASES.items():
        for phrase in phrases:
            L = len(phrase)
            for i in range(len(norms) - L + 1):
                if tuple(norms[i:i + L]) == phrase:
                    out.append((rule, tuple(range(i, i + L))))
    # single words
    for i, w in enumerate(norms):
        rule = _WORD_RULE.get(w)
        if rule is not None:
            out.append((rule, (i,)))
    return out
