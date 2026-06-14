"""Spoonerism indicators — a spoonerism clue names the Reverend Spooner.

Almost every spoonerism clue is flagged by "Spooner" in some form ("Spooner's",
"according to Spooner", "Reverend Spooner", "Dr Spooner says"). The reference table
carries no 'spoonerism' type, and the cue is this single recognisable name, so the
gate lives here. Without it the engine must abstain — an onset swap that happens to
land on a real phrase is not a spoonerism unless the clue says so.
"""

# core tokens: anything containing "spooner"
# lead words that attach to the name as part of the indicator phrase
_LEAD = {"according", "to", "reverend", "rev", "dr", "drs", "says", "said",
         "say", "old", "the"}


def _norm(text):
    return "".join(c for c in text.lower() if c.isalpha())


def find_indicator(words):
    """The spoonerism indicator's word positions (the Spooner name plus any attached
    lead words like "according to"), or None if Spooner is not named."""
    norms = [_norm(t.text) for t in words]
    core = [i for i, w in enumerate(norms) if "spooner" in w]
    if not core:
        return None
    lo = min(core)
    while lo - 1 >= 0 and norms[lo - 1] in _LEAD:
        lo -= 1
    hi = max(core)
    return list(range(lo, hi + 1))
