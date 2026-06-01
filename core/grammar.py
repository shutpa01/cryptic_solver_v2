"""Narrow grammar helper — definition-EXTENT only.

Not a router, not a guesser of mechanisms (that was dropped from the core path
as noisy). Its single job: given the clue words and which ones the DB confirmed
as the definition, decide how far the real definition phrase extends, by walking
spaCy's dependency parse outward and absorbing function words that are
grammatically bound to the confirmed head (its determiner, the preposition that
heads its phrase).

spaCy is loaded lazily and cached, so code paths that never call this stay light.
"""

_NLP = None


def _nlp():
    global _NLP
    if _NLP is None:
        import spacy
        _NLP = spacy.load("en_core_web_sm")
    return _NLP


def extend_definition_indices(clue_words, def_indices, wordplay_indices):
    """Grow a confirmed definition outward toward the wordplay.

    clue_words: list of surface word strings, in clue order.
    def_indices: indices (into clue_words) the DB confirmed as the definition.
    wordplay_indices: indices that carry a wordplay role (must NOT be absorbed).

    Returns the (possibly larger) set of definition indices. Absorbs ONLY a word
    that is (a) adjacent to the current definition span, (b) grammatically bound
    to a word already in the span (it is the head's determiner/prep, or its own
    head is in the span), (c) a function word (DET/ADP), and (d) not a wordplay
    word. Conservative: stops at the first word that fails any test.
    """
    if not def_indices:
        return set(def_indices)
    try:
        doc = _nlp()(" ".join(clue_words))
    except Exception:
        return set(def_indices)               # no grammar available -> unchanged
    if len(doc) != len(clue_words):
        return set(def_indices)               # tokenisation mismatch -> bail safe

    keep = set(def_indices)
    wp = set(wordplay_indices)

    # Walk LEFT from the leftmost definition word, absorbing bound function words.
    changed = True
    while changed:
        changed = False
        lo, hi = min(keep), max(keep)
        for adj in (lo - 1, hi + 1):
            if adj < 0 or adj >= len(doc) or adj in keep or adj in wp:
                continue
            tok = doc[adj]
            if tok.pos_ not in ("DET", "ADP"):
                continue
            # bound to the span if its head is in the span, or (for a preposition)
            # the word it governs is in the span.
            head_in = tok.head.i in keep
            child_in = any(c.i in keep for c in tok.children)
            if head_in or child_in:
                keep.add(adj)
                changed = True
    return keep
