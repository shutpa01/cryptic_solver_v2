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


_PARSE_CACHE = {}


def _parse(text):
    """spaCy parse of `text`, MEMOISED. Every engine, every definition split and the
    phrase-extent all re-parse the same clue text; without this they each pay a full
    spaCy parse, so one clue cost dozens of parses. Cache keyed by exact text; bounded."""
    doc = _PARSE_CACHE.get(text)
    if doc is None:
        if len(_PARSE_CACHE) > 4000:
            _PARSE_CACHE.clear()
        doc = _nlp()(text)
        _PARSE_CACHE[text] = doc
    return doc


def pos_tags(words):
    """Coarse spaCy POS, one tag per input word, ALIGNED to `words`.

    spaCy may split a word we keep whole (a contraction: "I'd" -> "I" + "'d"), so we
    align spaCy's tokens back to our words by consuming tokens until their letters
    cover each word, taking a content tag from the group where present. Returns []
    if spaCy is unavailable or the alignment fails (caller falls back to no POS).
    """
    import re
    if not words:
        return []
    try:
        doc = list(_parse(" ".join(words)))
    except Exception:
        return []

    def alnum(s):
        return re.sub(r"[^a-z0-9]", "", (s or "").lower())

    content = {"VERB", "NOUN", "ADJ", "PROPN", "NUM", "ADV"}
    tags, si = [], 0
    for w in words:
        target = alnum(w)
        group, acc = [], ""
        while si < len(doc) and len(acc) < len(target):
            group.append(doc[si])
            acc += alnum(doc[si].text)
            si += 1
        if acc != target or not group:
            return []                          # alignment broke -> no POS
        pos = next((t.pos_ for t in group if t.pos_ in content), group[0].pos_)
        tags.append(pos)
    return tags if len(tags) == len(words) else []


def wordplay_pos_tags(ctx, wordplay_tokens):
    """POS tags for `wordplay_tokens`, tagged in the FULL clue's grammatical context.

    The type engines must NOT tag the stripped wordplay fragment on its own: removing
    the definition changes a boundary word's part of speech — e.g. "gets" is a VERB in
    "...the fellow gets bandage" (a connective) but reads as a NOUN once "bandage" is
    gone, which wrongly rejects it as an unaccounted content word and blocks assembly.
    So tag the whole clue ONCE here and return each wordplay token's tag by its position
    in the clue. The wordplay tokens are the same token objects as the clue's, so the
    mapping is by identity. Falls back to None tags if tagging/alignment fails (callers
    already treat None as "no POS").

    This is the ONE place wordplay POS is derived, so a future tagging fix lands here
    and every engine inherits it (the engines just call this instead of pos_tags)."""
    clue_words = [t for t in ctx.clue_tokens if t.kind == "word"]
    full = pos_tags([t.text for t in clue_words])
    if len(full) != len(clue_words):
        return [None] * len(wordplay_tokens)
    pos_by_id = {id(t): full[i] for i, t in enumerate(clue_words)}
    return [pos_by_id.get(id(t)) for t in wordplay_tokens]


def phrase_extent(clue_words, def_indices):
    """Grow a definition seed to its WHOLE grammatical phrase so a bound multi-word
    definition is kept intact and never split.

    Absorbs an adjacent word ONLY if it is a DEPENDENT of a word already in the span
    (its spaCy head is in the span) — i.e. part of the same subtree. It never absorbs a
    word that GOVERNS the span (an ancestor). That direction is the whole point:
      - "Avoided dealing with": "dealing" depends on "Avoided", "with" on "dealing", so
        all three are one phrase -> kept together as the definition.
      - "... in scuffle": "scuffle" is a CHILD of the preposition "in" (in governs
        scuffle), so "in" is an ancestor, not a dependent -> NOT absorbed; it stays a
        link, definition is just "scuffle".
    Conservative: stops at the first adjacent word that is not a dependent of the span."""
    if not def_indices:
        return set(def_indices)
    try:
        doc = _parse(" ".join(clue_words))
    except Exception:
        return set(def_indices)
    if len(doc) != len(clue_words):
        return set(def_indices)
    keep = set(def_indices)
    changed = True
    while changed:
        changed = False
        lo, hi = min(keep), max(keep)
        for adj in (lo - 1, hi + 1):
            if adj < 0 or adj >= len(doc) or adj in keep:
                continue
            if doc[adj].head.i in keep:          # adj depends on the span -> same phrase
                keep.add(adj)
                changed = True

    # TRIM a DANGLING edge the grow added: a preposition/conjunction whose own object lies
    # OUTSIDE the span leaves an incomplete phrase — "A term" grows to "A term [for]" when
    # "for"'s object "Oxbridge" can't be reached. Seed words are never trimmed; a complete
    # phrase (intransitive "dealing with", whose "with" has no out-of-span child) is kept.
    seed = set(def_indices)
    _DANGLE = {"ADP", "CCONJ", "SCONJ", "PART"}
    changed = True
    while changed and len(keep) > 1:
        changed = False
        for edge in (max(keep), min(keep)):
            if edge in seed or doc[edge].pos_ not in _DANGLE:
                continue
            if any(ch.i not in keep for ch in doc[edge].children):
                keep.discard(edge)
                changed = True
                break
    return keep


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
        doc = _parse(" ".join(clue_words))
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
