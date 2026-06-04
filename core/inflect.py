"""Regular inflection variants for clue-word <-> DB matching.

The reference DB stores a word in one inflected form ("suffers") but not always
its base ("suffer"), so an exact-match lookup misses obvious equivalents. The rule
(memory: feedback-inflection-match): a clue word matches its regular inflections —
singular<->plural and the regular verb forms (suffer = suffers = suffered =
suffering).

Deliberately CONSERVATIVE: only regular -s/-es/-ies, -ed, -ing transforms, with a
minimum length guard, so we do not over-match the way a full stemmer would (short
words, "is"->"i"). Generated variants that are not real words simply never match a
DB row, so they are harmless; the only job here is to also offer the real variants.
"""

_MIN = 3        # do not strip a word down shorter than this


def word_variants(word):
    """Regular inflected forms of a single word, the word itself first."""
    w = (word or "").lower().strip()
    if not w:
        return []
    out = [w]

    def add(x):
        if x and len(x) >= 2 and x not in out:
            out.append(x)

    # plural / 3rd-person singular: -s / -es / -ies
    if w.endswith("ies") and len(w) > 4:
        add(w[:-3] + "y")               # parties -> party
    if w.endswith("es") and len(w) > _MIN + 1:
        add(w[:-2])                     # boxes -> box, passes -> pass
    if w.endswith("s") and not w.endswith("ss") and len(w) > _MIN:
        add(w[:-1])                     # suffers -> suffer
    add(w + "s")                        # suffer -> suffers
    add(w + "es")                       # box -> boxes
    if w.endswith("y") and len(w) > 2:
        add(w[:-1] + "ies")             # party -> parties

    # -ing
    if w.endswith("ing") and len(w) > _MIN + 1:
        base = w[:-3]
        add(base)                       # suffering -> suffer
        add(base + "e")                 # making -> make
    else:
        add(w + "ing")                  # suffer -> suffering
        if w.endswith("e") and len(w) > 2:
            add(w[:-1] + "ing")         # make -> making

    # -ed
    if w.endswith("ed") and len(w) > _MIN:
        base = w[:-2]
        add(base)                       # suffered -> suffer
        add(w[:-1])                     # used -> use
    else:
        add(w + "ed")                   # suffer -> suffered
        if w.endswith("e") and len(w) > 2:
            add(w + "d")                # use -> used

    return out


def phrase_variants(text):
    """Variants of a (possibly multi-word) phrase, inflecting the HEAD (last word).

    The morphological head of a definition/indicator phrase is its last word, so
    "is to suffer" -> "is to suffers" etc. A single word just yields its own
    variants. The original text is always first.
    """
    text = (text or "").strip()
    if not text:
        return [text]
    parts = text.split()
    if len(parts) == 1:
        return word_variants(parts[0])
    out = [text]
    head, prefix = parts[-1], parts[:-1]
    for v in word_variants(head):
        cand = " ".join(prefix + [v])
        if cand not in out:
            out.append(cand)
    return out
