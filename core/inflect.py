"""Regular inflection variants for clue-word <-> DB matching.

The reference DB stores a word in one inflected form ("suffers") but not always
its base ("suffer"), so an exact-match lookup misses obvious equivalents. The rule
(memory: feedback-inflection-match): a clue word matches its regular inflections —
singular<->plural and the regular verb forms (suffer = suffers = suffered =
suffering).

Deliberately CONSERVATIVE: only regular -s/-es/-ies, -ed, -ing transforms, with a
minimum length guard, so we do not over-match the way a full stemmer would (short
words, "is"->"i"). A generated variant must be a genuine inflection of the SAME
word — it must never wander into a DIFFERENT word. The earlier assumption that
"invented forms never match a DB row, so they are harmless" was FALSE and caused a
fabricated letter source: appending "-es" to "on" produced "ones" — which is one+s,
a form of "one", not of "on" — and "ones" IS a DB row (ones=I, ones=FLAT), so those
values were wrongly attributed to "on" (MANITOBA's "I <- on", INFLATION's "FLAT <-
on"). See memory feedback-no-derived-answer-letters. The fix: each suffix is only
applied where it is a valid English inflection of the word, so the bridge stays
within one word.
"""

_MIN = 3        # do not strip a word down shorter than this
# Endings after which the plural / 3rd-person "-es" is the correct suffix (box->boxes,
# pass->passes, buzz->buzzes, church->churches, dish->dishes, go->goes). Elsewhere the
# plural is "-s" (one->ones), so appending "-es" would invent a different word.
_ES_ENDINGS = ("s", "x", "z", "ch", "sh", "o")

# Closed-class function words DO NOT inflect (no plural / verb forms), so generating
# "-s/-es/-ed/-ing" forms for them only invents forms of OTHER words and imports their
# DB values — the same fabricated-letter breach as on->"ones" (feedback-no-derived-
# answer-letters), e.g. us->"uses"/"used" (forms of "use"). These get NO variants but
# themselves. Deliberately EXCLUDES words with a common verb/noun sense (up, out, off,
# over, under, down, round, near, will, can, may, are, be, ...) so a real inflection is
# never blocked; only unambiguous function words are listed.
_FUNCTION_WORDS = frozenset("""
a an the this that these those
i me my mine myself we us our ours ourselves you your yours yourself yourselves
he him his himself she her hers herself it its itself they them their theirs themselves
who whom whose which what
of to in on at by for with from into onto upon about above below beneath beside between
beyond against across among amongst before after behind during despite except inside
outside toward towards until unto via within without throughout through
and or but nor so yet as if than because although though while whilst whereas unless
whether since not no
""".split())


def word_variants(word):
    """Regular inflected forms of a single word, the word itself first."""
    w = (word or "").lower().strip()
    if not w:
        return []
    if w in _FUNCTION_WORDS:        # function words do not inflect — no variants but self
        return [w]
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
    if w.endswith(_ES_ENDINGS):
        add(w + "es")                   # box -> boxes (NOT on -> "ones")
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
