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
        # A DERIVED variant must be at least _MIN letters. A shorter "stem" is almost
        # always a malformed strip, not a real inflection — e.g. stripping "-ed" off
        # "used" yields "us", which is not the base ("use") and spuriously matches the
        # abbreviation us->US. The word itself is always kept (out starts with w).
        if x and len(x) >= _MIN and x not in out:
            out.append(x)

    # 1) STRIP an already-inflected input toward its base (singular / infinitive).
    #    Exactly one plural form applies, so these are mutually exclusive (an -ies word
    #    also ends -es and -s; without elif it produced "parti"/"partie" junk too).
    if w.endswith("ies") and len(w) > 4:
        add(w[:-3] + "y")               # parties -> party
    elif w.endswith("es") and len(w) > _MIN + 1:
        add(w[:-2])                     # boxes -> box, passes -> pass
    elif w.endswith("s") and not w.endswith("ss") and len(w) > _MIN:
        add(w[:-1])                     # suffers -> suffer
    if w.endswith("ing") and len(w) > _MIN + 1:
        add(w[:-3])                     # suffering -> suffer
        add(w[:-3] + "e")               # making -> make
    if w.endswith("ed") and len(w) > _MIN:
        add(w[:-2])                     # suffered -> suffer
        add(w[:-1])                     # used -> use ("us" is dropped by add's _MIN guard)

    # 2) BUILD forward forms ONLY from a BASE-like word — never STACK a suffix on a word
    #    that is already a plural / -ed / -ing form. Stacking is what produced "useds"
    #    (used+s) and "useding" (used+ing).
    if not w.endswith(("s", "ed", "ing")):
        if w.endswith("y") and len(w) > 2 and w[-2] not in "aeiou":
            add(w[:-1] + "ies")         # party -> parties (consonant + y)
        elif w.endswith(_ES_ENDINGS):
            add(w + "es")               # box -> boxes (NOT on -> "ones")
        else:
            add(w + "s")                # suffer -> suffers, use -> uses, boy -> boys
        if w.endswith("e"):
            add(w[:-1] + "ing")         # make -> making
            add(w + "d")                # use -> used
        else:
            add(w + "ing")              # suffer -> suffering
            add(w + "ed")               # suffer -> suffered

    return out


def phrase_variants(text):
    """Variants of a (possibly multi-word) phrase, inflecting EACH word in turn.

    The word that carries number/tense in a phrase is not always the last one — in a
    phrasal verb like "refers to" it is the FIRST word ("refers"), with "to" last and
    uninflectable. Inflecting only the last word left "refers to" and "refer to" as
    unrelated, breaking the singular<->plural rule. So we inflect each position
    independently (one word changed at a time, never the cross-product), giving
    "refers to" -> "refer to" / "referring to" / "referred to", and "names cite" ->
    "name cite" etc. The original text is always first; single words are unchanged.

    One-at-a-time (not the full product) keeps the set small and avoids inventing
    multi-word nonsense; function words still yield only themselves, so "to"/"in" add
    nothing.
    """
    text = (text or "").strip()
    if not text:
        return [text]
    parts = text.split()
    if len(parts) == 1:
        return word_variants(parts[0])
    out = [text]
    for i, w in enumerate(parts):
        for v in word_variants(w):
            cand = " ".join(parts[:i] + [v] + parts[i + 1:])
            if cand not in out:
                out.append(cand)
    return out
