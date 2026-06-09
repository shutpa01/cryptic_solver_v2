"""Curated literal-word lexicon — short function words a setter may use as their
OWN uppercased letters (it -> IT, a -> A, on -> ON), a *raw* reading the reference
DB does not provide (it stores synonyms/abbreviations, never a word's own letters).

This is an ADDITIVE value source for the container family (engines that look up a
value via lookup_all): it lets a piece that IS a function word contribute its
letters, e.g. VERITY = VER(IT)Y where "it" -> IT. It is NOT used by the charade
engine, which already reads literals through its own signature-pinned LIT_F slot.

WHY A CURATED LIST (not "any word -> its own letters"): a blanket rule explodes
false positives. The list is FUNCTION WORDS ONLY, and every entry was OBSERVED
acting as a literal in real solved clues (mined read-only from the 20 LIT_F charade
signatures over 53,983 clues; see _extract_literal_clues.py). Deliberately EXCLUDED,
because each is a different device or genuinely unsafe:
  - isolated/quoted single letters (a bare "P", "B") — a letter-token device,
  - proper nouns read as letters (Ron, Ted, Al) — un-lexicon-able,
  - content words read as letters (ants -> ANTS) — the false-positive class,
  - contractions (I'm, it's) — handled by the contraction layer.

The literal is only ever a CANDIDATE: the engine still requires exact reconstruction
of the answer, a confirmed definition, a matching signature, and links-classified-
last. So a 2-letter literal cannot be taken blindly — it survives only when the whole
clue rebuilds exactly. Grow this list from evidence, never by guessing.
"""

from core.wordplay import raw

# Function words only, each observed as a literal in real clues (count in the mine).
LITERAL_WORDS = frozenset({
    # articles / determiners
    "a", "an", "the", "no", "our", "her",
    # prepositions / particles
    "in", "to", "on", "at", "of", "for", "up", "out", "off",
    # conjunctions
    "or", "as", "and", "if", "so", "but",
    # pronouns
    "i", "it", "me", "us", "he", "one",
    # verbs / auxiliaries / adverbs
    "is", "are", "be", "do", "go", "was", "am", "not",
})


def literal_value(text):
    """The literal letters of `text` if it is a single curated function word,
    else None. Single-token only — a multi-word phrase is never a literal here."""
    if not text:
        return None
    t = text.strip().lower()
    if " " in t or t not in LITERAL_WORDS:
        return None
    return raw(text)
