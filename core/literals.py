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

# SEED ONLY — the live lexicon now lives in the `literal_words` table (cryptic_new.db),
# editable via the clue-page admin panel (mirrors selection_indicators' list->DB shed,
# the same anti-pattern deletion/substitution shed before it). This frozenset is kept as
# (a) the one-time DB seed and (b) the fallback when no provider is wired (bare imports /
# tests). At runtime the wiring installs a provider via set_words_provider that reads the
# table, so curate literals IN THE DB, never here.
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


# Provider set by the wiring: words() -> the live set of curated literal words (lowercased),
# read from the literal_words table. None until wired, then the seed frozenset is used as a
# fallback so a bare import / unit test still behaves.
_WORDS_PROVIDER = None


def set_words_provider(fn):
    global _WORDS_PROVIDER
    _WORDS_PROVIDER = fn


def _literal_words():
    if _WORDS_PROVIDER is not None:
        try:
            return _WORDS_PROVIDER() or frozenset()
        except Exception:
            return LITERAL_WORDS
    return LITERAL_WORDS


def literal_value(text):
    """The literal letters of `text` if it is a single curated function word,
    else None. Single-token only — a multi-word phrase is never a literal here.
    The lexicon is the live `literal_words` table via the wired provider (seed
    frozenset as fallback)."""
    if not text:
        return None
    t = text.strip().lower()
    if " " in t or t not in _literal_words():
        return None
    return raw(text)
