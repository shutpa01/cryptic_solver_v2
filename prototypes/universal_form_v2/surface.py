"""Surface-level operations on clue text (tokenisation, normalisation).

Mirrors `signature_solver/solver.py::_normalize_clue` followed by
whitespace `.split()` and per-token surrounding-punctuation strip.
This preserves internal apostrophes — "bird's" stays as one token —
so possessive handling lands at DB-lookup time (via
`signature_solver/db.py::_word_variants`), not at tokenisation time.

Used by both the verifier (`clipboard_verifier._surface_words`) and
the matcher (`tree_matcher`) to keep their notions of "a clue word"
identical.
"""
from __future__ import annotations

import unicodedata


# Surrounding punctuation we strip from each whitespace-split token.
# We deliberately keep apostrophes and hyphens (they're internal to
# tokens like "bird's", "won't", "twenty-one"). Dot, comma, semicolon,
# colon, exclamation/question mark, double-quote, parentheses are
# stripped from token edges only.
_TRIM_CHARS = ".,;:!?\"()"


def _normalize(text: str) -> str:
    """Mirrors signature_solver/solver.py::_normalize_clue."""
    if not text:
        return ""
    nfkd = unicodedata.normalize("NFKD", text)
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    for old, new in [
        ("‘", "'"),
        ("’", "'"),
        ("“", '"'),
        ("”", '"'),
        ("–", "-"),
        ("—", "-"),
    ]:
        ascii_text = ascii_text.replace(old, new)
    return ascii_text


def tokenize(text: str) -> list:
    """Return surface tokens preserving internal apostrophes/hyphens.

    "Cream tea uncovered, with fewer calories?"
        -> ["Cream", "tea", "uncovered", "with", "fewer", "calories"]
    "The wingless bird's screech"
        -> ["The", "wingless", "bird's", "screech"]
    "twenty-one"
        -> ["twenty-one"]
    """
    if not text:
        return []
    out = []
    for tok in _normalize(text).split():
        clean = tok.strip(_TRIM_CHARS)
        # Skip punctuation-only tokens (e.g. lone "-" from em-dashes
        # between phrases). Real words always contain at least one
        # alphabetic character.
        if clean and any(c.isalpha() for c in clean):
            out.append(clean)
    return out
