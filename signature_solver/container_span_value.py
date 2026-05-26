"""Compatibility shim for GT2 container span/value recovery."""
from __future__ import annotations


def find_container_span_value_suggestion(clue_text, answer, db, candidates):
    """Find one validated phrase -> shell candidate for a failed container.

    Return the legacy ``(phrase, value)`` shape from the first GT2 bundle.
    """
    from .gt2_candidate_generator import generate_gt2_candidates

    bundles = generate_gt2_candidates(clue_text, answer, db, candidates)
    if not bundles:
        return None
    bundle = bundles[0]
    phrase, values = next(iter(bundle.overlay_synonyms.items()))
    return phrase, values[0]


def span_suggestion_actually_used(sr_obj, phrase, value):
    if sr_obj is None or sr_obj.result is None:
        return False
    phrase_clean = phrase.lower().strip(",.;:!?\"'()-")
    value_clean = "".join(c for c in value.upper() if c.isalpha())
    for word, _tok, val, *_ in sr_obj.result.word_roles:
        word_clean = word.lower().strip(",.;:!?\"'()-")
        role_value = "".join(c for c in (val or "").upper() if c.isalpha())
        if word_clean == phrase_clean and role_value == value_clean:
            return True
    return False
