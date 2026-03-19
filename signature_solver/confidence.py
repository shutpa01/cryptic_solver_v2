"""Confidence scoring for signature solver results.

Starts at 100 and deducts for problems a user would spot.
By the time we reach scoring, assembly is already verified (pieces make the answer).

Penalties:
  -60  nonsense synonym (value not a real word)
  -30  circularity (answer used to explain itself)
  -20  operation missing its indicator
  -20  unconfirmed synonym (real word, plausible DB gap)
  -15  unconfirmed homophone
  -10  unconfirmed abbreviation
   -5  indicator not verified in DB
   -5  unverified link word

Score 0-100:
  80+ = HIGH — serve directly to user
  50-79 = MEDIUM — usable with caveats
  <50 = LOW — not user-facing
"""

from .tokens import *


# Operations that need an indicator to be explainable to a user
_FODDER_NEEDS_INDICATOR = {
    ANA_F: {ANA_I},
    HID_F: {HID_I},
    HOM_F: {HOM_I},
    POS_F: {POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
            POS_I_ALTERNATE, POS_I_HALF, POS_I_TRIM_FIRST,
            POS_I_TRIM_LAST, POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER},
    DEL_F: {DEL_I, POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
            POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER},
}


def score_result(result, words, answer, analyses, db):
    """Score an explanation by deducting for problems a user would spot.

    Starts at 100. Deducts for unverified pieces, missing indicators,
    nonsense words, circularity. Source doesn't matter — a user sees
    the explanation, not how it was produced.

    Args:
        result: SignatureResult with word_roles
        words: wordplay words
        answer: expected answer (uppercase, no spaces)
        analyses: word analyses
        db: RefDB instance

    Returns:
        (score 0-100, list of (reason, delta) tuples)
    """
    reasons = []
    roles = result.word_roles

    score = 100

    # --- Track which operations and indicators are present ---
    fodder_types_present = set()
    indicator_types_present = set()

    for word, tok, val in roles:

        # Link words
        if tok == LNK:
            if db.is_link_word(word.lower()) or db.get_indicator_types(word.lower()):
                continue
            score -= 5
            reasons.append((f"unverified link '{word}'", -5))
            continue

        # Indicators
        if tok in INDICATOR_TOKENS:
            indicator_types_present.add(tok)
            verified = _is_verified_indicator(word, tok, db)
            if not verified:
                score -= 5
                reasons.append((f"indicator '{word}' not verified in DB", -5))
            continue

        # Mechanically self-evident pieces — no penalty possible
        if tok in (ANA_F, HID_F, POS_F, DEL_F):
            fodder_types_present.add(tok)
            continue

        # RAW letters — already verified by assembly, no penalty
        if tok == RAW:
            continue

        # Synonyms
        if tok == SYN_F:
            w = word.lower().strip(".,;:!?\"'()-")
            if val and _is_confirmed_synonym(w, val, db):
                continue
            if val and db.is_real_word(val):
                score -= 20
                reasons.append((f"SYN '{word}'={val} unconfirmed (real word)", -20))
            else:
                score -= 60
                reasons.append((f"SYN '{word}'={val} not a real word", -60))
            continue

        # Abbreviations
        if tok == ABR_F:
            w = word.lower().strip(".,;:!?\"'()-")
            abbrs = db.get_abbreviations(w)
            if not (val and val in abbrs):
                score -= 10
                reasons.append((f"ABR '{word}'={val} unconfirmed", -10))
            continue

        # Homophones
        if tok == HOM_F:
            fodder_types_present.add(tok)
            w = word.lower().strip(".,;:!?\"'()-")
            if not (val and _is_confirmed_homophone(w, val, db)):
                score -= 15
                reasons.append((f"HOM '{word}'={val} unconfirmed", -15))
            continue

    # --- Structural: operations missing their indicator ---
    for fodder_tok in fodder_types_present:
        if fodder_tok in _FODDER_NEEDS_INDICATOR:
            needed = _FODDER_NEEDS_INDICATOR[fodder_tok]
            if not (needed & indicator_types_present):
                score -= 20
                reasons.append((f"missing indicator for {fodder_tok}", -20))

    # --- Circularity: definition reused as fodder ---
    circ = _check_circularity(roles, answer)
    if circ:
        score += circ
        reasons.append(("definition reused as fodder", circ))

    return max(0, min(100, score)), reasons


def _is_verified_indicator(word, tok, db):
    """Check if word is a known indicator of the expected type."""
    ind_types = db.get_indicator_types(word.lower())
    if not ind_types:
        return False

    expected = _token_to_db_types(tok)
    if not expected:
        if tok in (POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
                   POS_I_ALTERNATE, POS_I_HALF, POS_I_TRIM_FIRST,
                   POS_I_TRIM_LAST, POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER):
            expected = {"parts", "acrostic", "alternating", "selection", "deletion"}

    for wtype, _sub, conf in ind_types:
        if wtype in expected:
            return True

    return False


def _token_to_db_types(token):
    """Map indicator token back to DB wordplay_type strings."""
    return {
        ANA_I: {"anagram"},
        REV_I: {"reversal"},
        CON_I: {"container", "insertion"},
        DEL_I: {"deletion"},
        HID_I: {"hidden"},
        HOM_I: {"homophone"},
    }.get(token, set())


def _is_confirmed_synonym(word, val, db):
    """Check if val is a known synonym of word."""
    syns = db.get_synonyms(word)
    if val in syns:
        return True
    all_syns = db.get_synonyms(word, max_len=len(val) + 5)
    return val in all_syns


def _is_confirmed_homophone(word, val, db):
    """Check if val is a known homophone of word (direct or via synonym)."""
    homos = db.get_homophones(word)
    if val in homos:
        return True
    # Check synonym → homophone chain
    for syn in db.get_synonyms(word):
        if val in db.get_homophones(syn.lower()):
            return True
    return False


def _check_circularity(roles, answer):
    """Penalty if a synonym value used in the parse IS the answer.

    Exception: double definitions where the only fodder is a single SYN_F
    whose value equals the answer — that's correct by design.
    """
    fodder_roles = [(w, t, v) for w, t, v in roles if t in FODDER_TOKENS]
    for word, tok, val in fodder_roles:
        if tok == SYN_F and val == answer:
            if len(fodder_roles) == 1:
                return 0
            return -30
    return 0
