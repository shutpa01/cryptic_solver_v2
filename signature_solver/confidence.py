"""Confidence scoring for signature solver results.

By the time we reach scoring, we already know:
1. A signature pattern was matched
2. The assembly produces the answer (verified by _verify_combo)

So scoring checks: how well verified is each piece?

Score 0-100:
  80+ = high confidence, serve directly
  50-79 = medium, pass as strong evidence to API
  <50 = low, pass as weak evidence / discard
"""

from .tokens import *


def score_result(result, words, answer, analyses, db):
    """Score a SignatureResult for confidence.

    Logic:
    1. Signature matched + assembly verified = base 60
    2. Each piece verified in DB = bonus
    3. Each piece NOT verified = penalty
    4. All pieces verified = guaranteed 80+
    5. Circularity = hard penalty

    Returns:
        int score 0-100, plus list of (reason, delta) tuples
    """
    reasons = []
    roles = result.word_roles

    # --- Base: signature matched and assembly verified ---
    score = 60
    reasons.append(("signature matched + assembly verified", 60))

    # --- Check each piece ---
    n_pieces = 0
    n_verified = 0

    for word, tok, val in roles:
        if tok == LNK:
            # Link words: check they're real
            if db.is_link_word(word.lower()):
                continue  # fine, no bonus or penalty
            elif db.get_indicator_types(word.lower()):
                continue  # indicator used as link, acceptable
            else:
                score -= 5
                reasons.append((f"unverified link word '{word}'", -5))
            continue

        if tok in INDICATOR_TOKENS:
            n_pieces += 1
            verified, detail = _verify_indicator(word, tok, db)
            if verified:
                n_verified += 1
                score += detail
                reasons.append((f"indicator '{word}' verified", detail))
            else:
                score -= 5
                reasons.append((f"indicator '{word}' not in DB", -5))
            continue

        if tok in (SYN_F, ABR_F, HOM_F):
            n_pieces += 1
            verified, detail = _verify_lookup(word, tok, val, db)
            if verified:
                n_verified += 1
                score += detail
                reasons.append((f"{tok} '{word}'={val} verified", detail))
            else:
                score -= 5
                reasons.append((f"{tok} '{word}'={val} not in DB", -5))
            continue

        if tok in (ANA_F, HID_F, POS_F):
            # These are mechanically verified by assembly — always count as verified
            n_pieces += 1
            n_verified += 1
            score += 3
            reasons.append((f"{tok} '{word}' mechanically verified", 3))
            continue

        if tok == RAW:
            # Raw letters — the word itself is in the answer
            n_pieces += 1
            w_alpha = "".join(c for c in word.upper() if c.isalpha())
            if w_alpha in answer:
                n_verified += 1
                score += 2
                reasons.append((f"RAW '{word}' in answer", 2))
            else:
                score -= 3
                reasons.append((f"RAW '{word}' not in answer", -3))
            continue

    # --- Bonus if ALL pieces verified ---
    # If every piece is independently verified, the mechanical solve is confirmed.
    # Guarantee a score of 82 (HIGH threshold) — only circularity can pull it below.
    if n_pieces > 0 and n_verified == n_pieces:
        target = 82
        bonus = max(2, target - score)
        score += bonus
        reasons.append((f"all {n_pieces} pieces verified", bonus))

    # --- Circularity check ---
    circ = _check_circularity(roles, answer)
    if circ:
        score += circ
        reasons.append(("circularity penalty", circ))

    return max(0, min(100, score)), reasons


def _verify_indicator(word, tok, db):
    """Check if word is a known indicator of the expected type.

    Returns (verified: bool, score_delta: int).
    """
    ind_types = db.get_indicator_types(word.lower())
    if not ind_types:
        return False, 0

    expected = _token_to_db_types(tok)
    if not expected:
        # Positional indicators — check for 'parts' wordplay_type
        if tok in (POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
                   POS_I_ALTERNATE, POS_I_HALF, POS_I_TRIM_FIRST,
                   POS_I_TRIM_LAST, POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER):
            expected = {"parts", "acrostic", "alternating", "selection", "deletion"}

    found_conf = None
    for wtype, _sub, conf in ind_types:
        if wtype in expected:
            found_conf = conf
            break

    if found_conf is None:
        return False, 0

    if found_conf == "very_high":
        return True, 5
    elif found_conf == "high":
        return True, 4
    else:
        return True, 3


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


def _verify_lookup(word, tok, val, db):
    """Check if a synonym/abbreviation/homophone is in the DB.

    Returns (verified: bool, score_delta: int).
    """
    if not val or not isinstance(val, str):
        return False, 0

    w = word.lower().strip(".,;:!?\"'()-")

    if tok == ABR_F:
        abbrs = db.get_abbreviations(w)
        if val in abbrs:
            return True, 5
        return False, 0

    if tok == SYN_F:
        syns = db.get_synonyms(w)
        if val in syns:
            return True, 5
        # Try broader search
        all_syns = db.get_synonyms(w, max_len=len(val) + 5)
        if val in all_syns:
            return True, 3
        return False, 0

    if tok == HOM_F:
        homos = db.get_homophones(w)
        if val in homos:
            return True, 5
        # Check synonym → homophone chain
        syns = db.get_synonyms(w)
        for syn in syns:
            if val in db.get_homophones(syn.lower()):
                return True, 4
        return False, 0

    return False, 0


def _check_circularity(roles, answer):
    """Penalty if a synonym value used in the parse IS the answer."""
    for word, tok, val in roles:
        if tok == SYN_F and val == answer:
            return -30
    return 0
