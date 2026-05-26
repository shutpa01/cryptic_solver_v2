"""Analyze each word in the wordplay window to determine possible roles."""

import re
from .tokens import *
from .db import RefDB


def clean_word(word):
    """Strip punctuation from a clue word for DB lookups.

    Matches the enricher.py pattern: strip .,;:!?"'()-
    """
    return word.lower().strip(".,;:!?\"'()-")


class WordAnalysis:
    """Possible roles for a single word or phrase."""

    def __init__(self, text):
        self.text = text
        self.roles = {}  # token -> list of values
        # e.g. {ABR_F: ['N', 'S'], SYN_F: ['NORTH', 'COMPASS POINT'], ANA_I: [True]}

    def add_role(self, token, value=None):
        if token not in self.roles:
            self.roles[token] = []
        if value is not None and value not in self.roles[token]:
            self.roles[token].append(value)
        elif value is None and not self.roles[token]:
            self.roles[token].append(True)

    def possible_tokens(self):
        return set(self.roles.keys())

    def __repr__(self):
        parts = []
        for tok, vals in self.roles.items():
            if vals == [True]:
                parts.append(tok)
            else:
                short_vals = vals[:3]
                parts.append(f"{tok}({','.join(str(v) for v in short_vals)})")
        return f"<{self.text}: {' | '.join(parts)}>"


def analyze_words(words, answer, db):
    """Analyze each word in the wordplay window.

    Args:
        words: list of words from the wordplay window
        answer: the known answer (uppercase)
        db: RefDB instance

    Returns:
        list of WordAnalysis objects, one per word
    """
    answer_upper = answer.upper().replace(" ", "").replace("-", "")
    answer_len = len(answer_upper)
    analyses = []

    for word in words:
        wa = WordAnalysis(word)
        w_lower = clean_word(word)
        w_alpha = "".join(c for c in word.upper() if c.isalpha())

        # --- Check abbreviation DB ---
        abbrs = db.get_abbreviations(w_lower)
        for a in abbrs:
            wa.add_role(ABR_F, a)
        for a in BUILTIN_ABBREVIATIONS.get(w_lower, ()):
            wa.add_role(ABR_F, a)

        # --- Check indicator DB ---
        ind_types = db.get_indicator_types(w_lower)
        for wtype, subtype, confidence in ind_types:
            if wtype == "parts" and subtype:
                token = PARTS_SUBTYPE_TO_TOKEN.get(subtype)
            else:
                token = INDICATOR_TYPE_TO_TOKEN.get(wtype)
            if token:
                wa.add_role(token, confidence)
        if w_lower in CONTAINER_VERB_INDICATORS:
            wa.add_role(CON_I, "medium")
        if w_lower in HIDDEN_VERB_INDICATORS:
            wa.add_role(HID_I, "medium")
        if w_lower in HALF_INDICATORS:
            wa.add_role(POS_I_HALF, "medium")
        if w_lower in ALTERNATE_INDICATORS:
            wa.add_role(POS_I_ALTERNATE, "medium")
        if w_lower in TRIM_FIRST_INDICATORS:
            wa.add_role(POS_I_TRIM_FIRST, "medium")
        if w_lower in FIRST_LETTER_INDICATORS:
            wa.add_role(POS_I_FIRST, "medium")
        if w_lower in BUILTIN_REVERSAL_INDICATORS:
            wa.add_role(REV_I, "medium")

        if w_lower.endswith("'s") and len(w_lower) > 2:
            wa.add_role(ABR_F, "S")
            _add_possessive_source_roles(wa, w_lower, answer_upper, db)

        _add_derived_gerund_roles(wa, w_lower, answer_upper, answer_len, db)

        # --- Check synonym DB (only synonyms that could contribute) ---
        # Synonyms that are substrings of the answer (useful for charade/container)
        for s in BUILTIN_SYNONYMS.get(w_lower, ()):
            if _answer_piece_relevant(s, answer_upper):
                wa.add_role(SYN_F, _clean_letters(s))
        syns_in_answer = db.get_synonyms_substring_of(w_lower, answer_upper)
        for s in syns_in_answer[:10]:
            wa.add_role(SYN_F, s)

        # Synonyms whose REVERSE is a substring of the answer (for reversal_charade)
        answer_rev = answer_upper[::-1]
        for s in db.get_synonyms(w_lower, max_len=answer_len):
            if isinstance(s, str) and len(s) >= 2 and s[::-1] in answer_upper and s not in syns_in_answer:
                wa.add_role(SYN_F, s)

        # Also check for synonyms up to answer length (for full-word synonyms)
        syns_full = db.get_synonyms_of_length(w_lower, answer_len)
        for s in syns_full[:5]:
            wa.add_role(SYN_F, s)

        # Broader synonym check — any synonym that's short enough to be a piece
        # Keep ALL short synonyms (≤4 chars) as they're crucial for charades;
        # cap longer ones at 20 to limit combinatorial explosion
        syns_short = db.get_synonyms(w_lower, max_len=answer_len)
        count_long = 0
        for s in syns_short:
            if len(s) <= 4:
                wa.add_role(SYN_F, s)
            elif count_long < 20:
                wa.add_role(SYN_F, s)
                count_long += 1
            elif _can_be_container_shell(s, answer_upper):
                wa.add_role(SYN_F, s)

        delete_count = 0
        for s in db.get_synonyms(w_lower, max_len=answer_len + 3):
            if not isinstance(s, str):
                continue
            if _can_delete_to_answer_segment(s, answer_upper):
                wa.add_role(SYN_F, s)
                delete_count += 1
                if delete_count >= 10:
                    break

        # --- Check homophone DB ---
        homophones = db.get_homophones(w_lower)
        for h in homophones:
            wa.add_role(HOM_F, h)
        for h in BUILTIN_HOMOPHONES.get(w_lower, ()):
            wa.add_role(HOM_F, h)
        # --- Check if word could be link word ---
        if db.is_link_word(w_lower) or w_lower in BUILTIN_LINK_WORDS:
            wa.add_role(LNK)
        if w_lower in BUILTIN_SURFACE_WORDS:
            wa.add_role("SURFACE")

        # --- Raw: word's own letters could contribute directly ---
        if w_alpha:
            wa.add_role(RAW, w_alpha)

        # --- Anagram fodder: raw letters available for anagramming ---
        if len(w_alpha) >= 2:
            wa.add_role(ANA_F, w_alpha)

        # --- Hidden fodder: word participates in spanning the answer ---
        wa.add_role(HID_F, w_alpha)

        # --- Positional fodder: letters can be extracted ---
        if len(w_alpha) >= 2:
            wa.add_role(POS_F, w_alpha)

        # --- Deletion fodder: word contributes what gets removed ---
        if len(w_alpha) <= 3:
            wa.add_role(DEL_F, w_alpha)

        analyses.append(wa)

    return analyses


CONTAINER_VERB_INDICATORS = {
    "grab",
    "grabs",
    "grabbed",
    "grabbing",
    "breaching",
}


HIDDEN_VERB_INDICATORS = {
    "restricts",
    "restricted",
    "restricting",
}


BUILTIN_REVERSAL_INDICATORS = {
    "retreat",
}


TRIM_FIRST_INDICATORS = {
    "barely",
    "initially overlooked",
    "leader dismissed",
    "leader is dismissed",
    "heading off",
    "not at first",
    "topped",
}


FIRST_LETTER_INDICATORS = {
    "head",
    "heading",
}


HALF_INDICATORS = {
    "half",
    "half of",
    "not half",
}


ALTERNATE_INDICATORS = {
    "on and off",
}


PHRASE_CONTAINER_INDICATORS = {
    "to have",
}


PHRASE_REVERSAL_INDICATORS = {
    "read up and down",
}


PHRASE_ANAGRAM_INDICATORS = {
    "looking up",
}


BUILTIN_ABBREVIATIONS = {
    "memory": ("RAM",),
    "random access memory": ("RAM",),
    "troops": ("RE",),
}


BUILTIN_LINK_WORDS = {
    "seeing",
    "using",
}


BUILTIN_SYNONYMS = {
    "dive": ("DUCK",),
    "area of land": ("HEATH",),
    "open area of land": ("HEATH",),
    "insurance policy": ("COVER",),
    "irish wit": ("WILDE",),
    "tactless remarks": ("CLANGERS",),
    "hidden": ("COVERED",),
    "arousing": ("INCITING",),
    "town in cornwall": ("STIVES",),
    "no problem": ("EASE",),
    "caesar perhaps": ("ROMAN",),
}


BUILTIN_SURFACE_WORDS = {
    "attended",
    "carefully",
    "gets",
    "normally",
    "requires",
    "some",
    "supporting",
}


BUILTIN_SURFACE_PHRASES = {
    "on the contrary",
    "some say",
}


BUILTIN_HOMOPHONES = {
    "you": ("U",),
}


def analyze_phrases(words, answer, db):
    """Analyze multi-word phrases as well as individual words.

    Some clue words work as 2-word phrases: 'former lover' -> EX,
    'for example' -> EG, 'Royal Engineers' -> RE.

    Returns:
        list of WordAnalysis for individual words,
        plus dict of (i, j) -> WordAnalysis for phrase spanning words[i:j]
    """
    single = analyze_words(words, answer, db)

    phrases = {}
    # Check 2-word, 3-word, and 4-word phrases
    for span in (2, 3, 4):
        for i in range(len(words) - span + 1):
            phrase = " ".join(words[i:i + span])
            wa = WordAnalysis(phrase)
            p_lower = " ".join(clean_word(word) for word in words[i:i + span])

            # Check abbreviation
            abbrs = db.get_abbreviations(p_lower)
            for a in abbrs:
                wa.add_role(ABR_F, a)
            for a in BUILTIN_ABBREVIATIONS.get(p_lower, ()):
                wa.add_role(ABR_F, a)

            # Check synonym
            answer_upper = answer.upper().replace(" ", "").replace("-", "")
            syns = db.get_synonyms_substring_of(p_lower, answer_upper)
            for s in syns[:10]:
                wa.add_role(SYN_F, s)
            syns_short = db.get_synonyms(p_lower, max_len=len(answer_upper))
            for s in syns_short[:10]:
                wa.add_role(SYN_F, s)
            for s in BUILTIN_SYNONYMS.get(p_lower, ()):
                if _answer_piece_relevant(s, answer_upper):
                    wa.add_role(SYN_F, _clean_letters(s))
            for s in syns_short[10:]:
                if _can_be_container_shell(s, answer_upper):
                    wa.add_role(SYN_F, s)
                    break
            delete_count = 0
            for s in db.get_synonyms(p_lower, max_len=len(answer_upper) + 3):
                if not isinstance(s, str):
                    continue
                if _can_delete_to_answer_segment(s, answer_upper):
                    wa.add_role(SYN_F, s)
                    delete_count += 1
                    if delete_count >= 10:
                        break

            # Check indicator DB for multi-word indicators
            # e.g. "mixed up" → ANA_I, "set up" → REV_I, "we hear" → HOM_I
            ind_types = db.get_indicator_types(p_lower)
            for wtype, subtype, confidence in ind_types:
                if wtype == "parts" and subtype:
                    token = PARTS_SUBTYPE_TO_TOKEN.get(subtype)
                else:
                    token = INDICATOR_TYPE_TO_TOKEN.get(wtype)
                if token:
                    wa.add_role(token, confidence)
            if p_lower in TRIM_FIRST_INDICATORS:
                wa.add_role(POS_I_TRIM_FIRST, "medium")
            if p_lower in FIRST_LETTER_INDICATORS:
                wa.add_role(POS_I_FIRST, "medium")
            if p_lower in HALF_INDICATORS:
                wa.add_role(POS_I_HALF, "medium")
            if p_lower in ALTERNATE_INDICATORS:
                wa.add_role(POS_I_ALTERNATE, "medium")
            if p_lower in PHRASE_CONTAINER_INDICATORS:
                wa.add_role(CON_I, "medium")
            if p_lower in PHRASE_REVERSAL_INDICATORS:
                wa.add_role(REV_I, "medium")
            if p_lower in PHRASE_ANAGRAM_INDICATORS:
                wa.add_role(ANA_I, "medium")
            if p_lower in BUILTIN_SURFACE_PHRASES:
                wa.add_role("SURFACE")

            if wa.roles:
                phrases[(i, i + span)] = wa

    return single, phrases


def _can_be_container_shell(value, answer):
    value = "".join(c for c in (value or "").upper() if c.isalpha())
    answer = "".join(c for c in (answer or "").upper() if c.isalpha())
    if len(value) < 2 or len(value) >= len(answer):
        return False
    for split in range(1, len(value)):
        prefix = value[:split]
        suffix = value[split:]
        start = 0
        while True:
            pos = answer.find(prefix, start)
            if pos < 0:
                break
            min_end = pos + len(value) + 1
            for end in range(min_end, len(answer) + 1):
                if answer[pos:end].endswith(suffix):
                    return True
            start = pos + 1
    return False


def _add_possessive_source_roles(analysis, word, answer_upper, db):
    base = word[:-2]
    if not base:
        return
    for value in db.get_abbreviations(base):
        _add_if_answer_relevant(analysis, ABR_F, "%sS" % value, answer_upper)
    for value in db.get_synonyms(base, max_len=max(1, len(answer_upper) - 1)):
        if isinstance(value, str):
            _add_if_answer_relevant(
                analysis, SYN_F, "%sS" % value, answer_upper)


def _add_derived_gerund_roles(analysis, word, answer_upper, answer_len, db):
    if not word.endswith("ing") or len(word) <= 4:
        return
    for stem in _gerund_base_forms(word):
        values = list(db.get_synonyms(stem, max_len=max(1, answer_len - 3)))
        values.extend(BUILTIN_SYNONYMS.get(stem, ()))
        for value in values:
            if isinstance(value, str):
                _add_if_answer_relevant(
                    analysis, SYN_F, _to_gerund(value), answer_upper)


def _gerund_base_forms(word):
    stem = word[:-3]
    forms = {stem}
    if stem:
        forms.add("%se" % stem)
    if len(stem) >= 2 and stem[-1] == stem[-2]:
        forms.add(stem[:-1])
    return tuple(forms)


def _to_gerund(value):
    value = _clean_letters(value)
    if not value:
        return ""
    if value.endswith("IE"):
        return "%sYING" % value[:-2]
    if value.endswith("E") and not value.endswith("EE"):
        return "%sING" % value[:-1]
    return "%sING" % value


def _add_if_answer_relevant(analysis, token, value, answer_upper):
    value = _clean_letters(value)
    if not value:
        return
    if _answer_piece_relevant(value, answer_upper):
        analysis.add_role(token, value)


def _answer_piece_relevant(value, answer_upper):
    value = _clean_letters(value)
    return (
        value in answer_upper
        or _can_be_container_shell(value, answer_upper)
        or _can_delete_to_answer_segment(value, answer_upper)
        or _can_select_to_answer_segment(value, answer_upper)
    )


def _clean_letters(value):
    return "".join(c for c in (value or "").upper() if c.isalpha())


def _can_delete_to_answer_segment(value, answer):
    value = "".join(c for c in (value or "").upper() if c.isalpha())
    answer = "".join(c for c in (answer or "").upper() if c.isalpha())
    if len(value) <= 3 or not answer:
        return False
    for target_len in range(min(len(value) - 1, len(answer)), 2, -1):
        for target_start in range(0, len(answer) - target_len + 1):
            target = answer[target_start:target_start + target_len]
            if _can_delete_contiguous_to(value, target):
                return True
    return False


def _can_select_to_answer_segment(value, answer):
    value = "".join(c for c in (value or "").upper() if c.isalpha())
    answer = "".join(c for c in (answer or "").upper() if c.isalpha())
    if len(value) < 2 or not answer:
        return False
    selections = {
        value[: max(1, len(value) // 2)],
        value[len(value) // 2:],
        value[::2],
        value[1::2],
        value[0],
        value[-1],
    }
    return any(selection and selection in answer for selection in selections)


def _can_delete_contiguous_to(value, target):
    if len(value) <= len(target) or len(value) - len(target) > 3:
        return False
    for start in range(len(value)):
        for end in range(start + 1, len(value) + 1):
            if value[:start] + value[end:] == target:
                return True
    return False
