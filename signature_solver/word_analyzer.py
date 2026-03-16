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

        # --- Check indicator DB ---
        ind_types = db.get_indicator_types(w_lower)
        for wtype, subtype, confidence in ind_types:
            if wtype == "parts" and subtype:
                token = PARTS_SUBTYPE_TO_TOKEN.get(subtype)
            else:
                token = INDICATOR_TYPE_TO_TOKEN.get(wtype)
            if token:
                wa.add_role(token, confidence)

        # --- Check synonym DB (only synonyms that could contribute) ---
        # Synonyms that are substrings of the answer (useful for charade/container)
        syns_in_answer = db.get_synonyms_substring_of(w_lower, answer_upper)
        for s in syns_in_answer[:10]:
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

        # --- Check homophone DB ---
        homophones = db.get_homophones(w_lower)
        for h in homophones:
            wa.add_role(HOM_F, h)

        # --- Check if word could be link word ---
        if db.is_link_word(w_lower):
            wa.add_role(LNK)

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
            p_lower = phrase.lower()

            # Check abbreviation
            abbrs = db.get_abbreviations(p_lower)
            for a in abbrs:
                wa.add_role(ABR_F, a)

            # Check synonym
            answer_upper = answer.upper().replace(" ", "").replace("-", "")
            syns = db.get_synonyms_substring_of(p_lower, answer_upper)
            for s in syns[:10]:
                wa.add_role(SYN_F, s)
            syns_short = db.get_synonyms(p_lower, max_len=len(answer_upper))
            for s in syns_short[:10]:
                wa.add_role(SYN_F, s)

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

            if wa.roles:
                phrases[(i, i + span)] = wa

    return single, phrases
