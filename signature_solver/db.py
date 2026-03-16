"""Database lookups for the signature solver.

Wraps the existing ClueEnricher but provides a simpler interface
focused on what the signature solver needs: for each word, what
roles could it play?
"""

import os
import re
import sqlite3
from functools import lru_cache


class RefDB:
    """Reference database lookups — loads all tables into memory."""

    def __init__(self, db_path=None):
        if db_path is None:
            db_path = os.path.join(
                os.path.dirname(os.path.dirname(__file__)),
                "data", "cryptic_new.db"
            )
        self._load_all(db_path)

    def _load_all(self, db_path):
        conn = sqlite3.connect(db_path, timeout=30)

        # --- Indicators: word -> list of (wordplay_type, subtype, confidence) ---
        self.indicators = {}
        for word, wtype, subtype, confidence in conn.execute(
            "SELECT word, wordplay_type, subtype, confidence FROM indicators"
        ):
            w = word.lower().strip()
            if w not in self.indicators:
                self.indicators[w] = []
            self.indicators[w].append((wtype, subtype, confidence))

        # --- Abbreviations (wordplay table): word -> list of substitutions ---
        self.abbreviations = {}
        for indicator, substitution in conn.execute(
            "SELECT indicator, substitution FROM wordplay"
        ):
            w = indicator.lower().strip()
            if w not in self.abbreviations:
                self.abbreviations[w] = []
            sub = substitution.strip().upper()
            if sub and sub not in self.abbreviations[w]:
                self.abbreviations[w].append(sub)

        # --- Synonyms: word -> list of synonyms (uppercase) ---
        self.synonyms = {}
        for word, synonym in conn.execute(
            "SELECT word, synonym FROM synonyms_pairs"
        ):
            w = word.lower().strip()
            if w not in self.synonyms:
                self.synonyms[w] = []
            self.synonyms[w].append(synonym.strip().upper())

        # --- Definition-answer pairs (augmented): merge into synonyms ---
        n_da_new = 0
        for definition, answer in conn.execute(
            "SELECT definition, answer FROM definition_answers_augmented"
            " WHERE definition IS NOT NULL AND answer IS NOT NULL"
        ):
            w = definition.lower().strip()
            val = answer.strip().upper()
            if not w or not val:
                continue
            if w not in self.synonyms:
                self.synonyms[w] = []
            if val not in self.synonyms[w]:
                self.synonyms[w].append(val)
                n_da_new += 1

        # --- Homophones: word -> list of homophones ---
        self.homophones = {}
        for word, homophone in conn.execute(
            "SELECT word, homophone FROM homophones"
        ):
            w = word.lower().strip()
            if w not in self.homophones:
                self.homophones[w] = []
            self.homophones[w].append(homophone.strip().upper())

        conn.close()

        n_ind = sum(len(v) for v in self.indicators.values())
        n_abbr = sum(len(v) for v in self.abbreviations.values())
        n_syn = sum(len(v) for v in self.synonyms.values())
        n_hom = sum(len(v) for v in self.homophones.values())
        print(f"RefDB loaded: {n_ind} indicators, {n_abbr} abbreviations, "
              f"{n_syn} synonyms ({n_da_new} from def_answers), {n_hom} homophones")

    @staticmethod
    def _word_variants(word):
        """Generate normalized forms: strip punctuation, possessives and simple plurals."""
        w = word.lower().strip(".,;:!?\"'()-").strip()
        variants = [w]
        # Strip possessive 's
        if w.endswith("'s"):
            variants.append(w[:-2])
        # Strip plural possessive
        if w.endswith("s'"):
            variants.append(w[:-2])
        # Strip simple plural (but not boss, less, etc.)
        if len(w) >= 4 and w.endswith("s") and not w.endswith("ss"):
            variants.append(w[:-1])
        return variants

    def get_indicator_types(self, word):
        """Return set of wordplay types this word could indicate.

        Returns: list of (wordplay_type, subtype, confidence) tuples
        """
        results = []
        for v in self._word_variants(word):
            if v in self.indicators:
                results.extend(self.indicators[v])
        return results

    def get_abbreviations(self, word):
        """Return list of possible abbreviation expansions (uppercase).

        E.g. 'north' -> ['N'], 'hundred' -> ['C']
        """
        results = []
        for v in self._word_variants(word):
            if v in self.abbreviations:
                for a in self.abbreviations[v]:
                    if a not in results:
                        results.append(a)
        return results

    def get_synonyms(self, word, max_len=None):
        """Return list of synonyms (uppercase).

        Args:
            word: the clue word
            max_len: if set, only return synonyms up to this length
        """
        results = []
        for v in self._word_variants(word):
            if v in self.synonyms:
                for s in self.synonyms[v]:
                    if s not in results:
                        if max_len is None or len(s) <= max_len:
                            results.append(s)
        return results

    def get_synonyms_of_length(self, word, length):
        """Return synonyms that are exactly `length` characters."""
        return [s for s in self.get_synonyms(word) if len(s) == length]

    def get_synonyms_substring_of(self, word, answer):
        """Return synonyms that are substrings of the answer."""
        answer_upper = answer.upper()
        return [s for s in self.get_synonyms(word) if s in answer_upper and s != answer_upper]

    def get_homophones(self, word):
        """Return list of homophones (uppercase)."""
        results = []
        for v in self._word_variants(word):
            if v in self.homophones:
                for h in self.homophones[v]:
                    if h not in results:
                        results.append(h)
        return results

    def is_definition_of(self, phrase, answer):
        """Check if phrase is a synonym/definition of the answer.

        Checks both directions: phrase→answer and answer→phrase.
        Uses synonyms_pairs + definition_answers_augmented (already merged).
        """
        answer_clean = answer.upper().replace(" ", "").replace("-", "")
        phrase_clean = phrase.lower().strip(".,;:!?\"'()-").strip()

        # Check phrase → answer
        for s in self.get_synonyms(phrase_clean):
            if s.replace(" ", "").replace("-", "") == answer_clean:
                return True

        # Check answer → phrase (reverse direction)
        answer_lower = answer.lower().replace("-", " ")
        for s in self.get_synonyms(answer_lower):
            if s.replace(" ", "").replace("-", "").upper() == phrase_clean.upper().replace(" ", ""):
                return True

        return False

    def is_link_word(self, word):
        """Check if word is a common link word."""
        from .tokens import LINK_WORDS
        return word.lower().strip() in LINK_WORDS
