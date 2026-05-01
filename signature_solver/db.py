"""Database lookups for the signature solver.

Wraps the existing ClueEnricher but provides a simpler interface
focused on what the signature solver needs: for each word, what
roles could it play?
"""

import os
import re
import sqlite3
from functools import lru_cache


def _normalize_key(text):
    """Strip all punctuation from a lookup key, keeping only alphanumeric + spaces."""
    return re.sub(r"[^a-z0-9 ]", "", text.lower().strip()).strip()


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

        # --- Wordlist: set of known real words (uppercase) ---
        # Used by confidence scoring to distinguish real words from nonsense.
        # Built from reference DB tables first, then enriched from clues_master.
        self.wordlist = set()

        # --- Indicators: word -> list of (wordplay_type, subtype, confidence) ---
        self.indicators = {}
        for word, wtype, subtype, confidence in conn.execute(
            "SELECT word, wordplay_type, subtype, confidence FROM indicators"
        ):
            w = _normalize_key(word)
            if w not in self.indicators:
                self.indicators[w] = []
            self.indicators[w].append((wtype, subtype, confidence))

        # --- Abbreviations (wordplay table): word -> list of substitutions ---
        # category='dbe' rows are definition-by-example markers (say, perhaps,
        # for example, etc.) — they don't produce letters and must not be
        # loaded as abbreviations. They will be consumed by a future DBE solver.
        self.abbreviations = {}
        for indicator, substitution in conn.execute(
            "SELECT indicator, substitution FROM wordplay "
            "WHERE category IS NULL OR category != 'dbe'"
        ):
            w = _normalize_key(indicator)
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
            w = _normalize_key(word)
            if w not in self.synonyms:
                self.synonyms[w] = []
            self.synonyms[w].append(synonym.strip().upper())

        # --- Definition-answer pairs (augmented): merge into synonyms ---
        n_da_new = 0
        for definition, answer in conn.execute(
            "SELECT definition, answer FROM definition_answers_augmented"
            " WHERE definition IS NOT NULL AND answer IS NOT NULL"
        ):
            w = _normalize_key(definition)
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
            w = _normalize_key(word)
            if w not in self.homophones:
                self.homophones[w] = []
            self.homophones[w].append(homophone.strip().upper())

        # --- Build wordlist from reference DB ---
        # All synonym words and values
        for w, syns in self.synonyms.items():
            if len(w) >= 2:
                self.wordlist.add(w.upper())
            for s in syns:
                if len(s) >= 2:
                    self.wordlist.add(s)
        # All abbreviation words (not the short values — A, N, R aren't "words")
        for w in self.abbreviations:
            if len(w) >= 2:
                self.wordlist.add(w.upper())
        # All homophone words and values
        for w, homos in self.homophones.items():
            if len(w) >= 2:
                self.wordlist.add(w.upper())
            for h in homos:
                if len(h) >= 2:
                    self.wordlist.add(h)
        # All indicator words
        for w in self.indicators:
            if len(w) >= 2:
                self.wordlist.add(w.upper())

        conn.close()

        # --- Enrich wordlist from clues_master (clue texts + answers) ---
        clues_db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "clues_master.db"
        )
        if os.path.exists(clues_db_path):
            clues_conn = sqlite3.connect(clues_db_path, timeout=30)
            # All answers
            for (answer,) in clues_conn.execute(
                "SELECT DISTINCT answer FROM clues WHERE answer IS NOT NULL"
            ):
                a = answer.strip().upper()
                if len(a) >= 2:
                    self.wordlist.add(a)
            # All words from clue texts
            for (clue_text,) in clues_conn.execute(
                "SELECT clue_text FROM clues WHERE clue_text IS NOT NULL"
            ):
                for word in re.findall(r"[A-Za-z]+", clue_text):
                    w = word.upper()
                    if len(w) >= 2:
                        self.wordlist.add(w)
            clues_conn.close()

        n_ind = sum(len(v) for v in self.indicators.values())
        n_abbr = sum(len(v) for v in self.abbreviations.values())
        n_syn = sum(len(v) for v in self.synonyms.values())
        n_hom = sum(len(v) for v in self.homophones.values())
        print(f"RefDB loaded: {n_ind} indicators, {n_abbr} abbreviations, "
              f"{n_syn} synonyms ({n_da_new} from def_answers), {n_hom} homophones, "
              f"{len(self.wordlist):,} wordlist entries")

    @staticmethod
    def _word_variants(word):
        """Generate normalized forms: strip punctuation, possessives and simple plurals."""
        w = _normalize_key(word)
        variants = [w]
        # Strip possessive 's (now just trailing 's' after normalize stripped the apostrophe)
        if w.endswith("s") and len(w) >= 3:
            # Check original for possessive pattern
            orig = word.lower().strip()
            if orig.endswith("'s") or orig.endswith("\u2019s") or orig.endswith("s'"):
                variants.append(w[:-1])
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

    def is_extra_synonym(self, word, value):
        """Default: nothing is an extra (no overlay applied). The
        _SynonymOverlayRefDB overrides this to return True for
        DBE-injected pairs so the matcher can tag them in word_roles."""
        return False

    def with_extra_synonyms(self, extras):
        """Return a thin overlay over this RefDB that adds extra synonyms.

        extras: dict mapping clue-word (lowercase, normalised) -> list of
        uppercase candidate synonyms.

        The overlay forwards every method/attribute to the underlying
        RefDB except synonym lookups, which are unioned with the extras
        for the matching word. The original RefDB is never mutated.

        Pattern is intentionally narrow: a no-op when extras is empty,
        and the overlay can be passed anywhere a RefDB is expected.
        """
        if not extras:
            return self
        return _SynonymOverlayRefDB(self, extras)

    def with_extra_indicators(self, extras):
        """Return an overlay that adds extra indicator entries.

        extras: dict mapping clue-word (lowercase, normalised) -> list of
        (wordplay_type, subtype, confidence) tuples — same shape as
        get_indicator_types returns. Confidence is typically 'medium'
        for solver-injected guesses.

        Composes cleanly with with_extra_synonyms — call either order:
            db.with_extra_synonyms(s).with_extra_indicators(i)
            db.with_extra_indicators(i).with_extra_synonyms(s)
        Both produce a stack of overlays that delegate via __getattr__.
        """
        if not extras:
            return self
        return _IndicatorOverlayRefDB(self, extras)

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

    def is_real_word(self, word):
        """Check if word appears in our wordlist (known English words).

        Used by confidence scoring to distinguish plausible synonyms
        (real words not in our synonym DB) from nonsense (IFFLING, etc).
        """
        w = word.upper().strip()
        if not w or len(w) < 2:
            return False
        return w in self.wordlist


class _SynonymOverlayRefDB:
    """Thin wrapper that augments a base RefDB with extra synonyms for
    specific words. Created via RefDB.with_extra_synonyms(extras).

    All other methods and attributes pass through unchanged. The base
    RefDB is never mutated — extras live only on this wrapper instance,
    so concurrent solves can each have their own overlay.
    """

    __slots__ = ('_base', '_extras')

    def __init__(self, base, extras):
        self._base = base
        # Normalise keys + values once at construction; uppercase candidates.
        self._extras = {}
        for word, vals in extras.items():
            key = _normalize_key(word)
            cleaned = []
            seen = set()
            for v in vals:
                vu = v.upper().strip() if isinstance(v, str) else ''
                if vu and vu not in seen:
                    cleaned.append(vu)
                    seen.add(vu)
            if cleaned:
                self._extras[key] = cleaned

    def __getattr__(self, name):
        # Delegate any attribute we don't override to the base RefDB.
        return getattr(self._base, name)

    def _extra_for(self, word):
        for v in self._base._word_variants(word):
            if v in self._extras:
                return self._extras[v]
        return []

    def is_extra_synonym(self, word, value):
        """Did `value` come from the extras (DBE injection) for `word`?

        Helps the matcher tag DBE-sourced pieces in word_roles so the
        explanation can attribute them honestly.
        """
        if not value:
            return False
        v_upper = value.upper().strip()
        return v_upper in self._extra_for(word)

    # --- Augmented synonym methods ---

    def get_synonyms(self, word, max_len=None):
        results = self._base.get_synonyms(word, max_len=max_len)
        for s in self._extra_for(word):
            if s in results:
                continue
            if max_len is None or len(s) <= max_len:
                results.append(s)
        return results

    def get_synonyms_of_length(self, word, length):
        return [s for s in self.get_synonyms(word) if len(s) == length]

    def get_synonyms_substring_of(self, word, answer):
        answer_upper = answer.upper()
        return [s for s in self.get_synonyms(word)
                if s in answer_upper and s != answer_upper]

    # --- Composition: stack overlays without losing this one ---

    def with_extra_synonyms(self, extras):
        if not extras:
            return self
        # Merge into THIS overlay's extras rather than wrap again.
        merged = {k: list(v) for k, v in self._extras.items()}
        for word, vals in extras.items():
            from signature_solver.db import _normalize_key as _nk
            key = _nk(word)
            seen = set(merged.get(key, []))
            for v in vals:
                vu = v.upper().strip() if isinstance(v, str) else ''
                if vu and vu not in seen:
                    merged.setdefault(key, []).append(vu)
                    seen.add(vu)
        return _SynonymOverlayRefDB(self._base, merged)

    def with_extra_indicators(self, extras):
        if not extras:
            return self
        return _IndicatorOverlayRefDB(self, extras)


class _IndicatorOverlayRefDB:
    """Overlay that augments a base RefDB with extra indicator entries
    for specific words. Created via RefDB.with_extra_indicators(extras).

    extras: dict mapping word -> list of (wordplay_type, subtype, confidence).
    Same shape as RefDB.get_indicator_types returns, so unioning is direct.
    """

    __slots__ = ('_base', '_extras')

    def __init__(self, base, extras):
        self._base = base
        self._extras = {}
        for word, entries in extras.items():
            key = _normalize_key(word)
            cleaned = []
            for e in entries:
                if not isinstance(e, (tuple, list)) or len(e) < 2:
                    continue
                wtype = e[0]
                subtype = e[1] if len(e) >= 2 else None
                conf = e[2] if len(e) >= 3 else 'medium'
                cleaned.append((wtype, subtype, conf))
            if cleaned:
                self._extras[key] = cleaned

    def __getattr__(self, name):
        return getattr(self._base, name)

    def get_indicator_types(self, word):
        results = list(self._base.get_indicator_types(word))
        # Use the same word-variants helper as the base for consistency.
        # Walk up to the underlying RefDB to find the helper if needed.
        base = self._base
        while not hasattr(base, '_word_variants'):
            base = base._base
        for v in base._word_variants(word):
            if v in self._extras:
                for entry in self._extras[v]:
                    if entry not in results:
                        results.append(entry)
        return results

    # --- Composition ---

    def with_extra_synonyms(self, extras):
        if not extras:
            return self
        return _SynonymOverlayRefDB(self, extras)

    def with_extra_indicators(self, extras):
        if not extras:
            return self
        # Merge into THIS overlay's extras rather than wrap again.
        merged = {k: list(v) for k, v in self._extras.items()}
        # Walk up to the base RefDB for normalisation
        base = self._base
        while not hasattr(base, '_word_variants'):
            base = base._base
        from signature_solver.db import _normalize_key as _nk
        for word, entries in extras.items():
            key = _nk(word)
            existing = list(merged.get(key, []))
            for e in entries:
                if not isinstance(e, (tuple, list)) or len(e) < 2:
                    continue
                wtype = e[0]
                subtype = e[1] if len(e) >= 2 else None
                conf = e[2] if len(e) >= 3 else 'medium'
                tup = (wtype, subtype, conf)
                if tup not in existing:
                    existing.append(tup)
            merged[key] = existing
        return _IndicatorOverlayRefDB(self._base, merged)
