"""LiveDB — a drop-in replacement for signature_solver.db.RefDB that answers every
lookup with a LIVE INDEXED query instead of loading the whole reference DB (~1.7M
synonyms, ~1GB, 30-60s) into memory.

Why: the preload is what makes the server start slowly, hold a gigabyte in RAM (the
source of the long-session crash), and need a costly rebuild before a freshly-added
entry is seen. The reference tables are already indexed, and a single-word lookup is
~0.1ms (see core/_prove_livedb.py), so querying on demand is cheap — and an add is seen
immediately because the query reads the row that was just written.

It mirrors RefDB's interface and normalization exactly (RefDB._word_variants /
_normalize_key), so it can be substituted wherever a RefDB is expected. Per-word results
are memoised for the life of the instance; the cache is dropped when the page reloads.

NOTE the one index subtlety: definition_answers_augmented is indexed on
LOWER(definition), so its lookup MUST be `WHERE LOWER(definition)=?` (matching the
functional index) or it full-scans 643k rows (57ms vs 0.5ms).
"""

import os
import sqlite3

from signature_solver.db import RefDB, _normalize_key


def _default_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "cryptic_new.db")


def _pron_key(phonemes):
    """Normalised pronunciation key: ARPABET phonemes with the stress digits removed,
    so 'L IH1 S T' and 'L IH0 S T' compare equal. Exact (rhotic) match — the British
    non-rhotic adjustment (dropping a post-vowel R for fort/fought) is a later step."""
    return " ".join(ph.rstrip("012") for ph in (phonemes or "").split())


class LiveDB:
    """RefDB's lookups, on demand, no preload."""

    def __init__(self, db_path=None):
        self.path = db_path or _default_path()
        self._conn = sqlite3.connect(self.path, check_same_thread=False, timeout=30)
        self._cs, self._csl, self._ca, self._ci, self._ch = {}, {}, {}, {}, {}
        self._cp = {}                       # pronunciation cache (CMUdict phonemes)

    # normalization shared with RefDB so keys match the loaded behaviour exactly.
    def _word_variants(self, word):
        return RefDB._word_variants(word)

    # --- synonyms (synonyms_pairs + the merged definition_answers_augmented) ---------
    def get_synonyms(self, word, max_len=None):
        if word in self._cs:
            base = self._cs[word]
        else:
            out, seen = [], set()
            for v in self._word_variants(word):        # v is already a normalized key
                for (s,) in self._conn.execute(
                        "SELECT synonym FROM synonyms_pairs WHERE norm_word=?", (v,)):
                    su = (s or "").strip().upper()
                    if su and su not in seen:
                        seen.add(su); out.append(su)
                for (a,) in self._conn.execute(
                        "SELECT answer FROM definition_answers_augmented "
                        "WHERE norm_def=?", (v,)):       # normalized-key index (faithful to RefDB)
                    au = (a or "").strip().upper()
                    if au and au not in seen:
                        seen.add(au); out.append(au)
            self._cs[word] = out
            base = out
        if max_len is None:
            return base
        return [s for s in base if len(s) <= max_len]

    def get_synonyms_of_length(self, word, length):
        return [s for s in self.get_synonyms(word) if len(s) == length]

    def get_synonyms_substring_of(self, word, answer):
        au = answer.upper()
        return [s for s in self.get_synonyms(word) if s in au and s != au]

    # --- abbreviations (wordplay table, excluding DBE markers) -----------------------
    def get_abbreviations(self, word):
        if word in self._ca:
            return self._ca[word]
        out, seen = [], set()
        for v in self._word_variants(word):
            for (sub,) in self._conn.execute(
                    "SELECT substitution FROM wordplay WHERE norm_ind=? AND "
                    "(category IS NULL OR category != 'dbe')", (v,)):
                su = (sub or "").strip().upper()
                if su and su not in seen:
                    seen.add(su); out.append(su)
        self._ca[word] = out
        return out

    # --- indicators ------------------------------------------------------------------
    def get_indicator_types(self, word):
        if word in self._ci:
            return self._ci[word]
        out = []
        for v in self._word_variants(word):
            for row in self._conn.execute(
                    "SELECT wordplay_type, subtype, confidence FROM indicators WHERE norm_word=?",
                    (v,)):
                out.append(tuple(row))
        self._ci[word] = out
        return out

    # --- homophones ------------------------------------------------------------------
    def get_homophones(self, word):
        if word in self._ch:
            return self._ch[word]
        out, seen = [], set()
        for v in self._word_variants(word):
            for (h,) in self._conn.execute(
                    "SELECT homophone FROM homophones WHERE norm_word=?", (v,)):
                hu = (h or "").strip().upper()
                if hu and hu not in seen:
                    seen.add(hu); out.append(hu)
        self._ch[word] = out
        return out

    # --- pronunciations (CMUdict phonemes) -------------------------------------------
    def get_pronunciation(self, word):
        """List of phoneme strings (raw ARPABET) for the word, via the live table."""
        if word in self._cp:
            return self._cp[word]
        out, seen = [], set()
        for v in self._word_variants(word):
            for (p,) in self._conn.execute(
                    "SELECT phonemes FROM pronunciations WHERE norm_word=?", (v,)):
                p = (p or "").strip()
                if p and p not in seen:
                    seen.add(p); out.append(p)
        self._cp[word] = out
        return out

    def sounds_alike(self, word, target):
        """True if `word` is pronounced the same as `target`. Judged primarily by the
        CMUdict pronunciations (broad coverage), with the curated homophones table as a
        supplement for pairs CMUdict lacks. Stress is ignored; the match is exact."""
        # curated table (the small hand-built supplement), in either direction
        tclean = "".join(c for c in (target or "").upper() if c.isalpha())
        if tclean and tclean in [h.replace(" ", "") for h in self.get_homophones(word)]:
            return True
        # pronunciation match
        kw = {_pron_key(p) for p in self.get_pronunciation(word)}
        if not kw:
            return False
        kt = {_pron_key(p) for p in self.get_pronunciation(target)}
        return bool(kw & kt)

    # --- definitions (both directions, via the live synonym lookup) ------------------
    def is_definition_of(self, phrase, answer):
        answer_clean = answer.upper().replace(" ", "").replace("-", "")
        phrase_clean = phrase.lower().strip(".,;:!?\"'()-").strip()
        for s in self.get_synonyms(phrase_clean):
            if s.replace(" ", "").replace("-", "") == answer_clean:
                return True
        answer_lower = answer.lower().replace("-", " ")
        for s in self.get_synonyms(answer_lower):
            if s.replace(" ", "").replace("-", "").upper() == phrase_clean.upper().replace(" ", ""):
                return True
        return False

    def is_link_word(self, word):
        from signature_solver.tokens import LINK_WORDS
        return word.lower().strip() in LINK_WORDS

    def is_real_word(self, word):
        # Confidence-only (does NOT affect pass/fail, which is reconstruction-based).
        # Permissive live stub; a precise wordlist could be a live EXISTS query later.
        return bool(word) and len(word.strip()) >= 2

    def is_extra_synonym(self, word, value):
        return False

    # --- DBE / indicator overlays: reuse RefDB's overlay classes (they delegate to
    #     this base, which now has _word_variants + get_synonyms) -----------------------
    def with_extra_synonyms(self, extras):
        if not extras:
            return self
        from signature_solver.db import _SynonymOverlayRefDB
        return _SynonymOverlayRefDB(self, extras)

    def with_extra_indicators(self, extras):
        if not extras:
            return self
        from signature_solver.db import _IndicatorOverlayRefDB
        return _IndicatorOverlayRefDB(self, extras)
