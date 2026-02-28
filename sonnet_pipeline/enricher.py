"""Enrich a clue with lookups from all reference tables.

For each word/phrase in a clue, looks up:
- synonyms_pairs: what could this word mean?
- indicators: does this word signal a wordplay type?
- wordplay: does this word abbreviate to a letter/string?
- homophones: what does this word sound like?
- definition_answers_augmented: does this phrase define the answer?

Returns a structured context block to include in model training prompts.
"""

import re
import sqlite3
from pathlib import Path

DB_PATH = Path(__file__).parent.parent / "data" / "cryptic_new.db"


def _is_container_outer(syn, answer_clean):
    """Check if answer = syn with a contiguous block inserted (container outer).

    E.g. HEARTY is a container outer for HENPARTY because
    HE + NP + ARTY = HENPARTY (NP inserted at position 2 of HEARTY).
    """
    gap = len(answer_clean) - len(syn)
    if gap < 1 or len(syn) < 3:
        return False
    for i in range(1, len(syn) + 1):
        if answer_clean[:i] == syn[:i] and answer_clean[i + gap:] == syn[i:]:
            return True
    return False


def _is_fractured_substring(syn, answer_clean):
    """Check if syn appears in answer with a single fracture (one insertion gap).

    E.g. UNCLEAN in UNCLEVANYA: UNCLE + AN with V inserted between.
    Catches container+charade cases that _is_container_outer misses
    (where the synonym doesn't span the full answer).
    Requires at least 2 chars on each side of the split.
    """
    if len(syn) < 4:
        return False
    for split_pos in range(2, len(syn) - 1):
        prefix = syn[:split_pos]
        suffix = syn[split_pos:]
        idx = answer_clean.find(prefix)
        if idx >= 0 and answer_clean.find(suffix, idx + len(prefix) + 1) >= 0:
            return True
    return False


def _word_variants(word):
    """Generate normalized variants: strip possessives and simple plurals."""
    forms = [word]
    # she's → she, father's → father
    if word.endswith("'s"):
        forms.append(word[:-2])
    # supporters' → supporter
    elif word.endswith("s'"):
        forms.append(word[:-2])
    # supporters → supporter (not for short words or double-s like "boss")
    if len(word) >= 4 and word.endswith("s") and not word.endswith("ss"):
        stem = word[:-1]
        if stem not in forms:
            forms.append(stem)
    return forms


class ClueEnricher:
    def __init__(self, db_path=None):
        self.db_path = str(db_path or DB_PATH)
        self.conn = sqlite3.connect(self.db_path)
        self._load_indicators()
        self._load_wordplay()
        self._load_homophones()
        self._load_synonyms()
        self._load_definitions()

    def _load_indicators(self):
        """Pre-load indicators into a dict for fast lookup."""
        cur = self.conn.execute(
            "SELECT word, wordplay_type, confidence FROM indicators"
        )
        self.indicators = {}
        for word, wtype, conf in cur:
            tag = wtype if conf != 'low' else f"{wtype}?"
            self.indicators.setdefault(word.lower(), []).append(tag)

    def _load_wordplay(self):
        """Pre-load abbreviations into a dict."""
        cur = self.conn.execute(
            "SELECT indicator, substitution FROM wordplay"
        )
        self.abbreviations = {}
        for word, sub in cur:
            self.abbreviations.setdefault(word.lower(), []).append(sub)

    def _load_homophones(self):
        """Pre-load homophones into a dict."""
        cur = self.conn.execute(
            "SELECT word, homophone FROM homophones"
        )
        self.homophones = {}
        for word, homophone in cur:
            self.homophones.setdefault(word.lower(), []).append(homophone)

    def _load_synonyms(self):
        """Pre-load all synonyms into a dict, sorted by length per word."""
        cur = self.conn.execute(
            "SELECT word, synonym FROM synonyms_pairs"
        )
        raw = {}
        for word, synonym in cur:
            raw.setdefault(word.lower(), set()).add(synonym.upper())
        # Sort each word's synonyms by length then alphabetically
        self.synonyms = {}
        for word, syn_set in raw.items():
            self.synonyms[word] = sorted(syn_set, key=lambda s: (len(s), s))
        print(f"  Loaded {len(self.synonyms):,} synonym keys ({sum(len(v) for v in self.synonyms.values()):,} pairs)")

    def _load_definitions(self):
        """Pre-load definition->answer pairs into a set for fast lookup."""
        cur = self.conn.execute(
            "SELECT definition, answer FROM definition_answers_augmented"
        )
        self.definitions = set()
        for defn, answer in cur:
            if defn and answer:
                self.definitions.add((defn.lower(), answer.upper()))
        print(f"  Loaded {len(self.definitions):,} definition pairs")

    def lookup_synonyms(self, word, max_results=10, max_len=None, answer=None):
        """Look up synonyms for a word from pre-loaded dict.

        Tries possessive/plural variants if exact match is empty.
        If answer is provided, prioritize synonyms that are substrings of the
        answer -- these are the ones most likely to be wordplay components.
        """
        # Merge synonyms from all word variants, deduplicating
        all_syns = []
        seen = set()
        for variant in _word_variants(word.lower()):
            for s in self.synonyms.get(variant, []):
                if s not in seen:
                    all_syns.append(s)
                    seen.add(s)
        syns = all_syns
        if max_len is not None:
            syns = [s for s in syns if len(s) <= max_len]
        if answer and syns:
            answer_clean = re.sub(r"[^A-Z]", "", answer.upper())
            # Priority 1: substrings of answer (useful as charade pieces)
            in_answer = [s for s in syns if s in answer_clean and len(s) >= 2]
            # Priority 2: container outers (answer = syn with letters inserted)
            seen = set(in_answer)
            container_outer = [s for s in syns if s not in seen
                               and _is_container_outer(s, answer_clean)]
            seen.update(container_outer)
            # Priority 3: fractured substrings (syn split into 2 contiguous parts
            # in the answer, with a gap — catches container+charade)
            fractured = [s for s in syns if s not in seen
                         and _is_fractured_substring(s, answer_clean)]
            seen.update(fractured)
            # Priority 4: contain the answer (useful for deletion/outer_deletion)
            contains_answer = [s for s in syns if s not in seen
                               and answer_clean in s and len(s) > len(answer_clean)]
            seen.update(contains_answer)
            # Everything else
            rest = [s for s in syns if s not in seen]
            syns = in_answer + container_outer + fractured + contains_answer + rest
        return syns[:max_results]

    def lookup_abbreviations(self, word):
        """Look up abbreviations, trying possessive/plural variants."""
        result = []
        seen = set()
        for variant in _word_variants(word.lower()):
            for a in self.abbreviations.get(variant, []):
                if a not in seen:
                    result.append(a)
                    seen.add(a)
        return result

    def lookup_indicators(self, word):
        """Look up indicators, trying possessive/plural variants."""
        result = []
        for variant in _word_variants(word.lower()):
            result.extend(self.indicators.get(variant, []))
        return list(set(result))

    def lookup_definition(self, phrase, answer):
        """Check if a phrase defines the answer."""
        return (phrase.lower(), answer.upper()) in self.definitions

    def enrich(self, clue_text, answer, max_synonym_results=10):
        """Build enrichment context for a clue.

        Returns a formatted string with all DB lookups for each clue word/phrase.
        """
        words = clue_text.split()
        answer_len = len(answer.replace(" ", "").replace("-", ""))
        lines = []

        # Build word list, splitting hyphenated words into sub-words too
        word_list = []
        for word in words:
            w = word.lower().strip(".,;:!?\"'()")
            if w:
                word_list.append(w)
                # Also add hyphenated sub-words
                if "-" in w:
                    for part in w.split("-"):
                        part = part.strip()
                        if part and len(part) >= 2:
                            word_list.append(part)

        # Single words
        seen_words = set()
        for w in word_list:
            if not w or len(w) < 2 or w in seen_words:
                continue
            seen_words.add(w)

            entries = []
            answer_clean = re.sub(r"[^A-Z]", "", answer.upper())

            # Synonyms (filter by length - synonym can't be longer than answer)
            syns = self.lookup_synonyms(w, max_results=20, max_len=answer_len, answer=answer)
            syns = syns[:max_synonym_results]
            if syns:
                # Mark synonyms relevant to the answer (substrings, container outers, or fractured substrings)
                marked = []
                for s in syns:
                    if s in answer_clean and len(s) >= 2:
                        marked.append(s + "*")
                    elif _is_container_outer(s, answer_clean):
                        marked.append(s + "*")
                    elif _is_fractured_substring(s, answer_clean):
                        marked.append(s + "*")
                    else:
                        marked.append(s)
                entries.append(f"syn={','.join(marked)}")

            # Abbreviation (try word variants)
            abbrevs = []
            seen_abbr = set()
            for variant in _word_variants(w):
                for a in self.abbreviations.get(variant, []):
                    if a not in seen_abbr:
                        abbrevs.append(a)
                        seen_abbr.add(a)
            if abbrevs:
                marked = []
                for a in abbrevs:
                    a_upper = re.sub(r"[^A-Z]", "", a.upper())
                    if a_upper and a_upper in answer_clean:
                        marked.append(a + "*")
                    else:
                        marked.append(a)
                entries.append(f"abbr={','.join(marked)}")

            # Indicator (try word variants)
            ind_types = []
            for variant in _word_variants(w):
                ind_types.extend(self.indicators.get(variant, []))
            if ind_types:
                entries.append(f"ind={','.join(sorted(set(ind_types)))}")

            # Homophone (try word variants)
            homos = []
            seen_homo = set()
            for variant in _word_variants(w):
                for h in self.homophones.get(variant, []):
                    if h not in seen_homo:
                        homos.append(h)
                        seen_homo.add(h)
            if homos:
                entries.append(f"sounds={','.join(homos)}")

            # Anagram fodder: flag if literal letters overlap heavily with answer
            w_upper = re.sub(r"[^A-Z]", "", w.upper())
            if len(w_upper) >= 4 and len(answer_clean) >= 5:
                pool = list(answer_clean)
                match_count = 0
                for c in w_upper:
                    if c in pool:
                        pool.remove(c)
                        match_count += 1
                if (match_count >= 4
                        and match_count / len(answer_clean) >= 0.7
                        and match_count / len(w_upper) >= 0.7):
                    entries.append(f"fodder={w_upper}({match_count}/{len(answer_clean)} anagram match)")

            if entries:
                lines.append(f"  {w}: {'; '.join(entries)}")

        # Two-word phrases (for indicators like "mixed up", "going around")
        for i in range(len(words) - 1):
            phrase = (
                words[i].lower().strip(".,;:!?\"'()")
                + " "
                + words[i + 1].lower().strip(".,;:!?\"'()")
            )
            ind_types = self.indicators.get(phrase, [])
            if ind_types:
                lines.append(f"  {phrase}: ind={','.join(sorted(set(ind_types)))}")

            # Two-word synonym lookup
            syns = self.lookup_synonyms(phrase, max_results=10, max_len=answer_len, answer=answer)
            if syns:
                answer_clean = re.sub(r"[^A-Z]", "", answer.upper())
                marked = [s + "*" if s in answer_clean and len(s) >= 2 else s for s in syns]
                lines.append(f"  {phrase}: syn={','.join(marked)}")

        # Three-word phrases
        for i in range(len(words) - 2):
            phrase = " ".join(
                w.lower().strip(".,;:!?\"'()") for w in words[i : i + 3]
            )
            syns = self.lookup_synonyms(phrase, max_results=6, max_len=answer_len, answer=answer)
            syns = syns[:4]
            if syns:
                answer_clean = re.sub(r"[^A-Z]", "", answer.upper())
                marked = [s + "*" if s in answer_clean and len(s) >= 2 else s for s in syns]
                lines.append(f"  {phrase}: syn={','.join(marked)}")

        # Definition check (first 1-3 words and last 1-3 words)
        def_checks = []
        for n in range(1, min(4, len(words) + 1)):
            prefix = " ".join(words[:n]).strip(".,;:!?\"'()")
            suffix = " ".join(words[-n:]).strip(".,;:!?\"'()")
            if self.lookup_definition(prefix, answer):
                def_checks.append(f"  \"{prefix}\" -> {answer} (definition match)")
            if suffix != prefix and self.lookup_definition(suffix, answer):
                def_checks.append(f"  \"{suffix}\" -> {answer} (definition match)")

        result = ""
        if lines:
            result += "DB lookups:\n" + "\n".join(lines)
        if def_checks:
            result += "\nDefinitions:\n" + "\n".join(def_checks)

        return result

    def close(self):
        self.conn.close()
