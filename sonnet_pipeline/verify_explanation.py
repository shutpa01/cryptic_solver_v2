"""Mechanical verification of cryptic crossword explanations.

Verifies claims in explanations against reference data and string operations.
No AI needed — pure mechanical checks.

Checks performed:
1. ASSEMBLY: Do the claimed pieces concatenate to the answer?
2. DEFINITION: Does the claimed definition map to the answer in our DB?
3. SYNONYMS: Are claimed synonym relationships real?
4. ABBREVIATIONS: Are claimed abbreviations real?
5. INDICATORS: Are claimed indicators real for the claimed type?
6. HIDDEN: Is the answer actually hidden in the claimed text?
7. ANAGRAM: Do the claimed letters actually anagram to the result?
8. REVERSAL: Does reversing the claimed text give the result?
"""

import re
import sqlite3
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
REF_DB = str(PROJECT_ROOT / "data" / "cryptic_new.db")
CLUES_DB = str(PROJECT_ROOT / "data" / "clues_master.db")

# Link words allowed between / around the two windows of a Double Definition
# and in other "what's left over" checks. Canonical source: the LINKERS set
# in `enrichment/05_self_learning_enrichment.py`. Kept in sync manually —
# that module's name starts with a digit so it can't be imported directly.
LINKERS = {
    "of", "in", "the", "a", "an", "to", "for", "with", "and", "or",
    "by", "from", "as", "on", "at", "but", "so", "yet", "if", "not",
    "nor", "up", "it", "its", "into", "onto", "within", "without",
    "that", "which", "when", "where", "while", "how", "why", "who",
    "this", "these", "those", "such", "one", "ones", "some", "any",
    "all", "here", "there",
    "is", "are", "be", "been", "being", "was", "were",
    "has", "have", "had", "having",
    "will", "would", "could", "should", "must", "may", "might",
    "get", "gets", "got", "getting",
    "give", "gives", "gave", "given", "giving",
    "make", "makes", "made", "making",
    "need", "needs",
    "thus", "hence", "therefore", "maybe",
    "dont", "doesnt", "didnt", "wont", "wouldnt", "cant", "isnt", "arent",
    "once",
}


class ExplanationVerifier:
    def __init__(self, ref_db=None, clues_db=None):
        self.ref = sqlite3.connect(ref_db or REF_DB)
        self.ref.row_factory = sqlite3.Row
        # Cache lookups for speed
        self._syn_cache = {}
        self._abbr_cache = {}
        self._ind_cache = {}
        self._def_cache = {}
        # Load indicators table for clue-level indicator scanning
        self._indicators_by_word = {}
        for row in self.ref.execute("SELECT word, wordplay_type, subtype FROM indicators"):
            w = row[0].lower().strip()
            self._indicators_by_word.setdefault(w, []).append((row[1], row[2]))

    def is_synonym(self, word, target):
        """Check if word -> target is a known synonym pair.

        Checks both synonyms_pairs and definition_answers_augmented,
        in both directions.
        """
        key = (word.lower(), target.lower())
        if key not in self._syn_cache:
            w, t = word.lower(), target.lower()
            row = None
            for w1, w2 in [(w, t), (t, w)]:
                if row:
                    break
                row = self.ref.execute(
                    "SELECT 1 FROM synonyms_pairs WHERE LOWER(word) = ? AND LOWER(synonym) = ? LIMIT 1",
                    (w1, w2),
                ).fetchone()
                if not row:
                    row = self.ref.execute(
                        "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition) = ? AND LOWER(answer) = ? LIMIT 1",
                        (w1, w2),
                    ).fetchone()
            self._syn_cache[key] = row is not None
        return self._syn_cache[key]

    def is_abbreviation(self, word, letters):
        """Check if word abbreviates to letters."""
        key = (word.lower(), letters.upper())
        if key not in self._abbr_cache:
            row = self.ref.execute(
                "SELECT 1 FROM wordplay WHERE LOWER(indicator) = ? AND UPPER(substitution) = ? LIMIT 1",
                (word.lower(), letters.upper()),
            ).fetchone()
            if not row:
                row = self.ref.execute(
                    "SELECT 1 FROM synonyms_pairs WHERE LOWER(word) = ? AND UPPER(synonym) = ? LIMIT 1",
                    (word.lower(), letters.upper()),
                ).fetchone()
            self._abbr_cache[key] = row is not None
        return self._abbr_cache[key]

    def is_homophone(self, word1, word2):
        """Check if word1 and word2 are registered homophones.

        Looks up both directions in the `homophones` table since it stores
        pairs both ways.
        """
        w1, w2 = word1.lower().strip(), word2.lower().strip()
        if not w1 or not w2:
            return False
        row = self.ref.execute(
            "SELECT 1 FROM homophones WHERE "
            "(LOWER(word)=? AND LOWER(homophone)=?) OR "
            "(LOWER(word)=? AND LOWER(homophone)=?) LIMIT 1",
            (w1, w2, w2, w1),
        ).fetchone()
        return row is not None

    def is_indicator(self, word, wordplay_type):
        """Check if word is a known indicator for the given type."""
        key = (word.lower(), wordplay_type.lower())
        if key not in self._ind_cache:
            row = self.ref.execute(
                "SELECT 1 FROM indicators WHERE LOWER(word) = ? AND LOWER(wordplay_type) = ? LIMIT 1",
                (word.lower(), wordplay_type.lower()),
            ).fetchone()
            self._ind_cache[key] = row is not None
        return self._ind_cache[key]

    def definition_matches(self, definition, answer):
        """Check if definition -> answer exists in our DB (forward or reverse)."""
        key = (definition.lower(), answer.upper())
        if key not in self._def_cache:
            d_low = definition.lower().strip()
            a_up = answer.upper().strip()
            # Forward: definition -> answer
            row = self.ref.execute(
                "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition) = ? AND UPPER(answer) = ? LIMIT 1",
                (d_low, a_up),
            ).fetchone()
            if not row:
                row = self.ref.execute(
                    "SELECT 1 FROM synonyms_pairs WHERE LOWER(word) = ? AND UPPER(synonym) = ? LIMIT 1",
                    (d_low, a_up),
                ).fetchone()
            # Reverse: answer -> definition
            if not row:
                row = self.ref.execute(
                    "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition) = ? AND UPPER(answer) = ? LIMIT 1",
                    (a_up.lower(), d_low.upper()),
                ).fetchone()
            if not row:
                row = self.ref.execute(
                    "SELECT 1 FROM synonyms_pairs WHERE LOWER(word) = ? AND UPPER(synonym) = ? LIMIT 1",
                    (a_up.lower(), d_low.upper()),
                ).fetchone()
            self._def_cache[key] = row is not None
        return self._def_cache[key]

    def explanation_names_indicator(self, expl, op_type):
        """Find [op_type: "word"] annotation in the explanation and check
        the named word is a known indicator of op_type.

        Returns one of:
          ('verified', word) — annotation present and word IS an indicator
          ('wrong',    word) — annotation present but word is NOT an indicator
          ('missing',  None) — no annotation found in the explanation

        Accepts brackets that bundle multiple annotations:
          [container: "visiting"; reversal: "upset"]
        as well as a single annotation per bracket.
        """
        pattern = re.compile(
            r"\b" + re.escape(op_type) + r"\s*:\s*[\"']([^\"']+)[\"']",
            re.IGNORECASE,
        )
        m = pattern.search(expl or "")
        if not m:
            return ('missing', None)
        word = m.group(1).strip()
        if self.is_indicator(word, op_type):
            return ('verified', word)
        # Container annotations also accept insertion-type indicators.
        if op_type == 'container' and self.is_indicator(word, 'insertion'):
            return ('verified', word)
        return ('wrong', word)

    def clue_has_indicator(self, clue_text, required_type, required_subtypes=None):
        """Check if any word in the clue is a known indicator of the required type.

        Scans single words and 2-word phrases against the indicators table.
        If required_subtypes is provided, also checks the subtype matches.
        """
        clue_words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", clue_text.lower())
        # Single words
        for word in clue_words:
            entries = self._indicators_by_word.get(word, [])
            for wtype, subtype in entries:
                if wtype == required_type:
                    if required_subtypes is None or subtype in required_subtypes:
                        return True
        # 2-word phrases
        for i in range(len(clue_words) - 1):
            phrase = clue_words[i] + " " + clue_words[i + 1]
            entries = self._indicators_by_word.get(phrase, [])
            for wtype, subtype in entries:
                if wtype == required_type:
                    if required_subtypes is None or subtype in required_subtypes:
                        return True
        return False

    def check_assembly(self, pieces_letters, answer):
        """Do the pieces concatenate to the answer?"""
        assembled = "".join(pieces_letters).upper().replace(" ", "")
        target = answer.upper().replace(" ", "")
        return assembled == target

    def check_hidden(self, text, answer):
        """Is the answer hidden in the text (forward or reversed)?"""
        clean_text = re.sub(r"[^A-Z]", "", text.upper())
        clean_answer = answer.upper().replace(" ", "")
        return clean_answer in clean_text or clean_answer[::-1] in clean_text

    def check_anagram(self, fodder, result):
        """Do the fodder letters anagram to the result?"""
        f = sorted(re.sub(r"[^A-Z]", "", fodder.upper()))
        r = sorted(re.sub(r"[^A-Z]", "", result.upper()))
        return f == r

    def check_reversal(self, source, result):
        """Does reversing the source give the result?"""
        s = re.sub(r"[^A-Z]", "", source.upper())
        r = re.sub(r"[^A-Z]", "", result.upper())
        return s[::-1] == r

    def check_reversal_source(self, letters, source_word):
        """Verify a 'X (reversal of "Y")' piece claim.

        Returns True if some synonym, abbreviation, or the literal letters
        of source_word, when reversed, equals letters. False otherwise.
        Returns None if either input is empty.
        """
        L = re.sub(r"[^A-Z]", "", (letters or "").upper())
        sw = (source_word or "").lower().strip(".,;:!?\"'()-")
        if not L or not sw:
            return None
        target = L[::-1]  # the source candidate that, reversed, gives L
        # Direct: source_word's letters themselves
        sw_letters = re.sub(r"[^A-Z]", "", source_word.upper())
        if sw_letters == target:
            return True
        # Synonym lookup
        try:
            for syn in self.ref.execute(
                "SELECT synonym FROM synonyms_pairs WHERE LOWER(word)=?",
                (sw,),
            ):
                if re.sub(r"[^A-Z]", "", (syn[0] or "").upper()) == target:
                    return True
        except Exception:
            pass
        # Abbreviation lookup
        try:
            for sub in self.ref.execute(
                "SELECT substitution FROM wordplay WHERE LOWER(indicator)=?",
                (sw,),
            ):
                if re.sub(r"[^A-Z]", "", (sub[0] or "").upper()) == target:
                    return True
        except Exception:
            pass
        return False

    def check_deletion_mechanism(self, source, result):
        """Is `result` obtainable from `source` by a recognised cryptic deletion?

        Accepts head, tail, outer (both ends), and heart (middle) removals of
        one or more contiguous letters. Rejects arbitrary interior substring
        matches — those are not valid deletions in cryptic convention.
        """
        s = re.sub(r"[^A-Z]", "", (source or "").upper())
        r = re.sub(r"[^A-Z]", "", (result or "").upper())
        if not s or not r or len(r) >= len(s):
            return False
        # Head removal: drop 1..N from the front
        for n in range(1, len(s) - len(r) + 1):
            if s[n:] == r:
                return True
        # Tail removal: drop 1..N from the back
        for n in range(1, len(s) - len(r) + 1):
            if s[:-n] == r:
                return True
        # Outer removal: drop both ends (keep contiguous middle)
        if len(s) - len(r) >= 2:
            for start in range(1, len(s) - len(r)):
                end = start + len(r)
                if end < len(s) and s[start:end] == r:
                    return True
        # Heart removal: middle char(s) removed, keep head+tail
        gap = len(s) - len(r)
        if gap >= 1:
            for start in range(1, len(r)):
                if s[:start] + s[start + gap:] == r:
                    return True
        return False

    def check_positional(self, kind, letters, source):
        """Verify a positional-extraction claim: e.g. 'E is the last letter of "conclusion"'.

        Returns True if the claimed letters match the extraction, False if they don't,
        None if the claim can't be evaluated (empty source or letters).
        """
        src = re.sub(r"[^A-Za-z]", "", source or "").upper()
        L = (letters or "").strip().upper()
        n = len(L)
        if not src or not L:
            return None
        if kind == "first":
            return src.startswith(L)
        if kind == "last":
            return src.endswith(L)
        if kind == "middle":
            if len(src) < n:
                return False
            start = (len(src) - n) // 2
            return src[start:start + n] == L
        if kind == "outer":
            # Outer letters = first + last (drop middle). For n=2 the
            # standard form is the single first + single last char.
            if n == 2:
                return L[0] == src[0] and L[-1] == src[-1]
            # For n>2 the convention varies; accept first half + last half.
            half = n // 2
            return L == src[:half] + src[-(n - half):]
        if kind == "initial":
            return n == 1 and src.startswith(L)
        if kind == "final":
            return n == 1 and src.endswith(L)
        if kind == "odd":
            odd = "".join(src[i] for i in range(0, len(src), 2))
            return odd == L
        if kind == "even":
            even = "".join(src[i] for i in range(1, len(src), 2))
            return even == L
        if kind == "alternate":
            odd = "".join(src[i] for i in range(0, len(src), 2))
            even = "".join(src[i] for i in range(1, len(src), 2))
            return L in (odd, even)
        return None

    def check_container(self, pieces, answer):
        """Can one piece be inserted into another to form the answer?

        Tries each piece as the inner word, the rest concatenated as the outer.
        Tests all insertion positions in the outer word.
        Returns (True, inner, outer) on success, (False, None, None) on failure.
        """
        answer_clean = re.sub(r"[^A-Z]", "", answer.upper())
        for i, inner in enumerate(pieces):
            outer = "".join(pieces[:i] + pieces[i+1:]).upper()
            inner_clean = re.sub(r"[^A-Z]", "", inner.upper())
            # Try inserting inner at every position in outer
            for pos in range(1, len(outer)):
                candidate = outer[:pos] + inner_clean + outer[pos:]
                if candidate == answer_clean:
                    return True, inner_clean, outer
        return False, None, None

    def _classify_clue_words(self, clue_text, ai_explanation, definition_text):
        """Classify every clue word by the role the explanation claims for it.

        Returns dict with:
            classified: list of (word, role) tuples in clue order
            unaccounted: list of words neither role-claimed nor in LINK_WORDS
            link: list of words classified as link
            summary: short human-readable summary

        Roles (in claim-priority order):
            definition, synonym_source, abbreviation_source, positional_source,
            reversal_source, deletion_source, indicator, anagram_fodder,
            hidden_source, dbe_marker, link, unaccounted

        Order matters: a clue word is claimed by the FIRST role that owns it.
        Whatever remains after all role claims is matched against LINK_WORDS
        — present in list → link; not in list → unaccounted.
        """
        from signature_solver.tokens import (
            LINK_WORDS, DBE_MARKERS_SINGLE, DBE_MARKERS_MULTI,
        )

        clue = clue_text or ""
        # Strip enumeration "(7)", "(3,4)", etc. before tokenising
        clue = re.sub(r"\s*\([\d,\-\s/]+\)\s*$", "", clue)
        words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", clue.lower())
        if not words:
            return {"classified": [], "unaccounted": [], "link": [],
                    "summary": "no clue words"}

        expl = ai_explanation or ""
        claimed = {}  # word_index -> role

        # Quoted-value pattern: matches "..." OR '...' allowing the *other*
        # quote type inside (so synonym="queen's worker" captures full value).
        # Use group(1) or group(2) — whichever matched.
        Q = r'(?:"([^"]+)"|\'([^\']+)\')'

        def _qval(m):
            return m.group(1) if m.group(1) is not None else m.group(2)

        def _norm(w):
            """Strip trailing 's possessive for matching equivalence."""
            return w[:-2] if w.endswith("'s") else w

        def _claim_phrase(phrase, role):
            """Claim each word of `phrase` once against unclaimed clue words.
            Match is case-insensitive and possessive-tolerant
            (lincolnshire matches lincolnshire's, and vice versa)."""
            if not phrase:
                return
            phrase_words = re.findall(
                r"[a-zA-Z]+(?:'[a-zA-Z]+)?", phrase.lower())
            for pw in phrase_words:
                pw_norm = _norm(pw)
                for i, w in enumerate(words):
                    if i in claimed:
                        continue
                    if _norm(w) == pw_norm:
                        claimed[i] = role
                        break

        # 1. Definition phrase
        if definition_text:
            _claim_phrase(definition_text, "definition")

        # 2. Synonym sources: (synonym="X") and "synonym of \"X\""
        for m in re.finditer(
                r"synonym\s*=\s*" + Q, expl, re.IGNORECASE):
            _claim_phrase(_qval(m), "synonym_source")
        for m in re.finditer(
                r"synonym\s+of\s+" + Q, expl, re.IGNORECASE):
            _claim_phrase(_qval(m), "synonym_source")

        # 3. Abbreviation sources
        for m in re.finditer(
                r"abbreviation\s*=\s*" + Q, expl, re.IGNORECASE):
            _claim_phrase(_qval(m), "abbreviation_source")
        for m in re.finditer(
                r"abbreviation\s+of\s+" + Q, expl, re.IGNORECASE):
            _claim_phrase(_qval(m), "abbreviation_source")
        for m in re.finditer(
                r"abbr\.?\s*(?:of\s+)?" + Q, expl, re.IGNORECASE):
            _claim_phrase(_qval(m), "abbreviation_source")

        # 4. Positional sources: (first/last/middle/outer/etc. letter(s) of "X")
        for m in re.finditer(
                r"(?:first|last|middle|outer|initial|final|odd|even"
                r"|alternat(?:e|ing))\s+letters?\s+(?:of|in|from)\s+" + Q,
                expl, re.IGNORECASE):
            _claim_phrase(_qval(m), "positional_source")

        # 5. Reversal sources: (reversal of "X") or (reverse of "X")
        for m in re.finditer(
                r"(?:reversal|reverse)\s+of\s+" + Q,
                expl, re.IGNORECASE):
            _claim_phrase(_qval(m), "reversal_source")

        # 6. Deletion sources: deletion="X"
        for m in re.finditer(
                r"deletion\s*=\s*" + Q, expl, re.IGNORECASE):
            _claim_phrase(_qval(m), "deletion_source")

        # 7. Indicator phrases: [type: "X"; type: "Y"; ...]
        # Multi-indicator brackets use ';' separators inside one [...] pair,
        # so we walk each bracket's inner content and extract each
        # type:"phrase" pair. A single \[type:"X"\] regex would only catch
        # the FIRST indicator in a multi-indicator bracket.
        _IND_TYPES = (r"anagram|reversal|container|deletion|homophone|"
                      r"hidden|first\s+letters?|last\s+letters?|middle\s+letters?|"
                      r"outer\s+letters?|initial|final|odd|even|alternat(?:e|ing)|"
                      r"spoonerism|charade|cycle|cycling|insertion|"
                      r"acrostic|substitution|parts|selection")
        for bm in re.finditer(r"\[([^\]]+)\]", expl):
            inner = bm.group(1)
            for m in re.finditer(
                    r"(?:" + _IND_TYPES + r")\s*:\s*" + Q,
                    inner, re.IGNORECASE):
                _claim_phrase(_qval(m), "indicator")

        # 8. Anagram fodder: clue words appearing as uppercase tokens
        # between "anagram of" and the next "=" (greedy stop at last =).
        # Includes single-letter fodder (e.g. "anagram of I + ANOTHER").
        # First tries literal lowercased match; failing that, queries the
        # synonym/abbreviation DB so that fodder ET claims clue word "alien"
        # when the explanation lacks an inline (synonym=...) annotation.
        ana_match = re.search(
            r"anagram\s+(?:of\s+)?(.+?)\s*=", expl, re.IGNORECASE)
        if ana_match:
            fodder_words = re.findall(r"\b[A-Z]+\b", ana_match.group(1))
            for fw in fodder_words:
                fw_lower = fw.lower()
                fw_upper = fw.upper()
                # Step 1: literal match
                literal_claimed = False
                for i, w in enumerate(words):
                    if i in claimed:
                        continue
                    if _norm(w) == fw_lower:
                        claimed[i] = "anagram_fodder"
                        literal_claimed = True
                        break
                if literal_claimed:
                    continue
                # Step 2: synonym/abbreviation lookup against unclaimed words
                for i, w in enumerate(words):
                    if i in claimed:
                        continue
                    base = _norm(w)
                    try:
                        if (self.is_synonym(base, fw_upper)
                                or self.is_abbreviation(base, fw_upper)):
                            claimed[i] = "anagram_fodder"
                            break
                    except Exception:
                        pass

        # 9. Hidden span: hidden in "X" — claim clue words whose letters
        # contiguously appear in the span (case-insensitive substring check
        # over the letter-only normalisation).
        for m in re.finditer(
                r'hidden(?:\s+reversed)?\s+in\s+["\']([^"\']+)["\']',
                expl, re.IGNORECASE):
            span_letters = re.sub(r"[^a-z]", "", m.group(1).lower())
            for i, w in enumerate(words):
                if i in claimed:
                    continue
                if w and w in span_letters:
                    claimed[i] = "hidden_source"

        # 10. DBE markers (single + multi-word phrases)
        for i, w in enumerate(words):
            if i in claimed:
                continue
            if w in DBE_MARKERS_SINGLE:
                claimed[i] = "dbe_marker"
        for i in range(len(words) - 1):
            if i in claimed or (i + 1) in claimed:
                continue
            if (words[i], words[i + 1]) in DBE_MARKERS_MULTI:
                claimed[i] = "dbe_marker"
                claimed[i + 1] = "dbe_marker"

        # 11. Link words (last; leftover AND in LINK_WORDS)
        link_words = []
        unaccounted = []
        classified = []
        for i, w in enumerate(words):
            if i in claimed:
                classified.append((w, claimed[i]))
            elif w in LINK_WORDS:
                link_words.append(w)
                classified.append((w, "link"))
            else:
                unaccounted.append(w)
                classified.append((w, "unaccounted"))

        if unaccounted:
            summary = (f"{len(unaccounted)}/{len(words)} unaccounted: "
                       + ", ".join(unaccounted))
        else:
            summary = (f"all {len(words)} clue words accounted for "
                       f"({len(link_words)} link)")

        return {
            "classified": classified,
            "unaccounted": unaccounted,
            "link": link_words,
            "summary": summary,
        }

    def verify(self, clue_text, answer, definition, wordplay_type, ai_explanation):
        """Run all mechanical checks on an explanation.

        Returns dict with:
            score: 0-100
            checks: list of {check, passed, detail}
            verdict: HIGH/MEDIUM/LOW/FAIL
        """
        checks = []
        answer_clean = (answer or "").upper().replace(" ", "")

        if not ai_explanation or not answer:
            return {"score": 0, "checks": [], "verdict": "FAIL"}

        expl = ai_explanation

        # --- CHECK 1: Definition verification ---
        if definition:
            matched = self.definition_matches(definition, answer)
            checks.append({
                "check": "definition",
                "status": "verified" if matched else "unverifiable",
                "detail": f"'{definition}' -> {answer}: {'VERIFIED' if matched else 'not in DB'}",
            })
        else:
            checks.append({
                "check": "no_definition",
                "status": "wrong",
                "detail": "No definition provided",
            })

        # --- CHECK 2: Parse and verify pieces ---
        # Format 1: WORD (synonym of 'source') — Sonnet format
        pieces = re.findall(
            r"(\w+)\s*\(\s*synonym\s+of\s+[\"']([^\"']+)[\"']\s*\)",
            expl, re.IGNORECASE,
        )
        # Format 2: WORD (synonym="source") — mechanical solver format
        pieces += re.findall(
            r"(\w+)\s*\(\s*synonym\s*=\s*\"([^\"]+)\"\s*\)",
            expl, re.IGNORECASE,
        )
        for result_word, source_word in pieces:
            matched = self.is_synonym(source_word, result_word)
            # Also try individual words from multi-word source
            if not matched:
                for w in source_word.lower().split():
                    if len(w) > 2 and self.is_synonym(w, result_word):
                        matched = True
                        break
            # Single-letter "synonyms" not in DB are fabricated abbreviations — wrong
            if not matched and len(result_word.strip()) == 1:
                checks.append({
                    "check": "synonym",
                    "status": "wrong",
                    "detail": f"'{source_word}' = {result_word}: single-letter NOT in DB",
                })
            else:
                checks.append({
                    "check": "synonym",
                    "status": "verified" if matched else "unverifiable",
                    "detail": f"'{source_word}' = {result_word}: {'VERIFIED' if matched else 'not in DB'}",
                })

        # Check abbreviation claims — verify the specific clue_word → letters mapping
        # Format 1: WORD (abbr. of 'source') — Sonnet format
        abbrs = re.findall(
            r"(\w+)\s*\(\s*abbr\.?\s+(?:of\s+)?[\"']?([^\"')]+)[\"']?\s*\)",
            expl, re.IGNORECASE,
        )
        # Format 2: WORD (abbreviation="source") — mechanical solver format
        abbrs += re.findall(
            r"(\w+)\s*\(\s*abbreviation\s*=\s*\"([^\"]+)\"\s*\)",
            expl, re.IGNORECASE,
        )
        for letters, word in abbrs:
            word_clean = word.strip().lower()
            letters_clean = letters.strip().upper()
            # Check exact mapping: does THIS word abbreviate to THESE letters?
            matched = self.is_abbreviation(word_clean, letters_clean)
            if not matched:
                # Also try individual words if multi-word
                for w in word_clean.split():
                    if self.is_abbreviation(w, letters_clean):
                        matched = True
                        break
            checks.append({
                "check": "abbreviation",
                "status": "verified" if matched else "wrong",
                "detail": f"'{word_clean}' -> {letters_clean}: {'VERIFIED' if matched else 'NOT KNOWN'}",
            })

        # --- CHECK 2c: Source words must appear in the clue ---
        # Every synonym or abbreviation source cited in the explanation must
        # be a word or phrase that actually occurs in the clue. Catches
        # hallucinated sources where the DB supports source->letters but the
        # source isn't what the setter put in the clue.
        if clue_text:
            clue_norm = clue_text.lower()
            seen_phantom = set()
            for result_word, source_word in pieces:
                source_norm = source_word.lower().strip(" \t\"'?!.,;:")
                if not source_norm:
                    continue
                # Accept full-phrase substring match
                if source_norm in clue_norm:
                    continue
                # Accept if every content word of the source is a word in the clue
                source_tokens = [t for t in re.findall(r"[a-z']+", source_norm)
                                 if t not in LINKERS]
                clue_tokens = set(re.findall(r"[a-z']+", clue_norm))
                if source_tokens and all(t in clue_tokens for t in source_tokens):
                    continue
                key = ("syn", result_word.upper(), source_norm)
                if key in seen_phantom:
                    continue
                seen_phantom.add(key)
                checks.append({
                    "check": "source_not_in_clue",
                    "status": "wrong",
                    "detail": f"synonym source '{source_word}' not in clue "
                              f"for piece '{result_word}'",
                })
            for letters, word in abbrs:
                source_norm = word.lower().strip(" \t\"'?!.,;:")
                if not source_norm:
                    continue
                if source_norm in clue_norm:
                    continue
                source_tokens = [t for t in re.findall(r"[a-z']+", source_norm)
                                 if t not in LINKERS]
                clue_tokens = set(re.findall(r"[a-z']+", clue_norm))
                if source_tokens and all(t in clue_tokens for t in source_tokens):
                    continue
                key = ("abbr", letters.upper(), source_norm)
                if key in seen_phantom:
                    continue
                seen_phantom.add(key)
                checks.append({
                    "check": "source_not_in_clue",
                    "status": "wrong",
                    "detail": f"abbreviation source '{word}' not in clue "
                              f"for piece '{letters}'",
                })

        # --- CHECK 3: Assembly verification ---
        # Try arrow format: "WORD → LETTERS"
        arrow_pieces = re.findall(r"(\w+)\s*(?:->|\u2192)\s*([A-Z]+)", expl)
        if arrow_pieces:
            assembled = "".join(p[1] for p in arrow_pieces)
            asm_ok = assembled.upper() == answer_clean
            checks.append({
                "check": "assembly",
                "status": "verified" if asm_ok else "wrong",
                "detail": f"pieces={assembled} vs answer={answer_clean}: {'MATCH' if asm_ok else 'MISMATCH'}",
            })
        else:
            # Try new format: extract uppercase letter sequences before parentheses
            # Matches: "LETTERS(mechanism)" patterns and "inside LETTERS" and "reverse LETTERS"
            letter_pieces = re.findall(r"\b([A-Z]{1,})\s*\(", expl)
            # Also catch "inside WORD" and "reverse WORD" targets
            inside_targets = re.findall(r"inside\s+([A-Z]+)", expl)
            # Also catch "anagram of X+Y = ANSWER" — the answer itself confirms assembly
            anagram_eq = re.findall(r"=\s*([A-Z]+)", expl)

            assembly_found = False

            if letter_pieces:
                # Try charade: pieces should concatenate to answer
                assembled = "".join(letter_pieces)
                if assembled == answer_clean:
                    checks.append({
                        "check": "assembly",
                        "status": "verified",
                        "detail": f"pieces {'+'.join(letter_pieces)} = {answer_clean}: MATCH",
                    })
                    assembly_found = True

            if not assembly_found and anagram_eq:
                # Anagram format: "anagram of X+Y = ANSWER"
                if anagram_eq[0] == answer_clean:
                    # Extract fodder pieces
                    fodder_pieces = re.findall(r"\b([A-Z]{2,})\s*\(", expl)
                    fodder = "".join(fodder_pieces)
                    if sorted(fodder) == sorted(answer_clean):
                        checks.append({
                            "check": "assembly",
                            "status": "verified",
                            "detail": f"anagram of {fodder} = {answer_clean}: MATCH",
                        })

        # --- CHECK 4: Type-specific mechanical checks ---
        wtype = (wordplay_type or "").lower()

        # Hidden verification uses the span-naming convention: the explanation
        # must include `hidden in "..."` where UPPERCASE letters within the
        # quoted span concatenate to the answer (or reversed answer for
        # hidden_reversed). The span must also appear as a substring of the
        # clue text. This verifies the explanation's specific claim, not
        # just that the answer letters happen to be somewhere in the clue.
        if ("hidden" in wtype or "hidden in " in expl.lower()
                or "hidden reversed" in expl.lower()):
            is_reversed = (wtype == "hidden_reversed"
                           or "hidden reversed" in expl.lower())
            span_match = re.search(
                r"hidden(?:\s+reversed)?\s+in\s+[\"']([^\"']+)[\"']",
                expl, re.IGNORECASE,
            )
            if span_match:
                span = span_match.group(1)
                upper_letters = re.sub(r"[^A-Z]", "", span)
                target = answer_clean[::-1] if is_reversed else answer_clean
                letters_match = upper_letters == target
                span_letters = re.sub(r"[^a-z]", "", span.lower())
                clue_letters = re.sub(r"[^a-z]", "", clue_text.lower())
                span_in_clue = bool(span_letters) and span_letters in clue_letters
                if letters_match and span_in_clue:
                    checks.append({
                        "check": "hidden_word",
                        "status": "verified",
                        "detail": f"span '{span}' in clue; UPPERCASE "
                                  f"{'reverses to' if is_reversed else 'equals'} "
                                  f"{answer_clean}",
                    })
                else:
                    problems = []
                    if not letters_match:
                        problems.append(
                            f"UPPERCASE '{upper_letters}' != "
                            f"{'reversed ' if is_reversed else ''}answer '{target}'"
                        )
                    if not span_in_clue:
                        problems.append(f"span '{span}' not a substring of clue")
                    checks.append({
                        "check": "hidden_word",
                        "status": "wrong",
                        "detail": "; ".join(problems),
                    })
            else:
                checks.append({
                    "check": "hidden_word",
                    "status": "unverifiable",
                    "detail": "hidden explanation missing span in casing convention "
                              "hidden in \"...X...\"",
                })

        # --- CHECK 4b: Double Definition ---
        # A DD splits the clue into two windows, each of which defines the
        # answer. Both windows must map to the answer in the DB, both must
        # be substrings of the clue, and any clue words outside the two
        # windows must be on the LINKERS whitelist.
        if (wtype == "double_definition"
                or "double definition:" in expl.lower()):
            dd_match = re.search(
                r"double\s+definition:\s*(.+?)(?:;|$)",
                expl, re.IGNORECASE | re.DOTALL,
            )
            windows = []
            if dd_match:
                dd_content = dd_match.group(1)
                window_pairs = re.findall(
                    r"([^,=]+?)\s*=\s*(\w+)", dd_content,
                )
                for w, target in window_pairs:
                    if target.upper().replace(" ", "") == answer_clean:
                        windows.append(w.strip())

            if len(windows) >= 2:
                w1, w2 = windows[0], windows[1]
                # Strip leading/trailing punctuation for DB lookup — keeps
                # the display form intact while allowing `anecdote?` to match
                # a DB entry for `anecdote`.
                w1_lookup = w1.strip(" \t\"'?!.,;:")
                w2_lookup = w2.strip(" \t\"'?!.,;:")
                w1_ok = (self.definition_matches(w1_lookup, answer)
                         or self.is_synonym(w1_lookup, answer))
                w2_ok = (self.definition_matches(w2_lookup, answer)
                         or self.is_synonym(w2_lookup, answer))
                clue_lower = clue_text.lower()
                w1_in_clue = w1.lower() in clue_lower
                w2_in_clue = w2.lower() in clue_lower
                # Remainder: clue with both windows removed, remaining tokens
                # must all be link words.
                remainder_text = clue_lower
                if w1_in_clue:
                    remainder_text = remainder_text.replace(w1.lower(), " ", 1)
                if w2_in_clue:
                    remainder_text = remainder_text.replace(w2.lower(), " ", 1)
                remainder_tokens = re.findall(r"[a-z]+", remainder_text)
                non_link = [t for t in remainder_tokens if t not in LINKERS]
                all_links = not non_link

                if (w1_ok and w2_ok and w1_in_clue and w2_in_clue
                        and all_links):
                    checks.append({
                        "check": "dd",
                        "status": "verified",
                        "detail": f"DD verified: `{w1}` + `{w2}` both map to "
                                  f"{answer_clean}; remainder is link words",
                    })
                elif w1_ok or w2_ok:
                    checks.append({
                        "check": "dd",
                        "status": "unverifiable",
                        "detail": f"DD partial: `{w1}`={w1_ok}, `{w2}`={w2_ok} "
                                  f"(answer {answer_clean})"
                                  + (f", non-link remainder: {non_link}"
                                     if non_link else ""),
                    })
                else:
                    checks.append({
                        "check": "dd",
                        "status": "wrong",
                        "detail": f"DD: neither `{w1}` nor `{w2}` maps to "
                                  f"{answer_clean} in DB",
                    })
            else:
                checks.append({
                    "check": "dd",
                    "status": "unverifiable",
                    "detail": "DD format did not yield two 'window = ANSWER' pairs",
                })

        # --- CHECK 4c: Cryptic Definition ---
        # A CD has no wordplay mechanism. The only way to verify it is a
        # direct DB lookup: either the whole clue text, or the extracted
        # definition, maps to the answer in definition_answers_augmented
        # (or synonyms_pairs, via definition_matches which checks both).
        if wtype == "cryptic_definition":
            cd_clue_ok = self.definition_matches(clue_text, answer)
            cd_def_ok = bool(definition) and self.definition_matches(definition, answer)
            if cd_clue_ok or cd_def_ok:
                parts = []
                if cd_clue_ok:
                    parts.append("clue text maps to answer in DB")
                if cd_def_ok:
                    parts.append("definition maps to answer in DB")
                checks.append({
                    "check": "cd",
                    "status": "verified",
                    "detail": "; ".join(parts),
                })
            else:
                checks.append({
                    "check": "cd",
                    "status": "unverifiable",
                    "detail": "neither clue text nor definition maps to answer "
                              "in DB — CD cannot be verified",
                })

        # --- CHECK 4d: Homophone ---
        # The explanation asserts the answer sounds like another word. The
        # homophone relationship must exist in the `homophones` DB table
        # (either direction). Piece-level sub-homophones (e.g. CENS sounds
        # like SENSE within a charade) are also handled.
        if wtype == "homophone" or "sounds like" in expl.lower():
            # Primary pattern: `ANSWER sounds like WORD` (WORD may be bare
            # or in quotes)
            homophone_matches = re.findall(
                r"([A-Z]+)\s+sounds\s+like\s+[\"']?([A-Z]+)[\"']?",
                expl, re.IGNORECASE,
            )
            for answer_claim, sound_alike in homophone_matches:
                answer_claim_upper = answer_claim.upper()
                sound_alike_upper = sound_alike.upper()
                if answer_claim_upper == sound_alike_upper:
                    continue
                ok = self.is_homophone(answer_claim, sound_alike)
                checks.append({
                    "check": "homophone",
                    "status": "verified" if ok else "unverifiable",
                    "detail": f"'{answer_claim}' sounds like "
                              f"'{sound_alike}': "
                              f"{'in DB' if ok else 'not in DB'}",
                })
            if not homophone_matches and wtype == "homophone":
                checks.append({
                    "check": "homophone",
                    "status": "unverifiable",
                    "detail": "homophone explanation missing "
                              "'ANSWER sounds like WORD' form",
                })

        # --- CHECK 4e: Spoonerism ---
        # A Spoonerism swaps the initial consonant clusters of two words.
        # Mechanical letter-level verification: take the first two piece
        # words from the explanation, swap their initial consonant runs,
        # concatenate, and compare to the answer. This is strict and will
        # miss phonetic Spoonerisms where silent letters make the letter
        # count differ (e.g. FOE+NEAR -> NOE+FEAR ≠ NOFEAR) — those are
        # emitted as unverifiable for human review.
        if wtype == "spoonerism" or "spooner" in expl.lower():
            pieces_in_expl = re.findall(r"\b([A-Z]+)\s*\(", expl)
            if len(pieces_in_expl) >= 2:
                w1, w2 = pieces_in_expl[0], pieces_in_expl[1]

                def _initial_cluster(w):
                    m = re.match(r"([^AEIOU]+)", w)
                    return m.group(1) if m else ""

                c1 = _initial_cluster(w1)
                c2 = _initial_cluster(w2)
                swapped = (c2 + w1[len(c1):]) + (c1 + w2[len(c2):])
                if swapped == answer_clean:
                    checks.append({
                        "check": "spoonerism",
                        "status": "verified",
                        "detail": f"swap {w1}+{w2} -> {c2}{w1[len(c1):]}+"
                                  f"{c1}{w2[len(c2):]} = {answer_clean}",
                    })
                else:
                    checks.append({
                        "check": "spoonerism",
                        "status": "unverifiable",
                        "detail": f"mechanical swap of {w1}+{w2} gives "
                                  f"'{swapped}', answer is '{answer_clean}' "
                                  f"(likely phonetic Spoonerism)",
                    })
            else:
                checks.append({
                    "check": "spoonerism",
                    "status": "unverifiable",
                    "detail": "Spoonerism: fewer than two piece words found",
                })

        if wtype == "anagram" or "[anagram" in expl.lower():
            # Extract all uppercase letter groups between "anagram of" and "=".
            # Single-letter pieces (e.g. B from "B (abbreviation=\"black\")")
            # are included; their abbreviation source is validated separately
            # by the abbreviation check, and source-in-clue confirms the
            # source word is actually in the clue. The anagram check here
            # just needs the full letter count.
            # Greedy match: stop at the LAST `=` in the explanation (the one
            # before the answer), not any intermediate `=` inside annotations
            # like `abbreviation="..."`. This is important for multi-piece
            # fodder like `OTTER + L (abbreviation="large") + WEIR`.
            ana_section = re.search(r"anagram\s+(?:of\s+)?(.+)\s*=", expl)
            if ana_section:
                fodder_parts = re.findall(r"\b[A-Z]+\b", ana_section.group(1))
                fodder = "".join(fodder_parts)
            else:
                # Fallback: try original single-capture pattern
                ana_match = re.search(
                    r"anagram\s+(?:of\s+)?([A-Z\s+]+?)(?:\s*=|\s*anagrammed|\s*\[|\s*;|\s*\()",
                    expl, re.IGNORECASE,
                )
                fodder = ana_match.group(1).strip() if ana_match else ""
            if fodder:
                ana_ok = self.check_anagram(fodder, answer)
                checks.append({
                    "check": "anagram",
                    "status": "verified" if ana_ok else "wrong",
                    "detail": f"'{fodder}' anagrams to {answer_clean}: {'YES' if ana_ok else 'NO'}",
                })

                # --- Fodder provenance: verify each fodder word appears in the clue ---
                # For multi-word fodder (e.g. "anagram of BARON + STUDIES"), check that
                # each word appears in the clue text.  Single-letter pieces from
                # abbreviations (e.g. "N (abbreviation='name')") are trusted only when
                # the abbreviation itself is verified via DB -- that check already
                # happens above in CHECK 2 and earns its own +8.  Here we only check
                # words of 2+ letters from the fodder.
                if ana_ok and clue_text and fodder_parts:
                    # Normalise both sides: strip everything that isn't a letter,
                    # so "BRIDES" matches "bride's", "GT" matches "G&T", etc.
                    clue_norm = re.sub(r"[^a-z]", "", clue_text.lower())
                    all_found = True
                    for fp in fodder_parts:
                        fp_norm = re.sub(r"[^a-z]", "", fp.lower())
                        if fp_norm and fp_norm not in clue_norm:
                            all_found = False
                            break
                    if all_found:
                        checks.append({
                            "check": "fodder_provenance",
                            "status": "verified",
                            "detail": f"all fodder words {fodder_parts} found in clue text",
                        })

        is_hidden_reversed = "hidden reversed" in expl.lower() or wtype == "hidden_reversed"
        if not is_hidden_reversed and (wtype == "reversal" or "[reversal" in expl.lower()):
            rev_match = re.search(
                r"(\w+)\s*(?:reversed|reversal|backwards|reflected)",
                expl, re.IGNORECASE,
            )
            if not rev_match:
                rev_match = re.search(
                    r"reverse of (\w+)",
                    expl, re.IGNORECASE,
                )
            if rev_match:
                source = rev_match.group(1)
                rev_ok = self.check_reversal(source, answer)
                if rev_ok:
                    # Full answer is a reversal of one word
                    checks.append({
                        "check": "reversal",
                        "status": "verified",
                        "detail": f"'{source}' reversed = {answer_clean}: YES",
                    })
                else:
                    # Check if reversal is a piece within a charade.
                    # To avoid false positives, we require BOTH:
                    #   1. The reversed letters appear in the answer
                    #   2. The reversed letters appear as a claimed piece
                    #      in the explanation (uppercase before parenthesis)
                    source_clean = re.sub(r"[^A-Z]", "", source.upper())
                    reversed_piece = source_clean[::-1]
                    claimed_pieces = re.findall(r"\b([A-Z]{1,})\s*\(", expl)
                    if reversed_piece in answer_clean and reversed_piece in claimed_pieces:
                        checks.append({
                            "check": "reversal",
                            "status": "verified",
                            "detail": f"'{source}' reversed = {reversed_piece}, claimed piece in {answer_clean}: YES",
                        })
                    else:
                        checks.append({
                            "check": "reversal",
                            "status": "wrong",
                            "detail": f"'{source}' reversed = {reversed_piece}, not a valid piece in {answer_clean}: NO",
                        })

        if "[container:" in expl.lower() or "container" in wtype:
            # Extract uppercase pieces — from before bracket if bracket format,
            # otherwise from the whole explanation
            if "[container:" in expl.lower():
                search_text = expl[:expl.lower().index("[container:")]
            else:
                search_text = expl.split(";")[0] if ";" in expl else expl
            container_pieces = re.findall(r"\b([A-Z]+)\s*\(", search_text)
            if len(container_pieces) >= 2:
                cont_ok, inner, outer = self.check_container(container_pieces, answer)
                if cont_ok:
                    checks.append({
                        "check": "container",
                        "status": "verified",
                        "detail": f"{inner} inside {outer} = {answer_clean}: YES",
                    })
                # No penalty on failure — complex assembly may not parse simply

        # --- CHECK 5: Positional extraction claims ---
        # Covers: first/last/middle/initial/final letter(s); odd/even/alternating letters.
        # Verifies the claimed letters actually match the positional extraction of
        # the source word or phrase. Fabricated positional claims earn "wrong".
        positional_kinds = [
            ("first",     r"first\s+letters?"),
            ("last",      r"last\s+letters?"),
            ("middle",    r"middle\s+letters?"),
            ("outer",     r"outer\s+letters?"),
            ("initial",   r"initials?(?:\s+letter)?"),
            ("final",     r"finals?(?:\s+letter)?"),
            ("odd",       r"odd\s+letters?"),
            ("even",      r"even\s+letters?"),
            ("alternate", r"alternat(?:ing|e)\s+letters?"),
        ]
        seen_positional = set()
        for kind, phrase_re in positional_kinds:
            pat = re.compile(
                r"(\w+)\s*\(\s*" + phrase_re + r"\s+(?:of|in|from)\s+"
                r"[\"']?([^\")]+?)[\"']?\s*\)",
                re.IGNORECASE,
            )
            for letters, source in pat.findall(expl):
                key = (kind, letters.upper(), source.strip().lower())
                if key in seen_positional:
                    continue
                seen_positional.add(key)
                result = self.check_positional(kind, letters, source)
                if result is None:
                    continue
                checks.append({
                    "check": "positional",
                    "status": "verified" if result else "wrong",
                    "detail": f"'{letters}' as {kind} of '{source.strip()}': "
                              f"{'YES' if result else 'NO'}",
                })

        # --- CHECK 5a2: Reversal-source piece claims ---
        # Format: LETTERS (reversal of "source") — the piece comes from
        # reversing some synonym/abbreviation/literal of source.
        seen_rev_src = set()
        rev_src_pat = re.compile(
            r"(\w+)\s*\(\s*(?:reversal|reverse)\s+of\s+[\"']?([^\")]+?)[\"']?\s*\)",
            re.IGNORECASE,
        )
        for letters, source in rev_src_pat.findall(expl):
            key = (letters.upper(), source.strip().lower())
            if key in seen_rev_src:
                continue
            seen_rev_src.add(key)
            ok = self.check_reversal_source(letters, source)
            if ok is None:
                continue
            checks.append({
                "check": "reversal_source",
                "status": "verified" if ok else "wrong",
                "detail": f"'{letters}' as reversal of '{source.strip()}': "
                          f"{'YES' if ok else 'NO'}",
            })

        # --- CHECK 5b: Deletion claims ---
        # Format: LETTERS (deletion="source")
        deletion_claims = re.findall(
            r"(\w+)\s*\(\s*deletion\s*=\s*\"([^\"]+)\"[^)]*\)",
            expl, re.IGNORECASE,
        )
        for letters, source_word in deletion_claims:
            letters_clean = re.sub(r"[^A-Z]", "", letters.upper())
            source_clean = re.sub(r"[^A-Z]", "", source_word.upper())
            if source_clean and letters_clean:
                del_ok = self.check_deletion_mechanism(source_clean, letters_clean)
                verdict = ("valid head/tail/outer/heart deletion"
                           if del_ok else "not a recognised deletion mechanism")
                checks.append({
                    "check": "deletion",
                    "status": "verified" if del_ok else "wrong",
                    "detail": f"'{letters}' from '{source_word}': {verdict}",
                })

        # --- CHECK 5c: "X minus Y = Z" deletion format ---
        # Format: OLIVER (synonym="boy wanting more") minus R (river) = OLIVE
        minus_match = re.search(
            r"([A-Z]+)\s*\([^)]+\)\s+minus\s+([A-Z]+)\s*\([^)]+\)\s*=\s*([A-Z]+)",
            expl,
        )
        if minus_match:
            source_letters = minus_match.group(1)
            removed_letters = minus_match.group(2)
            result_letters = minus_match.group(3)
            # Verify: removing the letters from source gives the result
            remaining = source_letters
            for ch in removed_letters:
                pos = remaining.find(ch)
                if pos >= 0:
                    remaining = remaining[:pos] + remaining[pos+1:]
            del_ok = remaining == result_letters and result_letters == answer_clean
            checks.append({
                "check": "assembly",
                "status": "verified" if del_ok else "wrong",
                "detail": f"{source_letters} minus {removed_letters} = {result_letters}: {'MATCH' if del_ok else 'MISMATCH'}",
            })

        # --- CHECK 5d: "X with deletion = Y" format ---
        # Format: ATONED (synonym="made up") with deletion = TONED
        with_del_match = re.search(
            r"([A-Z]+)\s*\([^)]+\)\s+with\s+deletion\s*=\s*([A-Z]+)",
            expl,
        )
        if with_del_match:
            source_letters = with_del_match.group(1)
            result_letters = with_del_match.group(2)
            # Verify: result is source with head, tail, or outer letters removed
            del_ok = (
                source_letters[1:] == result_letters or           # head removed
                source_letters[:-1] == result_letters or          # tail removed
                (len(source_letters) >= 4 and
                 source_letters[1:-1] == result_letters) or       # outer removed
                result_letters in source_letters                   # substring
            )
            if del_ok and result_letters == answer_clean:
                checks.append({
                    "check": "assembly",
                    "status": "verified",
                    "detail": f"{source_letters} with deletion = {result_letters}: MATCH",
                })

        # --- CHECK 5e: Silent-piece catch ---
        # Every piece annotation `WORD (content)` must have content matching a
        # known verification pattern. If the content is free-form narrative or
        # an unsupported mechanism (e.g. "(anagram of 'x')", "(LO reversed)"),
        # the piece is unverified and cannot carry an explanation to HIGH.
        _valid_content_prefix = re.compile(
            r"^\s*(?:"
            r"synonym\s+of\s+[\"']"
            r"|synonym\s*=\s*\""
            r"|abbr\.?\s*(?:of\s+)?"
            r"|abbreviation\s*=\s*\""
            r"|(?:first|last|middle|outer|initial|final|odd|even|alternat(?:e|ing))"
            r"\s+letters?\s+(?:of|in|from)\s+"
            r"|(?:reversal|reverse)\s+of\s+"
            r"|from\s+clue\b"
            r"|deletion\s*=\s*\""
            r")",
            re.IGNORECASE,
        )
        _seen_silent = set()
        for m in re.finditer(r"\b(\w+)\s*\(([^)]*)\)", expl):
            word, content = m.group(1), m.group(2)
            content_stripped = content.strip()
            # Skip inline assembly visualizations — pure uppercase letters
            if content_stripped and re.fullmatch(r"[A-Z\s]+", content_stripped):
                continue
            # Skip empty parens
            if not content_stripped:
                continue
            # Skip if content starts with a known verification keyword
            if _valid_content_prefix.match(content_stripped):
                continue
            key = (word.upper(), content_stripped[:50])
            if key in _seen_silent:
                continue
            _seen_silent.add(key)
            checks.append({
                "check": "silent_piece",
                "status": "unverifiable",
                "detail": f"piece '{word}' has unrecognised source claim: "
                          f"'{content_stripped[:60]}'",
            })

        # --- CHECK 6: Trivial explanation detection ---
        # If the explanation is just "ANSWER(synonym of definition)" with no wordplay,
        # it's not a real explanation — it just restates the definition
        syn_pieces = re.findall(
            r"(\w+)\s*\(\s*synonym\s+of\s+[\"']([^\"']+)[\"']\s*\)",
            expl, re.IGNORECASE,
        )
        if syn_pieces and len(syn_pieces) == 1:
            result_word, source_word = syn_pieces[0]
            if result_word.upper().replace(" ", "") == answer_clean:
                checks.append({
                    "check": "trivial",
                    "status": "wrong",
                    "detail": f"explanation just restates answer as synonym of definition",
                })

        # --- CHECK 7: Indicator verification (operation-level only) ---
        # Piece-level positional mechanisms (first/last/middle letter) are
        # mechanically verified by check_positional() and no longer need
        # an indicator-DB match. Operation-level checks still apply.
        OPERATION_INDICATOR_REQUIREMENTS = {
            "anagram":          ("anagram", None),
            "reversal":         ("reversal", None),
            "container":        ("container", None),
            "deletion":         ("deletion", None),
            # hidden/hidden_reversed use span-naming convention for verification —
            # no indicator-DB check required.
            "homophone":        ("homophone", None),
        }

        # Check operation-level indicator: the explanation MUST name a
        # known indicator word for each operation that requires one. Just
        # having an indicator somewhere in the clue isn't enough — the
        # parse must show which word does the work, otherwise the
        # mechanism is hidden from the user.
        #
        # wordplay_type can be a comma-separated list (e.g. "container,
        # reversal") for compound parses; check each component.
        wtypes_raw = (wordplay_type or "").lower()
        wtypes_list = [w.strip().replace(" ", "_")
                        for w in wtypes_raw.split(",") if w.strip()]

        # Also flag piece-level reversal inside a non-reversal parse:
        # "X (reversal of "Y")" inside the explanation means a reversal
        # mechanism is in play even if wordplay_type is "container".
        if "reversal of" in (expl or "").lower() and "reversal" not in wtypes_list:
            wtypes_list.append("reversal")

        seen_op_checks = set()
        for wtype in wtypes_list:
            if wtype not in OPERATION_INDICATOR_REQUIREMENTS:
                continue
            req_type, _req_subtypes = OPERATION_INDICATOR_REQUIREMENTS[wtype]
            if req_type in seen_op_checks:
                continue
            seen_op_checks.add(req_type)
            status, word = self.explanation_names_indicator(expl, req_type)
            if status == 'verified':
                checks.append({
                    "check": "indicator",
                    "status": "verified",
                    "detail": f"'{word}' is a known {req_type} indicator",
                })
            elif status == 'wrong':
                checks.append({
                    "check": "indicator",
                    "status": "wrong",
                    "detail": f"'{word}' is not a known {req_type} indicator",
                })
            else:  # missing
                # Fallback: explanation didn't NAME an indicator, but does
                # the clue have one? Mark as 'unverifiable' (mechanism
                # hidden) rather than 'wrong' if at least the clue contains
                # an indicator. If the clue has no indicator at all, that's
                # 'wrong'.
                clue_has = self.clue_has_indicator(clue_text, req_type)
                if not clue_has and req_type == "container":
                    clue_has = self.clue_has_indicator(clue_text, "insertion")
                if clue_has:
                    checks.append({
                        "check": "indicator",
                        "status": "unverifiable",
                        "detail": f"explanation does not name the {req_type} "
                                  f"indicator (mechanism hidden from user)",
                    })
                else:
                    checks.append({
                        "check": "indicator",
                        "status": "wrong",
                        "detail": f"no '{req_type}' indicator found in clue "
                                  f"for {wtype} operation",
                    })

        # --- CHECK 7b: Clue-side indicator scan (mechanism-hiding catch) ---
        # The check above only fires for operations DECLARED in wordplay_type
        # or surfaced by a "reversal of" string in the explanation. A parse
        # that hides the mechanism completely (e.g. mechanical_v1 storing
        # `wordplay_type="charade"` with no reversal mention, despite the
        # clue containing "sent over") slips through with no demerit.
        #
        # This check independently scans the CLUE for indicators of each
        # operation type. If the clue has e.g. a reversal indicator and
        # the explanation neither declares the reversal in wordplay_type
        # nor names the indicator, score WRONG — the explanation is
        # hiding a mechanism the clue plainly contains.
        #
        # Honours the standing rule (2026-05-01): "verifier cannot ignore
        # an indicator present in the clue". Without this scan, the
        # earlier rule only covered the case where the explanation leaked
        # the operation accidentally; here we catch the case where the
        # explanation hides it on purpose.
        # Build the set of clue words and 2-word phrases (used below).
        _clue_words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?",
                                 (clue_text or "").lower())
        _clue_phrases = [_clue_words[i] + " " + _clue_words[i + 1]
                         for i in range(len(_clue_words) - 1)]
        _expl_lower = (expl or "").lower()

        for op_type in OPERATION_INDICATOR_REQUIREMENTS:
            req_type, _ = OPERATION_INDICATOR_REQUIREMENTS[op_type]
            if req_type in seen_op_checks:
                continue  # already addressed by check 7a above

            # Find specific clue word(s)/phrase(s) that are indicators
            # for this op type per the indicators DB.
            indicator_tokens = []
            for w in _clue_words:
                for wt, _st in self._indicators_by_word.get(w, []):
                    if wt == req_type or (req_type == "container"
                                          and wt == "insertion"):
                        indicator_tokens.append(w)
                        break
            for ph in _clue_phrases:
                for wt, _st in self._indicators_by_word.get(ph, []):
                    if wt == req_type or (req_type == "container"
                                          and wt == "insertion"):
                        indicator_tokens.append(ph)
                        break

            if not indicator_tokens:
                continue

            # If any of those exact words/phrases appear ANYWHERE in the
            # explanation, the parse has already addressed them (possibly
            # under a different op-type — ambiguous indicators like "over"
            # can be anagram OR reversal). Don't double-fault.
            if any(tok in _expl_lower for tok in indicator_tokens):
                continue

            # The clue has an indicator for this op-type, no other
            # op-type in the parse used it, and it's invisible in the
            # explanation. Mechanism hidden — flag WRONG.
            checks.append({
                "check": "indicator",
                "status": "wrong",
                "detail": f"clue contains a {req_type} indicator "
                          f"({indicator_tokens[0]!r}) but the parse "
                          f"hides this mechanism",
            })
            seen_op_checks.add(req_type)

        # Piece-level positional indicator checks have been removed.
        # `check_positional()` mechanically verifies that e.g. X really is
        # the last letter of Y — the indicator-DB check was redundant and
        # produced false negatives when the indicator DB lacked common
        # words like "outskirts of". Mechanical truth is self-verifying;
        # requiring an additional DB match was the mirror-trick problem.

        # --- CHECK 8: Word coverage ---
        # Classify every word in the clue by the role the explanation
        # claims for it. Words may be claimed as: definition, synonym
        # source, abbreviation source, positional source, reversal
        # source, deletion source, indicator, anagram fodder, hidden
        # source, dbe_marker. Whatever is left over is then checked
        # against the LINK_WORDS allow-list — leftover-AND-in-list →
        # link; leftover-and-not-in-list → unaccounted.
        # Per the user brief: every word must have a use. Unaccounted
        # words mark the explanation as incomplete and the score
        # reflects that via the wrong-branch penalty in the scoring
        # section (see -15 per word, capped at -50).
        try:
            coverage = self._classify_clue_words(
                clue_text, expl, definition)
            unaccounted = coverage["unaccounted"]
            checks.append({
                "check": "word_coverage",
                "status": "wrong" if unaccounted else "verified",
                "detail": coverage["summary"],
                "classified": coverage["classified"],
                "unaccounted": unaccounted,
            })
        except Exception:
            pass  # Diagnostic check must never disrupt verification

        # --- SCORING ---
        # Three states: VERIFIED (proven correct), UNVERIFIABLE (not in DB),
        # WRONG (proven incorrect by mechanical check)
        #
        # Philosophy: EARN your score. Unverified claims don't help you.
        #   VERIFIED = points earned (you proved your claim)
        #   UNVERIFIABLE = nothing (you haven't proven anything)
        #   WRONG = heavy penalty (you made a false claim)
        if not checks:
            return {"score": 20, "checks": [], "verdict": "LOW",
                    "detail": "No verifiable claims found",
                    "verified": 0, "unverifiable": 0, "wrong": 0}

        verified = sum(1 for c in checks if c["status"] == "verified")
        unverifiable = sum(1 for c in checks if c["status"] == "unverifiable")
        wrong = sum(1 for c in checks if c["status"] == "wrong")

        # Hybrid scoring:
        # - Mechanical checks (assembly, hidden, anagram, reversal, first_letter)
        #   are 100% reliable — high value verified, fatal if wrong
        # - DB checks (synonym, definition, abbreviation) have coverage gaps
        #   — moderate value verified, small penalty if unverifiable
        #
        # Start at 40 (baseline for having an explanation at all)
        # Earn up to 100, penalised down to 0
        score = 40

        # Assembly is only trustworthy if the pieces themselves are trustworthy
        # If any piece source is WRONG, assembly verification means nothing
        has_wrong = any(c["status"] == "wrong" for c in checks)

        for c in checks:
            if c["status"] == "verified":
                if c["check"] == "assembly":
                    if has_wrong:
                        score += 0  # Assembly passes but pieces are wrong — don't reward
                    else:
                        score += 25  # Pieces make the answer — strong
                elif c["check"] == "hidden_word":
                    if has_wrong:
                        score += 0
                    else:
                        score += 35  # Hidden substring is irrefutable — strongest mechanical proof
                elif c["check"] in ("anagram", "reversal", "container"):
                    if has_wrong:
                        score += 0
                    else:
                        score += 25  # Mechanism works — strong
                elif c["check"] == "fodder_provenance":
                    if has_wrong:
                        score += 0
                    else:
                        score += 10  # All fodder words confirmed in clue text
                elif c["check"] == "dd":
                    if has_wrong:
                        score += 0
                    else:
                        score += 40  # Both DD windows map to answer + clean remainder
                elif c["check"] == "cd":
                    if has_wrong:
                        score += 0
                    else:
                        score += 40  # CD: clue or definition maps to answer in DB
                elif c["check"] == "homophone":
                    if has_wrong:
                        score += 0
                    else:
                        score += 35  # Homophone pair in DB — mechanical proof
                elif c["check"] == "spoonerism":
                    if has_wrong:
                        score += 0
                    else:
                        score += 30  # Mechanical letter-level swap matches answer
                elif c["check"] == "definition":
                    score += 15  # Definition confirmed
                elif c["check"] == "synonym":
                    score += 10  # Synonym confirmed
                elif c["check"] == "abbreviation":
                    score += 8   # Abbreviation confirmed
                elif c["check"] == "first_letter":
                    score += 8   # First letter confirmed (legacy narrow check)
                elif c["check"] == "positional":
                    score += 10  # Positional extraction confirmed (mechanical)
                elif c["check"] == "deletion":
                    score += 8   # Deletion confirmed
                elif c["check"] == "indicator":
                    score += 10  # Operation indicator confirmed (named + valid)
                elif c["check"] == "reversal_source":
                    score += 10  # Reversal source verified via DB lookup
            elif c["status"] == "wrong":
                if c["check"] == "assembly":
                    score -= 50  # Pieces don't make the answer — fatal
                elif c["check"] in ("hidden_word", "anagram", "reversal"):
                    score -= 50  # Mechanism doesn't work — fatal
                elif c["check"] == "deletion":
                    score -= 30  # Trivially verifiable, no excuse
                elif c["check"] == "first_letter":
                    score -= 30  # Trivially verifiable, no excuse
                elif c["check"] == "positional":
                    score -= 30  # Fabricated positional claim — mechanically disprovable
                elif c["check"] == "abbreviation":
                    score -= 30  # Abbreviation not in DB — likely fabricated
                elif c["check"] == "trivial":
                    score -= 40  # Just restating the definition, not an explanation
                elif c["check"] == "indicator":
                    score -= 50  # Operation claimed without indicator — fatal
                elif c["check"] == "no_definition":
                    score -= 30  # Every cryptic clue must have a definition
                elif c["check"] == "synonym":
                    if "single-letter" in c.get("detail", ""):
                        score -= 30  # Fabricated single-letter abbreviation disguised as synonym
                    else:
                        score -= 5   # Could be DB gap
                elif c["check"] == "definition":
                    score -= 5   # Could be DB gap
                elif c["check"] == "dd":
                    score -= 40  # DD claimed but neither window maps to answer — likely bogus
                elif c["check"] == "source_not_in_clue":
                    score -= 30  # Phantom source — piece claims a word the clue doesn't use
                elif c["check"] == "word_coverage":
                    # Every clue word must have an assigned use per the brief.
                    # Penalty: -25 per unaccounted word, no cap. The score
                    # floor at 0 takes care of clues with many missing words.
                    # Calibrated so 1 unaccounted word in a HIGH-scoring
                    # explanation drops it below the HIGH threshold (70):
                    # 85→60 = MEDIUM, 100→75 = still HIGH only if otherwise
                    # immaculate.
                    n = len(c.get("unaccounted", []))
                    if n > 0:
                        score -= 25 * n
            elif c["status"] == "unverifiable":
                # A piece that the verifier tried to check against the DB and could
                # not confirm is not an explained piece — it's a gap. Penalise so
                # that unverifiable claims cannot ride through to HIGH on the back
                # of mechanically-verified assembly alone.
                if c["check"] == "synonym":
                    score -= 30
                elif c["check"] == "definition":
                    score -= 10
                elif c["check"] == "abbreviation":
                    score -= 30
                elif c["check"] == "silent_piece":
                    score -= 30  # Piece annotation doesn't match any known verification pattern
                elif c["check"] == "dd":
                    score -= 10  # DD partial or malformed — not both windows verified
                elif c["check"] == "cd":
                    score -= 10  # CD can't be verified — whole clue / definition not in DB
                elif c["check"] == "homophone":
                    score -= 20  # Homophone pair not in DB — claim unverified
                elif c["check"] == "spoonerism":
                    score -= 10  # Spoonerism letter-swap doesn't match — likely phonetic
                elif c["check"] == "indicator":
                    # Mechanism hidden from user — explanation didn't name
                    # the licensing indicator. Significant penalty so the
                    # parse cannot ride to HIGH (or even MEDIUM) on
                    # assembly alone.
                    score -= 30

        score = min(100, max(0, score))

        if score >= 70:
            verdict = "HIGH"
        elif score >= 50:
            verdict = "MEDIUM"
        elif score >= 30:
            verdict = "LOW"
        else:
            verdict = "FAIL"

        # MEDIUM requires at least one verified piece beyond just the definition.
        # An explanation with only a definition match and no wordplay evidence
        # is not a real explanation. Exception: mechanical DDs are trusted
        # (handled by caller passing mechanical_dd=True).
        if verdict == "MEDIUM":
            wordplay_checks = [c for c in checks
                               if c["status"] == "verified"
                               and c["check"] not in ("definition",)]
            if not wordplay_checks:
                verdict = "LOW"

        return {
            "score": score,
            "checks": checks,
            "verified": verified,
            "unverifiable": unverifiable,
            "wrong": wrong,
            "verdict": verdict,
        }


def verify_puzzle(source, puzzle_number):
    """Verify all explanations for a puzzle and print results."""
    clues_conn = sqlite3.connect(CLUES_DB)
    clues_conn.row_factory = sqlite3.Row

    rows = clues_conn.execute("""
        SELECT clue_number, direction, clue_text, answer, definition,
               wordplay_type, ai_explanation
        FROM clues
        WHERE source = ? AND puzzle_number = ?
        ORDER BY direction, CAST(clue_number AS INTEGER)
    """, (source, str(puzzle_number))).fetchall()

    verifier = ExplanationVerifier()

    print(f"\n{'='*70}")
    print(f"VERIFICATION: {source} #{puzzle_number} ({len(rows)} clues)")
    print(f"{'='*70}\n")

    verdicts = Counter()
    for r in rows:
        result = verifier.verify(
            r["clue_text"], r["answer"], r["definition"],
            r["wordplay_type"], r["ai_explanation"],
        )
        verdicts[result["verdict"]] += 1

        status = result["verdict"]
        score = result["score"]
        v = result.get("verified", 0)
        u = result.get("unverifiable", 0)
        w = result.get("wrong", 0)
        checks_str = ""
        for c in result.get("checks", []):
            if c["status"] == "verified":
                mark = "+"
            elif c["status"] == "wrong":
                mark = "X"
            else:
                mark = "?"
            checks_str += f"\n      [{mark}] {c['check']}: {c['detail']}"

        print(f"{r['clue_number']:>3s}{r['direction'][0]} [{status:6s} {score:3d}] {r['answer'] or '?':15s} | {r['clue_text'][:45]}")
        print(f"      verified={v} unverifiable={u} WRONG={w}")
        if checks_str:
            print(checks_str)
        print()

    print(f"\n{'='*70}")
    print(f"SUMMARY: HIGH={verdicts['HIGH']} MEDIUM={verdicts['MEDIUM']} "
          f"LOW={verdicts['LOW']} FAIL={verdicts['FAIL']}")
    print(f"{'='*70}")

    clues_conn.close()


if __name__ == "__main__":
    import sys
    if len(sys.argv) >= 3:
        verify_puzzle(sys.argv[1], sys.argv[2])
    else:
        print("Usage: python verify_explanation.py <source> <puzzle_number>")
        print("Example: python verify_explanation.py times 29495")
