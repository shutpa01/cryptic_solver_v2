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

        if "hidden" in wtype or "hidden in" in expl.lower() or "hidden reversed" in expl.lower():
            hidden_ok = self.check_hidden(clue_text, answer)
            checks.append({
                "check": "hidden_word",
                "status": "verified" if hidden_ok else "wrong",
                "detail": f"'{answer_clean}' hidden in clue: {'YES' if hidden_ok else 'NO'}",
            })

        if wtype == "anagram" or "[anagram" in expl.lower():
            # Extract all uppercase letter groups between "anagram of" and "="
            ana_section = re.search(r"anagram\s+(?:of\s+)?(.+?)\s*=", expl)
            if ana_section:
                fodder_parts = re.findall(r"[A-Z]{2,}", ana_section.group(1))
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

        # --- CHECK 5: First letter claims ---
        first_letter_claims = re.findall(
            r"(\w)\s*\(\s*first\s+letter\s+of\s+[\"']?(\w+)",
            expl, re.IGNORECASE,
        )
        for letter, word in first_letter_claims:
            fl_ok = word.upper().startswith(letter.upper())
            checks.append({
                "check": "first_letter",
                "status": "verified" if fl_ok else "wrong",
                "detail": f"first of '{word}' = {letter}: {'YES' if fl_ok else 'NO'}",
            })

        # --- CHECK 5b: Deletion claims ---
        # Format: LETTERS (deletion="source")
        deletion_claims = re.findall(
            r"(\w+)\s*\(\s*deletion\s*=\s*\"([^\"]+)\"\s*\)",
            expl, re.IGNORECASE,
        )
        for letters, source_word in deletion_claims:
            letters_clean = re.sub(r"[^A-Z]", "", letters.upper())
            source_clean = re.sub(r"[^A-Z]", "", source_word.upper())
            if source_clean and letters_clean:
                del_ok = (letters_clean in source_clean or
                          source_clean.startswith(letters_clean) or
                          source_clean.endswith(letters_clean))
                checks.append({
                    "check": "deletion",
                    "status": "verified" if del_ok else "wrong",
                    "detail": f"'{letters}' from '{source_word}': {'YES' if del_ok else 'NOT a substring'}",
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

        # --- CHECK 7: Indicator verification ---
        # For operations that require indicators, verify an indicator word
        # exists in the clue text. Without an indicator, the claimed operation
        # is unjustified and the explanation cannot be trusted.
        #
        # Mechanisms needing indicators:
        MECHANISM_INDICATOR_REQUIREMENTS = {
            "reversal":     ("reversal", None),
            "deletion":     ("deletion", None),
            "hidden":       ("hidden", None),
            "sound_of":     ("homophone", None),
            "first_letter": ("parts", {"first_use", "first_delete"}),
            "last_letter":  ("parts", {"last_use", "last_delete", "last", "last letter",
                                       "tail_delete"}),
        }
        # Also check the overall wordplay_type for operation-level indicators
        OPERATION_INDICATOR_REQUIREMENTS = {
            "anagram":          ("anagram", None),
            "reversal":         ("reversal", None),
            "container":        ("container", None),
            "deletion":         ("deletion", None),
            "hidden":           ("hidden", None),
            "hidden_reversed":  ("hidden", None),
            "homophone":        ("homophone", None),
        }

        # Check operation-level indicator
        wtype = (wordplay_type or "").lower().replace(" ", "_")
        if wtype in OPERATION_INDICATOR_REQUIREMENTS:
            req_type, req_subtypes = OPERATION_INDICATOR_REQUIREMENTS[wtype]
            if not self.clue_has_indicator(clue_text, req_type, req_subtypes):
                # Also check 'insertion' for container
                found = False
                if req_type == "container":
                    found = self.clue_has_indicator(clue_text, "insertion")
                if not found:
                    checks.append({
                        "check": "indicator",
                        "status": "wrong",
                        "detail": f"no '{req_type}' indicator found in clue for {wtype} operation",
                    })

        # Check piece-level indicators from explanation text
        # Look for mechanism claims like "last letter of", "reversed"
        last_letter_claims = re.findall(
            r"(\w)\s*\(\s*last\s+letter\s+of\s+[\"']?(\w+)",
            expl, re.IGNORECASE,
        )
        for letter, word in last_letter_claims:
            req_type, req_subtypes = MECHANISM_INDICATOR_REQUIREMENTS["last_letter"]
            if not self.clue_has_indicator(clue_text, req_type, req_subtypes):
                checks.append({
                    "check": "indicator",
                    "status": "wrong",
                    "detail": f"no 'last letter' indicator in clue for '{word}' -> {letter}",
                })

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
                elif c["check"] in ("hidden_word", "anagram", "reversal", "container"):
                    if has_wrong:
                        score += 0
                    else:
                        score += 25  # Mechanism works — strong
                elif c["check"] == "definition":
                    score += 15  # Definition confirmed
                elif c["check"] == "synonym":
                    score += 10  # Synonym confirmed
                elif c["check"] == "abbreviation":
                    score += 8   # Abbreviation confirmed
                elif c["check"] == "first_letter":
                    score += 8   # First letter confirmed
                elif c["check"] == "deletion":
                    score += 8   # Deletion confirmed
            elif c["status"] == "wrong":
                if c["check"] == "assembly":
                    score -= 50  # Pieces don't make the answer — fatal
                elif c["check"] in ("hidden_word", "anagram", "reversal"):
                    score -= 50  # Mechanism doesn't work — fatal
                elif c["check"] == "deletion":
                    score -= 30  # Trivially verifiable, no excuse
                elif c["check"] == "first_letter":
                    score -= 30  # Trivially verifiable, no excuse
                elif c["check"] == "abbreviation":
                    score -= 15  # Likely wrong
                elif c["check"] == "trivial":
                    score -= 40  # Just restating the definition, not an explanation
                elif c["check"] == "indicator":
                    score -= 50  # Operation claimed without indicator — fatal
                elif c["check"] == "no_definition":
                    score -= 30  # Every cryptic clue must have a definition
                elif c["check"] in ("definition", "synonym"):
                    score -= 5   # Could be DB gap
            # unverifiable = no change

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
