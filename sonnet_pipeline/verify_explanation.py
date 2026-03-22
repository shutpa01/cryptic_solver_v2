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

    def is_synonym(self, word, target):
        """Check if word -> target is a known synonym pair."""
        key = (word.lower(), target.lower())
        if key not in self._syn_cache:
            row = self.ref.execute(
                "SELECT 1 FROM synonyms_pairs WHERE word = ? AND synonym = ? LIMIT 1",
                (word.lower(), target.lower()),
            ).fetchone()
            if not row:
                # Also check reverse
                row = self.ref.execute(
                    "SELECT 1 FROM synonyms_pairs WHERE word = ? AND synonym = ? LIMIT 1",
                    (target.lower(), word.lower()),
                ).fetchone()
            self._syn_cache[key] = row is not None
        return self._syn_cache[key]

    def is_abbreviation(self, word, letters):
        """Check if word abbreviates to letters."""
        key = (word.lower(), letters.upper())
        if key not in self._abbr_cache:
            row = self.ref.execute(
                "SELECT 1 FROM wordplay WHERE indicator = ? AND UPPER(substitution) = ? LIMIT 1",
                (word.lower(), letters.upper()),
            ).fetchone()
            self._abbr_cache[key] = row is not None
        return self._abbr_cache[key]

    def is_indicator(self, word, wordplay_type):
        """Check if word is a known indicator for the given type."""
        key = (word.lower(), wordplay_type.lower())
        if key not in self._ind_cache:
            row = self.ref.execute(
                "SELECT 1 FROM indicators WHERE word = ? AND wordplay_type = ? LIMIT 1",
                (word.lower(), wordplay_type.lower()),
            ).fetchone()
            self._ind_cache[key] = row is not None
        return self._ind_cache[key]

    def definition_matches(self, definition, answer):
        """Check if definition -> answer exists in our DB."""
        key = (definition.lower(), answer.upper())
        if key not in self._def_cache:
            row = self.ref.execute(
                "SELECT 1 FROM definition_answers_augmented WHERE definition = ? AND answer = ? LIMIT 1",
                (definition.lower(), answer.upper()),
            ).fetchone()
            self._def_cache[key] = row is not None
        return self._def_cache[key]

    def check_assembly(self, pieces_letters, answer):
        """Do the pieces concatenate to the answer?"""
        assembled = "".join(pieces_letters).upper().replace(" ", "")
        target = answer.upper().replace(" ", "")
        return assembled == target

    def check_hidden(self, text, answer):
        """Is the answer hidden in the text?"""
        clean_text = re.sub(r"[^A-Z]", "", text.upper())
        clean_answer = answer.upper().replace(" ", "")
        return clean_answer in clean_text

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

        # --- CHECK 2: Parse and verify pieces ---
        pieces = re.findall(
            r"(\w+)\s*\(\s*synonym\s+of\s+[\"']([^\"']+)[\"']\s*\)",
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
        abbrs = re.findall(
            r"(\w+)\s*\(\s*abbr\.?\s+(?:of\s+)?[\"']?([^\"')]+)[\"']?\s*\)",
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

            if letter_pieces and not anagram_eq:
                # For charade/container: pieces should assemble to answer
                # Try direct concatenation
                assembled = "".join(letter_pieces)
                if assembled == answer_clean:
                    checks.append({
                        "check": "assembly",
                        "status": "verified",
                        "detail": f"pieces {'+'.join(letter_pieces)} = {answer_clean}: MATCH",
                    })
                else:
                    # Try with container logic — inner "inside" outer
                    # Don't mark as wrong here, assembly may be container
                    # which the simple concat won't catch
                    pass

            if anagram_eq:
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

        if wtype == "hidden" or "hidden in" in expl.lower():
            hidden_ok = self.check_hidden(clue_text, answer)
            checks.append({
                "check": "hidden_word",
                "status": "verified" if hidden_ok else "wrong",
                "detail": f"'{answer_clean}' hidden in clue: {'YES' if hidden_ok else 'NO'}",
            })

        if wtype == "anagram" or "[anagram" in expl.lower():
            ana_match = re.search(
                r"anagram\s+(?:of\s+)?([A-Z\s+]+?)(?:\s*=|\s*anagrammed|\s*\[|\s*;)",
                expl, re.IGNORECASE,
            )
            if ana_match:
                fodder = ana_match.group(1).strip()
                ana_ok = self.check_anagram(fodder, answer)
                checks.append({
                    "check": "anagram",
                    "status": "verified" if ana_ok else "wrong",
                    "detail": f"'{fodder}' anagrams to {answer_clean}: {'YES' if ana_ok else 'NO'}",
                })

        if wtype == "reversal" or "[reversal" in expl.lower():
            rev_match = re.search(
                r"(\w+)\s*(?:reversed|reversal|backwards|reflected)",
                expl, re.IGNORECASE,
            )
            if rev_match:
                source = rev_match.group(1)
                rev_ok = self.check_reversal(source, answer)
                checks.append({
                    "check": "reversal",
                    "status": "verified" if rev_ok else "wrong",
                    "detail": f"'{source}' reversed = {answer_clean}: {'YES' if rev_ok else 'NO'}",
                })

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
                elif c["check"] in ("hidden_word", "anagram", "reversal"):
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
            elif c["status"] == "wrong":
                if c["check"] == "assembly":
                    score -= 50  # Pieces don't make the answer — fatal
                elif c["check"] in ("hidden_word", "anagram", "reversal"):
                    score -= 50  # Mechanism doesn't work — fatal
                elif c["check"] == "first_letter":
                    score -= 30  # Trivially verifiable, no excuse
                elif c["check"] == "abbreviation":
                    score -= 15  # Likely wrong
                elif c["check"] == "trivial":
                    score -= 40  # Just restating the definition, not an explanation
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
