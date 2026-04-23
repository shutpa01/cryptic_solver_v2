"""Piece-level cryptic clue verifier.

Verifies each piece of a cryptic clue explanation by its mechanism,
then checks assembly (pieces concatenate to answer).

Works from structured components (ai_pieces), not by parsing explanation text.
This is more reliable than regex-based extraction from free text.

Mechanism verification:
  - synonym:          DB lookup (synonyms_pairs + definition_answers_augmented, both directions)
  - abbreviation:     DB lookup (wordplay table)
  - literal:          letters == word (trivial)
  - first_letter:     letters == word[0] (or first N letters)
  - last_letter:      letters == word[-1] (or last N letters)
  - core_letters:     letters == word[1:-1] (middle)
  - outer_letters:    letters == word[0] + word[-1]
  - alternate_letters: every other letter of word
  - deletion:         letters is a subword of source (prefix, suffix, or with middle removed)
  - reversal:         reverse(letters) is synonym/abbreviation of clue_word
  - anagram_fodder:   sorted(letters) == sorted(answer or sub-answer)
  - sound_of:         no mechanical check (unverifiable)
"""

import re
import sqlite3
from functools import lru_cache


def norm(s):
    """Strip non-alpha, uppercase."""
    return re.sub(r"[^A-Za-z]", "", s or "").upper()


class PieceVerifier:
    """Verifies individual pieces and assemblies against the reference DB.

    Uses RefDB for in-memory lookups — no per-query SQLite hits.
    """

    def __init__(self, ref_db=None):
        if ref_db is None:
            import os, sys
            sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))
            from signature_solver.db import RefDB
            ref_db = RefDB()
        self._ref_db = ref_db
        # Also keep a direct DB connection for definition_matches
        import os
        ref_db_path = os.path.join(
            os.path.dirname(os.path.dirname(__file__)),
            "data", "cryptic_new.db"
        )
        self.ref = sqlite3.connect(f"file:{ref_db_path}?mode=ro", uri=True)
        self._def_cache = {}

    def _is_synonym(self, word, letters):
        """Check if word -> letters exists in synonyms (includes definition_answers_augmented).

        Direct dict lookup on RefDB.synonyms for speed — avoids building full synonym lists.
        """
        from signature_solver.db import _normalize_key
        w = _normalize_key(word.strip().strip(".,;:!?\"'()-"))
        lt = letters.upper().strip()
        if not w or not lt:
            return False
        # Forward: word -> letters
        syns = self._ref_db.synonyms.get(w, [])
        if lt in syns:
            return True
        # Reverse: letters -> word
        rev_key = _normalize_key(lt)
        rev_syns = self._ref_db.synonyms.get(rev_key, [])
        if w.upper() in rev_syns:
            return True
        return False

    def _is_abbreviation(self, word, letters):
        """Check if word -> letters exists in abbreviations (wordplay table)."""
        from signature_solver.db import _normalize_key
        w = _normalize_key(word.strip().strip(".,;:!?\"'()-"))
        lt = letters.upper().strip()
        if not w or not lt:
            return False
        abbrs = self._ref_db.abbreviations.get(w, [])
        return lt in abbrs

    def _check_any_word(self, text, letters):
        """Try words and word pairs from text against synonym and abbreviation tables.

        Handles cases where clue_word contains gloss text like "'about', to be returned"
        or "Post Office = mail delivery" instead of just the clue word.
        """
        from signature_solver.db import _normalize_key
        lt = letters.upper().strip()
        if not lt:
            return False
        words = re.findall(r'[a-zA-Z]+', text)
        # Try individual words
        for w in words:
            if len(w) < 2:
                continue
            wkey = _normalize_key(w)
            if lt in self._ref_db.synonyms.get(wkey, []):
                return True
            if lt in self._ref_db.abbreviations.get(wkey, []):
                return True
        # Try consecutive word pairs
        for i in range(len(words) - 1):
            phrase = words[i] + " " + words[i + 1]
            pkey = _normalize_key(phrase)
            if lt in self._ref_db.synonyms.get(pkey, []):
                return True
            if lt in self._ref_db.abbreviations.get(pkey, []):
                return True
        return False

    def verify_piece(self, piece, answer=None):
        """Verify a single piece by its mechanism.

        Args:
            piece: dict with clue_word, letters, mechanism
            answer: full answer (needed for anagram_fodder check)

        Returns:
            dict with: status ('verified', 'unverifiable', 'wrong'), detail (str)
        """
        mech = piece.get("mechanism", "").lower()
        word = piece.get("clue_word", "")
        letters = piece.get("letters", "")

        if not letters:
            return {"status": "wrong", "detail": "no letters"}

        word_norm = norm(word)
        letters_norm = norm(letters)

        # --- synonym ---
        if mech == "synonym":
            if word_norm == letters_norm:
                return {"status": "unverifiable", "detail": f"self-synonym: {letters}={word}"}
            if self._is_synonym(word, letters):
                return {"status": "verified", "detail": f'"{word}" -> {letters}: synonym in DB'}
            if self._is_abbreviation(word, letters):
                return {"status": "verified", "detail": f'"{word}" -> {letters}: abbreviation in DB'}
            # Try extracting individual words from gloss text
            if self._check_any_word(word, letters):
                return {"status": "verified", "detail": f'word in "{word}" -> {letters}: in DB'}
            # Single-letter "synonyms" not in DB are fabricated abbreviations — wrong
            if len(letters_norm) == 1:
                return {"status": "wrong", "detail": f'"{word}" -> {letters}: single-letter synonym NOT in DB'}
            return {"status": "unverifiable", "detail": f'"{word}" -> {letters}: not in DB'}

        # --- abbreviation ---
        if mech == "abbreviation":
            if self._is_abbreviation(word, letters):
                return {"status": "verified", "detail": f'"{word}" -> {letters}: abbreviation in DB'}
            if self._is_synonym(word, letters):
                return {"status": "verified", "detail": f'"{word}" -> {letters}: synonym in DB'}
            if self._check_any_word(word, letters):
                return {"status": "verified", "detail": f'word in "{word}" -> {letters}: in DB'}
            # Abbreviations must be in the DB — if not found, it's wrong
            return {"status": "wrong", "detail": f'"{word}" -> {letters}: abbreviation NOT in DB'}

        # --- literal ---
        if mech == "literal":
            if word_norm == letters_norm:
                return {"status": "verified", "detail": f'literal: {letters}'}
            # Might be a partial literal
            if letters_norm in word_norm:
                return {"status": "verified", "detail": f'literal substring: {letters} in {word}'}
            # Fallback: often mislabelled by TFTT parser — check if it's actually a synonym/abbreviation
            if self._is_synonym(word, letters):
                return {"status": "verified", "detail": f'mislabelled literal, actually synonym: "{word}" -> {letters}'}
            if self._is_abbreviation(word, letters):
                return {"status": "verified", "detail": f'mislabelled literal, actually abbreviation: "{word}" -> {letters}'}
            if self._check_any_word(word, letters):
                return {"status": "verified", "detail": f'mislabelled literal, word in "{word}" -> {letters}'}
            return {"status": "unverifiable", "detail": f'literal mismatch: {letters} not in {word}'}

        # --- first_letter ---
        if mech == "first_letter":
            if word_norm and letters_norm == word_norm[0]:
                return {"status": "verified", "detail": f'first letter of "{word}" = {letters}'}
            # Multi-word: first letter of each word
            words = [norm(w) for w in word.split() if norm(w)]
            if words and letters_norm == "".join(w[0] for w in words[:len(letters_norm)]):
                return {"status": "verified", "detail": f'first letters of "{word}" = {letters}'}
            return {"status": "wrong", "detail": f'first letter mismatch: {word} -> {letters}'}

        # --- last_letter ---
        if mech == "last_letter":
            if word_norm and letters_norm == word_norm[-1]:
                return {"status": "verified", "detail": f'last letter of "{word}" = {letters}'}
            words = [norm(w) for w in word.split() if norm(w)]
            if words and letters_norm == "".join(w[-1] for w in words[:len(letters_norm)]):
                return {"status": "verified", "detail": f'last letters of "{word}" = {letters}'}
            return {"status": "wrong", "detail": f'last letter mismatch: {word} -> {letters}'}

        # --- core_letters ---
        if mech in ("core_letters", "inner_letters"):
            if len(word_norm) >= 3 and letters_norm == word_norm[1:-1]:
                return {"status": "verified", "detail": f'core of "{word}" = {letters}'}
            return {"status": "wrong", "detail": f'core mismatch: {word} -> {letters}'}

        # --- outer_letters ---
        if mech == "outer_letters":
            if len(word_norm) >= 2 and letters_norm == word_norm[0] + word_norm[-1]:
                return {"status": "verified", "detail": f'outer letters of "{word}" = {letters}'}
            return {"status": "wrong", "detail": f'outer mismatch: {word} -> {letters}'}

        # --- alternate_letters ---
        if mech in ("alternate_letters", "alternating"):
            odd = word_norm[0::2]
            even = word_norm[1::2]
            if letters_norm == odd:
                return {"status": "verified", "detail": f'odd letters of "{word}" = {letters}'}
            if letters_norm == even:
                return {"status": "verified", "detail": f'even letters of "{word}" = {letters}'}
            return {"status": "wrong", "detail": f'alternate mismatch: {word} -> {letters}'}

        # --- deletion ---
        if mech == "deletion":
            # Check if letters is a prefix, suffix, or subword of source
            source = piece.get("source", word)
            source_norm = norm(source)
            if not source_norm:
                source_norm = word_norm
            if source_norm.startswith(letters_norm):
                return {"status": "verified", "detail": f'prefix deletion: {source} -> {letters}'}
            if source_norm.endswith(letters_norm):
                return {"status": "verified", "detail": f'suffix deletion: {source} -> {letters}'}
            if letters_norm in source_norm:
                return {"status": "verified", "detail": f'internal deletion: {source} -> {letters}'}
            return {"status": "wrong", "detail": f'deletion: {letters} not in {source}'}

        # --- truncation ---
        if mech == "truncation":
            if word_norm.startswith(letters_norm) or word_norm.endswith(letters_norm):
                return {"status": "verified", "detail": f'truncation of "{word}" = {letters}'}
            return {"status": "wrong", "detail": f'truncation mismatch: {word} -> {letters}'}

        # --- reversal ---
        if mech == "reversal":
            reversed_letters = letters_norm[::-1]
            if self._is_synonym(word, reversed_letters):
                return {"status": "verified", "detail": f'reverse of "{word}" synonym {reversed_letters} = {letters}'}
            if self._is_abbreviation(word, reversed_letters):
                return {"status": "verified", "detail": f'reverse of "{word}" abbr {reversed_letters} = {letters}'}
            if word_norm[::-1] == letters_norm:
                return {"status": "verified", "detail": f'reverse of "{word}" = {letters}'}
            # Try extracting words from gloss text
            if self._check_any_word(word, reversed_letters):
                return {"status": "verified", "detail": f'word in "{word}" reversed -> {letters}: in DB'}
            return {"status": "unverifiable", "detail": f'reversal: {word} -> {letters}'}

        # --- anagram_fodder ---
        if mech == "anagram_fodder":
            if answer:
                answer_norm = norm(answer)
                if sorted(letters_norm) == sorted(answer_norm):
                    return {"status": "verified", "detail": f'anagram: {letters} -> {answer}'}
            # Can't fully verify without knowing what it anagrams to
            return {"status": "unverifiable", "detail": f'anagram fodder: {letters}'}

        # --- sound_of / homophone ---
        if mech in ("sound_of", "homophone"):
            return {"status": "unverifiable", "detail": f'homophone: {word} -> {letters}'}

        # --- hidden ---
        if mech == "hidden":
            return {"status": "verified", "detail": f'hidden in "{word}"'}

        # --- unknown mechanism ---
        return {"status": "unverifiable", "detail": f'unknown mechanism: {mech}'}

    def verify_assembly(self, pieces, answer, wordplay_type=None):
        """Check that pieces assemble to the answer.

        Tries concatenation, container (insertion), and single-piece reversal.
        Anagram fallback only allowed when wordplay_type includes 'anagram'.
        """
        answer_norm = norm(answer)
        piece_letters = [norm(p.get("letters", "")) for p in pieces]
        assembled = "".join(piece_letters)
        wt = (wordplay_type or "").lower()
        is_anagram_type = "anagram" in wt

        # Direct concatenation
        if assembled == answer_norm:
            return {"status": "verified", "detail": f"assembly: {assembled} = {answer_norm}"}

        # Anagram (sorted letters match) — only for anagram types
        if is_anagram_type:
            has_anagram = any(p.get("mechanism") == "anagram_fodder" for p in pieces)
            if has_anagram and sorted(assembled) == sorted(answer_norm):
                return {"status": "verified", "detail": f"anagram assembly: sorted({assembled}) = sorted({answer_norm})"}

        # Container: try inserting each piece into the concatenation of others
        if len(piece_letters) >= 2:
            for i in range(len(piece_letters)):
                inner = piece_letters[i]
                outer = "".join(piece_letters[:i] + piece_letters[i+1:])
                if not inner or not outer:
                    continue
                for k in range(1, len(outer)):
                    if outer[:k] + inner + outer[k:] == answer_norm:
                        return {"status": "verified", "detail": f"container assembly: {inner} inside {outer} = {answer_norm}"}

        # Reversal: try reversing one piece
        if len(piece_letters) >= 1:
            for i in range(len(piece_letters)):
                trial = list(piece_letters)
                trial[i] = trial[i][::-1]
                if "".join(trial) == answer_norm:
                    return {"status": "verified", "detail": f"reversal assembly: piece {i} reversed = {answer_norm}"}

        # Anagram of all letters — only for anagram types
        if is_anagram_type and sorted(assembled) == sorted(answer_norm) and assembled != answer_norm:
            return {"status": "verified", "detail": f"anagram assembly: sorted({assembled}) = sorted({answer_norm})"}

        return {"status": "wrong", "detail": f"assembly mismatch: {assembled} != {answer_norm}"}

    def verify_definition(self, definition, answer):
        """Check if definition maps to answer in DB (both directions)."""
        if not definition or not answer:
            return {"status": "wrong", "detail": "no definition"}

        if self._is_synonym(definition, answer):
            return {"status": "verified", "detail": f'"{definition}" -> {answer}: in DB'}
        return {"status": "unverifiable", "detail": f'"{definition}" -> {answer}: not in DB'}

    def verify_clue(self, clue_text, answer, definition, wordplay_type, components):
        """Full verification of a clue from structured components.

        Args:
            clue_text: original clue text
            answer: known answer
            definition: definition text (or None)
            wordplay_type: e.g. "charade", "anagram", "hidden"
            components: dict with ai_pieces, assembly, wordplay_type

        Returns:
            dict with score (0-100), verdict (HIGH/MEDIUM/LOW/FAIL), checks list,
            and positional_breakdown list
        """
        checks = []
        answer_norm = norm(answer)

        # --- Definition ---
        if definition and definition.lower() not in ("double definition",):
            def_result = self.verify_definition(definition, answer)
            checks.append({"check": "definition", **def_result})
        elif definition and definition.lower() == "double definition":
            checks.append({"check": "definition", "status": "verified",
                           "detail": "double definition"})
        else:
            checks.append({"check": "no_definition", "status": "wrong",
                           "detail": "no definition provided"})

        # --- Pieces ---
        pieces = components.get("ai_pieces", []) if components else []
        positional = []
        pos = 0

        for p in pieces:
            letters = p.get("letters", "")
            letters_len = len(norm(letters))
            piece_result = self.verify_piece(p, answer=answer)
            checks.append({"check": f"piece:{p.get('mechanism', '?')}",
                           **piece_result})

            # Build positional breakdown
            positional.append({
                "start": pos + 1,
                "end": pos + letters_len,
                "letters": norm(letters),
                "clue_word": p.get("clue_word", ""),
                "mechanism": p.get("mechanism", ""),
                "verified": piece_result["status"] == "verified",
            })
            pos += letters_len

        # --- Assembly ---
        if pieces:
            asm_result = self.verify_assembly(pieces, answer, wordplay_type)
            checks.append({"check": "assembly", **asm_result})

        # --- Type-specific mechanical checks ---
        wt = (wordplay_type or "").lower()

        if "hidden" in wt:
            clue_letters = norm(clue_text)
            if answer_norm in clue_letters or answer_norm[::-1] in clue_letters:
                checks.append({"check": "hidden_word", "status": "verified",
                               "detail": f"{answer_norm} hidden in clue"})
            else:
                checks.append({"check": "hidden_word", "status": "wrong",
                               "detail": f"{answer_norm} NOT hidden in clue"})

        if wt == "anagram" and pieces:
            # Fodder may be in clue_word (parser) or letters (solver) — try both
            fodder_from_cw = "".join(norm(p.get("clue_word", "")) for p in pieces
                                     if p.get("mechanism") == "anagram_fodder")
            fodder_from_lt = "".join(norm(p.get("letters", "")) for p in pieces
                                     if p.get("mechanism") == "anagram_fodder")
            # Use clue_word if it's different from answer (real fodder), else use letters
            fodder = fodder_from_cw if fodder_from_cw != answer_norm else fodder_from_lt
            if fodder and sorted(fodder) == sorted(answer_norm) and fodder != answer_norm:
                checks.append({"check": "anagram", "status": "verified",
                               "detail": f"sorted({fodder}) = sorted({answer_norm})"})

            # Provenance check: fodder clue_words should appear in the clue text
            if clue_text:
                clue_norm = norm(clue_text)
                for p in pieces:
                    if p.get("mechanism") != "anagram_fodder":
                        continue
                    cw = norm(p.get("clue_word", ""))
                    if cw and cw not in clue_norm:
                        checks.append({"check": "anagram_provenance", "status": "wrong",
                                       "detail": f'fodder "{p.get("clue_word")}" not found in clue text'})

        # Container check: can one piece be inserted into another to form the answer?
        if wt == "container" and len(pieces) >= 2:
            piece_letters = [norm(p.get("letters", "")) for p in pieces]
            container_ok = False
            for i in range(len(piece_letters)):
                inner = piece_letters[i]
                outer = "".join(piece_letters[:i] + piece_letters[i+1:])
                if not inner or not outer:
                    continue
                for k in range(1, len(outer)):
                    if outer[:k] + inner + outer[k:] == answer_norm:
                        container_ok = True
                        break
                if container_ok:
                    break
            if container_ok:
                checks.append({"check": "container", "status": "verified",
                               "detail": f"container assembly verified"})

        # Reversal check: does reversing a piece give the answer?
        if wt == "reversal" and pieces:
            all_letters = "".join(norm(p.get("letters", "")) for p in pieces)
            if all_letters[::-1] == answer_norm:
                checks.append({"check": "reversal", "status": "verified",
                               "detail": f"reverse of {all_letters} = {answer_norm}"})
            # Also check if assembly with one piece reversed works
            elif len(pieces) >= 2:
                for i in range(len(pieces)):
                    trial = list(norm(p.get("letters", "")) for p in pieces)
                    trial[i] = trial[i][::-1]
                    if "".join(trial) == answer_norm:
                        checks.append({"check": "reversal", "status": "verified",
                                       "detail": f"charade with reversal of piece {i}"})
                        break

        # --- Scoring ---
        return self._score(checks, positional)

    def _score(self, checks, positional):
        """Score based on verified/unverifiable/wrong checks.

        Philosophy:
          - definition_only (no piece checks) = LOW (not enough evidence)
          - verified assembly + verified definition = HIGH baseline
          - each verified piece adds confidence
          - any wrong piece is a heavy penalty
          - unverifiable pieces are neutral
        """
        verified = [c for c in checks if c["status"] == "verified"]
        unverifiable = [c for c in checks if c["status"] == "unverifiable"]
        wrong = [c for c in checks if c["status"] == "wrong"]

        has_def = any(c["check"] == "definition" and c["status"] == "verified" for c in checks)
        has_assembly = any(c["check"] == "assembly" and c["status"] == "verified" for c in checks)
        has_mechanical = any(c["check"] in ("hidden_word", "anagram", "container", "reversal")
                            and c["status"] == "verified" for c in checks)
        has_wrong = len(wrong) > 0
        has_no_def = any(c["check"] == "no_definition" for c in checks)

        piece_checks = [c for c in checks if c["check"].startswith("piece:")]
        pieces_verified = sum(1 for c in piece_checks if c["status"] == "verified")
        pieces_wrong = sum(1 for c in piece_checks if c["status"] == "wrong")
        pieces_total = len(piece_checks)

        # If anagram is verified, all fodder pieces count as verified
        # The sorted letter match IS the verification for the whole set
        if has_mechanical and any(c["check"] == "anagram" and c["status"] == "verified" for c in checks):
            fodder_count = sum(1 for c in piece_checks if "anagram_fodder" in c["check"])
            pieces_verified = max(pieces_verified, fodder_count)

        # Start at 0 and earn points
        score = 0

        # Definition: +15
        if has_def:
            score += 15

        # Assembly verified: +25
        if has_assembly:
            score += 25

        # Mechanical proof (hidden/anagram): +25
        if has_mechanical:
            score += 25

        # Piece verification: up to +40
        if pieces_total > 0:
            piece_ratio = pieces_verified / pieces_total
            score += int(40 * piece_ratio)

        # Penalties
        if pieces_wrong > 0:
            score -= 30 * pieces_wrong
        # Anagram provenance failure: fodder words not found in clue text
        provenance_wrong = sum(1 for c in checks
                               if c["check"] == "anagram_provenance" and c["status"] == "wrong")
        if provenance_wrong > 0:
            score -= 30
        # Missing definition is only a penalty when there's no mechanical proof
        # and no verified assembly. A verified anagram/hidden IS the proof.
        # A verified charade assembly (pieces concatenate to answer) is strong evidence.
        if has_no_def and not has_mechanical and not has_assembly:
            score -= 15

        # Clamp
        score = max(0, min(100, score))

        # Verdict
        if score >= 80:
            verdict = "HIGH"
        elif score >= 40:
            verdict = "MEDIUM"
        elif score >= 20:
            verdict = "LOW"
        else:
            verdict = "FAIL"

        # Special case: definition only, no pieces = LOW regardless
        if has_def and not piece_checks and not has_mechanical and not has_assembly:
            verdict = "LOW"
            score = min(score, 30)

        return {
            "score": score,
            "verdict": verdict,
            "checks": checks,
            "positional": positional,
            "verified": len(verified),
            "unverifiable": len(unverifiable),
            "wrong": len(wrong),
        }
