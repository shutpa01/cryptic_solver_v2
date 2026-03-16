"""Catalog-driven signature solver.

Simple loop: analyze words → match against catalog → execute → done.
The complexity lives in the catalog (data) and matcher (logic),
not in ad-hoc strategy functions.
"""

from .tokens import *
from .db import RefDB
from .word_analyzer import analyze_words, analyze_phrases, clean_word
from .catalog import CATALOG
from .matcher import match_signatures
from .positional_catalog import POSITIONAL_CATALOG
from .positional_matcher import match_positional
from .base_catalog import BASE_CATALOG
from .base_matcher import match_base
from . import executor
from .confidence import score_result


class SignatureResult:
    """Result of a successful signature match."""

    def __init__(self, signature, word_roles, explanation_parts):
        self.signature = signature          # list of tokens
        self.word_roles = word_roles        # list of (word, token, value)
        self.explanation_parts = explanation_parts  # human-readable breakdown

    def signature_str(self):
        return " · ".join(self.signature)

    def __repr__(self):
        return f"<SignatureResult: {self.signature_str()}>"


class SolveResult:
    """Full result from solve(), including evidence for API fallback."""

    def __init__(self, result, confidence, confidence_reasons, analyses, phrases):
        self.result = result              # SignatureResult or None
        self.confidence = confidence      # 0-100
        self.confidence_reasons = confidence_reasons
        self.analyses = analyses          # word analyses for API evidence
        self.phrases = phrases            # phrase analyses

    @property
    def solved(self):
        return self.result is not None

    @property
    def high_confidence(self):
        return self.confidence >= 80

    @property
    def medium_confidence(self):
        return 50 <= self.confidence < 80

    def evidence_summary(self):
        """Format word analyses as structured evidence for an API call."""
        lines = []
        for wa in self.analyses:
            roles = []
            for tok, vals in wa.roles.items():
                if tok in (HID_F, ANA_F, POS_F):
                    continue
                if vals == [True]:
                    roles.append(tok)
                else:
                    short = vals[:5]
                    roles.append(f"{tok}({','.join(str(v) for v in short)})")
            if roles:
                lines.append(f"  {wa.text}: {' | '.join(roles)}")
        return "\n".join(lines)


def extract_definition_candidates(clue_words, answer, db, max_def_words=4):
    """Try definition candidates from both ends of the clue.

    For each candidate, check if it's a synonym of the answer using
    definition_answers_augmented and synonyms_pairs.

    Returns list of (definition_phrase, wordplay_words) tuples,
    ordered by likelihood (exact DB matches first, shorter defs first).
    """
    candidates = []

    for n in range(1, min(max_def_words + 1, len(clue_words))):
        # Try n words from the start
        def_phrase = " ".join(clue_words[:n])
        wp_words = clue_words[n:]
        if wp_words and db.is_definition_of(def_phrase, answer):
            candidates.append((def_phrase, wp_words))

        # Try n words from the end
        def_phrase = " ".join(clue_words[-n:])
        wp_words = clue_words[:-n]
        if wp_words and db.is_definition_of(def_phrase, answer):
            candidates.append((def_phrase, wp_words))

    return candidates


def _normalize_clue(text):
    """Normalize clue text: strip smart quotes, accented chars, etc."""
    import unicodedata
    # Normalize unicode (e.g. à -> a, é -> e)
    nfkd = unicodedata.normalize('NFKD', text)
    ascii_text = ''.join(c for c in nfkd if not unicodedata.combining(c))
    # Replace smart quotes/dashes with plain equivalents
    for old, new in [('\u2018', "'"), ('\u2019', "'"), ('\u201c', '"'),
                     ('\u201d', '"'), ('\u2013', '-'), ('\u2014', '-')]:
        ascii_text = ascii_text.replace(old, new)
    return ascii_text


def solve_clue(clue_text, answer, db, min_confidence=0):
    """Solve a raw clue: extract definition candidates, then solve each.

    Args:
        clue_text: the full clue text (string)
        answer: the known answer
        db: RefDB instance
        min_confidence: minimum confidence score to accept (0-100)

    Returns:
        SolveResult (best result across all definition candidates)
    """
    clue_words = _normalize_clue(clue_text).strip().split()
    answer_clean = answer.upper().replace(" ", "").replace("-", "")

    candidates = extract_definition_candidates(clue_words, answer_clean, db)

    best_sr = None

    for def_phrase, wp_words in candidates:
        # Determine def position: if wp_words are after def, def is at start
        if clue_words[:len(clue_words) - len(wp_words)] == clue_words[:len(def_phrase.split())]:
            def_pos = 'start'
        else:
            def_pos = 'end'

        sr = solve(wp_words, answer_clean, db, min_confidence, def_pos=def_pos)
        if sr.solved:
            sr.definition = def_phrase
            if best_sr is None or sr.confidence > best_sr.confidence:
                best_sr = sr
            if sr.high_confidence:
                break

    if best_sr is not None:
        return best_sr

    # No definition candidate worked — fall back to best unsolved result
    if candidates:
        def_phrase, wp_words = candidates[0]
        sr = solve(wp_words, answer_clean, db, min_confidence=0)
        sr.definition = def_phrase
        return sr

    # No definition candidates found at all — try full clue as wordplay
    sr = solve(clue_words, answer_clean, db, min_confidence=0)
    sr.definition = None
    return sr


def solve(wordplay_words, answer, db, min_confidence=0, def_pos=None):
    """Solve using catalog-driven signature matching.

    Args:
        wordplay_words: list of words from the wordplay window
        answer: the known answer (uppercase, no spaces/hyphens)
        db: RefDB instance
        min_confidence: minimum confidence score to accept (0-100)
        def_pos: "start" or "end" (where definition is) or None

    Returns:
        SolveResult with .result, .confidence, .confidence_reasons, .analyses
    """
    answer_clean = answer.upper().replace(" ", "").replace("-", "")

    # Step 1: Analyse all words
    analyses, phrases = analyze_phrases(wordplay_words, answer, db)

    # Step 2: Match against catalogs and execute
    best_result = None
    best_confidence = -1
    best_reasons = []

    def _process_match(entry, assignment):
        """Process a verified match — returns (result, confidence, reasons) or None."""
        # Execute (builds explanation)
        success, explanation, pieces = executor.execute_signature(
            entry, assignment, wordplay_words, answer_clean
        )
        if not success:
            return None

        # Build SignatureResult
        sig_tokens = [tok for _, tok, _ in pieces]
        for ind_tok, ind_idx in assignment['indicator_indices'].items():
            sig_tokens.append(ind_tok)
        for lnk_idx in assignment.get('lnk_indices', set()):
            sig_tokens.append(LNK)

        word_roles = list(pieces)
        for ind_tok, ind_idx in assignment['indicator_indices'].items():
            if isinstance(ind_idx, tuple):
                phrase = " ".join(wordplay_words[k] for k in range(ind_idx[0], ind_idx[1]))
                word_roles.append((phrase, ind_tok, None))
            else:
                word_roles.append((wordplay_words[ind_idx], ind_tok, None))
        for lnk_idx in assignment.get('lnk_indices', set()):
            word_roles.append((wordplay_words[lnk_idx], LNK, None))

        sig_result = SignatureResult(sig_tokens, word_roles, [explanation])
        confidence, reasons = score_result(
            sig_result, wordplay_words, answer_clean, analyses, db
        )
        return sig_result, confidence, reasons

    # Try base matcher first (flexible spans, collapsed patterns)
    for entry, assignment in match_base(
        wordplay_words, analyses, phrases, BASE_CATALOG,
        answer_clean, db, def_pos=def_pos
    ):
        result = _process_match(entry, assignment)
        if result is None:
            continue

        sig_result, confidence, reasons = result
        if confidence > best_confidence:
            best_result = sig_result
            best_confidence = confidence
            best_reasons = reasons

        if confidence >= 80:
            break

    # If base matcher found high confidence, return immediately
    if best_confidence >= 80:
        return SolveResult(best_result, best_confidence, best_reasons,
                           analyses, phrases)

    # Fall back to positional matcher (fixed spans, more entries)
    for entry, assignment in match_positional(
        wordplay_words, analyses, phrases, POSITIONAL_CATALOG,
        answer_clean, db, def_pos=def_pos
    ):
        result = _process_match(entry, assignment)
        if result is None:
            continue

        sig_result, confidence, reasons = result
        if confidence > best_confidence:
            best_result = sig_result
            best_confidence = confidence
            best_reasons = reasons

        if confidence >= 80:
            break

    if best_confidence >= 80:
        return SolveResult(best_result, best_confidence, best_reasons,
                           analyses, phrases)

    # Fall back to old catalog matcher for patterns not in other catalogs
    for entry, assignment in match_signatures(
        wordplay_words, analyses, phrases, CATALOG, answer_clean, db
    ):
        result = _process_match(entry, assignment)
        if result is None:
            continue

        sig_result, confidence, reasons = result
        if confidence > best_confidence:
            best_result = sig_result
            best_confidence = confidence
            best_reasons = reasons

        if confidence >= 80:
            break

    # Return best result
    if best_result is not None and best_confidence >= min_confidence:
        return SolveResult(best_result, best_confidence, best_reasons,
                           analyses, phrases)

    return SolveResult(None, max(0, best_confidence), best_reasons,
                       analyses, phrases)
