from itertools import combinations
from collections import Counter
import re

from resources import norm_letters

# Evidence system integration (safe import with fallback)
EVIDENCE_SYSTEM_AVAILABLE = False
_evidence_detector = None
ComprehensiveWordplayDetector = None

try:
    from stages.evidence import ComprehensiveWordplayDetector
    EVIDENCE_SYSTEM_AVAILABLE = True
except ImportError as e:
    print(f"EVIDENCE IMPORT FAILED: {e}")


def _get_evidence_detector():
    global _evidence_detector
    if _evidence_detector is None and EVIDENCE_SYSTEM_AVAILABLE:
        _evidence_detector = ComprehensiveWordplayDetector()
    return _evidence_detector


def _normalize_enumeration(enumeration):
    """
    Normalize enumeration to integer letter count.
    "(2,5,8)" → 15, "(7)" → 7, "8" → 8, 8 → 8
    """
    if isinstance(enumeration, int):
        return enumeration
    if isinstance(enumeration, str):
        digits = re.findall(r'\d+', enumeration)
        return sum(int(d) for d in digits) if digits else 0
    return 0


def generate_anagram_hypotheses(clue_text, enumeration, candidates,
                                definition_words=None):
    """
    Enhanced anagram hypothesis generation:
    1. Run original logic first (preserves all existing hits)
    2. For unresolved cases, try evidence system
    3. Return combined results

    definition_words: set of lowercase word strings already attributed as the
    definition. These are excluded from the anagram fodder search so the solver
    never accidentally uses definition words as fodder.
    """

    # Normalize enumeration: "(2,5,8)" → 15
    enumeration = _normalize_enumeration(enumeration)

    # STEP 1: Run original logic (exactly as before)
    original_hypotheses = _generate_anagram_hypotheses_original(
        clue_text, enumeration, candidates, definition_words)

    # STEP 2: If original found hits, return them (preserves existing behavior)
    if original_hypotheses:
        return original_hypotheses

    # STEP 3: If no original hits AND evidence system available, try evidence system
    if EVIDENCE_SYSTEM_AVAILABLE:
        try:
            evidence_hypotheses = _generate_anagram_hypotheses_evidence(clue_text,
                                                                        enumeration,
                                                                        candidates)
            return evidence_hypotheses
        except Exception as e:
            print(f"EVIDENCE SYSTEM ERROR: {e}")
            # Silent fallback - don't break existing functionality
            pass

    # STEP 4: Return empty if nothing found (same as original behavior)
    return []


def _generate_anagram_hypotheses_original(clue_text, enumeration, candidates,
                                          definition_words=None):
    """
    Original anagram detection logic (preserved exactly).
    Stage A: Free anagram hypothesis generation (provisional).
    Includes Stage-B hygiene: reject trivial self-anagrams.

    definition_words: set of lowercase normalised word strings to exclude from
    the fodder search. These words have already been attributed as the definition
    at source and must not be considered as anagram fodder.
    """

    clue_lc = clue_text.lower()
    excluded = {norm_letters(w).lower() for w in definition_words} if definition_words else set()

    # ---- normalise candidates to letter counters ----
    candidate_counters = {}
    for cand in candidates:
        norm = norm_letters(cand)
        if len(norm) == enumeration:
            candidate_counters[cand] = Counter(norm)

    if not candidate_counters:
        return []

    # ---- tokenise clue ----
    # Keep words that contain letters (allow punctuation like apostrophes)
    words = []
    for w in clue_text.split():
        letters_only = ''.join(c for c in w if c.isalpha())
        if letters_only:  # Keep word if it has any letters
            words.append(w)  # Keep original form for display

    word_counters = [(w, Counter(norm_letters(w))) for w in words]

    # ---- Wire principle: exclude definition words from fodder pool ----
    if excluded:
        word_counters = [(w, c) for w, c in word_counters
                         if norm_letters(w).lower() not in excluded]

    hypotheses = []

    for r in range(1, len(word_counters) + 1):
        for idxs in combinations(range(len(word_counters)), r):
            chosen = [word_counters[i] for i in idxs]

            combined = Counter()
            for _, ctr in chosen:
                combined += ctr

            if sum(combined.values()) != enumeration:
                continue

            for candidate, cand_ctr in candidate_counters.items():
                if combined != cand_ctr:
                    continue

                # ---- STAGE B HYGIENE: reject self-anagrams ----
                # If the candidate appears verbatim in the clue, reject
                if candidate.lower() in clue_lc:
                    continue

                used_words = [w for w, _ in chosen]
                unused_words = [
                    w for i, (w, _) in enumerate(word_counters)
                    if i not in idxs
                ]

                hypotheses.append({
                    "answer": candidate,
                    "fodder_words": used_words,
                    "fodder_letters": "".join(sorted(combined.elements())),
                    "unused_words": unused_words,
                    "definition_words": list(definition_words) if definition_words else [],
                    "candidate_source": candidate,
                    "solve_type": "anagram_exact",
                    "confidence": "provisional",
                })

    return hypotheses


def _generate_anagram_hypotheses_evidence(clue_text, enumeration, candidates):
    """
    Evidence system anagram detection for unresolved cases.
    Converts evidence system output to standard hypothesis format.
    """

    detector = _get_evidence_detector()

    # Filter candidates to correct length (matching original behavior)
    filtered_candidates = []
    for cand in candidates:
        norm = norm_letters(cand)
        if len(norm) == enumeration:
            filtered_candidates.append(cand)

    if not filtered_candidates:
        return []

    # Use evidence system to find anagram evidence
    evidence_list = detector.analyze_clue_for_anagram_evidence(
        clue_text=clue_text,
        candidates=filtered_candidates,
        enumeration=f"({enumeration})",  # Convert to enumeration format
        debug=False
    )

    # Convert evidence objects to hypothesis format (matching original interface)
    hypotheses = []

    for evidence in evidence_list:
        # Calculate unused words (words not in fodder)
        clue_words = [w for w in clue_text.split() if w.isalpha()]
        unused_words = [w for w in clue_words if w not in evidence.fodder_words]

        # Map evidence types to solve types for reporting
        solve_type_map = {
            "exact": "anagram_evidence_exact",
            "partial": "anagram_evidence_partial",
            "deletion": "anagram_evidence_deletion",
            "insertion": "anagram_evidence_insertion"
        }

        hypothesis = {
            "answer": evidence.candidate,
            "fodder_words": evidence.fodder_words,
            "fodder_letters": evidence.fodder_letters,
            "unused_words": unused_words,
            "candidate_source": evidence.candidate,
            "solve_type": solve_type_map.get(evidence.evidence_type, "anagram_evidence"),
            "confidence": evidence.confidence,
            # Additional fields for enhanced functionality (preserve evidence system data)
            "evidence_type": evidence.evidence_type,
            "score_boost": detector.calculate_anagram_score_boost(evidence),
            "needed_letters": getattr(evidence, 'needed_letters', ''),
            "excess_letters": getattr(evidence, 'excess_letters', ''),
            # Indicator and attribution data for compound analysis
            "indicator_words": getattr(evidence, 'indicator_words', []),
            "indicator_position": getattr(evidence, 'indicator_position', -1),
            "link_words": getattr(evidence, 'link_words', []),
            "remaining_words": getattr(evidence, 'remaining_words', [])
        }

        hypotheses.append(hypothesis)

    return hypotheses