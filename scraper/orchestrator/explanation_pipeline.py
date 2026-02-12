#!/usr/bin/env python3
"""
Explanation Pipeline - Adapts existing solver stages to this project.

Flow:
1. Load clues from clues_master.db (tagged as 'anagram')
2. Build graph from cryptic_new.db
3. Run stages with ANSWER as only candidate:
   - DD stage
   - Lurker/Hidden stage
   - Anagram stage (+ compound)
4. Report what solved at each stage
5. (Future) API quality control on explanations

Key principle: Faithful to original stage code, just adapting paths/DBs.
"""

import os
import re
import sqlite3
from typing import List, Dict, Any, Optional, Tuple, Set
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

# =============================================================================
# CONFIGURATION
# =============================================================================

# Database paths
CLUES_MASTER_DB = Path(r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")
CRYPTIC_NEW_DB = Path(
    r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db")

# Cohort selection - EDIT THESE
SOURCE = "telegraph"  # times, guardian, telegraph, bigdave44, etc. (None = all)
PUZZLE_NUMBER = '27868'  # e.g., "28456" or "28450-28460" (None = any)
WORDPLAY_TYPE = None  # anagram, charade, hidden, etc. (None = any)
COUNT = 50  # Max clues to load
RANDOM = True  # True = random selection, False = sequential


# =============================================================================
# NORMALISATION (from resources.py - exact copy)
# =============================================================================

def clean_key(x: str) -> str:
    if not x:
        return ""
    x = x.strip().lower()
    x = x.replace("'", "'").replace("'", "'")
    x = x.replace("'", "")
    x = re.sub(r"^[^a-z]+", "", x)
    x = re.sub(r"[^a-z]+$", "", x)
    return x


def clean_val(x: str) -> str:
    if not x:
        return ""
    x = x.strip()
    x = re.sub(r"^[^A-Za-z]+", "", x)
    x = re.sub(r"[^A-Za-z]+$", "", x)
    return x


def norm_letters(s: str) -> str:
    return re.sub(r"[^A-Za-z]", "", s or "").lower()


def parse_enum(en):
    return sum(map(int, re.findall(r"\d+", en or "")))


# =============================================================================
# GRAPH BUILDING (from resources.py - exact copy)
# =============================================================================

def add_pair(graph, a, b):
    ak = clean_key(a)
    bv = clean_val(b)
    if ak and bv:
        graph.setdefault(ak, set()).add(bv)

    bk = clean_key(b)
    av = clean_val(a)
    if bk and av:
        graph.setdefault(bk, set()).add(av)


def load_graph(conn: sqlite3.Connection) -> dict:
    """Load definition/synonym graph from cryptic_new.db."""
    cur = conn.cursor()
    G = {}

    # Use augmented table as specified
    cur.execute("SELECT definition, answer FROM definition_answers_augmented")
    for d, ans in cur.fetchall():
        add_pair(G, d, ans)

    cur.execute("SELECT word, synonym FROM synonyms_pairs")
    for w, s in cur.fetchall():
        add_pair(G, w, s)

    return {k: list(v) for k, v in G.items()}


def build_wordlist(conn: sqlite3.Connection) -> List[str]:
    """Build wordlist from cryptic_new.db."""
    cur = conn.cursor()
    words = set()

    cur.execute("SELECT DISTINCT answer FROM definition_answers_augmented")
    for (a,) in cur.fetchall():
        v = clean_val(a)
        if v:
            words.add(v)

    cur.execute("SELECT DISTINCT word FROM synonyms_pairs")
    for (w,) in cur.fetchall():
        v = clean_val(w)
        if v:
            words.add(v)

    cur.execute("SELECT DISTINCT synonym FROM synonyms_pairs")
    for (s,) in cur.fetchall():
        v = clean_val(s)
        if v:
            words.add(v)

    return sorted(words)


# =============================================================================
# DD STAGE (from dd_stage.py - faithful copy with minor path fixes)
# =============================================================================

def _extract_words(clue_text: str) -> list:
    """Extract words from clue, removing enumeration."""
    text = re.sub(r'\(\d+(?:,\d+)*\)\s*$', '', clue_text).strip()
    words = text.split()
    return words


def _get_candidates_for_phrase(phrase: str, graph: dict, total_len: int = None) -> dict:
    """Get all candidates for windows generated from a phrase."""
    candidates = defaultdict(list)

    # Generate definition windows (simplified version)
    words = phrase.split()
    windows = []
    for i in range(len(words)):
        for j in range(i + 1, len(words) + 1):
            window = ' '.join(words[i:j])
            windows.append(window)

    for window in windows:
        key = clean_key(window)
        if not key:
            continue

        graph_candidates = graph.get(key)
        if not graph_candidates:
            continue

        for cand in graph_candidates:
            cand_norm = norm_letters(cand)

            if total_len is not None and len(cand_norm) != total_len:
                continue

            candidates[cand_norm].append((window, cand))

    return candidates


def generate_dd_hypotheses(clue_text: str, graph: dict, total_len: int = None) -> List[
    Dict]:
    """
    True Double Definition detection.
    Returns list of hypotheses if DD found.
    """
    if not clue_text:
        return []

    words = _extract_words(clue_text)

    if len(words) < 2:
        return []

    for split_point in range(1, len(words)):
        left_phrase = ' '.join(words[:split_point])
        right_phrase = ' '.join(words[split_point:])

        left_candidates = _get_candidates_for_phrase(left_phrase, graph, total_len)
        right_candidates = _get_candidates_for_phrase(right_phrase, graph, total_len)

        overlap = set(left_candidates.keys()) & set(right_candidates.keys())

        if overlap:
            cand_norm = next(iter(overlap))
            left_window, left_answer = left_candidates[cand_norm][0]
            right_window, right_answer = right_candidates[cand_norm][0]

            return [{
                "answer": left_answer,
                "windows": [left_window, right_window],
                "left_definition": left_phrase,
                "right_definition": right_phrase,
                "split_point": split_point,
                "solve_type": "double_definition",
            }]

    return []


# =============================================================================
# LURKER/HIDDEN STAGE (from lurker_stage.py - faithful copy)
# =============================================================================

def _letters_only_stream(clue_text: str) -> str:
    return "".join(ch.lower() for ch in clue_text if ch.isalpha())


def _word_spans_letters_only(clue_text: str) -> list:
    """Word spans in letters-only coordinates."""
    spans = []
    idx = 0
    in_word = False
    start = 0

    for ch in clue_text:
        if ch.isalpha():
            if not in_word:
                in_word = True
                start = idx
            idx += 1
        else:
            if in_word:
                spans.append((start, idx))
                in_word = False

    if in_word:
        spans.append((start, idx))

    return spans


def _is_valid_lurker_span(span: tuple, word_spans: list) -> bool:
    """A valid lurker span must cross at least one word boundary (span 2+ words)."""
    s, e = span

    touched = []
    for ws, we in word_spans:
        if s < we and e > ws:
            touched.append((ws, we))

    # Must span at least 2 words
    if len(touched) < 2:
        return False

    # First word: span must start inside it (not at beginning)
    first_ws, first_we = touched[0]
    if not (first_ws < s < first_we):
        return False

    # Last word: span must end inside it (not at end)
    last_ws, last_we = touched[-1]
    if not (last_ws < e < last_we):
        return False

    # Check words are contiguous
    for i in range(len(touched) - 1):
        if touched[i][1] != touched[i + 1][0]:
            return False

    return True


def generate_lurker_hypotheses(clue_text: str, enumeration: int, candidates: List[str]) -> \
        List[Dict]:
    """Generate lurker/hidden word hypotheses."""
    if not candidates or not isinstance(enumeration, int) or enumeration <= 0:
        return []

    norm_candidates = {
        norm_letters(c): c
        for c in candidates
        if len(norm_letters(c)) == enumeration
    }
    if not norm_candidates:
        return []

    stream = _letters_only_stream(clue_text)
    n = len(stream)
    L = enumeration
    if n < L:
        return []

    word_spans = _word_spans_letters_only(clue_text)
    hypotheses = []

    for i in range(0, n - L + 1):
        span = (i, i + L)

        if not _is_valid_lurker_span(span, word_spans):
            continue

        window = stream[i:i + L]

        if window in norm_candidates:
            hypotheses.append({
                "answer": norm_candidates[window],
                "direction": "forward",
                "letters": window,
                "span": span,
                "solve_type": "hidden_word",
                "confidence": "high",
            })

        rev = window[::-1]
        if rev in norm_candidates:
            hypotheses.append({
                "answer": norm_candidates[rev],
                "direction": "reverse",
                "letters": rev,
                "span": span,
                "solve_type": "hidden_reversed",
                "confidence": "high",
            })

    return hypotheses


# =============================================================================
# ANAGRAM STAGE (from anagram_stage.py - faithful copy)
# =============================================================================

from itertools import combinations
from collections import Counter


def generate_anagram_hypotheses(clue_text: str, enumeration: int,
                                candidates: List[str]) -> List[Dict]:
    """
    Anagram hypothesis generation.
    Finds word combinations that anagram to candidates.
    """
    clue_lc = clue_text.lower()

    # Normalise candidates to letter counters
    candidate_counters = {}
    for cand in candidates:
        norm = norm_letters(cand)
        if len(norm) == enumeration:
            candidate_counters[cand] = Counter(norm)

    if not candidate_counters:
        return []

    # Tokenise clue
    words = []
    for w in clue_text.split():
        letters_only = ''.join(c for c in w if c.isalpha())
        if letters_only:
            words.append(w)

    word_counters = [(w, Counter(norm_letters(w))) for w in words]

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

                # Reject self-anagrams
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
                    "solve_type": "anagram",
                    "confidence": "high",
                })

    return hypotheses


# =============================================================================
# COMPOUND ANAGRAM STAGE (using full evidence system + proper validation)
# =============================================================================

# Import the comprehensive evidence system
try:
    from anagram_evidence_system import ComprehensiveWordplayDetector, AnagramEvidence

    EVIDENCE_SYSTEM_AVAILABLE = True
except ImportError as e:
    EVIDENCE_SYSTEM_AVAILABLE = False
    print(f"Warning: anagram_evidence_system.py import failed: {e}")
except Exception as e:
    EVIDENCE_SYSTEM_AVAILABLE = False
    print(f"Warning: anagram_evidence_system.py error: {e}")

# Import the compound wordplay analyzer for proper validation
try:
    from compound_wordplay_analyzer import CompoundWordplayAnalyzer, WordRole

    COMPOUND_ANALYZER_AVAILABLE = True
except ImportError as e:
    COMPOUND_ANALYZER_AVAILABLE = False
    print(f"Warning: compound_wordplay_analyzer.py import failed: {e}")
except Exception as e:
    COMPOUND_ANALYZER_AVAILABLE = False
    print(f"Warning: compound_wordplay_analyzer.py error: {e}")

# Import the explanation builder for quality assessment
try:
    from explanation_builder import ExplanationBuilder

    EXPLANATION_BUILDER_AVAILABLE = True
except ImportError as e:
    EXPLANATION_BUILDER_AVAILABLE = False
    print(f"Warning: explanation_builder.py import failed: {e}")
except Exception as e:
    EXPLANATION_BUILDER_AVAILABLE = False
    print(f"Warning: explanation_builder.py error: {e}")

# Import the unified parse builder for general wordplay
try:
    from unified_parse_builder import UnifiedParseBuilder

    UNIFIED_PARSER_AVAILABLE = True
except ImportError as e:
    UNIFIED_PARSER_AVAILABLE = False
    print(f"Warning: unified_parse_builder.py import failed: {e}")
except Exception as e:
    UNIFIED_PARSER_AVAILABLE = False
    print(f"Warning: unified_parse_builder.py error: {e}")


class CompoundAnagramStage:
    """
    Uses the full ComprehensiveWordplayDetector for compound anagram detection.
    Integrates with CompoundWordplayAnalyzer for proper validation.
    Handles: exact, deletion, doubling, and compound (substitution + anagram).
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._detector = None
        self._analyzer = None
        self._explainer = None

    def _get_detector(self):
        if self._detector is None and EVIDENCE_SYSTEM_AVAILABLE:
            self._detector = ComprehensiveWordplayDetector(self.db_path)
        return self._detector

    def _get_analyzer(self):
        if self._analyzer is None and COMPOUND_ANALYZER_AVAILABLE:
            self._analyzer = CompoundWordplayAnalyzer(self.db_path)
        return self._analyzer

    def _get_explainer(self):
        if self._explainer is None and EXPLANATION_BUILDER_AVAILABLE:
            self._explainer = ExplanationBuilder()
        return self._explainer

    def close(self):
        if self._analyzer:
            self._analyzer.close()
            self._analyzer = None
        self._detector = None

    def detect(self, clue_text: str, answer: str, debug: bool = False) -> Optional[Dict]:
        """
        Detect compound/complex anagram using evidence system + proper validation.

        Returns hypothesis dict if found and validated, None otherwise.
        """
        detector = self._get_detector()
        if not detector:
            return None

        # Detect indicators
        indicators = detector.detect_wordplay_indicators(clue_text)

        if not indicators.get('anagram') and not indicators.get('anagram_matches'):
            return None

        # Test the answer as candidate
        answer_upper = answer.upper().replace(' ', '')
        answer_letters = len(answer_upper)
        enumeration = str(answer_letters)

        evidence = detector.test_anagram_evidence(
            candidate=answer_upper,
            clue_text=clue_text,
            indicators=indicators,
            enumeration=enumeration,
            debug=debug
        )

        if not evidence:
            return None

        # Get fodder info
        fodder_letters = evidence.fodder_letters or ''
        fodder_len = len(''.join(c for c in fodder_letters if c.isalpha()))
        fodder_words = evidence.fodder_words or []

        # Get substitution letters if any
        sub_letters_count = 0
        compound_substitutions = []
        if hasattr(evidence,
                   'compound_substitutions') and evidence.compound_substitutions:
            compound_substitutions = evidence.compound_substitutions
            for _, letters, _ in compound_substitutions:
                sub_letters_count += len(letters)

        # =================================================================
        # CRITICAL VALIDATION: Letter math must work
        # From explanation_builder.py assess_quality()
        # =================================================================
        total_letters = fodder_len + sub_letters_count

        if total_letters != answer_letters:
            if debug:
                print(
                    f"    REJECTED: Letter math failed: {fodder_len} + {sub_letters_count} = {total_letters} != {answer_letters}")
            return None

        # Require at least one real fodder word (not just punctuation)
        valid_fodder = [w for w in fodder_words if any(c.isalpha() for c in w)]
        if not valid_fodder:
            if debug:
                print(f"    REJECTED: No valid fodder words")
            return None

        # =================================================================
        # USE COMPOUNDWORDPLAYANALYZER FOR FULL VALIDATION (if available)
        # For now, skip analyzer quality check - just use letter math
        # The analyzer expects a more complex case dict format
        # =================================================================
        quality = 'medium'  # Letter math passed, so at least medium quality

        # TODO: Integrate analyzer properly when case format is correct
        # analyzer = self._get_analyzer()
        # if analyzer:
        #     try:
        #         result = analyzer.analyze_case(case)
        #         quality = result.get('explanation', {}).get('quality', 'none')
        #     except Exception as e:
        #         pass

        # =================================================================
        # BUILD HYPOTHESIS
        # =================================================================
        hypothesis = {
            "answer": answer,
            "fodder_words": fodder_words,
            "fodder_letters": fodder_letters,
            "evidence_type": evidence.evidence_type,
            "confidence": evidence.confidence,
            "indicator_words": evidence.indicator_words,
            "remaining_words": evidence.remaining_words,
            "solve_type": f"anagram_{evidence.evidence_type}",
            "quality": quality,
        }

        # Add compound-specific info if present
        if compound_substitutions:
            hypothesis["compound_substitutions"] = compound_substitutions
            sub_parts = [f"{phrase}→{letters}" for phrase, letters, cat in
                         compound_substitutions]
            hypothesis[
                "derivation"] = f"{' + '.join(sub_parts)} + anagram({fodder_letters.upper()}) = {answer}"

        # Add deletion info if present
        if hasattr(evidence, 'deletion_info') and evidence.deletion_info:
            hypothesis["deletion_info"] = evidence.deletion_info

        # Add doubling info if present
        if hasattr(evidence, 'doubling_info') and evidence.doubling_info:
            hypothesis["doubling_info"] = evidence.doubling_info

        return hypothesis


# =============================================================================
# PIPELINE ORCHESTRATOR
# =============================================================================

@dataclass
class PipelineResult:
    """Result from pipeline processing."""
    clue_id: int
    clue_text: str
    answer: str
    enumeration: str
    solved_by: Optional[str]  # 'dd', 'hidden', 'anagram', None
    hypothesis: Optional[Dict]
    explanation: Optional[Dict]


@dataclass
class CohortConfig:
    """Configuration for selecting a cohort of clues."""
    source: Optional[str] = None  # e.g., 'times', 'guardian', 'telegraph'
    puzzle_number: Optional[str] = 'telegraph'  # e.g., '28456' or range '28450-28460'
    wordplay_type: Optional[str] = '31087'  # e.g., 'anagram', 'charade'
    count: int = 50  # Max clues to return
    random: bool = True  # Random selection vs sequential

    def describe(self) -> str:
        """Human-readable description of the cohort."""
        parts = []
        if self.source:
            parts.append(f"source={self.source}")
        if self.puzzle_number:
            parts.append(f"puzzle={self.puzzle_number}")
        if self.wordplay_type:
            parts.append(f"wordplay={self.wordplay_type}")
        parts.append(f"count={self.count}")
        if self.random:
            parts.append("random")
        return ', '.join(parts) if parts else "all clues"


def list_available_sources() -> Dict[str, int]:
    """List available sources and their clue counts."""
    conn = sqlite3.connect(CLUES_MASTER_DB)
    cur = conn.cursor()
    cur.execute("""
        SELECT source, COUNT(*) as cnt 
        FROM clues 
        WHERE source IS NOT NULL 
        GROUP BY source 
        ORDER BY cnt DESC
    """)
    sources = {row[0]: row[1] for row in cur.fetchall()}
    conn.close()
    return sources


def list_puzzles_for_source(source: str, limit: int = 20) -> List[Dict]:
    """List recent puzzles for a source."""
    conn = sqlite3.connect(CLUES_MASTER_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT puzzle_number, COUNT(*) as clue_count,
               MIN(clue_text) as sample_clue
        FROM clues 
        WHERE source = ?
        AND puzzle_number IS NOT NULL
        GROUP BY puzzle_number
        ORDER BY puzzle_number DESC
        LIMIT ?
    """, (source, limit))
    puzzles = [dict(row) for row in cur.fetchall()]
    conn.close()
    return puzzles


def load_clues(config: CohortConfig = None, **kwargs) -> List[Dict]:
    """Load clues from clues_master.db with flexible filtering.

    Args:
        config: CohortConfig object with selection criteria
        **kwargs: Alternative to config - individual parameters

    Returns:
        List of clue dictionaries
    """
    # Support both config object and keyword arguments
    if config is None:
        config = CohortConfig(**kwargs)

    conn = sqlite3.connect(CLUES_MASTER_DB)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    # Build WHERE clause dynamically
    conditions = [
        "answer IS NOT NULL",
        "clue_text IS NOT NULL",
        "LENGTH(answer) >= 3"
    ]
    params = []

    if config.source:
        conditions.append("LOWER(source) = LOWER(?)")
        params.append(config.source)

    if config.puzzle_number:
        if '-' in str(config.puzzle_number):
            # Range: '28450-28460'
            start, end = config.puzzle_number.split('-')
            conditions.append("CAST(puzzle_number AS INTEGER) BETWEEN ? AND ?")
            params.extend([int(start), int(end)])
        else:
            conditions.append("puzzle_number = ?")
            params.append(config.puzzle_number)

    if config.wordplay_type:
        conditions.append("LOWER(wordplay_type) = LOWER(?)")
        params.append(config.wordplay_type)

    where_clause = " AND ".join(conditions)
    order_clause = "ORDER BY RANDOM()" if config.random else "ORDER BY id"

    query = f"""
        SELECT id, clue_text, answer, enumeration, wordplay_type, 
               source, puzzle_number
        FROM clues
        WHERE {where_clause}
        {order_clause}
        LIMIT ?
    """
    params.append(config.count)

    cur.execute(query, params)
    clues = [dict(row) for row in cur.fetchall()]
    conn.close()
    return clues


def load_complete_puzzle(source: str, puzzle_number: str) -> List[Dict]:
    """Load all clues from a specific puzzle (for realistic scoring)."""
    config = CohortConfig(
        source=source,
        puzzle_number=puzzle_number,
        count=100,  # Puzzles typically have 28-32 clues
        random=False  # Keep original order
    )
    return load_clues(config)


def run_pipeline(clues: List[Dict], graph: dict, wordlist: List[str]) -> List[
    PipelineResult]:
    """
    Run clues through the pipeline stages.

    Each stage receives the ANSWER as the only candidate.
    If a stage solves it, we record and move on.
    If not, we pass to the next stage.

    Stages:
    1. DD (double definition)
    2. Hidden/Lurker
    3. Pure Anagram
    4. Compound Anagram (via evidence system)
    5. General Parser (charade, substitution, reversal, etc.)
    """
    import time

    results = []

    stats = {'dd': 0, 'hidden': 0, 'anagram': 0, 'compound': 0, 'general': 0,
             'unsolved': 0}
    timing = {'dd': 0, 'hidden': 0, 'anagram': 0, 'compound': 0, 'general': 0}

    # Initialize compound detector (uses full evidence system)
    compound_stage = CompoundAnagramStage(str(CRYPTIC_NEW_DB))

    # Initialize general parser
    general_parser = None
    if UNIFIED_PARSER_AVAILABLE:
        general_parser = UnifiedParseBuilder(str(CRYPTIC_NEW_DB))

    total_start = time.time()

    for i, clue in enumerate(clues):
        clue_text = clue['clue_text']
        answer = clue['answer']
        enum_str = clue.get('enumeration', '')
        enum_int = parse_enum(enum_str)

        # Progress indicator
        if (i + 1) % 10 == 0:
            elapsed = time.time() - total_start
            print(f"  Processing {i + 1}/{len(clues)} ({elapsed:.1f}s elapsed)...")

        # Single candidate = the known answer
        candidates = [answer]

        result = PipelineResult(
            clue_id=clue['id'],
            clue_text=clue_text,
            answer=answer,
            enumeration=enum_str,
            solved_by=None,
            hypothesis=None,
            explanation=None
        )

        # Stage 1: DD
        t0 = time.time()
        dd_hyps = generate_dd_hypotheses(clue_text, graph, enum_int)
        timing['dd'] += time.time() - t0
        if dd_hyps:
            # Check if DD found our answer
            for hyp in dd_hyps:
                if norm_letters(hyp['answer']) == norm_letters(answer):
                    result.solved_by = 'dd'
                    result.hypothesis = hyp
                    stats['dd'] += 1
                    break

        # Stage 2: Hidden/Lurker (if not solved)
        if not result.solved_by:
            t0 = time.time()
            hidden_hyps = generate_lurker_hypotheses(clue_text, enum_int, candidates)
            timing['hidden'] += time.time() - t0
            if hidden_hyps:
                for hyp in hidden_hyps:
                    if norm_letters(hyp['answer']) == norm_letters(answer):
                        result.solved_by = 'hidden'
                        result.hypothesis = hyp
                        stats['hidden'] += 1
                        break

        # Stage 3: Pure Anagram (if not solved)
        if not result.solved_by:
            t0 = time.time()
            anagram_hyps = generate_anagram_hypotheses(clue_text, enum_int, candidates)
            timing['anagram'] += time.time() - t0
            if anagram_hyps:
                for hyp in anagram_hyps:
                    if norm_letters(hyp['answer']) == norm_letters(answer):
                        result.solved_by = 'anagram'
                        result.hypothesis = hyp
                        stats['anagram'] += 1
                        break

        # Stage 4: Compound Anagram via Evidence System (if not solved by pure anagram)
        if not result.solved_by and EVIDENCE_SYSTEM_AVAILABLE:
            t0 = time.time()
            compound_hyp = compound_stage.detect(clue_text, answer)
            timing['compound'] += time.time() - t0
            if compound_hyp:
                result.solved_by = 'compound'
                result.hypothesis = compound_hyp
                stats['compound'] += 1

        # Stage 5: General Parser (charade, substitution, reversal, etc.)
        if not result.solved_by and general_parser:
            t0 = time.time()
            parse_result = general_parser.parse(clue_text, answer, definition_words=None,
                                                debug=False)
            timing['general'] += time.time() - t0
            if parse_result:
                # Debug: show what parser found
                num_contribs = len(
                    parse_result.contributions) if parse_result.contributions else 0
                if num_contribs > 0:
                    print(
                        f"  DEBUG General: {answer} - {num_contribs} contributions, is_complete={parse_result.is_complete}")
                    for c in parse_result.contributions:
                        print(f"    -> {c.source_words} → {c.letters} ({c.operation})")

                # =============================================================
                # VALIDATION: Must have 2+ contributions
                # Single synonym lookup is NOT a valid charade parse
                # =============================================================
                is_valid_complete = False
                if parse_result.is_complete and len(parse_result.contributions) >= 2:
                    is_valid_complete = True
                # =============================================================

                if is_valid_complete:
                    result.solved_by = 'general'
                    stats['general'] += 1
                # Always store the hypothesis (complete or partial)
                result.hypothesis = {
                    'answer': answer,
                    'derivation': parse_result.derivation,
                    'contributions': [
                        {
                            'source': c.source_words,
                            'operation': c.operation,
                            'letters': c.letters,
                            'indicator': c.indicator
                        }
                        for c in parse_result.contributions
                    ],
                    'definition': parse_result.definition_words,
                    'link_words': parse_result.link_words,
                    'unparsed_words': parse_result.unparsed_words,
                    'letters_explained': ''.join(
                        c.letters for c in parse_result.contributions),
                    'letters_remaining': '',  # Computed below if needed
                    'confidence': parse_result.confidence,
                    'progress_summary': f"{len(parse_result.contributions)} contributions",
                    'diagnostic': {},  # Empty dict so .get() works
                    'solve_type': 'general' if is_valid_complete else 'partial'
                }

        if not result.solved_by:
            stats['unsolved'] += 1

        results.append(result)

    compound_stage.close()
    if general_parser:
        general_parser.close()

    total_time = time.time() - total_start

    # Print summary
    print("\n" + "=" * 60)
    print("PIPELINE RESULTS")
    print("=" * 60)
    print(f"Total clues: {len(clues)}")
    print(f"  Solved by DD:       {stats['dd']}")
    print(f"  Solved by Hidden:   {stats['hidden']}")
    print(f"  Solved by Anagram:  {stats['anagram']}")
    print(f"  Solved by Compound: {stats['compound']}")
    print(f"  Solved by General:  {stats['general']}")
    print(f"  Unsolved:           {stats['unsolved']}")

    total_solved = stats['dd'] + stats['hidden'] + stats['anagram'] + stats['compound'] + \
                   stats['general']
    pct = (total_solved / len(clues) * 100) if clues else 0
    print(f"\nSolve rate: {total_solved}/{len(clues)} ({pct:.1f}%)")

    # Timing breakdown
    print(f"\n--- TIMING ({total_time:.1f}s total) ---")
    for stage, t in timing.items():
        pct_time = (t / total_time * 100) if total_time > 0 else 0
        print(f"  {stage:12s}: {t:6.2f}s ({pct_time:4.1f}%)")
    print(f"  {'per clue':12s}: {total_time / len(clues) * 1000:.0f}ms avg")

    return results


def main():
    import sys
    from datetime import datetime

    # Uses config from top of file (SOURCE, PUZZLE_NUMBER, etc.)

    config = CohortConfig(
        source=SOURCE,
        puzzle_number=PUZZLE_NUMBER,
        wordplay_type=WORDPLAY_TYPE,
        count=COUNT,
        random=RANDOM
    )

    # Handle --list-sources and --list-puzzles from command line (still useful)
    for arg in sys.argv[1:]:
        if arg == '--list-sources':
            print("\n" + "=" * 60)
            print("AVAILABLE SOURCES")
            print("=" * 60)
            if not CLUES_MASTER_DB.exists():
                print(f"ERROR: Database not found at {CLUES_MASTER_DB}")
                return
            sources = list_available_sources()
            for source, count in sources.items():
                print(f"  {source:20s}: {count:>8,} clues")
            print(f"\nTotal: {sum(sources.values()):,} clues")
            return
        elif arg.startswith('--list-puzzles='):
            list_puzzles_source = arg.split('=')[1]
            print("\n" + "=" * 60)
            print(f"RECENT PUZZLES FOR: {list_puzzles_source}")
            print("=" * 60)
            if not CLUES_MASTER_DB.exists():
                print(f"ERROR: Database not found at {CLUES_MASTER_DB}")
                return
            puzzles = list_puzzles_for_source(list_puzzles_source)
            if not puzzles:
                print(f"  No puzzles found for source '{list_puzzles_source}'")
                return
            for p in puzzles:
                print(f"  #{p['puzzle_number']:>8s}  ({p['clue_count']} clues)")
            return

    # Main pipeline execution
    print("=" * 60)
    print("EXPLANATION PIPELINE - FULL CASCADE v3")
    print("=" * 60)
    print(f"Cohort: {config.describe()}")

    # Debug: check systems available
    print(f"\n*** EVIDENCE_SYSTEM_AVAILABLE = {EVIDENCE_SYSTEM_AVAILABLE} ***")
    print(f"*** COMPOUND_ANALYZER_AVAILABLE = {COMPOUND_ANALYZER_AVAILABLE} ***")
    print(f"*** EXPLANATION_BUILDER_AVAILABLE = {EXPLANATION_BUILDER_AVAILABLE} ***")
    print(f"*** UNIFIED_PARSER_AVAILABLE = {UNIFIED_PARSER_AVAILABLE} ***")

    # Load graph from cryptic_new.db
    print("\nLoading graph from cryptic_new.db...")
    if not CRYPTIC_NEW_DB.exists():
        print(f"ERROR: Database not found at {CRYPTIC_NEW_DB}")
        return

    cryptic_conn = sqlite3.connect(CRYPTIC_NEW_DB)
    graph = load_graph(cryptic_conn)
    wordlist = build_wordlist(cryptic_conn)
    cryptic_conn.close()
    print(f"  Graph entries: {len(graph):,}")
    print(f"  Wordlist size: {len(wordlist):,}")

    # Load clues based on cohort config
    print(f"\nLoading clues from clues_master.db...")
    if not CLUES_MASTER_DB.exists():
        print(f"ERROR: Database not found at {CLUES_MASTER_DB}")
        return

    clues = load_clues(config)
    print(f"  Loaded {len(clues)} clues")

    if not clues:
        print("  ERROR: No clues found matching criteria")
        return

    # Show source breakdown if mixed
    source_counts = {}
    for c in clues:
        src = c.get('source', 'unknown') or 'unknown'
        source_counts[src] = source_counts.get(src, 0) + 1
    if len(source_counts) > 1:
        print(f"  By source: {source_counts}")

    # Count by wordplay type
    type_counts = {}
    for c in clues:
        wt = c.get('wordplay_type', 'unknown') or 'unknown'
        type_counts[wt] = type_counts.get(wt, 0) + 1
    print(f"  By wordplay type: {type_counts}")

    # Run pipeline
    results = run_pipeline(clues, graph, wordlist)

    # Output to file
    OUTPUT_DIR = Path(
        r"C:\Users\shute\PycharmProjects\AI_Solver\Solver\orchestrator\logs")
    OUTPUT_DIR.mkdir(exist_ok=True)

    # Build filename from cohort config
    filename_parts = ["pipeline"]
    if config.source:
        filename_parts.append(config.source)
    if config.puzzle_number:
        filename_parts.append(f"puzzle{config.puzzle_number}")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_parts.append(timestamp)
    output_file = OUTPUT_DIR / f"{'_'.join(filename_parts)}.txt"

    lines = []
    lines.append("=" * 70)
    lines.append("EXPLANATION PIPELINE - FULL CASCADE RESULTS")
    lines.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Cohort: {config.describe()}")
    if config.source:
        lines.append(f"Source: {config.source}")
    if config.puzzle_number:
        lines.append(f"Puzzle: {config.puzzle_number}")
    lines.append(f"Clues loaded: {len(clues)}")
    lines.append("=" * 70)
    lines.append("")

    # Group by solved_by
    by_stage = {'dd': [], 'hidden': [], 'anagram': [], 'compound': [], 'general': [],
                None: []}
    for r in results:
        by_stage[r.solved_by].append(r)

    for stage_name, stage_results in by_stage.items():
        if not stage_results:
            continue

        stage_label = stage_name.upper() if stage_name else "UNSOLVED"
        lines.append(f"\n{'=' * 70}")
        lines.append(f"STAGE: {stage_label} ({len(stage_results)} clues)")
        lines.append("=" * 70)

        for r in stage_results:
            lines.append(f"\n[{stage_label}] {r.clue_text}")
            lines.append(f"  Answer: {r.answer}")

            # Get clue words for analysis
            clue_words = [w.strip('.,;:!?"\'') for w in r.clue_text.split()]

            if r.hypothesis:
                if r.solved_by == 'dd':
                    # Double Definition - show both definitions
                    windows = r.hypothesis.get('windows', [])
                    lines.append(f"  Wordplay: Double Definition")
                    lines.append(f"  Definition 1: \"{windows[0]}\"")
                    lines.append(
                        f"  Definition 2: \"{windows[1] if len(windows) > 1 else '?'}\"")

                elif r.solved_by == 'hidden':
                    # Hidden word
                    direction = r.hypothesis.get('direction', 'forward')
                    span = r.hypothesis.get('span', (0, 0))

                    # Common hidden word indicators
                    hidden_indicators = {
                        'in', 'within', 'inside', 'among', 'amongst', 'amid', 'amidst',
                        'part of', 'some', 'held by', 'held in', 'concealed', 'hidden',
                        'buried', 'contains', 'containing', 'houses', 'covers',
                        'picked up', 'found in', 'seen in', 'spotted in'
                    }
                    reversal_indicators = {
                        'back', 'returned', 'up', 'over', 'recalled', 'brought back',
                        'picked up', 'lifted', 'raised', 'reflected', 'reversed'
                    }

                    # Parse clue to find indicator and definition
                    clue_words = [w.strip('.,;:!?"\'()[]') for w in r.clue_text.split()]
                    indicator_found = []

                    for i, word in enumerate(clue_words):
                        word_lower = word.lower()
                        if word_lower in hidden_indicators:
                            indicator_found.append(word)
                        if direction == 'reverse' and word_lower in reversal_indicators:
                            indicator_found.append(word)
                        # Check two-word indicators
                        if i < len(clue_words) - 1:
                            two_word = f"{word_lower} {clue_words[i + 1].lower().strip('.,;:!?\"')}"
                            if two_word in hidden_indicators or two_word in reversal_indicators:
                                indicator_found.append(f"{word} {clue_words[i + 1]}")

                    lines.append(
                        f"  Wordplay: Hidden word {'(reversed)' if direction == 'reverse' else ''}")
                    if indicator_found:
                        lines.append(f"  Indicator: {', '.join(set(indicator_found))}")

                elif r.solved_by == 'anagram':
                    fodder = r.hypothesis.get('fodder_words', [])
                    unused = r.hypothesis.get('unused_words', [])
                    fodder_letters = r.hypothesis.get('fodder_letters', '')

                    # Common anagram indicators
                    anagram_indicators = {
                        'about', 'absurd', 'adjusted', 'agitated', 'altered', 'amended',
                        'anew', 'around', 'arranged', 'astray', 'awful', 'bad', 'badly',
                        'battered', 'bizarre', 'broken', 'bust', 'chaotic', 'cocktail',
                        'confused', 'contrived', 'converted', 'cooked', 'crazy',
                        'crushed',
                        'curious', 'damaged', 'dancing', 'deplorable', 'deranged',
                        'destroyed',
                        'different', 'disordered', 'disrupted', 'distorted', 'disturbed',
                        'drunk', 'dynamic', 'eccentric', 'edited', 'erratic', 'exploded',
                        'false', 'fancy', 'faulty', 'fermented', 'fixed', 'flexible',
                        'flying', 'foolish', 'forged', 'foul', 'free', 'fresh', 'funny',
                        'garbled', 'ground', 'haphazard', 'improper', 'incorrect',
                        'injured',
                        'insane', 'irregular', 'juggled', 'jumbled', 'lunatic', 'mad',
                        'mangled', 'maybe', 'medley', 'melted', 'messed', 'minced',
                        'misguided', 'mishap', 'mistreated', 'mixed', 'modified',
                        'moving',
                        'muddled', 'mutilated', 'nasty', 'new', 'novel', 'nuts', 'nutty',
                        'odd', 'off', 'ordered', 'organised', 'organized', 'out',
                        'peculiar',
                        'perhaps', 'played', 'playing', 'poor', 'possibly', 'prepared',
                        'processed', 'queer', 'quirky', 'random', 'rearranged', 'rebuilt',
                        'reformed', 'remodelled', 'reordered', 'reorganised',
                        'reorganized',
                        'repaired', 'reset', 'reshaped', 'reshuffled', 'restored',
                        'revised',
                        'revolutionary', 'reworked', 'rigged', 'rocky', 'rotten', 'rough',
                        'ruined', 'rum', 'run', 'running', 'scattered', 'scrambled',
                        'shaken',
                        'shattered', 'shifted', 'shot', 'shuffled', 'sick', 'silly',
                        'sloppy',
                        'smashed', 'somehow', 'sorted', 'sloshed', 'spilt', 'spoilt',
                        'spread',
                        'stewed', 'stirred', 'strange', 'tangled', 'tattered', 'terrible',
                        'thrashing', 'torn', 'tortured', 'trained', 'transformed',
                        'translated',
                        'treated', 'tricky', 'troubled', 'tumbling', 'twisted', 'undone',
                        'unfortunate', 'unhinged', 'unkempt', 'unlikely', 'unruly',
                        'unsettled',
                        'unusual', 'upset', 'varied', 'vile', 'volatile', 'wandering',
                        'wasted',
                        'wayward', 'weird', 'whipped', 'wicked', 'wild', 'woolly',
                        'worked',
                        'worried', 'wound', 'wrecked', 'writhing', 'wrong', 'wrongly'
                    }

                    # Classify unused words into indicator vs definition
                    indicator_words = []
                    definition_words = []
                    for word in unused:
                        word_clean = word.lower().strip('.,;:!?"\'()[]')
                        if word_clean in anagram_indicators:
                            indicator_words.append(word)
                        else:
                            definition_words.append(word)

                    lines.append(f"  Wordplay: Anagram")
                    lines.append(f"  Fodder: {' + '.join(fodder)}")
                    if indicator_words:
                        lines.append(f"  Indicator: {', '.join(indicator_words)}")
                    if definition_words:
                        lines.append(f"  Definition: \"{' '.join(definition_words)}\"")

                elif r.solved_by == 'compound':
                    subs = r.hypothesis.get('compound_substitutions', [])
                    fodder = r.hypothesis.get('fodder_words', [])
                    remaining = r.hypothesis.get('remaining_words', [])
                    indicator_words = r.hypothesis.get('indicator_words', [])
                    derivation = r.hypothesis.get('derivation', '')

                    lines.append(f"  Wordplay: Compound Anagram")
                    if subs:
                        for phrase, letters, category in subs:
                            lines.append(
                                f"  Substitution: {phrase} → {letters} ({category})")
                    if fodder:
                        lines.append(f"  Fodder: {' + '.join(fodder)}")
                    if derivation:
                        lines.append(f"  Derivation: {derivation}")
                    if indicator_words:
                        lines.append(f"  Indicator: {', '.join(indicator_words)}")
                    if remaining:
                        lines.append(f"  Definition: \"{' '.join(remaining)}\"")

                elif r.solved_by == 'general':
                    derivation = r.hypothesis.get('derivation', '')
                    definition = r.hypothesis.get('definition', [])
                    unparsed = r.hypothesis.get('unparsed_words', [])

                    lines.append(f"  Derivation: {derivation}")
                    if definition:
                        lines.append(f"  Definition: \"{' '.join(definition)}\"")
                    if unparsed:
                        lines.append(f"  Unused words: {', '.join(unparsed)}")

                elif not r.solved_by:
                    # UNSOLVED - show partial progress and diagnostic info
                    confidence = r.hypothesis.get('confidence', 0)
                    letters_explained = r.hypothesis.get('letters_explained', '')
                    letters_remaining = r.hypothesis.get('letters_remaining',
                                                         r.answer.upper())
                    unparsed = r.hypothesis.get('unparsed_words', [])
                    definition = r.hypothesis.get('definition', [])
                    contributions = r.hypothesis.get('contributions', [])
                    diagnostic = r.hypothesis.get('diagnostic', {})

                    # Use diagnostic contributions if sequential found nothing
                    diag_contribs = diagnostic.get('contributions', [])
                    if not contributions and diag_contribs:
                        contributions = diag_contribs
                        # Recalculate letters_explained from diagnostic
                        letters_explained = diagnostic.get('letters_found_display', '')
                        letters_remaining = diagnostic.get('letters_missing',
                                                           r.answer.upper())
                        unparsed = diagnostic.get('unused_words', [])
                        confidence = len(letters_explained) / len(
                            r.answer.replace(' ', '').replace('-', '')) if r.answer else 0

                    if definition:
                        lines.append(f"  Definition (guess): \"{' '.join(definition)}\"")

                    if contributions:
                        lines.append(f"  Found ({confidence * 100:.0f}% of answer):")
                        for c in contributions:
                            # Handle both dict format (from diagnostic) and list format
                            if isinstance(c, dict):
                                source = c.get('source', '?')
                                op = c.get('type', c.get('operation', '?'))
                                letters = c.get('letters', '?')
                                indicator = c.get('indicator', '')
                            else:
                                source = c.get('source', ['?'])
                                source = ' '.join(source) if isinstance(source,
                                                                        list) else source
                                op = c.get('operation', '?')
                                letters = c.get('letters', '?')
                                indicator = c.get('indicator', '')
                            ind_str = f" [{indicator}]" if indicator else ""
                            source_str = source if isinstance(source, str) else ' '.join(
                                source)
                            lines.append(f"    {source_str} → {letters} ({op}){ind_str}")
                        lines.append(f"  Explained: {letters_explained}")

                    if letters_remaining:
                        lines.append(f"  MISSING: {letters_remaining}")

                    if unparsed:
                        lines.append(f"  Unused words: {', '.join(unparsed)}")

                    # Show suggestions for gaps
                    gaps = diagnostic.get('gaps', [])
                    potential_gaps = diagnostic.get('potential_gaps', [])
                    if gaps and unparsed:
                        lines.append(f"  --- Suggestions ---")
                        seen = set()
                        for pg in potential_gaps[:5]:
                            sug = f"    {pg.get('candidate_word', '?')} → {pg.get('missing_letters', '?')}?"
                            if sug not in seen:
                                lines.append(sug)
                                seen.add(sug)

                    if not contributions and not letters_explained:
                        lines.append(f"  No wordplay components identified")

    # Summary
    lines.append("\n" + "=" * 70)
    lines.append("SUMMARY")
    lines.append("=" * 70)
    lines.append(f"Total clues: {len(results)}")
    lines.append(f"  Solved by DD:       {len(by_stage['dd'])}")
    lines.append(f"  Solved by Hidden:   {len(by_stage['hidden'])}")
    lines.append(f"  Solved by Anagram:  {len(by_stage['anagram'])}")
    lines.append(f"  Solved by Compound: {len(by_stage['compound'])}")
    lines.append(f"  Solved by General:  {len(by_stage['general'])}")
    lines.append(f"  Unsolved:           {len(by_stage[None])}")

    # Add partial progress stats
    partial_progress = [r for r in by_stage[None] if
                        r.hypothesis and r.hypothesis.get('confidence', 0) > 0]
    if partial_progress:
        lines.append(f"    (with partial progress: {len(partial_progress)})")

    total_solved = len(results) - len(by_stage[None])
    pct = (total_solved / len(results) * 100) if results else 0
    lines.append(f"\nSolve rate: {total_solved}/{len(results)} ({pct:.1f}%)")

    # Write file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(lines))

    print(f"\nResults written to: {output_file}")

    # Show sample results on console
    print("\n" + "=" * 60)
    print("SAMPLE RESULTS (first 10)")
    print("=" * 60)

    for r in results[:10]:
        status = r.solved_by.upper() if r.solved_by else "UNSOLVED"
        print(f"\n[{status}] {r.clue_text}")
        print(f"  Answer: {r.answer}")
        if r.hypothesis:
            if r.solved_by == 'anagram':
                fodder = r.hypothesis.get('fodder_words', [])
                unused = r.hypothesis.get('unused_words', [])
                derivation = f"anagram({'+'.join(fodder)}) = {r.answer}"
                print(f"  Derivation: {derivation}")
                if unused:
                    print(f"  Definition: {unused[0]}")
            elif r.solved_by == 'compound':
                print(f"  Derivation: {r.hypothesis.get('derivation', '')}")
            elif r.solved_by == 'general':
                print(f"  Derivation: {r.hypothesis.get('derivation', '')}")
                definition = r.hypothesis.get('definition', [])
                if definition:
                    print(f"  Definition: {' '.join(definition)}")
            elif r.solved_by == 'dd':
                windows = r.hypothesis.get('windows', [])
                print(f"  Derivation: {windows[0]} = {r.answer} = {windows[1]}")
            elif r.solved_by == 'hidden':
                direction = r.hypothesis.get('direction', 'forward')
                print(
                    f"  Derivation: hidden {'(reversed)' if direction == 'reverse' else ''}")
            elif not r.solved_by:
                # Show partial progress for unsolved
                confidence = r.hypothesis.get('confidence', 0)
                definition = r.hypothesis.get('definition', [])
                if definition:
                    print(f"  Definition: {' '.join(definition)}")
                if confidence > 0:
                    print(
                        f"  Found ({confidence * 100:.0f}%): {r.hypothesis.get('derivation', '')}")
                    print(f"  MISSING: {r.hypothesis.get('letters_remaining', '')}")
                else:
                    print(f"  No wordplay components identified")


if __name__ == "__main__":
    main()