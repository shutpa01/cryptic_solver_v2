#!/usr/bin/env python3
"""
General Wordplay Analyzer - Orchestrator for complete puzzle solving.

1. Calls anagram_analysis.main() to run the full pipeline
   (DD, lurker, anagram, compound stages)
2. Processes non-anagram clues (stage_anagram hit_found=0) through general wordplay
3. Writes a unified puzzle report pulling from ALL stage tables

This file does NOT modify any existing pipeline files.
"""

import sys
import sqlite3
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from datetime import datetime

# Add project root to path

# Import the compound analyzer to reuse its analysis methods
from stages.compound import (
    CompoundWordplayAnalyzer, WordRole, norm_letters
)
from stages.unified_explanation import ExplanationBuilder
# Database paths
PIPELINE_DB_PATH = Path(
    r'C:\Users\shute\PycharmProjects\cryptic_solver_V2\pipeline_stages.db')
CRYPTIC_DB_PATH = Path(
    r'C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\cryptic_new.db')


def get_pipeline_connection():
    """Get pipeline database connection."""
    return sqlite3.connect(PIPELINE_DB_PATH)


def get_cryptic_connection():
    """Get cryptic database connection."""
    return sqlite3.connect(CRYPTIC_DB_PATH)


# ======================================================================
# STAGE: General wordplay table management
# ======================================================================

def init_general_table():
    """Create stage_general table if it doesn't exist."""
    conn = get_pipeline_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_general (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            clue_id INTEGER,
            clue_text TEXT,
            answer TEXT,
            formula TEXT,
            quality TEXT,
            definition_window TEXT,
            substitutions TEXT,
            operation_indicators TEXT,
            fully_resolved INTEGER,
            letters_still_needed TEXT,
            unresolved_words TEXT,
            breakdown TEXT,
            word_roles TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_general_run ON stage_general(run_id)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_general_clue ON stage_general(clue_id)")

    conn.commit()
    conn.close()


def clear_stage_general(run_id: int):
    """Clear previous general stage data."""
    conn = get_pipeline_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM stage_general WHERE run_id = ?", (run_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        print(f"  Cleared {deleted} old records from stage_general")


# ======================================================================
# STAGE: Load non-anagram clues
# ======================================================================

def load_non_anagram_clues(run_id: int = 0) -> List[Dict[str, Any]]:
    """
    Load clues for general wordplay analysis from TWO sources:
    1. stage_anagram WHERE hit_found = 0 (no anagram evidence at all)
    2. stage_compound WHERE fully_resolved = 0 (anagram evidence didn't help)

    EXCLUDES clues already solved by DD or lurker stages.
    Also joins with stage_definition to get definition support data.
    """
    conn = get_pipeline_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # First, get clue_ids already solved by DD or lurker
    cursor.execute("""
        SELECT clue_id FROM stage_dd WHERE run_id = ? AND hit_found = 1
        UNION
        SELECT clue_id FROM stage_lurker WHERE run_id = ? AND hit_found = 1
    """, (run_id, run_id))
    excluded_ids = {row['clue_id'] for row in cursor.fetchall()}

    if excluded_ids:
        print(f"  Excluding {len(excluded_ids)} clues already solved by DD/lurker")

    # Source 1: No anagram evidence (excluding DD/lurker hits)
    cursor.execute("""
        SELECT 
            a.clue_id,
            a.clue_text,
            a.answer,
            a.unused_words,
            d.candidates,
            d.support
        FROM stage_anagram a
        LEFT JOIN stage_definition d 
            ON a.clue_id = d.clue_id AND a.run_id = d.run_id
        WHERE a.run_id = ? AND a.hit_found = 0
        ORDER BY a.clue_id
    """, (run_id,))
    anagram_rows = [row for row in cursor.fetchall() if
                    row['clue_id'] not in excluded_ids]

    # Source 2: Compound rejects and weak partial anagrams (excluding DD/lurker hits)
    cursor.execute("""
        SELECT 
            c.clue_id,
            c.clue_text,
            c.answer,
            d.candidates,
            d.support
        FROM stage_compound c
        LEFT JOIN stage_definition d 
            ON c.clue_id = d.clue_id AND c.run_id = d.run_id
        WHERE c.run_id = ? AND c.fully_resolved IN (0, 2)
        ORDER BY c.clue_id
    """, (run_id,))
    compound_rows = [row for row in cursor.fetchall() if
                     row['clue_id'] not in excluded_ids]

    conn.close()

    seen_ids = set()
    results = []

    def _parse_row(row, source_label):
        clue_id = row['clue_id']
        if clue_id in seen_ids:
            return None
        seen_ids.add(clue_id)

        record = {
            'id': clue_id,
            'clue': row['clue_text'],
            'answer': row['answer'] or '',
            'source_stage': source_label,
        }

        # Parse answer (may be stored as "likely_answer|db_answer")
        answer_parts = record['answer'].split('|')
        record['answer'] = answer_parts[-1] if answer_parts else ''

        # Parse JSON fields safely
        try:
            record['unused_words'] = json.loads(
                row['unused_words'] or '[]') if 'unused_words' in row.keys() else []
        except (json.JSONDecodeError, KeyError):
            record['unused_words'] = []

        try:
            record['candidates'] = json.loads(row['candidates'] or '[]')
        except json.JSONDecodeError:
            record['candidates'] = []

        try:
            record['support'] = json.loads(row['support'] or '{}')
        except json.JSONDecodeError:
            record['support'] = {}

        # Extract enumeration from clue text
        enum_match = re.search(r'\([\d,]+\)\s*$', record['clue'])
        if enum_match:
            record['enumeration'] = enum_match.group().strip('()')
        else:
            record['enumeration'] = ''

        return record

    # Process anagram misses first
    for row in anagram_rows:
        rec = _parse_row(row, 'anagram_miss')
        if rec:
            results.append(rec)

    anagram_miss_count = len(results)

    # Then compound rejects
    for row in compound_rows:
        rec = _parse_row(row, 'compound_reject')
        if rec:
            results.append(rec)

    compound_reject_count = len(results) - anagram_miss_count
    print(
        f"  Sources: {anagram_miss_count} anagram misses + {compound_reject_count} compound rejects")

    return results


# ======================================================================
# STAGE: Definition finding
# ======================================================================

def find_definition_window(answer: str, clue_text: str, support: Dict[str, Any]) -> \
Optional[str]:
    """Find the definition window for a given answer.

    After locating the initial phrase, attempts to extend it by testing adjacent
    clue words against definition_answers_augmented (longest extension first).
    This removes spurious words from the wordplay pool rather than leaving them
    unresolved.
    """
    answer_upper = answer.upper().replace(' ', '')
    definition = None

    # Strategy 1: Check support from definition stage
    if support:
        for key, value in support.items():
            if isinstance(value, list):
                for v in value:
                    if isinstance(v, str) and v.upper().replace(' ', '') == answer_upper:
                        definition = key
                        break
                if definition:
                    break
                if key.upper().replace(' ', '') == answer_upper:
                    definition = value[0] if value else None
                    break

    # Strategy 2: Database lookup
    if not definition:
        definition = _find_definition_from_db(answer_upper, clue_text)

    if not definition:
        return None

    # Strategy 3: Extend definition to cover adjacent clue words.
    # Tries longest extension first so we never stop too early.
    return _try_extend_definition(answer_upper, clue_text, definition)


def _normalise_def_word(w: str) -> str:
    """Lowercase, strip punctuation and possessive 's for definition matching."""
    w = w.lower().strip(".,;:!?\"'")
    if w.endswith("'s"):
        w = w[:-2]
    elif w.endswith("s'"):
        w = w[:-1]
    return w.strip()


def _find_definition_from_db(answer: str, clue_text: str) -> Optional[str]:
    """Find definition by looking up known definitions in definition_answers_augmented.

    CRITICAL: Only accepts definitions at the START or END of the clue, never mid-clue.

    Strategy 1: known definitions for this answer → check if any appear in the clue.
    Strategy 2 (fallback): try 1-5 word windows from clue start/end → look up in DB,
                           normalising for possessives and punctuation.
    """
    if not answer:
        return None

    answer_upper = answer.upper().replace(' ', '')

    conn = get_cryptic_connection()
    cursor = conn.cursor()

    # ── Strategy 1 (unchanged) ────────────────────────────────────────────────
    cursor.execute("""
        SELECT DISTINCT definition
        FROM definition_answers_augmented
        WHERE UPPER(REPLACE(answer, ' ', '')) = ?
    """, (answer_upper,))

    known_definitions = [row[0] for row in cursor.fetchall()]
    known_definitions.sort(key=len, reverse=True)

    clue_text_lower = clue_text.lower()
    for defn in known_definitions:
        defn_lower = defn.lower()
        if clue_text_lower.startswith(defn_lower + ' ') or \
           clue_text_lower.startswith(defn_lower) and len(defn_lower) == len(clue_text_lower) or \
           clue_text_lower.endswith(' ' + defn_lower) or \
           clue_text_lower.endswith(defn_lower) and len(defn_lower) == len(clue_text_lower):
            conn.close()
            return defn

    # ── Strategy 2: window scan ───────────────────────────────────────────────
    # Try longest windows first; always leave at least 1 word for wordplay.
    clue_words = clue_text.split()
    max_size = min(5, len(clue_words) - 1)

    for size in range(max_size, 0, -1):
        for words in (clue_words[:size], clue_words[-size:]):
            window = ' '.join(_normalise_def_word(w) for w in words)
            cursor.execute("""
                SELECT 1 FROM definition_answers_augmented
                WHERE LOWER(definition) = ?
                AND UPPER(REPLACE(answer, ' ', '')) = ?
            """, (window, answer_upper))
            if cursor.fetchone():
                conn.close()
                return ' '.join(words)   # original text, for display
            # Also check synonyms_pairs (e.g. "football team" → "rangers")
            cursor.execute("""
                SELECT 1 FROM synonyms_pairs
                WHERE LOWER(word) = ?
                AND UPPER(REPLACE(synonym, ' ', '')) = ?
            """, (window, answer_upper))
            if cursor.fetchone():
                conn.close()
                return ' '.join(words)

    conn.close()
    return None


def _try_extend_definition(answer_upper: str, clue_text: str, definition: str) -> str:
    """Extend a found definition phrase by testing adjacent clue words against the DB.

    After the initial definition is found (e.g. "bully"), checks whether adding
    neighbouring clue words produces a longer phrase that also maps to the answer
    in definition_answers_augmented.  Tests from the longest possible extension
    down to 1 extra word and returns the first (longest) match found.

    Only extends inward from the clue edge where the definition sits — never
    reaches across wordplay words in the middle of the clue.
    Returns the original definition unchanged if no extension matches.
    """
    if not definition:
        return definition

    clue_clean = re.sub(r'\s*\([\d,\s\-]+\)\s*$', '', clue_text)
    clue_words = clue_clean.split()
    def_lower = definition.lower().strip()
    clue_lower = clue_clean.lower()

    at_end   = clue_lower.endswith(def_lower)
    at_start = clue_lower.startswith(def_lower)

    if not at_start and not at_end:
        return definition

    def_count = len(definition.split())

    conn = get_cryptic_connection()
    cursor = conn.cursor()
    result = definition

    if at_end:
        def_start_idx = len(clue_words) - def_count
        max_extra = min(5, def_start_idx)
        # Try longest extension first
        for extra in range(max_extra, 0, -1):
            extended_words = clue_words[def_start_idx - extra:]
            phrase = ' '.join(_normalise_def_word(w) for w in extended_words)
            cursor.execute(
                "SELECT 1 FROM definition_answers_augmented "
                "WHERE LOWER(definition) = ? AND UPPER(REPLACE(answer,' ','')) = ?",
                (phrase.lower(), answer_upper)
            )
            if cursor.fetchone():
                result = ' '.join(extended_words)
                break

    elif at_start:
        max_extra = min(5, len(clue_words) - def_count)
        # Try longest extension first
        for extra in range(max_extra, 0, -1):
            extended_words = clue_words[:def_count + extra]
            phrase = ' '.join(_normalise_def_word(w) for w in extended_words)
            cursor.execute(
                "SELECT 1 FROM definition_answers_augmented "
                "WHERE LOWER(definition) = ? AND UPPER(REPLACE(answer,' ','')) = ?",
                (phrase.lower(), answer_upper)
            )
            if cursor.fetchone():
                result = ' '.join(extended_words)
                break

    conn.close()
    return result


# ======================================================================
# STAGE: Save general results
# ======================================================================

def save_stage_general(run_id: int, records: List[Dict[str, Any]]):
    """Save general wordplay analysis results."""
    conn = get_pipeline_connection()
    cursor = conn.cursor()

    for rec in records:
        explanation = rec.get('explanation', {})
        compound = rec.get('compound_solution') or {}
        word_roles = rec.get('word_roles', [])

        word_roles_data = []
        for wr in word_roles:
            if hasattr(wr, '__dict__'):
                word_roles_data.append(wr.__dict__)
            elif isinstance(wr, dict):
                word_roles_data.append(wr)

        cursor.execute("""
            INSERT INTO stage_general 
            (run_id, clue_id, clue_text, answer, formula, quality, definition_window,
             substitutions, operation_indicators, fully_resolved, letters_still_needed,
             unresolved_words, breakdown, word_roles)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            rec.get('id'),
            rec.get('clue'),
            rec.get('answer', ''),
            explanation.get('formula', ''),
            explanation.get('quality', ''),
            rec.get('definition_window', ''),
            json.dumps(compound.get('substitutions', [])),
            json.dumps(compound.get('operation_indicators', [])),
            1 if compound.get('fully_resolved') else 0,
            compound.get('letters_still_needed', ''),
            json.dumps(compound.get('unresolved_words', [])),
            json.dumps(explanation.get('breakdown', [])),
            json.dumps(word_roles_data)
        ))

    conn.commit()
    conn.close()
    print(f"  Saved {len(records)} records to stage_general (run_id={run_id})")


# ======================================================================
# General Wordplay Analyzer
# ======================================================================

class GeneralWordplayAnalyzer:
    """
    Analyzes general wordplay clues (non-anagram).
    Uses the compound analyzer's machinery with empty anagram letters.
    """

    def __init__(self):
        self.compound_analyzer = CompoundWordplayAnalyzer()
        self.explainer = ExplanationBuilder()

    def close(self):
        if self.compound_analyzer:
            self.compound_analyzer.close()

    def analyze_case(self, record: Dict[str, Any]) -> Dict[str, Any]:
        """Analyze a single non-anagram clue using general wordplay techniques."""
        clue_text = record.get('clue', '')
        answer = record.get('answer', '').upper().replace(' ', '')
        support = record.get('support', {})

        definition_window = find_definition_window(answer, clue_text, support)

        clue_clean = re.sub(r'\s*\([\d,]+\)\s*$', '', clue_text)
        clue_words = clue_clean.replace('(', ' ').replace(')', ' ').split()

        word_roles = []
        accounted_words = set()

        if definition_window:
            for w in definition_window.split():
                for cw in clue_words:
                    if cw.lower().strip('.,;:!?"\'') == w.lower().strip('.,;:!?"\''):
                        word_roles.append(WordRole(cw, 'definition', answer, 'pipeline'))
                        accounted_words.add(cw.lower())
                        break

        remaining_words = [w for w in clue_words
                           if w.lower() not in accounted_words]

        compound_solution = None
        if remaining_words:
            compound_solution = self.compound_analyzer._analyze_remaining_words(
                remaining_words,
                '',  # Empty anagram letters
                answer,
                word_roles,
                accounted_words,
                clue_words,
                definition_window
            )

            # RETRY: If letters still needed and indicators were assigned,
            # retry with those indicators skipped so they can be tried as synonyms
            if (compound_solution and
                    compound_solution.get('letters_still_needed') and
                    compound_solution.get('operation_indicators')):

                # Collect indicator words to skip
                skip_words = set()
                for ind_word, ind_type, ind_subtype in compound_solution[
                    'operation_indicators']:
                    for w in ind_word.split():
                        skip_words.add(w.lower())

                # Rebuild word_roles and accounted_words from scratch (definition only)
                word_roles_retry = []
                accounted_retry = set()
                if definition_window:
                    for w in definition_window.split():
                        for cw in clue_words:
                            if cw.lower().strip('.,;:!?"\'') == w.lower().strip(
                                    '.,;:!?"\''):
                                word_roles_retry.append(
                                    WordRole(cw, 'definition', answer, 'pipeline'))
                                accounted_retry.add(cw.lower())
                                break

                remaining_retry = [w for w in clue_words
                                   if w.lower() not in accounted_retry]

                retry_solution = self.compound_analyzer._analyze_remaining_words(
                    remaining_retry,
                    '',
                    answer,
                    word_roles_retry,
                    accounted_retry,
                    clue_words,
                    definition_window,
                    skip_indicator_words=skip_words
                )

                # Use retry if it either fully resolves all letters, or
                # accounts for more clue words (fewer unresolved). More words
                # explained = more genuine cryptic mechanisms found, even if
                # fewer answer letters are matched (greedy synonyms can inflate
                # letter counts without being correct).
                if retry_solution:
                    retry_unresolved = len(retry_solution.get('unresolved_words', []))
                    pass1_unresolved = len(compound_solution.get('unresolved_words', []))
                    retry_fully_resolved = not retry_solution.get('letters_still_needed')

                    if retry_fully_resolved or retry_unresolved < pass1_unresolved:
                        compound_solution = retry_solution
                        word_roles = word_roles_retry
                        accounted_words = accounted_retry

        explanation = self.explainer.build_explanation(
            record,
            word_roles,
            [],  # No fodder words
            '',  # No fodder letters
            None,  # No anagram indicator
            definition_window,
            compound_solution,
            clue_words,
            answer
        )

        accounted_normalized = {norm_letters(w) for w in accounted_words}
        remaining_unresolved = [w for w in clue_words
                                if norm_letters(w) not in accounted_normalized]

        return {
            'id': record.get('id'),
            'clue': clue_text,
            'answer': answer,
            'definition_window': definition_window,
            'word_roles': word_roles,
            'compound_solution': compound_solution,
            'explanation': explanation,
            'remaining_unresolved': remaining_unresolved
        }

    def analyze_cohort(self, records: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze all non-anagram clues."""
        results = []
        for i, record in enumerate(records):
            try:
                result = self.analyze_case(record)
                results.append(result)
                if (i + 1) % 100 == 0:
                    print(f"  Processed {i + 1}/{len(records)} clues...")
            except Exception as e:
                print(f"  Warning: Error analyzing clue {record.get('id')}: {e}")
                continue
        return results


# ======================================================================
# Unified Puzzle Report
# ======================================================================

def run_general_analysis(run_id: int = 0):
    """Run general wordplay analysis on non-anagram clues."""
    init_general_table()
    clear_stage_general(run_id)

    print(
        f"\nLoading clues for general analysis (anagram misses + compound rejects, run_id={run_id})...")
    non_anagram_clues = load_non_anagram_clues(run_id)
    print(f"  Total: {len(non_anagram_clues)} clues for general wordplay")

    if not non_anagram_clues:
        print("  No non-anagram clues to analyze.")
        return []

    print(f"\nAnalyzing general wordplay...")
    analyzer = GeneralWordplayAnalyzer()

    try:
        results = analyzer.analyze_cohort(non_anagram_clues)
    finally:
        analyzer.close()

    print(f"\nSaving results...")
    save_stage_general(run_id, results)

    return results


# DEAD CODE: Use report.py main() instead. Kept temporarily for Phase 1 cleanup.
def main():
    """
    Complete puzzle solver:
    1. Run anagram_analysis (which runs the full pipeline)
    2. Run general wordplay on remaining clues
    3. Write unified puzzle report
    """
    print("=" * 60)
    print("COMPLETE PUZZLE SOLVER")
    print("=" * 60)

    # Step 1: Run the existing pipeline via anagram_analysis
    print("\nSTEP 1: Running pipeline (DD, lurker, anagram, compound)...")
    print("-" * 60)

    import anagram_analysis
    anagram_analysis.main()

    # Step 2: Run general wordplay on non-anagram clues
    print("\n" + "-" * 60)
    print("STEP 2: Running general wordplay analysis...")
    print("-" * 60)

    run_general_analysis(run_id=0)

    # Step 3: Write unified report
    print("\n" + "-" * 60)
    print("STEP 3: Writing unified puzzle report...")
    print("-" * 60)

    from report import get_puzzle_info, get_all_clue_ids, resolve_clue, format_report, PIPELINE_DB, CLUES_DB, OUTPUT_FILE
    puzzle_info = get_puzzle_info(PIPELINE_DB, 0)
    clue_ids = get_all_clue_ids(PIPELINE_DB, 0)
    results = [resolve_clue(PIPELINE_DB, CLUES_DB, 0, cid) for cid in clue_ids]
    report = format_report(results, puzzle_info)
    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {OUTPUT_FILE}")
    print("\n" + report)


if __name__ == '__main__':
    main()