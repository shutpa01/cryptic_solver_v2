#!/usr/bin/env python3
"""
Pipeline Persistence Module

Maintains SQLite database tracking each pipeline stage.
Keeps only the latest run (run_id=0), clearing all data on each new run.

Location: data/pipeline_persistence.py
Database: data/pipeline_stages.db
"""

import sqlite3
import json
from datetime import datetime
from pathlib import Path
from typing import List, Dict, Any, Optional
import re

# Database path (same folder as this file)
DB_PATH = Path(__file__).parent / 'pipeline_stages.db'


def get_connection():
    """Get database connection."""
    return sqlite3.connect(DB_PATH)


def init_tables():
    """Create all tables if they don't exist."""
    conn = get_connection()
    cursor = conn.cursor()

    # Meta table for run management
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS pipeline_meta (
            key TEXT PRIMARY KEY,
            value TEXT
        )
    """)

    # Stage: Input (initial clue cohort)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_input (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            clue_id INTEGER,
            clue_text TEXT,
            answer TEXT,
            enumeration TEXT,
            source TEXT,
            puzzle_number TEXT,
            wordplay_type TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_input_run ON stage_input(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_input_clue ON stage_input(clue_id)")

    # Stage: Definition (after definition candidate matching)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_definition (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            clue_id INTEGER,
            clue_text TEXT,
            answer TEXT,
            candidates TEXT,
            candidate_count INTEGER,
            answer_in_candidates INTEGER,
            support TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_def_run ON stage_definition(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_def_clue ON stage_definition(clue_id)")

    # Stage: Anagram (after brute force detection)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_anagram (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            clue_id INTEGER,
            clue_text TEXT,
            answer TEXT,
            hit_found INTEGER,
            fodder_words TEXT,
            fodder_letters TEXT,
            matched_candidate TEXT,
            solve_type TEXT,
            confidence TEXT,
            unused_words TEXT,
            all_hypotheses TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ana_run ON stage_anagram(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_ana_clue ON stage_anagram(clue_id)")

    # Stage: Evidence (after evidence scoring/ranking)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_evidence (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            clue_id INTEGER,
            clue_text TEXT,
            answer TEXT,
            evidence_found INTEGER,
            top_candidate TEXT,
            top_candidate_score REAL,
            answer_rank_original INTEGER,
            answer_rank_evidence INTEGER,
            ranking_improved INTEGER,
            scored_candidates TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_evi_run ON stage_evidence(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_evi_clue ON stage_evidence(clue_id)")

    # Stage: Double Definition (checked FIRST, before definition gate)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_dd (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            clue_id INTEGER,
            clue_text TEXT,
            answer TEXT,
            hit_found INTEGER,
            matched_answer TEXT,
            windows TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dd_run ON stage_dd(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_dd_clue ON stage_dd(clue_id)")

    # Stage: Definition Failed (clues that failed the definition gate)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_definition_failed (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            clue_id INTEGER,
            clue_text TEXT,
            answer TEXT,
            candidate_count INTEGER,
            candidates_sample TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deffail_run ON stage_definition_failed(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_deffail_clue ON stage_definition_failed(clue_id)")

    # Stage: Lurker (hidden word detection)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_lurker (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            clue_id INTEGER,
            clue_text TEXT,
            answer TEXT,
            hit_found INTEGER,
            lurker_answer TEXT,
            container_text TEXT,
            start_pos INTEGER,
            end_pos INTEGER
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lurker_run ON stage_lurker(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_lurker_clue ON stage_lurker(clue_id)")

    # Stage: Compound (after compound analysis with explanations)
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_compound (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            clue_id INTEGER,
            clue_text TEXT,
            answer TEXT,
            formula TEXT,
            quality TEXT,
            definition_window TEXT,
            anagram_fodder TEXT,
            anagram_indicator TEXT,
            substitutions TEXT,
            operation_indicators TEXT,
            positional_indicators TEXT,
            fully_resolved INTEGER,
            letters_still_needed TEXT,
            unresolved_words TEXT,
            remaining_unresolved TEXT,
            breakdown TEXT,
            word_roles TEXT
        )
    """)
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_comp_run ON stage_compound(run_id)")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_comp_clue ON stage_compound(clue_id)")

    conn.commit()
    conn.close()


def start_run() -> int:
    """
    Start a new pipeline run.
    Always uses run_id=0, clearing all previous data.
    Returns the run_id to use (always 0).
    """
    init_tables()

    conn = get_connection()
    cursor = conn.cursor()

    # Always use run_id=0 - single version only
    new_run_id = 0

    # Clear ALL old data from ALL runs
    tables = ['stage_input', 'stage_dd', 'stage_definition', 'stage_definition_failed',
              'stage_anagram', 'stage_lurker', 'stage_evidence', 'stage_compound']
    for table in tables:
        cursor.execute(f"DELETE FROM {table}")
        deleted = cursor.rowcount
        if deleted > 0:
            print(f"  Cleared {deleted} old records from {table}")

    # Reset auto-increment sequences
    cursor.execute("DELETE FROM sqlite_sequence")

    # Update meta
    cursor.execute("""
        INSERT OR REPLACE INTO pipeline_meta (key, value) 
        VALUES ('current_run_id', ?)
    """, (str(new_run_id),))

    cursor.execute("""
        INSERT OR REPLACE INTO pipeline_meta (key, value) 
        VALUES ('run_timestamp', ?)
    """, (datetime.now().isoformat(),))

    conn.commit()
    conn.close()

    print(f"Pipeline run started: run_id={new_run_id}")
    return new_run_id


def save_stage_input(run_id: int, records: List[Dict[str, Any]]):
    """Save input stage data."""
    conn = get_connection()
    cursor = conn.cursor()

    for rec in records:
        cursor.execute("""
            INSERT INTO stage_input 
            (run_id, clue_id, clue_text, answer, enumeration, source, puzzle_number, wordplay_type)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            rec.get('id'),
            rec.get('clue_text') or rec.get('clue'),
            rec.get('answer'),
            rec.get('enumeration'),
            rec.get('source'),
            rec.get('puzzle_number'),
            rec.get('wordplay_type')
        ))

    conn.commit()
    conn.close()
    print(f"Saved {len(records)} records to stage_input (run_id={run_id})")


def save_stage_definition(run_id: int, records: List[Dict[str, Any]]):
    """Save definition stage data."""
    conn = get_connection()
    cursor = conn.cursor()

    for rec in records:
        candidates = rec.get('definition_candidates', [])

        # Normalize: strip all non-alpha characters, uppercase
        # Matches gate logic in pipeline_simulator.py using norm_letters()
        answer_normalized = re.sub(r"[^A-Za-z]", "", rec.get('answer', '') or "").upper()
        candidates_normalized = [re.sub(r"[^A-Za-z]", "", c or "").upper() for c in candidates] if candidates else []
        answer_in_candidates = 1 if answer_normalized in candidates_normalized else 0

        cursor.execute("""
            INSERT INTO stage_definition 
            (run_id, clue_id, clue_text, answer, candidates, candidate_count, answer_in_candidates, support)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            rec.get('id'),
            rec.get('clue_text') or rec.get('clue'),
            rec.get('answer'),
            json.dumps(candidates[:50]) if candidates else '[]',
            # Limit to 50 for storage
            len(candidates) if candidates else 0,
            answer_in_candidates,
            json.dumps(rec.get('support', {}))
        ))

    conn.commit()
    conn.close()
    print(f"Saved {len(records)} records to stage_definition (run_id={run_id})")


def save_stage_dd(run_id: int, records: List[Dict[str, Any]]):
    """Save double definition stage data."""
    conn = get_connection()
    cursor = conn.cursor()

    for rec in records:
        dd_hits = rec.get('double_definition', [])
        top_hit = dd_hits[0] if dd_hits else {}

        cursor.execute("""
            INSERT INTO stage_dd 
            (run_id, clue_id, clue_text, answer, hit_found, matched_answer, windows)
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            rec.get('id'),
            rec.get('clue_text') or rec.get('clue'),
            rec.get('answer'),
            1 if dd_hits else 0,
            top_hit.get('answer', ''),
            json.dumps(top_hit.get('windows', []))
        ))

    conn.commit()
    conn.close()
    print(f"Saved {len(records)} records to stage_dd (run_id={run_id})")


def save_stage_definition_failed(run_id: int, records: List[Dict[str, Any]]):
    """Save definition failed stage data (clues that failed the gate)."""
    conn = get_connection()
    cursor = conn.cursor()

    for rec in records:
        candidates = rec.get('definition_candidates', [])

        cursor.execute("""
            INSERT INTO stage_definition_failed 
            (run_id, clue_id, clue_text, answer, candidate_count, candidates_sample)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            rec.get('id'),
            rec.get('clue_text') or rec.get('clue'),
            rec.get('answer'),
            len(candidates) if candidates else 0,
            json.dumps(candidates[:20]) if candidates else '[]'  # Sample for debugging
        ))

    conn.commit()
    conn.close()
    print(f"Saved {len(records)} records to stage_definition_failed (run_id={run_id})")


def save_stage_lurker(run_id: int, records: List[Dict[str, Any]]):
    """Save lurker stage data."""
    conn = get_connection()
    cursor = conn.cursor()

    for rec in records:
        lurkers = rec.get('lurkers', [])
        top_hit = lurkers[0] if lurkers else {}

        cursor.execute("""
            INSERT INTO stage_lurker 
            (run_id, clue_id, clue_text, answer, hit_found, lurker_answer, 
             container_text, start_pos, end_pos)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            rec.get('id'),
            rec.get('clue_text') or rec.get('clue'),
            rec.get('answer'),
            1 if lurkers else 0,
            top_hit.get('answer', ''),
            top_hit.get('container', '') or top_hit.get('letters', ''),
            top_hit.get('span', (-1, -1))[0],
            top_hit.get('span', (-1, -1))[1]
        ))

    conn.commit()
    conn.close()
    print(f"Saved {len(records)} records to stage_lurker (run_id={run_id})")


def save_stage_anagram(run_id: int, records: List[Dict[str, Any]]):
    """Save anagram stage data."""
    conn = get_connection()
    cursor = conn.cursor()

    for rec in records:
        anagrams = rec.get('anagrams', [])
        top_hit = anagrams[0] if anagrams else {}

        cursor.execute("""
            INSERT INTO stage_anagram 
            (run_id, clue_id, clue_text, answer, hit_found, fodder_words, fodder_letters,
             matched_candidate, solve_type, confidence, unused_words, all_hypotheses)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            rec.get('id'),
            rec.get('clue_text') or rec.get('clue'),
            rec.get('answer'),
            1 if anagrams and top_hit.get('solve_type') == 'anagram_exact' else 0,
            json.dumps(top_hit.get('fodder_words', [])),
            top_hit.get('fodder_letters', ''),
            top_hit.get('answer', ''),
            top_hit.get('solve_type', ''),
            str(top_hit.get('confidence', '')),
            json.dumps(top_hit.get('unused_words', [])),
            json.dumps(anagrams[:10])  # Store top 10 hypotheses
        ))

    conn.commit()
    conn.close()
    print(f"Saved {len(records)} records to stage_anagram (run_id={run_id})")


def save_stage_evidence(run_id: int, records: List[Dict[str, Any]]):
    """Save evidence stage data."""
    conn = get_connection()
    cursor = conn.cursor()

    for rec in records:
        evidence = rec.get('evidence_analysis', {})
        scored = evidence.get('scored_candidates', [])
        top = scored[0] if scored else {}

        # Convert scored_candidates to JSON-serializable format
        scored_serializable = []
        for sc in scored[:10]:
            sc_dict = {}
            for k, v in sc.items():
                if hasattr(v, '__dict__'):
                    # Convert dataclass/object to dict
                    sc_dict[k] = v.__dict__
                else:
                    sc_dict[k] = v
            scored_serializable.append(sc_dict)

        cursor.execute("""
            INSERT INTO stage_evidence 
            (run_id, clue_id, clue_text, answer, evidence_found, top_candidate,
             top_candidate_score, answer_rank_original, answer_rank_evidence,
             ranking_improved, scored_candidates)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            rec.get('id'),
            rec.get('clue_text') or rec.get('clue'),
            rec.get('answer'),
            1 if evidence.get('evidence_found') else 0,
            top.get('candidate', ''),
            top.get('score', 0.0),
            evidence.get('answer_rank_original'),
            evidence.get('answer_rank_evidence'),
            1 if evidence.get('ranking_improved') else 0,
            json.dumps(scored_serializable)
        ))

    conn.commit()
    conn.close()
    print(f"Saved {len(records)} records to stage_evidence (run_id={run_id})")


def save_stage_compound(run_id: int, records: List[Dict[str, Any]]):
    """Save compound analysis stage data."""
    conn = get_connection()
    cursor = conn.cursor()

    for rec in records:
        explanation = rec.get('explanation', {})
        compound = rec.get('compound_solution') or {}
        anagram_comp = rec.get('anagram_component') or {}
        word_roles = rec.get('word_roles', [])

        # Convert word_roles to serializable format
        word_roles_data = []
        for wr in word_roles:
            if hasattr(wr, '__dict__'):
                word_roles_data.append(wr.__dict__)
            elif isinstance(wr, dict):
                word_roles_data.append(wr)

        # Get likely_answer and db_answer (with fallback for old format)
        likely_answer = rec.get('likely_answer', rec.get('answer', ''))
        db_answer = rec.get('db_answer', rec.get('answer', ''))

        cursor.execute("""
            INSERT INTO stage_compound 
            (run_id, clue_id, clue_text, answer, formula, quality, definition_window,
             anagram_fodder, anagram_indicator, substitutions, operation_indicators,
             positional_indicators, fully_resolved, letters_still_needed, unresolved_words,
             remaining_unresolved, breakdown, word_roles)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            rec.get('id'),
            rec.get('clue_text') or rec.get('clue'),
            f"{likely_answer}|{db_answer}",  # Store both, pipe-separated
            explanation.get('formula', ''),
            explanation.get('quality', ''),
            rec.get('definition_window', ''),
            json.dumps(anagram_comp.get('fodder_words', [])),
            anagram_comp.get('indicator', ''),
            json.dumps(compound.get('substitutions', [])),
            json.dumps(compound.get('operation_indicators', [])),
            json.dumps(compound.get('positional_indicators', [])),
            compound.get('fully_resolved', 0),
            compound.get('letters_still_needed', ''),
            json.dumps(compound.get('unresolved_words', [])),
            json.dumps(rec.get('remaining_unresolved', [])),
            json.dumps(explanation.get('breakdown', [])),
            json.dumps(word_roles_data)
        ))

    conn.commit()
    conn.close()
    print(f"Saved {len(records)} records to stage_compound (run_id={run_id})")


def save_stage(stage_name: str, run_id: int, records: List[Dict[str, Any]]):
    """
    Generic save function - routes to appropriate stage saver.
    """
    savers = {
        'input': save_stage_input,
        'dd': save_stage_dd,
        'definition': save_stage_definition,
        'definition_failed': save_stage_definition_failed,
        'anagram': save_stage_anagram,
        'lurker': save_stage_lurker,
        'evidence': save_stage_evidence,
        'compound': save_stage_compound
    }

    saver = savers.get(stage_name)
    if saver:
        saver(run_id, records)
    else:
        print(f"WARNING: Unknown stage '{stage_name}'")


# ============================================================================
# QUERY HELPERS
# ============================================================================

def get_clue_journey(clue_text_pattern: str) -> Dict[str, Any]:
    """
    Get a clue's data through all stages.
    Uses LIKE pattern matching on clue_text.
    Returns data from both runs for comparison.
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    result = {'run_0': {}, 'run_1': {}}

    stages = ['stage_input', 'stage_dd', 'stage_definition', 'stage_definition_failed',
              'stage_anagram', 'stage_lurker', 'stage_evidence', 'stage_compound']

    for stage in stages:
        for run_id in [0, 1]:
            cursor.execute(f"""
                SELECT * FROM {stage} 
                WHERE run_id = ? AND clue_text LIKE ?
            """, (run_id, f'%{clue_text_pattern}%'))

            row = cursor.fetchone()
            if row:
                result[f'run_{run_id}'][stage] = dict(row)

    conn.close()
    return result


def get_stage_summary(run_id: int = None) -> Dict[str, Any]:
    """
    Get summary statistics for each stage.
    If run_id is None, uses current run.
    """
    conn = get_connection()
    cursor = conn.cursor()

    if run_id is None:
        cursor.execute("SELECT value FROM pipeline_meta WHERE key = 'current_run_id'")
        row = cursor.fetchone()
        run_id = int(row[0]) if row else 0

    summary = {'run_id': run_id}

    # Input count
    cursor.execute("SELECT COUNT(*) FROM stage_input WHERE run_id = ?", (run_id,))
    summary['input_count'] = cursor.fetchone()[0]

    # DD stats (checked first in pipeline)
    cursor.execute("""
        SELECT COUNT(*), SUM(hit_found) 
        FROM stage_dd WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    summary['dd_count'] = row[0]
    summary['dd_hits'] = row[1] or 0

    # Definition stats
    cursor.execute("""
        SELECT COUNT(*), SUM(answer_in_candidates) 
        FROM stage_definition WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    summary['definition_count'] = row[0]
    summary['definition_answer_matches'] = row[1] or 0

    # Definition failed stats
    cursor.execute("SELECT COUNT(*) FROM stage_definition_failed WHERE run_id = ?", (run_id,))
    summary['definition_failed_count'] = cursor.fetchone()[0]

    # Anagram stats
    cursor.execute("""
        SELECT COUNT(*), SUM(hit_found) 
        FROM stage_anagram WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    summary['anagram_count'] = row[0]
    summary['anagram_hits'] = row[1] or 0

    # Lurker stats
    cursor.execute("""
        SELECT COUNT(*), SUM(hit_found) 
        FROM stage_lurker WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    summary['lurker_count'] = row[0]
    summary['lurker_hits'] = row[1] or 0

    # Evidence stats
    cursor.execute("""
        SELECT COUNT(*), SUM(evidence_found), SUM(ranking_improved)
        FROM stage_evidence WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    summary['evidence_count'] = row[0]
    summary['evidence_found'] = row[1] or 0
    summary['ranking_improved'] = row[2] or 0

    # Compound stats
    cursor.execute("""
        SELECT COUNT(*), SUM(fully_resolved),
               SUM(CASE WHEN quality = 'solved' THEN 1 ELSE 0 END),
               SUM(CASE WHEN quality = 'high' THEN 1 ELSE 0 END),
               SUM(CASE WHEN quality = 'medium' THEN 1 ELSE 0 END),
               SUM(CASE WHEN quality = 'low' THEN 1 ELSE 0 END)
        FROM stage_compound WHERE run_id = ?
    """, (run_id,))
    row = cursor.fetchone()
    summary['compound_count'] = row[0]
    summary['fully_resolved'] = row[1] or 0
    summary['quality_solved'] = row[2] or 0
    summary['quality_high'] = row[3] or 0
    summary['quality_medium'] = row[4] or 0
    summary['quality_low'] = row[5] or 0

    conn.close()
    return summary


def compare_runs() -> Dict[str, Any]:
    """Compare statistics between run 0 and run 1."""
    return {
        'run_0': get_stage_summary(0),
        'run_1': get_stage_summary(1)
    }


def find_failures(run_id: int = None, stage: str = 'anagram') -> List[Dict]:
    """
    Find clues that failed at a specific stage.
    """
    conn = get_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    if run_id is None:
        cursor.execute("SELECT value FROM pipeline_meta WHERE key = 'current_run_id'")
        row = cursor.fetchone()
        run_id = int(row[0]) if row else 0

    if stage == 'definition':
        cursor.execute("""
            SELECT * FROM stage_definition 
            WHERE run_id = ? AND answer_in_candidates = 0
        """, (run_id,))
    elif stage == 'definition_failed':
        cursor.execute("""
            SELECT * FROM stage_definition_failed 
            WHERE run_id = ?
        """, (run_id,))
    elif stage == 'dd':
        cursor.execute("""
            SELECT * FROM stage_dd 
            WHERE run_id = ? AND hit_found = 0
        """, (run_id,))
    elif stage == 'anagram':
        cursor.execute("""
            SELECT * FROM stage_anagram 
            WHERE run_id = ? AND hit_found = 0
        """, (run_id,))
    elif stage == 'lurker':
        cursor.execute("""
            SELECT * FROM stage_lurker 
            WHERE run_id = ? AND hit_found = 0
        """, (run_id,))
    elif stage == 'compound':
        cursor.execute("""
            SELECT * FROM stage_compound 
            WHERE run_id = ? AND fully_resolved = 0
        """, (run_id,))
    else:
        return []

    results = [dict(row) for row in cursor.fetchall()]
    conn.close()
    return results


def print_clue_journey(clue_text_pattern: str):
    """Pretty print a clue's journey through all stages."""
    journey = get_clue_journey(clue_text_pattern)

    print(f"\n{'=' * 70}")
    print(f"CLUE JOURNEY: *{clue_text_pattern}*")
    print('=' * 70)

    for run_key in ['run_0', 'run_1']:
        run_data = journey[run_key]
        if not run_data:
            continue

        print(f"\n--- {run_key.upper()} ---")

        for stage, data in run_data.items():
            print(f"\n  {stage}:")
            for key, value in data.items():
                if key in ('id', 'run_id'):
                    continue
                # Truncate long values
                val_str = str(value)
                if len(val_str) > 80:
                    val_str = val_str[:80] + '...'
                print(f"    {key}: {val_str}")


# ============================================================================
# CLI for quick queries
# ============================================================================

if __name__ == "__main__":
    import sys

    if len(sys.argv) < 2:
        print("Usage:")
        print("  python pipeline_persistence.py summary")
        print("  python pipeline_persistence.py compare")
        print("  python pipeline_persistence.py journey <clue_pattern>")
        print("  python pipeline_persistence.py failures <stage>")
        print("  python pipeline_persistence.py clean    # Reset database")
        print("  python pipeline_persistence.py counts   # Show actual row counts")
        sys.exit(0)

    cmd = sys.argv[1]

    if cmd == 'summary':
        print("\n" + "=" * 60)
        print("PIPELINE STAGE SUMMARY - BY RUN")
        print("=" * 60)

        for run_id in [0, 1]:
            summary = get_stage_summary(run_id)
            if summary.get('input_count', 0) == 0:
                continue

            print(f"\n--- RUN {run_id} ---")
            print(f"  Input clues:              {summary.get('input_count', 0)}")
            print(
                f"  DD hits:                  {summary.get('dd_hits', 0)} / {summary.get('dd_count', 0)}")
            print(
                f"  Definition failed:        {summary.get('definition_failed_count', 0)}")
            print(
                f"  Definition matches:       {summary.get('definition_answer_matches', 0)} / {summary.get('definition_count', 0)}")
            print(
                f"  Anagram hits:             {summary.get('anagram_hits', 0)} / {summary.get('anagram_count', 0)}")
            print(
                f"  Lurker hits:              {summary.get('lurker_hits', 0)} / {summary.get('lurker_count', 0)}")
            print(
                f"  Evidence found:           {summary.get('evidence_found', 0)} / {summary.get('evidence_count', 0)}")
            print(f"  Ranking improved:         {summary.get('ranking_improved', 0)}")
            print(
                f"  Compound - fully resolved:{summary.get('fully_resolved', 0)} / {summary.get('compound_count', 0)}")
            print(f"  Quality breakdown:")
            print(f"    solved:  {summary.get('quality_solved', 0)}")
            print(f"    high:    {summary.get('quality_high', 0)}")
            print(f"    medium:  {summary.get('quality_medium', 0)}")
            print(f"    low:     {summary.get('quality_low', 0)}")

    elif cmd == 'compare':
        print("\n" + "=" * 60)
        print("PIPELINE RUN COMPARISON")
        print("=" * 60)

        s0 = get_stage_summary(0)
        s1 = get_stage_summary(1)

        print(f"\n{'Metric':<30} {'Run 0':>10} {'Run 1':>10} {'Diff':>10}")
        print("-" * 60)

        metrics = [
            ('Input clues', 'input_count'),
            ('DD hits', 'dd_hits'),
            ('Definition failed', 'definition_failed_count'),
            ('Definition matches', 'definition_answer_matches'),
            ('Anagram hits', 'anagram_hits'),
            ('Lurker hits', 'lurker_hits'),
            ('Evidence found', 'evidence_found'),
            ('Ranking improved', 'ranking_improved'),
            ('Fully resolved', 'fully_resolved'),
            ('Quality: solved', 'quality_solved'),
            ('Quality: high', 'quality_high'),
            ('Quality: medium', 'quality_medium'),
            ('Quality: low', 'quality_low'),
        ]

        for label, key in metrics:
            v0 = s0.get(key, 0) or 0
            v1 = s1.get(key, 0) or 0
            diff = v1 - v0
            diff_str = f"+{diff}" if diff > 0 else str(diff)
            print(f"{label:<30} {v0:>10} {v1:>10} {diff_str:>10}")

    elif cmd == 'journey' and len(sys.argv) > 2:
        pattern = ' '.join(sys.argv[2:])
        print_clue_journey(pattern)

    elif cmd == 'failures' and len(sys.argv) > 2:
        stage = sys.argv[2]
        failures = find_failures(stage=stage)
        print(f"Found {len(failures)} failures at {stage} stage:")
        for f in failures[:10]:
            print(f"  - {f.get('clue_text', '')[:60]}...")

    elif cmd == 'counts':
        # Show actual row counts per run_id
        conn = get_connection()
        cursor = conn.cursor()

        print("\n" + "=" * 60)
        print("ACTUAL ROW COUNTS BY RUN_ID")
        print("=" * 60)

        tables = ['stage_input', 'stage_dd', 'stage_definition', 'stage_definition_failed',
                  'stage_anagram', 'stage_lurker', 'stage_evidence', 'stage_compound']

        print(f"\n{'Table':<25} {'Run 0':>10} {'Run 1':>10} {'Total':>10}")
        print("-" * 55)

        for table in tables:
            cursor.execute(f"""
                SELECT run_id, COUNT(*) FROM {table} GROUP BY run_id
            """)
            counts = {row[0]: row[1] for row in cursor.fetchall()}
            r0 = counts.get(0, 0)
            r1 = counts.get(1, 0)
            print(f"{table:<25} {r0:>10} {r1:>10} {r0 + r1:>10}")

        conn.close()

    elif cmd == 'clean':
        # Reset the entire database
        confirm = input("This will DELETE ALL pipeline data. Type 'yes' to confirm: ")
        if confirm.lower() == 'yes':
            conn = get_connection()
            cursor = conn.cursor()

            tables = ['stage_input', 'stage_dd', 'stage_definition', 'stage_definition_failed',
                      'stage_anagram', 'stage_lurker', 'stage_evidence', 'stage_compound',
                      'pipeline_meta']

            for table in tables:
                cursor.execute(f"DELETE FROM {table}")
                print(f"  Cleared {table}")

            # Reset sequences
            cursor.execute("DELETE FROM sqlite_sequence")
            print("  Reset auto-increment sequences")

            conn.commit()
            conn.close()
            print("\nDatabase cleaned. Next run will start fresh with run_id=0")
        else:
            print("Cancelled.")

    else:
        print(f"Unknown command: {cmd}")