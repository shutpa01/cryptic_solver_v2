"""Regression tests for the opt-in Stage Three write layer."""

import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.atomic_parse_store import ensure_table
from signature_solver.stage_three_write_layer import (
    write_stage_three_puzzle_results,
)


def run_tests():
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    ref = sqlite3.connect(":memory:")
    ref.row_factory = sqlite3.Row
    _create_tables(conn, ref)

    summary = {
        "source": "test",
        "puzzle_number": "1",
        "total": 2,
        "pass": 0,
        "review": 2,
        "clues": [{
            "clue_id": 1,
            "clue": "Large reptile heading off fabulous bird",
            "answer": "ROC",
            "stage_one_context": {
                "schema": "clue_context:v1",
                "tokens": [{"index": 0, "text": "Large"}],
            },
            "stage_two_casefile": {
                "schema": "stage_two_casefile:v1",
                "status": "needs_enrichment",
                "source_candidates": [{"text": "Large reptile"}],
            },
            "stage_three_proof": {
                "schema": "stage_three_proof:v1",
                "status": "REVIEW",
                "checks": [{"name": "conditional_facts",
                            "status": "REVIEW"}],
            },
            "status": "REVIEW",
            "bucket": "db_enrichment_review",
            "failed_checks": ["conditional_facts"],
            "mechanical_failed": [],
            "purpose_failed": ["word_purpose_candidates"],
            "review_items": [{
                "clue_id": 1,
                "review_type": "stage_three:mechanism_indicator_evidence",
                "summary": "Purpose evidence needed for heading off",
                "payload": {"request": {"text": "heading off"}},
            }],
            "pending_enrichments": [{
                "type": "synonym",
                "word": "Large reptile",
                "letters": "CROC",
                "answer": "ROC",
                "clue_text": "Large reptile heading off fabulous bird",
                "source": "test",
                "puzzle_number": "1",
            }],
        }, {
            "clue_id": 2,
            "clue": "Already known",
            "answer": "AMI",
            "status": "REVIEW",
            "bucket": "db_enrichment_review",
            "failed_checks": ["conditional_facts"],
            "mechanical_failed": [],
            "purpose_failed": [],
            "review_items": [],
            "pending_enrichments": [{
                "type": "synonym",
                "word": "French friend",
                "letters": "AMI",
                "answer": "AMI",
                "clue_text": "Already known",
                "source": "test",
                "puzzle_number": "1",
            }],
        }],
    }

    counts = write_stage_three_puzzle_results(summary, conn=conn, ref_conn=ref)
    assert counts["artifacts"] == 2
    assert counts["review_items"] == 1
    assert counts["pending_enrichments"] == 1
    assert counts["skipped_pending"] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM atomic_parse_artifacts").fetchone()[0] == 2
    assert conn.execute(
        "SELECT COUNT(*) FROM atomic_parse_review_items").fetchone()[0] == 1
    assert conn.execute(
        "SELECT COUNT(*) FROM pending_enrichments").fetchone()[0] == 1
    artifact = conn.execute(
        """SELECT clue_context_json, stage_two_casefile_json,
                  stage_three_proof_json
           FROM atomic_parse_artifacts
           WHERE clue_id = 1"""
    ).fetchone()
    assert '"tokens"' in artifact["clue_context_json"]
    assert '"stage_two_casefile:v1"' in artifact["stage_two_casefile_json"]
    assert '"stage_three_proof:v1"' in artifact["stage_three_proof_json"]

    second = write_stage_three_puzzle_results(summary, conn=conn, ref_conn=ref)
    assert second["pending_enrichments"] == 0
    assert second["skipped_pending"] == 2
    assert second["review_items"] == 0
    assert second["skipped_review_items"] == 1

    print("Stage Three write layer contract passed")
    return True


def _create_tables(conn, ref):
    ensure_table(conn)
    conn.execute("""
        CREATE TABLE pending_enrichments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            word TEXT NOT NULL,
            letters TEXT NOT NULL,
            answer TEXT,
            clue_text TEXT,
            source TEXT,
            puzzle_number INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    conn.execute("""
        CREATE TABLE rejected_enrichments (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            type TEXT NOT NULL,
            word TEXT NOT NULL,
            letters TEXT NOT NULL,
            rejected_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)
    ref.execute("""
        CREATE TABLE synonyms_pairs (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            word TEXT NOT NULL,
            synonym TEXT NOT NULL,
            source TEXT
        )
    """)
    ref.execute("""
        CREATE TABLE definition_answers_augmented (
            definition TEXT,
            answer TEXT,
            source TEXT
        )
    """)
    ref.execute(
        "INSERT INTO synonyms_pairs (word, synonym) VALUES (?, ?)",
        ("French friend", "AMI"),
    )
    conn.commit()
    ref.commit()


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
