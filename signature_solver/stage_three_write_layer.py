"""Opt-in persistence for Stage Three dry-run actions.

The proof gate itself stays read-only.  This module is the explicit write
boundary: it stores Stage Three artifacts, human review items, and concrete DB
enrichment candidates in the existing queues.
"""
from __future__ import annotations

import sqlite3
from pathlib import Path

from .atomic_parse_store import (
    finish_atomic_run,
    start_atomic_run,
    write_atomic_artifact,
    write_review_item,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
REF_DB = PROJECT_ROOT / "data" / "cryptic_new.db"
SOLVER_VERSION = "stage_three_proof:v1"


def write_stage_three_puzzle_results(summary, conn=None, ref_conn=None):
    """Persist one Stage Three puzzle summary and return write counts."""
    own_conn = conn is None
    own_ref = ref_conn is None
    if own_conn:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
        conn.row_factory = sqlite3.Row
    if own_ref:
        ref_conn = sqlite3.connect(str(REF_DB), timeout=30)
        ref_conn.row_factory = sqlite3.Row
    run_id = None
    counts = {
        "run_id": None,
        "artifacts": 0,
        "review_items": 0,
        "pending_enrichments": 0,
        "skipped_pending": 0,
        "skipped_review_items": 0,
    }
    try:
        run_id = start_atomic_run(
            summary["source"],
            summary["puzzle_number"],
            summary["total"],
            conn=conn,
            solver_version=SOLVER_VERSION,
        )
        counts["run_id"] = run_id
        for clue in summary.get("clues") or []:
            artifact_id = _write_stage_three_artifact(conn, clue)
            counts["artifacts"] += 1
            for item in clue.get("review_items") or []:
                if _review_item_exists(conn, clue["clue_id"], item):
                    counts["skipped_review_items"] += 1
                    continue
                write_review_item(
                    run_id,
                    artifact_id,
                    clue["clue_id"],
                    item["review_type"],
                    item["summary"],
                    item.get("payload") or {},
                    conn=conn,
                )
                counts["review_items"] += 1
            for row in clue.get("pending_enrichments") or []:
                if _skip_pending_enrichment(conn, ref_conn, row):
                    counts["skipped_pending"] += 1
                    continue
                _insert_pending_enrichment(conn, row)
                counts["pending_enrichments"] += 1
        finish_atomic_run(
            run_id,
            summary.get("pass", 0),
            summary.get("review", 0),
            conn=conn,
        )
        if own_conn:
            conn.commit()
        return counts
    except Exception:
        if own_conn:
            conn.rollback()
        raise
    finally:
        if own_ref:
            ref_conn.close()
        if own_conn:
            conn.close()


def _write_stage_three_artifact(conn, clue):
    artifact = {
        "status": (
            "solved" if clue.get("status") == "PASS" else "partial"
        ),
        "confidence": None,
        "solver_version": SOLVER_VERSION,
        "clue_context": clue.get("stage_one_context"),
        "annotations": [],
        "gt2_bundles": [],
        "token_parses": [],
        "stage_two_casefile": clue.get("stage_two_casefile"),
        "stage_three_proof": clue.get("stage_three_proof"),
        "wfw": [{
            "schema": "stage_three_puzzle_result:v1",
            "status": clue.get("status"),
            "bucket": clue.get("bucket"),
            "failed_checks": clue.get("failed_checks") or [],
            "mechanical_failed": clue.get("mechanical_failed") or [],
            "purpose_failed": clue.get("purpose_failed") or [],
            "review_items": clue.get("review_items") or [],
            "pending_enrichments": clue.get("pending_enrichments") or [],
        }],
    }
    return write_atomic_artifact(
        clue["clue_id"],
        clue.get("clue") or "",
        clue.get("answer") or "",
        artifact,
        conn=conn,
    )


def _review_item_exists(conn, clue_id, item):
    row = conn.execute(
        """SELECT 1
           FROM atomic_parse_review_items
           WHERE clue_id = ?
             AND status = 'open'
             AND review_type = ?
             AND summary = ?
           LIMIT 1""",
        (clue_id, item.get("review_type"), item.get("summary")),
    ).fetchone()
    return row is not None


def _skip_pending_enrichment(conn, ref_conn, row):
    return (
        _pending_exists(conn, row)
        or _pending_rejected(conn, row)
        or _already_in_reference_db(ref_conn, row)
    )


def _pending_exists(conn, row):
    existing = conn.execute(
        """SELECT 1
           FROM pending_enrichments
           WHERE type = ?
             AND LOWER(word) = ?
             AND UPPER(letters) = ?
           LIMIT 1""",
        (
            row.get("type"),
            (row.get("word") or "").lower(),
            (row.get("letters") or "").upper(),
        ),
    ).fetchone()
    return existing is not None


def _pending_rejected(conn, row):
    rejected = conn.execute(
        """SELECT 1
           FROM rejected_enrichments
           WHERE type = ?
             AND LOWER(word) = ?
             AND UPPER(letters) = ?
           LIMIT 1""",
        (
            row.get("type"),
            (row.get("word") or "").lower(),
            (row.get("letters") or "").upper(),
        ),
    ).fetchone()
    return rejected is not None


def _already_in_reference_db(ref_conn, row):
    etype = row.get("type")
    word = row.get("word") or ""
    letters = row.get("letters") or ""
    if etype == "synonym":
        existing = ref_conn.execute(
            """SELECT 1
               FROM synonyms_pairs
               WHERE LOWER(word) = LOWER(?)
                 AND UPPER(synonym) = UPPER(?)
               LIMIT 1""",
            (word, letters),
        ).fetchone()
        return existing is not None
    if etype == "definition":
        existing = ref_conn.execute(
            """SELECT 1
               FROM definition_answers_augmented
               WHERE LOWER(definition) = LOWER(?)
                 AND UPPER(answer) = UPPER(?)
               LIMIT 1""",
            (word, letters),
        ).fetchone()
        return existing is not None
    return False


def _insert_pending_enrichment(conn, row):
    conn.execute(
        """INSERT INTO pending_enrichments
           (type, word, letters, answer, clue_text, source, puzzle_number,
            created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, CURRENT_TIMESTAMP)""",
        (
            row.get("type"),
            row.get("word"),
            (row.get("letters") or "").upper(),
            row.get("answer") or "",
            row.get("clue_text") or "",
            row.get("source"),
            row.get("puzzle_number"),
        ),
    )
