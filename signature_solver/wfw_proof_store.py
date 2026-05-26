"""Persistence for WFW proof attempts.

This table is the durable WFW authority.  Other systems may propose evidence,
but a row here records whether WFW itself proved the answer assembly.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
SCHEMA_VERSION = "wfw_proof_attempts:v1"


_DDL = """
CREATE TABLE IF NOT EXISTS wfw_proof_attempts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clue_id INTEGER NOT NULL,
    source TEXT,
    puzzle_number TEXT,
    status TEXT NOT NULL,
    proof_source TEXT NOT NULL,
    proof_json TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_INDEX_CLUE_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_wfw_proof_attempts_clue "
    "ON wfw_proof_attempts (clue_id, created_at)"
)

_INDEX_PUZZLE_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_wfw_proof_attempts_puzzle "
    "ON wfw_proof_attempts (source, puzzle_number, status)"
)


def ensure_table(conn=None):
    """Create the WFW proof table if it is missing."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        conn.execute(_DDL)
        conn.execute(_INDEX_CLUE_DDL)
        conn.execute(_INDEX_PUZZLE_DDL)
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def write_wfw_proof_attempt(clue_id, source, puzzle_number, proof_record,
                            conn=None):
    """Append one WFW proof attempt and return its row id."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_table(conn)
        proof = _normalise_proof(proof_record)
        cur = conn.execute(
            """INSERT INTO wfw_proof_attempts
               (clue_id, source, puzzle_number, status, proof_source,
                proof_json, schema_version)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                clue_id,
                source,
                str(puzzle_number) if puzzle_number is not None else None,
                proof["status"],
                proof["source"],
                _dump(proof),
                SCHEMA_VERSION,
            ),
        )
        if own:
            conn.commit()
        return cur.lastrowid
    finally:
        if own:
            conn.close()


def get_latest_wfw_proof_attempt(clue_id, conn=None):
    """Return the newest WFW proof attempt for a clue, or None."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        try:
            row = conn.execute(
                """SELECT id, clue_id, source, puzzle_number, status,
                          proof_source, proof_json, schema_version, created_at
                   FROM wfw_proof_attempts
                   WHERE clue_id = ?
                   ORDER BY created_at DESC, id DESC
                   LIMIT 1""",
                (clue_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        return _row_to_attempt(row)
    finally:
        if own:
            conn.close()


def _normalise_proof(proof_record):
    if hasattr(proof_record, "as_dict"):
        proof = proof_record.as_dict()
    else:
        proof = dict(proof_record or {})
    proof.setdefault("status", "wfw_review")
    proof.setdefault("source", "unknown")
    return proof


def _row_to_attempt(row):
    return {
        "id": row[0],
        "clue_id": row[1],
        "source": row[2],
        "puzzle_number": row[3],
        "status": row[4],
        "proof_source": row[5],
        "proof": _load(row[6]) or {},
        "schema_version": row[7],
        "created_at": row[8],
    }


def _dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load(value):
    if not value:
        return None
    return json.loads(value)
