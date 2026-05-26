"""Persistence for stage-one clue contexts.

This table is additive audit data.  It records the atom-backed context that a
solve attempt received before later assembly, proof, or display decisions.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
SCHEMA_VERSION = "solver_stage_contexts:v1"
STAGE_NAME = "stage_01_context"


_DDL = """
CREATE TABLE IF NOT EXISTS solver_stage_contexts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clue_id INTEGER NOT NULL,
    source TEXT,
    puzzle_number TEXT,
    stage_name TEXT NOT NULL,
    status TEXT NOT NULL,
    schema_version TEXT NOT NULL,
    context_json TEXT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_INDEX_CLUE_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_solver_stage_contexts_clue "
    "ON solver_stage_contexts (clue_id, stage_name, created_at)"
)

_INDEX_PUZZLE_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_solver_stage_contexts_puzzle "
    "ON solver_stage_contexts (source, puzzle_number, stage_name, created_at)"
)


def ensure_table(conn=None):
    """Create the additive stage-context table if it is missing."""
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


def write_stage_context(clue_id, source, puzzle_number, context, status="built",
                        conn=None):
    """Append one serialised stage-one context and return its row id."""
    if context is None:
        return None
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_table(conn)
        cur = conn.execute(
            """INSERT INTO solver_stage_contexts
               (clue_id, source, puzzle_number, stage_name, status,
                schema_version, context_json)
               VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                clue_id,
                source,
                str(puzzle_number) if puzzle_number is not None else None,
                STAGE_NAME,
                status,
                SCHEMA_VERSION,
                _dump(context.as_dict()),
            ),
        )
        if own:
            conn.commit()
        return cur.lastrowid
    finally:
        if own:
            conn.close()


def write_solve_result_context(clue_id, source, puzzle_number, solve_result,
                               status="built", conn=None):
    """Append the stage-one context carried by a SolveResult."""
    context = getattr(solve_result, "stage_one_context", None)
    if context is None:
        context = getattr(solve_result, "clue_context", None)
    if context is None:
        unified = getattr(solve_result, "wfw_unified_result", None)
        context = getattr(unified, "stage_one_context", None)
        if context is None:
            context = getattr(unified, "clue_context", None)
    return write_stage_context(
        clue_id, source, puzzle_number, context, status=status, conn=conn)


def get_latest_stage_context(clue_id, conn=None):
    """Return the newest stage-one context audit row for a clue, or None."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        try:
            row = conn.execute(
                """SELECT id, clue_id, source, puzzle_number, stage_name,
                          status, schema_version, context_json, created_at
                   FROM solver_stage_contexts
                   WHERE clue_id = ? AND stage_name = ?
                   ORDER BY created_at DESC, id DESC
                   LIMIT 1""",
                (clue_id, STAGE_NAME),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        return {
            "id": row[0],
            "clue_id": row[1],
            "source": row[2],
            "puzzle_number": row[3],
            "stage_name": row[4],
            "status": row[5],
            "schema_version": row[6],
            "context": _load(row[7]),
            "created_at": row[8],
        }
    finally:
        if own:
            conn.close()


def _dump(value):
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load(value):
    if not value:
        return None
    return json.loads(value)
