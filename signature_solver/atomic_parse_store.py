"""Persistence for evidence-preserving atomic parser artifacts.

This is intentionally additive.  The table created here does not replace
``structured_explanations`` or ``clue_word_roles``; it keeps the richer
token/span/atom/parse record that those older display tables cannot hold.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"
SCHEMA_VERSION = "atomic_parse_artifacts:v1"


_DDL = """
CREATE TABLE IF NOT EXISTS atomic_parse_artifacts (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clue_id INTEGER NOT NULL,
    clue_text TEXT NOT NULL,
    answer TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL,
    solver_version TEXT NOT NULL,
    clue_context_json TEXT,
    annotations_json TEXT,
    gt2_bundles_json TEXT,
    token_parses_json TEXT,
    stage_two_casefile_json TEXT,
    stage_three_proof_json TEXT,
    wfw_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_INDEX_CLUE_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_atomic_parse_artifacts_clue "
    "ON atomic_parse_artifacts (clue_id, created_at)"
)

_STATE_DDL = """
CREATE TABLE IF NOT EXISTS clue_pipeline_state (
    clue_id INTEGER PRIMARY KEY,
    source TEXT,
    puzzle_number TEXT,
    clue_text TEXT NOT NULL,
    answer TEXT NOT NULL,
    status TEXT NOT NULL,
    confidence REAL,
    solver_version TEXT NOT NULL,
    stage_one_json TEXT,
    annotations_json TEXT,
    stage_two_json TEXT,
    stage_three_json TEXT,
    wfw_json TEXT,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
)
"""

_STATE_PUZZLE_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_clue_pipeline_state_puzzle "
    "ON clue_pipeline_state (source, puzzle_number)"
)

_RUN_DDL = """
CREATE TABLE IF NOT EXISTS atomic_parse_runs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source TEXT NOT NULL,
    puzzle_number TEXT NOT NULL,
    status TEXT NOT NULL,
    total_clues INTEGER NOT NULL DEFAULT 0,
    complete_clues INTEGER NOT NULL DEFAULT 0,
    review_clues INTEGER NOT NULL DEFAULT 0,
    solver_version TEXT NOT NULL,
    started_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    finished_at TIMESTAMP
)
"""

_REVIEW_DDL = """
CREATE TABLE IF NOT EXISTS atomic_parse_review_items (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER,
    artifact_id INTEGER NOT NULL,
    clue_id INTEGER NOT NULL,
    status TEXT NOT NULL DEFAULT 'open',
    review_type TEXT NOT NULL,
    summary TEXT NOT NULL,
    payload_json TEXT,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    resolved_at TIMESTAMP
)
"""

_INDEX_REVIEW_CLUE_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_atomic_parse_review_clue "
    "ON atomic_parse_review_items (clue_id, status, created_at)"
)


def ensure_table(conn=None):
    """Create the additive atomic artifact table if it is missing."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        conn.execute(_DDL)
        _ensure_column(conn, "atomic_parse_artifacts",
                       "stage_two_casefile_json", "TEXT")
        _ensure_column(conn, "atomic_parse_artifacts",
                       "stage_three_proof_json", "TEXT")
        conn.execute(_INDEX_CLUE_DDL)
        conn.execute(_STATE_DDL)
        conn.execute(_STATE_PUZZLE_INDEX_DDL)
        conn.execute(_RUN_DDL)
        conn.execute(_REVIEW_DDL)
        conn.execute(_INDEX_REVIEW_CLUE_DDL)
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def write_atomic_artifact(clue_id, clue_text, answer, artifact, conn=None):
    """Append one atomic parser artifact row and return its row id."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_table(conn)
        row = _normalise_artifact(clue_id, clue_text, answer, artifact)
        cur = conn.execute(
            """INSERT INTO atomic_parse_artifacts
               (clue_id, clue_text, answer, status, confidence, solver_version,
                clue_context_json, annotations_json, gt2_bundles_json,
                token_parses_json, stage_two_casefile_json,
                stage_three_proof_json, wfw_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                row["clue_id"],
                row["clue_text"],
                row["answer"],
                row["status"],
                row["confidence"],
                row["solver_version"],
                _dump(row["clue_context"]),
                _dump(row["annotations"]),
                _dump(row["gt2_bundles"]),
                _dump(row["token_parses"]),
                _dump(row["stage_two_casefile"]),
                _dump(row["stage_three_proof"]),
                _dump(row["wfw"]),
            ),
        )
        if own:
            conn.commit()
        return cur.lastrowid
    finally:
        if own:
            conn.close()


def write_solve_result_artifact(clue_id, clue_text, answer, solve_result,
                                conn=None, solver_version=SCHEMA_VERSION):
    """Serialise and append the atomic data carried by a SolveResult."""
    artifact = artifact_from_solve_result(solve_result, solver_version)
    return write_atomic_artifact(clue_id, clue_text, answer, artifact, conn)


def upsert_pipeline_state(clue_id, source, puzzle_number, clue_text, answer,
                          artifact, conn=None):
    """Write the current app-facing pipeline state for one clue.

    Unlike atomic_parse_artifacts, this table has exactly one row per clue.
    """
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_table(conn)
        row = _normalise_artifact(clue_id, clue_text, answer, artifact)
        conn.execute(
            """INSERT INTO clue_pipeline_state
               (clue_id, source, puzzle_number, clue_text, answer, status,
                confidence, solver_version, stage_one_json, annotations_json,
                stage_two_json, stage_three_json, wfw_json, updated_at)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                       CURRENT_TIMESTAMP)
               ON CONFLICT(clue_id) DO UPDATE SET
                   source = excluded.source,
                   puzzle_number = excluded.puzzle_number,
                   clue_text = excluded.clue_text,
                   answer = excluded.answer,
                   status = excluded.status,
                   confidence = excluded.confidence,
                   solver_version = excluded.solver_version,
                   stage_one_json = excluded.stage_one_json,
                   annotations_json = excluded.annotations_json,
                   stage_two_json = excluded.stage_two_json,
                   stage_three_json = excluded.stage_three_json,
                   wfw_json = excluded.wfw_json,
                   updated_at = CURRENT_TIMESTAMP""",
            (
                row["clue_id"],
                source,
                str(puzzle_number) if puzzle_number is not None else None,
                row["clue_text"],
                row["answer"],
                row["status"],
                row["confidence"],
                row["solver_version"],
                _dump(row["clue_context"]),
                _dump(row["annotations"]),
                _dump(row["stage_two_casefile"]),
                _dump(row["stage_three_proof"]),
                _dump(row["wfw"]),
            ),
        )
        if own:
            conn.commit()
        return clue_id
    finally:
        if own:
            conn.close()


def upsert_solve_result_pipeline_state(
        clue_id, source, puzzle_number, clue_text, answer, solve_result,
        conn=None, solver_version=SCHEMA_VERSION,
        status_override=None, confidence_override=None):
    """Serialise a SolveResult into the current clue pipeline state.

    status_override and confidence_override, when provided, take precedence
    over the values computed by artifact_from_solve_result. Used by
    _mark_current_state_solved to record legacy-engine solves as
    status='solved', confidence=100 while still persisting the stage_three
    evidence produced by the signature solver.

    Also writes to wfw_proof_attempts whenever stage_three_proof is present,
    so the two tables always stay in sync.
    """
    artifact = artifact_from_solve_result(solve_result, solver_version)
    if status_override is not None:
        artifact["status"] = status_override
    if confidence_override is not None:
        artifact["confidence"] = confidence_override
    upsert_pipeline_state(
        clue_id, source, puzzle_number, clue_text, answer, artifact, conn)
    stage_three = (
        getattr(solve_result, "stage_three_proof", None)
        if solve_result is not None else None
    )
    if stage_three is not None:
        from signature_solver.wfw_proof_store import write_wfw_proof_attempt
        proof_dict = stage_three.as_dict()
        proof_dict["status"] = (
            "wfw_proven" if stage_three.status == "PASS" else "wfw_review"
        )
        proof_dict["source"] = "stage_three_pipeline"
        write_wfw_proof_attempt(
            clue_id, source, puzzle_number, proof_dict, conn=conn)
    return clue_id


def get_pipeline_state(clue_id, conn=None):
    """Return the current pipeline state for a clue, or None."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        try:
            row = conn.execute(
                """SELECT clue_id, source, puzzle_number, clue_text, answer,
                          status, confidence, solver_version, stage_one_json,
                          annotations_json, stage_two_json, stage_three_json,
                          wfw_json, updated_at
                   FROM clue_pipeline_state
                   WHERE clue_id = ?""",
                (clue_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        return {
            "clue_id": row[0],
            "source": row[1],
            "puzzle_number": row[2],
            "clue_text": row[3],
            "answer": row[4],
            "status": row[5],
            "confidence": row[6],
            "solver_version": row[7],
            "stage_one": _load(row[8]),
            "annotations": _load(row[9]) or [],
            "stage_two": _load(row[10]),
            "stage_three": _load(row[11]),
            "wfw": _load(row[12]) or [],
            "updated_at": row[13],
        }
    finally:
        if own:
            conn.close()


def start_atomic_run(source, puzzle_number, total_clues, conn=None,
                     solver_version=SCHEMA_VERSION):
    """Create a durable production run row and return its id."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_table(conn)
        cur = conn.execute(
            """INSERT INTO atomic_parse_runs
               (source, puzzle_number, status, total_clues, solver_version)
               VALUES (?, ?, ?, ?, ?)""",
            (source, str(puzzle_number), "running", total_clues,
             solver_version),
        )
        if own:
            conn.commit()
        return cur.lastrowid
    finally:
        if own:
            conn.close()


def finish_atomic_run(run_id, complete_clues, review_clues, conn=None,
                      status="complete"):
    """Mark a production run finished."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_table(conn)
        conn.execute(
            """UPDATE atomic_parse_runs
               SET status = ?, complete_clues = ?, review_clues = ?,
                   finished_at = CURRENT_TIMESTAMP
               WHERE id = ?""",
            (status, complete_clues, review_clues, run_id),
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def write_review_item(run_id, artifact_id, clue_id, review_type, summary,
                      payload=None, conn=None):
    """Append a human-review item linked to an atomic artifact."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_table(conn)
        cur = conn.execute(
            """INSERT INTO atomic_parse_review_items
               (run_id, artifact_id, clue_id, review_type, summary,
                payload_json)
               VALUES (?, ?, ?, ?, ?, ?)""",
            (run_id, artifact_id, clue_id, review_type, summary,
             _dump(payload or {})),
        )
        if own:
            conn.commit()
        return cur.lastrowid
    finally:
        if own:
            conn.close()


def artifact_from_solve_result(solve_result, solver_version=SCHEMA_VERSION):
    """Return a JSON-serialisable artifact from the current SolveResult."""
    if solve_result is None:
        return {
            "status": "failed",
            "confidence": None,
            "solver_version": solver_version,
            "clue_context": None,
            "annotations": [],
            "gt2_bundles": [],
            "token_parses": [],
            "stage_two_casefile": None,
            "stage_three_proof": None,
            "wfw": [],
        }

    context = getattr(solve_result, "clue_context", None)
    context_dict = context.as_dict() if context is not None else None
    annotations = []
    if context is not None:
        annotations = [ann.as_dict() for ann in context.annotations]

    gt2_bundles = []
    for attr in ("gt2_candidate_bundles", "gt2_evidence_bundles"):
        for bundle in getattr(solve_result, attr, []) or []:
            gt2_bundles.append(bundle.as_dict())

    token_parses = [
        parse.as_dict()
        for parse in getattr(solve_result, "token_parses", []) or []
    ]
    wfw = list(getattr(solve_result, "wfw_token_parses", []) or [])
    confidence = getattr(solve_result, "confidence", None)
    status = "solved" if getattr(solve_result, "high_confidence", False) else "partial"
    if solve_result is None:
        status = "failed"

    return {
        "status": status,
        "confidence": confidence,
        "solver_version": solver_version,
        "clue_context": context_dict,
        "annotations": annotations,
        "gt2_bundles": gt2_bundles,
        "token_parses": token_parses,
        "stage_two_casefile": _as_dict_or_none(
            getattr(solve_result, "stage_two_casefile", None)),
        "stage_three_proof": _as_dict_or_none(
            getattr(solve_result, "stage_three_proof", None)),
        "wfw": wfw,
    }


def get_latest_atomic_artifact(clue_id, conn=None):
    """Return the newest parsed atomic artifact for a clue, or None."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        try:
            row = conn.execute(
                """SELECT id, clue_id, clue_text, answer, status, confidence,
                          solver_version, clue_context_json, annotations_json,
                          gt2_bundles_json, token_parses_json,
                          stage_two_casefile_json, stage_three_proof_json,
                          wfw_json, created_at
                   FROM atomic_parse_artifacts
                   WHERE clue_id = ?
                   ORDER BY created_at DESC, id DESC
                   LIMIT 1""",
                (clue_id,),
            ).fetchone()
        except sqlite3.OperationalError:
            return None
        if row is None:
            return None
        return _row_to_artifact(row)
    finally:
        if own:
            conn.close()


def _normalise_artifact(clue_id, clue_text, answer, artifact):
    artifact = artifact or {}
    return {
        "clue_id": clue_id,
        "clue_text": clue_text or "",
        "answer": answer or "",
        "status": artifact.get("status") or "partial",
        "confidence": artifact.get("confidence"),
        "solver_version": artifact.get("solver_version") or SCHEMA_VERSION,
        "clue_context": artifact.get("clue_context"),
        "annotations": artifact.get("annotations") or [],
        "gt2_bundles": artifact.get("gt2_bundles") or [],
        "token_parses": artifact.get("token_parses") or [],
        "stage_two_casefile": artifact.get("stage_two_casefile"),
        "stage_three_proof": artifact.get("stage_three_proof"),
        "wfw": artifact.get("wfw") or [],
    }


def _row_to_artifact(row):
    return {
        "id": row[0],
        "clue_id": row[1],
        "clue_text": row[2],
        "answer": row[3],
        "status": row[4],
        "confidence": row[5],
        "solver_version": row[6],
        "clue_context": _load(row[7]),
        "annotations": _load(row[8]) or [],
        "gt2_bundles": _load(row[9]) or [],
        "token_parses": _load(row[10]) or [],
        "stage_two_casefile": _load(row[11]),
        "stage_three_proof": _load(row[12]),
        "wfw": _load(row[13]) or [],
        "created_at": row[14],
    }


def _ensure_column(conn, table, column, column_type):
    existing = {
        row[1] for row in conn.execute("PRAGMA table_info(%s)" % table)
    }
    if column not in existing:
        conn.execute("ALTER TABLE %s ADD COLUMN %s %s" % (
            table, column, column_type))


def _as_dict_or_none(value):
    if value is None:
        return None
    if hasattr(value, "as_dict"):
        return value.as_dict()
    return value


def _dump(value):
    if value is None:
        return None
    return json.dumps(value, sort_keys=True, separators=(",", ":"))


def _load(value):
    if not value:
        return None
    return json.loads(value)
