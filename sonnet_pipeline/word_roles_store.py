"""clue_word_roles persistence — captures the verifier's per-word
classification so it can be displayed on the clue page and amended by
admin review.

Table shape:
    clue_id INTEGER, word_index INTEGER, word_text TEXT,
    role TEXT, source TEXT ('auto'|'manual'), updated_at TIMESTAMP
    PRIMARY KEY (clue_id, word_index)

Write rules:
    - write_auto_roles: writes auto-classified rows but DOES NOT
      overwrite rows whose source is 'manual' (preserves user input).
    - write_manual_role: writes/overwrites a single row with
      source='manual'. Use this from the admin review UI.

The table is created lazily on the first write. Reads are safe
against the table not existing yet (return empty).
"""

import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"


_DDL = """
CREATE TABLE IF NOT EXISTS clue_word_roles (
    clue_id INTEGER NOT NULL,
    word_index INTEGER NOT NULL,
    word_text TEXT NOT NULL,
    role TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'auto',
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (clue_id, word_index)
)
"""

_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_clue_word_roles_clue "
    "ON clue_word_roles (clue_id)"
)


def ensure_table(conn=None):
    """Create the table if missing. Pass an existing connection or omit
    to use a short-lived one against the default DB."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=10)
    try:
        conn.execute(_DDL)
        conn.execute(_INDEX_DDL)
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def write_auto_roles(clue_id, classified, conn=None):
    """Replace auto-classified rows for one clue with the latest
    verifier output. Manual rows are preserved.

    Args:
        clue_id: integer clue id from clues.id
        classified: list of (word_text, role) tuples in clue order
                    (the 'classified' field returned by
                    ExplanationVerifier._classify_clue_words).
        conn: optional open sqlite3.Connection; if None, opens its own.
    """
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=10)
    try:
        ensure_table(conn)
        # Look up which positions are manually-owned; leave them alone.
        manual_positions = {
            row[0] for row in conn.execute(
                "SELECT word_index FROM clue_word_roles "
                "WHERE clue_id = ? AND source = 'manual'",
                (clue_id,),
            )
        }
        # Clear auto rows for this clue (manual rows stay because we
        # filter the delete by source='auto').
        conn.execute(
            "DELETE FROM clue_word_roles WHERE clue_id = ? AND source = 'auto'",
            (clue_id,),
        )
        # Insert fresh auto rows, skipping positions claimed by manual rows.
        for idx, (word_text, role) in enumerate(classified):
            if idx in manual_positions:
                continue
            conn.execute(
                "INSERT INTO clue_word_roles "
                "(clue_id, word_index, word_text, role, source) "
                "VALUES (?, ?, ?, ?, 'auto')",
                (clue_id, idx, word_text, role),
            )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def write_manual_role(clue_id, word_index, word_text, role, conn=None):
    """Write or overwrite a single word's role as a manual assignment.

    Use from the admin review UI when filling in unaccounted words or
    correcting a misclassified one. Manual rows survive future
    auto-write passes.
    """
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=10)
    try:
        ensure_table(conn)
        conn.execute(
            "INSERT INTO clue_word_roles "
            "(clue_id, word_index, word_text, role, source, updated_at) "
            "VALUES (?, ?, ?, ?, 'manual', CURRENT_TIMESTAMP) "
            "ON CONFLICT(clue_id, word_index) DO UPDATE SET "
            "role = excluded.role, word_text = excluded.word_text, "
            "source = 'manual', updated_at = CURRENT_TIMESTAMP",
            (clue_id, word_index, word_text, role),
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def get_roles(clue_id, conn=None):
    """Return a list of (word_index, word_text, role, source) tuples
    for one clue, ordered by word_index. Empty list if no rows or the
    table doesn't exist yet."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=10)
    try:
        try:
            rows = conn.execute(
                "SELECT word_index, word_text, role, source "
                "FROM clue_word_roles WHERE clue_id = ? "
                "ORDER BY word_index",
                (clue_id,),
            ).fetchall()
        except sqlite3.OperationalError:
            # Table doesn't exist yet
            return []
        return [tuple(r) for r in rows]
    finally:
        if own:
            conn.close()
