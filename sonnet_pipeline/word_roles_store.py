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
    letters TEXT,
    piece_key INTEGER,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    PRIMARY KEY (clue_id, word_index)
)
"""

_INDEX_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_clue_word_roles_clue "
    "ON clue_word_roles (clue_id)"
)

# Columns that may be missing on older instances of the table. ensure_table
# adds them with ALTER TABLE on first use.
_EXPECTED_COLUMNS = {"clue_id", "word_index", "word_text", "role", "source",
                     "letters", "piece_key", "updated_at"}


def ensure_table(conn=None):
    """Create the table if missing, or add new columns if it pre-dates them.
    Pass an existing connection or omit to use a short-lived one."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        conn.execute(_DDL)
        conn.execute(_INDEX_DDL)
        # Migrate older tables: add letters / piece_key if missing.
        existing = {r[1] for r in conn.execute(
            "PRAGMA table_info(clue_word_roles)").fetchall()}
        if "letters" not in existing:
            conn.execute("ALTER TABLE clue_word_roles ADD COLUMN letters TEXT")
        if "piece_key" not in existing:
            conn.execute("ALTER TABLE clue_word_roles ADD COLUMN piece_key INTEGER")
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def _normalise_classified(item):
    """Accept either a (word, role) tuple or a longer
    (word, role, letters, piece_key) tuple. Returns a 4-tuple."""
    if len(item) >= 4:
        return item[0], item[1], item[2], item[3]
    if len(item) == 3:
        return item[0], item[1], item[2], None
    return item[0], item[1], None, None


def write_auto_roles(clue_id, classified, conn=None):
    """Replace auto-classified rows for one clue with the latest
    verifier output. Manual rows are preserved.

    Args:
        clue_id: integer clue id from clues.id
        classified: list of tuples. Each entry can be one of:
            (word_text, role)
            (word_text, role, letters)
            (word_text, role, letters, piece_key)
            Missing fields default to None.
        conn: optional open sqlite3.Connection; if None, opens its own.
    """
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
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
        for idx, item in enumerate(classified):
            if idx in manual_positions:
                continue
            word_text, role, letters, piece_key = _normalise_classified(item)
            conn.execute(
                "INSERT INTO clue_word_roles "
                "(clue_id, word_index, word_text, role, source, letters, piece_key) "
                "VALUES (?, ?, ?, ?, 'auto', ?, ?)",
                (clue_id, idx, word_text, role, letters, piece_key),
            )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def write_manual_role(clue_id, word_index, word_text, role,
                      letters=None, piece_key=None, conn=None):
    """Write or overwrite a single word's role as a manual assignment.

    Use from the admin review UI when filling in unaccounted words or
    correcting a misclassified one. Manual rows survive future
    auto-write passes.

    `letters` and `piece_key` are optional — leave as None for roles
    that don't contribute letters (definition, link, indicator, ...).
    """
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_table(conn)
        conn.execute(
            "INSERT INTO clue_word_roles "
            "(clue_id, word_index, word_text, role, source, letters, piece_key, updated_at) "
            "VALUES (?, ?, ?, ?, 'manual', ?, ?, CURRENT_TIMESTAMP) "
            "ON CONFLICT(clue_id, word_index) DO UPDATE SET "
            "role = excluded.role, word_text = excluded.word_text, "
            "source = 'manual', letters = excluded.letters, "
            "piece_key = excluded.piece_key, updated_at = CURRENT_TIMESTAMP",
            (clue_id, word_index, word_text, role, letters, piece_key),
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def get_roles(clue_id, conn=None):
    """Return a list of (word_index, word_text, role, source, letters, piece_key)
    tuples for one clue, ordered by word_index. Empty list if no rows or
    the table doesn't exist yet."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        try:
            rows = conn.execute(
                "SELECT word_index, word_text, role, source, letters, piece_key "
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
