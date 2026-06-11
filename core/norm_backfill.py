"""Fill in any blank normalized-key columns in the reference DB.

LiveDB looks up reference entries by a stored, tidied (normalised) copy of the word
— for speed. Rows added by hand (e.g. in DB Browser) have that copy blank, so the
fast lookup cannot find them: the entry exists but is invisible to every engine.

This runs at every wiring build (app start AND the Reload button) and fills the
tidied copy for any row missing it, so manual additions are picked up after a
reload without changing the fast-lookup design. Only rows where the column IS NULL
are touched, so it is cheap and idempotent.

Uses the SAME normalisation (signature_solver.db._normalize_key) the lookups use,
so a backfilled key matches exactly what a query asks for.
"""
import sqlite3

from signature_solver.db import _normalize_key

# (table, source word column, normalized-key column) — every reference table whose
# LiveDB lookup keys off a normalized column. Mirrors core/_migrate_norm_keys.py.
_SPECS = [
    ("synonyms_pairs", "word", "norm_word"),
    ("definition_answers_augmented", "definition", "norm_def"),
    ("wordplay", "indicator", "norm_ind"),
    ("indicators", "word", "norm_word"),
    ("homophones", "word", "norm_word"),
    ("pronunciations", "word", "norm_word"),
]


def backfill_null_norm_keys(db_path):
    """Fill blank normalized-key columns across the reference tables. Returns the
    total number of rows filled. Best-effort: a missing table or column is skipped,
    never fatal (so it is safe on any DB shape)."""
    filled = 0
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.create_function("pynorm", 1, lambda s: _normalize_key(s or ""))
        for table, key, norm in _SPECS:
            try:
                cols = {r[1] for r in conn.execute("PRAGMA table_info(%s)" % table)}
                if key not in cols or norm not in cols:
                    continue
                # Treat EVERY blank form the same: NULL, empty string, or
                # whitespace-only. A hand-entered row (DB Browser) can leave the
                # matching column as '' rather than NULL, which a NULL-only fill
                # silently skips — leaving the row invisible to every lookup.
                n = conn.execute(
                    "UPDATE %s SET %s = pynorm(%s) "
                    "WHERE %s IS NULL OR TRIM(%s) = ''"
                    % (table, norm, key, norm, norm)).rowcount
                filled += n
            except sqlite3.Error:
                continue
        conn.commit()
    finally:
        conn.close()
    return filled
