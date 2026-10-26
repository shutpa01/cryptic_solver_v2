"""Keep the normalized-key columns in the reference DB correct.

LiveDB looks up reference entries by a stored, tidied (normalised) copy of the word
— for speed. A row whose stored copy is BLANK (hand-added in DB Browser) or WRONG
(e.g. a row edited so its word changed but the key did not — 'Ms Winfrey' left with
key 'mrs winfrey') is invisible to every engine: the entry exists but the fast
lookup cannot find it.

Every wiring build fills MISSING keys (cheap, indexed). The WFW Reload button and the
nightly cascade run the FULL reconcile: any row whose key != _normalize_key(word) is
repaired (blank or wrong), so manual additions AND edits self-heal after a Reload or
overnight without changing the fast-lookup design. Idempotent — a clean DB updates zero
rows. (Until 2026-10-26 the full reconcile ran on every build: 8–10 s each time.)

Uses the SAME normalisation (signature_solver.db._normalize_key) the lookups use,
so a reconciled key matches exactly what a query asks for.

NOTE: the write needs the DB to be writable. If another connection holds it (a
running server's sibling, or DB Browser with the DB open), the UPDATE fails with
'database is locked' — this is now reported (a warning), not swallowed silently, so
a stale key is never invisibly left unrepaired.
"""
import sqlite3
import sys

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


def backfill_null_norm_keys(db_path, full=False):
    """Fill the normalized-key columns across the reference tables.

    full=False (every wiring build): only NULL or blank keys — rows hand-added without one.
    The key columns are indexed, so this is ~instant.
    full=True (the WFW Reload button and the nightly cascade): ALSO repair keys that no longer
    equal _normalize_key(word) (a row whose word was edited in DB Browser). That needs the
    Python normaliser on all ~2.15M rows — measured 8–10 s, and it ran on every wiring build,
    including the rebuild inside an Approve-all (2026-10-26: 0 rows repaired, 8 s spent).

    Returns the total number of rows repaired. Best-effort on shape (a missing table or
    column is skipped), but a LOCKED database is reported (warning) rather than swallowed,
    so a stale key is never silently left unrepaired."""
    filled = 0
    conn = sqlite3.connect(db_path, timeout=30)
    try:
        conn.create_function("pynorm", 1, lambda s: _normalize_key(s or ""))
        for table, key, norm in _SPECS:
            cols = {r[1] for r in conn.execute("PRAGMA table_info(%s)" % table)}
            if key not in cols or norm not in cols:
                continue
            try:
                if full:
                    # Repair every drifted key: NULL, blank, OR a value that no longer equals
                    # the normalised word (a row whose word was edited keeps the old key).
                    # `norm IS NULL` is listed explicitly because `NULL <> x` is NULL (false).
                    where = "%s IS NULL OR %s <> pynorm(%s)" % (norm, norm, key)
                else:
                    # Only missing keys. A blank key on a blank word is already correct, so
                    # it is left alone rather than rewritten on every build.
                    where = ("%s IS NULL OR (%s = '' AND pynorm(%s) <> '')"
                             % (norm, norm, key))
                n = conn.execute(
                    "UPDATE %s SET %s = pynorm(%s) WHERE %s"
                    % (table, norm, key, where)).rowcount
                filled += n
            except sqlite3.OperationalError as e:
                sys.stderr.write(
                    "norm_backfill: could not reconcile %s (%s) — the DB may be locked "
                    "by another connection (running server / open DB Browser); "
                    "hand-added or edited keys will stay invisible until this succeeds.\n"
                    % (table, e))
        conn.commit()
    finally:
        conn.close()
    return filled
