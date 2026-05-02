"""Gate-keeper for pending_enrichments inserts.

Many code paths queue enrichment suggestions (synonym, abbreviation,
definition, indicator) when the verifier or solver decides the lookup
"missed" the DB. The decision is sometimes wrong: the data IS in the
reference DB, just under a different table or matching path the caller
didn't check. Without this gate, those false-positive suggestions
clutter the user's review queue and waste time.

Usage:

    from sonnet_pipeline.enrichment_gate import already_in_reference_db

    if already_in_reference_db(typ, word, letters):
        # don't queue — it's already known
        continue

The reference DB is `data/cryptic_new.db` (the canonical synonym /
abbreviation / indicator / definition store). The pending queue lives
in `data/clues_master.db`.
"""
import sqlite3
from pathlib import Path

_REF_DB_PATH = Path(__file__).resolve().parent.parent / "data" / "cryptic_new.db"

# Module-level connection cache — cheap to keep open, much cheaper than
# reconnecting per-call. Read-only intent; we never write to ref DB here.
_REF_CONN = None


def _conn():
    global _REF_CONN
    if _REF_CONN is None:
        _REF_CONN = sqlite3.connect(str(_REF_DB_PATH))
    return _REF_CONN


def already_in_reference_db(typ, word, letters, ref_db_path=None):
    """Return True if the suggested enrichment already exists in cryptic_new.db.

    Checks the appropriate tables for the suggestion type:
      - 'synonym'      → synonyms_pairs (both directions),
                         synonyms (text), definition_answers_augmented
      - 'definition'   → definition_answers_augmented, synonyms_pairs
      - 'abbreviation' → synonyms_pairs, wordplay (substitution),
                         synonyms (text)
      - 'indicator'    → indicators (matching wordplay_type)
      - 'homophone'    → homophones (both directions)

    Returns False on unrecognised type or any DB error (fail-open: better
    to queue a probable duplicate than miss a real gap).
    """
    if not typ or not word or not letters:
        return False

    if ref_db_path is not None:
        conn = sqlite3.connect(str(ref_db_path))
        try:
            return _check(conn, typ, word, letters)
        finally:
            conn.close()
    return _check(_conn(), typ, word, letters)


def _check(conn, typ, word, letters):
    w = (word or "").strip().lower()
    l = (letters or "").strip().upper()
    t = (typ or "").strip().lower()
    if not w or not l:
        return False

    try:
        if t == "synonym" or t == "definition":
            # synonyms_pairs (both directions)
            r = conn.execute(
                "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
                (w, l),
            ).fetchone()
            if r:
                return True
            r = conn.execute(
                "SELECT 1 FROM synonyms_pairs WHERE UPPER(word)=? AND LOWER(synonym)=? LIMIT 1",
                (l, w),
            ).fetchone()
            if r:
                return True
            # definition_answers_augmented (both directions)
            r = conn.execute(
                "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
                (w, l),
            ).fetchone()
            if r:
                return True
            r = conn.execute(
                "SELECT 1 FROM definition_answers_augmented WHERE UPPER(definition)=? AND LOWER(answer)=? LIMIT 1",
                (l, w),
            ).fetchone()
            if r:
                return True
            # broader synonyms text blob
            r = conn.execute(
                "SELECT synonyms FROM synonyms WHERE LOWER(word)=?",
                (w,),
            ).fetchone()
            if r and r[0] and l.lower() in (r[0] or "").lower():
                return True
            return False

        if t == "abbreviation":
            r = conn.execute(
                "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
                (w, l),
            ).fetchone()
            if r:
                return True
            r = conn.execute(
                "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND UPPER(substitution)=? LIMIT 1",
                (w, l),
            ).fetchone()
            if r:
                return True
            r = conn.execute(
                "SELECT synonyms FROM synonyms WHERE LOWER(word)=?",
                (w,),
            ).fetchone()
            if r and r[0] and l.lower() in (r[0] or "").lower():
                return True
            return False

        if t == "indicator":
            # indicator entries store wordplay_type in the `letters` slot of
            # the queue (e.g. ('indicator', 'crushed', 'ANAGRAM')).
            r = conn.execute(
                "SELECT 1 FROM indicators WHERE LOWER(word)=? AND LOWER(wordplay_type)=? LIMIT 1",
                (w, l.lower()),
            ).fetchone()
            return bool(r)

        if t == "homophone":
            r = conn.execute(
                "SELECT 1 FROM homophones WHERE (LOWER(word)=? AND LOWER(homophone)=?) "
                "OR (LOWER(word)=? AND LOWER(homophone)=?) LIMIT 1",
                (w, l.lower(), l.lower(), w),
            ).fetchone()
            return bool(r)
    except Exception:
        return False
    return False
