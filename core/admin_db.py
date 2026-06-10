"""Admin writes to the reference DB (cryptic_new.db) for the true-test UI.

Mirrors the dashboard's manual-enrichment inserts exactly (same tables, columns,
dedup), so adding here is identical to Accepting in the dashboard — just done
inline while testing a clue. Because the engine's defines()/indicator_types()
predicates do a LIVE query on a miss, a definition or indicator added here is seen
on the very next re-run with no server restart. (Synonyms are stored for future
engines; the hidden engine does not consume them yet.)

Each adder returns a short status string for the UI.
"""

import os
import sqlite3

CRYPTIC_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "data", "cryptic_new.db")
MASTER_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "clues_master.db")


def _conn():
    return sqlite3.connect(CRYPTIC_DB, timeout=30)


def _mconn():
    return sqlite3.connect(MASTER_DB, timeout=30)


# --- per-clue enrichment queue (pending_enrichments lives in clues_master.db) -------

def pending_for_clue(clue_text, answer):
    """The queued enrichments for one clue: [(id, type, word, letters, answer)].
    Matched on the clue text + answer (pending_enrichments has no clue_id)."""
    conn = _mconn()
    try:
        return conn.execute(
            "SELECT id, type, word, letters, answer FROM pending_enrichments "
            "WHERE clue_text=? AND answer=? ORDER BY id", (clue_text, answer)).fetchall()
    finally:
        conn.close()


def delete_pending(pending_id):
    """Drop one queued enrichment (after it has been approved into the reference DB)."""
    conn = _mconn()
    try:
        conn.execute("DELETE FROM pending_enrichments WHERE id=?", (int(pending_id),))
        conn.commit()
    finally:
        conn.close()


def reject_pending(pending_id):
    """Move one queued enrichment to rejected_enrichments and drop it from the queue,
    so it is recorded as rejected and never re-suggested."""
    conn = _mconn()
    try:
        row = conn.execute("SELECT type, word, letters FROM pending_enrichments "
                           "WHERE id=?", (int(pending_id),)).fetchone()
        if not row:
            return "Nothing to reject."
        conn.execute("INSERT INTO rejected_enrichments (type, word, letters, "
                     "rejected_at) VALUES (?, ?, ?, datetime('now'))", row)
        conn.execute("DELETE FROM pending_enrichments WHERE id=?", (int(pending_id),))
        conn.commit()
        return "Rejected: %r" % (row[1],)
    finally:
        conn.close()


def add_definition(definition, answer):
    definition = (definition or "").strip()
    answer = (answer or "").strip()
    if not definition or not answer:
        return "Definition and answer are both required."
    conn = _conn()
    try:
        if conn.execute("SELECT 1 FROM definition_answers_augmented "
                        "WHERE definition=? AND answer=?",
                        (definition, answer)).fetchone():
            return "Already present: %r → %s" % (definition, answer)
        conn.execute("INSERT INTO definition_answers_augmented "
                     "(definition, answer, source) VALUES (?, ?, 'admin')",
                     (definition, answer))
        conn.commit()
        return "Added definition: %r → %s" % (definition, answer)
    finally:
        conn.close()


def add_synonym(word, synonym):
    word = (word or "").strip()
    synonym = (synonym or "").strip()
    if not word or not synonym:
        return "Word and synonym are both required."
    conn = _conn()
    try:
        if conn.execute("SELECT 1 FROM synonyms_pairs WHERE word=? AND synonym=?",
                        (word, synonym)).fetchone():
            return "Already present: %r = %r" % (word, synonym)
        conn.execute("INSERT INTO synonyms_pairs (word, synonym, source) "
                     "VALUES (?, ?, 'admin')", (word, synonym))
        conn.commit()
        return "Added synonym: %r = %r (note: not used by the hidden engine yet)" % (
            word, synonym)
    finally:
        conn.close()


def add_indicator(word, wordplay_type):
    word = (word or "").strip()
    wp = (wordplay_type or "").strip().lower()
    if not word or not wp:
        return "Indicator word and type are both required."
    conn = _conn()
    try:
        if conn.execute("SELECT 1 FROM indicators WHERE word=? AND "
                        "wordplay_type=? AND subtype IS NULL",
                        (word, wp)).fetchone():
            return "Already present: %r (%s)" % (word, wp)
        conn.execute("INSERT INTO indicators "
                     "(word, wordplay_type, subtype, confidence, source) "
                     "VALUES (?, ?, NULL, 'high', 'admin')", (word, wp))
        conn.commit()
        return "Added indicator: %r (%s)" % (word, wp)
    finally:
        conn.close()
