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

from signature_solver.db import _normalize_key   # LiveDB matches on this normalized key

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
                     "(definition, answer, source, norm_def) VALUES (?, ?, 'admin', ?)",
                     (definition, answer, _normalize_key(definition)))
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
        # CASE-INSENSITIVE dedup: clue words are often capitalised (start of clue / proper
        # nouns) while the DB stores them lower-case, so an exact-case check let duplicates
        # through (e.g. 'Good' vs 'good'). Match on lower(word)/upper(synonym).
        if conn.execute("SELECT 1 FROM synonyms_pairs WHERE lower(word)=lower(?) "
                        "AND upper(synonym)=upper(?)", (word, synonym)).fetchone():
            return "Already present: %r = %r" % (word, synonym)
        conn.execute("INSERT INTO synonyms_pairs (word, synonym, source, norm_word) "
                     "VALUES (?, ?, 'admin', ?)", (word, synonym, _normalize_key(word)))
        conn.commit()
        return "Added synonym: %r = %r (note: not used by the hidden engine yet)" % (
            word, synonym)
    finally:
        conn.close()


def add_substitution(word, value):
    """Add a wordplay SUBSTITUTION (word -> its cryptic value: an abbreviation, Roman numeral,
    compass point, symbol, ...) to the `wordplay` table (cryptic_new.db) — NOT synonyms_pairs, so
    it is labelled 'Substitution', not 'Synonym'. Case-insensitive dedup; category 'admin'."""
    word = (word or "").strip()
    value = (value or "").strip().upper()
    if not word or not value:
        return "Word and substitution value are both required."
    conn = _conn()
    try:
        if conn.execute("SELECT 1 FROM wordplay WHERE lower(indicator)=lower(?) "
                        "AND upper(substitution)=?", (word, value)).fetchone():
            return "Already present: %r -> %s" % (word, value)
        conn.execute("INSERT INTO wordplay (indicator, substitution, category, notes, norm_ind) "
                     "VALUES (?, ?, 'admin', 'added via hand-solver', ?)",
                     (word, value, _normalize_key(word)))
        conn.commit()
        return "Added substitution: %r -> %s" % (word, value)
    finally:
        conn.close()


def add_link_word(word):
    """Add a joining/link word to the link_words table (cryptic_new.db). A link word is
    glue an engine may skip between pieces (e.g. 'has' in 'X has Y'); it carries no
    letters. Deduped case-insensitively; source 'admin'."""
    word = (word or "").strip()
    if not word:
        return "Link word is required."
    conn = _conn()
    try:
        if conn.execute("SELECT 1 FROM link_words WHERE lower(word)=lower(?)",
                        (word,)).fetchone():
            return "Already a link word: %r" % word
        conn.execute("INSERT INTO link_words (word, source, notes) "
                     "VALUES (?, 'admin', 'added via clue admin panel')", (word,))
        conn.commit()
        return "Added link word: %r" % word
    finally:
        conn.close()


def add_literal(word):
    """Add a curated literal word to the literal_words table (cryptic_new.db). A literal
    is a short function word a setter may use as its OWN uppercased letters (it->IT,
    pe->PE) — a raw reading the reference DB does not otherwise provide. Single token only
    (a literal is never a phrase, matching core.literals.literal_value). Stored lowercased,
    deduped case-insensitively; source 'admin'."""
    word = (word or "").strip().lower()
    if not word:
        return "Literal word is required."
    if " " in word:
        return "A literal must be a single word, not a phrase."
    conn = _conn()
    try:
        if conn.execute("SELECT 1 FROM literal_words WHERE lower(word)=?",
                        (word,)).fetchone():
            return "Already a literal: %r" % word
        conn.execute("INSERT INTO literal_words (word, source) VALUES (?, 'admin')",
                     (word,))
        conn.commit()
        return "Added literal: %r -> %s" % (word, word.upper())
    finally:
        conn.close()


def add_homophone(word, homophone):
    """Add a homophone PAIR to the homophones table (cryptic_new.db). Stored BIDIRECTIONALLY
    (word->homophone AND homophone->word) sharing a new group_id, with norm_word set on each
    so the live forward lookup (get_homophones, keyed on norm_word) sees it at once — matching
    the existing table's two-row-per-pair shape. Lowercased; deduped case-insensitively."""
    word = (word or "").strip().lower()
    homophone = (homophone or "").strip().lower()
    if not word or not homophone:
        return "Both the word and the homophone (sounds-like) are required."
    if word == homophone:
        return "A homophone pair needs two different spellings."
    conn = _conn()
    try:
        if conn.execute("SELECT 1 FROM homophones WHERE lower(word)=? AND lower(homophone)=?",
                        (word, homophone)).fetchone():
            return "Already present: %r sounds like %r" % (word, homophone)
        gid = conn.execute("SELECT COALESCE(MAX(group_id), 0) + 1 FROM homophones").fetchone()[0]
        conn.execute("INSERT INTO homophones (word, homophone, group_id, norm_word) "
                     "VALUES (?, ?, ?, ?)", (word, homophone, gid, _normalize_key(word)))
        conn.execute("INSERT INTO homophones (word, homophone, group_id, norm_word) "
                     "VALUES (?, ?, ?, ?)", (homophone, word, gid, _normalize_key(homophone)))
        conn.commit()
        return "Added homophone: %r sounds like %r" % (word, homophone)
    finally:
        conn.close()


def add_indicator(word, wordplay_type, subtype=None):
    word = (word or "").strip()
    wp = (wordplay_type or "").strip().lower()
    sub = (subtype or "").strip().lower() or None
    if not word or not wp:
        return "Indicator word and type are both required."
    # A SELECTION indicator names WHICH letters to take (first/last/outer/middle/alternate).
    # Without that rule it is meaningless — the solver could not know which letter to use —
    # so a selection MUST carry a subtype that maps to a real rule. Reject rule-less or
    # unknown-rule selections here (the write layer), so a dead selection can never exist.
    if wp == "selection":
        from core.selection_indicators import SUBTYPE_RULE, CLUE_PAGE_SUBTYPES
        if not sub:
            return ("A selection indicator needs a sub-type (which letters it takes): "
                    "one of %s." % ", ".join(CLUE_PAGE_SUBTYPES))
        if ("selection", sub) not in SUBTYPE_RULE:
            return ("Unknown selection sub-type %r. Use one of: %s."
                    % (sub, ", ".join(CLUE_PAGE_SUBTYPES)))
    # A LETTER-SHIFT (cyclic rotation by one) is meaningless without a direction — the solver
    # could not know which end moves — so, like selection, it MUST carry a valid sub-type.
    if wp == "letter_shift":
        if not sub:
            return ("A letter-shift indicator needs a sub-type (which way it moves): "
                    "last_front or first_end.")
        if sub not in ("last_front", "first_end"):
            return ("Unknown letter-shift sub-type %r. Use last_front or first_end." % sub)
    # A POSITIONAL (charade re-ordering) indicator is meaningless without a direction — the
    # solver could not know whether the piece goes after or before — so, like selection and
    # letter-shift, it MUST carry a valid direction sub-type (core skips a blank-subtype row).
    if wp == "charade_positional":
        if not sub:
            return ("A positional indicator needs a sub-type (which way it re-orders): "
                    "after or before.")
        if sub not in ("after", "before", "after_down", "before_down"):
            return ("Unknown positional sub-type %r. Use after or before "
                    "(or after_down/before_down for down clues)." % sub)
    label = "%s%s" % (wp, ("/" + sub) if sub else "")
    conn = _conn()
    try:
        if sub is None:
            dup = conn.execute("SELECT 1 FROM indicators WHERE word=? AND "
                               "wordplay_type=? AND subtype IS NULL",
                               (word, wp)).fetchone()
        else:
            dup = conn.execute("SELECT 1 FROM indicators WHERE word=? AND "
                               "wordplay_type=? AND subtype=?",
                               (word, wp, sub)).fetchone()
        if dup:
            return "Already present: %r (%s)" % (word, label)
        conn.execute("INSERT INTO indicators "
                     "(word, wordplay_type, subtype, confidence, source, norm_word) "
                     "VALUES (?, ?, ?, 'high', 'admin', ?)",
                     (word, wp, sub, _normalize_key(word)))
        conn.commit()
        return "Added indicator: %r (%s)" % (word, label)
    finally:
        conn.close()


# --- deletes (hand-solver: prune a POLLUTING reference row; RECOVERABLE) --------------
# Each delete copies the row(s) into deleted_entries before removing them, so a misclick is
# recoverable and you can see what has been pruned. Matching is case-insensitive on the
# word/definition. A delete that hits nothing is reported, not silently ignored (the value
# may be a bidirectional-lookup artifact, not a real row).

def _ensure_deleted_table(conn):
    conn.execute(
        "CREATE TABLE IF NOT EXISTS deleted_entries ("
        "kind TEXT, word TEXT, value TEXT, wordplay_type TEXT, subtype TEXT, "
        "answer TEXT, source TEXT, deleted_at TEXT)")


def _record_deleted(conn, kind, word="", value="", wptype="", subtype="", answer="",
                    source=""):
    _ensure_deleted_table(conn)
    conn.execute("INSERT INTO deleted_entries (kind,word,value,wordplay_type,subtype,"
                 "answer,source,deleted_at) VALUES (?,?,?,?,?,?,?,datetime('now'))",
                 (kind, word, value, wptype, subtype, answer, source))


def delete_synonym(word, synonym):
    """Delete the (word -> synonym) pair(s) from synonyms_pairs (recoverable). Returns a
    status string; reports 0 when there is no direct row (e.g. a reverse-lookup artifact)."""
    word = (word or "").strip(); synonym = (synonym or "").strip()
    if not word or not synonym:
        return "Word and synonym are both required."
    conn = _conn()
    try:
        rows = conn.execute("SELECT word,synonym,source FROM synonyms_pairs "
                            "WHERE lower(word)=lower(?) AND upper(synonym)=upper(?)",
                            (word, synonym)).fetchall()
        for w, s, src in rows:
            _record_deleted(conn, "synonym", word=w, value=s, source=src or "")
        conn.execute("DELETE FROM synonyms_pairs WHERE lower(word)=lower(?) "
                     "AND upper(synonym)=upper(?)", (word, synonym))
        conn.commit()
        n = len(rows)
        return ("Deleted synonym %r = %r (%d row%s; recoverable)." %
                (word, synonym, n, "" if n == 1 else "s")) if n else \
               ("No synonyms_pairs row for %r = %r — nothing deleted (likely a "
                "bidirectional-lookup match, not a stored row)." % (word, synonym))
    finally:
        conn.close()


def delete_definition(definition, answer):
    """Delete the (definition -> answer) row(s) from definition_answers_augmented."""
    definition = (definition or "").strip(); answer = (answer or "").strip()
    if not definition or not answer:
        return "Definition and answer are both required."
    conn = _conn()
    try:
        rows = conn.execute("SELECT definition,answer,source FROM definition_answers_augmented "
                            "WHERE lower(definition)=lower(?) AND upper(answer)=upper(?)",
                            (definition, answer)).fetchall()
        for d, a, src in rows:
            _record_deleted(conn, "definition", word=d, answer=a, source=src or "")
        conn.execute("DELETE FROM definition_answers_augmented WHERE lower(definition)=lower(?) "
                     "AND upper(answer)=upper(?)", (definition, answer))
        conn.commit()
        n = len(rows)
        return ("Deleted definition %r -> %s (%d row%s; recoverable)." %
                (definition, answer, n, "" if n == 1 else "s")) if n else \
               ("No definition row for %r -> %s." % (definition, answer))
    finally:
        conn.close()


def delete_indicator(word, wordplay_type, subtype=None):
    """Delete an indicator typing from indicators (optionally pinned to a subtype)."""
    word = (word or "").strip(); wp = (wordplay_type or "").strip().lower()
    sub = (subtype or "").strip().lower() or None
    if not word or not wp:
        return "Indicator word and type are both required."
    conn = _conn()
    try:
        if sub is None:
            rows = conn.execute("SELECT word,wordplay_type,subtype,source FROM indicators "
                                "WHERE lower(word)=lower(?) AND wordplay_type=?",
                                (word, wp)).fetchall()
            conn.execute("DELETE FROM indicators WHERE lower(word)=lower(?) AND "
                         "wordplay_type=?", (word, wp))
        else:
            rows = conn.execute("SELECT word,wordplay_type,subtype,source FROM indicators "
                                "WHERE lower(word)=lower(?) AND wordplay_type=? AND subtype=?",
                                (word, wp, sub)).fetchall()
            conn.execute("DELETE FROM indicators WHERE lower(word)=lower(?) AND "
                         "wordplay_type=? AND subtype=?", (word, wp, sub))
        for w, t, s, src in rows:
            _record_deleted(conn, "indicator", word=w, wptype=t, subtype=s or "",
                            source=src or "")
        conn.commit()
        n = len(rows)
        return ("Deleted indicator %r (%s%s) (%d row%s; recoverable)." %
                (word, wp, ("/" + sub) if sub else "", n, "" if n == 1 else "s")) if n else \
               ("No indicator row for %r (%s)." % (word, wp))
    finally:
        conn.close()


def delete_link(word):
    """Delete a link word from link_words."""
    word = (word or "").strip()
    if not word:
        return "Link word is required."
    conn = _conn()
    try:
        rows = conn.execute("SELECT word,source FROM link_words WHERE lower(word)=lower(?)",
                            (word,)).fetchall()
        for w, src in rows:
            _record_deleted(conn, "link", word=w, source=src or "")
        conn.execute("DELETE FROM link_words WHERE lower(word)=lower(?)", (word,))
        conn.commit()
        n = len(rows)
        return ("Deleted link word %r (%d row%s; recoverable)." %
                (word, n, "" if n == 1 else "s")) if n else ("No link row for %r." % word)
    finally:
        conn.close()
