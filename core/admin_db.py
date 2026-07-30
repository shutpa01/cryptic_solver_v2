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


def has_synonym(word, synonym):
    """True when (word = synonym) is already in synonyms_pairs (case-insensitive,
    matching add_synonym's dedup). The prefill honesty gate's check: an AI-proposed
    synonym piece is trusted only when the reference DB already backs it — otherwise
    it is provisional, never harvested (mirrors has_homophone / has_spoonerism)."""
    word = (word or "").strip()
    synonym = (synonym or "").strip()
    if not word or not synonym:
        return False
    conn = _conn()
    try:
        return conn.execute("SELECT 1 FROM synonyms_pairs WHERE lower(word)=lower(?) "
                            "AND upper(synonym)=upper(?)", (word, synonym)).fetchone() is not None
    finally:
        conn.close()


def has_substitution(word, value):
    """True when (word -> value) is already in the wordplay table (case-insensitive,
    matching add_substitution's dedup). The prefill honesty gate's check for an
    abbreviation/symbol piece — an AI-proposed substitution not in the DB is
    provisional, never harvested."""
    word = (word or "").strip()
    value = (value or "").strip().upper()
    if not word or not value:
        return False
    conn = _conn()
    try:
        return conn.execute("SELECT 1 FROM wordplay WHERE lower(indicator)=lower(?) "
                            "AND upper(substitution)=?", (word, value)).fetchone() is not None
    finally:
        conn.close()


def has_indicator(word, wordplay_type):
    """True when (word) is already typed as `wordplay_type` in the indicators table
    (case-insensitive on word, exact on type — matching triage's is_present check and
    add_indicator's dedup). The prefill honesty gate's check for an indicator piece: an
    AI-proposed indicator is trusted only when the reference DB already types it that way —
    otherwise it is provisional, queued, never harvested (mirrors has_synonym / is_definition)."""
    word = (word or "").strip()
    wp = (wordplay_type or "").strip().lower()
    if not word or not wp:
        return False
    conn = _conn()
    try:
        return conn.execute("SELECT 1 FROM indicators WHERE lower(word)=lower(?) "
                            "AND wordplay_type=?", (word, wp)).fetchone() is not None
    finally:
        conn.close()


def db_derives(word, value):
    """True if the SOLVER'S OWN lookup can derive `value` from `word` — the same
    bidirectional, inflection-aware synonym/abbreviation lookup the engines use
    (core.live_db.LiveDB.get_synonyms / get_abbreviations). The prefill honesty gate
    MUST use THIS, not the directional has_synonym / has_substitution reference-table
    checks: those miss a pair the DB stores the OTHER way round — e.g. 'in' -> HOME is
    stored only as Home->IN (synonyms_pairs) / home->in (wordplay), so the directional
    check re-queued a value the engine can already resolve (clue 10081049 FATHOM,
    2026-07-23). The gate only guards against a FABRICATED source; anything the engine
    can derive from the DB is, by definition, not fabricated, so it must not be queued.
    Best-effort: any error -> False (treat as an enrichment gap, never a false pass)."""
    word = (word or "").strip()
    value = (value or "").strip().upper()
    if not word or not value:
        return False
    try:
        from core.live_db import LiveDB
        db = LiveDB()
        try:
            return value in db.get_synonyms(word) or value in db.get_abbreviations(word)
        finally:
            try:
                db._conn.close()
            except Exception:
                pass
    except Exception:
        return False


def is_definition(phrase, answer):
    """True if the reference DB already backs `phrase` as a definition of `answer` — the
    SAME two tests the engines' defines() use (core.engine_registry):
      1. a synonym-defined answer  (core.live_db.LiveDB.is_definition_of, inflection-aware),
      2. an explicit definition_answers_augmented row (matched on the indexed norm_def key).
    The prefill honesty gate uses THIS so an AI definition the DB already backs is never
    re-queued, while an unbacked one is queued and made provisional. Exact-normalised (no
    inflection growth) so it errs toward QUEUING an uncertain definition, never toward a
    silent trust. Best-effort: any error -> False (treat as an enrichment gap, never a
    false pass)."""
    phrase = (phrase or "").strip()
    answer = (answer or "").strip()
    if not phrase or not answer:
        return False
    # 1) synonym-based definition (LiveDB — the engine's is_definition_of)
    try:
        from core.live_db import LiveDB
        db = LiveDB()
        try:
            if db.is_definition_of(phrase, answer):
                return True
        finally:
            try:
                db._conn.close()
            except Exception:
                pass
    except Exception:
        pass
    # 2) an explicit definition_answers_augmented row (idx_daa_norm on norm_def), the
    #    answer compared letters-only so 'DINING ROOM'/'DININGROOM' both match.
    key = _normalize_key(phrase)
    na = answer.upper().replace(" ", "").replace("-", "")
    try:
        conn = _conn()
        try:
            for (ans,) in conn.execute(
                    "SELECT answer FROM definition_answers_augmented WHERE norm_def=?",
                    (key,)):
                if (ans or "").upper().replace(" ", "").replace("-", "") == na:
                    return True
        finally:
            conn.close()
    except Exception:
        pass
    return False


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


_SPOONERISMS_DDL = """CREATE TABLE IF NOT EXISTS spoonerisms (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    source_phrase TEXT NOT NULL,
    answer_phrase TEXT NOT NULL,
    norm_source TEXT,
    norm_answer TEXT,
    created_at TEXT DEFAULT (datetime('now'))
)"""


def _norm_phrase(s):
    """Letters-only lowercase key for spoonerism matching — spacing and punctuation
    never matter to the sound pair (THE DEAR YACHT == the dear yacht == THEDEARYACHT)."""
    return "".join(c for c in (s or "").lower() if c.isalpha())


def add_spoonerism(source_phrase, answer_phrase):
    """Add a vetted spoonerism PAIR (cryptic_new.db): the source phrase whose transposed
    sounds give the answer phrase, e.g. THE DEAR YACHT -> THE YEAR DOT. Human-curated
    sound knowledge, like the homophones table — the pair IS the justification the manual
    commit gate checks, so nothing here is derived or verified phonetically. Directional
    (source -> answer), one row per pair, deduped on letters-only keys."""
    src = (source_phrase or "").strip().lower()
    ans = (answer_phrase or "").strip().lower()
    ns, na = _norm_phrase(src), _norm_phrase(ans)
    if not ns or not na:
        return "Both the source phrase and the answer are required."
    if ns == na:
        return "A spoonerism pair needs two different phrases."
    conn = _conn()
    try:
        conn.execute(_SPOONERISMS_DDL)
        if conn.execute("SELECT 1 FROM spoonerisms WHERE norm_source=? AND norm_answer=?",
                        (ns, na)).fetchone():
            return "Already present: spoonerism %r -> %r" % (src, ans)
        conn.execute("INSERT INTO spoonerisms (source_phrase, answer_phrase, norm_source, "
                     "norm_answer) VALUES (?, ?, ?, ?)", (src, ans, ns, na))
        conn.commit()
        return "Added spoonerism: %r -> %r" % (src, ans)
    finally:
        conn.close()


def has_spoonerism(source_phrase, answer_phrase):
    """True when the (source -> answer) pair is in the spoonerisms table (letters-only,
    case-insensitive). The manual commit gate's check — a spoonerism piece is only
    accepted when the human has vetted the pair."""
    ns, na = _norm_phrase(source_phrase), _norm_phrase(answer_phrase)
    if not ns or not na:
        return False
    conn = _conn()
    try:
        conn.execute(_SPOONERISMS_DDL)
        return conn.execute("SELECT 1 FROM spoonerisms WHERE norm_source=? AND norm_answer=?",
                            (ns, na)).fetchone() is not None
    finally:
        conn.close()


def has_homophone(spoken, answer_phrase):
    """True when 'spoken' is a SANCTIONED homophone of the answer letters — i.e. the pair
    is in the homophones table (letters-only, case-insensitive, either direction). The
    manual homophone gate's check: a homophone piece is only sanctioned once the human has
    approved the sound-alike pair (mirrors has_spoonerism). Unsanctioned pairs live in
    pending_enrichments (see queue_homophone) until approved."""
    ns, na = _norm_phrase(spoken), _norm_phrase(answer_phrase)
    if not ns or not na:
        return False
    conn = _conn()
    try:
        for (w, h) in conn.execute("SELECT word, homophone FROM homophones"):
            nw, nh = _norm_phrase(w), _norm_phrase(h)
            if (nw == ns and nh == na) or (nw == na and nh == ns):
                return True
        return False
    finally:
        conn.close()


def queue_homophone(spoken, answer_phrase, clue_text, source, puzzle_number):
    """Queue a TENTATIVE homophone pair (spoken sounds like the answer letters) to
    pending_enrichments, so it shows in the enrichment queue for the human to Approve
    (-> add_homophone, sanctioned) or Reject. No write to the live homophones table here —
    the pair is not trusted until approved (user rule 2026-07-17: homophones are infinite,
    so gate on approval, not pre-population). Deduped against the existing queue."""
    sp = (spoken or "").strip()
    al = "".join(c for c in (answer_phrase or "").upper() if c.isalpha())
    if not sp or not al:
        return "A tentative homophone needs a spoken word and answer letters."
    conn = _mconn()
    try:
        if conn.execute("SELECT 1 FROM pending_enrichments WHERE type='homophone' "
                        "AND lower(word)=? AND upper(letters)=?",
                        (sp.lower(), al)).fetchone():
            return "Tentative homophone already queued: %r sounds like %r" % (sp, al)
        conn.execute(
            "INSERT INTO pending_enrichments (type, word, letters, answer, clue_text, "
            "source, puzzle_number, created_at) VALUES ('homophone', ?, ?, ?, ?, ?, ?, "
            "datetime('now'))",
            (sp, al, answer_phrase, clue_text or "", source or "", str(puzzle_number or "")))
        conn.commit()
        return "Queued tentative homophone: %r sounds like %r (approve to sanction)" % (sp, al)
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


def delete_substitution(word, value):
    """Delete the (word -> value) row(s) from the wordplay table (recoverable). The
    spurious-abbreviation case the hand-solver keeps hitting (e.g. a rogue word -> X
    row that blocks the honest assembly)."""
    word = (word or "").strip(); value = (value or "").strip()
    if not word or not value:
        return "Word and value are both required."
    conn = _conn()
    try:
        rows = conn.execute("SELECT indicator, substitution FROM wordplay "
                            "WHERE lower(indicator)=lower(?) AND upper(substitution)=upper(?)",
                            (word, value)).fetchall()
        for w, s in rows:
            _record_deleted(conn, "substitution", word=w, value=s)
        conn.execute("DELETE FROM wordplay WHERE lower(indicator)=lower(?) "
                     "AND upper(substitution)=upper(?)", (word, value))
        conn.commit()
        n = len(rows)
        return ("Deleted abbreviation %r -> %r (%d row%s; recoverable)." %
                (word, value, n, "" if n == 1 else "s")) if n else \
               ("No wordplay row for %r -> %r — nothing deleted." % (word, value))
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


def delete_homophone(word, homophone):
    """Delete a homophone pair (BOTH directions — pairs are stored as two rows) from the
    homophones table (recoverable)."""
    word = (word or "").strip(); homophone = (homophone or "").strip()
    if not word or not homophone:
        return "Word and homophone are both required."
    conn = _conn()
    try:
        rows = conn.execute(
            "SELECT word, homophone FROM homophones WHERE "
            "(lower(word)=lower(?) AND lower(homophone)=lower(?)) OR "
            "(lower(word)=lower(?) AND lower(homophone)=lower(?))",
            (word, homophone, homophone, word)).fetchall()
        for w, h in rows:
            _record_deleted(conn, "homophone", word=w, value=h)
        conn.execute(
            "DELETE FROM homophones WHERE "
            "(lower(word)=lower(?) AND lower(homophone)=lower(?)) OR "
            "(lower(word)=lower(?) AND lower(homophone)=lower(?))",
            (word, homophone, homophone, word))
        conn.commit()
        n = len(rows)
        return ("Deleted homophone %r ~ %r (%d row%s; recoverable)." %
                (word, homophone, n, "" if n == 1 else "s")) if n else \
               ("No homophones row for %r ~ %r — nothing deleted." % (word, homophone))
    finally:
        conn.close()


def delete_spoonerism(source_phrase, answer_phrase):
    """Delete a vetted spoonerism pair (recoverable). Letters-only matching, same as the
    lookup — spacing never matters."""
    ns, na = _norm_phrase(source_phrase), _norm_phrase(answer_phrase)
    if not ns or not na:
        return "Source phrase and answer are both required."
    conn = _conn()
    try:
        conn.execute(_SPOONERISMS_DDL)
        rows = conn.execute("SELECT source_phrase, answer_phrase FROM spoonerisms "
                            "WHERE norm_source=? AND norm_answer=?", (ns, na)).fetchall()
        for s, a in rows:
            _record_deleted(conn, "spoonerism", word=s, value=a)
        conn.execute("DELETE FROM spoonerisms WHERE norm_source=? AND norm_answer=?",
                     (ns, na))
        conn.commit()
        n = len(rows)
        return ("Deleted spoonerism %r -> %r (%d row%s; recoverable)." %
                (source_phrase, answer_phrase, n, "" if n == 1 else "s")) if n else \
               ("No spoonerisms row for %r -> %r — nothing deleted."
                % (source_phrase, answer_phrase))
    finally:
        conn.close()


def search_pairs(word, value):
    """THE standalone delete flow's search (user design 2026-07-14): find every reference-DB
    row matching a typed (word, partner) pair — case-insensitive exact on each GIVEN field,
    at least one required — across synonyms, abbreviations (wordplay), indicators,
    definitions, homophones, spoonerisms and link words. Returns [{kind, word, value}]
    where `value` is what /hsdelete expects for that kind (indicator = 'type' or
    'type/subtype'; link = ''). A search, never a delete.

    WIDENING (user 2026-07-14, the one/is case): the lookup layer matches inflected forms,
    so a rogue pair the solver used may be STORED under a different word (one -> IS came
    from the row ones -> IS). When an exact pair search finds NOTHING, automatically re-run
    partner-only ("stored under a different word") and word-only ("same word, different
    partner"), each hit carrying a `note` saying which — so the row is always findable
    from what the user actually saw on the clue."""
    word = (word or "").strip()
    value = (value or "").strip()
    if not word and not value:
        return []
    out = _search_pairs_exact(word, value)
    if not out and word and value:
        # 1) same partner, RELATED word (shares a stem: ones/one, running/run) — the
        #    inflected-storage case; these are almost always the row the user means.
        for r in _search_pairs_exact(word, value, stem=True):
            r["note"] = "stored under a related word"
            out.append(r)
        # 2) still nothing: same partner under ANY word (capped, alphabetical).
        if not out:
            for r in _search_pairs_exact("", value):
                r["note"] = "stored under a different word"
                out.append(r)
        # 3) still nothing: the typed word with any partner (a partner typo).
        if not out:
            for r in _search_pairs_exact(word, ""):
                r["note"] = "same word, different partner"
                out.append(r)
    return out[:100]


def _search_pairs_exact(word, value, stem=False):
    """One pass of the pair search (see search_pairs). With stem=True the word matches
    RELATED stored words too — either string extends the other (one ~ ones, run ~
    running) — so a pair the lookup layer reached via inflection is findable."""
    out = []
    conn = _conn()

    def _where(col_w, col_v):
        conds, params = [], []
        if word:
            if stem:      # either string extends the other (one ~ ones, run ~ running)
                conds.append("(lower(%s) LIKE lower(?) || '%%' OR "
                             "lower(?) LIKE lower(%s) || '%%')" % (col_w, col_w))
                params.extend([word, word])
            else:
                conds.append("lower(%s)=lower(?)" % col_w)
                params.append(word)
        if value and col_v:
            conds.append("lower(%s)=lower(?)" % col_v)
            params.append(value)
        return " AND ".join(conds), params

    try:
        specs = [
            ("synonym", "synonyms_pairs", "word", "synonym"),
            ("substitution", "wordplay", "indicator", "substitution"),
            ("definition", "definition_answers_augmented", "definition", "answer"),
            ("homophone", "homophones", "word", "homophone"),
            ("spoonerism", "spoonerisms", "source_phrase", "answer_phrase"),
        ]
        for kind, table, cw, cv in specs:
            w, p = _where(cw, cv)
            if not w:
                continue
            try:
                for a, b in conn.execute(
                        "SELECT %s, %s FROM %s WHERE %s LIMIT 40" % (cw, cv, table, w), p):
                    out.append({"kind": kind, "word": a or "", "value": b or ""})
            except Exception:
                pass                                   # table may not exist yet (spoonerisms)
        # indicators: the partner is the TYPE (or type/subtype)
        if word:
            try:
                wp, _, sub = value.partition("/")
                if stem:
                    q = "SELECT word, wordplay_type, COALESCE(subtype,'') FROM indicators " \
                        "WHERE (lower(word) LIKE lower(?) || '%' OR " \
                        "lower(?) LIKE lower(word) || '%')"
                    params = [word, word]
                else:
                    q = "SELECT word, wordplay_type, COALESCE(subtype,'') FROM indicators " \
                        "WHERE lower(word)=lower(?)"
                    params = [word]
                if wp:
                    q += " AND lower(wordplay_type)=lower(?)"
                    params.append(wp)
                if sub:
                    q += " AND lower(COALESCE(subtype,''))=lower(?)"
                    params.append(sub)
                for w2, t, s in conn.execute(q + " LIMIT 40", params):
                    out.append({"kind": "indicator", "word": w2,
                                "value": ("%s/%s" % (t, s)) if s else t})
            except Exception:
                pass
        # link words: no partner
        if word and not value:
            try:
                for (w2,) in conn.execute("SELECT word FROM link_words WHERE "
                                          "lower(word)=lower(?) LIMIT 10", (word,)):
                    out.append({"kind": "link", "word": w2, "value": ""})
            except Exception:
                pass
    finally:
        conn.close()
    return out[:100]


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
