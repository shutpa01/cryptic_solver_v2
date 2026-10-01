"""The clue page's prose block — one definition of where drafts live and what a
record looks like.

    {"10094480": {"sentence": "Net means EARN, and picked up tells you it sounds
                               like ERNE.",
                  "gloss":    "A sea eagle, especially the white tailed variety.",
                  "answer":   "ERNE",
                  "approved": false}}

Written unapproved by `scripts/draft_prose.py`, ticked on /hs, and served only when
`approved` is true.

IT LIVES IN clues_master.db, TABLE clue_prose (user, 2026-10-01: "put the prose in
the DB where it belongs and include it in the deploy"). It used to be the file
logs/prose.json, on the reasoning that a draft is local working state. But an
approved sentence is page content for a clue, and the file only reached the droplet
with a CODE deploy — so on 10-01 every clue of the day went live with its answer and
no prose, because the morning's deploys were database-only. In the DB it travels
with every database upload, the 00:05 auto_deploy included.

The record shape callers see is unchanged — a dict per clue, keyed by the id as a
string — so no caller had to change. Absent `author`/`refused` are simply missing
keys, exactly as they were in the file.

WHY THIS MODULE EXISTS AT ALL. The drafter and the /hs tick both need the path, the
key and the record shape, and this session has twice been bitten by the same fact
living in two places — the clue-type label in two renderers, the homophone middle
read in one place and not another. One definition, imported by both.

APPROVAL IS A SEPARATE ACT FROM DRAFTING. `save_draft` never sets `approved`, and
`set_approved` never rewrites the text unless the user supplies it. So a re-run of
the drafter cannot silently un-approve something the user has already ticked, and
ticking cannot silently alter what was drafted.
"""

import sqlite3
import time

from core import store

# Additive only. One row per clue; the columns are the old JSON record's keys.
_DDL = """CREATE TABLE IF NOT EXISTS clue_prose (
    clue_id    INTEGER PRIMARY KEY,
    sentence   TEXT NOT NULL DEFAULT '',
    gloss      TEXT NOT NULL DEFAULT '',
    answer     TEXT NOT NULL DEFAULT '',
    approved   INTEGER NOT NULL DEFAULT 0,
    refused    TEXT,
    author     TEXT,
    facts_hash TEXT NOT NULL DEFAULT '',
    updated_at TEXT
)"""
_COLS = ("sentence", "gloss", "answer", "approved", "refused", "author", "facts_hash")


def _conn(db_path=None):
    return store.connect(db_path)


def _rec(row):
    """A DB row -> the record shape the file always had."""
    sentence, gloss, answer, approved, refused, author, facts_hash = row
    rec = {"sentence": sentence or "", "gloss": gloss or "", "answer": answer or "",
           "approved": bool(approved), "facts_hash": facts_hash or ""}
    if refused:
        rec["refused"] = refused
    if author:
        rec["author"] = author
    return rec


def load(db_path=None):
    """Every record, keyed by clue id as a string. No table yet = {}."""
    try:
        with _conn(db_path) as c:
            rows = c.execute("SELECT clue_id, " + ", ".join(_COLS)
                             + " FROM clue_prose").fetchall()
    except sqlite3.OperationalError:
        return {}
    return {str(r[0]): _rec(r[1:]) for r in rows}


def get(clue_id, data=None):
    """One clue's record, or None. `data` (from load()) lets a caller read once."""
    if data is not None:
        rec = data.get(str(clue_id))
        return rec if isinstance(rec, dict) else None
    try:
        with _conn() as c:
            row = c.execute("SELECT " + ", ".join(_COLS)
                            + " FROM clue_prose WHERE clue_id = ?",
                            (int(clue_id),)).fetchone()
    except (sqlite3.OperationalError, ValueError, TypeError):
        return None
    return _rec(row) if row else None


def put(clue_id, rec, db_path=None):
    """Write one clue's whole record (replacing any previous one)."""
    with _conn(db_path) as c:
        c.execute(_DDL)
        c.execute("INSERT OR REPLACE INTO clue_prose (clue_id, " + ", ".join(_COLS)
                  + ", updated_at) VALUES (?,?,?,?,?,?,?,?,?)",
                  (int(clue_id), rec.get("sentence") or "", rec.get("gloss") or "",
                   rec.get("answer") or "", 1 if rec.get("approved") else 0,
                   rec.get("refused") or None, rec.get("author") or None,
                   rec.get("facts_hash") or "",
                   time.strftime("%Y-%m-%d %H:%M:%S")))


def sentence_case(text):
    """First letter up, the rest left exactly as written.

    The model hands back a dictionary-style gloss — "a university city in England"
    — and a sentence is a sentence wherever it is printed (user, 2026-09-25). Only
    the first character changes: ALL CAPS values, proper nouns and everything after
    the opening letter are untouched.
    """
    t = (text or "").lstrip()
    return t[:1].upper() + t[1:] if t else t


def save_draft(clue_id, sentence, gloss, answer="", facts_hash=""):
    """File a NEW draft, unapproved. Refuses to touch a record already approved —
    an overnight re-run must never undo the user's tick.

    `facts_hash` fingerprints the record the prose was written from, so a later
    commit can tell whether the reading actually changed. See `facts_unchanged`.
    """
    if _keep(get(clue_id)):
        return False
    put(clue_id, {"sentence": sentence_case(sentence),
                  "gloss": sentence_case(gloss),
                  "answer": (answer or "").upper(), "approved": False,
                  "facts_hash": facts_hash})
    return True


def save_refusal(clue_id, reason, facts_hash=""):
    """Record that NO prose could honestly be written, and why.

    A refusal is a RESULT, not a blank (user, 2026-09-24: a clue that simply showed
    nothing was indistinguishable from the feature being broken). It is also the
    signal most worth seeing, because the drafter refuses when the record does not
    support the sentence — which is usually a fault in the READING, not the prose.

    Carries `facts_hash` for the same reason a draft does: a reading that has not
    changed should not spend another model call to be refused again. Never touches
    an approved record, exactly as save_draft does not.
    """
    if _keep(get(clue_id)):
        return False
    put(clue_id, {"sentence": "", "gloss": "", "answer": "", "approved": False,
                  "refused": (reason or "").strip() or "no reason given",
                  "facts_hash": facts_hash})
    return True


def _keep(rec):
    """True when the drafter must leave this record alone: the user ticked it, or
    the user wrote it. A machine draft may replace a machine draft or a refusal,
    never the user's own words."""
    rec = rec or {}
    return bool(rec.get("approved") or rec.get("author") == "user")


def save_user_text(clue_id, sentence, gloss, approved):
    """File prose the USER typed on /hs, for a clue with no draft or a refused one.

    The box is always there (user, 2026-09-27: "We need a proper process, where I
    just type it in HS") — times 5235 19a BRAIN OF BRITAIN was refused for
    "invents BBC" and there was nowhere to write the sentence by hand. Marked
    `author: user`, so no later drafter run overwrites it (`_keep`). The refusal,
    if any, is dropped: the user's text is the answer to it. Keeps the old
    `facts_hash` so an unchanged reading is not sent to the model again.
    """
    old = get(clue_id) or {}
    put(clue_id, {"sentence": sentence_case((sentence or "").strip()),
                  "gloss": sentence_case((gloss or "").strip()),
                  "answer": old.get("answer", ""), "approved": bool(approved),
                  "author": "user", "facts_hash": old.get("facts_hash", "")})
    return True


def facts_unchanged(clue_id, facts_hash, data=None):
    """True when a draft exists and was written from EXACTLY these facts.

    This is what makes drafting at the end of the nightly the cheap option. The
    user accepts the vast majority of prefill readings unchanged, so by the time
    they commit, the prose sitting in the box was written from the same record
    they are committing — and re-drafting it would spend 8-11 seconds and a model
    call to produce the same paragraph. Only a reading the user CHANGED has a
    different fingerprint, and only that one is drafted again.
    """
    rec = get(clue_id, data)
    return bool(rec and facts_hash and rec.get("facts_hash") == facts_hash)


def set_approved(clue_id, approved, sentence=None, gloss=None):
    """Tick or untick, optionally keeping edits the user made in the box.

    Returns False when there is nothing filed for this clue — approving something
    that does not exist would file a row that no draft ever wrote.
    """
    rec = get(clue_id)
    if rec is None:
        return False
    if sentence is not None:
        rec["sentence"] = sentence.strip()
    if gloss is not None:
        rec["gloss"] = gloss.strip()
    rec["approved"] = bool(approved)
    put(clue_id, rec)
    return True


def approved_text(clue_id, data=None):
    """(sentence, gloss) when the user has ticked it, else None.

    The serving path calls THIS and nothing else, so an unapproved draft cannot
    reach a page by any route.
    """
    rec = get(clue_id, data)
    if not rec or not rec.get("approved") or rec.get("refused"):
        return None
    return (rec.get("sentence") or "", rec.get("gloss") or "")
