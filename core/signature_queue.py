"""Pending-signature queue — auto-DISCOVERED catalog signatures awaiting human approval.

The cascade can DISCOVER the signature a failing clue needs and PROVE it (the signature,
trialled in-memory, must produce a clean pass via itself — see
catalog_creator.auto_discover_and_queue). Rather than file it straight into the catalog,
we queue it here so a human approves it, exactly like the enrichment queue for
synonyms/definitions/indicators.

A queued row stores only what add_signature needs (operation, roles, n_words, def_pos +
the signature string) plus the triggering clue and the verified parse text for review.
Approve -> catalog_creator.add_signature (origin='approved'); Reject -> remembered so the
same shape is not re-queued. Lives in clues_master.db alongside the catalog.
"""

import json
import os
import sqlite3

_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data", "clues_master.db")


def _conn(db_path=None):
    return sqlite3.connect(db_path or _DB, timeout=30)


def ensure_table(con):
    con.execute("""
        CREATE TABLE IF NOT EXISTS pending_signatures (
            id          INTEGER PRIMARY KEY,
            signature   TEXT NOT NULL,
            operation   TEXT NOT NULL,
            roles       TEXT NOT NULL,      -- JSON list
            n_words     TEXT NOT NULL,      -- JSON list
            def_pos     TEXT,
            clue_id     INTEGER,
            clue_text   TEXT,
            answer      TEXT,
            parse_text  TEXT,               -- the verified would-be parse, for review
            status      TEXT NOT NULL DEFAULT 'pending',   -- pending | rejected
            created_at  TEXT
        )""")
    con.commit()


def _in_catalog(con, signature):
    return con.execute("SELECT 1 FROM catalog_templates WHERE signature=?",
                       (signature,)).fetchone() is not None


def queue(cand_min, clue_id, clue_text, answer, parse_text, created_at, db_path=None):
    """Queue a discovered signature for approval. `cand_min` = {signature, operation,
    roles, n_words, def_pos}. No-op (returns None) if the signature is already in the
    catalog, already queued, or previously rejected. Returns the new row id otherwise."""
    con = _conn(db_path)
    try:
        ensure_table(con)
        sig = cand_min["signature"]
        if _in_catalog(con, sig):
            return None
        dup = con.execute("SELECT 1 FROM pending_signatures WHERE signature=?",
                          (sig,)).fetchone()
        if dup:
            return None
        rid = con.execute("INSERT INTO pending_signatures(signature,operation,roles,"
                          "n_words,def_pos,clue_id,clue_text,answer,parse_text,status,"
                          "created_at) VALUES(?,?,?,?,?,?,?,?,?,'pending',?)",
                          (sig, cand_min["operation"], json.dumps(cand_min["roles"]),
                           json.dumps(cand_min["n_words"]), cand_min["def_pos"],
                           clue_id, clue_text, answer, parse_text, created_at)).lastrowid
        con.commit()
        return rid
    finally:
        con.close()


def is_known(signature, db_path=None):
    """True if the signature is already in the catalog OR already pending/rejected — so
    the discovery hook can cheaply skip re-proposing it."""
    con = _conn(db_path)
    try:
        ensure_table(con)
        if _in_catalog(con, signature):
            return True
        return con.execute("SELECT 1 FROM pending_signatures WHERE signature=?",
                           (signature,)).fetchone() is not None
    finally:
        con.close()


def list_pending(db_path=None):
    con = _conn(db_path)
    try:
        ensure_table(con)
        rows = con.execute(
            "SELECT id,signature,operation,def_pos,clue_id,clue_text,answer,parse_text,"
            "created_at FROM pending_signatures WHERE status='pending' ORDER BY id").fetchall()
        cols = ["id", "signature", "operation", "def_pos", "clue_id", "clue_text",
                "answer", "parse_text", "created_at"]
        return [dict(zip(cols, r)) for r in rows]
    finally:
        con.close()


def count_pending(db_path=None):
    con = _conn(db_path)
    try:
        ensure_table(con)
        return con.execute(
            "SELECT COUNT(*) FROM pending_signatures WHERE status='pending'").fetchone()[0]
    finally:
        con.close()


def approve(row_id, db_path=None):
    """File a queued signature into the catalog (origin='approved') and remove it from the
    queue. Returns (template_id, signature) or (None, reason)."""
    from core import catalog_creator
    con = _conn(db_path)
    try:
        ensure_table(con)
        row = con.execute("SELECT signature,operation,roles,n_words,def_pos "
                          "FROM pending_signatures WHERE id=? AND status='pending'",
                          (row_id,)).fetchone()
        if not row:
            return (None, "not found / not pending")
        sig, operation, roles_j, nwords_j, def_pos = row
    finally:
        con.close()
    cand = {"operation": operation, "roles": json.loads(roles_j),
            "n_words": json.loads(nwords_j), "def_pos": def_pos}
    tid = catalog_creator.add_signature(
        cand, note="approved from auto-discovery queue", db_path=db_path,
        backup=True, origin="approved")
    con = _conn(db_path)
    try:
        con.execute("DELETE FROM pending_signatures WHERE id=?", (row_id,))
        con.commit()
    finally:
        con.close()
    if tid is None:
        return (None, "already in catalog (removed from queue)")
    return (tid, sig)


def reject(row_id, db_path=None):
    """Mark a queued signature rejected (kept so the shape is not re-queued)."""
    con = _conn(db_path)
    try:
        ensure_table(con)
        con.execute("UPDATE pending_signatures SET status='rejected' WHERE id=?",
                    (row_id,))
        con.commit()
    finally:
        con.close()
