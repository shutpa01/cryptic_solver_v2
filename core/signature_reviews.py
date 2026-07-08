"""Review-capture log for PENDING-only signatures (signature-tiers build §5 Steps 7-8).

When a human confirms (→ the answer IS what the pending-only signature reconstructed) or
rejects (→ it is not / hand-solve) a pending-only signature fire, a row is written here.
That accumulated track record is what makes promotion computable: a pending-only signature
becomes eligible to be promoted to PASS-tier when it has enough clean confirmations across
enough distinct puzzles (§9 decision 5).

  table signature_reviews(id, template_id, clue_id, verdict, reviewed_at)
    verdict: 'confirm' (human agreed → the fire was faithful) | 'reject' (not faithful)

This module owns the table (idempotent CREATE), the write path, and the promotion queries.
It does NOT itself change any verdict or tier automatically — promotion is a human, one-click
action (promote()), and only after a regression hunt (run separately). Claude never calls
promote(): triage is diagnose-only.
"""

import datetime
import os
import sqlite3

_CLUES_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data",
                         "clues_master.db")

# Promotion eligibility thresholds (§9 decision 5) — a pending-only signature earns PASS-tier
# candidacy through use, never automatically.
MIN_CONFIRMATIONS = 10
MAX_REJECTIONS = 0
MIN_DISTINCT_PUZZLES = 3


def _connect(db_path=None):
    return sqlite3.connect(db_path or _CLUES_DB, timeout=30)


def ensure_table(con):
    """Create the signature_reviews table if absent (idempotent)."""
    con.execute(
        "CREATE TABLE IF NOT EXISTS signature_reviews ("
        "id INTEGER PRIMARY KEY, template_id INTEGER NOT NULL, clue_id INTEGER, "
        "verdict TEXT NOT NULL, reviewed_at TEXT NOT NULL)")
    con.execute("CREATE INDEX IF NOT EXISTS ix_sigrev_template "
                "ON signature_reviews(template_id)")


def record(template_id, clue_id, verdict, db_path=None, reviewed_at=None):
    """Log one human review of a pending-only signature fire.

    verdict must be 'confirm' or 'reject'. Returns the new row id."""
    if verdict not in ("confirm", "reject"):
        raise ValueError("verdict must be 'confirm' or 'reject', got %r" % verdict)
    stamp = reviewed_at or datetime.datetime.utcnow().isoformat(timespec="seconds")
    con = _connect(db_path)
    try:
        ensure_table(con)
        rid = con.execute("SELECT COALESCE(MAX(id),0)+1 FROM signature_reviews").fetchone()[0]
        con.execute("INSERT INTO signature_reviews(id,template_id,clue_id,verdict,"
                    "reviewed_at) VALUES(?,?,?,?,?)",
                    (rid, template_id, clue_id, verdict, stamp))
        con.commit()
        return rid
    finally:
        con.close()


def stats(template_id, db_path=None):
    """(confirmations, rejections, distinct_puzzles) for one template's reviews.

    distinct_puzzles counts the distinct (source, puzzle_number) among CONFIRM rows (the
    evidence that matters for promotion), resolved through the clues table."""
    con = _connect(db_path)
    try:
        ensure_table(con)
        conf = con.execute("SELECT COUNT(*) FROM signature_reviews "
                           "WHERE template_id=? AND verdict='confirm'",
                           (template_id,)).fetchone()[0]
        rej = con.execute("SELECT COUNT(*) FROM signature_reviews "
                          "WHERE template_id=? AND verdict='reject'",
                          (template_id,)).fetchone()[0]
        puz = con.execute(
            "SELECT COUNT(*) FROM (SELECT DISTINCT c.source, c.puzzle_number "
            "FROM signature_reviews sr JOIN clues c ON c.id = sr.clue_id "
            "WHERE sr.template_id=? AND sr.verdict='confirm')",
            (template_id,)).fetchone()[0]
        return conf, rej, puz
    finally:
        con.close()


def is_eligible(template_id, db_path=None):
    """True iff the template meets the promotion thresholds (§9 decision 5)."""
    conf, rej, puz = stats(template_id, db_path=db_path)
    return (conf >= MIN_CONFIRMATIONS and rej <= MAX_REJECTIONS
            and puz >= MIN_DISTINCT_PUZZLES)


def eligible_for_promotion(db_path=None):
    """All pending-tier template ids currently eligible for promotion, with their stats.

    Returns list of dicts {template_id, confirmations, rejections, distinct_puzzles}. Only
    considers templates that are STILL tier='pending' (an already-promoted one is skipped)."""
    con = _connect(db_path)
    try:
        ensure_table(con)
        has_tier = any(r[1] == "tier" for r in
                       con.execute("PRAGMA table_info(catalog_templates)"))
        pending_ids = ([r[0] for r in con.execute(
            "SELECT id FROM catalog_templates WHERE tier='pending'")]
            if has_tier else [])
        reviewed = [r[0] for r in con.execute(
            "SELECT DISTINCT template_id FROM signature_reviews")]
    finally:
        con.close()
    out = []
    for tid in pending_ids:
        if tid not in reviewed:
            continue
        conf, rej, puz = stats(tid, db_path=db_path)
        if (conf >= MIN_CONFIRMATIONS and rej <= MAX_REJECTIONS
                and puz >= MIN_DISTINCT_PUZZLES):
            out.append({"template_id": tid, "confirmations": conf,
                        "rejections": rej, "distinct_puzzles": puz})
    return out


def promote(template_id, db_path=None):
    """One-click promotion of a pending-only signature to PASS-tier (human-committed).

    Guarded on eligibility so a caller cannot promote a signature that has not earned it.
    Returns True on promotion, False if not eligible / not pending. THE REGRESSION HUNT
    (promotion moves the signature into the pass-capable cascade — an ordering change) is a
    SEPARATE step the human runs before/after calling this; this function only flips the tier.
    Claude never calls this — triage is diagnose-only."""
    if not is_eligible(template_id, db_path=db_path):
        return False
    con = _connect(db_path)
    try:
        row = con.execute("SELECT tier FROM catalog_templates WHERE id=?",
                          (template_id,)).fetchone()
        if row is None or row[0] != "pending":
            return False
        con.execute("UPDATE catalog_templates SET tier='pass' WHERE id=?", (template_id,))
        con.commit()
        return True
    finally:
        con.close()
