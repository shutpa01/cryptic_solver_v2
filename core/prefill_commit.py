"""File a nightly-prefill reading as a PENDING committed parse (never pass).

The publish-first review used to mean walking every prefilled clue in /hs
(open, check the seeded grid, Commit) — page swapping and scrolling. User
design 2026-07-12 (memory: prefill-pending-commits): the prefill COMMITS its
validated reading, but ONLY EVER as status='pending' with solved_by='prefill',
so the whole puzzle is reviewable from the clue page in one scroll. Confirm
there (/prefillconfirm) = the user's one click: re-validate, promote to a
FROZEN manual pass, and harvest the reusable pieces to the reference DB.

Safety by construction, like pending-only signatures:
- a prefill filing can NEVER be a pass — only the user's Confirm can;
- a pending parse is never served to the public (the site reads pass only);
- NO freeze and NO reference-DB harvest here — both belong to the Confirm;
- never overwrites user state (existing hand-solver assignments or a frozen
  manual solve) and never re-runs anything.

Used by the nightly prefill (scripts/prompts/nightly_prefill.md). Building +
validation are core.wfw_web._build_manual_parse — the SAME gate the user's
own /hs commit goes through (tile/word coverage, fodder rules, selection
derivation), so an invalid reading cannot be filed at all.

CLI (one clue per call, assignments as a JSON file or literal):
    python -m core.prefill_commit <clue_id> <assignments.json | JSON-string>
"""

import json
import sys

from core import store


def file_pending_prefill(clue_id, assignments):
    """File `assignments` (the /hs payload list) as a pending prefill reading.
    Returns {"ok": bool, "msg": str}. On ok: wfw_hs_assignments seeded AND a
    pending manual parse stored (solved_by='prefill'). Never overwrites."""
    conn = store.connect()
    try:
        if store.get_hs_assignments(conn, clue_id):
            return {"ok": False, "msg": "clue %d already has hand-solver state "
                    "— not overwritten" % clue_id}
        if store.is_frozen(conn, clue_id):
            return {"ok": False, "msg": "clue %d is a frozen manual solve "
                    "— not touched" % clue_id}
    finally:
        conn.close()

    from core.wfw_web import _build_manual_parse
    # verify_db=True: the prefill is an AI reading, so a synonym/abbreviation piece the
    # reference DB does not back is filed PROVISIONAL (never trusted, never harvested) and
    # queued for review (queue_provisional, below — this comment claimed the queueing for
    # four months before any code did it) — an AI cannot fabricate a trusted letter-source.
    built = _build_manual_parse(clue_id, assignments, verify_db=True)
    if not built["ok"]:
        return built                     # the validation gate's own message

    parse = built["parse"]
    parse.status = "pending"             # NEVER pass: the user is the only path
    parse.solved_by = "prefill"          # the clue page badges + gates on this
    parse.warnings = list(parse.warnings or []) + [
        "prefill reading — awaiting your review (Confirm on the clue page)"]

    conn = store.connect()
    try:
        store.set_hs_assignments(conn, clue_id, json.dumps(assignments))
        store.save_parse(conn, clue_id, parse, built["ctx"])
        conn.commit()
    finally:
        conn.close()
    queued, unqueueable = queue_provisional(clue_id, parse)
    msg = "filed PENDING prefill for clue %d (%d piece%s)" % (
        clue_id, built["n_sources"], "" if built["n_sources"] == 1 else "s")
    if queued:
        msg += "; queued for review: %s" % "; ".join(queued)
    if unqueueable:
        msg += "; NO queue type (approve in /hs Status): %s" % "; ".join(unqueueable)
    return {"ok": True, "msg": msg}


def queue_provisional(clue_id, parse):
    """Put every provisional (source='pending') piece of `parse` in pending_enrichments, so the
    clue page's Approve / Approve-all can accept it and Confirm then passes.

    Without this, Confirm refused the reading ("accept it in the review queue") while the queue
    held nothing for it — only pieces the 00:09 cascade happened to queue were ever there — and
    Approve-all approved 0 and refused again (Gdn 30161 7d DIAMONDS / 4d PAINT STRIPPER,
    2026-11-11). Pieces are found by wfw_web._provisional_pieces, the SAME list Confirm's
    approval path files, so what is queued is exactly what blocks the pass.

    Row shape matches what the queue already holds and what pending_for_clue / _drop_queued
    match on: clue_text exactly as in clues, answer letters-only; letters = the value for
    synonym/substitution/homophone, the answer for a definition, the TYPE for an indicator.
    Deduped (same type + word + letters + clue). Returns (queued labels, unqueueable labels)."""
    from core import admin_db
    from core.wfw_web import _provisional_pieces, _QUEUE_TYPE, _raw_letters, _is_backed

    conn = admin_db._mconn()
    try:
        row = conn.execute("SELECT clue_text, answer, source, puzzle_number FROM clues "
                           "WHERE id=?", (clue_id,)).fetchone()
        if not row:
            return [], []
        clue_text, answer, src, pnum = row
        ans = _raw_letters(answer)
        queued, unqueueable = [], []
        for _role, _ord, _piece, add, label in _provisional_pieces(parse):
            qtype = _QUEUE_TYPE.get(add[0]) if add else None
            if qtype is None:
                unqueueable.append(label)
                continue
            if _is_backed(add):         # the DB holds it already: nothing to review
                continue
            word = (add[1] or "").strip()
            if qtype == "indicator":
                letters = (add[2] or "").strip().lower()
            elif qtype == "definition":
                letters = ans
            else:
                letters = "".join(c for c in (add[2] or "").upper() if c.isalpha())
            if not word or not letters:
                unqueueable.append(label)
                continue
            # The queue holds ONE row per (type, word, letters) across ALL clues (UNIQUE
            # constraint). A pair already queued by another clue stays where it is;
            # queued_rows_for() finds it for this clue's Approve-all.
            other = conn.execute("SELECT clue_text FROM pending_enrichments WHERE type=? AND "
                                 "lower(trim(word))=? AND lower(trim(letters))=?",
                                 (qtype, word.lower(), letters.lower())).fetchone()
            if other is not None:
                queued.append(label + (" (already queued)" if other[0] == clue_text
                                       else " (already queued by another clue)"))
                continue
            cur = conn.execute("INSERT OR IGNORE INTO pending_enrichments (type, word, letters, "
                               "answer, clue_text, source, puzzle_number, created_at) "
                               "VALUES (?, ?, ?, ?, ?, ?, ?, datetime('now'))",
                               (qtype, word, letters, ans, clue_text, src or "",
                                str(pnum or "")))
            queued.append(label if cur.rowcount else label + " (already queued)")
        conn.commit()
        return queued, unqueueable
    finally:
        conn.close()


def queued_rows_for(parse):
    """The queue rows that would settle `parse`'s provisional pieces, wherever they were
    queued: [(id, type, word, letters, answer)], the shape of admin_db.pending_for_clue.
    pending_for_clue matches on THIS clue's text only, but the queue is unique per pair, so
    a pair first queued by another clue lives under that clue's text — Approve-all must still
    approve it, or Confirm refuses on a piece the user was never shown (2026-11-11)."""
    from core import admin_db
    from core.wfw_web import _provisional_pieces, _QUEUE_TYPE, _raw_letters

    ans = _raw_letters(parse.answer_text)
    out = []
    conn = admin_db._mconn()
    try:
        for _role, _ord, _piece, add, _label in _provisional_pieces(parse):
            qtype = _QUEUE_TYPE.get(add[0]) if add else None
            if qtype is None:
                continue
            if qtype == "indicator":
                letters = (add[2] or "").strip().lower()
            elif qtype == "definition":
                letters = ans
            else:
                letters = "".join(c for c in (add[2] or "").upper() if c.isalpha())
            row = conn.execute("SELECT id, type, word, letters, answer FROM pending_enrichments "
                               "WHERE type=? AND lower(trim(word))=? AND lower(trim(letters))=?",
                               (qtype, (add[1] or "").strip().lower(), letters.lower())).fetchone()
            if row:
                out.append(tuple(row))
        return out
    finally:
        conn.close()


def main(argv):
    if len(argv) != 2:
        print(__doc__)
        return 2
    clue_id = int(argv[0])
    raw = argv[1]
    try:
        with open(raw, encoding="utf-8") as f:
            assignments = json.load(f)
    except OSError:
        assignments = json.loads(raw)
    out = file_pending_prefill(clue_id, assignments)
    print(out["msg"])
    return 0 if out["ok"] else 1


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
