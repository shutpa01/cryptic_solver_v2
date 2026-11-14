"""Overnight auto-approve — the user's morning "pending has prose -> approve" check, done
by the nightly so those clues can be served hours earlier (user, 2026-11-14):

    "ATM all i do is check that pending clues have prose, if they do I approve the
    explanation and the prose because they have proved to be extremely reliable...
    automate my 'checking' of the pending and get those clues on the site as part of
    the overnight job, then I fix the genuine fails."

The user chose option A: approve the AI-proposed reference entries too, unseen.

For each clue of the day's serving puzzles:
  1. A PENDING PREFILL reading whose prose is drafted (not refused) -> exactly the
     Approve-all click (core.wfw_web.approve_all: every queued enrichment into the
     reference DB, then Confirm through the hand-commit gate, frozen as a pass). If it
     is now a pass, its prose is approved. If Confirm refuses, nothing is ticked and
     the clue stays for the user.
  2. An ENGINE PASS with drafted, unapproved prose -> the prose is approved.
Left for the user: fails, INVALIDs, pendings with no prose or refused prose, readings
Confirm refuses, and the prose of the user's own hand solves (solved_by 'manual').

    python scripts/auto_approve.py --date 2026-11-14
    python scripts/auto_approve.py --source times --pnum 29700
    python scripts/auto_approve.py --date 2026-11-14 --dry-run   # list, change nothing
"""

import argparse
import sqlite3
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


# The nightly captures this output as UTF-8; on a Windows pipe Python would otherwise
# write cp1252 and crash on the first character it cannot encode.
try:
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
except Exception:
    pass


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def _clues(conn, date, source, pnum, served):
    ph = ",".join("?" for _ in served)
    sql = ("SELECT c.id, c.source, c.puzzle_number, c.clue_number, c.direction, c.answer, "
           "       s.status, COALESCE(s.solved_by, ''), "
           "       p.clue_id IS NOT NULL, COALESCE(p.approved, 0), p.refused, "
           "       COALESCE(p.sentence, '') "
           "FROM clues c "
           "LEFT JOIN wfw_solve s ON s.clue_id = c.id "
           "LEFT JOIN clue_prose p ON p.clue_id = c.id "
           "WHERE c.source IN (%s) " % ph)
    args = list(served)
    if pnum:
        sql += "AND c.source = ? AND c.puzzle_number = ? "
        args += [source, str(pnum)]
    else:
        sql += "AND c.publication_date = ? "
        args.append(date)
    return conn.execute(sql + "ORDER BY c.source, c.puzzle_number, c.direction, "
                              "CAST(c.clue_number AS INTEGER)", args).fetchall()


def main():
    ap = argparse.ArgumentParser(description="Overnight auto-approve of prefilled clues")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD: every serving puzzle that day")
    ap.add_argument("--source", default=None)
    ap.add_argument("--pnum", default=None)
    ap.add_argument("--dry-run", action="store_true", help="list what would be done")
    args = ap.parse_args()
    if not (args.date or (args.source and args.pnum)):
        ap.error("need --date, or --source + --pnum")

    from core import store, prose_store
    from web.serving import SERVED_SOURCES

    conn = store.connect()
    try:
        rows = _clues(conn, args.date, args.source, args.pnum, SERVED_SOURCES)
    finally:
        conn.close()
    scope = ("%s %s" % (args.source, args.pnum)) if args.pnum else args.date
    log("auto-approve %s: %d clues%s" % (scope, len(rows), "  [DRY RUN]" if args.dry_run else ""))

    confirmed, prose_ticked, refused, left = 0, 0, [], []
    wfw_web = None
    for (cid, src, pn, num, dirn, ans, status, by, has_prose, approved, prose_refused,
         sentence) in rows:
        tag = "%s %s %s%s %s" % (src, pn, num, (dirn or "")[:1], ans)
        drafted = bool(has_prose and not prose_refused and sentence.strip())

        if status == "pending" and by == "prefill":
            if not drafted:
                left.append("%s — pending, %s" % (
                    tag, "prose refused" if prose_refused else "no prose"))
                continue
            if args.dry_run:
                log("  would Approve-all + approve prose: %s" % tag)
                continue
            if wfw_web is None:
                from core import wfw_web          # heavy import; only when needed
            msg = wfw_web.approve_all(cid, draft_prose=False)
            c2 = store.connect()
            try:
                now = c2.execute("SELECT status FROM wfw_solve WHERE clue_id = ?",
                                 (cid,)).fetchone()
            finally:
                c2.close()
            if now and now[0] == "pass":
                prose_store.set_approved(cid, True)
                confirmed += 1
                log("  CONFIRMED %s — %s" % (tag, msg))
            else:
                refused.append("%s — %s" % (tag, msg))
                log("  NOT confirmed %s — %s" % (tag, msg))

        elif status == "pass" and by not in ("manual", "prefill"):
            if drafted and not approved:
                if args.dry_run:
                    log("  would approve prose (engine pass): %s" % tag)
                    continue
                prose_store.set_approved(cid, True)
                prose_ticked += 1

        elif status in (None, "fail", "pending"):
            left.append("%s — %s" % (tag, status or "no parse"))

    log("auto-approve %s done: %d prefill readings confirmed, %d engine-pass prose "
        "approved, %d refused by Confirm, %d left for you"
        % (scope, confirmed, prose_ticked, len(refused), len(left) + len(refused)))
    for line in refused:
        log("  REFUSED: " + line)
    for line in left:
        log("  FOR YOU: " + line)
    return 0


if __name__ == "__main__":
    sys.exit(main())
