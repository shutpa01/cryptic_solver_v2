"""Queue the provisional pieces of PASSED prefill readings that never reached the review queue.

Background (2026-11-11): core/prefill_commit filed DB-unbacked pieces as source='pending' but
never queued them, so a reading passed through /hs Status before Status approved provisional
pieces (or by any other route) kept pieces that never entered the reference DB. This puts each
such piece in pending_enrichments via prefill_commit.queue_provisional — the review queue, NOT
the reference DB — so the user Approves or Rejects each one as normal.

    .venv/Scripts/python.exe scripts/backfill_prefill_queue.py           # list only
    .venv/Scripts/python.exe scripts/backfill_prefill_queue.py --write   # queue them
    add --status pending to look at still-pending prefill readings instead of passes
"""
import argparse
import io
import sqlite3
import sys

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
sys.path.insert(0, ".")

from core import store                                   # noqa: E402
from core import admin_db                                # noqa: E402
from core.wfw_web import (_provisional_pieces, _QUEUE_TYPE, _raw_letters,  # noqa: E402
                          _is_backed, _refresh_provisional_flags)
from core.prefill_commit import queue_provisional        # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--write", action="store_true")
    ap.add_argument("--status", default="pass")
    args = ap.parse_args()

    c = sqlite3.connect("data/clues_master.db")
    ids = [r[0] for r in c.execute(
        "SELECT DISTINCT p.clue_id FROM wfw_piece p JOIN wfw_solve s ON s.clue_id=p.clue_id "
        "WHERE p.source='pending' AND s.solved_by='prefill' AND s.status=? ORDER BY p.clue_id",
        (args.status,))]
    n_new = n_have = n_none = n_backed = 0
    for cid in ids:
        conn = store.connect()
        try:
            parse = store.load_parse(conn, cid)
        finally:
            conn.close()
        src, pnum, num, d, ctext, ans = c.execute(
            "SELECT source, puzzle_number, clue_number, direction, clue_text, answer "
            "FROM clues WHERE id=?", (cid,)).fetchone()
        have = admin_db.pending_for_clue(ctext, _raw_letters(ans))
        print("%d %s %s %s%s %s" % (cid, src, pnum, num, d[0], ans))
        for _r, _o, _p, add, label in _provisional_pieces(parse):
            q = _QUEUE_TYPE.get(add[0]) if add else None
            inq = q and any(t == q and (w or "").strip().lower() == add[1].strip().lower()
                            for _i, t, w, _l, _a in have)
            backed = bool(add) and _is_backed(add)
            tag = ("NO QUEUE TYPE" if q is None else "DB backs it now — stale flag" if backed
                   else "already queued" if inq else "to queue")
            n_none += q is None
            n_backed += bool(q and backed)
            n_have += bool(q and not backed and inq)
            n_new += bool(q and not backed and not inq)
            print("    [%s] %s" % (tag, label))
        if args.write:
            cleared = _refresh_provisional_flags(cid)
            if cleared:
                print("    -> stale flag cleared: %s" % "; ".join(cleared))
            conn = store.connect()
            try:
                parse = store.load_parse(conn, cid)
            finally:
                conn.close()
            queued, unq = queue_provisional(cid, parse)
            print("    -> queued: %s%s" % ("; ".join(queued) or "-",
                                         ("  | unqueueable: " + "; ".join(unq)) if unq else ""))
    print("\n%d clues; %d pieces to queue, %d already queued, %d DB-backed (stale flag), "
          "%d with no queue type%s" % (len(ids), n_new, n_have, n_backed, n_none,
                                       "" if args.write else "  (list only)"))


if __name__ == "__main__":
    main()
