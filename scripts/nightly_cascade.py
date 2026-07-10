"""Nightly WFW cascade — solve today's newly scraped clues (phase 6, 2026-07-10).

Runs every clue of the SERVED papers (telegraph / times / guardian) through the
WFW cascade with DB-only batch wiring (zero AI calls) and persists the parses to
wfw_solve — so the morning starts with engine results and my prefill only has the
genuine fails to read.

Rules:
- Only clues WITH an answer (prize/answerless puzzles are hand-solved in the
  morning — never cascaded blind).
- Only clues with NO stored wfw parse yet (a re-run never overwrites work), and
  never a frozen manual solve (same guard as the hand-solver's re-run).
- Serving papers only: the unserved papers (independent / dailymail) are scraped
  for the corpus but not solved nightly.

Usage:
    python scripts/nightly_cascade.py                 # today's puzzles
    python scripts/nightly_cascade.py --date 2026-07-10
    python scripts/nightly_cascade.py --source telegraph --pnum 31287   # one puzzle
"""

import argparse
import sqlite3
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

SERVE_SOURCES = ["telegraph", "times", "guardian"]


def log(msg):
    print("[%s] %s" % (time.strftime("%H:%M:%S"), msg), flush=True)


def clues_to_solve(target_date=None, source=None, pnum=None):
    """(clue_id, clue_text, answer, source, pnum, direction, enumeration) rows
    needing a first cascade: answer present, no wfw parse, not frozen."""
    con = sqlite3.connect(str(ROOT / "data" / "clues_master.db"), timeout=30)
    try:
        where, params = [], []
        if source and pnum:
            where.append("c.source = ? AND c.puzzle_number = ?")
            params += [source, str(pnum)]
        else:
            where.append("c.publication_date = ?")
            params.append(target_date)
            where.append("c.source IN (%s)" % ",".join("?" * len(SERVE_SOURCES)))
            params += SERVE_SOURCES
        rows = con.execute(
            """SELECT c.id, c.clue_text, c.answer, c.source, c.puzzle_number,
                      c.direction, c.enumeration
               FROM clues c
               WHERE %s
                 AND c.answer IS NOT NULL AND c.answer != ''
                 AND c.clue_text IS NOT NULL AND c.clue_text != ''
                 AND c.id NOT IN (SELECT clue_id FROM wfw_solve)
                 AND c.id NOT IN (SELECT clue_id FROM wfw_frozen)
               ORDER BY c.source, c.puzzle_number,
                        CASE c.direction WHEN 'across' THEN 0 ELSE 1 END,
                        CAST(c.clue_number AS INTEGER)""" % " AND ".join(where),
            params).fetchall()
        skipped = 0
        if not source:
            skipped = con.execute(
                """SELECT COUNT(*) FROM clues c
                   WHERE c.publication_date = ?
                     AND c.source IN (%s)
                     AND (c.answer IS NULL OR c.answer = '')"""
                % ",".join("?" * len(SERVE_SOURCES)),
                [target_date] + SERVE_SOURCES).fetchone()[0]
        return rows, skipped
    finally:
        con.close()


def main():
    ap = argparse.ArgumentParser(description="Nightly WFW cascade")
    ap.add_argument("--date", default=None, help="YYYY-MM-DD (default today)")
    ap.add_argument("--source", default=None)
    ap.add_argument("--pnum", default=None)
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()
    target = args.date or date.today().isoformat()

    rows, skipped_answerless = clues_to_solve(
        target_date=target, source=args.source, pnum=args.pnum)
    scope = ("%s #%s" % (args.source, args.pnum) if args.source
             else "%s (%s)" % (target, "/".join(SERVE_SOURCES)))
    log("cascade scope %s: %d clues to solve, %d answerless skipped"
        % (scope, len(rows), skipped_answerless))
    if not rows:
        return 0
    if args.dry_run:
        for cid, text, *_ in rows[:10]:
            log("  would solve %d: %s" % (cid, text[:60]))
        return 0

    from core import engine_registry
    from core.wfw_web import enum_space   # spaces multi-word answers per enumeration
    wiring = engine_registry.db_only(engine_registry.make_db_wiring())

    counts = {}
    per_puzzle = {}
    for cid, clue_text, answer, src, pn, direction, enum in rows:
        try:
            answer = enum_space(answer, enum)
            _ctx, parse, _name = engine_registry.solve_clue_text(
                clue_text, answer, wiring, source=src, puzzle_number=pn,
                clue_id=cid, direction=direction)
            status = parse.status if parse is not None else "fail"
        except Exception as e:
            log("  ERROR clue %d: %s" % (cid, e))
            status = "error"
        counts[status] = counts.get(status, 0) + 1
        key = (src, pn)
        per_puzzle.setdefault(key, {}).setdefault(status, 0)
        per_puzzle[key][status] += 1

    for (src, pn), c in sorted(per_puzzle.items()):
        log("  %s #%s: %s" % (src, pn, ", ".join(
            "%s=%d" % kv for kv in sorted(c.items()))))
    log("cascade done: %s" % ", ".join("%s=%d" % kv for kv in sorted(counts.items())))
    return 0


if __name__ == "__main__":
    sys.exit(main())
