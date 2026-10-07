#!/usr/bin/env python3
"""Announce INDIVIDUAL clue URLs to IndexNow, without touching the puzzle ledger.

WHY THIS EXISTS (2026-09-17). scripts/indexnow_notify.py announces a puzzle as ONE
atomic unit, only once every clue is served, and records it in `sent_puzzle` — which
is write-once. That is right for the daily deploy and wrong for an experiment: we
want to announce a handful of already-passing clue pages the moment they are live,
hours before the rest of the puzzle is confirmed, and we must NOT let that write a
ledger row, because the morning run would then skip the puzzle and the remaining
clue pages would never be announced at all.

So this script:
  * never opens logs/indexnow_state.db — the puzzle ledger is untouched, and the
    normal deploy announce still fires exactly as it does today;
  * announces only clue URLs that web.serving.is_served says get a public page, so
    it can never push a URL that answers 410;
  * DOES NOTHING unless you pass --send. Announcing is an outward-facing act that
    cannot be recalled, so the default is a dry run that prints the list.

Typical use on the night of the experiment:

    # see what arm 1 of 2 would announce for today's puzzles
    python scripts/announce_clues.py --arm 1/2

    # actually announce it
    python scripts/announce_clues.py --arm 1/2 --send

    # ...and the other arm later, from the same deployed state
    python scripts/announce_clues.py --arm 2/2 --send

Arms are INTERLEAVED, not split down the middle: arm 1 of 2 takes the 1st, 3rd,
5th... clue in puzzle order. Bing appears to work a submitted list in order (on
2026-09-17 the Guardian's indexed pages were the first ten by clue id), so a
contiguous split would confound "which arm" with "how far down the list".
"""

import argparse
import sys
import time
from datetime import date
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from web import create_app, indexnow

BASE = "https://" + indexnow.HOST
LOG_PATH = ROOT / "logs" / "announce_clues.log"

# A hand-run experiment should never push a large batch. The daily deploy is the
# only thing that announces at volume, and it has its own guard.
MAX_URLS_WITHOUT_FORCE = 40


def _log(line):
    """Append to the run log AND print it, so the terminal and the file agree."""
    stamp = time.strftime("%Y-%m-%d %H:%M:%S")
    LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(LOG_PATH, "a", encoding="utf-8") as fh:
        fh.write("[%s] %s\n" % (stamp, line))
    print(line)


def collect_clue_urls(db, target_date=None, source=None, puzzle=None):
    """Served clue pages for a date (or one puzzle), in puzzle reading order.

    SAME serving truth as the clue route and the sitemap (web.serving.is_served), so
    a URL here always answers 200. Deliberately does NOT require the whole puzzle to
    be served — that is the entire point of this script.
    """
    from web.serving import SERVED_SOURCES, is_served
    from web.routes.clue import generate_clue_slug

    where, params = [], []
    if source and puzzle:
        where.append("c.source = ? AND c.puzzle_number = ?")
        params += [source, str(puzzle)]
    else:
        where.append("c.publication_date = ?")
        params.append(target_date)
    where.append("c.source IN (%s)" % ",".join("?" * len(SERVED_SOURCES)))
    params += list(SERVED_SOURCES)

    rows = db.execute(
        """SELECT c.id, c.source, c.puzzle_number, c.clue_text, w.status
             FROM clues c
             JOIN wfw_solve w ON w.clue_id = c.id AND w.status IN ('pass', 'invalid')
            WHERE %s
              AND c.clue_text IS NOT NULL AND c.clue_text != ''
              AND c.answer IS NOT NULL AND c.answer != ''
            ORDER BY c.source, c.puzzle_number,
                     CASE c.direction WHEN 'across' THEN 0 ELSE 1 END,
                     CAST(c.clue_number AS INTEGER)""" % " AND ".join(where),
        params).fetchall()

    out = []
    for r in rows:
        if not is_served(r["source"], r["id"]):
            continue                       # invalid-without-comment, frozen, etc.
        slug = generate_clue_slug(r["clue_text"], clue_id=r["id"])
        if slug:
            out.append((r["source"], str(r["puzzle_number"]), r["status"],
                        BASE + "/clue/" + slug))
    return out


def parse_arm(spec):
    """'1/2' -> (0, 2). Returns (index, stride) or None."""
    if not spec:
        return None
    try:
        k, n = spec.split("/")
        k, n = int(k), int(n)
    except ValueError:
        raise SystemExit("--arm wants K/N, e.g. 1/2")
    if not (1 <= k <= n):
        raise SystemExit("--arm K must be between 1 and N")
    return (k - 1, n)


def main():
    ap = argparse.ArgumentParser(
        description="Announce individual served clue URLs to IndexNow. "
                    "Dry run unless --send. Never touches the puzzle ledger.")
    ap.add_argument("--date", metavar="YYYY-MM-DD", default=None,
                    help="publication date to announce (default: today)")
    ap.add_argument("--source", default=None, help="one source, e.g. guardian")
    ap.add_argument("--puzzle", default=None, help="one puzzle number (needs --source)")
    ap.add_argument("--arm", metavar="K/N", default=None,
                    help="announce only every Nth clue starting at K (interleaved)")
    ap.add_argument("--send", action="store_true",
                    help="ACTUALLY submit. Without this the script only prints the list.")
    ap.add_argument("--force", action="store_true",
                    help="allow a batch larger than %d URLs" % MAX_URLS_WITHOUT_FORCE)
    args = ap.parse_args()

    if args.puzzle and not args.source:
        raise SystemExit("--puzzle needs --source")
    arm = parse_arm(args.arm)
    target = args.date or date.today().isoformat()

    app = create_app("development")
    with app.app_context():
        from web.db import get_db
        rows = collect_clue_urls(get_db(), target_date=target,
                                 source=args.source, puzzle=args.puzzle)

    scope = ("%s #%s" % (args.source, args.puzzle)) if args.puzzle else target
    if not rows:
        _log("announce_clues: %s — no served clue pages. Nothing to do." % scope)
        return 0

    by_puzzle = {}
    for src, pnum, _st, _u in rows:
        by_puzzle[(src, pnum)] = by_puzzle.get((src, pnum), 0) + 1
    summary = ", ".join("%s #%s: %d" % (s, p, n) for (s, p), n in sorted(by_puzzle.items()))
    _log("announce_clues: %s — %d served clue page(s) [%s]" % (scope, len(rows), summary))

    if arm:
        k, n = arm
        rows = rows[k::n]
        _log("  arm %s -> %d URL(s)" % (args.arm, len(rows)))

    urls = [u for _s, _p, _st, u in rows]
    for u in urls:
        print("   " + u)

    if not urls:
        _log("  nothing in this arm; nothing sent.")
        return 0

    if not args.send:
        _log("  DRY RUN — nothing submitted. Re-run with --send to announce these "
             "%d URL(s)." % len(urls))
        return 0

    if len(urls) > MAX_URLS_WITHOUT_FORCE and not args.force:
        _log("  REFUSING: %d URLs is more than this script's %d limit. The daily "
             "deploy is what announces at volume. Pass --force if you mean it."
             % (len(urls), MAX_URLS_WITHOUT_FORCE))
        return 1

    status, body = indexnow.submit(urls, timeout=15)
    if status in (200, 202):
        _log("  SENT %d URL(s) — HTTP %s %s. Puzzle ledger untouched." % (len(urls), status, body))
        for u in urls:
            _log("    sent %s" % u)
        return 0
    _log("  FAILED: HTTP %s %s — nothing recorded, safe to re-run." % (status, body))
    return 1


if __name__ == "__main__":
    sys.exit(main())
