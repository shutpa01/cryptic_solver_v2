"""Notify IndexNow (Bing/Yandex/...) of clue + puzzle URLs served since the last deploy.

Run AFTER content is live on the droplet (the deploy step uploads the DB, then calls
this) so every announced URL returns 200. It uses the app's OWN serving rule
(is_served / served_puzzle_numbers) so it can never announce a URL that would 410 — the
same one-truth the sitemap and the 410 gate share.

DEFAULT = incremental: announce only what has been served SINCE the last successful run,
tracked by a watermark (logs/indexnow_watermark.txt = the DB clock at the last run). So
deploying once per finished puzzle announces just that puzzle, each URL exactly once —
the "be first" pattern, no redundant resubmission.

    python scripts/indexnow_notify.py            # incremental (new since last deploy)
    python scripts/indexnow_notify.py --all      # every served URL (one-off backfill)
    python scripts/indexnow_notify.py --days 14  # a rolling window (manual; ignores watermark)
    python scripts/indexnow_notify.py --dry-run  # print URLs, submit nothing, don't advance
    python scripts/indexnow_notify.py --init      # set the watermark to now, announce nothing
"""

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on sys.path

from web import create_app, indexnow

ROOT = Path(__file__).resolve().parent.parent
WATERMARK_FILE = ROOT / "logs" / "indexnow_watermark.txt"
BASE = "https://" + indexnow.HOST


def _db_now(db, offset=None):
    """The DB's own clock (UTC, same frame as wfw_solve.created_at), optionally offset
    e.g. '-2 days'. Keeps watermark comparisons free of client-timezone skew."""
    expr = "datetime('now')" if not offset else "datetime('now', ?)"
    args = () if not offset else (offset,)
    return db.execute("SELECT " + expr, args).fetchone()[0]


def _read_watermark():
    try:
        return WATERMARK_FILE.read_text(encoding="utf-8").strip() or None
    except OSError:
        return None


def _write_watermark(ts):
    WATERMARK_FILE.parent.mkdir(parents=True, exist_ok=True)
    WATERMARK_FILE.write_text(ts, encoding="utf-8")


def collect_urls(db, since):
    """Served clue+puzzle production URLs. since=None -> every served URL; otherwise only
    those whose SERVE time (wfw_solve.created_at) is after `since`. The final gate is the
    app's own is_served / served_puzzle_numbers, so a 410 URL can never be announced."""
    from web.serving import SERVED_SOURCES, is_served, served_puzzle_numbers
    from web.routes.clue import generate_clue_slug
    from web.models import classify_puzzle

    ph = ",".join("?" for _ in SERVED_SOURCES)
    urls = []

    # clues served (optionally: since `since`)
    csql = (f"SELECT c.id, c.source, c.clue_text FROM clues c "
            f"JOIN wfw_solve w ON w.clue_id = c.id AND w.status IN ('pass','invalid') "
            f"WHERE c.source IN ({ph}) AND c.clue_text IS NOT NULL "
            f"  AND c.answer IS NOT NULL AND c.answer != ''")
    cparams = list(SERVED_SOURCES)
    if since is not None:
        csql += " AND w.created_at > ?"
        cparams.append(since)
    for r in db.execute(csql, cparams).fetchall():
        if is_served(r["source"], r["id"]):
            slug = generate_clue_slug(r["clue_text"], clue_id=r["id"])
            if slug:
                urls.append(f"{BASE}/clue/{slug}")

    # puzzles now fully served (optionally: with a clue solved since `since`)
    served = served_puzzle_numbers()
    psql = (f"SELECT c.source, c.puzzle_number, MAX(c.publication_date) AS pub, "
            f"       MAX(w.created_at) AS last_solve FROM clues c "
            f"JOIN wfw_solve w ON w.clue_id = c.id AND w.status IN ('pass','invalid') "
            f"WHERE c.source IN ({ph}) AND c.puzzle_number IS NOT NULL "
            f"  AND c.clue_text IS NOT NULL GROUP BY c.source, c.puzzle_number")
    pparams = list(SERVED_SOURCES)
    if since is not None:
        psql += " HAVING MAX(w.created_at) > ?"
        pparams.append(since)
    for r in db.execute(psql, pparams).fetchall():
        if (r["source"], str(r["puzzle_number"])) in served:
            slug, _ = classify_puzzle(r["source"], r["puzzle_number"], r["pub"])
            if slug:
                urls.append(f"{BASE}/{r['source']}/{slug}/{r['puzzle_number']}")

    return urls


def main():
    ap = argparse.ArgumentParser(description="Notify IndexNow of served URLs")
    ap.add_argument("--all", action="store_true", help="every served URL (one-off backfill)")
    ap.add_argument("--days", type=int, default=None,
                    help="rolling N-day window (manual; ignores/does not advance the watermark)")
    ap.add_argument("--dry-run", action="store_true", help="print URLs, submit nothing")
    ap.add_argument("--init", action="store_true",
                    help="set the watermark to now and announce nothing (baseline after a backfill)")
    args = ap.parse_args()

    app = create_app("development")
    with app.app_context():
        from web.db import get_db
        db = get_db()
        now = _db_now(db)

        if args.init:
            _write_watermark(now)
            print("IndexNow: watermark initialised to %s — nothing announced." % now)
            return 0

        # decide the `since` boundary and whether this run advances the watermark
        advance = False
        if args.all:
            since, scope = None, "ALL served"
        elif args.days is not None:
            since = _db_now(db, "-%d days" % args.days)
            scope = "last %d day(s)" % args.days
        else:
            since = _read_watermark() or _db_now(db, "-2 days")   # first run: a safe 2-day catch
            scope = "new since %s" % since
            advance = True

        urls = collect_urls(db, since)

    print("IndexNow: %d URL(s) in scope (%s)." % (len(urls), scope))
    if args.dry_run:
        for u in urls:
            print("  " + u)
        print("[dry-run] nothing submitted; watermark unchanged.")
        return 0

    if urls:
        status, body = indexnow.submit(urls)
        ok = status in (200, 202)
        print("IndexNow submit -> HTTP %s  %s"
              % (status, "accepted" if ok else "REJECTED: " + body))
        if not ok:
            return 1                       # do NOT advance the watermark on failure — retry next run
    else:
        print("Nothing new to announce.")

    if advance:
        _write_watermark(now)              # everything served up to `now` is now announced
        print("Watermark advanced to %s." % now)
    return 0


if __name__ == "__main__":
    sys.exit(main())
