"""Notify IndexNow (Bing/Yandex/...) of recently-served clue + puzzle URLs.

Run AFTER the content is live on the droplet (the deploy step uploads the DB, then
calls this) so every announced URL returns 200. It uses the app's OWN serving rule
(is_served / served_puzzle_numbers) so it can never announce a URL that would 410 —
the same one-truth the sitemap and the 410 gate share.

Scope: clues solved and puzzles published within --days (default 3). In the week-only
model that is exactly the fresh content; --all announces the whole served set (one-off).

    python scripts/indexnow_notify.py              # last 3 days
    python scripts/indexnow_notify.py --days 14
    python scripts/indexnow_notify.py --all        # every served URL (one-off backfill)
    python scripts/indexnow_notify.py --dry-run    # print URLs, submit nothing
"""

import argparse
import sys
from datetime import date, timedelta
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on sys.path

from web import create_app, indexnow

BASE = "https://" + indexnow.HOST


def recent_served_urls(days=3, all_urls=False):
    """The production URLs of served clues/puzzles that are new within `days`
    (or every served URL when all_urls). Gated by the live serving rule."""
    from web.db import get_db
    from web.serving import SERVED_SOURCES, is_served, served_puzzle_numbers
    from web.routes.clue import generate_clue_slug
    from web.models import classify_puzzle

    db = get_db()
    ph = ",".join("?" for _ in SERVED_SOURCES)
    cutoff = None if all_urls else (date.today() - timedelta(days=days)).isoformat()
    urls = []

    # --- clue pages (served, recently solved or published) ---
    sql = (f"SELECT c.id, c.source, c.clue_text, c.publication_date, "
           f"       w.created_at AS solved_at "
           f"FROM clues c "
           f"JOIN wfw_solve w ON w.clue_id = c.id AND w.status IN ('pass','invalid') "
           f"WHERE c.source IN ({ph}) AND c.clue_text IS NOT NULL "
           f"  AND c.answer IS NOT NULL AND c.answer != ''")
    params = list(SERVED_SOURCES)
    if cutoff:
        sql += " AND (w.created_at >= ? OR c.publication_date >= ?)"
        params += [cutoff, cutoff]
    for r in db.execute(sql, params).fetchall():
        if not is_served(r["source"], r["id"]):
            continue
        slug = generate_clue_slug(r["clue_text"], clue_id=r["id"])
        if slug:
            urls.append(f"{BASE}/clue/{slug}")

    # --- puzzle pages (fully-served, recently published) ---
    served = served_puzzle_numbers()
    psql = (f"SELECT source, puzzle_number, MAX(publication_date) AS pub "
            f"FROM clues WHERE source IN ({ph}) AND puzzle_number IS NOT NULL "
            f"  AND clue_text IS NOT NULL")
    pparams = list(SERVED_SOURCES)
    if cutoff:
        psql += " AND publication_date >= ?"
        pparams.append(cutoff)
    psql += " GROUP BY source, puzzle_number"
    for r in db.execute(psql, pparams).fetchall():
        if (r["source"], str(r["puzzle_number"])) not in served:
            continue
        slug, _ = classify_puzzle(r["source"], r["puzzle_number"], r["pub"])
        if slug:
            urls.append(f"{BASE}/{r['source']}/{slug}/{r['puzzle_number']}")

    return urls


def main():
    ap = argparse.ArgumentParser(description="Notify IndexNow of recently-served URLs")
    ap.add_argument("--days", type=int, default=3)
    ap.add_argument("--all", action="store_true", help="every served URL (one-off backfill)")
    ap.add_argument("--dry-run", action="store_true", help="print URLs, submit nothing")
    args = ap.parse_args()

    app = create_app("development")
    with app.app_context():
        urls = recent_served_urls(days=args.days, all_urls=args.all)

    scope = "ALL served" if args.all else "last %d day(s)" % args.days
    print("IndexNow: %d served URL(s) in scope (%s)." % (len(urls), scope))
    if args.dry_run:
        for u in urls:
            print("  " + u)
        print("[dry-run] nothing submitted.")
        return 0
    if not urls:
        print("Nothing to submit.")
        return 0
    status, body = indexnow.submit(urls)
    ok = status in (200, 202)
    print("IndexNow submit -> HTTP %s  %s" % (status, "accepted" if ok else "REJECTED: " + body))
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
