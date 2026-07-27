"""Notify IndexNow (Bing/Yandex/...) of clue + puzzle URLs — each puzzle exactly ONCE.

Run AFTER content is live on the droplet (the deploy step uploads the DB, then calls
this). It uses the app's OWN serving rule (is_served / served_puzzle_numbers) so it can
never announce a URL that would 410 — the same one-truth the sitemap and the 410 gate
share.

WHAT'S SENT, AND WHY ONCE
-------------------------
A clue's URL is permanent, so "have we already told Bing about it?" is the only question
that matters — a timestamp is irrelevant (an old clue that gets re-saved must NOT be
re-announced). We track it at the PUZZLE level, because a puzzle is only ever deployed
once every clue is solved (user rule), so a puzzle is a complete, stable unit. The ledger
(logs/indexnow_state.db) records every (source, puzzle_number) whose URLs have been sent.
A normal run sends only puzzles NOT in the ledger — the puzzle page + all its clue pages —
then records the puzzle. Sent exactly once, ever.

    python scripts/indexnow_notify.py                    # send puzzles not yet sent, record them
    python scripts/indexnow_notify.py --dry-run          # show what WOULD be sent; send nothing
    python scripts/indexnow_notify.py --seed-before DATE  # mark puzzles published <= DATE as sent
    python scripts/indexnow_notify.py --seed-all         # mark EVERY served puzzle as sent
    python scripts/indexnow_notify.py --force            # allow a big send with an empty ledger

ONE-TIME MIGRATION: Bing already has the existing pages, so seed the ledger first with
everything already sent, then a normal run only announces genuinely new puzzles. Seeding
by publication date (--seed-before) lets the very recent puzzles fall through as the
catch-up, while everything older is marked done.
"""

import argparse
import sqlite3
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))   # repo root on sys.path

from web import create_app, indexnow

ROOT = Path(__file__).resolve().parent.parent
LEDGER_DB = ROOT / "logs" / "indexnow_state.db"
BASE = "https://" + indexnow.HOST


def _db_now(db):
    """The app DB's own clock, used only to stamp the ledger row (informational)."""
    return db.execute("SELECT datetime('now')").fetchone()[0]


# --- the sent-ledger: one row per puzzle we've announced -----------------------------

def _ledger_conn():
    LEDGER_DB.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(LEDGER_DB))
    conn.execute("CREATE TABLE IF NOT EXISTS sent_puzzle ("
                 "source TEXT NOT NULL, puzzle_number TEXT NOT NULL, "
                 "sent_at TEXT, PRIMARY KEY (source, puzzle_number))")
    conn.commit()
    return conn


def _ledger_sent(conn):
    return {(r[0], r[1]) for r in
            conn.execute("SELECT source, puzzle_number FROM sent_puzzle").fetchall()}


def _ledger_record(conn, source, number, when):
    """Record a puzzle as sent. Returns 1 if newly recorded, 0 if it was already there."""
    cur = conn.execute(
        "INSERT OR IGNORE INTO sent_puzzle (source, puzzle_number, sent_at) VALUES (?, ?, ?)",
        (source, str(number), when))
    conn.commit()
    return cur.rowcount


# --- gather every served puzzle's URLs, grouped by puzzle ----------------------------

def collect_puzzle_urls(db):
    """Every fully-served puzzle -> its production URLs (the puzzle page + each served clue
    page), grouped by (source, puzzle_number). Same serving truth as the sitemap/410 gate,
    so a URL that would 410 is never included. Returns:
        {(source, number): {"urls": [puzzle_url, clue_url, ...], "pub": "YYYY-MM-DD"}}
    """
    from web.serving import SERVED_SOURCES, is_served, served_puzzle_numbers
    from web.routes.clue import generate_clue_slug
    from web.models import classify_puzzle

    ph = ",".join("?" for _ in SERVED_SOURCES)
    served = served_puzzle_numbers()          # (source, str(number)) of FULLY-served puzzles
    out = {}

    # 1) Seed EVERY served puzzle with its own page URL, straight from the served set — so a
    #    puzzle can NEVER be silently dropped just because none of its clues generated a URL.
    psql = (f"SELECT c.source, c.puzzle_number, MAX(c.publication_date) AS pub FROM clues c "
            f"WHERE c.source IN ({ph}) AND c.puzzle_number IS NOT NULL "
            f"GROUP BY c.source, c.puzzle_number")
    for r in db.execute(psql, list(SERVED_SOURCES)).fetchall():
        key = (r["source"], str(r["puzzle_number"]))
        if key not in served:
            continue
        entry = out.setdefault(key, {"urls": [], "pub": r["pub"]})
        slug, _ = classify_puzzle(r["source"], r["puzzle_number"], r["pub"])
        if slug:
            entry["urls"].append(f"{BASE}/{r['source']}/{slug}/{r['puzzle_number']}")

    # 2) Append each served clue's own page URL under its (already-present) puzzle.
    csql = (f"SELECT c.id, c.source, c.puzzle_number, c.clue_text FROM clues c "
            f"JOIN wfw_solve w ON w.clue_id = c.id AND w.status IN ('pass','invalid') "
            f"WHERE c.source IN ({ph}) AND c.clue_text IS NOT NULL "
            f"  AND c.answer IS NOT NULL AND c.answer != ''")
    for r in db.execute(csql, list(SERVED_SOURCES)).fetchall():
        key = (r["source"], str(r["puzzle_number"]))
        if key not in out:                     # puzzle not fully served -> skip its clues
            continue
        if not is_served(r["source"], r["id"]):
            continue
        slug = generate_clue_slug(r["clue_text"], clue_id=r["id"])
        if slug:
            out[key]["urls"].append(f"{BASE}/clue/{slug}")

    return out


def main():
    ap = argparse.ArgumentParser(description="Notify IndexNow of served puzzle URLs (once each)")
    ap.add_argument("--dry-run", action="store_true",
                    help="show the NEW puzzles that would be sent; submit nothing, record nothing")
    ap.add_argument("--seed-before", metavar="YYYY-MM-DD", default=None,
                    help="mark served puzzles published on/before this date as already sent "
                         "(no submission) — the one-time migration boundary")
    ap.add_argument("--seed-all", action="store_true",
                    help="mark EVERY served puzzle as already sent (no submission)")
    ap.add_argument("--force", action="store_true",
                    help="allow a large send even when the ledger is empty (bypass the guard)")
    args = ap.parse_args()

    app = create_app("development")
    with app.app_context():
        from web.db import get_db
        db = get_db()
        now = _db_now(db)
        puzzles = collect_puzzle_urls(db)

    conn = _ledger_conn()
    sent = _ledger_sent(conn)

    # ---- seeding (one-time migration): record without sending ----
    if args.seed_all or args.seed_before:
        added = 0
        for (source, number), entry in puzzles.items():
            if args.seed_before and (entry["pub"] or "") > args.seed_before:
                continue
            added += _ledger_record(conn, source, number, now)
        scope = "ALL served" if args.seed_all else ("published on/before " + args.seed_before)
        print("IndexNow seed (%s): %d puzzle(s) newly marked sent; ledger now holds %d."
              % (scope, added, len(_ledger_sent(conn))))
        return 0

    # ---- normal run: send puzzles not yet in the ledger ----
    new = {k: v for k, v in puzzles.items() if k not in sent}
    total_urls = sum(len(v["urls"]) for v in new.values())
    print("IndexNow: %d served puzzles, %d already sent, %d new (%d URLs)."
          % (len(puzzles), len(sent), len(new), total_urls))

    # safety guard: never blast the whole site because the ledger was never seeded
    if not sent and len(new) > 20 and not (args.dry_run or args.force):
        print("REFUSING: the ledger is empty and %d puzzles look 'new'. Seed first "
              "(--seed-before YYYY-MM-DD or --seed-all), or pass --force to override."
              % len(new))
        return 1

    if args.dry_run:
        for (source, number), v in sorted(new.items()):
            print("  NEW  %-11s %-8s  %2d URL(s)  (pub %s)"
                  % (source, number, len(v["urls"]), v["pub"]))
        print("[dry-run] nothing submitted; ledger unchanged.")
        return 0

    if not new:
        print("Nothing new to announce.")
        return 0

    sent_ok = fail = 0
    for (source, number), v in sorted(new.items()):
        if not v["urls"]:                                # served but nothing to send — never silent
            fail += 1
            print("  WARN   %-11s %-8s  served but produced 0 URLs — not sent, not recorded "
                  "(needs a look)" % (source, number))
            continue
        status, body = indexnow.submit(v["urls"])       # one puzzle's URLs
        if status in (200, 202):
            _ledger_record(conn, source, number, now)    # record ONLY on success
            sent_ok += 1
            print("  sent   %-11s %-8s  %2d URL(s)  OK" % (source, number, len(v["urls"])))
        else:
            fail += 1
            print("  FAILED %-11s %-8s  HTTP %s  %s  (not recorded, will retry next run)"
                  % (source, number, status, body))
    print("Done: %d puzzle(s) sent, %d failed." % (sent_ok, fail))
    return 1 if fail else 0


if __name__ == "__main__":
    sys.exit(main())
