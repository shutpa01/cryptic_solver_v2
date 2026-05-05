"""One-shot: retract future puzzle URLs from Google's index.

Reads impressions/submitted_future_urls.json (the list of placeholder
puzzle URLs previously submitted to Google's Indexing API as
URL_UPDATED). For URLs that have NOT since become real puzzles, sends
URL_DELETED telling Google to drop them. URLs that have since become
real published puzzles are SKIPPED — we want those indexed.

"Real" is determined by querying data/clues_master.db: a (source, pnum)
pair with at least one clue row counts as real. Local DB and prod DB
are in sync after the most recent dashboard deploy, so the local DB
correctly reflects what prod will serve.

Idempotent and resumable: tracks retracted URLs in
impressions/retracted_future_urls.json. Re-running skips URLs already
retracted.

Daily quota for the Indexing API is 200 calls.

Usage:
    python scripts/retract_future_puzzle_urls.py            # plan only
    python scripts/retract_future_puzzle_urls.py --confirm  # actually send
"""
import argparse
import json
import re
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
SUBMITTED_PATH = PROJECT_ROOT / "impressions" / "submitted_future_urls.json"
RETRACTED_PATH = PROJECT_ROOT / "impressions" / "retracted_future_urls.json"
SA_PATH = PROJECT_ROOT / "impressions" / "indexing_service_account.json"
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"

URL_RE = re.compile(r"^https://justcordelia\.com/(\w+)/cryptic/(\d+)$")


def parse_url(url):
    m = URL_RE.match(url)
    if not m:
        return None, None
    return m.group(1), m.group(2)


def is_real_puzzle(conn, source, pnum):
    """Return True if at least one clue row exists for (source, pnum)."""
    row = conn.execute(
        "SELECT 1 FROM clues WHERE source = ? AND puzzle_number = ? LIMIT 1",
        (source, pnum),
    ).fetchone()
    return row is not None


def load_json_set(path):
    if path.exists():
        with open(path, encoding="utf-8") as f:
            return set(json.load(f))
    return set()


def save_json_set(path, s):
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(sorted(s), f, indent=2)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--confirm", action="store_true",
                        help="Required to actually submit URL_DELETED calls")
    args = parser.parse_args()

    if not SUBMITTED_PATH.exists():
        print(f"ERROR: {SUBMITTED_PATH} not found")
        sys.exit(1)
    if not SA_PATH.exists():
        print(f"ERROR: {SA_PATH} not found")
        sys.exit(1)

    submitted = load_json_set(SUBMITTED_PATH)
    already_retracted = load_json_set(RETRACTED_PATH)
    candidates = sorted(submitted - already_retracted)

    print(f"Submitted future URLs:        {len(submitted)}")
    print(f"Already retracted:            {len(already_retracted)}")
    print(f"Candidates (not yet retracted): {len(candidates)}")

    if not candidates:
        print("Nothing to do.")
        return

    # DB pre-flight: only retract URLs whose puzzle is NOT in clues_master.db.
    # If the puzzle is in the DB it is now a real published puzzle and we
    # want it indexed — DO NOT retract.
    print("\nChecking each URL against local clues_master.db...")
    print("=" * 60)
    conn = sqlite3.connect(str(CLUES_DB))
    todo = []          # not in DB -> retract
    skip_real = []     # in DB -> real puzzle, leave indexed
    skip_bad = []      # URL doesn't match expected pattern
    for i, url in enumerate(candidates, 1):
        source, pnum = parse_url(url)
        if source is None:
            skip_bad.append(url)
            print(f"  [{i:>3}/{len(candidates)}] BAD URL    {url}")
            continue
        if is_real_puzzle(conn, source, pnum):
            skip_real.append(url)
            print(f"  [{i:>3}/{len(candidates)}] real, skip {url}")
        else:
            todo.append(url)
            print(f"  [{i:>3}/{len(candidates)}] retract    {url}")
    conn.close()

    print("=" * 60)
    print(f"To retract (no DB row):       {len(todo)}")
    print(f"Skip (now real, in DB):       {len(skip_real)}")
    print(f"Skip (URL didn't parse):      {len(skip_bad)}")

    if not todo:
        print("\nNothing to retract.")
        return

    if not args.confirm:
        print("\nPlan only. Re-run with --confirm to actually send URL_DELETED.")
        return

    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    creds = service_account.Credentials.from_service_account_file(
        str(SA_PATH),
        scopes=["https://www.googleapis.com/auth/indexing"],
    )
    service = build("indexing", "v3", credentials=creds)

    print(f"\nSubmitting URL_DELETED for {len(todo)} URLs...")
    print("=" * 60)
    retracted_now = set(already_retracted)
    ok = 0
    err = 0
    for i, url in enumerate(todo, 1):
        try:
            service.urlNotifications().publish(
                body={"url": url, "type": "URL_DELETED"}
            ).execute()
            retracted_now.add(url)
            ok += 1
            print(f"  [{i:>3}/{len(todo)}] OK  {url}")
        except Exception as e:
            err += 1
            print(f"  [{i:>3}/{len(todo)}] ERR {url} -- {e}")
        # Persist progress every 10 calls so a crash doesn't lose state
        if i % 10 == 0:
            save_json_set(RETRACTED_PATH, retracted_now)

    save_json_set(RETRACTED_PATH, retracted_now)

    print("=" * 60)
    print(f"Retracted: {ok}    Errors: {err}")
    print(f"Skipped (now real, 200): {len(skip_real)}")
    print(f"State saved to: {RETRACTED_PATH}")


if __name__ == "__main__":
    main()
