#!/usr/bin/env python3
"""Times Grid Backfill — extract grids from local JSON files and fetch grids
from the Times public archive for puzzles >= #29337.

Phase 1: Extract grids from local JSON backup files (instant).
Phase 2: Walk the Times archive sitemap, fetch puzzle pages via HTTP,
         extract API URLs, download grid solutions.

Puzzles below #29337 don't expose the API URL in their HTML — those need
the Selenium-based times_harvest.py separately.

Usage:
    python scraper/times/times_grid_backfill.py
    python scraper/times/times_grid_backfill.py --test    # 10 puzzles only
"""

import argparse
import json
import re
import sqlite3
import time
from pathlib import Path

import requests

SCRIPT_DIR = Path(__file__).resolve().parent
DB_PATH = SCRIPT_DIR.parent.parent / "data" / "clues_master.db"

HTTP_CUTOFF = 29337  # Puzzles below this don't expose API URL in page HTML

SITEMAP_URL = "https://www.thetimes.com/html-puzzles-sitemap"

API_URL_PATTERN = re.compile(
    r'(https://feeds\.thetimes\.(?:com|co\.uk)/puzzles/sp/[^"\']+)'
)

# Patterns for puzzle links on week pages
PUZZLE_LINK_PATTERNS = {
    'cryptic': re.compile(r'href="(/puzzles/crossword/times-cryptic-no-(\d+)-[^"]+)"'),
    'sunday-cryptic': re.compile(r'href="(/puzzles/crossword/sunday-times-cryptic-no-(\d+)-[^"]+)"'),
}


def get_puzzles_needing_grids(conn):
    """Return set of (puzzle_number_str) for Times puzzles without grids."""
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT p.puzzle_number
        FROM clues p
        LEFT JOIN puzzle_grids g ON p.source = g.source AND p.puzzle_number = g.puzzle_number
        WHERE p.source = 'times'
          AND g.source IS NULL
    """)
    return set(row[0] for row in c.fetchall())


def save_grid(conn, puzzle_number, solution, rows, cols, url):
    """Save grid to puzzle_grids. api_folder stores the source URL."""
    conn.execute("""
        INSERT OR REPLACE INTO puzzle_grids
        (source, puzzle_number, solution, grid_rows, grid_cols, api_folder)
        VALUES (?, ?, ?, ?, ?, ?)
    """, ('times', str(puzzle_number), solution, rows, cols, url))
    conn.commit()


def phase1_local_jsons(conn, needing_grids):
    """Extract grids from local JSON backup files."""
    print("=" * 60)
    print("PHASE 1: Local JSON files")
    print("=" * 60)

    json_files = list(SCRIPT_DIR.glob("times_*.json"))
    print(f"Found {len(json_files)} local JSON files")

    saved = 0
    skipped = 0

    for json_path in sorted(json_files):
        # Extract puzzle number from filename: times_cryptic_29481.json
        m = re.match(r'times_(?:cryptic|sunday-cryptic)_(\d+)\.json', json_path.name)
        if not m:
            continue

        pnum = m.group(1)
        if pnum not in needing_grids:
            skipped += 1
            continue

        try:
            data = json.loads(json_path.read_text())
            copy = data.get('data', {}).get('copy', {})
            settings = copy.get('settings', {})
            solution = settings.get('solution', '')
            gridsize = copy.get('gridsize', {})
            rows = int(gridsize.get('rows', 15))
            cols = int(gridsize.get('cols', 15))

            if solution and len(solution) == rows * cols:
                save_grid(conn, pnum, solution, rows, cols, f"local:{json_path.name}")
                needing_grids.discard(pnum)
                saved += 1
                print(f"  #{pnum}: OK ({rows}x{cols})")
            else:
                print(f"  #{pnum}: no solution in JSON (competition?)")
        except Exception as e:
            print(f"  #{pnum}: error: {e}")

    print(f"Phase 1 complete: {saved} grids saved, {skipped} already had grids")
    return saved


def phase2_http_backfill(conn, needing_grids, test=False):
    """Fetch grids from the Times archive via HTTP."""
    print(f"\n{'=' * 60}")
    print("PHASE 2: HTTP backfill from archive")
    print("=" * 60)

    # Filter to only puzzles in HTTP range
    http_eligible = set()
    for pnum in needing_grids:
        try:
            if int(pnum) >= HTTP_CUTOFF:
                http_eligible.add(pnum)
        except ValueError:
            continue

    print(f"Puzzles needing grids (total): {len(needing_grids)}")
    print(f"Puzzles eligible for HTTP (>= #{HTTP_CUTOFF}): {len(http_eligible)}")
    print(f"Puzzles below cutoff (need Selenium): {len(needing_grids) - len(http_eligible)}")

    if not http_eligible:
        print("Nothing to do in Phase 2")
        return 0

    # Step 1: Get all week links from sitemap
    print(f"\nFetching archive sitemap...")
    try:
        r = requests.get(SITEMAP_URL, timeout=15)
        if r.status_code != 200:
            print(f"Sitemap returned {r.status_code}")
            return 0
    except Exception as e:
        print(f"Sitemap error: {e}")
        return 0

    week_links = re.findall(r'href="(/html-puzzles-sitemap/\d{4}-\d{2}-\d)"', r.text)
    print(f"Found {len(week_links)} week pages in archive")

    # Step 2: Process each week page
    saved = 0
    failed = 0
    failed_nums = []
    processed = 0

    for i, week_path in enumerate(week_links):
        week_url = f"https://www.thetimes.com{week_path}"

        try:
            wr = requests.get(week_url, timeout=15)
            if wr.status_code != 200:
                continue
        except Exception:
            continue

        # Find all puzzle links on this week page
        puzzles_this_week = []
        for puzzle_type, pattern in PUZZLE_LINK_PATTERNS.items():
            for path, num in pattern.findall(wr.text):
                if num in http_eligible:
                    puzzles_this_week.append((puzzle_type, num, path))

        if not puzzles_this_week:
            continue

        print(f"\n  Week {week_path}: {len(puzzles_this_week)} puzzles to fetch")

        for puzzle_type, pnum, path in puzzles_this_week:
            if test and processed >= 10:
                print(f"\n  TEST MODE: stopping after {processed} puzzles")
                return saved

            puzzle_page_url = f"https://www.thetimes.com{path}"

            # Fetch puzzle page to get API URL
            try:
                pr = requests.get(puzzle_page_url, timeout=15)
                if pr.status_code != 200:
                    print(f"    #{pnum}: page returned {pr.status_code}")
                    failed += 1
                    failed_nums.append(int(pnum))
                    continue
            except Exception as e:
                print(f"    #{pnum}: page error: {e}")
                failed += 1
                failed_nums.append(int(pnum))
                continue

            api_match = API_URL_PATTERN.search(pr.text)
            if not api_match:
                print(f"    #{pnum}: no API URL in page")
                failed += 1
                failed_nums.append(int(pnum))
                time.sleep(0.5)
                continue

            api_base = api_match.group(1).rstrip('/')
            api_url = api_base + '/data.json'

            # Fetch API JSON
            try:
                ar = requests.get(api_url, timeout=15)
                if ar.status_code != 200:
                    print(f"    #{pnum}: API returned {ar.status_code}")
                    failed += 1
                    failed_nums.append(int(pnum))
                    time.sleep(0.5)
                    continue
                data = ar.json()
            except Exception as e:
                print(f"    #{pnum}: API error: {e}")
                failed += 1
                failed_nums.append(int(pnum))
                time.sleep(0.5)
                continue

            # Extract grid
            copy = data.get('data', {}).get('copy', {})
            settings = copy.get('settings', {})
            solution = settings.get('solution', '')
            gridsize = copy.get('gridsize', {})
            rows = int(gridsize.get('rows', 15))
            cols = int(gridsize.get('cols', 15))

            if solution and len(solution) == rows * cols:
                save_grid(conn, pnum, solution, rows, cols, api_url)
                http_eligible.discard(pnum)
                saved += 1
                print(f"    #{pnum}: OK ({rows}x{cols})")
            else:
                # Competition puzzle — no solution yet
                competition = data.get('data', {}).get('competitioncrossword', 0)
                if competition:
                    print(f"    #{pnum}: competition puzzle (no solution)")
                else:
                    print(f"    #{pnum}: no solution in API response")
                failed += 1
                failed_nums.append(int(pnum))

            processed += 1
            time.sleep(0.5)

        # Progress
        if (i + 1) % 50 == 0:
            print(f"\n  --- {i + 1}/{len(week_links)} weeks scanned, {saved} grids saved ---")

    print(f"\n{'=' * 60}")
    print("PHASE 2 COMPLETE")
    print(f"{'=' * 60}")
    print(f"Succeeded: {saved}")
    print(f"Failed: {failed}")

    if failed_nums:
        failed_nums.sort()
        print(f"\nFailed puzzle numbers ({len(failed_nums)}):")
        print(f"  {failed_nums[:50]}")
        if len(failed_nums) > 50:
            print(f"  ... and {len(failed_nums) - 50} more")

    # Report remaining
    still_needing = len(http_eligible)
    if still_needing:
        print(f"\nStill without grids in HTTP range: {still_needing}")

    return saved


def backfill(test=False):
    conn = sqlite3.connect(str(DB_PATH))

    # Ensure puzzle_grids table exists
    conn.execute("""
        CREATE TABLE IF NOT EXISTS puzzle_grids (
            source TEXT NOT NULL,
            puzzle_number TEXT NOT NULL,
            solution TEXT,
            grid_rows INTEGER NOT NULL DEFAULT 15,
            grid_cols INTEGER NOT NULL DEFAULT 15,
            api_folder TEXT,
            api_type TEXT,
            api_id TEXT,
            PRIMARY KEY (source, puzzle_number)
        )
    """)

    needing_grids = get_puzzles_needing_grids(conn)
    print(f"Times puzzles without grids: {len(needing_grids)}")

    p1 = phase1_local_jsons(conn, needing_grids)
    p2 = phase2_http_backfill(conn, needing_grids, test=test)

    print(f"\n{'=' * 60}")
    print("BACKFILL SUMMARY")
    print(f"{'=' * 60}")
    print(f"Phase 1 (local JSON): {p1} grids")
    print(f"Phase 2 (HTTP):       {p2} grids")
    print(f"Total new grids:      {p1 + p2}")

    # Final count
    c = conn.cursor()
    c.execute("SELECT COUNT(*) FROM puzzle_grids WHERE source='times'")
    total = c.fetchone()[0]
    print(f"Total Times grids in DB: {total}")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Times Grid Backfill")
    parser.add_argument("--test", action="store_true",
                        help="Phase 2: stop after 10 puzzles")
    args = parser.parse_args()

    backfill(test=args.test)
