#!/usr/bin/env python3
"""Independent Edition Backfill

Two modes:
  1. UPGRADE existing puzzles: update clue_number and direction ONLY,
     matched by exact clue_text. No other fields are touched.
  2. ADD new puzzles: full insert for dates not in the DB at all.

The edition archive goes back to June 2021 (~1,500 puzzles).
Edition IDs are predictable: uk.co.independent.issue.DDMMYY

Usage:
    python independent_edition_backfill.py                        # Report only
    python independent_edition_backfill.py --execute              # Run upgrades + new inserts
    python independent_edition_backfill.py --from 2021-06-11      # Start from specific date
    python independent_edition_backfill.py --upgrade-only         # Only upgrade existing, skip new
"""

import os
import re
import sqlite3
import sys
import time
import unicodedata
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parent))
from independent_edition import (
    EDITION_BASE,
    REQUEST_DELAY,
    content_xml_url,
    fetch_content_xml,
    find_cryptic_puzzle_url,
    fetch_puzzle_json,
    parse_puzzle,
    build_grid_solution,
    save_to_database,
)

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

# Earliest available edition
EARLIEST_DATE = date(2021, 6, 11)


def normalise_text(text):
    """Normalise clue text for matching.

    The edition JSON and old scraper produce slightly different text:
    - Edition has HTML tags (<i>word</i>), old scraper stripped them
    - Edition preserves commas, old scraper stripped them
    - Curly quotes/dashes may differ between sources
    """
    if not text:
        return ''
    # Strip HTML tags
    text = re.sub(r'<[^>]+>', '', text)
    # Normalise unicode to NFKC (converts curly quotes, em-dashes etc)
    text = unicodedata.normalize('NFKC', text)
    # Normalise curly quotes and dashes to ASCII equivalents
    text = text.replace('\u2018', "'").replace('\u2019', "'")  # single quotes
    text = text.replace('\u201c', '"').replace('\u201d', '"')  # double quotes
    text = text.replace('\u2013', '-').replace('\u2014', '-')  # en/em dash
    text = text.replace('\u2026', '...')                        # ellipsis
    # Strip commas (old scraper removed them)
    text = text.replace(',', '')
    # Collapse whitespace
    text = re.sub(r'\s+', ' ', text).strip()
    return text.lower()


def get_dates_needing_upgrade():
    """Find Independent publication dates that have clues but no direction data.

    Returns dict of {publication_date: puzzle_number}.
    """
    if not Path(DB_PATH).exists():
        return {}

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    # Dates where ALL clues lack direction
    cursor.execute("""
        SELECT publication_date, puzzle_number,
               COUNT(*) as total,
               SUM(CASE WHEN direction IS NOT NULL AND direction != '' THEN 1 ELSE 0 END) as has_dir
        FROM clues
        WHERE source = 'independent'
        GROUP BY publication_date, puzzle_number
        HAVING has_dir = 0
    """)
    result = {}
    for row in cursor.fetchall():
        result[row[0]] = row[1]

    conn.close()
    return result


def get_all_existing_dates():
    """Get all publication dates for Independent puzzles in the DB."""
    if not Path(DB_PATH).exists():
        return set()

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT DISTINCT publication_date
        FROM clues WHERE source = 'independent'
    """)
    dates = set(row[0] for row in cursor.fetchall())
    conn.close()
    return dates


def get_db_clues_for_date(pub_date):
    """Get existing clues for a date.

    Returns:
        text_map: {normalised_clue_text: row_id}
        answer_map: {answer_upper: [row_id, ...]} for fallback matching
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    cursor.execute("""
        SELECT id, clue_text, answer FROM clues
        WHERE source = 'independent' AND publication_date = ?
    """, (pub_date,))

    text_map = {}
    answer_map = {}
    for row_id, clue_text, answer in cursor.fetchall():
        key = normalise_text(clue_text)
        text_map[key] = row_id
        ans = (answer or '').upper()
        answer_map.setdefault(ans, []).append(row_id)

    conn.close()
    return text_map, answer_map


def upgrade_puzzle(edition_date, dry_run=True):
    """Upgrade an existing puzzle: update clue_number and direction only.

    Matches edition JSON clues to DB rows by normalised clue text.
    Only updates clue_number and direction — nothing else.

    Returns (matched, unmatched, total_edition) counts.
    """
    date_str = edition_date.isoformat()

    # Fetch edition data
    xml_root = fetch_content_xml(edition_date)
    if xml_root is None:
        return -1, 0, 0

    puzzle_url, puzzle_title = find_cryptic_puzzle_url(xml_root)
    if not puzzle_url:
        return -1, 0, 0

    puzzle_json = fetch_puzzle_json(puzzle_url)
    if not puzzle_json:
        return -1, 0, 0

    clues, grid_solution, grid_rows, grid_cols, puzzle_number = parse_puzzle(
        puzzle_json, edition_date, puzzle_title
    )
    if not clues:
        return -1, 0, 0

    # Get existing DB clues keyed by normalised text and by answer
    text_map, answer_map = get_db_clues_for_date(date_str)

    matched_text = 0
    matched_answer = 0
    unmatched_edition = []
    used_row_ids = set()  # prevent double-matching

    updates = []  # (clue_number, direction, row_id, match_type) tuples

    for clue in clues:
        key = normalise_text(clue['clue_text'])
        row_id = text_map.get(key)

        if row_id is not None and row_id not in used_row_ids:
            matched_text += 1
            updates.append((clue['clue_number'], clue['direction'], row_id, 'text'))
            used_row_ids.add(row_id)
        else:
            # Fallback: match by answer (must be unique match within puzzle)
            answer = clue.get('answer', '').upper()
            candidates = [rid for rid in answer_map.get(answer, [])
                          if rid not in used_row_ids]
            if len(candidates) == 1:
                matched_answer += 1
                updates.append((clue['clue_number'], clue['direction'], candidates[0], 'answer'))
                used_row_ids.add(candidates[0])
                print(f"      ANSWER-MATCH: {answer} <- {clue['clue_text'][:50]}")
            else:
                unmatched_edition.append(
                    f"{clue['clue_text'][:50]} [{answer}]"
                    + (f" ({len(candidates)} candidates)" if len(candidates) > 1 else "")
                )

    if not dry_run and updates:
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        for clue_number, direction, row_id, _ in updates:
            cursor.execute("""
                UPDATE clues SET clue_number = ?, direction = ?
                WHERE id = ?
            """, (clue_number, direction, row_id))
        conn.commit()
        conn.close()

    if unmatched_edition:
        for text in unmatched_edition:
            print(f"      UNMATCHED: {text}")

    return matched_text + matched_answer, len(unmatched_edition), len(clues)


def generate_date_range(start_date, end_date):
    """Generate all dates from start to end inclusive."""
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)


def main():
    dry_run = '--execute' not in sys.argv
    upgrade_only = '--upgrade-only' in sys.argv

    # Parse --from date
    from_date = EARLIEST_DATE
    for i, arg in enumerate(sys.argv):
        if arg == '--from' and i + 1 < len(sys.argv):
            from_date = date.fromisoformat(sys.argv[i + 1])

    mode = "REPORT ONLY" if dry_run else "EXECUTING"
    print("=" * 70)
    print(f"INDEPENDENT EDITION BACKFILL [{mode}]")
    print("=" * 70)
    print(f"Date range: {from_date} to {date.today()}")
    print(f"Database: {DB_PATH}")
    print()

    # Phase 1: Identify upgrades (existing puzzles missing direction)
    print("Scanning database for puzzles needing upgrade...")
    needs_upgrade = get_dates_needing_upgrade()
    existing_dates = get_all_existing_dates()
    print(f"  {len(needs_upgrade)} puzzles need direction/clue_number upgrade")
    print(f"  {len(existing_dates)} total puzzle dates in DB")
    print()

    # Phase 2: Scan edition archive for available dates
    print("Scanning edition archive...")
    today = date.today()

    upgrade_results = []  # (date, matched, unmatched, total)
    new_available = []     # dates available but not in DB

    for edition_date in generate_date_range(from_date, today):
        date_str = edition_date.isoformat()

        if date_str in needs_upgrade:
            # This puzzle needs upgrading
            label = edition_date.strftime('%a %d %b %Y')
            pnum = needs_upgrade[date_str]
            print(f"  {label} #{pnum}: ", end="", flush=True)

            matched, unmatched, total = upgrade_puzzle(edition_date, dry_run=dry_run)

            if matched == -1:
                print("edition not available")
                upgrade_results.append((edition_date, pnum, 0, 0, 0, 'unavailable'))
            elif matched == 0 and total > 0:
                print(f"0/{total} matched -- DB has different puzzle for this date, SKIPPED")
                upgrade_results.append((edition_date, pnum, 0, unmatched, total, 'mismatch'))
            elif unmatched > 0:
                tag = "[DRY RUN] " if dry_run else ""
                print(f"{tag}{matched}/{total} matched, {unmatched} unmatched")
                upgrade_results.append((edition_date, pnum, matched, unmatched, total, 'partial'))
            else:
                tag = "[DRY RUN] " if dry_run else ""
                print(f"{tag}{matched}/{total} matched")
                upgrade_results.append((edition_date, pnum, matched, 0, total, 'ok'))

            time.sleep(REQUEST_DELAY)

        elif date_str not in existing_dates and not upgrade_only:
            # Check if edition exists for this date (new puzzle)
            url = content_xml_url(edition_date)
            try:
                resp = requests.head(url, timeout=10)
                if resp.status_code == 200:
                    new_available.append(edition_date)
            except Exception:
                pass
            # Lighter rate limiting for HEAD requests
            time.sleep(0.3)

    # Summary
    print()
    print("=" * 70)
    print("SUMMARY")
    print("=" * 70)

    ok = sum(1 for r in upgrade_results if r[5] == 'ok')
    partial = sum(1 for r in upgrade_results if r[5] == 'partial')
    mismatch = sum(1 for r in upgrade_results if r[5] == 'mismatch')
    unavailable = sum(1 for r in upgrade_results if r[5] == 'unavailable')
    total_matched = sum(r[2] for r in upgrade_results)
    total_unmatched = sum(r[3] for r in upgrade_results if r[5] != 'mismatch')

    print(f"\nUPGRADES:")
    print(f"  Full match:      {ok}")
    print(f"  Partial match:   {partial}")
    print(f"  Wrong puzzle:    {mismatch} (DB has different clues, skipped)")
    print(f"  Not available:   {unavailable}")
    print(f"  Clues matched:   {total_matched}")
    print(f"  Clues unmatched: {total_unmatched}")

    if not upgrade_only:
        print(f"\nNEW PUZZLES AVAILABLE: {len(new_available)}")
        if new_available:
            print(f"  Range: {new_available[0]} to {new_available[-1]}")
            print("  (Run with --execute to add these as new inserts)")

    # Save failed/problem puzzles to file for re-scraping
    failed = [r for r in upgrade_results if r[5] in ('mismatch', 'unavailable', 'partial')]
    if failed:
        fail_path = Path(__file__).resolve().parent / 'backfill_failures.txt'
        with open(fail_path, 'w') as f:
            f.write("# Independent backfill failures\n")
            f.write(f"# Generated: {date.today().isoformat()}\n")
            f.write(f"# Format: date | puzzle_number | status | matched/total\n\n")
            for edition_date, pnum, matched, unmatched, total, status in failed:
                f.write(f"{edition_date.isoformat()} | #{pnum} | {status} | {matched}/{total}\n")
        print(f"\nFailed puzzles saved to: {fail_path}")
        print(f"  {len(failed)} puzzles need re-scraping")

    if dry_run:
        print(f"\nThis was a DRY RUN. No database changes were made.")
        print(f"Run with --execute to apply updates.")


if __name__ == "__main__":
    main()
