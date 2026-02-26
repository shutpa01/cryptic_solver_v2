#!/usr/bin/env python3
"""Independent Crossword Scraper

Scrapes clues AND answers from independentcrossword.co.uk
Saves directly to the clues table in clues_master.db.

Usage:
    python independent_all.py              # Today's puzzle
    python independent_all.py --backfill   # All puzzles from archive
    python independent_all.py --days=30    # Last 30 days
"""

import re
import os
import sys
import sqlite3
import requests
import time
from datetime import datetime, date, timedelta
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

BASE_URL = "https://independentcrossword.co.uk"
ARCHIVE_URL = f"{BASE_URL}/archive/independent-cryptic-crossword"

# Rate limiting
REQUEST_DELAY = 1  # seconds between requests


def get_puzzle_url_for_date(puzzle_date):
    """Construct puzzle URL from date."""
    # Format: independent-cryptic-crossword-28-january-2026-answers
    day = puzzle_date.day
    month = puzzle_date.strftime('%B').lower()
    year = puzzle_date.year
    return f"{BASE_URL}/independent-cryptic-crossword-{day}-{month}-{year}-answers"


def fetch_page(url):
    """Fetch page with error handling."""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.text
        elif response.status_code == 404:
            return None
        else:
            print(f"  HTTP {response.status_code}: {url}")
            return None
    except Exception as e:
        print(f"  Error fetching {url}: {e}")
        return None


def parse_puzzle_page(html_text, puzzle_date):
    """Parse puzzle page and extract clues with answers."""
    soup = BeautifulSoup(html_text, 'html.parser')

    clues = []

    # Find all clue entries
    data_divs = soup.find_all('div', class_='data')

    for div in data_divs:
        try:
            # Get clue text
            clue_link = div.find('a', class_='main-btn')
            if not clue_link:
                continue

            clue_text = clue_link.get_text(strip=True)
            # Remove any trailing letter count span text
            clue_text = re.sub(r'\s*\d+\s*$', '', clue_text)

            # Get answer from letter boxes
            letter_boxes = div.find_all('div', class_='letter_box')
            answer = ''
            for box in letter_boxes:
                # The letter is the text content minus the span number
                box_text = box.get_text(strip=True)
                # Remove leading digits (the position number)
                letter = re.sub(r'^\d+', '', box_text)
                answer += letter

            if clue_text and answer:
                clues.append({
                    'clue_text': clue_text,
                    'answer': answer.upper(),
                    'puzzle_date': puzzle_date.strftime('%Y-%m-%d')
                })
        except Exception as e:
            continue

    return clues


def get_archive_dates():
    """Scrape archive page to get all available puzzle dates."""
    print(f"Fetching archive: {ARCHIVE_URL}")
    html_text = fetch_page(ARCHIVE_URL)
    if not html_text:
        print("Failed to fetch archive page")
        return []

    soup = BeautifulSoup(html_text, 'html.parser')
    dates = []

    # Find all archive links
    # Pattern: "The Independent's Cryptic Crossword 28 January 2026 Answers"
    for link in soup.find_all('a', href=True):
        text = link.get_text(strip=True)
        match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})\s+Answers', text)
        if match:
            day = int(match.group(1))
            month_str = match.group(2)
            year = int(match.group(3))

            try:
                puzzle_date = datetime.strptime(f"{day} {month_str} {year}",
                                                "%d %B %Y").date()
                if puzzle_date not in dates:
                    dates.append(puzzle_date)
            except ValueError:
                continue

    # Sort oldest to newest
    dates.sort()
    print(f"Found {len(dates)} puzzles in archive")
    return dates


def puzzle_already_fetched(puzzle_date):
    """Check if puzzle is already in the clues table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = 'independent' AND publication_date = ?
    """, (puzzle_date.strftime('%Y-%m-%d'),))

    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def save_to_database(clues, puzzle_date):
    """Save clues directly to clues table."""
    if not clues:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    clue_count = 0

    for i, clue in enumerate(clues, 1):
        cursor.execute("""
            INSERT OR IGNORE INTO clues
            (source, publication_date, clue_number, clue_text, answer)
            VALUES (?, ?, ?, ?, ?)
        """, (
            'independent',
            clue['puzzle_date'],
            str(i),
            clue['clue_text'],
            clue['answer'],
        ))
        clue_count += 1

    conn.commit()
    conn.close()
    return clue_count


def scrape_puzzle(puzzle_date, force=False):
    """Scrape a single puzzle by date."""
    date_str = puzzle_date.strftime('%d %B %Y')

    if not force and puzzle_already_fetched(puzzle_date):
        print(f"  {date_str}: already in database - skipping")
        return 0

    url = get_puzzle_url_for_date(puzzle_date)
    html_text = fetch_page(url)

    if not html_text:
        print(f"  {date_str}: not found")
        return 0

    clues = parse_puzzle_page(html_text, puzzle_date)

    if not clues:
        print(f"  {date_str}: no clues found")
        return 0

    count = save_to_database(clues, puzzle_date)
    print(f"  {date_str}: saved {count} clues")
    return count


def scrape_today():
    """Scrape today's puzzle."""
    today = date.today()
    print(f"Scraping today's puzzle: {today.strftime('%d %B %Y')}")
    return scrape_puzzle(today)


def scrape_recent_days(days=30):
    """Scrape recent days."""
    print(f"Scraping last {days} days...")
    total = 0

    for i in range(days):
        puzzle_date = date.today() - timedelta(days=i)
        count = scrape_puzzle(puzzle_date)
        total += count
        if count > 0:
            time.sleep(REQUEST_DELAY)

    return total


def scrape_backfill():
    """Scrape all puzzles from archive."""
    print("Starting backfill from archive...")

    dates = get_archive_dates()
    if not dates:
        print("No dates found in archive")
        return 0

    total_saved = 0
    total_skipped = 0

    for i, puzzle_date in enumerate(dates):
        print(f"[{i + 1}/{len(dates)}]", end="")

        if puzzle_already_fetched(puzzle_date):
            print(f"  {puzzle_date}: already in database - skipping")
            total_skipped += 1
            continue

        count = scrape_puzzle(puzzle_date)
        if count > 0:
            total_saved += count
            time.sleep(REQUEST_DELAY)

    print(
        f"\nBackfill complete: {total_saved} clues saved, {total_skipped} puzzles skipped")
    return total_saved


def main():
    print("=" * 60)
    print("INDEPENDENT CROSSWORD SCRAPER")
    print("(from independentcrossword.co.uk - includes answers!)")
    print(f"Date: {date.today().strftime('%A, %d %B %Y')}")
    print("=" * 60)
    print(f"Database: {DB_PATH}")

    # Parse arguments
    if '--backfill' in sys.argv:
        scrape_backfill()
    elif any(arg.startswith('--days=') for arg in sys.argv):
        for arg in sys.argv:
            if arg.startswith('--days='):
                days = int(arg.split('=')[1])
                scrape_recent_days(days)
                break
    else:
        scrape_today()

    print("\nDone!")


if __name__ == "__main__":
    main()
