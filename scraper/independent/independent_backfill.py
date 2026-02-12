#!/usr/bin/env python3
"""Independent Crossword Automated Backfill

Automatically cycles through all archive months and scrapes puzzles.
No browser needed - pure HTTP requests.

Usage:
    python independent_auto_backfill.py
    python independent_auto_backfill.py --start-year=2023 --start-month=6
"""

import re
import os
import sys
import sqlite3
import requests
import time
from datetime import datetime, date
from pathlib import Path
from bs4 import BeautifulSoup
from dotenv import load_dotenv

load_dotenv()

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")

BASE_URL = "https://www.independentcrossword.co.uk"

# Months in lowercase for URL
MONTHS = ['january', 'february', 'march', 'april', 'may', 'june',
          'july', 'august', 'september', 'october', 'november', 'december']

REQUEST_DELAY = 0.5  # seconds between requests


def fetch_page(url):
    """Fetch page with requests."""
    try:
        response = requests.get(url, timeout=30)
        if response.status_code == 200:
            return response.text
        return None
    except:
        return None


def parse_puzzle_page(html, puzzle_date_str):
    """Parse puzzle page and extract clues with answers."""
    soup = BeautifulSoup(html, 'html.parser')
    clues = []

    data_divs = soup.find_all('div', class_='data')

    for div in data_divs:
        try:
            clue_link = div.find('a', class_='main-btn')
            if not clue_link:
                continue

            clue_text = clue_link.get_text(strip=True)
            clue_text = re.sub(r'\s*\d+\s*$', '', clue_text)

            letter_boxes = div.find_all('div', class_='letter_box')
            answer = ''
            for box in letter_boxes:
                box_text = box.get_text(strip=True)
                letter = re.sub(r'^\d+', '', box_text)
                answer += letter

            if clue_text and answer:
                clues.append({
                    'clue_text': clue_text,
                    'answer': answer.upper(),
                    'puzzle_date': puzzle_date_str
                })
        except:
            continue

    return clues


def puzzle_already_fetched(puzzle_date_str):
    """Check if puzzle is already in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS independent_clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_type TEXT DEFAULT 'cryptic',
            puzzle_number TEXT,
            puzzle_date TEXT,
            setter TEXT,
            clue_number TEXT,
            direction TEXT,
            clue_text TEXT,
            enumeration TEXT,
            answer TEXT,
            explanation TEXT,
            published INTEGER DEFAULT 0,
            fetched_at TEXT
        )
    """)

    cursor.execute("""
        SELECT COUNT(*) FROM independent_clues 
        WHERE puzzle_date = ?
    """, (puzzle_date_str,))

    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def save_to_database(clues, puzzle_date_str):
    """Save clues to database."""
    if not clues:
        return 0

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    fetched_at = datetime.now().isoformat()
    clue_count = 0

    for i, clue in enumerate(clues, 1):
        cursor.execute("""
            INSERT INTO independent_clues 
            (puzzle_type, puzzle_date, clue_number, clue_text, answer, fetched_at)
            VALUES (?, ?, ?, ?, ?, ?)
        """, (
            'cryptic',
            clue['puzzle_date'],
            str(i),
            clue['clue_text'],
            clue['answer'],
            fetched_at
        ))
        clue_count += 1

    conn.commit()
    conn.close()
    return clue_count


def extract_puzzle_links_from_archive(html):
    """Extract all puzzle URLs from archive page."""
    soup = BeautifulSoup(html, 'html.parser')
    puzzles = []

    links = soup.find_all('a', href=True)

    for link in links:
        try:
            href = link.get('href', '')
            text = link.get_text(strip=True)

            # Match: "The Independent's Cryptic Crossword 28 January 2025 Answers"
            match = re.search(r'(\d{1,2})\s+(\w+)\s+(\d{4})\s*Answers?', text,
                              re.IGNORECASE)
            if match:
                day = int(match.group(1))
                month_str = match.group(2)
                year = int(match.group(3))

                try:
                    puzzle_date = datetime.strptime(f"{day} {month_str} {year}",
                                                    "%d %B %Y").date()
                    puzzle_date_str = puzzle_date.strftime('%Y-%m-%d')

                    # Make sure href is absolute
                    if href.startswith('/'):
                        url = BASE_URL + href
                    elif href.startswith('http'):
                        url = href
                    else:
                        month_lower = month_str.lower()
                        url = f"{BASE_URL}/independent-cryptic-crossword-{day}-{month_lower}-{year}-answers"

                    puzzles.append({
                        'url': url,
                        'date_str': puzzle_date_str,
                        'display': f"{day} {month_str} {year}"
                    })
                except ValueError:
                    continue
        except:
            continue

    return puzzles


def scrape_month(year, month_name):
    """Scrape all puzzles for a given month."""
    archive_url = f"{BASE_URL}/archive/independent-cryptic-crossword/{year}/{month_name}"

    html = fetch_page(archive_url)
    if not html:
        return 0, 0, True  # empty month

    puzzles = extract_puzzle_links_from_archive(html)
    if not puzzles:
        return 0, 0, True  # no puzzles found

    saved = 0
    skipped = 0

    for puzzle in puzzles:
        date_str = puzzle['date_str']
        display = puzzle['display']
        url = puzzle['url']

        if puzzle_already_fetched(date_str):
            skipped += 1
            continue

        # Fetch puzzle page
        puzzle_html = fetch_page(url)
        if not puzzle_html:
            continue

        # Parse and save
        clues = parse_puzzle_page(puzzle_html, date_str)
        if clues:
            count = save_to_database(clues, date_str)
            print(f"      {display}: {count} clues")
            saved += count

        time.sleep(REQUEST_DELAY)

    return saved, skipped, False


def main():
    print("=" * 60)
    print("INDEPENDENT CROSSWORD AUTO BACKFILL")
    print(f"Date: {date.today().strftime('%A, %d %B %Y')}")
    print("=" * 60)
    print(f"Database: {DB_PATH}")

    # Parse arguments - default start from April 2025
    start_year = 2025
    start_month = 4  # April
    end_year = 2015  # Go back to 2015

    for arg in sys.argv[1:]:
        if arg.startswith('--start-year='):
            start_year = int(arg.split('=')[1])
        elif arg.startswith('--start-month='):
            start_month = int(arg.split('=')[1])
        elif arg.startswith('--end-year='):
            end_year = int(arg.split('=')[1])

    print(f"\nStarting from: {MONTHS[start_month - 1].title()} {start_year}")
    print(f"Going back to: {end_year}")
    print()

    total_saved = 0
    total_skipped = 0
    empty_months = 0

    # Iterate through years and months backwards
    year = start_year
    month = start_month

    while year >= end_year:
        month_name = MONTHS[month - 1]
        print(f"  {month_name.title()} {year}:", end=" ", flush=True)

        saved, skipped, is_empty = scrape_month(year, month_name)

        if is_empty:
            print("(empty/not found)")
            empty_months += 1
        else:
            print(f"saved {saved} clues, skipped {skipped} puzzles")
            total_saved += saved
            total_skipped += skipped

        # Move to previous month
        month -= 1
        if month < 1:
            month = 12
            year -= 1

        time.sleep(REQUEST_DELAY)

    print("\n" + "=" * 60)
    print("BACKFILL COMPLETE")
    print("=" * 60)
    print(f"Total clues saved: {total_saved}")
    print(f"Puzzles skipped (already in DB): {total_skipped}")
    print(f"Empty months: {empty_months}")


if __name__ == "__main__":
    main()