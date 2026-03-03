#!/usr/bin/env python3
"""timesforthetimes_scraper.py — Scrape clue explanations from timesforthetimes.co.uk

Fetches Times Cryptic puzzle explanations and updates matching rows in clues_master.db.
Each page follows the pattern: https://timesforthetimes.co.uk/times-cryptic-XXXXX

Usage:
    python timesforthetimes_scraper.py                   # fetch gap from last explained to today
    python timesforthetimes_scraper.py --start 29404     # from specific puzzle number
    python timesforthetimes_scraper.py --start 29404 --end 29469  # specific range
    python timesforthetimes_scraper.py --daily           # just today's puzzle
    python timesforthetimes_scraper.py --dry-run --start 29469   # parse without writing

Writes to: data/clues_master.db (updates existing clue rows with explanation + definition)
Matches rows by: source='times' AND puzzle_number=? AND UPPER(answer)=?
"""

import argparse
import re
import sqlite3
import time
from datetime import date, timedelta
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MASTER_DB = BASE_DIR / 'data' / 'clues_master.db'
BASE_URL = 'https://timesforthetimes.co.uk/times-cryptic-{}'
RATE_LIMIT = 2.5

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

# Known blog abbreviations that are ALL-CAPS but not answers
NOT_ANSWERS = {
    'DD', 'CD', 'NHO', 'LOI', 'COD', 'FOI', 'POI', 'NI',
    'UK', 'US', 'TV', 'BBC', 'ITV', 'DP', 'RHS', 'LHS',
    'ACROSS', 'DOWN', 'EDIT', 'NB', 'PS',
}

ENUM_PAT = re.compile(r'\(([\d,\-\s]+)\)\s*$')


# ============================================================
# Helpers
# ============================================================

def is_answer(text: str) -> bool:
    """Return True if text looks like a crossword answer (ALL CAPS, 3+ letters)."""
    text = text.strip()
    if text in NOT_ANSWERS:
        return False
    if len(text) < 3:
        return False
    # Must be uppercase letters, spaces, hyphens, apostrophes only
    if not re.match(r'^[A-Z][A-Z\s\-\']{2,}$', text):
        return False
    # Must contain at least 3 actual letters
    if sum(1 for c in text if c.isalpha()) < 3:
        return False
    return True


# ============================================================
# Database helpers
# ============================================================

def get_last_explained_puzzle(conn: sqlite3.Connection) -> int:
    """Get the highest Times puzzle number that already has an explanation in clues_master."""
    row = conn.execute("""
        SELECT MAX(CAST(puzzle_number AS INTEGER))
        FROM clues
        WHERE source = 'times'
          AND explanation IS NOT NULL AND explanation != ''
    """).fetchone()
    return row[0] if (row and row[0]) else 27388


def puzzle_has_explanations(conn: sqlite3.Connection, puzzle_number: int) -> bool:
    """Check if this puzzle already has explanations in clues_master."""
    row = conn.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = 'times'
          AND puzzle_number = ?
          AND explanation IS NOT NULL AND explanation != ''
    """, (str(puzzle_number),)).fetchone()
    return row[0] > 0


def update_clues(conn: sqlite3.Connection, puzzle_number: int, clues: list) -> int:
    """Update existing clue rows in clues_master.db with explanation and definition.

    Matches by source='times' AND puzzle_number=? AND UPPER(answer)=?.
    Returns count of rows updated.
    """
    updated = 0
    puzzle_str = str(puzzle_number)

    for clue in clues:
        answer = clue['answer'].upper().replace(' ', '')

        # Try to match by puzzle_number and answer
        result = conn.execute("""
            UPDATE clues
            SET explanation = ?,
                definition = ?
            WHERE source = 'times'
              AND puzzle_number = ?
              AND UPPER(REPLACE(answer, ' ', '')) = ?
              AND (explanation IS NULL OR explanation = '')
        """, (
            clue['explanation'],
            clue.get('definition', ''),
            puzzle_str,
            answer,
        ))

        if result.rowcount > 0:
            updated += result.rowcount

    conn.commit()
    return updated


# ============================================================
# HTTP fetch
# ============================================================

def fetch_page(puzzle_number: int) -> str | None:
    url = BASE_URL.format(puzzle_number)
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
        elif resp.status_code == 404:
            return None
        else:
            print(f'HTTP {resp.status_code}')
            return None
    except Exception as e:
        print(f'fetch error: {e}')
        return None


# ============================================================
# HTML parser
# ============================================================

def parse_page(html: str, puzzle_number: int) -> list:
    """
    Parse a timesforthetimes page and extract clue entries.
    Returns list of dicts: clue_number, direction, clue_text, enumeration,
                           answer, definition, explanation.

    The site uses a clean table structure inside <section class="xwd-blog">:

        <table class="clues">
          <thead><tr><th>Across</th></tr></thead>
          <tbody>
            <tr>
              <td class="num">1</td>
              <td class="clue"><span>...clue text... <u>definition</u>...(7)</span></td>
            </tr>
            <tr>
              <td></td>
              <td class="ans"><strong>ANSWER</strong> – explanation text...</td>
            </tr>
          </tbody>
        </table>

    Definition text is underlined via style="text-decoration: underline".
    """
    soup = BeautifulSoup(html, 'html.parser')

    clues = []

    # Find all clue tables inside xwd-blog
    blog = soup.find('section', class_='xwd-blog')
    if not blog:
        # Fallback: look anywhere for table.clues
        tables = soup.find_all('table', class_='clues')
    else:
        tables = blog.find_all('table', class_='clues')

    if not tables:
        return []

    for table in tables:
        # Determine direction from <thead>
        thead = table.find('thead')
        if not thead:
            continue
        header_text = thead.get_text(strip=True).lower()
        if 'across' in header_text:
            direction = 'across'
            dir_suffix = 'a'
        elif 'down' in header_text:
            direction = 'down'
            dir_suffix = 'd'
        else:
            continue

        # Walk rows in pairs: num/clue row, then ans row
        rows = table.find('tbody').find_all('tr') if table.find('tbody') else table.find_all('tr')

        pending_num = None
        pending_clue_text = ''
        pending_enumeration = ''
        pending_definition = ''

        for row in rows:
            num_td = row.find('td', class_='num')
            clue_td = row.find('td', class_='clue')
            ans_td = row.find('td', class_='ans')

            if num_td and clue_td:
                # Clue row: extract number, clue text, definition
                pending_num = num_td.get_text(strip=True)

                # Full clue text (plain)
                pending_clue_text = clue_td.get_text(separator=' ', strip=True)

                # Enumeration: last (...) in the clue text
                m = ENUM_PAT.search(pending_clue_text)
                pending_enumeration = m.group(1).strip() if m else ''

                # Definition: underlined span(s)
                underlined = []
                for span in clue_td.find_all(True):
                    style = span.get('style', '')
                    if 'underline' in style or span.name == 'u':
                        underlined.append(span.get_text(strip=True))
                pending_definition = ' / '.join(underlined) if underlined else ''

            elif ans_td and pending_num:
                # Answer row: extract answer and explanation
                strong = ans_td.find(['strong', 'b'])
                answer = strong.get_text(strip=True) if strong else ''

                if not answer or not is_answer(answer):
                    # Try any bold ALL-CAPS text
                    for tag in ans_td.find_all(['strong', 'b']):
                        candidate = tag.get_text(strip=True)
                        if is_answer(candidate):
                            answer = candidate
                            break

                if not answer:
                    pending_num = None
                    continue

                # Explanation: everything after the <strong> tag
                full_text = ans_td.get_text(separator=' ', strip=True)
                expl = ''
                ans_pos = full_text.find(answer)
                if ans_pos >= 0:
                    expl = full_text[ans_pos + len(answer):].strip()
                    expl = re.sub(r'^[\s\-–—:\.]+', '', expl).strip()

                clues.append({
                    'clue_number': f'{pending_num}{dir_suffix}',
                    'direction': direction,
                    'clue_text': pending_clue_text,
                    'enumeration': pending_enumeration,
                    'answer': answer,
                    'definition': pending_definition,
                    'explanation': expl,
                })
                pending_num = None
                pending_clue_text = ''
                pending_enumeration = ''
                pending_definition = ''

    return clues


# ============================================================
# Puzzle number estimation
# ============================================================

def estimate_current_puzzle() -> int:
    """Estimate today's Times Cryptic puzzle number (Mon-Sat, no Sunday)."""
    ref_date = date(2026, 1, 26)
    ref_number = 29449
    today = date.today()
    days = 0
    current = ref_date
    while current < today:
        current += timedelta(days=1)
        if current.weekday() != 6:  # not Sunday
            days += 1
    return ref_number + days


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scrape timesforthetimes.co.uk explanations into clues_master.db'
    )
    parser.add_argument('--start', type=int, help='First puzzle number to fetch')
    parser.add_argument('--end', type=int, help='Last puzzle number (inclusive)')
    parser.add_argument('--daily', action='store_true',
                        help='Fetch only today\'s estimated puzzle')
    parser.add_argument('--dry-run', action='store_true',
                        help='Fetch and parse but do not write to DB')
    args = parser.parse_args()

    conn = sqlite3.connect(MASTER_DB)

    if args.daily:
        start = end = estimate_current_puzzle()
    else:
        last_explained = get_last_explained_puzzle(conn)
        start = args.start if args.start else (last_explained + 1)
        end = args.end if args.end else estimate_current_puzzle()

    print('timesforthetimes scraper')
    print(f'DB:    {MASTER_DB}')
    print(f'Range: puzzles {start} to {end}')
    if args.dry_run:
        print('DRY RUN — no DB writes')
    print()

    total_updated = 0
    fetched = 0
    failed = 0

    for puzzle_num in range(start, end + 1):
        if not args.dry_run and puzzle_has_explanations(conn, puzzle_num):
            print(f'  {puzzle_num}: already has explanations — skipping')
            continue

        print(f'  {puzzle_num}: fetching...', end=' ', flush=True)
        html = fetch_page(puzzle_num)

        if html is None:
            print('not found')
            failed += 1
            time.sleep(1)
            continue

        clues = parse_page(html, puzzle_num)

        if not clues:
            print('parsed 0 clues — page may have different format')
            failed += 1
            time.sleep(RATE_LIMIT)
            continue

        print(f'parsed {len(clues)} clues', end='')

        if not args.dry_run:
            n = update_clues(conn, puzzle_num, clues)
            total_updated += n
            print(f', updated {n} rows')
        else:
            print()
            for c in clues[:4]:
                print(
                    f'    [{c["clue_number"]}] {c["answer"]!r:25} '
                    f'clue: {c["clue_text"][:35]!r:37} '
                    f'expl: {c["explanation"][:50]!r}'
                )

        fetched += 1
        time.sleep(RATE_LIMIT)

    conn.close()
    print()
    print(f'Done. Fetched: {fetched}, Updated: {total_updated}, Failed/not found: {failed}')


if __name__ == '__main__':
    main()
