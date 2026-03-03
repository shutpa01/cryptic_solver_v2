#!/usr/bin/env python3
"""bigdave_scraper.py — Scrape clue explanations from bigdave44.com

Fetches Telegraph cryptic puzzle explanations (hints) and updates matching
rows in clues_master.db.

Covers two Big Dave categories:
  - DT Cryptic: bigdave44.com/category/crosswords/dt-cryptic-crosswords/
  - Toughie:    bigdave44.com/category/crosswords/toughie-crosswords/

Post format:
  - Answers in <span class="spoiler"><span class="hidden-content">ANSWER</span></span>
  - Definitions underlined with <u> tags
  - Clue numbers like "8a", "1d"
  - Explanations as prose text after the spoiler span

Usage:
    python bigdave_scraper.py                          # backfill DT + Toughie
    python bigdave_scraper.py --category dt             # DT cryptic only
    python bigdave_scraper.py --category toughie        # Toughie only
    python bigdave_scraper.py --limit 50               # stop after 50 puzzles
    python bigdave_scraper.py --start-page 100         # start from category page 100
    python bigdave_scraper.py --daily                   # today's puzzles only
    python bigdave_scraper.py --dry-run                 # parse without DB writes

Writes to: data/clues_master.db (updates existing clue rows with explanation)
Matches rows by: source='telegraph' + puzzle_number + UPPER(answer)
"""

import argparse
import re
import sqlite3
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MASTER_DB = BASE_DIR / 'data' / 'clues_master.db'
RATE_LIMIT = 2.5

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

CATEGORIES = {
    'dt': {
        'url': 'https://bigdave44.com/category/crosswords/dt-cryptic-crosswords/',
        'num_re': re.compile(r'DT[\s\-]*(\d{4,6})', re.IGNORECASE),
        'skip_words': ['toughie', 'hint', 'full review', 'saturday', 'sunday'],
        'label': 'DT',
    },
    'toughie': {
        'url': 'https://bigdave44.com/category/crosswords/toughie-crosswords/',
        'num_re': re.compile(r'Toughie[\s\-]*(\d{3,6})', re.IGNORECASE),
        'skip_words': ['hint', 'full review', 'sunday toughie'],
        'label': 'Toughie',
    },
}


# ============================================================
# HTTP helpers
# ============================================================

def fetch(url: str, retries: int = 3) -> requests.Response | None:
    for attempt in range(retries):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=30)
            if resp.status_code == 200:
                return resp
            elif resp.status_code == 404:
                return None
            else:
                print(f'  HTTP {resp.status_code} on attempt {attempt + 1}')
        except Exception as e:
            print(f'  Fetch error (attempt {attempt + 1}): {e}')
        time.sleep(2)
    return None


# ============================================================
# Category page crawler
# ============================================================

def get_post_links(cat: dict, page: int = 1) -> list[dict]:
    """Fetch category page and return list of {title, url, puzzle_number}."""
    base_url = cat['url']
    url = base_url if page == 1 else f'{base_url}page/{page}/'
    resp = fetch(url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    posts = []
    num_re = cat['num_re']
    skip_words = cat['skip_words']

    seen_numbers = set()
    for a in soup.find_all('a', rel='bookmark'):
        title = a.get_text(strip=True)
        href = a.get('href', '')

        # Skip unwanted posts (hints-only, full reviews, etc.)
        title_lower = title.lower()
        if any(skip in title_lower for skip in skip_words):
            continue

        # Extract puzzle number
        m = num_re.search(title)
        if not m:
            m = num_re.search(href)
        if not m:
            continue

        puzzle_number = m.group(1)

        # Deduplicate (same puzzle can appear multiple times on one page)
        if puzzle_number in seen_numbers:
            continue
        seen_numbers.add(puzzle_number)

        posts.append({
            'title': title,
            'url': href,
            'puzzle_number': puzzle_number,
        })

    return posts


def get_max_page(cat: dict) -> int:
    """Get the maximum page number from category pagination."""
    resp = fetch(cat['url'])
    if not resp:
        return 1
    soup = BeautifulSoup(resp.text, 'html.parser')

    max_page = 1
    for a in soup.find_all('a'):
        href = a.get('href', '')
        m = re.search(r'/page/(\d+)/', href)
        if m:
            max_page = max(max_page, int(m.group(1)))
    return max_page


# ============================================================
# HTML parser
# ============================================================

def parse_post(html: str) -> list[dict]:
    """Parse a Big Dave blog post and extract clue entries.

    Returns list of dicts: {clue_number, direction, answer, explanation}
    """
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', class_='entry-content')
    if not content:
        return []

    clues = []
    direction = None

    # Walk through content elements
    for element in content.children:
        if not isinstance(element, Tag):
            continue

        text = element.get_text(strip=True)
        if not text:
            continue

        # Direction headers
        text_lower = text.lower()
        if text_lower in ('across', 'down'):
            direction = text_lower
            continue
        # Check for bold direction header
        bold = element.find(['strong', 'b'])
        if bold:
            bold_text = bold.get_text(strip=True).lower()
            if bold_text in ('across', 'down'):
                direction = bold_text
                continue

        if not direction:
            continue

        # Look for clue number at start of element text
        # No word boundary — direction letter often abuts the next word (e.g. "8aVery")
        clue_match = re.match(r'^(\d+)\s*([adAD])(?=\s|[A-Z]|$)', text)
        if not clue_match:
            continue

        clue_num = clue_match.group(1)
        dir_letter = clue_match.group(2).lower()
        row_dir = 'across' if dir_letter == 'a' else 'down'

        # Extract answer from spoiler span
        answer = ''
        spoiler = element.find('span', class_='spoiler')
        if spoiler:
            hidden = spoiler.find('span', class_='hidden-content')
            if hidden:
                answer = hidden.get_text(strip=True)
            else:
                answer = spoiler.get_text(strip=True)

        if not answer:
            # Fallback: look for bold all-caps text
            for tag in element.find_all(['strong', 'b']):
                candidate = tag.get_text(strip=True)
                if candidate == candidate.upper() and len(candidate) >= 3 and re.match(r'^[A-Z\s\-\']+$', candidate):
                    answer = candidate
                    break

        if not answer or len(answer) < 2:
            continue

        # Extract explanation: all text after the answer
        full_text = element.get_text(separator=' ', strip=True)
        explanation = ''

        # Find the answer in the full text and take everything after
        ans_pos = full_text.upper().find(answer.upper())
        if ans_pos >= 0:
            explanation = full_text[ans_pos + len(answer):].strip()
            # Clean leading separators (colon, dash, etc.)
            explanation = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', explanation).strip()

        clues.append({
            'clue_number': clue_num,
            'direction': row_dir,
            'answer': answer.upper(),
            'explanation': explanation,
        })

    return clues


# ============================================================
# Database helpers
# ============================================================

def puzzle_has_explanations(conn: sqlite3.Connection, puzzle_number: str) -> bool:
    row = conn.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = 'telegraph'
          AND puzzle_number = ?
          AND explanation IS NOT NULL AND explanation != ''
    """, (puzzle_number,)).fetchone()
    return row[0] > 0


def puzzle_exists(conn: sqlite3.Connection, puzzle_number: str) -> bool:
    row = conn.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = 'telegraph' AND puzzle_number = ?
    """, (puzzle_number,)).fetchone()
    return row[0] > 0


def update_clues(conn: sqlite3.Connection, puzzle_number: str,
                 clues: list[dict]) -> int:
    updated = 0
    for clue in clues:
        answer = clue['answer'].upper().replace(' ', '').replace('-', '')
        explanation = clue['explanation']
        if not explanation:
            continue

        result = conn.execute("""
            UPDATE clues
            SET explanation = ?
            WHERE source = 'telegraph'
              AND puzzle_number = ?
              AND UPPER(REPLACE(REPLACE(answer, ' ', ''), '-', '')) = ?
              AND (explanation IS NULL OR explanation = '')
        """, (explanation, puzzle_number, answer))

        if result.rowcount > 0:
            updated += result.rowcount

    conn.commit()
    return updated


# ============================================================
# Main
# ============================================================

def scrape_category(cat: dict, conn: sqlite3.Connection, args) -> dict:
    """Scrape one category. Returns stats dict."""
    label = cat['label']
    total_updated = 0
    total_fetched = 0
    total_skipped = 0
    total_no_match = 0

    if args.daily:
        max_page = 1
    else:
        max_page = get_max_page(cat)
        print(f'  {label} category pages: {max_page}')

    page = args.start_page
    puzzles_done = 0
    consecutive_empty = 0

    while page <= max_page:
        if args.limit and puzzles_done >= args.limit:
            print(f'\nReached limit of {args.limit} puzzles')
            break

        print(f'\n--- {label} Page {page} ---')
        posts = get_post_links(cat, page)

        if not posts:
            consecutive_empty += 1
            print(f'  No {label} posts found (empty streak: {consecutive_empty})')
            if consecutive_empty >= 5:
                print(f'  Stopping after {consecutive_empty} consecutive empty pages')
                break
            page += 1
            continue

        consecutive_empty = 0

        for post in posts:
            if args.limit and puzzles_done >= args.limit:
                break

            pnum = post['puzzle_number']

            # Skip if puzzle doesn't exist in DB
            if not args.dry_run and not puzzle_exists(conn, pnum):
                total_no_match += 1
                continue

            # Skip if already has explanations
            if not args.dry_run and puzzle_has_explanations(conn, pnum):
                total_skipped += 1
                continue

            print(f'  {label} {pnum}: fetching {post["url"][:60]}...', end=' ', flush=True)
            resp = fetch(post['url'])
            time.sleep(RATE_LIMIT)

            if not resp:
                print('FAILED')
                continue

            clues = parse_post(resp.text)

            if not clues:
                print('parsed 0 clues')
                continue

            print(f'parsed {len(clues)} clues', end='')

            if not args.dry_run:
                n = update_clues(conn, pnum, clues)
                total_updated += n
                print(f', updated {n} rows')
            else:
                print()
                for c in clues[:3]:
                    print(
                        f'    [{c["clue_number"]}{c["direction"][0]}] '
                        f'{c["answer"]!r:25} '
                        f'expl: {c["explanation"][:60]!r}'
                    )

            total_fetched += 1
            puzzles_done += 1

        page += 1
        if args.daily:
            break

    return {
        'updated': total_updated,
        'fetched': total_fetched,
        'skipped': total_skipped,
        'no_match': total_no_match,
    }


def main():
    parser = argparse.ArgumentParser(
        description='Scrape bigdave44.com explanations into clues_master.db'
    )
    parser.add_argument('--category', choices=['dt', 'toughie', 'all'], default='all',
                        help='Which category to scrape (default: all)')
    parser.add_argument('--start-page', type=int, default=1,
                        help='Start from this category page number')
    parser.add_argument('--limit', type=int, default=0,
                        help='Stop after updating this many puzzles per category (0 = no limit)')
    parser.add_argument('--daily', action='store_true',
                        help='Only scrape page 1 (today\'s puzzles)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse and print, do not write to DB')
    args = parser.parse_args()

    conn = sqlite3.connect(MASTER_DB)

    print('bigdave44 scraper')
    print(f'DB:  {MASTER_DB}')
    if args.dry_run:
        print('DRY RUN -- no DB writes')
    print()

    if args.category == 'all':
        cats_to_run = list(CATEGORIES.items())
    else:
        cats_to_run = [(args.category, CATEGORIES[args.category])]

    grand_updated = 0
    grand_fetched = 0
    grand_skipped = 0
    grand_no_match = 0

    for cat_key, cat in cats_to_run:
        print(f'\n{"=" * 60}')
        print(f'Category: {cat["label"]}')
        print(f'{"=" * 60}')

        stats = scrape_category(cat, conn, args)
        grand_updated += stats['updated']
        grand_fetched += stats['fetched']
        grand_skipped += stats['skipped']
        grand_no_match += stats['no_match']

        print(f'\n  {cat["label"]}: Fetched {stats["fetched"]}, Updated {stats["updated"]}, '
              f'Skipped {stats["skipped"]}, No match {stats["no_match"]}')

    conn.close()
    print(f'\n{"=" * 60}')
    print(f'Done. Fetched: {grand_fetched}, Updated: {grand_updated}, '
          f'Skipped (already done): {grand_skipped}, '
          f'No DB match: {grand_no_match}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
