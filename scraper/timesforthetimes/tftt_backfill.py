#!/usr/bin/env python3
"""tftt_backfill.py — Backfill Times explanations from timesforthetimes.co.uk

Two-phase approach:
  Phase 1: Discover all post URLs via WordPress REST API (fast, no rate limit concern)
  Phase 2: Fetch and parse each post page for clue explanations

Saves to data/times_explanations.db with columns:
  puzzle_number, clue_text, answer, definition, explanation

Usage:
    python tftt_backfill.py discover             # Phase 1: build URL index
    python tftt_backfill.py scrape               # Phase 2: fetch and parse pages
    python tftt_backfill.py scrape --start 27000 # Phase 2: from specific puzzle
    python tftt_backfill.py scrape --dry-run     # test parse without writing
    python tftt_backfill.py stats                # show DB statistics
"""

import argparse
import json
import re
import sqlite3
import time
import sys
from pathlib import Path

import requests
from bs4 import BeautifulSoup

BASE_DIR = Path(__file__).resolve().parent.parent.parent
TARGET_DB = BASE_DIR / 'data' / 'times_explanations.db'
INDEX_FILE = Path(__file__).resolve().parent / 'tftt_post_index.json'
PROGRESS_FILE = Path(__file__).resolve().parent / 'tftt_backfill_progress.txt'

WP_API = 'https://timesforthetimes.co.uk/wp-json/wp/v2/posts'
# Category IDs: 11 = Daily Cryptic, 21 = Weekend Cryptic
CRYPTIC_CATEGORIES = [11, 21]

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

RATE_LIMIT = 2.5  # seconds between page fetches

NOT_ANSWERS = {
    'DD', 'CD', 'NHO', 'LOI', 'COD', 'FOI', 'POI', 'NI',
    'UK', 'US', 'TV', 'BBC', 'ITV', 'DP', 'RHS', 'LHS',
    'ACROSS', 'DOWN', 'EDIT', 'NB', 'PS', 'IMO', 'FWIW',
}

ENUM_PAT = re.compile(r'\(([\d,\-\s]+)\)\s*$')


# ============================================================
# Helpers
# ============================================================

def is_answer(text):
    text = text.strip()
    if text in NOT_ANSWERS:
        return False
    if len(text) < 3:
        return False
    if not re.match(r'^[A-Z][A-Z\s\-\']{2,}$', text):
        return False
    if sum(1 for c in text if c.isalpha()) < 3:
        return False
    return True


def extract_puzzle_number(slug, link):
    """Extract the 5-digit puzzle number from slug or link."""
    # Try slug first — most have the number somewhere
    m = re.search(r'(\d{5})', slug)
    if m:
        return int(m.group(1))
    m = re.search(r'(\d{5})', link)
    if m:
        return int(m.group(1))
    return None


# ============================================================
# Phase 1: Discover post URLs via WP REST API
# ============================================================

def discover_posts():
    """Fetch all cryptic post URLs from WordPress API and save to index file."""
    print('Phase 1: Discovering post URLs via WP REST API')
    print()

    all_posts = {}  # puzzle_number -> {slug, link, title, date}

    for cat_id in CRYPTIC_CATEGORIES:
        page = 1
        while True:
            url = (f'{WP_API}?categories={cat_id}'
                   f'&per_page=100&page={page}'
                   f'&order=asc&orderby=date'
                   f'&_fields=slug,link,title,date')

            resp = requests.get(url, headers=HEADERS, timeout=30)

            if resp.status_code != 200:
                print(f'  API returned {resp.status_code} on page {page}')
                break

            posts = json.loads(resp.text)
            if not posts:
                break

            total_pages = int(resp.headers.get('X-WP-TotalPages', 0))
            total_posts = int(resp.headers.get('X-WP-Total', 0))

            for p in posts:
                pn = extract_puzzle_number(p['slug'], p['link'])
                if pn:
                    all_posts[pn] = {
                        'slug': p['slug'],
                        'link': p['link'],
                        'title': p['title']['rendered'] if isinstance(p['title'], dict) else p['title'],
                        'date': p['date'],
                    }

            cat_name = 'Daily' if cat_id == 11 else 'Weekend'
            print(f'  {cat_name} cat={cat_id}: page {page}/{total_pages} '
                  f'({len(posts)} posts, {len(all_posts)} total indexed)')

            if page >= total_pages:
                break
            page += 1
            time.sleep(0.3)  # gentle on the API

    # Save index
    # Sort by puzzle number for output
    sorted_posts = dict(sorted(all_posts.items()))
    with open(INDEX_FILE, 'w', encoding='utf-8') as f:
        json.dump(sorted_posts, f, indent=2, ensure_ascii=False)

    pn_list = sorted(all_posts.keys())
    print()
    print(f'Indexed {len(all_posts)} posts')
    print(f'Puzzle range: {min(pn_list)} to {max(pn_list)}')
    print(f'Saved to {INDEX_FILE}')


# ============================================================
# Phase 2: Fetch and parse pages
# ============================================================

def init_db(db_path):
    conn = sqlite3.connect(db_path, timeout=30)
    conn.execute('''CREATE TABLE IF NOT EXISTS clues (
        puzzle_number INTEGER,
        clue_text TEXT,
        answer TEXT,
        definition TEXT,
        explanation TEXT,
        PRIMARY KEY (puzzle_number, answer)
    )''')
    conn.commit()
    return conn


def puzzle_exists(conn, puzzle_number):
    row = conn.execute(
        'SELECT COUNT(*) FROM clues WHERE puzzle_number = ?',
        (puzzle_number,)
    ).fetchone()
    return row[0] > 0


def save_clues(conn, puzzle_number, clues):
    saved = 0
    for c in clues:
        try:
            conn.execute(
                'INSERT OR IGNORE INTO clues VALUES (?,?,?,?,?)',
                (puzzle_number, c['clue_text'], c['answer'],
                 c.get('definition', ''), c['explanation'])
            )
            saved += 1
        except Exception:
            pass
    conn.commit()
    return saved


def fetch_page(url):
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code == 200:
            return resp.text
        elif resp.status_code == 429:
            print('  rate limited — sleeping 30s')
            time.sleep(30)
            resp = requests.get(url, headers=HEADERS, timeout=15)
            return resp.text if resp.status_code == 200 else None
        else:
            return None
    except Exception as e:
        print(f'  fetch error: {e}')
        return None


def parse_page(html):
    """Parse explanations from a timesforthetimes page.

    Handles multiple formats:
    1. New: <table class="clues"> with <td class="num">, <td class="clue">, <td class="ans">
    2. Old: <table> (no class) with plain <td> pairs
    3. Very old: freeform paragraphs with <strong>ANSWER</strong> explanation<br/>
    """
    soup = BeautifulSoup(html, 'html.parser')
    clues = []

    # Find all non-calendar tables
    all_tables = soup.find_all('table')
    content_tables = [t for t in all_tables
                      if 'wp-calendar' not in str(t.get('class', []))]

    if not content_tables:
        # Try freeform paragraph format (pre-~2017 posts)
        return _parse_freeform(soup)

    for table in content_tables:
        direction = _detect_direction(table)
        if not direction:
            continue

        rows = _get_data_rows(table)
        pending_clue_text = ''
        pending_definition = ''
        pending_num = None

        for row in rows:
            tds = row.find_all('td')
            if len(tds) < 2:
                continue

            first_text = tds[0].get_text(strip=True)

            # New format: check for class="num" / class="clue" / class="ans"
            has_num_class = 'num' in (tds[0].get('class') or [])
            has_clue_class = 'clue' in (tds[1].get('class') or []) if len(tds) > 1 else False
            has_ans_class = 'ans' in (tds[1].get('class') or []) if len(tds) > 1 else False

            if has_num_class or (first_text and first_text.isdigit()):
                clue_td = tds[1]

                # Check if this row has a bold answer inline (single-row format)
                inline_answer = _extract_answer(clue_td)
                if inline_answer:
                    explanation = _extract_explanation(clue_td, inline_answer)
                    # Extract definition from underlined text
                    underlined = []
                    for el in clue_td.find_all(True):
                        style = el.get('style', '')
                        if 'underline' in style or el.name == 'u':
                            underlined.append(el.get_text(strip=True))
                    definition = ' / '.join(underlined) if underlined else ''

                    clues.append({
                        'clue_text': '',
                        'answer': inline_answer,
                        'definition': definition,
                        'explanation': explanation,
                    })
                    pending_num = None
                    continue

                # Two-row format: clue row, then answer row
                pending_num = first_text
                pending_clue_text = clue_td.get_text(separator=' ', strip=True)
                pending_clue_text = ENUM_PAT.sub('', pending_clue_text).strip()

                underlined = []
                for el in clue_td.find_all(True):
                    style = el.get('style', '')
                    if 'underline' in style or el.name == 'u':
                        underlined.append(el.get_text(strip=True))
                pending_definition = ' / '.join(underlined) if underlined else ''

            elif pending_num and (has_ans_class or not first_text):
                # Answer/explanation row (two-row format)
                ans_td = tds[1] if len(tds) >= 2 else tds[0]
                answer = _extract_answer(ans_td)
                if not answer:
                    pending_num = None
                    continue

                explanation = _extract_explanation(ans_td, answer)

                clues.append({
                    'clue_text': pending_clue_text,
                    'answer': answer,
                    'definition': pending_definition,
                    'explanation': explanation,
                })
                pending_num = None
                pending_clue_text = ''
                pending_definition = ''

    return clues


def _parse_freeform(soup):
    """Parse old-format posts where clues are in paragraphs with <strong> answers.

    Format: number <strong>ANSWER</strong> explanation text<br/>
    Direction markers: <strong>ACROSS</strong> and <strong>DOWN</strong>
    """
    entry = soup.find('div', class_='entry-content')
    if not entry:
        return []

    clues = []
    # Get all strong/b tags in the entry
    # Walk through the HTML content looking for the pattern:
    # digit(s) followed by <strong>CAPS_WORD</strong> followed by text
    html_str = str(entry)

    # Pattern: clue number, then bold answer, then explanation until next clue or <br>
    # Match: optional whitespace, 1-2 digits, space, <strong>ANSWER</strong> explanation
    pattern = re.compile(
        r'(?:^|<br\s*/?>|<p>)\s*'          # start of line / <br> / <p>
        r'(\d{1,2})[.\s]*'                    # clue number (optional period)
        r'<(?:strong|b)>'                     # opening bold
        r'([A-Z][A-Z\s\-\']{2,}?)'           # ANSWER in caps
        r'\s*</(?:strong|b)>\s*'              # closing bold
        r'(.*?)'                              # explanation text
        r'(?=<br|<p>|</p>|</div>|\d{1,2}[.\s]*<(?:strong|b)>)',  # lookahead
        re.DOTALL
    )

    # Also try plain-text format: number. ANSWER – explanation (no bold)
    plain_pattern = re.compile(
        r'(?:^|<br\s*/?>|<p>)\s*'              # start of line / <br> / <p>
        r'(\d{1,2})[.\s]+'                      # clue number with period/space
        r'([A-Z][A-Z\s\-\']{2,}?)'              # ANSWER in caps
        r'\s*[\u2013\u2014\-]+\s*'               # dash separator
        r'(.*?)'                                  # explanation text
        r'(?=<br|<p>|</p>|</div>|\d{1,2}[.\s]+[A-Z]{3})',  # lookahead
        re.DOTALL
    )

    for pat in [pattern, plain_pattern]:
        for m in pat.finditer(html_str):
            num = m.group(1)
            answer = m.group(2).strip()
            expl_html = m.group(3).strip()

            if not is_answer(answer):
                continue

            # Skip if already found this answer
            if any(c['answer'] == answer for c in clues):
                continue

            # Clean HTML from explanation
            expl_soup = BeautifulSoup(expl_html, 'html.parser')
            explanation = expl_soup.get_text(separator=' ', strip=True)
            # Strip leading punctuation
            explanation = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', explanation).strip()

            if explanation:
                clues.append({
                    'clue_text': '',
                    'answer': answer,
                    'definition': '',
                    'explanation': explanation,
                })

    return clues


def _detect_direction(table):
    thead = table.find('thead')
    if thead:
        text = thead.get_text(strip=True).lower()
        if 'across' in text:
            return 'across'
        if 'down' in text:
            return 'down'

    rows = table.find_all('tr')
    if rows:
        text = rows[0].get_text(strip=True).lower()
        if 'across' in text:
            return 'across'
        if 'down' in text:
            return 'down'
    return None


def _get_data_rows(table):
    tbody = table.find('tbody')
    if tbody:
        return tbody.find_all('tr')
    rows = table.find_all('tr')
    if not rows:
        return []
    first_text = rows[0].get_text(strip=True).lower()
    if 'across' in first_text or 'down' in first_text:
        return rows[1:]
    return rows


def _extract_answer(td):
    for tag in td.find_all(['strong', 'b']):
        candidate = tag.get_text(strip=True)
        if is_answer(candidate):
            return candidate
        # Try uppercase — some posts use lowercase bold answers
        upper = candidate.upper()
        if is_answer(upper):
            return upper
    text = td.get_text(strip=True)
    match = re.match(r'^([A-Z][A-Z\s\-\']{2,})', text)
    if match and is_answer(match.group(1).strip()):
        return match.group(1).strip()
    return None


def _extract_explanation(td, answer):
    full_text = td.get_text(separator=' ', strip=True)
    # Try exact match first, then case-insensitive
    ans_pos = full_text.find(answer)
    if ans_pos < 0:
        ans_pos = full_text.lower().find(answer.lower())
    if ans_pos >= 0:
        expl = full_text[ans_pos + len(answer):].strip()
        expl = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', expl).strip()
        return expl
    return full_text


def save_progress(puzzle_number):
    PROGRESS_FILE.write_text(str(puzzle_number))


def load_progress():
    if PROGRESS_FILE.exists():
        try:
            return int(PROGRESS_FILE.read_text().strip())
        except ValueError:
            pass
    return None


def scrape_pages(args):
    """Phase 2: Fetch and parse post pages using the discovered index."""
    if not INDEX_FILE.exists():
        print(f'No index file found at {INDEX_FILE}')
        print('Run "python tftt_backfill.py discover" first.')
        return

    with open(INDEX_FILE, 'r', encoding='utf-8') as f:
        index = json.load(f)

    conn = init_db(TARGET_DB)

    existing = conn.execute('SELECT COUNT(DISTINCT puzzle_number) FROM clues').fetchone()[0]
    total_clues = conn.execute('SELECT COUNT(*) FROM clues').fetchone()[0]

    # Filter and sort puzzle numbers
    puzzle_numbers = sorted(int(k) for k in index.keys())

    if args.start:
        puzzle_numbers = [p for p in puzzle_numbers if p >= args.start]
    if args.end:
        puzzle_numbers = [p for p in puzzle_numbers if p <= args.end]
    if args.resume:
        last = load_progress()
        if last:
            puzzle_numbers = [p for p in puzzle_numbers if p > last]
            print(f'Resuming after puzzle {last}')

    print(f'Phase 2: Scraping post pages')
    print(f'DB: {TARGET_DB}')
    print(f'Existing: {existing} puzzles, {total_clues} clues')
    print(f'Index has {len(index)} posts, processing {len(puzzle_numbers)} puzzles')
    if args.dry_run:
        print('DRY RUN')
    print()

    fetched = 0
    skipped = 0
    failed = 0
    total_saved = 0

    for pn in puzzle_numbers:
        pn_str = str(pn)
        if pn_str not in index:
            continue

        if not args.dry_run and puzzle_exists(conn, pn):
            skipped += 1
            continue

        post = index[pn_str]
        url = post['link']

        html = fetch_page(url)
        if html is None:
            failed += 1
            print(f'  {pn}: FETCH FAILED')
            save_progress(pn)
            time.sleep(1)
            continue

        clues = parse_page(html)

        if not clues:
            failed += 1
            print(f'  {pn}: no clues parsed')
            save_progress(pn)
            time.sleep(1)
            continue

        if args.dry_run:
            print(f'  {pn}: {len(clues)} clues')
            for c in clues[:2]:
                print(f'    {c["answer"]:20} expl: {c["explanation"][:60]}')
        else:
            n = save_clues(conn, pn, clues)
            total_saved += n
            print(f'  {pn}: saved {n} clues')

        fetched += 1
        save_progress(pn)
        time.sleep(RATE_LIMIT)

        if fetched % 100 == 0:
            total_now = conn.execute('SELECT COUNT(*) FROM clues').fetchone()[0]
            puzzles_now = conn.execute('SELECT COUNT(DISTINCT puzzle_number) FROM clues').fetchone()[0]
            print(f'  --- Progress: {fetched} fetched, {puzzles_now} puzzles, {total_now} clues ---')

    total_final = conn.execute('SELECT COUNT(*) FROM clues').fetchone()[0]
    puzzles_final = conn.execute('SELECT COUNT(DISTINCT puzzle_number) FROM clues').fetchone()[0]
    conn.close()

    print()
    print(f'Done.')
    print(f'  Fetched: {fetched}')
    print(f'  Skipped (already had): {skipped}')
    print(f'  Failed/no clues: {failed}')
    print(f'  Clues saved this run: {total_saved}')
    print(f'  DB total: {puzzles_final} puzzles, {total_final} clues')


def show_stats():
    """Show DB statistics."""
    if not TARGET_DB.exists():
        print('No database found.')
        return
    conn = sqlite3.connect(TARGET_DB, timeout=30)
    total = conn.execute('SELECT COUNT(*) FROM clues').fetchone()[0]
    puzzles = conn.execute('SELECT COUNT(DISTINCT puzzle_number) FROM clues').fetchone()[0]
    minp = conn.execute('SELECT MIN(puzzle_number) FROM clues').fetchone()[0]
    maxp = conn.execute('SELECT MAX(puzzle_number) FROM clues').fetchone()[0]
    with_expl = conn.execute("SELECT COUNT(*) FROM clues WHERE explanation != ''").fetchone()[0]
    with_def = conn.execute("SELECT COUNT(*) FROM clues WHERE definition != ''").fetchone()[0]
    conn.close()

    print(f'times_explanations.db stats:')
    print(f'  Puzzles: {puzzles} (range {minp} to {maxp})')
    print(f'  Total clues: {total}')
    print(f'  With explanations: {with_expl}')
    print(f'  With definitions: {with_def}')

    if INDEX_FILE.exists():
        with open(INDEX_FILE, 'r') as f:
            index = json.load(f)
        print(f'  Index: {len(index)} posts discovered')
        remaining = len(index) - puzzles
        print(f'  Remaining to scrape: ~{remaining}')


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Backfill Times explanations from timesforthetimes.co.uk'
    )
    sub = parser.add_subparsers(dest='command')

    sub.add_parser('discover', help='Phase 1: Discover all post URLs via WP API')

    scrape_p = sub.add_parser('scrape', help='Phase 2: Fetch and parse pages')
    scrape_p.add_argument('--start', type=int, help='Start from this puzzle number')
    scrape_p.add_argument('--end', type=int, help='End at this puzzle number')
    scrape_p.add_argument('--resume', action='store_true', help='Resume from last progress')
    scrape_p.add_argument('--dry-run', action='store_true', help='Parse but do not write')

    sub.add_parser('stats', help='Show DB statistics')

    args = parser.parse_args()

    if args.command == 'discover':
        discover_posts()
    elif args.command == 'scrape':
        scrape_pages(args)
    elif args.command == 'stats':
        show_stats()
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
