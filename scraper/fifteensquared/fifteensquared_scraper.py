#!/usr/bin/env python3
"""fifteensquared_scraper.py — Scrape clue explanations from fifteensquared.net

Fetches Guardian and Independent cryptic puzzle explanations and updates matching
rows in clues_master.db.

Two HTML formats are supported:
  - Modern (fts-table): structured <table> with fts-group/fts-subgroup classes
  - Legacy (paragraph): <p> tags with color-coded spans (pre-~2020)

Usage:
    python fifteensquared_scraper.py                          # backfill from last explained
    python fifteensquared_scraper.py --source guardian         # Guardian only
    python fifteensquared_scraper.py --source independent      # Independent only
    python fifteensquared_scraper.py --limit 50                # stop after 50 puzzles
    python fifteensquared_scraper.py --start-page 100          # start from category page 100
    python fifteensquared_scraper.py --daily                   # today's puzzles only
    python fifteensquared_scraper.py --dry-run                 # parse without DB writes

Writes to: data/clues_master.db (updates existing clue rows with explanation)
Matches rows by: source + puzzle_number + UPPER(answer)
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
RATE_LIMIT = 2.5  # seconds between requests

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

# Category page URLs for each source
CATEGORY_URLS = {
    'guardian': 'https://www.fifteensquared.net/category/guardian/',
    'independent': 'https://www.fifteensquared.net/category/independent/',
}

# Regex to extract puzzle number from post titles
# Guardian: "Guardian Cryptic crossword No 29,931 by Vulcan" or "Guardian 29,934 – Pangakupu"
# Independent: "Independent 12,286 by Liari"
PUZZLE_NUM_RE = re.compile(
    r'(?:Guardian\s+(?:Cryptic\s+(?:crossword\s+)?)?(?:No\s+)?|Independent\s+)'
    r'([\d,]+)',
    re.IGNORECASE,
)

# Map post title keywords to DB source values
SOURCE_MAP = {
    'guardian': 'guardian',
    'independent': 'independent',
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

def get_post_links(category_url: str, page: int = 1) -> list[dict]:
    """Fetch a category page and return list of {title, url, puzzle_number, source}."""
    url = category_url if page == 1 else f'{category_url}page/{page}/'
    resp = fetch(url)
    if not resp:
        return []

    soup = BeautifulSoup(resp.text, 'html.parser')
    posts = []

    for a in soup.find_all('a', rel='bookmark'):
        title = a.get_text(strip=True)
        href = a.get('href', '')

        # Skip non-cryptic posts (quiptic, quick, prize, genius, etc.)
        title_lower = title.lower()
        if any(skip in title_lower for skip in
               ['quiptic', 'quick cryptic', 'prize', 'genius', 'everyman',
                'on sunday', 'sunday']):
            continue

        # Extract puzzle number
        m = PUZZLE_NUM_RE.search(title)
        if not m:
            continue
        puzzle_number = m.group(1).replace(',', '')

        # Determine source
        if 'independent' in title_lower:
            source = 'independent'
        elif 'guardian' in title_lower:
            source = 'guardian'
        else:
            continue

        # Extract publication date from URL: /YYYY/MM/DD/...
        date_match = re.search(r'/(\d{4})/(\d{2})/(\d{2})/', href)
        pub_date = f'{date_match.group(1)}-{date_match.group(2)}-{date_match.group(3)}' if date_match else None

        posts.append({
            'title': title,
            'url': href,
            'puzzle_number': puzzle_number,
            'source': source,
            'pub_date': pub_date,
        })

    return posts


def get_max_page(category_url: str) -> int:
    """Get the maximum page number from category pagination."""
    resp = fetch(category_url)
    if not resp:
        return 1
    soup = BeautifulSoup(resp.text, 'html.parser')

    # Look for page numbers in pagination links
    max_page = 1
    for a in soup.find_all('a'):
        href = a.get('href', '')
        m = re.search(r'/page/(\d+)/', href)
        if m:
            max_page = max(max_page, int(m.group(1)))
    return max_page


# ============================================================
# HTML parsers
# ============================================================

def parse_fts_table(content: Tag) -> list[dict]:
    """Parse modern fts-table format posts."""
    fts_div = content.find('div', class_=lambda c: c and 'fts' in c)
    if not fts_div:
        return []

    table = fts_div.find('table')
    if not table:
        return []

    clues = []
    direction = None
    rows = table.find_all('tr')

    i = 0
    while i < len(rows):
        row = rows[i]

        # Check for direction header
        group_td = row.find('td', class_='fts-group')
        if group_td:
            text = group_td.get_text(strip=True).lower()
            if 'across' in text:
                direction = 'across'
            elif 'down' in text:
                direction = 'down'
            i += 1
            continue

        # Clue row: has fts-subgroup td with a number
        subgroup_tds = row.find_all('td', class_='fts-subgroup')
        if subgroup_tds and direction:
            clue_num = subgroup_tds[0].get_text(strip=True)
            if not clue_num or not clue_num[0].isdigit():
                i += 1
                continue

            # Answer is in the second fts-subgroup TD (positional)
            answer = ''
            if len(subgroup_tds) >= 2:
                answer = subgroup_tds[1].get_text(strip=True)

            # Fallback: look for any ALL-CAPS text in spans or bold tags
            if not answer or answer != answer.upper() or len(answer) < 3:
                answer = ''
                for span in row.find_all('span'):
                    text = span.get_text(strip=True)
                    if text == text.upper() and len(text) >= 3 and re.match(r'^[A-Z\s\-\']+$', text):
                        answer = text
                        break
            if not answer:
                for tag in row.find_all(['strong', 'b']):
                    text = tag.get_text(strip=True)
                    if text == text.upper() and len(text) >= 3 and re.match(r'^[A-Z\s\-\']+$', text):
                        answer = text
                        break

            # Look for explanation in next row
            explanation = ''
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                # Explanation row typically has no fts-subgroup or fts-group
                if not next_row.find('td', class_='fts-group') and not (
                    next_row.find('td', class_='fts-subgroup') and
                    next_row.find('td', class_='fts-subgroup').get_text(strip=True)[:1].isdigit()
                ):
                    # Get all text from the explanation row
                    explanation = next_row.get_text(separator=' ', strip=True)
                    i += 1  # skip the explanation row

            if answer:
                clues.append({
                    'clue_number': clue_num,
                    'direction': direction,
                    'answer': answer,
                    'explanation': explanation,
                })

        i += 1

    return clues


def parse_paragraph_format(content: Tag) -> list[dict]:
    """Parse legacy paragraph-based format posts.

    Structure: <p> tags containing:
      - Blue span with clue number + clue text
      - Red/dark bold span with ANSWER
      - Explanation text after the answer
    Direction headers are <strong>Across</strong> / <strong>Down</strong>
    """
    clues = []
    direction = None

    for element in content.children:
        if not isinstance(element, Tag):
            continue

        text = element.get_text(strip=True)

        # Direction headers
        if element.name in ('p', 'h2', 'h3', 'h4'):
            lower = text.lower()
            if lower in ('across', 'down') or (
                element.find(['strong', 'b']) and
                element.find(['strong', 'b']).get_text(strip=True).lower() in ('across', 'down')
            ):
                if 'across' in lower:
                    direction = 'across'
                elif 'down' in lower:
                    direction = 'down'
                continue

        if not direction:
            continue

        if element.name != 'p':
            continue

        # Look for answer in bold/strong (all-caps, 3+ letters)
        answer = ''
        answer_tag = None
        for tag in element.find_all(['strong', 'b']):
            candidate = tag.get_text(strip=True)
            # Remove any enumeration suffix like (7)
            candidate = re.sub(r'\s*\([\d,\-\s]+\)\s*$', '', candidate)
            if candidate == candidate.upper() and len(candidate) >= 3 and re.match(r'^[A-Z\s\-\']+$', candidate):
                answer = candidate
                answer_tag = tag
                break

        if not answer:
            continue

        # Extract clue number from the beginning of the paragraph
        full_text = element.get_text(separator=' ', strip=True)
        num_match = re.match(r'^(\d+[\s,]*(?:\d+)?)', full_text)
        clue_num = num_match.group(1).strip().rstrip(',') if num_match else ''

        # Extract explanation: text after the answer
        explanation = ''
        ans_pos = full_text.find(answer)
        if ans_pos >= 0:
            explanation = full_text[ans_pos + len(answer):].strip()
            # Clean leading separators
            explanation = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', explanation).strip()

        if clue_num:
            clues.append({
                'clue_number': clue_num,
                'direction': direction,
                'answer': answer,
                'explanation': explanation,
            })

    return clues


def parse_plain_table(content: Tag) -> list[dict]:
    """Parse plain <table> format posts (no fts classes).

    Two sub-variants:
      A) Single table with "Across"/"Down" header rows (colspan)
      B) Separate tables for Across and Down, direction from context or clue number suffix

    Each data row has 3 cells: [clue_number, answer, clue_text + explanation]
    """
    tables = content.find_all('table')
    if not tables:
        return []

    clues = []
    direction = None

    # Determine if direction comes from preceding <p> elements or from within table
    # Build a list of (element_type, element) in document order
    for element in content.children:
        if not isinstance(element, Tag):
            continue

        # Check for direction header in <p>, <h2>, etc.
        if element.name in ('p', 'h2', 'h3', 'h4'):
            text = element.get_text(strip=True).lower()
            if text in ('across', 'down'):
                direction = text
                continue
            bold = element.find(['strong', 'b'])
            if bold and bold.get_text(strip=True).lower() in ('across', 'down'):
                direction = bold.get_text(strip=True).lower()
                continue

        if element.name != 'table':
            continue

        rows = element.find_all('tr')
        for row in rows:
            tds = row.find_all('td')
            if not tds:
                continue

            # Check for direction header row (colspan, contains "Across"/"Down")
            first_text = tds[0].get_text(strip=True).lower()
            if first_text in ('across', 'down'):
                direction = first_text
                continue
            # Bold direction header
            bold = tds[0].find(['strong', 'b'])
            if bold and bold.get_text(strip=True).lower() in ('across', 'down'):
                direction = bold.get_text(strip=True).lower()
                continue

            if len(tds) < 3:
                continue

            # Extract clue number from TD[0]
            clue_num_text = tds[0].get_text(strip=True)
            if not clue_num_text or not clue_num_text[0].isdigit():
                continue

            # Detect direction from clue number suffix (e.g. "1a", "2d")
            row_dir = direction
            suffix_match = re.match(r'^(\d+)\s*([adAD])$', clue_num_text)
            if suffix_match:
                clue_num_text = suffix_match.group(1)
                row_dir = 'across' if suffix_match.group(2).lower() == 'a' else 'down'

            if not row_dir:
                continue

            # Answer from TD[1] — may be bold or plain text
            answer_td = tds[1]
            answer = ''
            bold_tag = answer_td.find(['b', 'strong'])
            if bold_tag:
                answer = bold_tag.get_text(strip=True)
            if not answer:
                answer = answer_td.get_text(strip=True)

            # Clean answer
            answer = re.sub(r'\s*\([\d,\-\s]+\)\s*$', '', answer).strip()
            if not answer or len(answer) < 3:
                continue

            # Explanation from TD[2] — text after the clue
            td2_text = tds[2].get_text(separator=' ', strip=True)
            # The clue text is typically in a colored span; explanation follows after
            # Try to split on the enumeration pattern or on the answer appearing
            explanation = ''

            # Look for enumeration as separator: everything after (\d+) is explanation
            enum_match = re.search(r'\([\d,\-\s]+\)\s*', td2_text)
            if enum_match:
                explanation = td2_text[enum_match.end():].strip()
            else:
                # Fallback: take entire text (it's likely all explanation)
                explanation = td2_text

            # Clean leading separators
            explanation = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', explanation).strip()

            clues.append({
                'clue_number': clue_num_text,
                'direction': row_dir,
                'answer': answer.upper(),
                'explanation': explanation,
            })

    return clues


def parse_post(html: str) -> list[dict]:
    """Parse a fifteensquared blog post. Auto-detects format."""
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', class_='entry-content')
    if not content:
        return []

    # Try modern fts-table format first
    clues = parse_fts_table(content)
    if clues:
        return clues

    # Try plain table format
    clues = parse_plain_table(content)
    if clues:
        return clues

    # Fall back to legacy paragraph format
    return parse_paragraph_format(content)


# ============================================================
# Database helpers
# ============================================================

def puzzle_has_explanations(conn: sqlite3.Connection, source: str, puzzle_number: str) -> bool:
    """Check if this puzzle already has explanations."""
    row = conn.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = ?
          AND puzzle_number = ?
          AND explanation IS NOT NULL AND explanation != ''
    """, (source, puzzle_number)).fetchone()
    return row[0] > 0


def date_has_explanations(conn: sqlite3.Connection, source: str, pub_date: str) -> bool:
    """Check if clues on this date already have explanations (for sources without puzzle_number)."""
    row = conn.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = ?
          AND publication_date = ?
          AND explanation IS NOT NULL AND explanation != ''
    """, (source, pub_date)).fetchone()
    return row[0] > 0


def puzzle_exists(conn: sqlite3.Connection, source: str, puzzle_number: str) -> bool:
    """Check if we have any clues for this puzzle in the DB."""
    row = conn.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = ? AND puzzle_number = ?
    """, (source, puzzle_number)).fetchone()
    return row[0] > 0


def date_has_clues(conn: sqlite3.Connection, source: str, pub_date: str) -> bool:
    """Check if we have any clues for this date in the DB."""
    row = conn.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = ? AND publication_date = ?
    """, (source, pub_date)).fetchone()
    return row[0] > 0


def update_clues(conn: sqlite3.Connection, source: str, puzzle_number: str,
                 clues: list[dict]) -> int:
    """Update existing clue rows with explanations. Returns count of rows updated."""
    updated = 0
    for clue in clues:
        answer = clue['answer'].upper().replace(' ', '').replace('-', '')
        explanation = clue['explanation']
        if not explanation:
            continue

        result = conn.execute("""
            UPDATE clues
            SET explanation = ?
            WHERE source = ?
              AND puzzle_number = ?
              AND UPPER(REPLACE(REPLACE(answer, ' ', ''), '-', '')) = ?
              AND (explanation IS NULL OR explanation = '')
        """, (explanation, source, puzzle_number, answer))

        if result.rowcount > 0:
            updated += result.rowcount

    conn.commit()
    return updated


def update_clues_by_date(conn: sqlite3.Connection, source: str, pub_date: str,
                         puzzle_number: str, clues: list[dict]) -> int:
    """Update clue rows by date + answer (for sources without reliable puzzle_number).
    Also backfills puzzle_number if provided."""
    updated = 0
    for clue in clues:
        answer = clue['answer'].upper().replace(' ', '').replace('-', '')
        explanation = clue['explanation']
        if not explanation:
            continue

        result = conn.execute("""
            UPDATE clues
            SET explanation = ?,
                puzzle_number = COALESCE(NULLIF(puzzle_number, ''), ?)
            WHERE source = ?
              AND publication_date = ?
              AND UPPER(REPLACE(REPLACE(answer, ' ', ''), '-', '')) = ?
              AND (explanation IS NULL OR explanation = '')
        """, (explanation, puzzle_number, source, pub_date, answer))

        if result.rowcount > 0:
            updated += result.rowcount

    conn.commit()
    return updated


# ============================================================
# Main
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Scrape fifteensquared.net explanations into clues_master.db'
    )
    parser.add_argument('--source', choices=['guardian', 'independent'],
                        help='Scrape only this source (default: both)')
    parser.add_argument('--start-page', type=int, default=1,
                        help='Start from this category page number')
    parser.add_argument('--limit', type=int, default=0,
                        help='Stop after updating this many puzzles (0 = no limit)')
    parser.add_argument('--daily', action='store_true',
                        help='Only scrape page 1 (today\'s puzzles)')
    parser.add_argument('--dry-run', action='store_true',
                        help='Parse and print, do not write to DB')
    args = parser.parse_args()

    sources = [args.source] if args.source else ['guardian', 'independent']
    conn = sqlite3.connect(MASTER_DB)

    print('fifteensquared scraper')
    print(f'DB:      {MASTER_DB}')
    print(f'Sources: {", ".join(sources)}')
    if args.dry_run:
        print('DRY RUN -- no DB writes')
    print()

    total_updated = 0
    total_fetched = 0
    total_skipped = 0
    total_no_match = 0

    for source in sources:
        cat_url = CATEGORY_URLS[source]
        print(f'=== {source.upper()} ===')
        print(f'Category: {cat_url}')

        if args.daily:
            max_page = 1
        else:
            max_page = get_max_page(cat_url)
            print(f'Pages: {max_page}')

        page = args.start_page
        puzzles_done = 0
        consecutive_empty = 0

        while page <= max_page:
            if args.limit and puzzles_done >= args.limit:
                print(f'\nReached limit of {args.limit} puzzles')
                break

            print(f'\n--- Page {page} ---')
            posts = get_post_links(cat_url, page)

            if not posts:
                consecutive_empty += 1
                print(f'  No posts found (empty streak: {consecutive_empty})')
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
                psource = post['source']
                pub_date = post.get('pub_date')
                use_date_match = (psource == 'independent')

                # Skip if not matching requested source
                if psource not in sources:
                    continue

                if not args.dry_run:
                    if use_date_match and pub_date:
                        # Independent: match by date (puzzle_number often NULL)
                        if not date_has_clues(conn, psource, pub_date):
                            total_no_match += 1
                            continue
                        if date_has_explanations(conn, psource, pub_date):
                            total_skipped += 1
                            continue
                    else:
                        # Guardian: match by puzzle_number
                        if not puzzle_exists(conn, psource, pnum):
                            total_no_match += 1
                            continue
                        if puzzle_has_explanations(conn, psource, pnum):
                            total_skipped += 1
                            continue

                print(f'  {psource} #{pnum}: fetching {post["url"][:70]}...', end=' ', flush=True)
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
                    if use_date_match and pub_date:
                        n = update_clues_by_date(conn, psource, pub_date, pnum, clues)
                    else:
                        n = update_clues(conn, psource, pnum, clues)
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

    conn.close()
    print(f'\n{"=" * 60}')
    print(f'Done. Fetched: {total_fetched}, Updated: {total_updated}, '
          f'Skipped (already done): {total_skipped}, '
          f'No DB match: {total_no_match}')
    print(f'{"=" * 60}')


if __name__ == '__main__':
    main()
