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
    """Parse modern fts-table format posts.

    Two sub-variants:
      A) fts-table: structured <table> with fts-group/fts-subgroup TD classes
      B) fts-list: nested <div> elements with fts-group/fts-subgroup div classes
    """
    fts_div = content.find('div', class_=lambda c: c and 'fts' in c)
    if not fts_div:
        return []

    # Variant B: fts-list (div-based, no table)
    fts_classes = fts_div.get('class', [])
    if 'fts-list' in fts_classes:
        return _parse_fts_list(fts_div)

    # Variant A: fts-table (table-based)
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


def _parse_fts_list(fts_div: Tag) -> list[dict]:
    """Parse fts-list format: nested divs with fts-group/fts-subgroup classes.

    Structure:
      div.fts.fts-list
        div.fts-group  → "ACROSS" or "DOWN" (direction header)
        div             → wrapper containing per-clue divs
          div.fts-group → one per clue
            div.fts-subgroup → clue text (e.g. "8. Rugged comedian... (8)")
            div.fts-subgroup → answer (e.g. "CARPETED")
            div.fts-subgroup → explanation
    """
    clues = []
    direction = None

    for child in fts_div.children:
        if not isinstance(child, Tag):
            continue

        child_classes = child.get('class', [])

        # Direction header: top-level fts-group with "ACROSS" or "DOWN"
        if 'fts-group' in child_classes:
            text = child.get_text(strip=True).lower()
            if 'across' in text:
                direction = 'across'
            elif 'down' in text:
                direction = 'down'
            continue

        if not direction:
            continue

        # Wrapper div containing fts-group divs (one per clue)
        clue_groups = child.find_all('div', class_='fts-group', recursive=False)
        for group in clue_groups:
            subgroups = group.find_all('div', class_='fts-subgroup', recursive=False)
            if len(subgroups) < 2:
                continue

            # First subgroup: clue text (starts with number)
            clue_text = subgroups[0].get_text(strip=True)
            num_match = re.match(r'^(\d+[\s./]*(?:\d+)?)', clue_text)
            if not num_match:
                continue
            clue_num = num_match.group(1).strip().rstrip('.')

            # Second subgroup: answer
            answer = subgroups[1].get_text(strip=True)
            if not answer or len(answer) < 2:
                continue

            # Third subgroup (if present): explanation
            explanation = ''
            if len(subgroups) >= 3:
                explanation = subgroups[2].get_text(separator=' ', strip=True)

            clues.append({
                'clue_number': clue_num,
                'direction': direction,
                'answer': answer.upper(),
                'explanation': explanation,
            })

    return clues


def _extract_answer_from_explanation(expl_line: str) -> tuple[str, str]:
    """Extract answer and remaining explanation from an explanation line.

    Common patterns:
      - "= ANSWER" or "= ANSWER rest"
      - "ANSWER (definition) rest"
      - "(wordplay)* ANSWER"
      - "explanation text = ANSWER"

    Returns (answer, explanation) or ('', '') if no answer found.
    """
    expl_line = expl_line.strip()
    if not expl_line:
        return '', ''

    # Pattern 1: "= ANSWER" anywhere in the line
    eq_match = re.search(r'=\s+([A-Z][A-Z\s\-\']*[A-Z])\b', expl_line)
    if eq_match:
        answer = eq_match.group(1).strip()
        if len(answer.replace(' ', '').replace('-', '')) >= 2:
            return answer, expl_line

    # Pattern 2: ALL-CAPS word(s) at the start or standalone
    caps_match = re.match(r'^\(?[^A-Z]*\)?\s*([A-Z][A-Z\s\-\']*[A-Z])\b', expl_line)
    if caps_match:
        answer = caps_match.group(1).strip()
        if len(answer.replace(' ', '').replace('-', '')) >= 2:
            return answer, expl_line

    # Pattern 3: Find any ALL-CAPS word(s) in the line (2+ consecutive caps letters)
    caps_match = re.search(r'\b([A-Z][A-Z\s\-\']{0,}[A-Z])\b', expl_line)
    if caps_match:
        answer = caps_match.group(1).strip()
        if len(answer.replace(' ', '').replace('-', '')) >= 2:
            return answer, expl_line

    # Pattern 4: Single all-caps word (e.g., "AFRO")
    caps_match = re.search(r'\b([A-Z]{3,})\b', expl_line)
    if caps_match:
        return caps_match.group(1), expl_line

    return '', ''


def _split_p_on_br(element: Tag) -> list[str]:
    """Split a <p> element's content on <br/> tags, returning text lines."""
    html_str = str(element)
    html_str = re.sub(r'<br\s*/?>', '\n', html_str)
    line_soup = BeautifulSoup(html_str, 'html.parser')
    return [l.strip() for l in line_soup.get_text().split('\n') if l.strip()]


def parse_paragraph_format(content: Tag) -> list[dict]:
    """Parse legacy paragraph-based format posts.

    Handles multiple sub-variants:
      A) <p> tags with bold ANSWER in <strong>/<b>
      B) <p> tags with clue on line 1 and explanation (containing CAPS answer) on line 2,
         separated by <br/>
      C) Direction headers combined with first clue via <br/>

    Direction headers: <strong>Across</strong> / <strong>Down</strong>
    (may be standalone or combined with first clue in same <p> via <br/>)
    """
    clues = []
    direction = None

    # Clue number regex: matches "8", "12/14", "1,17", "1/17"
    CLUE_NUM_RE = re.compile(r'^(\d+[\s/,]*\d*)\s')

    def _process_clue_pair(clue_line: str, expl_line: str, cur_dir: str):
        """Process a clue_line + explanation_line pair."""
        clue_line = clue_line.strip()
        expl_line = expl_line.strip()
        if not clue_line:
            return

        num_match = CLUE_NUM_RE.match(clue_line)
        if not num_match:
            return
        clue_num = num_match.group(1).strip().rstrip(',').rstrip('.')

        answer, explanation = _extract_answer_from_explanation(expl_line)
        if not answer:
            return

        clues.append({
            'clue_number': clue_num,
            'direction': cur_dir,
            'answer': answer.upper(),
            'explanation': explanation,
        })

    for element in content.children:
        if not isinstance(element, Tag):
            continue

        text = element.get_text(strip=True)
        if not text:
            continue

        # Direction headers
        if element.name in ('p', 'h2', 'h3', 'h4'):
            lower = text.lower()

            # Check for standalone direction header
            if lower in ('across', 'down'):
                direction = lower
                continue

            # Check for bold direction header (may be combined with first clue)
            bold_tag = element.find(['strong', 'b'])
            if bold_tag:
                bold_text = bold_tag.get_text(strip=True).lower()
                if bold_text in ('across', 'down'):
                    direction = bold_text

                    # The direction header may be combined with clue text via <br/>
                    if element.name == 'p':
                        lines = _split_p_on_br(element)
                        # lines[0] = "Across"/"Down", lines[1] = clue, lines[2] = explanation
                        if len(lines) >= 3:
                            _process_clue_pair(lines[1], lines[2], direction)
                        # Could have more clue pairs: lines[3]=clue, lines[4]=expl, etc.
                        i = 3
                        while i + 1 < len(lines):
                            _process_clue_pair(lines[i], lines[i + 1], direction)
                            i += 2
                    continue

        if not direction:
            continue

        if element.name != 'p':
            continue

        # First try: look for answer in bold/strong (all-caps, 3+ letters)
        answer = ''
        for tag in element.find_all(['strong', 'b']):
            candidate = tag.get_text(strip=True)
            candidate = re.sub(r'\s*\([\d,\-\s]+\)\s*$', '', candidate)
            if candidate == candidate.upper() and len(candidate) >= 3 and re.match(r'^[A-Z\s\-\']+$', candidate):
                answer = candidate
                break

        if answer:
            # Extract clue number from the beginning of the paragraph
            full_text = element.get_text(separator=' ', strip=True)
            num_match = re.match(r'^(\d+[\s,/]*(?:\d+)?)', full_text)
            clue_num = num_match.group(1).strip().rstrip(',') if num_match else ''

            # Extract explanation: text after the answer
            explanation = ''
            ans_pos = full_text.find(answer)
            if ans_pos >= 0:
                explanation = full_text[ans_pos + len(answer):].strip()
                explanation = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', explanation).strip()

            if clue_num:
                clues.append({
                    'clue_number': clue_num,
                    'direction': direction,
                    'answer': answer,
                    'explanation': explanation,
                })
            continue

        # Second try: split paragraph on <br/> and pair clue+explanation lines
        lines = _split_p_on_br(element)
        if len(lines) >= 2:
            # Pair lines: lines[0]=clue, lines[1]=explanation
            _process_clue_pair(lines[0], lines[1], direction)
            # Handle additional pairs in same paragraph
            i = 2
            while i + 1 < len(lines):
                _process_clue_pair(lines[i], lines[i + 1], direction)
                i += 2
        elif len(lines) == 1:
            # Single line — might have answer inline after enumeration
            line = lines[0]
            num_match = CLUE_NUM_RE.match(line)
            if num_match:
                clue_num = num_match.group(1).strip().rstrip(',').rstrip('.')
                rest = line[num_match.end():]
                # Look for answer after enumeration
                enum_match = re.search(r'\([\d,\-\s]+\)\s*', rest)
                if enum_match:
                    after_enum = rest[enum_match.end():]
                    answer, explanation = _extract_answer_from_explanation(after_enum)
                    if answer:
                        clues.append({
                            'clue_number': clue_num,
                            'direction': direction,
                            'answer': answer.upper(),
                            'explanation': explanation if explanation else after_enum,
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
            # Check for direction header in <th> elements (common variant)
            ths = row.find_all('th')
            if ths:
                # Get all text from the th, including nested spans
                th_text = ths[0].get_text(strip=True).lower()
                if 'across' in th_text:
                    direction = 'across'
                    continue
                elif 'down' in th_text:
                    direction = 'down'
                    continue

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


def parse_two_col_detail_table(content: Tag) -> list[dict]:
    """Parse 2-column table format: TD[0]=clue number, TD[1]=detail (clue + answer + explanation).

    Common on fifteensquared for Independent puzzles. Structure:
      - Header row: "No" | "Detail" (bold, skip)
      - Direction row: "Across"/"Down" (bold in TD[0], skip)
      - Data rows: clue_number | detail cell containing:
          - Red-coloured span with clue text
          - Blue bold ANSWER
          - Explanation text in subsequent <p> tags

    Only matches tables that have exactly 2 columns with a "No"/"Detail" header
    or direction headers in bold.
    """
    tables = content.find_all('table')
    if not tables:
        return []

    clues = []
    direction = None

    for element in content.children:
        if not isinstance(element, Tag):
            continue

        # Check for direction header in <p>, <h2>, etc. before table
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

        # Check if this is a 2-col detail table
        rows = element.find_all('tr')
        if not rows:
            continue

        # Verify it's a 2-column table (check first few data rows)
        col_counts = []
        for r in rows[:5]:
            tds = r.find_all('td')
            if tds:
                col_counts.append(len(tds))
        if not col_counts or max(col_counts) > 2:
            continue  # Not a 2-col table

        for row in rows:
            # Check for direction header in <th> elements
            ths = row.find_all('th')
            if ths:
                th_text = ths[0].get_text(strip=True).lower()
                if 'across' in th_text:
                    direction = 'across'
                    continue
                elif 'down' in th_text:
                    direction = 'down'
                    continue

            tds = row.find_all('td')
            if not tds:
                continue

            # Skip header row ("No" | "Detail")
            first_text = tds[0].get_text(strip=True)
            if first_text.lower() in ('no', 'no.', '#', 'number'):
                continue

            # Check for direction header in first TD
            first_lower = first_text.lower()
            if first_lower in ('across', 'down'):
                direction = first_lower
                continue
            bold = tds[0].find(['strong', 'b'])
            if bold and bold.get_text(strip=True).lower() in ('across', 'down'):
                direction = bold.get_text(strip=True).lower()
                continue

            if len(tds) != 2:
                continue
            if not direction:
                continue

            # TD[0] = clue number
            clue_num_text = first_text.strip()
            if not clue_num_text or not clue_num_text[0].isdigit():
                continue

            # Clean clue number (handle "12/14" linked clues)
            clue_num_text = clue_num_text.split()[0]  # take first part

            # TD[1] = detail cell: find answer in blue bold
            detail_td = tds[1]
            answer = ''
            explanation_parts = []

            # Look for answer in bold tags (typically blue-coloured)
            for tag in detail_td.find_all(['strong', 'b']):
                candidate = tag.get_text(strip=True)
                # Remove enumeration suffix
                candidate = re.sub(r'\s*\([\d,\-\s]+\)\s*$', '', candidate)
                if (candidate == candidate.upper() and len(candidate) >= 2
                        and re.match(r'^[A-Z\s\-\']+$', candidate)):
                    answer = candidate
                    break

            if not answer:
                # Fallback: look for ALL-CAPS word in the text after enumeration
                full_text = detail_td.get_text(separator=' ', strip=True)
                # Find text after the enumeration
                enum_match = re.search(r'\(\d[\d,\-\s]*\)', full_text)
                if enum_match:
                    after_enum = full_text[enum_match.end():]
                    caps_match = re.search(r'\b([A-Z][A-Z\s\-\']{1,}[A-Z])\b', after_enum)
                    if caps_match:
                        answer = caps_match.group(1).strip()

            if not answer:
                continue

            # Extract explanation: text after the answer
            # Get all <p> tags in the detail cell — first <p> is usually the clue,
            # subsequent ones are answer + explanation
            p_tags = detail_td.find_all('p')
            if len(p_tags) >= 2:
                # Skip first paragraph (clue text), collect rest as explanation
                expl_texts = []
                for p in p_tags[1:]:
                    p_text = p.get_text(separator=' ', strip=True)
                    if p_text:
                        expl_texts.append(p_text)
                explanation = ' '.join(expl_texts)
            else:
                # Single block — try to extract after the enumeration+answer
                full_text = detail_td.get_text(separator=' ', strip=True)
                ans_pos = full_text.find(answer)
                if ans_pos >= 0:
                    explanation = full_text[ans_pos + len(answer):].strip()
                else:
                    explanation = ''

            # Clean leading separators
            explanation = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', explanation).strip()

            clues.append({
                'clue_number': clue_num_text,
                'direction': direction,
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

    # Try 2-column detail table format (common on Independent)
    clues = parse_two_col_detail_table(content)
    if clues:
        return clues

    # Try plain 3-column table format
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
