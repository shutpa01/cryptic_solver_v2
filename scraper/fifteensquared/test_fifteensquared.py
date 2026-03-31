#!/usr/bin/env python3
"""test_fifteensquared.py — Offline test of fifteensquared URL discovery + HTML parsing.

Tests:
  1. WP API discovery: can we find the post URL for a given puzzle number + source?
  2. HTML parsing: can we extract clue_number, answer, clue_text, definition, explanation?
  3. Cross-check: does clue count match clues_master.db?

No DB writes, no Haiku calls. Pure scraping validation.

Usage:
    python scraper/fifteensquared/test_fifteensquared.py
    python scraper/fifteensquared/test_fifteensquared.py --puzzle 29958 --source guardian
    python scraper/fifteensquared/test_fifteensquared.py --puzzle 12304 --source independent
"""

import argparse
import re
import sqlite3
import sys
import time
from pathlib import Path

import requests
from bs4 import BeautifulSoup, Tag

BASE_DIR = Path(__file__).resolve().parent.parent.parent
MASTER_DB = BASE_DIR / 'data' / 'clues_master.db'

HEADERS = {
    'User-Agent': (
        'Mozilla/5.0 (Windows NT 10.0; Win64; x64) '
        'AppleWebKit/537.36 (KHTML, like Gecko) '
        'Chrome/120.0.0.0 Safari/537.36'
    )
}

WP_API = 'https://www.fifteensquared.net/wp-json/wp/v2/posts'

# fifteensquared WP category IDs
CATEGORY_IDS = {
    'guardian': 7,
    'independent': 8,
}

RATE_LIMIT = 1.5

# Known blog abbreviations that are ALL-CAPS but not answers
NOT_ANSWERS = {
    'DD', 'CD', 'NHO', 'LOI', 'COD', 'FOI', 'POI', 'NI',
    'UK', 'US', 'TV', 'BBC', 'ITV', 'DP', 'RHS', 'LHS',
    'ACROSS', 'DOWN', 'EDIT', 'NB', 'PS',
}

ENUM_PAT = re.compile(r'\(([\d,\-\s/]+)\)\s*$')


# ============================================================
# URL Discovery via WordPress API
# ============================================================

def discover_post_url(puzzle_number, source, pub_date=None):
    """Find the fifteensquared post URL for a puzzle.

    Strategy:
      1. WP API search by puzzle number within source category
      2. If search fails and pub_date provided, try date-range filter

    Returns (url, title, method) or (None, None, None).
    """
    cat_id = CATEGORY_IDS.get(source)
    if not cat_id:
        return None, None, None

    # Strategy 1: WP API search
    url, title = _search_wp_api(puzzle_number, cat_id)
    if url:
        return url, title, 'wp_search'

    # Strategy 2: date-range filter (needs pub_date)
    if pub_date:
        url, title = _search_wp_by_date(puzzle_number, cat_id, pub_date)
        if url:
            return url, title, 'wp_date'

    return None, None, None


def _search_wp_api(puzzle_number, cat_id):
    """Search WP API for a post containing the puzzle number."""
    try:
        resp = requests.get(WP_API, headers=HEADERS, timeout=15, params={
            'search': str(puzzle_number),
            'categories': str(cat_id),
            'per_page': 10,
        })
        if resp.status_code != 200:
            print(f'    WP search returned HTTP {resp.status_code}')
            return None, None

        posts = resp.json()
        if not posts:
            return None, None

        # Find post whose title/slug contains our puzzle number
        pnum_str = str(puzzle_number)
        for post in posts:
            title = post.get('title', {}).get('rendered', '')
            slug = post.get('slug', '')
            link = post.get('link', '')
            # Check if puzzle number appears in title or slug
            # Title may have commas: "No 29,958" → check both with and without
            title_clean = title.replace(',', '').replace('.', '')
            if pnum_str in title_clean or pnum_str in slug:
                return link, title

        return None, None
    except Exception as e:
        print(f'    WP search error: {e}')
        return None, None


def _search_wp_by_date(puzzle_number, cat_id, pub_date):
    """Search WP API by date range + category, then match puzzle number in title."""
    try:
        # Search posts from day before to day after
        from datetime import datetime, timedelta
        dt = datetime.strptime(pub_date, '%Y-%m-%d')
        after = (dt - timedelta(days=1)).strftime('%Y-%m-%dT00:00:00')
        before = (dt + timedelta(days=2)).strftime('%Y-%m-%dT00:00:00')

        resp = requests.get(WP_API, headers=HEADERS, timeout=15, params={
            'categories': str(cat_id),
            'after': after,
            'before': before,
            'per_page': 10,
        })
        if resp.status_code != 200:
            return None, None

        posts = resp.json()
        pnum_str = str(puzzle_number)
        for post in posts:
            title = post.get('title', {}).get('rendered', '')
            slug = post.get('slug', '')
            link = post.get('link', '')
            title_clean = title.replace(',', '').replace('.', '')
            if pnum_str in title_clean or pnum_str in slug:
                return link, title
        return None, None
    except Exception as e:
        print(f'    WP date search error: {e}')
        return None, None


# ============================================================
# HTML Parsers — enhanced with clue_text and definition
# ============================================================

def is_answer(text):
    """Return True if text looks like a crossword answer."""
    text = text.strip()
    if text in NOT_ANSWERS:
        return False
    if len(text) < 3:
        return False
    if not re.match(r'^[A-Z][A-Z\s\-\']+$', text):
        return False
    if sum(1 for c in text if c.isalpha()) < 3:
        return False
    return True


def _extract_definition(element):
    """Extract underlined text (definition) from an HTML element.

    Handles both <u> tags and style="text-decoration: underline".
    """
    parts = []
    for tag in element.find_all(True):
        style = tag.get('style', '')
        if tag.name == 'u' or 'underline' in style:
            text = tag.get_text(strip=True)
            if text:
                parts.append(text)
    return ' '.join(parts) if parts else ''


def _extract_clue_text(element):
    """Extract the clue text from a colored span/font element.

    Returns (clue_text, enumeration).
    """
    # Try font color=blue (old format)
    font = element.find('font', color=True)
    if font:
        text = font.get_text(separator=' ', strip=True)
        enum_m = ENUM_PAT.search(text)
        enumeration = enum_m.group(1).strip() if enum_m else ''
        return text, enumeration

    # Try colored spans (fts format) — blue (#4682b4) or red (#ff0000)
    clue_spans = []
    for span in element.find_all('span'):
        style = span.get('style', '')
        if '#4682b4' in style or '#ff0000' in style or 'blue' in style.lower():
            clue_spans.append(span.get_text(strip=True))
    if clue_spans:
        text = ' '.join(clue_spans)
        # Remove leading clue number (e.g. "1. " or "1 ")
        text = re.sub(r'^\d+[\s,]*(?:\d+)?\.\s*', '', text)
        enum_m = ENUM_PAT.search(text)
        enumeration = enum_m.group(1).strip() if enum_m else ''
        return text, enumeration

    # Fallback: full text up to enumeration
    text = element.get_text(separator=' ', strip=True)
    enum_m = ENUM_PAT.search(text)
    enumeration = enum_m.group(1).strip() if enum_m else ''
    return text, enumeration


def parse_plain_table(content):
    """Parse old-style <table cellpadding> format (e.g. Guardian/Brummie).

    One row per clue with 3 TDs:
      TD[0]: clue number
      TD[1]: <b>ANSWER</b>
      TD[2]: <font color='blue'>clue text with <u>definition</u> (enum)</font><br/>explanation
    """
    clues = []
    direction = None

    for element in content.children:
        if not isinstance(element, Tag):
            continue

        # Direction headers outside tables
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

            # Direction header row
            first_text = tds[0].get_text(strip=True)
            first_lower = first_text.lower()
            if first_lower in ('across', 'down'):
                direction = first_lower
                continue
            bold = tds[0].find(['strong', 'b'])
            if bold and bold.get_text(strip=True).lower() in ('across', 'down'):
                direction = bold.get_text(strip=True).lower()
                continue

            if len(tds) < 3 or not direction:
                continue

            # TD[0]: clue number
            clue_num = tds[0].get_text(strip=True)
            if not clue_num or not clue_num[0].isdigit():
                continue

            # TD[1]: answer (bold)
            answer_td = tds[1]
            bold_tag = answer_td.find(['b', 'strong'])
            answer = bold_tag.get_text(strip=True) if bold_tag else answer_td.get_text(strip=True)
            answer = answer.strip()
            if not is_answer(answer):
                continue

            # TD[2]: clue text + explanation
            td2 = tds[2]

            # Clue text from colored element
            clue_text, enumeration = _extract_clue_text(td2)

            # Definition from underlined text
            definition = _extract_definition(td2)

            # Explanation: text after <br> or after enumeration
            explanation = ''
            # Method 1: look for <br> tag — explanation follows it
            br = td2.find('br')
            if br:
                # Collect all text after the <br>
                parts = []
                for sibling in br.next_siblings:
                    if isinstance(sibling, Tag):
                        parts.append(sibling.get_text(separator=' ', strip=True))
                    elif isinstance(sibling, str):
                        parts.append(sibling.strip())
                explanation = ' '.join(p for p in parts if p).strip()
            else:
                # Fallback: everything after the enumeration
                full = td2.get_text(separator=' ', strip=True)
                enum_m = ENUM_PAT.search(full)
                if enum_m:
                    explanation = full[enum_m.end():].strip()

            # Clean leading separators from explanation
            explanation = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', explanation).strip()

            dir_suffix = 'a' if direction == 'across' else 'd'
            clues.append({
                'clue_number': f'{clue_num}{dir_suffix}',
                'direction': direction,
                'answer': answer.upper(),
                'clue_text': clue_text,
                'enumeration': enumeration,
                'definition': definition,
                'explanation': explanation,
            })

    return clues


def parse_fts_list(content):
    """Parse fts-list format (e.g. Independent/Dalibor).

    Structure:
      <div class="fts fts-list">
        <div class="fts-group"><strong>ACROSS</strong></div>
        <div>
          <div class="fts-group">
            <div class="fts-subgroup">  <!-- clue number + text + definition -->
            <div class="fts-subgroup">  <!-- answer (bold) -->
            <div class="fts-subgroup">  <!-- explanation -->
          </div>
        </div>
    """
    fts_div = content.find('div', class_=lambda c: c and 'fts-list' in c)
    if not fts_div:
        return []

    clues = []
    direction = None

    # Walk through all fts-group divs
    for group in fts_div.find_all('div', class_=lambda c: c and 'fts-group' in c, recursive=True):
        # Check if this is a direction header
        text = group.get_text(strip=True)
        if text.upper() in ('ACROSS', 'DOWN'):
            direction = text.lower()
            continue

        if not direction:
            continue

        # Look for fts-subgroup children (direct or nested)
        subgroups = group.find_all('div', class_=lambda c: c and 'fts-subgroup' in c, recursive=True)
        if len(subgroups) < 2:
            continue

        # Subgroup 0: clue number + text + definition
        sg0 = subgroups[0]
        sg0_text = sg0.get_text(strip=True)

        # Extract clue number from start
        num_match = re.match(r'^(\d+[\s,]*(?:\d+)?)\.\s*', sg0_text)
        if not num_match:
            num_match = re.match(r'^(\d+[\s,]*(?:\d+)?)\s+', sg0_text)
        if not num_match:
            continue
        clue_num = num_match.group(1).strip().rstrip('.,')

        # Clue text and definition from subgroup 0
        clue_text, enumeration = _extract_clue_text(sg0)
        definition = _extract_definition(sg0)

        # Subgroup 1: answer
        sg1 = subgroups[1]
        answer = sg1.get_text(strip=True)
        if not is_answer(answer):
            continue

        # Subgroup 2: explanation (if present)
        explanation = ''
        if len(subgroups) >= 3:
            explanation = subgroups[2].get_text(separator=' ', strip=True)
        explanation = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', explanation).strip()

        dir_suffix = 'a' if direction == 'across' else 'd'
        clues.append({
            'clue_number': f'{clue_num}{dir_suffix}',
            'direction': direction,
            'answer': answer.upper(),
            'clue_text': clue_text,
            'enumeration': enumeration,
            'definition': definition,
            'explanation': explanation,
        })

    return clues


def parse_two_col_table(content):
    """Parse 2-column styled table format (e.g. Independent/Mog).

    Structure:
      <table>
        <tr>
          <td style="border:...">5</td>
          <td style="border:...">
            <span style="color: #ff0000">clue text with <u>definition</u> (enum)</span>
            <p><span style="color: #0000ff"><strong>ANSWER</strong></span> (explanation)</p>
            <p>breakdown...</p>
          </td>
        </tr>
    """
    clues = []
    direction = None

    for element in content.children:
        if not isinstance(element, Tag):
            continue

        # Direction headers in paragraphs
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

        for row in element.find_all('tr'):
            tds = row.find_all('td')

            # Direction header in table
            if len(tds) >= 1:
                first = tds[0].get_text(strip=True).lower()
                if first in ('across', 'down'):
                    direction = first
                    continue

            if len(tds) < 2 or not direction:
                continue

            # TD[0]: clue number
            clue_num = tds[0].get_text(strip=True).strip('.')
            if not clue_num or not clue_num[0].isdigit():
                continue

            # TD[1]: clue text + answer + explanation all mixed in <p> tags
            td1 = tds[1]

            # Find answer: blue bold text
            answer = ''
            for tag in td1.find_all(['strong', 'b']):
                parent_style = ''
                parent = tag.parent
                while parent and parent != td1:
                    parent_style += parent.get('style', '')
                    parent = parent.parent
                tag_style = tag.get('style', '') + parent_style
                candidate = tag.get_text(strip=True)
                if is_answer(candidate) and ('0000ff' in tag_style or 'blue' in tag_style):
                    answer = candidate
                    break
            # Fallback: any bold ALL-CAPS
            if not answer:
                for tag in td1.find_all(['strong', 'b']):
                    candidate = tag.get_text(strip=True)
                    if is_answer(candidate):
                        answer = candidate
                        break

            if not answer:
                continue

            # Clue text: from red/colored spans BEFORE any <p> tags
            clue_text = ''
            enumeration = ''
            definition = ''
            # Get top-level spans (not inside <p>)
            for child in td1.children:
                if isinstance(child, Tag) and child.name == 'p':
                    break  # stop at first <p>
                if isinstance(child, Tag) and child.name == 'span':
                    style = child.get('style', '')
                    if '#ff0000' in style or '#4682b4' in style:
                        clue_text = child.get_text(separator=' ', strip=True)
                        definition = _extract_definition(child)
                        enum_m = ENUM_PAT.search(clue_text)
                        enumeration = enum_m.group(1).strip() if enum_m else ''
                        break

            # Explanation: from <p> tags — take the one with most detail
            # Skip first (definition line) and last (bare answer), pick middle
            paragraphs = td1.find_all('p')
            expl_candidates = []
            answer_clean = re.sub(r'[^A-Z]', '', answer)
            for p in paragraphs:
                p_text = p.get_text(separator=' ', strip=True)
                if not p_text or p_text == '\xa0':
                    continue
                # Skip paragraphs that are JUST the answer
                p_clean = re.sub(r'[^A-Z]', '', p_text.upper())
                if p_clean == answer_clean:
                    continue
                expl_candidates.append(p_text)

            # The mechanism explanation is typically the longest non-definition paragraph
            # or the second one (first is definition, second is breakdown)
            explanation = ''
            if len(expl_candidates) >= 2:
                explanation = expl_candidates[1]  # breakdown paragraph
            elif expl_candidates:
                explanation = expl_candidates[0]
            explanation = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', explanation).strip()

            dir_suffix = 'a' if direction == 'across' else 'd'
            clues.append({
                'clue_number': f'{clue_num}{dir_suffix}',
                'direction': direction,
                'answer': answer.upper(),
                'clue_text': clue_text,
                'enumeration': enumeration,
                'definition': definition,
                'explanation': explanation,
            })

    return clues


def parse_fts_table(content):
    """Parse modern fts-table format (e.g. Independent/Kairos).

    Two rows per clue:
      Row 1: fts-subgroup TDs — number, answer, clue text (with underline definition)
      Row 2: fts-subgroup class, colspan=2 empty TD, then explanation TD
    """
    fts_div = content.find('div', class_=lambda c: c and 'fts-table' in c)
    if not fts_div:
        # Also check for generic fts div that contains a table
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

        # Direction header: fts-group class
        group_td = row.find('td', class_=lambda c: c and 'fts-group' in c)
        if group_td:
            text = group_td.get_text(strip=True).lower()
            if 'across' in text:
                direction = 'across'
            elif 'down' in text:
                direction = 'down'
            i += 1
            continue

        # Clue row: has fts-subgroup TDs
        subgroup_tds = row.find_all('td', class_=lambda c: c and 'fts-subgroup' in c)
        if subgroup_tds and direction:
            clue_num = subgroup_tds[0].get_text(strip=True)
            if not clue_num or not clue_num[0].isdigit():
                i += 1
                continue

            # Answer from TD[1]
            answer = ''
            if len(subgroup_tds) >= 2:
                answer = subgroup_tds[1].get_text(strip=True)

            # Fallback: find ALL-CAPS text in spans
            if not answer or not is_answer(answer):
                answer = ''
                for span in row.find_all('span'):
                    text = span.get_text(strip=True)
                    if is_answer(text):
                        answer = text
                        break

            if not answer:
                i += 1
                continue

            # Clue text + definition from TD[2]
            clue_text = ''
            enumeration = ''
            definition = ''
            if len(subgroup_tds) >= 3:
                clue_td = subgroup_tds[2]
                clue_text, enumeration = _extract_clue_text(clue_td)
                definition = _extract_definition(clue_td)

            # Explanation from next row
            explanation = ''
            if i + 1 < len(rows):
                next_row = rows[i + 1]
                next_group = next_row.find('td', class_=lambda c: c and 'fts-group' in c)
                # Check it's not a new direction header or a new clue row
                if not next_group:
                    next_subgroups = next_row.find_all('td', class_=lambda c: c and 'fts-subgroup' in c)
                    is_new_clue = False
                    for td in next_subgroups:
                        t = td.get_text(strip=True)
                        if t and t[0].isdigit():
                            is_new_clue = True
                            break
                    if not is_new_clue:
                        explanation = next_row.get_text(separator=' ', strip=True)
                        i += 1  # skip explanation row

            explanation = re.sub(r'^[\s\-\u2013\u2014:\.]+', '', explanation).strip()

            dir_suffix = 'a' if direction == 'across' else 'd'
            clues.append({
                'clue_number': f'{clue_num}{dir_suffix}',
                'direction': direction,
                'answer': answer.upper(),
                'clue_text': clue_text,
                'enumeration': enumeration,
                'definition': definition,
                'explanation': explanation,
            })

        i += 1

    return clues


def parse_paragraph_format(content):
    """Parse paragraph-based format where clues are in <p> tags, not tables.

    Pattern:
      <p><strong>Across</strong></p>
      <p>1. Clue text with <u>definition</u> underlined (5)</p>
      <p><strong>ANSWER</strong>: explanation text</p>
      ... optional commentary paragraphs ...
    """
    clues = []
    paras = content.find_all('p')
    if not paras:
        return clues

    direction = None
    pending_num = None
    pending_clue_text = ''
    pending_enumeration = ''
    pending_definition = ''

    for p in paras:
        text = p.get_text(strip=True)
        if not text:
            continue

        text_lower = text.lower().strip()

        # Section header: Across / Down (often bold)
        if text_lower in ('across', 'down'):
            direction = 'across' if text_lower == 'across' else 'down'
            pending_num = None
            continue

        if direction is None:
            continue

        # Clue line: starts with number followed by period or space
        clue_match = re.match(r'^(\d+)\s*[.\s]', text)
        if clue_match and not p.find(['strong', 'b']):
            # This looks like a clue line (has number, no bold answer)
            pending_num = clue_match.group(1)
            pending_clue_text = text

            m = ENUM_PAT.search(pending_clue_text)
            pending_enumeration = m.group(1).strip() if m else ''

            pending_definition = _extract_definition(p)
            continue

        # Also catch clue lines where the number is bold
        if clue_match and p.find(['strong', 'b']):
            # Could be clue or answer — check if the bold text is ALL CAPS (answer)
            bold = p.find(['strong', 'b'])
            bold_text = bold.get_text(strip=True) if bold else ''
            if not is_answer(bold_text):
                # Bold number, not an answer — treat as clue line
                pending_num = clue_match.group(1)
                pending_clue_text = text
                m = ENUM_PAT.search(pending_clue_text)
                pending_enumeration = m.group(1).strip() if m else ''
                pending_definition = _extract_definition(p)
                continue

        # Answer line: has bold text that looks like an answer
        if pending_num and p.find(['strong', 'b']):
            bold = p.find(['strong', 'b'])
            if not bold:
                continue

            answer = bold.get_text(strip=True)
            # Clean answer: remove trailing colon, dash etc
            answer = re.sub(r'[\s:–—\-]+$', '', answer).strip()

            if not is_answer(answer):
                # Try other bold elements
                for tag in p.find_all(['strong', 'b']):
                    candidate = re.sub(r'[\s:–—\-]+$', '', tag.get_text(strip=True)).strip()
                    if is_answer(candidate):
                        answer = candidate
                        break

            if not is_answer(answer):
                continue

            # Explanation: everything after the answer
            full_text = p.get_text(separator=' ', strip=True)
            expl = ''
            ans_pos = full_text.find(answer)
            if ans_pos >= 0:
                expl = full_text[ans_pos + len(answer):].strip()
                expl = re.sub(r'^[\s\-–—:\.]+', '', expl).strip()

            dir_suffix = 'a' if direction == 'across' else 'd'
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


def _is_cross_reference(clue):
    """Check if a clue is just a cross-reference like 'See 1' or 'See 1 across'."""
    text = clue.get('clue_text', '').strip()
    if not text:
        return not clue.get('explanation', '').strip()
    return bool(re.match(r'^See\s+\d', text, re.IGNORECASE))


def parse_post(html):
    """Parse a fifteensquared blog post. Auto-detects format.

    Returns list of dicts with: clue_number, direction, answer,
    clue_text, enumeration, definition, explanation.
    Filters out cross-reference entries ("See 1", etc.).
    """
    soup = BeautifulSoup(html, 'html.parser')
    content = soup.find('div', class_='entry-content')
    if not content:
        return [], 'no entry-content div'

    # Try each format in order of specificity
    clues = parse_fts_table(content)
    fmt = 'fts_table'

    if not clues:
        clues = parse_fts_list(content)
        fmt = 'fts_list'

    if not clues:
        clues = parse_plain_table(content)
        fmt = 'plain_table'

    if not clues:
        clues = parse_two_col_table(content)
        fmt = 'two_col_table'

    if not clues:
        clues = parse_paragraph_format(content)
        fmt = 'paragraph'

    if not clues:
        return [], 'no_format_matched'

    # Filter out cross-references
    filtered = [c for c in clues if not _is_cross_reference(c)]
    return filtered, fmt


# ============================================================
# Validation helpers
# ============================================================

def get_db_clue_count(source, puzzle_number):
    """Get clue count from clues_master.db for comparison."""
    try:
        conn = sqlite3.connect(MASTER_DB)
        row = conn.execute(
            "SELECT COUNT(*) FROM clues WHERE source = ? AND puzzle_number = ?",
            (source, str(puzzle_number)),
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return -1


def get_db_date_clue_count(source, pub_date):
    """Get clue count by date from clues_master.db."""
    try:
        conn = sqlite3.connect(MASTER_DB)
        row = conn.execute(
            "SELECT COUNT(*) FROM clues WHERE source = ? AND publication_date = ?",
            (source, pub_date),
        ).fetchone()
        conn.close()
        return row[0] if row else 0
    except Exception:
        return -1


def get_db_answers(source, puzzle_number):
    """Get all answers for a puzzle from clues_master.db."""
    try:
        conn = sqlite3.connect(MASTER_DB)
        rows = conn.execute(
            "SELECT UPPER(REPLACE(REPLACE(answer, ' ', ''), '-', '')) FROM clues WHERE source = ? AND puzzle_number = ?",
            (source, str(puzzle_number)),
        ).fetchall()
        conn.close()
        return {r[0] for r in rows if r[0]}
    except Exception:
        return set()


# ============================================================
# Test runner
# ============================================================

def test_puzzle(puzzle_number, source, pub_date=None, verbose=True):
    """Test URL discovery + parsing for a single puzzle. Returns success bool."""
    print(f'\n{"=" * 70}')
    print(f'TEST: {source} #{puzzle_number}' + (f' ({pub_date})' if pub_date else ''))
    print('=' * 70)

    # Step 1: URL discovery
    print('\n[1] URL Discovery...')
    url, title, method = discover_post_url(puzzle_number, source, pub_date)

    if not url:
        print('  FAIL: could not find post URL')
        return False

    print(f'  OK: found via {method}')
    print(f'  URL:   {url}')
    print(f'  Title: {title}')

    # Step 2: Fetch page
    print('\n[2] Fetching page...')
    try:
        resp = requests.get(url, headers=HEADERS, timeout=15)
        if resp.status_code != 200:
            print(f'  FAIL: HTTP {resp.status_code}')
            return False
        print(f'  OK: {len(resp.text)} bytes')
    except Exception as e:
        print(f'  FAIL: {e}')
        return False

    # Step 3: Parse HTML
    print('\n[3] Parsing HTML...')
    clues, fmt = parse_post(resp.text)

    if not clues:
        print(f'  FAIL: parsed 0 clues (format: {fmt})')
        return False

    print(f'  OK: {len(clues)} clues parsed (format: {fmt})')

    # Step 4: Validate fields
    print('\n[4] Field validation...')
    issues = []
    has_clue_text = 0
    has_definition = 0
    has_explanation = 0
    across_count = 0
    down_count = 0

    for c in clues:
        if c['clue_text']:
            has_clue_text += 1
        if c['definition']:
            has_definition += 1
        if c['explanation']:
            has_explanation += 1
        if c['direction'] == 'across':
            across_count += 1
        else:
            down_count += 1

        # Check answer is valid
        if not is_answer(c['answer']):
            issues.append(f'  bad answer: {c["clue_number"]} = {c["answer"]!r}')

    print(f'  Across: {across_count}, Down: {down_count}')
    print(f'  With clue_text:   {has_clue_text}/{len(clues)}')
    print(f'  With definition:  {has_definition}/{len(clues)}')
    print(f'  With explanation: {has_explanation}/{len(clues)}')

    if issues:
        for issue in issues[:5]:
            print(issue)

    # Step 5: Cross-check against DB
    print('\n[5] DB cross-check...')
    db_count = get_db_clue_count(source, puzzle_number)
    if db_count > 0:
        print(f'  DB has {db_count} clues for {source} #{puzzle_number}')
        match_pct = round(100 * len(clues) / db_count) if db_count else 0
        print(f'  Blog has {len(clues)} clues ({match_pct}% of DB)')

        # Check answer overlap
        db_answers = get_db_answers(source, puzzle_number)
        blog_answers = {re.sub(r'[^A-Z]', '', c['answer']) for c in clues}
        overlap = db_answers & blog_answers
        blog_only = blog_answers - db_answers
        db_only = db_answers - blog_answers
        print(f'  Answer overlap: {len(overlap)}/{len(db_answers)} DB answers matched')
        if blog_only:
            print(f'  Blog-only answers: {blog_only}')
        if db_only and len(db_only) <= 5:
            print(f'  DB-only answers: {db_only}')
    elif pub_date:
        db_count = get_db_date_clue_count(source, pub_date)
        if db_count > 0:
            print(f'  DB has {db_count} clues for {source} on {pub_date} (no puzzle_number)')
        else:
            print(f'  No DB clues found for {source} #{puzzle_number} or date {pub_date}')
    else:
        print(f'  No DB clues found for {source} #{puzzle_number}')

    # Print sample clues
    if verbose:
        print('\n[6] Sample clues:')
        for c in clues[:4]:
            print(f'  {c["clue_number"]:5} {c["answer"]:20} def={c["definition"][:30]!r:32} expl={c["explanation"][:50]!r}')
            if c['clue_text']:
                print(f'        clue: {c["clue_text"][:70]!r}')

    # Overall verdict
    ok = len(clues) >= 10 and has_explanation >= len(clues) * 0.8
    print(f'\n  VERDICT: {"PASS" if ok else "FAIL"}')
    return ok


def test_wp_api_pagination(source, max_pages=3):
    """Test WP API pagination for building a post index."""
    cat_id = CATEGORY_IDS[source]
    print(f'\n{"=" * 70}')
    print(f'WP API PAGINATION TEST: {source} (category {cat_id})')
    print('=' * 70)

    total_posts = 0
    total_with_number = 0
    puzzle_numbers = []

    for page in range(1, max_pages + 1):
        print(f'\n  Page {page}...')
        try:
            resp = requests.get(WP_API, headers=HEADERS, timeout=15, params={
                'categories': str(cat_id),
                'per_page': 20,
                'page': page,
            })
            if resp.status_code != 200:
                print(f'  HTTP {resp.status_code}')
                break

            # Report total from headers
            if page == 1:
                total_header = resp.headers.get('X-WP-Total', '?')
                pages_header = resp.headers.get('X-WP-TotalPages', '?')
                print(f'  Total posts: {total_header}, Total pages: {pages_header}')

            posts = resp.json()
            if not posts:
                break

            for post in posts:
                total_posts += 1
                title = post.get('title', {}).get('rendered', '')
                slug = post.get('slug', '')
                link = post.get('link', '')
                date = post.get('date', '')[:10]

                # Extract puzzle number from title
                title_clean = title.replace(',', '').replace('.', '').replace('&#8211;', '-')
                pnum_match = re.search(r'(\d{4,5})', title_clean)
                pnum = pnum_match.group(1) if pnum_match else None

                if pnum:
                    total_with_number += 1
                    puzzle_numbers.append(pnum)

                print(f'    {date} | #{pnum or "???":>6} | {title[:60]}')

        except Exception as e:
            print(f'  Error: {e}')
            break

        time.sleep(RATE_LIMIT)

    print(f'\n  Summary: {total_posts} posts scanned, {total_with_number} with puzzle number extracted')
    if puzzle_numbers:
        print(f'  Number range: {min(puzzle_numbers)} - {max(puzzle_numbers)}')
    return total_with_number > 0


def main():
    parser = argparse.ArgumentParser(description='Test fifteensquared URL discovery + parsing')
    parser.add_argument('--puzzle', type=int, help='Single puzzle number to test')
    parser.add_argument('--source', choices=['guardian', 'independent'], help='Source for single puzzle')
    parser.add_argument('--date', help='Publication date YYYY-MM-DD (helps URL discovery)')
    parser.add_argument('--pagination', action='store_true', help='Test WP API pagination instead')
    parser.add_argument('--quick', action='store_true', help='Only run 2 test puzzles')
    args = parser.parse_args()

    if args.pagination:
        for src in (['guardian', 'independent'] if not args.source else [args.source]):
            test_wp_api_pagination(src)
        return

    if args.puzzle and args.source:
        test_puzzle(args.puzzle, args.source, args.date)
        return

    # Default: test matrix of known puzzles
    test_cases = [
        # (puzzle_number, source, pub_date, description)
        (29958, 'guardian', '2026-03-19', 'Guardian Enigmatist (today)'),
        (29957, 'guardian', '2026-03-18', 'Guardian Brummie (yesterday)'),
        (29956, 'guardian', '2026-03-17', 'Guardian Alia'),
        (12307, 'independent', '2026-03-19', 'Independent Dalibor (today)'),
        (12304, 'independent', '2026-03-16', 'Independent Kairos'),
        (12305, 'independent', '2026-03-17', 'Independent Mog'),
    ]

    if args.quick:
        test_cases = test_cases[:2]

    results = []
    for pnum, source, pub_date, desc in test_cases:
        print(f'\n\n{"#" * 70}')
        print(f'# {desc}')
        print(f'{"#" * 70}')
        ok = test_puzzle(pnum, source, pub_date)
        results.append((desc, ok))
        time.sleep(RATE_LIMIT)

    # Also test WP API pagination
    print(f'\n\n{"#" * 70}')
    print(f'# WP API Pagination')
    print(f'{"#" * 70}')
    for src in ['guardian', 'independent']:
        test_wp_api_pagination(src, max_pages=2)
        time.sleep(RATE_LIMIT)

    # Final summary
    print(f'\n\n{"=" * 70}')
    print('FINAL SUMMARY')
    print('=' * 70)
    passed = sum(1 for _, ok in results if ok)
    total = len(results)
    for desc, ok in results:
        status = 'PASS' if ok else 'FAIL'
        print(f'  [{status}] {desc}')
    print(f'\n  {passed}/{total} puzzles passed')


if __name__ == '__main__':
    main()
