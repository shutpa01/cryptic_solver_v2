"""Re-parse the fifteensquared Indy 12352 blog and update DB with real per-clue explanations.

The scraper returned only the answer for each clue (single_para format quirk).
This script fetches the page directly, parses the inline format used by Kairos
(number, clue, ANSWER, explanation, ... in one continuous paragraph), and
overwrites the answer-only stubs in clues.explanation.
"""
import re
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent

sys.path.insert(0, str(ROOT / 'scraper' / 'fifteensquared'))
from test_fifteensquared import HEADERS  # type: ignore
import requests

URL = 'https://www.fifteensquared.net/2026/05/11/independent-12352-kairos/'

resp = requests.get(URL, headers=HEADERS, timeout=15)
html = resp.text

# Extract entry-content body
m = re.search(r'<div[^>]*class="[^"]*entry-content[^"]*"[^>]*>(.+?)<div[^>]*class="[^"]*entry-meta',
              html, re.DOTALL)
if not m:
    m = re.search(r'<div[^>]*class="[^"]*entry-content[^"]*"[^>]*>(.+)', html, re.DOTALL)
body_html = m.group(1)

# Decode common entities
body = body_html.replace('&#8217;', "'").replace('&#8216;', "'") \
                .replace('&#8220;', '"').replace('&#8221;', '"') \
                .replace('&nbsp;', ' ').replace('&amp;', '&')
# Strip tags but preserve text spacing
body = re.sub(r'<br[^>]*>', '\n', body)
body = re.sub(r'</p>', '\n', body)
body = re.sub(r'<[^>]+>', '', body)
body = re.sub(r'[ \t]+', ' ', body)

# Connect to DB for answer lookup
conn = sqlite3.connect(str(ROOT / 'data' / 'clues_master.db'))
clues_rows = conn.execute("""
    SELECT id, clue_number, direction, answer
    FROM clues WHERE source='independent' AND puzzle_number=12352
    ORDER BY direction, CAST(clue_number AS INTEGER)
""").fetchall()

# Build map: (number_int, direction_letter) -> (id, answer)
clues_map = {}
for cid, num, dirn, ans in clues_rows:
    clues_map[(int(num), dirn[0].lower())] = (cid, ans)

# Locate Across and Down sections
across_idx = body.find('Across')
down_idx = body.find('Down', across_idx + 1)
end_idx = re.search(r'Categories\b|Many thanks', body[down_idx:])
end_idx = down_idx + (end_idx.start() if end_idx else len(body) - down_idx)

across_text = body[across_idx + len('Across'):down_idx]
down_text = body[down_idx + len('Down'):end_idx]


def parse_section(text, direction):
    """Parse 'NN clue_text ANSWER explanation NN clue_text ANSWER ...' inline.

    Strategy: find each clue number boundary by matching against the
    known clue number list for the puzzle's direction. Boundaries are
    ints in increasing order. Slice between boundaries.
    """
    nums_in_dir = sorted(
        n for (n, d) in clues_map if d == direction
    )
    # Find each number's start position. Pattern: " NN " preceded by space or start.
    positions = []
    cursor = 0
    for n in nums_in_dir:
        pat = re.compile(rf'(?<!\d){n}(?!\d)')
        m = pat.search(text, cursor)
        if m:
            positions.append((n, m.start(), m.end()))
            cursor = m.end()
        else:
            positions.append((n, None, None))

    results = []
    for i, (n, start, end) in enumerate(positions):
        if start is None:
            results.append((n, None))
            continue
        next_start = positions[i + 1][1] if i + 1 < len(positions) else len(text)
        chunk = text[end:next_start].strip()
        results.append((n, chunk))
    return results


def split_clue_chunk(chunk, answer_letters):
    """Given a clue chunk 'clue_text ANSWER explanation', return explanation.
    The answer appears in UPPERCASE (with spaces). Find its boundary and slice."""
    if not chunk:
        return None
    # Normalize answer to a regex that allows space and hyphen between letters
    spaced = r'\s*'.join(re.escape(c) for c in answer_letters)
    m = re.search(r'\b(' + spaced + r')\b', chunk)
    if not m:
        return None
    expl = chunk[m.end():].strip()
    return expl


print(f"Across letters: {len(across_text)} chars")
print(f"Down letters: {len(down_text)} chars\n")

across = parse_section(across_text, 'a')
down = parse_section(down_text, 'd')

updates = []
for direction_letter, parsed_list in (('a', across), ('d', down)):
    for n, chunk in parsed_list:
        key = (n, direction_letter)
        if key not in clues_map:
            continue
        cid, ans = clues_map[key]
        ans_letters = re.sub(r'[^A-Z]', '', ans.upper())
        expl = split_clue_chunk(chunk, ans_letters)
        if expl:
            updates.append((cid, n, direction_letter, ans, expl[:400]))
        else:
            print(f"  PARSE-MISS {n}{direction_letter} {ans}: chunk[:200] = {(chunk or '')[:200]!r}")

print(f"\nParsed {len(updates)} explanations.\n")
for cid, n, d, ans, expl in updates:
    print(f"  {n}{d} {ans:18}: {expl[:160]}")

# Apply updates: overwrite the answer-only stubs
print()
applied = 0
for cid, n, d, ans, expl in updates:
    conn.execute("UPDATE clues SET explanation = ? WHERE id = ?", (expl, cid))
    applied += 1
conn.commit()
conn.close()
print(f"\nUpdated {applied} rows in clues.explanation.")
