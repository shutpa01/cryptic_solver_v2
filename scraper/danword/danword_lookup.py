#!/usr/bin/env python3
"""Danword Answer Lookup for Prize Puzzles

Looks up crossword clue answers on danword.com using Selenium + Google Custom Search.
Validates answers against the puzzle grid, then writes to the clues DB.

Usage:
    python -m scraper.danword.danword_lookup --source telegraph --puzzle 3358
    python -m scraper.danword.danword_lookup --source telegraph --puzzle 3358 --dry-run
"""

import argparse
import io
import json
import os
import random
import re
import sqlite3
import subprocess
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

# Fix Windows console encoding
if sys.stdout.encoding != 'utf-8':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DB_PATH = os.getenv('DB_PATH', str(PROJECT_ROOT / 'data' / 'clues_master.db'))
TELEGRAPH_JSON_DIR = PROJECT_ROOT / 'scraper' / 'telegraph'
TIMES_JSON_DIR = PROJECT_ROOT / 'scraper' / 'times'

DANWORD_URL = 'https://www.danword.com'


def get_chrome_version_main():
    """Read installed Chrome major version from Windows registry."""
    try:
        result = subprocess.run(
            ['reg', 'query', r'HKEY_CURRENT_USER\Software\Google\Chrome\BLBeacon', '/v', 'version'],
            capture_output=True, text=True, timeout=5,
        )
        match = re.search(r'(\d+)\.', result.stdout)
        return int(match.group(1)) if match else None
    except Exception:
        return None


# ---------- DB helpers --------------------------------------------------

def get_answerless_clues(source, puzzle_number):
    """Fetch clues without answers for a puzzle. Returns list of dicts."""
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()
    cur.execute("""
        SELECT id, clue_number, direction, clue_text, enumeration
        FROM clues
        WHERE source = ? AND puzzle_number = ?
          AND (answer IS NULL OR answer = '')
        ORDER BY direction, CAST(clue_number AS INTEGER)
    """, (source, str(puzzle_number)))
    rows = [dict(r) for r in cur.fetchall()]
    conn.close()
    return rows


def write_answers(answers):
    """Write answers to DB. answers = list of (answer, clue_id)."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    for answer, clue_id in answers:
        cur.execute("UPDATE clues SET answer = ? WHERE id = ?", (answer, clue_id))
    conn.commit()
    conn.close()


# ---------- Danword Selenium lookup ------------------------------------

def setup_driver():
    """Launch Chrome for danword scraping."""
    import undetected_chromedriver as uc

    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    chrome_ver = get_chrome_version_main()
    print(f"Chrome version: {chrome_ver}")
    driver = uc.Chrome(options=options, version_main=chrome_ver)
    return driver


def lookup_clue(driver, clue_text):
    """Search danword for a clue and return the answer (or None).

    Uses the Google Custom Search widget embedded on danword.com.
    """
    from selenium.webdriver.common.by import By
    from selenium.webdriver.common.keys import Keys
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

    # Always navigate fresh to danword home — prevents stale CSE widget issues
    driver.get(DANWORD_URL)
    time.sleep(3)

    # Find the Google CSE search input
    try:
        wait = WebDriverWait(driver, 10)
        search_input = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "input.gsc-input")))
    except Exception:
        try:
            search_input = driver.find_element(By.CSS_SELECTOR, "input[type='text']")
        except Exception:
            print("    Could not find search input")
            return None

    # Type the clue and search
    search_input.clear()
    time.sleep(0.3)
    search_input.send_keys(clue_text)
    time.sleep(0.5)
    search_input.send_keys(Keys.RETURN)

    # Wait for Google CSE results to load
    time.sleep(4)

    # Find result links — Google CSE renders results with .gs-title class
    result_links = []
    for selector in ["a.gs-title", ".gsc-results a[href*='danword.com/crossword/']"]:
        try:
            result_links = driver.find_elements(By.CSS_SELECTOR, selector)
            if result_links:
                break
        except Exception:
            continue

    if not result_links:
        print("    No search results found")
        return None

    # Navigate to the first result's URL directly (avoids click issues)
    try:
        href = result_links[0].get_attribute('href') or result_links[0].get_attribute('data-ctorig')
        if href and 'danword.com' in href:
            driver.get(href)
        else:
            result_links[0].click()
    except Exception:
        try:
            driver.execute_script("arguments[0].click();", result_links[0])
        except Exception:
            print("    Could not navigate to result")
            return None

    time.sleep(2)

    # Extract answer from the page
    return extract_answer_from_page(driver)


def extract_answer_from_page(driver):
    """Parse the answer from a danword answer page."""
    from selenium.webdriver.common.by import By

    try:
        # Answer is in <ul id="answerDisplay"> with each letter in <li class="box">
        boxes = driver.find_elements(By.CSS_SELECTOR, "#answerDisplay .box")
        if boxes:
            answer = ''.join(box.text.strip() for box in boxes)
            return answer.upper() if answer else None

        # Fallback: parse from page source
        source = driver.page_source
        matches = re.findall(r'<li class="box">([A-Za-z])</li>', source)
        if matches:
            return ''.join(matches).upper()

    except Exception as e:
        print(f"    Error extracting answer: {e}")

    return None


def parse_enum_length(enumeration):
    """Parse enumeration string to total letter count. '(5,3)' -> 8, '(9)' -> 9."""
    if not enumeration:
        return 0
    digits = re.findall(r'\d+', enumeration)
    return sum(int(d) for d in digits)


# ---------- Grid validation --------------------------------------------

def find_puzzle_json(source, puzzle_number):
    """Find the saved puzzle JSON file for grid validation."""
    if source == 'telegraph':
        # Try various filename patterns
        for pattern in [
            f"telegraph_prize-cryptic_{puzzle_number}.json",
            f"telegraph_prize-toughie_{puzzle_number}.json",
        ]:
            path = TELEGRAPH_JSON_DIR / pattern
            if path.exists():
                return path

        # Also try matching by puzzle number in title
        for f in TELEGRAPH_JSON_DIR.glob("telegraph_prize-*.json"):
            try:
                data = json.loads(f.read_text())
                copy = data.get('json', {}).get('copy', {})
                title = copy.get('title', '')
                if str(puzzle_number) in title:
                    return f
            except Exception:
                continue

    elif source == 'times':
        # Times cryptic JSON files
        for pattern in [
            f"times_cryptic_{puzzle_number}.json",
            f"times_sunday-cryptic_{puzzle_number}.json",
        ]:
            path = TIMES_JSON_DIR / pattern
            if path.exists():
                return path

    return None


def _parse_range(val):
    """Parse '3-7' -> (3, 7) or '5' -> (5, 5)."""
    s = str(val)
    if '-' in s:
        a, b = s.split('-', 1)
        return int(a), int(b)
    return int(s), int(s)


def _build_word_cells(words):
    """Build {word_id: [(row, col), ...]} from words array.

    Across words: x is a range (cols), y is a single row.
    Down words: x is a single col, y is a range (rows).
    """
    word_cells = {}
    for w in words:
        wid = w['id']
        x_lo, x_hi = _parse_range(w['x'])
        y_lo, y_hi = _parse_range(w['y'])
        cells = []
        if x_lo != x_hi:
            # Across word — y is fixed, x varies
            for col in range(x_lo, x_hi + 1):
                cells.append((y_lo, col))
        else:
            # Down word — x is fixed, y varies
            for row in range(y_lo, y_hi + 1):
                cells.append((row, x_lo))
        word_cells[wid] = cells
    return word_cells


def _build_clue_to_word(clues_sections):
    """Map (clue_number_str, direction) -> word_id from clues array."""
    mapping = {}
    for section in clues_sections:
        title = section.get('title', '').lower()
        direction = 'across' if 'across' in title else 'down'
        for clue in section.get('clues', []):
            num = str(clue['number'])
            mapping[(num, direction)] = clue['word']
    return mapping


def validate_grid(json_path, clue_answers):
    """Validate answers using grid crossings. Works for both Telegraph and Times.

    Args:
        json_path: path to the puzzle JSON file
        clue_answers: dict of {(clue_number_str, direction): answer}

    Returns:
        (is_valid, error_count, total_crossings)
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    # Navigate to the copy object (different nesting for Telegraph vs Times)
    if 'json' in data:
        copy = data['json'].get('copy', {})
    elif 'data' in data:
        copy = data['data'].get('copy', {})
    else:
        copy = data.get('copy', {})

    words = copy.get('words', [])
    clues_sections = copy.get('clues', [])

    if not words or not clues_sections:
        return None, 0, 0  # Can't validate

    word_cells = _build_word_cells(words)
    clue_to_word = _build_clue_to_word(clues_sections)

    # Place answers into a virtual grid: {(row, col): {word_id: letter}}
    cell_letters = {}  # {(row, col): {word_id: letter}}
    placed = 0

    for (num, direction), answer in clue_answers.items():
        wid = clue_to_word.get((num, direction))
        if wid is None:
            continue
        cells = word_cells.get(wid, [])
        clean = re.sub(r'[^A-Za-z]', '', answer).upper()
        if len(clean) != len(cells):
            continue  # Length mismatch, skip
        placed += 1
        for i, (row, col) in enumerate(cells):
            if (row, col) not in cell_letters:
                cell_letters[(row, col)] = {}
            cell_letters[(row, col)][wid] = clean[i]

    # Check crossings — cells where two or more words overlap
    errors = 0
    total = 0
    error_details = []

    for (row, col), letters_by_word in cell_letters.items():
        if len(letters_by_word) < 2:
            continue
        total += 1
        unique_letters = set(letters_by_word.values())
        if len(unique_letters) > 1:
            errors += 1
            error_details.append(f"  ({row},{col}): {dict(letters_by_word)}")

    if error_details:
        for detail in error_details[:5]:
            print(detail)

    return errors == 0, errors, total


def build_solution_string(json_path, clue_answers):
    """Build a flat solution string from JSON grid structure + answers.

    Returns (solution_str, rows, cols) or None if not possible.
    The string has letters for white cells and spaces for black cells.
    """
    with open(json_path, encoding='utf-8') as f:
        data = json.load(f)

    if 'json' in data:
        copy = data['json'].get('copy', {})
        grid_array = data['json'].get('grid')
    elif 'data' in data:
        copy = data['data'].get('copy', {})
        grid_array = None
    else:
        copy = data.get('copy', {})
        grid_array = None

    gridsize = copy.get('gridsize', {})
    cols = int(gridsize.get('cols', 15))
    rows = int(gridsize.get('rows', 15))
    words = copy.get('words', [])
    clues_sections = copy.get('clues', [])

    if not words or not clues_sections:
        return None

    word_cells = _build_word_cells(words)
    clue_to_word = _build_clue_to_word(clues_sections)

    # Determine which cells are black vs white
    black_cells = set()
    white_cells = set()

    if grid_array:
        # Telegraph: use grid array to identify black cells
        for r_idx, row in enumerate(grid_array):
            for c_idx, cell in enumerate(row):
                if cell.get('Blank') == 'blank':
                    black_cells.add((r_idx + 1, c_idx + 1))  # 1-indexed
                else:
                    white_cells.add((r_idx + 1, c_idx + 1))
    else:
        # Times: infer from word positions
        for cells in word_cells.values():
            for cell in cells:
                white_cells.add(cell)

    # Place answers into grid
    grid_letters = {}
    for (num, direction), answer in clue_answers.items():
        wid = clue_to_word.get((num, direction))
        if wid is None:
            continue
        cells = word_cells.get(wid, [])
        clean = re.sub(r'[^A-Za-z]', '', answer).upper()
        if len(clean) != len(cells):
            continue
        for i, (row, col) in enumerate(cells):
            grid_letters[(row, col)] = clean[i]

    # Build solution string
    solution = []
    for r in range(1, rows + 1):
        for c in range(1, cols + 1):
            if (r, c) in grid_letters:
                solution.append(grid_letters[(r, c)])
            elif (r, c) in black_cells or (r, c) not in white_cells:
                solution.append(' ')
            else:
                solution.append(' ')  # White cell without a letter

    return ''.join(solution), rows, cols


def update_puzzle_grid_solution(source, puzzle_number, solution, rows, cols):
    """Update the puzzle_grids table with a solution string."""
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()
    cur.execute("""UPDATE puzzle_grids SET solution = ?, grid_rows = ?, grid_cols = ?
                   WHERE source = ? AND puzzle_number = ?""",
                (solution, rows, cols, source, str(puzzle_number)))
    if cur.rowcount == 0:
        cur.execute("""INSERT OR IGNORE INTO puzzle_grids (source, puzzle_number, solution, grid_rows, grid_cols)
                       VALUES (?, ?, ?, ?, ?)""",
                    (source, str(puzzle_number), solution, rows, cols))
    conn.commit()
    conn.close()


# ---------- Main orchestrator ------------------------------------------

def lookup_puzzle(source, puzzle_number, dry_run=False):
    """Look up all clues for a puzzle on danword and optionally write answers."""
    clues = get_answerless_clues(source, puzzle_number)
    if not clues:
        print(f"No answerless clues found for {source} #{puzzle_number}")
        return 0, 0

    print(f"\nLooking up {len(clues)} clues for {source} #{puzzle_number} on danword...")

    driver = setup_driver()
    found = {}  # {clue_id: (answer, clue_number, direction)}

    try:
        for i, clue in enumerate(clues):
            clue_text = clue['clue_text']
            enum = clue['enumeration']
            expected_len = parse_enum_length(enum)
            num = clue['clue_number']
            direction = clue['direction']
            clue_id = clue['id']

            d = 'A' if direction == 'across' else 'D'
            print(f"  [{i+1}/{len(clues)}] {num:>2}{d}: {clue_text[:60]}...", end='')

            answer = lookup_clue(driver, clue_text)

            if answer:
                # Validate length
                clean_answer = re.sub(r'[^A-Za-z]', '', answer).upper()
                if expected_len and len(clean_answer) != expected_len:
                    print(f" -> {answer} (WRONG LENGTH: {len(clean_answer)} != {expected_len})")
                else:
                    print(f" -> {clean_answer}")
                    found[clue_id] = (clean_answer, num, direction)
            else:
                print(" -> NOT FOUND")

            # Brief delay between searches (longer every 10 to avoid rate limiting)
            if i < len(clues) - 1:
                if (i + 1) % 10 == 0:
                    delay = random.uniform(3, 5)
                    print(f"    (longer pause: {delay:.0f}s)")
                else:
                    delay = random.uniform(1.5, 2.5)
                time.sleep(delay)

    finally:
        driver.quit()

    print(f"\nFound: {len(found)}/{len(clues)} answers")

    if not found:
        return 0, len(clues)

    # Grid validation
    json_path = find_puzzle_json(source, puzzle_number)
    grid_valid = None

    if json_path:
        print(f"\nValidating against grid: {json_path.name}")
        clue_answers = {(num, direction): answer for answer, num, direction in found.values()}
        is_valid, errors, total = validate_grid(str(json_path), clue_answers)
        grid_valid = is_valid
        if is_valid:
            print(f"  Grid validation PASSED ({total} crossings checked)")
            print(f"Grid: PASSED ({total} crossings)")
        else:
            print(f"  Grid validation FAILED: {errors}/{total} crossing errors")
            print(f"Grid: FAILED ({errors}/{total} crossings)")
    else:
        print(f"\nNo puzzle JSON found for grid validation")
        print(f"Grid: NO_JSON")

    # Write to DB
    if dry_run:
        print(f"\nDRY RUN: would write {len(found)} answers to DB")
    elif grid_valid is False:
        print(f"\nSkipping DB write — grid validation failed")
    else:
        tag = "(grid validated)" if grid_valid else "(no grid validation)"
        print(f"\nWriting {len(found)} answers to DB {tag}")
        write_answers([(answer, clue_id) for clue_id, (answer, _, _) in found.items()])

        # Build and store grid solution for the web app
        if json_path and grid_valid:
            result = build_solution_string(str(json_path), clue_answers)
            if result:
                sol, rows, cols = result
                update_puzzle_grid_solution(source, puzzle_number, sol, rows, cols)
                print(f"  Grid solution stored ({rows}x{cols})")

    return len(found), len(clues)


def main():
    parser = argparse.ArgumentParser(description='Look up prize puzzle answers on danword')
    parser.add_argument('--source', required=True, choices=['telegraph', 'times'],
                        help='Puzzle source')
    parser.add_argument('--puzzle', required=True, help='Puzzle number')
    parser.add_argument('--dry-run', action='store_true',
                        help='Find answers but do not write to DB')
    args = parser.parse_args()

    print(f"Danword Lookup -- {args.source} #{args.puzzle}")
    print(f"Database: {DB_PATH}")

    found, total = lookup_puzzle(args.source, args.puzzle, dry_run=args.dry_run)

    print(f"\nDone: {found}/{total} answers {'found' if args.dry_run else 'written'}")
    sys.exit(0 if found > 0 else 1)


if __name__ == '__main__':
    main()
