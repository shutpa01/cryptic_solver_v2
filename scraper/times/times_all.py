#!/usr/bin/env python3
"""The Times Cryptic Crossword Scraper

Uses undetected-chromedriver to log in and discover API ID from network traffic.
No longer relies on offset calculations - captures the real API URL.

Requires .env file with:
TIMES_EMAIL=your_email
TIMES_PASSWORD=your_password
"""

import requests
import json
import sqlite3
import os
import sys
import html
import time
import re
from datetime import datetime, date, timedelta
from pathlib import Path
from dotenv import load_dotenv

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")

TIMES_EMAIL = os.getenv('TIMES_EMAIL')
TIMES_PASSWORD = os.getenv('TIMES_PASSWORD')

# Puzzle type configurations
PUZZLE_TYPES = {
    'cryptic': {
        'feed_type': 'crosswordcryptic',
        'name': 'Times Cryptic',
        'days': [0, 1, 2, 3, 4, 5],  # Mon-Sat
        'page_url': 'https://www.thetimes.com/puzzles/crossword'
    },
    'quick-cryptic': {
        'feed_type': 'crosswordquickcryptic',
        'name': 'Times Quick Cryptic',
        'days': [0, 1, 2, 3, 4, 5],  # Mon-Sat
        'page_url': 'https://www.thetimes.com/puzzles/crossword'
    },
    'sunday-cryptic': {
        'feed_type': 'crosswordcryptic',  # Same feed type, different ID series
        'name': 'Sunday Times Cryptic',
        'days': [6],  # Sunday only
        'page_url': 'https://www.thetimes.com/puzzles/crossword'
    }
}

BASE_FEED_URL = "https://feeds.thetimes.com/puzzles/sp"
LOGIN_URL = "https://login.thetimes.co.uk/"


def estimate_puzzle_number(puzzle_type, target_date=None):
    """Estimate puzzle number based on reference data."""
    if target_date is None:
        target_date = date.today()

    if puzzle_type == 'cryptic' or puzzle_type == 'quick-cryptic':
        # Reference: puzzle 29449 on Mon Jan 26, 2026
        ref_date = date(2026, 1, 26)
        ref_number = 29449

        # Count publishing days (Mon-Sat, no Sunday)
        days = 0
        current = ref_date
        while current < target_date:
            current += timedelta(days=1)
            if current.weekday() != 6:  # Not Sunday
                days += 1
        while current > target_date:
            current -= timedelta(days=1)
            if current.weekday() != 6:
                days -= 1

        return ref_number + days

    elif puzzle_type == 'sunday-cryptic':
        # Reference: puzzle 5200 on Sun Jan 26, 2026
        ref_date = date(2026, 1, 26)
        ref_number = 5200

        # Count Sundays
        weeks_diff = (target_date - ref_date).days // 7
        return ref_number + weeks_diff

    return None


def get_puzzle_via_network_capture(puzzle_type='cryptic'):
    """
    Log in and capture the API URL from network traffic.
    Returns the puzzle JSON data, puzzle number, and API ID.
    """
    if not TIMES_EMAIL or not TIMES_PASSWORD:
        raise ValueError("Missing TIMES_EMAIL or TIMES_PASSWORD in .env file")

    print("Launching browser with network logging enabled...")

    config = PUZZLE_TYPES.get(puzzle_type, PUZZLE_TYPES['cryptic'])

    # Enable performance logging to capture network requests
    options = uc.ChromeOptions()
    options.add_argument('--start-maximized')
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    driver = uc.Chrome(options=options, version_main=144)

    api_id = None
    api_url = None
    json_data = None
    puzzle_number = None

    try:
        # === LOGIN ===
        print(f"Navigating to login page...")
        driver.get(LOGIN_URL)
        time.sleep(3)

        wait = WebDriverWait(driver, 15)

        # Check for Auth0 iframe
        try:
            iframe = wait.until(EC.presence_of_element_located(
                (By.CSS_SELECTOR, "iframe[name='auth0-lock-widget-frame']")))
            print("Switching to Auth0 iframe...")
            driver.switch_to.frame(iframe)
        except:
            print("No iframe found, continuing on main page...")

        print("Entering email...")
        email_field = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "input.auth0-lock-input[name='email']")))
        email_field.clear()
        email_field.send_keys(TIMES_EMAIL)

        print("Entering password...")
        password_field = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "input.auth0-lock-input[name='password']")))
        password_field.clear()
        password_field.send_keys(TIMES_PASSWORD)

        print("Clicking login...")
        login_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "button[type='submit']")))
        login_btn.click()

        print("Waiting for login to complete...")
        driver.switch_to.default_content()
        WebDriverWait(driver, 30).until(lambda d: "login" not in d.current_url.lower())
        print(f"Logged in! URL: {driver.current_url}")

        # === NAVIGATE TO PUZZLE ===
        puzzle_url = config['page_url']
        print(f"Navigating to crossword page: {puzzle_url}")
        driver.get(puzzle_url)
        time.sleep(5)

        # Calculate expected puzzle number
        expected_number = estimate_puzzle_number(puzzle_type)
        print(f"Looking for puzzle #{expected_number}...")

        # Find the link with the exact puzzle number
        links = driver.find_elements(By.TAG_NAME, "a")
        target_link = None

        for link in links:
            href = link.get_attribute("href") or ""

            if puzzle_type == 'sunday-cryptic':
                # Look for sunday-times-cryptic-no-XXXXX
                match = re.search(r'sunday-times-cryptic-no-(\d+)', href)
            else:
                # Look for times-cryptic-no-XXXXX but NOT sunday
                if 'sunday' in href.lower():
                    continue
                match = re.search(r'times-cryptic-no-(\d+)', href)

            if match:
                found_number = int(match.group(1))
                if found_number == expected_number:
                    target_link = link
                    puzzle_number = found_number
                    print(f"Found exact match: puzzle #{puzzle_number}")
                    break

        if not target_link:
            # Fallback: find highest puzzle number
            print(f"Exact match not found, looking for highest number...")
            candidates = []
            for link in links:
                href = link.get_attribute("href") or ""
                if puzzle_type == 'sunday-cryptic':
                    match = re.search(r'sunday-times-cryptic-no-(\d+)', href)
                else:
                    if 'sunday' in href.lower():
                        continue
                    match = re.search(r'times-cryptic-no-(\d+)', href)
                if match:
                    candidates.append((int(match.group(1)), link))

            if candidates:
                candidates.sort(key=lambda x: x[0], reverse=True)
                puzzle_number = candidates[0][0]
                target_link = candidates[0][1]
                print(f"Using highest number: puzzle #{puzzle_number}")

        if not target_link:
            print("ERROR: Could not find puzzle link!")
            driver.quit()
            return None, None, None

        feed_type = config['feed_type']

        print(f"Clicking puzzle #{puzzle_number}...")
        target_link.click()

        # Wait for puzzle to load
        print("Waiting for puzzle to load...")
        time.sleep(10)

        print(f"Current URL: {driver.current_url}")

        # Extract puzzle number from URL
        url_match = re.search(r'no-(\d+)', driver.current_url)
        if url_match:
            puzzle_number = int(url_match.group(1))
            print(f"Puzzle number from URL: {puzzle_number}")

        # === CAPTURE API URL FROM PUZZLE IFRAME ===
        # The Times embeds the puzzle in an iframe:
        #   <iframe id="puzzle-iframe" src="https://feeds.thetimes.com/puzzles/sp/{feed_type}/{date}/{id}/?...">
        # Reading the iframe src is simpler and more reliable than network interception.
        print("Looking for puzzle iframe...")
        try:
            iframes = driver.find_elements(By.TAG_NAME, "iframe")
            for iframe in iframes:
                src = iframe.get_attribute("src") or ""
                if "feeds.thetimes" in src and feed_type in src:
                    # Strip query string, append data.json
                    base = src.split("?")[0].rstrip("/")
                    api_url = base + "/data.json"
                    print(f"Found iframe src: {src[:80]}")
                    print(f"Constructed API URL: {api_url}")
                    # Extract API ID: .../feed_type/YYYYMMDD/ID/data.json
                    match = re.search(rf'{feed_type}/(\d+)/(\d+)', api_url)
                    if match:
                        api_id = match.group(2)
                        print(f"Extracted API ID: {api_id}")
                    break
        except Exception as e:
            print(f"  Iframe search failed: {e}")

        if not api_url:
            # Fallback: search page source
            print("Searching page source for API URL...")
            page_source = driver.page_source
            match = re.search(
                r'(https://feeds\.thetimes\.(?:com|co\.uk)/puzzles/sp/\w+/\d{8}/\d+/data\.json)',
                page_source
            )
            if match:
                api_url = match.group(1)
                print(f"Found API URL in page source: {api_url}")
                # Extract API ID
                id_match = re.search(rf'{feed_type}/(\d+)/(\d+)/data\.json', api_url)
                if id_match:
                    api_id = id_match.group(2)

        if not api_url:
            print("WARNING: Could not find API URL in network logs or page source")
            print("Saving page for debugging...")
            debug_file = Path(__file__).parent / "times_page_debug.html"
            with open(debug_file, 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
            print(f"Saved to {debug_file}")
            driver.quit()
            return None, puzzle_number, api_id

        # === FETCH PUZZLE DATA ===
        print(f"Fetching puzzle data from API...")

        # Get cookies from browser session
        cookies = {c['name']: c['value'] for c in driver.get_cookies()}

        headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json',
            'Referer': 'https://www.thetimes.com/',
        }

        response = requests.get(api_url, headers=headers, cookies=cookies, timeout=30)

        if response.status_code == 200:
            json_data = response.json()
            print("Successfully fetched puzzle data!")
        else:
            print(f"Failed to fetch puzzle data: {response.status_code}")
            # Try without cookies (the API might be public)
            response = requests.get(api_url, headers=headers, timeout=30)
            if response.status_code == 200:
                json_data = response.json()
                print("Fetched puzzle data (no cookies needed)")

        driver.quit()
        print("Browser closed.")

        return json_data, puzzle_number, api_id

    except Exception as e:
        print(f"Error: {e}")
        screenshot_file = Path(__file__).parent / "times_error.png"
        try:
            driver.save_screenshot(str(screenshot_file))
            print(f"Saved screenshot to {screenshot_file}")
        except:
            pass
        driver.quit()
        raise


def parse_puzzle(json_data):
    """Parse The Times puzzle JSON."""
    data = json_data.get('data', {})
    copy = data.get('copy', {})
    meta = data.get('meta', {})

    puzzle_number = meta.get('number') or copy.get('id')
    title = copy.get('title', '')
    setter = copy.get('setter', '')
    puzzle_date_raw = copy.get('date-publish', '')
    # Convert to ISO date
    if puzzle_date_raw and ',' in puzzle_date_raw:
        try:
            puzzle_date = datetime.strptime(puzzle_date_raw, '%A, %d %B %Y').strftime('%Y-%m-%d')
        except ValueError:
            puzzle_date = puzzle_date_raw
    else:
        puzzle_date = puzzle_date_raw
    puzzle_type_name = copy.get('crosswordtype', 'Times Cryptic')
    is_competition = data.get('competitioncrossword', 0) == 1

    print(f"Title: {title}")
    print(f"Puzzle #: {puzzle_number}")
    print(f"Date: {puzzle_date}")
    print(f"Competition: {is_competition}")

    # Build word_id -> solution mapping
    word_solutions = {}
    for word in copy.get('words', []):
        word_id = word.get('id')
        solution = word.get('solution', '')
        if word_id and solution:
            word_solutions[word_id] = solution.upper()

    # Parse clues
    across_clues = []
    down_clues = []
    answers_found = 0
    total_clues = 0

    clue_sections = copy.get('clues', [])
    for section in clue_sections:
        section_title = section.get('title', '').lower()
        is_across = 'across' in section_title

        for clue in section.get('clues', []):
            clue_text = html.unescape(clue.get('clue', ''))
            word_id = clue.get('word')
            answer = word_solutions.get(word_id, '')

            clue_obj = {
                'number': clue.get('number'),
                'clue': clue_text,
                'enumeration': clue.get('format', ''),
                'length': clue.get('length'),
                'answer': answer
            }

            if answer:
                answers_found += 1
            total_clues += 1

            if is_across:
                across_clues.append(clue_obj)
            else:
                down_clues.append(clue_obj)

    print(f"Clues: {len(across_clues)} across, {len(down_clues)} down")
    print(f"Answers: {answers_found}/{total_clues}")

    if is_competition and answers_found == 0:
        print("Note: Prize puzzle - answers hidden until competition closes")

    return {
        'puzzle_number': puzzle_number,
        'puzzle_type': puzzle_type_name,
        'title': title,
        'setter': setter,
        'date': puzzle_date,
        'is_competition': is_competition,
        'across': across_clues,
        'down': down_clues
    }


def puzzle_exists(puzzle_number, puzzle_type):
    """Check if puzzle already in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS times_clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_type TEXT,
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
        SELECT COUNT(*) FROM times_clues 
        WHERE puzzle_number = ? AND puzzle_type LIKE ?
    """, (str(puzzle_number), f'%{puzzle_type}%'))

    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def save_to_database(puzzle_data):
    """Save puzzle clues to database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS times_clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_type TEXT,
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
        CREATE INDEX IF NOT EXISTS idx_times_puzzle 
        ON times_clues(puzzle_number, puzzle_type)
    """)

    puzzle_number = puzzle_data.get('puzzle_number', '')
    puzzle_type = puzzle_data.get('puzzle_type', 'Times Cryptic')
    puzzle_date = puzzle_data.get('date', '')
    setter = puzzle_data.get('setter', '')
    fetched_at = datetime.now().isoformat()

    clue_count = 0
    for direction in ['across', 'down']:
        for clue in puzzle_data.get(direction, []):
            cursor.execute("""
                INSERT INTO times_clues 
                (puzzle_type, puzzle_number, puzzle_date, setter, clue_number, direction, 
                 clue_text, enumeration, answer, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                puzzle_type,
                str(puzzle_number),
                puzzle_date,
                setter,
                str(clue.get('number', '')),
                direction,
                clue.get('clue', ''),
                clue.get('enumeration', ''),
                clue.get('answer', ''),
                fetched_at
            ))
            clue_count += 1

    conn.commit()
    conn.close()

    print(f"Saved {clue_count} clues to database")
    return clue_count


def main():
    print("=" * 60)
    print("THE TIMES CROSSWORD SCRAPER")
    print("(Network capture mode - no offset calculations)")
    print("=" * 60)

    today = date.today()
    print(f"Today: {today.strftime('%A, %d %B %Y')}")

    # Parse arguments
    force = '--force' in sys.argv

    # Get puzzle type
    puzzle_type = None
    for arg in sys.argv[1:]:
        if arg in PUZZLE_TYPES:
            puzzle_type = arg
            break

    # Auto-detect based on day of week if not specified
    if puzzle_type is None:
        if today.weekday() == 6:  # Sunday
            puzzle_type = 'sunday-cryptic'
        else:
            puzzle_type = 'cryptic'

    config = PUZZLE_TYPES[puzzle_type]
    print(f"Puzzle type: {config['name']}")

    # Get puzzle via network capture
    print("\n" + "-" * 40)
    json_data, puzzle_number, api_id = get_puzzle_via_network_capture(puzzle_type)
    print("-" * 40 + "\n")

    if not json_data:
        print("Failed to fetch puzzle data")
        return

    if puzzle_number:
        print(f"Puzzle number: {puzzle_number}")
    if api_id:
        print(f"API ID: {api_id}")

    # Check if already fetched
    if not force and puzzle_exists(puzzle_number, puzzle_type):
        print(f"\nPuzzle {puzzle_number} already in database. Use --force to re-fetch.")
        return

    # Parse
    puzzle_data = parse_puzzle(json_data)

    if not puzzle_data.get('puzzle_number'):
        print("Failed to parse puzzle")
        return

    # Use URL puzzle number if parse didn't find one
    if not puzzle_data['puzzle_number'] and puzzle_number:
        puzzle_data['puzzle_number'] = puzzle_number

    # Delete existing if force mode
    if force and puzzle_exists(puzzle_number, puzzle_type):
        print(f"\nRemoving existing puzzle {puzzle_number}...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM times_clues 
            WHERE puzzle_number = ? AND puzzle_type LIKE ?
        """, (str(puzzle_number), f'%{puzzle_type}%'))
        conn.commit()
        conn.close()

    # Save
    print(f"\nSaving to database...")
    save_to_database(puzzle_data)

    # Save JSON backup
    script_dir = Path(__file__).parent
    json_path = script_dir / f"times_{puzzle_type}_{puzzle_number}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON backup: {json_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()