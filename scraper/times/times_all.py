#!/usr/bin/env python3
"""The Times Cryptic Crossword Scraper

Uses undetected-chromedriver to log in and discover API ID from network traffic.
No longer relies on offset calculations - captures the real API URL.
Saves clues directly to the clues table in clues_master.db.

Uses a SINGLE browser session for all puzzle types — login once, fetch all.

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
import subprocess
from datetime import datetime, date, timedelta
from pathlib import Path
from dotenv import load_dotenv

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()


def get_chrome_version_main() -> int | None:
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

SCRIPT_DIR = Path(__file__).parent
DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

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
    'sunday-cryptic': {
        'feed_type': 'crosswordcryptic',  # Same feed type, different ID series
        'name': 'Sunday Times Cryptic',
        'days': [6],  # Sunday only
        'page_url': 'https://www.thetimes.com/puzzles/crossword'
    }
}

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
        # Reference: puzzle 5205 on Sun Mar 1, 2026
        ref_date = date(2026, 3, 1)
        ref_number = 5205

        # Count Sundays
        weeks_diff = (target_date - ref_date).days // 7
        return ref_number + weeks_diff

    return None


# ── Browser lifecycle ─────────────────────────────────────────────────────

PROFILE_DIR = SCRIPT_DIR / '.chrome_profile'


def create_browser():
    """Launch Chrome with a persistent profile so session cookies survive between runs."""
    if not TIMES_EMAIL or not TIMES_PASSWORD:
        raise ValueError("Missing TIMES_EMAIL or TIMES_PASSWORD in .env file")

    # Clean up stale lock files from previous crashed sessions
    for lock_file in ['SingletonLock', 'SingletonSocket', 'SingletonCookie']:
        lock_path = PROFILE_DIR / lock_file
        if lock_path.exists():
            try:
                lock_path.unlink()
            except Exception:
                pass

    print("Launching browser...")
    options = uc.ChromeOptions()
    options.add_argument('--start-maximized')
    options.add_argument(f'--user-data-dir={PROFILE_DIR}')
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})

    chrome_ver = get_chrome_version_main()
    print(f"Chrome version detected: {chrome_ver}")
    return uc.Chrome(options=options, version_main=chrome_ver)


def is_logged_in(driver):
    """Check if we already have an active session by navigating to the puzzles page."""
    print("Checking for existing session...")
    driver.get('https://www.thetimes.com/puzzles/crossword')
    time.sleep(5)

    # If we can see puzzle links, we're logged in
    links = driver.find_elements(By.TAG_NAME, "a")
    for link in links:
        href = link.get_attribute("href") or ""
        if re.search(r'times-cryptic-no-\d+', href):
            print("Already logged in (session cookies valid)")
            return True

    # Check if we were redirected to login
    if "login" in driver.current_url.lower():
        print("Session expired — need to log in")
        return False

    print("Could not find puzzle links — need to log in")
    return False


def login(driver):
    """Log in to The Times. Returns the driver on success, raises on failure."""
    print(f"Navigating to login page...")
    driver.get(LOGIN_URL)
    time.sleep(3)

    wait = WebDriverWait(driver, 20)
    print(f"Login page URL: {driver.current_url}")

    # Check for Auth0 iframe (older Times login used one)
    try:
        iframe = WebDriverWait(driver, 5).until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "iframe[name='auth0-lock-widget-frame']")))
        print("Switching to Auth0 iframe...")
        driver.switch_to.frame(iframe)
    except Exception:
        print("No iframe found, continuing on main page...")

    # Wait for form to render
    print("Waiting for login form to be ready...")
    wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "input[name='email']")))
    time.sleep(2)

    # Enter credentials
    print("Entering email...")
    email_field = driver.find_element(By.CSS_SELECTOR, "input[name='email']")
    email_field.click()
    email_field.clear()
    email_field.send_keys(TIMES_EMAIL)

    print("Entering password...")
    password_field = driver.find_element(By.CSS_SELECTOR, "input[name='password']")
    password_field.click()
    password_field.clear()
    password_field.send_keys(TIMES_PASSWORD)

    time.sleep(0.5)

    print("Clicking login...")
    submit_btn = driver.find_element(By.CSS_SELECTOR, "button.auth0-lock-submit")
    submit_btn.click()

    print("Waiting for login redirect...")
    driver.switch_to.default_content()

    # Auth0 redirect chain crashes WebDriver if polled too aggressively.
    # Sleep first, then gently check if we landed somewhere useful.
    time.sleep(15)

    for check in range(6):
        try:
            url = driver.current_url.lower()
            if "login" not in url:
                print(f"Logged in! URL: {driver.current_url}")
                return driver
            if "thetimes.co.uk" in url and "/login" not in url:
                print(f"Logged in! URL: {driver.current_url}")
                return driver
        except Exception:
            pass  # WebDriver temporarily unavailable during redirect
        time.sleep(3)

    # Final check: maybe we're logged in but URL still looks odd
    try:
        current = driver.current_url
        page_text = driver.page_source[:5000].lower()
        if "my account" in page_text or "www.thetimes.co.uk" in current:
            print(f"Login appears successful despite URL. URL: {current}")
            return driver
    except Exception:
        pass

    raise Exception("Login failed — could not confirm redirect after 30s")


# ── Puzzle fetching (uses existing browser session) ───────────────────────

def fetch_puzzle(driver, puzzle_type):
    """Navigate to a puzzle and fetch its data via the API.

    Uses an already-logged-in browser session.
    Returns (json_data, puzzle_number, api_id) or (None, None, None) on failure.
    """
    config = PUZZLE_TYPES[puzzle_type]
    puzzle_url = config['page_url']

    print(f"\n{'=' * 60}")
    print(f"FETCHING: {config['name']}")
    print(f"{'=' * 60}")

    # Navigate to crossword listing page
    print(f"Navigating to crossword page: {puzzle_url}")
    for nav_attempt in range(3):
        try:
            driver.get(puzzle_url)
            break
        except Exception:
            if nav_attempt < 2:
                print(f"  Navigation failed (attempt {nav_attempt + 1}/3), retrying in 5s...")
                time.sleep(5)
            else:
                raise
    time.sleep(5)

    # Calculate expected puzzle number
    expected_number = estimate_puzzle_number(puzzle_type)
    print(f"Looking for puzzle #{expected_number}...")

    # Find the link with the exact puzzle number
    links = driver.find_elements(By.TAG_NAME, "a")
    target_href = None
    puzzle_number = None

    for link in links:
        href = link.get_attribute("href") or ""

        if puzzle_type == 'sunday-cryptic':
            match = re.search(r'sunday-times-cryptic-no-(\d+)', href)
        else:
            if 'sunday' in href.lower():
                continue
            match = re.search(r'times-cryptic-no-(\d+)', href)

        if match:
            found_number = int(match.group(1))
            if found_number == expected_number:
                target_href = href
                puzzle_number = found_number
                print(f"Found exact match: puzzle #{puzzle_number}")
                break

    if not target_href:
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
                candidates.append((int(match.group(1)), href))

        if candidates:
            candidates.sort(key=lambda x: x[0], reverse=True)
            puzzle_number = candidates[0][0]
            target_href = candidates[0][1]
            print(f"Using highest number: puzzle #{puzzle_number}")

    if not target_href:
        print("ERROR: Could not find puzzle link!")
        return None, None, None

    # Navigate directly to the puzzle page
    print(f"Navigating to puzzle #{puzzle_number}: {target_href}")
    driver.get(target_href)

    print("Waiting for puzzle to load...")
    time.sleep(10)

    print(f"Current URL: {driver.current_url}")

    # Extract puzzle number from URL
    url_match = re.search(r'no-(\d+)', driver.current_url)
    if url_match:
        puzzle_number = int(url_match.group(1))
        print(f"Puzzle number from URL: {puzzle_number}")

    # Find the puzzle iframe
    api_url = None
    api_id = None

    print("Looking for puzzle iframe...")
    try:
        iframes = driver.find_elements(By.TAG_NAME, "iframe")
        for iframe in iframes:
            src = iframe.get_attribute("src") or ""
            if "feeds.thetimes" in src:
                base = src.split("?")[0].rstrip("/")
                api_url = base + "/data.json"
                print(f"Found iframe src: {src[:120]}")
                print(f"Constructed API URL: {api_url}")
                id_match = re.search(r'/sp/(\w+)/(\d{8})/(\d+)', api_url)
                if id_match:
                    detected_feed = id_match.group(1)
                    api_id = id_match.group(3)
                    print(f"Detected feed type: {detected_feed}")
                    print(f"Extracted API ID: {api_id}")
                break
        else:
            all_srcs = [iframe.get_attribute("src") or "(empty)" for iframe in iframes]
            print(f"No feeds.thetimes iframe found. {len(iframes)} iframes on page:")
            for s in all_srcs:
                print(f"  {s[:120]}")
    except Exception as e:
        print(f"  Iframe search failed: {e}")

    if not api_url:
        # Fallback: search page source
        print("Searching page source for API URL...")
        page_source = driver.page_source
        match = re.search(
            r'(https://feeds\.thetimes\.(?:com|co\.uk)/puzzles/sp/\w+/\d{8}/\d+(?:/data\.json)?)',
            page_source
        )
        if match:
            api_url = match.group(1)
            if not api_url.endswith('/data.json'):
                api_url += '/data.json'
            print(f"Found API URL in page source: {api_url}")
            id_match = re.search(r'/sp/(\w+)/(\d{8})/(\d+)', api_url)
            if id_match:
                api_id = id_match.group(3)

    if not api_url:
        print("WARNING: Could not find API URL")
        debug_file = SCRIPT_DIR / "times_page_debug.html"
        with open(debug_file, 'w', encoding='utf-8') as f:
            f.write(driver.page_source)
        print(f"Saved debug page to {debug_file}")
        return None, puzzle_number, api_id

    # Fetch puzzle data via API
    print(f"Fetching puzzle data from API...")
    cookies = {c['name']: c['value'] for c in driver.get_cookies()}
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
        'Accept': 'application/json',
        'Referer': 'https://www.thetimes.com/',
    }

    response = requests.get(api_url, headers=headers, cookies=cookies, timeout=30)

    json_data = None
    if response.status_code == 200:
        json_data = response.json()
        print("Successfully fetched puzzle data!")
    else:
        print(f"Failed to fetch puzzle data: {response.status_code}")
        response = requests.get(api_url, headers=headers, timeout=30)
        if response.status_code == 200:
            json_data = response.json()
            print("Fetched puzzle data (no cookies needed)")

    return json_data, puzzle_number, api_id


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
            clue_text = re.sub(r'<[^>]+>', '', html.unescape(clue.get('clue', '')))
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
    """Check if puzzle already in the clues table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = 'times' AND puzzle_number = ?
    """, (str(puzzle_number),))

    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def save_to_database(puzzle_data):
    """Save puzzle clues directly to clues table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    puzzle_number = puzzle_data.get('puzzle_number', '')
    puzzle_date = puzzle_data.get('date', '')

    clue_count = 0
    for direction in ['across', 'down']:
        for clue in puzzle_data.get(direction, []):
            cursor.execute("""
                INSERT OR IGNORE INTO clues
                (source, puzzle_number, publication_date, clue_number, direction,
                 clue_text, enumeration, answer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'times',
                str(puzzle_number),
                puzzle_date,
                str(clue.get('number', '')),
                direction,
                clue.get('clue', ''),
                clue.get('enumeration', ''),
                clue.get('answer', ''),
            ))
            clue_count += 1

    conn.commit()
    conn.close()

    print(f"Saved {clue_count} clues to clues table")
    return clue_count


def process_puzzle(driver, puzzle_type, force=False):
    """Fetch, parse and save a single puzzle type. Returns True on success."""
    json_data, puzzle_number, api_id = fetch_puzzle(driver, puzzle_type)

    if not json_data:
        print(f"Failed to fetch data")
        return False

    if puzzle_number:
        print(f"Puzzle number: {puzzle_number}")
    if api_id:
        print(f"API ID: {api_id}")

    # Check if already fetched
    if not force and puzzle_exists(puzzle_number, puzzle_type):
        print(f"Puzzle {puzzle_number} already in database.")
        return True

    # Parse
    puzzle_data = parse_puzzle(json_data)

    if not puzzle_data.get('puzzle_number'):
        print("Failed to parse puzzle")
        return False

    if not puzzle_data['puzzle_number'] and puzzle_number:
        puzzle_data['puzzle_number'] = puzzle_number

    # Delete existing if force mode
    if force and puzzle_exists(puzzle_number, puzzle_type):
        print(f"Removing existing puzzle {puzzle_number}...")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            DELETE FROM clues
            WHERE source = 'times' AND puzzle_number = ?
        """, (str(puzzle_number),))
        conn.commit()
        conn.close()

    # Save
    save_to_database(puzzle_data)

    # Save JSON backup
    json_path = SCRIPT_DIR / f"times_{puzzle_type}_{puzzle_number}.json"
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"JSON backup: {json_path}")

    return True


def main():
    print("=" * 60)
    print("THE TIMES CROSSWORD SCRAPER")
    print("=" * 60)

    today = date.today()
    print(f"Today: {today.strftime('%A, %d %B %Y')}")
    print(f"Database: {DB_PATH}")

    force = '--force' in sys.argv

    # If a specific type was requested on the command line, run only that
    explicit_type = None
    for arg in sys.argv[1:]:
        if arg in PUZZLE_TYPES:
            explicit_type = arg
            break

    if explicit_type:
        types_to_run = [explicit_type]
    else:
        # Run all puzzle types scheduled for today's day of week
        types_to_run = [
            pt for pt, cfg in PUZZLE_TYPES.items()
            if today.weekday() in cfg['days']
        ]

    if not types_to_run:
        print("No puzzle types scheduled for today.")
        return

    print(f"Puzzle types to fetch: {', '.join(types_to_run)}")

    # Single browser session with persistent profile
    driver = create_browser()
    try:
        if not is_logged_in(driver):
            login(driver)

        results = {}
        for pt in types_to_run:
            try:
                results[pt] = process_puzzle(driver, pt, force=force)
            except Exception as e:
                print(f"Error fetching {pt}: {e}")
                results[pt] = False

    except Exception as e:
        print(f"Login/browser error: {e}")
        try:
            driver.save_screenshot(str(SCRIPT_DIR / "times_error.png"))
            with open(str(SCRIPT_DIR / "times_error.html"), 'w', encoding='utf-8') as f:
                f.write(driver.page_source)
        except Exception:
            pass
        results = {pt: False for pt in types_to_run}
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        print("Browser closed.")

    # Summary
    print(f"\n{'=' * 60}")
    print("SUMMARY")
    print(f"{'=' * 60}")
    all_ok = True
    for pt, ok in results.items():
        status = "OK" if ok else "FAILED"
        print(f"  {PUZZLE_TYPES[pt]['name']:25} {status}")
        if not ok:
            all_ok = False

    if not all_ok:
        sys.exit(1)


if __name__ == "__main__":
    main()
