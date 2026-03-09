#!/usr/bin/env python3
"""Telegraph Daily Scraper

Logs in via browser, finds today's puzzle API IDs from the puzzles page,
fetches each puzzle JSON, and saves clues directly to the clues table in clues_master.db.

Requires .env with TELEGRAPH_EMAIL and TELEGRAPH_PASSWORD.
"""

import html
import re
import os
import json
import sys
import time
import sqlite3
import subprocess
import requests
from datetime import datetime, date
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

LOGIN_URL = "https://secure.telegraph.co.uk/customer/secure/login/"
PUZZLES_URL = "https://www.telegraph.co.uk/puzzles/"
PROFILE_DIR = SCRIPT_DIR / '.chrome_profile'

# Map link types to (puzzle_type, is_prize)
# is_prize = True means skip date check (prize links show closing date, not publication date)
# NOTE: The puzzle number in the URL is often wrong (e.g. #31169 for a prize cryptic).
# The API returns the correct puzzle data regardless — dedup happens via puzzle_already_fetched.
TYPE_MAP = {
    'cryptic-crossword': ('cryptic',       False),
    'toughie-crossword': ('toughie',       False),
    'prize-cryptic':     ('prize-cryptic', True),
    'prize-toughie':     ('prize-toughie', True),
}


def dismiss_cookie_consent(driver):
    """Dismiss the cookie consent popup if present."""
    try:
        wait = WebDriverWait(driver, 5)
        iframe = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "iframe[id*='sp_message_iframe']")))
        driver.switch_to.frame(iframe)
        # Click "Essential cookies only" or "I Accept"
        for selector in [
            "button[title='Essential cookies only']",
            "button[title='I Accept']",
            "button[title='Accept']",
            "button.sp_choice_type_11",
            "button.sp_choice_type_12",
        ]:
            try:
                btn = driver.find_element(By.CSS_SELECTOR, selector)
                btn.click()
                print("Dismissed cookie consent.")
                break
            except Exception:
                continue
        driver.switch_to.default_content()
        time.sleep(1)
    except Exception:
        driver.switch_to.default_content()


def login(driver):
    """Log in to Telegraph using credentials from .env."""
    email = os.getenv("TELEGRAPH_EMAIL")
    password = os.getenv("TELEGRAPH_PASSWORD")
    if not email or not password:
        raise ValueError("Missing TELEGRAPH_EMAIL or TELEGRAPH_PASSWORD in .env")

    print("Navigating to login page...")
    driver.get(LOGIN_URL)
    time.sleep(3)
    dismiss_cookie_consent(driver)

    wait = WebDriverWait(driver, 15)

    # Save login page HTML for debugging
    debug_html = SCRIPT_DIR / "telegraph_login_debug.html"
    with open(debug_html, 'w', encoding='utf-8') as f:
        f.write(driver.page_source)
    print(f"Saved login page HTML: {debug_html.name}")

    # Find email field — try multiple selectors for old/new UI
    print("Entering email...")
    email_field = None
    for selector in [(By.ID, "email"), (By.CSS_SELECTOR, "input[type='email']"),
                     (By.CSS_SELECTOR, "input[name='email']"),
                     (By.CSS_SELECTOR, "input[placeholder*='email']")]:
        try:
            email_field = wait.until(EC.presence_of_element_located(selector))
            print(f"  Found email field via {selector}")
            break
        except Exception:
            continue
    if not email_field:
        raise Exception("Could not find email field")
    email_field.click()
    time.sleep(0.5)
    email_field.clear()
    email_field.send_keys(email)
    time.sleep(1)

    # Verify email was entered — React inputs may ignore send_keys
    val = email_field.get_attribute('value')
    if not val or val == email_field.get_attribute('placeholder'):
        print("  send_keys failed, trying React-compatible JS injection...")
        driver.execute_script("""
            var el = arguments[0];
            var val = arguments[1];
            var setter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value').set;
            setter.call(el, val);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        """, email_field, email)
        time.sleep(0.5)
        # Double-check
        val = email_field.get_attribute('value')
        if not val:
            print("  React setter also failed, trying ActionChains...")
            from selenium.webdriver.common.action_chains import ActionChains
            email_field.click()
            email_field.clear()
            ActionChains(driver).click(email_field).send_keys(email).perform()
            time.sleep(0.5)

    print("Clicking continue...")
    # The new Telegraph login renders buttons via JS/shadow DOM.
    # Use JavaScript to find and click the Continue button.
    clicked = driver.execute_script("""
        // Try standard selectors first
        var selectors = ['button.screen-cta', 'a.screen-cta', 'button[type="submit"]'];
        for (var s of selectors) {
            var el = document.querySelector(s);
            if (el && el.offsetParent !== null) { el.click(); return 'css:' + s; }
        }
        // Search all elements for "Continue" text
        var all = document.querySelectorAll('button, a, div[role="button"], span[role="button"]');
        for (var el of all) {
            if (el.textContent.trim() === 'Continue' && el.offsetParent !== null) {
                el.click(); return 'text:Continue';
            }
        }
        // Last resort: find any clickable element with Continue
        var everything = document.querySelectorAll('*');
        for (var el of everything) {
            if (el.childElementCount === 0 && el.textContent.trim() === 'Continue') {
                el.closest('button, a, [role="button"]')?.click() || el.parentElement.click();
                return 'parent:Continue';
            }
        }
        return null;
    """)
    if clicked:
        print(f"  Clicked via {clicked}")
    else:
        from selenium.webdriver.common.keys import Keys
        print("  No button found via JS, pressing Enter...")
        email_field.send_keys(Keys.RETURN)

    # Wait for password step to load
    time.sleep(5)

    print("Entering password...")
    password_field = None
    for selector in [(By.ID, "password"), (By.CSS_SELECTOR, "input[type='password']"),
                     (By.CSS_SELECTOR, "input[name='password']")]:
        try:
            password_field = WebDriverWait(driver, 10).until(
                EC.presence_of_element_located(selector))
            break
        except Exception:
            continue
    if not password_field:
        # Save debug screenshot before failing
        driver.save_screenshot(str(SCRIPT_DIR / "telegraph_password_debug.png"))
        raise Exception("Could not find password field — login UI may have changed")
    password_field.click()
    time.sleep(0.5)
    password_field.clear()
    password_field.send_keys(password)
    time.sleep(1)

    # Verify password was entered — same React issue as email
    val = password_field.get_attribute('value')
    if not val:
        print("  Password send_keys failed, trying React-compatible JS injection...")
        driver.execute_script("""
            var el = arguments[0];
            var val = arguments[1];
            var setter = Object.getOwnPropertyDescriptor(
                window.HTMLInputElement.prototype, 'value').set;
            setter.call(el, val);
            el.dispatchEvent(new Event('input', { bubbles: true }));
            el.dispatchEvent(new Event('change', { bubbles: true }));
        """, password_field, password)
        time.sleep(0.5)

    print("Clicking login...")
    # Use same JS approach that worked for the first Continue button
    clicked = driver.execute_script("""
        var selectors = ['button.screen-cta', 'a.screen-cta', 'button[type="submit"]'];
        for (var s of selectors) {
            var el = document.querySelector(s);
            if (el && el.offsetParent !== null) { el.click(); return 'css:' + s; }
        }
        var all = document.querySelectorAll('button, a, div[role="button"], span[role="button"]');
        for (var el of all) {
            var t = el.textContent.trim();
            if ((t === 'Continue' || t === 'Log in' || t === 'Sign in') && el.offsetParent !== null) {
                el.click(); return 'text:' + t;
            }
        }
        return null;
    """)
    if clicked:
        print(f"  Clicked via {clicked}")
    else:
        from selenium.webdriver.common.keys import Keys
        print("  No button found via JS, pressing Enter...")
        password_field.send_keys(Keys.RETURN)

    print("Waiting for login to complete...")
    time.sleep(15)

    for check in range(6):
        try:
            if "login" not in driver.current_url.lower():
                print(f"Logged in! URL: {driver.current_url}")
                return
        except Exception:
            pass
        time.sleep(3)

    # Check if we actually landed on the site
    try:
        if "telegraph.co.uk" in driver.current_url and "login" not in driver.current_url.lower():
            print(f"Logged in! URL: {driver.current_url}")
            return
    except Exception:
        pass

    raise Exception("Login failed — could not confirm redirect after 30s")


def harvest_today(driver, already_on_page=False):
    """Extract today's puzzle API IDs from the puzzles page."""
    today = date.today()
    # Page may show "1 Mar, 2026" or "01 Mar, 2026" — match both
    today_str = f"{today.day} {today.strftime('%b')}, {today.year}"
    today_str_padded = f"{today.day:02d} {today.strftime('%b')}, {today.year}"

    if not already_on_page:
        print(f"\nNavigating to puzzles page...")
        driver.get(PUZZLES_URL)
        print("Waiting for puzzles to load (JS app)...")
        time.sleep(20)

    all_links = driver.find_elements(By.TAG_NAME, "a")
    print(f"Found {len(all_links)} links on page")
    print(f"Looking for puzzles dated: {today_str}")

    puzzles = []
    seen_types = set()
    seen_apis = set()  # Deduplicate by API ID (page may have multiple links to same puzzle)

    for link in all_links:
        try:
            href = link.get_attribute("href") or ""
            text = link.text.strip()

            if not href or "Archive" in text:
                continue

            # Match puzzle URL pattern: #crossword/{folder}/{type}-{api_id}
            match = re.search(r'#crossword/([^/]+)/(.+)-(\d+)', href)
            if not match:
                continue

            folder, link_type, api_id = match.groups()

            # Extract puzzle number (NOTE: URL number is unreliable for prize puzzles)
            num_match = re.search(r'number=(\d+)', href)
            puzzle_num = num_match.group(1) if num_match else None

            # Extract date from link text
            date_match = re.search(r'(\d{1,2} \w{3}, \d{4})', text)
            link_date = date_match.group(1) if date_match else None

            if link_type not in TYPE_MAP:
                continue

            puzzle_type, is_prize = TYPE_MAP[link_type]

            # Log all puzzle links we find (first occurrence of each type)
            if link_type not in seen_types:
                seen_types.add(link_type)
                print(f"  [scan] {link_type} #{puzzle_num} date='{link_date}' api={api_id}")

            # Only take today's puzzles
            # Prize types show closing date not publication date, so skip date check
            if not is_prize and link_date not in (today_str, today_str_padded):
                continue

            # Deduplicate by API ID
            api_key = f"{link_type}-{api_id}"
            if api_key in seen_apis:
                continue
            seen_apis.add(api_key)

            print(f"  Found: {puzzle_type} #{puzzle_num} -> API {api_id} ({link_date})")

            puzzles.append({
                'folder': folder,
                'link_type': link_type,
                'puzzle_type': puzzle_type,
                'api_id': api_id,
                'puzzle_number': puzzle_num,
            })

        except Exception as e:
            continue

    return puzzles


def puzzle_already_fetched(puzzle_type, puzzle_number):
    """Check if puzzle is already in the clues table."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT COUNT(*) FROM clues
        WHERE source = 'telegraph' AND puzzle_number = ?
    """, (str(puzzle_number),))

    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def extract_puzzle_number_from_title(title: str) -> str | None:
    """Extract the real puzzle number from the API title.

    Titles look like: "Cryptic Crossword No 31174",
    "Prize Cryptic No 3358", "Prize Toughie No 214", "Toughie Crossword No 3644".
    The URL puzzle number is sometimes wrong (especially for prize-cryptic),
    so we trust the API title instead.
    """
    match = re.search(r'No\.?\s*(\d+)', title)
    return match.group(1) if match else None


def fetch_and_save(puzzle):
    """Fetch puzzle JSON from API and save clues directly to clues table."""
    folder = puzzle['folder']
    link_type = puzzle['link_type']
    api_id = puzzle['api_id']
    puzzle_type = puzzle['puzzle_type']
    url_puzzle_number = puzzle['puzzle_number']

    url = f"https://puzzlesdata.telegraph.co.uk/puzzles/{folder}/{link_type}-{api_id}.json"
    print(f"  Fetching: {url}")

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"  Error: HTTP {response.status_code}")
            return 'failed'
    except Exception as e:
        print(f"  Error: {e}")
        return 'failed'

    data = response.json()

    # Save raw JSON for prize puzzles (preserves grid data for answer validation)
    if puzzle_type.startswith('prize'):
        json_path = SCRIPT_DIR / f"telegraph_{link_type}_{api_id}.json"
        with open(json_path, 'w') as f:
            json.dump(data, f)
        print(f"  Saved prize JSON: {json_path.name}")

    copy = data.get('json', {}).get('copy', {})

    title = copy.get('title', '')
    date_publish = copy.get('date-publish', '')
    # Convert to ISO date
    if date_publish and ',' in date_publish:
        try:
            date_publish = datetime.strptime(date_publish, '%A, %d %B %Y').strftime('%Y-%m-%d')
        except ValueError:
            pass

    # Use the real puzzle number from the API title, not the (often wrong) URL number
    real_number = extract_puzzle_number_from_title(title)
    if real_number and real_number != url_puzzle_number:
        print(f"  URL had #{url_puzzle_number}, API title says #{real_number} — using API number")
    puzzle_number = real_number or url_puzzle_number

    already_exists = puzzle_already_fetched(puzzle_type, puzzle_number)

    # Parse clues
    clues_groups = copy.get('clues', [])
    across = []
    down = []

    for group in clues_groups:
        direction = group.get('title', '').lower()
        for clue in group.get('clues', []):
            clue_obj = {
                'number': clue.get('number', ''),
                'clue': re.sub(r'<[^>]+>', '', html.unescape(clue.get('clue', ''))),
                'answer': clue.get('answer', ''),
                'enumeration': clue.get('format', '')
            }
            if direction == 'across':
                across.append(clue_obj)
            elif direction == 'down':
                down.append(clue_obj)

    print(f"  Title: {title}")
    print(f"  Puzzle #{puzzle_number} | {len(across)} across, {len(down)} down")

    # Save directly to clues table (INSERT OR IGNORE handles dedup)
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    inserted = 0

    for direction, clue_list in [('across', across), ('down', down)]:
        for clue in clue_list:
            cursor.execute("""
                INSERT INTO clues
                (source, puzzle_number, publication_date, clue_number, direction,
                 clue_text, enumeration, answer)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(source, puzzle_number, clue_number, direction, publication_date)
                DO UPDATE SET answer = excluded.answer
                WHERE answer IS NULL OR answer = ''
            """, (
                'telegraph',
                str(puzzle_number),
                date_publish,
                str(clue['number']),
                direction,
                clue['clue'],
                clue['enumeration'],
                clue['answer'],
            ))
            if cursor.rowcount > 0:
                inserted += 1

    # Store grid solution string for direct grid rendering
    settings = copy.get('settings', {})
    grid_solution = settings.get('solution', '')
    gridsize = copy.get('gridsize', {})
    grid_rows = int(gridsize.get('rows', 15))
    grid_cols = int(gridsize.get('cols', 15))

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS puzzle_grids (
            source TEXT NOT NULL,
            puzzle_number TEXT NOT NULL,
            solution TEXT,
            grid_rows INTEGER NOT NULL DEFAULT 15,
            grid_cols INTEGER NOT NULL DEFAULT 15,
            api_folder TEXT,
            api_type TEXT,
            api_id TEXT,
            PRIMARY KEY (source, puzzle_number)
        )
    """)

    has_solution = grid_solution and len(grid_solution) == grid_rows * grid_cols
    cursor.execute("""
        INSERT INTO puzzle_grids
        (source, puzzle_number, solution, grid_rows, grid_cols, api_folder, api_type, api_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        ON CONFLICT(source, puzzle_number) DO UPDATE SET
            solution = COALESCE(excluded.solution, puzzle_grids.solution),
            api_folder = COALESCE(excluded.api_folder, puzzle_grids.api_folder),
            api_type = COALESCE(excluded.api_type, puzzle_grids.api_type),
            api_id = COALESCE(excluded.api_id, puzzle_grids.api_id)
    """, (
        'telegraph', str(puzzle_number),
        grid_solution if has_solution else None,
        grid_rows, grid_cols,
        folder, link_type, api_id,
    ))

    conn.commit()
    conn.close()

    if already_exists and inserted == 0:
        print(f"  Already complete (#{puzzle_number}) - skipping")
        return 'skipped'
    elif already_exists and inserted > 0:
        print(f"  Repaired #{puzzle_number}: added {inserted} missing clues")
        return 'fetched'
    else:
        print(f"  Saved {len(across) + len(down)} clues as #{puzzle_number}")
        return 'fetched'


def backfill_grid_solutions():
    """Re-fetch the grid solution string for puzzles that have API coordinates but no solution.

    Prize puzzles are scraped before solutions are published — this picks up the
    grid layout once Telegraph releases it.  Answers are handled separately by
    the DANWORD backfill in puzzle_scraper.py, so this only touches puzzle_grids.
    No browser needed — uses direct API calls.
    """
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    try:
        gaps = cursor.execute("""
            SELECT puzzle_number, api_folder, api_type, api_id
            FROM puzzle_grids
            WHERE source = 'telegraph'
              AND solution IS NULL
              AND api_folder IS NOT NULL AND api_id IS NOT NULL
        """).fetchall()
    except Exception:
        gaps = []

    if not gaps:
        print("No grid solution gaps to backfill.")
        return 0

    print(f"Backfilling {len(gaps)} puzzle grid solutions...")
    filled = 0

    for puzzle_number, api_folder, api_type, api_id in gaps:
        url = f"https://puzzlesdata.telegraph.co.uk/puzzles/{api_folder}/{api_type}-{api_id}.json"
        try:
            response = requests.get(url, timeout=30)
            if response.status_code != 200:
                continue
            data = response.json()
        except Exception:
            continue

        copy = data.get('json', {}).get('copy', {})
        settings = copy.get('settings', {})
        solution = settings.get('solution', '')
        gridsize = copy.get('gridsize', {})
        grid_rows = int(gridsize.get('rows', 15))
        grid_cols = int(gridsize.get('cols', 15))

        if not solution or len(solution) != grid_rows * grid_cols:
            continue

        cursor.execute("""
            UPDATE puzzle_grids SET solution = ?, grid_rows = ?, grid_cols = ?
            WHERE source = 'telegraph' AND puzzle_number = ?
        """, (solution, grid_rows, grid_cols, puzzle_number))
        conn.commit()
        filled += 1
        print(f"  #{puzzle_number}: grid solution filled")

    conn.close()
    print(f"Backfilled {filled}/{len(gaps)} grid solutions.")
    return filled


def is_logged_in(driver):
    """Check if we already have an active Telegraph session."""
    print("Checking for existing session...")
    driver.get(PUZZLES_URL)
    time.sleep(20)  # JS app needs time to load

    # If we can see puzzle links, we're logged in
    links = driver.find_elements(By.TAG_NAME, "a")
    for link in links:
        href = link.get_attribute("href") or ""
        if re.search(r'#crossword/.+-(cryptic|toughie)', href):
            print("Already logged in (session cookies valid)")
            return True

    if "login" in driver.current_url.lower():
        print("Session expired — need to log in")
        return False

    print("Could not find puzzle links — need to log in")
    return False


def main():
    today = date.today()
    print("=" * 60)
    print("TELEGRAPH DAILY SCRAPER")
    print(f"Today: {today.strftime('%A, %d %B %Y')}")
    print("=" * 60)
    print(f"Database: {DB_PATH}")

    # Launch browser with persistent profile
    print("\nLaunching browser...")

    # Clean up stale lock files from previous crashed sessions
    for lock_file in ['SingletonLock', 'SingletonSocket', 'SingletonCookie']:
        lock_path = PROFILE_DIR / lock_file
        if lock_path.exists():
            try:
                lock_path.unlink()
            except Exception:
                pass

    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    options.add_argument(f'--user-data-dir={PROFILE_DIR}')
    options.add_argument("--no-first-run")
    options.add_argument("--no-default-browser-check")

    chrome_ver = get_chrome_version_main()
    print(f"Chrome version detected: {chrome_ver}")
    driver = uc.Chrome(options=options, version_main=chrome_ver)

    # Persistent profile may restore old tabs — settle and focus
    time.sleep(3)
    try:
        if len(driver.window_handles) > 1:
            for handle in driver.window_handles[1:]:
                driver.switch_to.window(handle)
                driver.close()
            driver.switch_to.window(driver.window_handles[0])
    except Exception:
        pass

    try:
        already_on_page = is_logged_in(driver)
        if not already_on_page:
            login(driver)
        puzzles = harvest_today(driver, already_on_page=already_on_page)
    except Exception as e:
        print(f"Error during harvest: {e}")
        screenshot = SCRIPT_DIR / "telegraph_error.png"
        try:
            driver.save_screenshot(str(screenshot))
            print(f"Screenshot saved: {screenshot}")
        except Exception:
            pass
        sys.exit(1)
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        print("Browser closed.")

    if not puzzles:
        print("\nNo puzzles found for today.")
        sys.exit(1)

    # Fetch and save each puzzle
    print(f"\n{'=' * 60}")
    print(f"FETCHING {len(puzzles)} PUZZLES")
    print(f"{'=' * 60}")

    stats = {'fetched': 0, 'skipped': 0, 'failed': 0}

    for puzzle in puzzles:
        print(f"\n{puzzle['puzzle_type']} #{puzzle['puzzle_number']}")
        result = fetch_and_save(puzzle)
        stats[result] += 1

    print(f"\n{'=' * 60}")
    print(f"DONE - Fetched: {stats['fetched']}, "
          f"Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    print(f"{'=' * 60}")

    # Backfill grid solutions for prize puzzles whose solutions have since been released
    print(f"\n{'=' * 60}")
    print("GRID SOLUTION BACKFILL")
    print(f"{'=' * 60}")
    backfill_grid_solutions()


if __name__ == "__main__":
    main()
