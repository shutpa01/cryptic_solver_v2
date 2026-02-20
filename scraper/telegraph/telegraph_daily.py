#!/usr/bin/env python3
"""Telegraph Daily Scraper

Logs in via browser, finds today's puzzle API IDs from the puzzles page,
fetches each puzzle JSON, and saves clues to the database.

Requires .env with TELEGRAPH_EMAIL and TELEGRAPH_PASSWORD.
"""

import html
import re
import os
import json
import time
import sqlite3
import requests
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db")

LOGIN_URL = "https://secure.telegraph.co.uk/customer/secure/login/"
PUZZLES_URL = "https://www.telegraph.co.uk/puzzles/"

# Map harvest link types to our DB puzzle_type values
TYPE_MAP = {
    'cryptic-crossword': 'cryptic',
    'prize-cryptic': 'saturday-cryptic',
    'toughie-crossword': 'toughie',
    'prize-toughie': 'prize-toughie',
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
    dismiss_cookie_consent(driver)

    wait = WebDriverWait(driver, 15)

    print("Entering email...")
    email_field = wait.until(EC.presence_of_element_located((By.ID, "email")))
    email_field.clear()
    email_field.send_keys(email)

    print("Clicking continue...")
    # Try <a> first (old UI), fall back to <button> (new UI)
    for cta_selector in ["a.screen-cta", "button.screen-cta", "button[type='submit']"]:
        try:
            continue_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, cta_selector)))
            continue_btn.click()
            break
        except Exception:
            continue

    print("Entering password...")
    password_field = wait.until(EC.presence_of_element_located((By.ID, "password")))
    password_field.clear()
    password_field.send_keys(password)

    print("Clicking login...")
    for cta_selector in ["a.screen-cta", "button.screen-cta", "button[type='submit']"]:
        try:
            login_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, cta_selector)))
            login_btn.click()
            break
        except Exception:
            continue

    print("Waiting for login to complete...")
    wait.until(lambda d: "login" not in d.current_url.lower())
    print(f"Logged in! URL: {driver.current_url}")


def harvest_today(driver):
    """Navigate to puzzles page and extract today's puzzle API IDs."""
    today = date.today()
    today_str = f"{today.day} {today.strftime('%b')}, {today.year}"

    print(f"\nNavigating to puzzles page...")
    driver.get(PUZZLES_URL)

    print("Waiting for puzzles to load (JS app)...")
    time.sleep(20)

    all_links = driver.find_elements(By.TAG_NAME, "a")
    print(f"Found {len(all_links)} links on page")
    print(f"Looking for puzzles dated: {today_str}")

    puzzles = []

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

            # Extract puzzle number
            num_match = re.search(r'number=(\d+)', href)
            puzzle_num = num_match.group(1) if num_match else None

            # Extract date from link text
            date_match = re.search(r'(\d{1,2} \w{3}, \d{4})', text)
            link_date = date_match.group(1) if date_match else None

            # Only take today's puzzles
            if link_date != today_str:
                continue

            if link_type not in TYPE_MAP:
                continue

            puzzle_type = TYPE_MAP[link_type]

            print(f"  Found: {link_type} #{puzzle_num} -> API {api_id} ({link_date})")

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
    """Check if puzzle is already in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS telegraph_clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_type TEXT,
            puzzle_number TEXT,
            puzzle_date TEXT,
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
        SELECT COUNT(*) FROM telegraph_clues
        WHERE puzzle_type = ? AND puzzle_number = ?
    """, (puzzle_type, str(puzzle_number)))

    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def fetch_and_save(puzzle):
    """Fetch puzzle JSON from API and save clues to database."""
    folder = puzzle['folder']
    link_type = puzzle['link_type']
    api_id = puzzle['api_id']
    puzzle_type = puzzle['puzzle_type']
    puzzle_number = puzzle['puzzle_number']

    if puzzle_already_fetched(puzzle_type, puzzle_number):
        print(f"  Already in database - skipping")
        return 'skipped'

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
    copy = data.get('json', {}).get('copy', {})

    title = copy.get('title', '')
    date_publish = copy.get('date-publish', '')
    # Convert to ISO date
    if date_publish and ',' in date_publish:
        try:
            date_publish = datetime.strptime(date_publish, '%A, %d %B %Y').strftime('%Y-%m-%d')
        except ValueError:
            pass

    # Parse clues
    clues_groups = copy.get('clues', [])
    across = []
    down = []

    for group in clues_groups:
        direction = group.get('title', '').lower()
        for clue in group.get('clues', []):
            clue_obj = {
                'number': clue.get('number', ''),
                'clue': html.unescape(clue.get('clue', '')),
                'answer': clue.get('answer', ''),
                'enumeration': clue.get('format', '')
            }
            if direction == 'across':
                across.append(clue_obj)
            elif direction == 'down':
                down.append(clue_obj)

    print(f"  Title: {title}")
    print(f"  Clues: {len(across)} across, {len(down)} down")

    # Save to database
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
    fetched_at = datetime.now().isoformat()
    clue_count = 0

    for direction, clue_list in [('across', across), ('down', down)]:
        for clue in clue_list:
            cursor.execute("""
                INSERT INTO telegraph_clues
                (puzzle_type, puzzle_number, puzzle_date, clue_number, direction,
                 clue_text, enumeration, answer, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                puzzle_type,
                str(puzzle_number),
                date_publish,
                str(clue['number']),
                direction,
                clue['clue'],
                clue['enumeration'],
                clue['answer'],
                fetched_at
            ))
            clue_count += 1

    conn.commit()
    conn.close()

    print(f"  Saved {clue_count} clues")
    return 'fetched'


def main():
    today = date.today()
    print("=" * 60)
    print("TELEGRAPH DAILY SCRAPER")
    print(f"Today: {today.strftime('%A, %d %B %Y')}")
    print("=" * 60)
    print(f"Database: {DB_PATH}")

    # Launch browser
    print("\nLaunching browser...")
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")

    driver = uc.Chrome(options=options, version_main=144)

    try:
        login(driver)
        puzzles = harvest_today(driver)
    except Exception as e:
        print(f"Error during harvest: {e}")
        screenshot = SCRIPT_DIR / "telegraph_error.png"
        try:
            driver.save_screenshot(str(screenshot))
            print(f"Screenshot saved: {screenshot}")
        except Exception:
            pass
        return
    finally:
        try:
            driver.quit()
        except Exception:
            pass
        print("Browser closed.")

    if not puzzles:
        print("\nNo puzzles found for today.")
        return

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


if __name__ == "__main__":
    main()
