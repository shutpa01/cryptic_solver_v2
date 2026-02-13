#!/usr/bin/env python3
"""Telegraph Gap-Fill Script

Logs in via browser, navigates to specific months in the date picker,
harvests API IDs for missing puzzles, and fetches them.

Targets:
  - Toughies 3402-3420 (January 2025)
  - Toughies 3629-3634 (February 2026)
  - Cryptics 31125-31160 (January-February 2026)
"""

import re
import os
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

TYPE_MAP = {
    'cryptic-crossword': 'cryptic',
    'prize-cryptic': 'saturday-cryptic',
    'toughie-crossword': 'toughie',
    'prize-toughie': 'prize-toughie',
}

# Months to visit: (year, month_name)
TARGET_MONTHS = [
    (2025, 'January'),
    (2026, 'January'),
    (2026, 'February'),
]


def dismiss_cookie_consent(driver):
    """Dismiss the cookie consent popup if present."""
    try:
        wait = WebDriverWait(driver, 5)
        iframe = wait.until(EC.presence_of_element_located(
            (By.CSS_SELECTOR, "iframe[id*='sp_message_iframe']")))
        driver.switch_to.frame(iframe)
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


def find_clickable(driver, selectors, timeout=15):
    """Try multiple selectors and return the first clickable element found."""
    wait = WebDriverWait(driver, timeout)
    for selector_type, selector in selectors:
        try:
            elem = wait.until(EC.element_to_be_clickable((selector_type, selector)))
            return elem
        except Exception:
            continue
    return None


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

    wait = WebDriverWait(driver, 20)

    print("Entering email...")
    email_field = wait.until(EC.presence_of_element_located((By.ID, "email")))
    email_field.clear()
    email_field.send_keys(email)
    time.sleep(1)

    print("Clicking continue...")
    continue_btn = find_clickable(driver, [
        (By.CSS_SELECTOR, "button[type='submit']"),
        (By.XPATH, "//button[contains(text(), 'Continue')]"),
        (By.CSS_SELECTOR, "a.screen-cta"),
    ])
    if continue_btn:
        js_click(driver, continue_btn)
    else:
        raise RuntimeError("Could not find Continue button")

    time.sleep(3)

    # Check for technical difficulties - retry once
    try:
        error_elem = driver.find_element(By.XPATH,
            "//*[contains(text(), 'technical difficulties')]")
        if error_elem:
            print("  Technical difficulties detected, retrying in 10s...")
            time.sleep(10)
            continue_btn = find_clickable(driver, [
                (By.CSS_SELECTOR, "button[type='submit']"),
                (By.XPATH, "//button[contains(text(), 'Continue')]"),
                (By.CSS_SELECTOR, "a.screen-cta"),
            ])
            if continue_btn:
                js_click(driver, continue_btn)
            time.sleep(3)
    except Exception:
        pass

    print("Entering password...")
    password_field = wait.until(EC.presence_of_element_located((By.ID, "password")))
    password_field.clear()
    password_field.send_keys(password)
    time.sleep(1)

    # Check if already logged in (page navigated after password entry)
    if "login" not in driver.current_url.lower():
        print(f"Already logged in! URL: {driver.current_url}")
        return

    print("Clicking login...")
    login_btn = find_clickable(driver, [
        (By.CSS_SELECTOR, "button[type='submit']"),
        (By.XPATH, "//button[contains(text(), 'Log in')]"),
        (By.XPATH, "//button[contains(text(), 'Sign in')]"),
        (By.CSS_SELECTOR, "a.screen-cta"),
    ], timeout=5)
    if login_btn:
        js_click(driver, login_btn)
    else:
        # Maybe auto-submitted, check URL
        if "login" not in driver.current_url.lower():
            print(f"Logged in! URL: {driver.current_url}")
            return
        raise RuntimeError("Could not find Login button")

    print("Waiting for login to complete...")
    wait.until(lambda d: "login" not in d.current_url.lower())
    print(f"Logged in! URL: {driver.current_url}")


def js_click(driver, element):
    """Click element using JavaScript to bypass intercept issues."""
    driver.execute_script("arguments[0].click();", element)


def click_date_button(driver):
    """Click the Date filter button to open the date picker."""
    try:
        buttons = driver.find_elements(By.CSS_SELECTOR, "button.toggle")
        for btn in buttons:
            btn_text = btn.text.strip()
            if btn_text.startswith("Date") or re.match(r'^[A-Z][a-z]{2} \d{4}', btn_text):
                js_click(driver, btn)
                time.sleep(2)
                return True
        print(f"  Toggle button texts: {[b.text.strip()[:20] for b in buttons]}")
        return False
    except Exception as e:
        print(f"  Date button error: {e}")
        return False


def get_current_picker_year(driver):
    """Get the currently displayed year in the date picker."""
    try:
        year_elem = driver.find_element(By.CSS_SELECTOR, "span.year")
        return int(year_elem.text)
    except:
        return None


def click_year_arrow(driver, direction='left'):
    """Click the year navigation arrow."""
    try:
        if direction == 'left':
            arrow = driver.find_element(By.CSS_SELECTOR, "button.select-prev-year")
        else:
            arrow = driver.find_element(By.CSS_SELECTOR, "button.select-next-year")
        js_click(driver, arrow)
        time.sleep(0.5)
        return True
    except Exception as e:
        print(f"  Year arrow error: {e}")
        return False


def navigate_to_year(driver, target_year):
    """Navigate the date picker to the target year."""
    picker_year = get_current_picker_year(driver)
    if not picker_year:
        print("  Could not read picker year")
        return False

    while picker_year > target_year:
        if not click_year_arrow(driver, 'left'):
            return False
        time.sleep(0.5)
        picker_year = get_current_picker_year(driver)

    while picker_year < target_year:
        if not click_year_arrow(driver, 'right'):
            return False
        time.sleep(0.5)
        picker_year = get_current_picker_year(driver)

    return picker_year == target_year


def select_month(driver, month_name):
    """Click on a specific month in the date picker."""
    try:
        month_div = driver.find_element(By.XPATH,
            f"//div[contains(@class, 'month') and text()='{month_name}']")
        js_click(driver, month_div)
        time.sleep(5)
        return True
    except:
        try:
            month_div = driver.find_element(By.XPATH,
                f"//div[contains(@class, 'month') and contains(text(), '{month_name[:3]}')]")
            js_click(driver, month_div)
            time.sleep(5)
            return True
        except:
            return False


def extract_puzzles_from_page(driver):
    """Extract all puzzle links from the current page."""
    puzzles = []
    all_links = driver.find_elements(By.TAG_NAME, "a")

    for link in all_links:
        try:
            href = link.get_attribute("href") or ""
            text = link.text.strip()

            if not href or "Archive" in text:
                continue

            match = re.search(r'#crossword/([^/]+)/(.+)-(\d+)', href)
            if not match:
                continue

            folder, link_type, api_id = match.groups()

            # Only interested in cryptic and toughie types
            if link_type not in TYPE_MAP:
                continue

            num_match = re.search(r'number=(\d+)', href)
            puzzle_num = num_match.group(1) if num_match else None

            if puzzle_num:
                puzzles.append({
                    'folder': folder,
                    'link_type': link_type,
                    'puzzle_type': TYPE_MAP[link_type],
                    'api_id': api_id,
                    'puzzle_number': puzzle_num,
                })
        except:
            continue

    return puzzles


def puzzle_already_fetched(puzzle_type, puzzle_number):
    """Check if puzzle is already in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()
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
        return 'skipped'

    url = f"https://puzzlesdata.telegraph.co.uk/puzzles/{folder}/{link_type}-{api_id}.json"

    try:
        response = requests.get(url, timeout=30)
        if response.status_code != 200:
            print(f"    HTTP {response.status_code} for {puzzle_type} #{puzzle_number}")
            return 'failed'
    except Exception as e:
        print(f"    Error fetching {puzzle_type} #{puzzle_number}: {e}")
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

    clues_groups = copy.get('clues', [])
    across = []
    down = []

    for group in clues_groups:
        direction = group.get('title', '').lower()
        for clue in group.get('clues', []):
            clue_obj = {
                'number': clue.get('number', ''),
                'clue': clue.get('clue', ''),
                'answer': clue.get('answer', ''),
                'enumeration': clue.get('format', '')
            }
            if direction == 'across':
                across.append(clue_obj)
            elif direction == 'down':
                down.append(clue_obj)

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
    return 'fetched'


def main():
    print("=" * 60)
    print("TELEGRAPH GAP-FILL")
    print("=" * 60)
    print(f"Database: {DB_PATH}")
    print(f"Target months: {TARGET_MONTHS}")

    print("\nLaunching browser...")
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = uc.Chrome(options=options, version_main=144)

    all_puzzles = []

    try:
        login(driver)

        # Archive pages with their target months
        ARCHIVES = [
            {
                'type': 'toughie-crossword',
                'url': 'https://www.telegraph.co.uk/puzzles/puzzle/toughie-crossword/?icid=engagement_onsite-asset_Onsite-asset_04-03_NCC-11_Toughie_archive_HP',
                'months': [(2025, 'January'), (2026, 'February')],
            },
            {
                'type': 'cryptic-crossword',
                'url': 'https://www.telegraph.co.uk/puzzles/puzzle/cryptic-crossword/',
                'months': [(2026, 'January'), (2026, 'February')],
            },
        ]

        for archive in ARCHIVES:
            archive_type = archive['type']
            print(f"\n{'=' * 50}")
            print(f"  ARCHIVE: {archive_type}")
            print(f"{'=' * 50}")

            # Load archive page
            driver.get(archive['url'])
            time.sleep(15)

            # First harvest default view (recent puzzles)
            puzzles = extract_puzzles_from_page(driver)
            new_puzzles = [p for p in puzzles if not puzzle_already_fetched(
                p['puzzle_type'], p['puzzle_number'])]
            if new_puzzles:
                print(f"  Default view: {len(new_puzzles)} new puzzles")
                for p in new_puzzles:
                    print(f"    {p['puzzle_type']} #{p['puzzle_number']} (API {p['api_id']})")
                all_puzzles.extend(new_puzzles)

            # Now navigate to each target month using the date picker
            for target_year, target_month in archive['months']:
                print(f"\n  --- {target_month} {target_year} ---")

                if not click_date_button(driver):
                    print("    Could not open date picker")
                    continue
                time.sleep(1)

                if not navigate_to_year(driver, target_year):
                    print(f"    Could not navigate to year {target_year}")
                    continue

                if not select_month(driver, target_month):
                    print(f"    Could not select {target_month}")
                    continue

                time.sleep(5)

                puzzles = extract_puzzles_from_page(driver)
                new_puzzles = [p for p in puzzles if not puzzle_already_fetched(
                    p['puzzle_type'], p['puzzle_number'])]

                print(f"    Found {len(puzzles)} puzzles, {len(new_puzzles)} new")
                for p in new_puzzles:
                    print(f"      {p['puzzle_type']} #{p['puzzle_number']} (API {p['api_id']})")

                all_puzzles.extend(new_puzzles)

    except Exception as e:
        print(f"\nError during harvest: {e}")
        import traceback
        traceback.print_exc()
        screenshot = SCRIPT_DIR / "telegraph_gapfill_error.png"
        driver.save_screenshot(str(screenshot))
        print(f"Screenshot saved: {screenshot}")
    finally:
        driver.quit()
        print("\nBrowser closed.")

    if not all_puzzles:
        print("\nNo new puzzles to fetch.")
        return

    # Fetch and save
    print(f"\n{'=' * 60}")
    print(f"FETCHING {len(all_puzzles)} PUZZLES")
    print(f"{'=' * 60}")

    stats = {'fetched': 0, 'skipped': 0, 'failed': 0}

    for puzzle in all_puzzles:
        result = fetch_and_save(puzzle)
        stats[result] += 1
        if result == 'fetched':
            print(f"  {puzzle['puzzle_type']} #{puzzle['puzzle_number']}: saved")

    print(f"\n{'=' * 60}")
    print(f"DONE - Fetched: {stats['fetched']}, "
          f"Skipped: {stats['skipped']}, Failed: {stats['failed']}")
    print(f"{'=' * 60}")


if __name__ == "__main__":
    main()
