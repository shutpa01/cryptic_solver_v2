#!/usr/bin/env python3
"""Telegraph Backfill Harvester - navigates date picker to collect all puzzle URLs.

Uses undetected-chromedriver to navigate the Telegraph puzzles archive,
going through each month/year to harvest puzzle API IDs.

Requires .env file with:
TELEGRAPH_EMAIL=your_email
TELEGRAPH_PASSWORD=your_password
"""

import re
import os
import json
import time
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
HARVEST_FILE = SCRIPT_DIR / "telegraph_api_mapping.json"

PUZZLES_URL = "https://www.telegraph.co.uk/puzzles/"
LOGIN_URL = "https://secure.telegraph.co.uk/customer/secure/login/"

# Puzzle types to harvest
PUZZLE_TYPES = ['cryptic-crossword', 'toughie-crossword', 'prize-toughie',
                'prize-cryptic']


def get_credentials():
    """Get credentials from .env file."""
    email = os.getenv("TELEGRAPH_EMAIL")
    password = os.getenv("TELEGRAPH_PASSWORD")
    if not email or not password:
        raise ValueError("Missing TELEGRAPH_EMAIL or TELEGRAPH_PASSWORD in .env file")
    return {"email": email, "password": password}


def load_existing_harvest():
    """Load existing harvest data if it exists."""
    if HARVEST_FILE.exists():
        with open(HARVEST_FILE, 'r') as f:
            return json.load(f)
    return {}


def save_harvest(data):
    """Save harvest data to JSON file."""
    with open(HARVEST_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    total = sum(len(v) for v in data.values())
    print(f"  [Saved {total} API URLs]")


def login(driver, creds):
    """Log in to Telegraph."""
    print("Navigating to login page...")
    driver.get(LOGIN_URL)

    wait = WebDriverWait(driver, 15)

    print("Entering email...")
    email_field = wait.until(EC.presence_of_element_located((By.ID, "email")))
    email_field.clear()
    email_field.send_keys(creds["email"])

    print("Clicking continue...")
    continue_btn = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "a.screen-cta")))
    continue_btn.click()

    print("Entering password...")
    password_field = wait.until(EC.presence_of_element_located((By.ID, "password")))
    password_field.clear()
    password_field.send_keys(creds["password"])

    print("Clicking login...")
    login_btn = wait.until(EC.element_to_be_clickable((By.CSS_SELECTOR, "a.screen-cta")))
    login_btn.click()

    print("Waiting for login to complete...")
    wait.until(lambda d: "login" not in d.current_url.lower())
    print(f"Logged in! URL: {driver.current_url}")


def extract_puzzles_from_page(driver):
    """Extract all puzzle links from current page."""
    puzzles = {}

    all_links = driver.find_elements(By.TAG_NAME, "a")

    for link in all_links:
        try:
            href = link.get_attribute("href") or ""
            text = link.text.strip()

            if not href or "Archive" in text:
                continue

            # Extract API ID from URL pattern: #crossword/{folder}/{type}-{api_id}
            match = re.search(r'#crossword/([^/]+)/(.+)-(\d+)', href)
            if not match:
                continue

            folder, ptype, api_id = match.groups()

            # Extract puzzle number from URL
            num_match = re.search(r'number=(\d+)', href)
            puzzle_num = num_match.group(1) if num_match else None

            if puzzle_num:
                key = f"{ptype}-{puzzle_num}"
                if key not in puzzles:
                    puzzles[key] = {
                        'type': ptype,
                        'puzzle_number': puzzle_num,
                        'api_id': api_id,
                        'folder': folder
                    }
        except:
            continue

    return puzzles


def dismiss_overlays(driver):
    """Try to dismiss any cookie banners or popups."""
    overlay_selectors = [
        "//button[contains(text(), 'Accept')]",
        "//button[contains(text(), 'OK')]",
        "//button[contains(text(), 'Got it')]",
        "//button[contains(text(), 'Close')]",
        "//*[contains(@class, 'close')]",
        "//*[contains(@class, 'dismiss')]",
        "//button[contains(@class, 'cookie')]",
    ]

    for selector in overlay_selectors:
        try:
            elem = driver.find_element(By.XPATH, selector)
            driver.execute_script("arguments[0].click();", elem)
            time.sleep(1)
        except:
            continue


def js_click(driver, element):
    """Click element using JavaScript to bypass intercept issues."""
    driver.execute_script("arguments[0].click();", element)


def navigate_to_puzzle_type(driver, puzzle_type):
    """Select a puzzle type using the filter dropdown."""
    # First make sure we're on the puzzles page
    if 'puzzles' not in driver.current_url:
        driver.get(PUZZLES_URL)
        time.sleep(5)
        dismiss_overlays(driver)

    # Map internal names to display names
    display_names = {
        'cryptic-crossword': 'Cryptic Crossword',
        'toughie-crossword': 'Toughie',
        'prize-toughie': 'Prize Toughie',
        'prize-cryptic': 'Prize Cryptic'
    }

    display_name = display_names.get(puzzle_type, puzzle_type)
    print(f"  Selecting filter: {display_name}")

    try:
        # Click on "All puzzles" or current filter to open dropdown
        filter_selectors = [
            "//div[contains(text(), 'All puzzles')]",
            "//div[contains(text(), 'Cryptic')]",
            "//div[contains(text(), 'Toughie')]",
            "//button[contains(text(), 'All puzzles')]",
            "//*[contains(@class, 'filter')]//*[contains(@class, 'select')]",
            "//*[contains(@class, 'dropdown')]",
        ]

        filter_btn = None
        for selector in filter_selectors:
            try:
                filter_btn = driver.find_element(By.XPATH, selector)
                break
            except:
                continue

        if filter_btn:
            js_click(driver, filter_btn)
            time.sleep(1)

            # Select the puzzle type from dropdown
            option = driver.find_element(By.XPATH,
                                         f"//*[contains(text(), '{display_name}')]")
            js_click(driver, option)
            time.sleep(3)
            return True
        else:
            print(f"    Could not find filter button")
            return False
    except Exception as e:
        print(f"    Could not select filter: {e}")
        return False


def click_date_button(driver):
    """Click the Date filter button to open the date picker."""
    try:
        # Find button.toggle that starts with "Date" OR contains a month/year pattern
        buttons = driver.find_elements(By.CSS_SELECTOR, "button.toggle")
        for btn in buttons:
            btn_text = btn.text.strip()
            # Match "Date" or month patterns like "Jan 2025", "Feb 2024", etc.
            if btn_text.startswith("Date") or re.match(r'^[A-Z][a-z]{2} \d{4}', btn_text):
                js_click(driver, btn)
                time.sleep(2)  # Give more time for picker to open
                return True
        # Debug: print what we found
        print(f"    Toggle button texts: {[b.text.strip()[:20] for b in buttons]}")
        return False
    except Exception as e:
        print(f"    Date button error: {e}")
        return False


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
        print(f"    Year arrow error: {e}")
        return False


def get_current_picker_year(driver):
    """Get the currently displayed year in the date picker."""
    try:
        year_elem = driver.find_element(By.CSS_SELECTOR, "span.year")
        return int(year_elem.text)
    except:
        return None


def select_month(driver, month_name):
    """Click on a specific month in the date picker."""
    try:
        month_div = driver.find_element(By.XPATH,
                                        f"//div[contains(@class, 'month') and text()='{month_name}']")
        js_click(driver, month_div)
        time.sleep(3)
        return True
    except:
        try:
            # Try partial match
            month_div = driver.find_element(By.XPATH,
                                            f"//div[contains(@class, 'month') and contains(text(), '{month_name[:3]}')]")
            js_click(driver, month_div)
            time.sleep(3)
            return True
        except:
            return False


def harvest_all_puzzles(driver, harvest_data, start_year=2017):
    """Navigate through all dates and harvest puzzle URLs."""
    current_year = date.today().year
    current_month = date.today().month
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

    total_harvested = 0
    total_skipped = 0

    # Navigate to puzzles page first
    print(f"\nNavigating to {PUZZLES_URL}")
    driver.get(PUZZLES_URL)
    time.sleep(5)
    dismiss_overlays(driver)
    time.sleep(2)

    for puzzle_type in PUZZLE_TYPES:
        print(f"\n{'=' * 60}")
        print(f"HARVESTING: {puzzle_type}")
        print(f"{'=' * 60}")

        if puzzle_type not in harvest_data:
            harvest_data[puzzle_type] = {}

        # Select puzzle type filter
        navigate_to_puzzle_type(driver, puzzle_type)
        time.sleep(3)

        # Skip 2026 - start from 2025
        # Go through each year from 2025 back to start_year
        for year in range(current_year - 1, start_year - 1, -1):
            print(f"\n  Year {year}:")

            # All 12 months for past years
            months_to_process = months

            # Open date picker
            if not click_date_button(driver):
                print(f"    Could not open date picker for year {year}")
                continue

            time.sleep(1)

            # Navigate to correct year by clicking left arrow
            picker_year = get_current_picker_year(driver)
            if picker_year:
                while picker_year > year:
                    if not click_year_arrow(driver, 'left'):
                        print(f"    Could not navigate to year {year}")
                        break
                    time.sleep(0.5)
                    picker_year = get_current_picker_year(driver)

            # Now process each month in this year
            for month in months_to_process:
                # Select the month
                if not select_month(driver, month):
                    print(f"    {month}: could not select")
                    # Re-open picker for next month
                    click_date_button(driver)
                    time.sleep(2)
                    continue

                time.sleep(5)  # Wait for puzzles to load - needs several seconds

                # Extract puzzles from page
                puzzles = extract_puzzles_from_page(driver)

                new_count = 0
                for key, puzzle in puzzles.items():
                    if puzzle['type'] != puzzle_type:
                        continue

                    pnum = puzzle['puzzle_number']
                    if pnum not in harvest_data[puzzle_type]:
                        harvest_data[puzzle_type][pnum] = {
                            'api_id': puzzle['api_id'],
                            'folder': puzzle['folder']
                        }
                        new_count += 1
                        total_harvested += 1
                    else:
                        total_skipped += 1

                if new_count > 0:
                    print(f"    {month}: +{new_count} puzzles")
                    save_harvest(harvest_data)
                else:
                    print(f"    {month}: no new puzzles")

                # Re-open date picker for next month (it closes after selecting)
                if not click_date_button(driver):
                    print(f"    Could not re-open date picker")
                    break
                time.sleep(2)

                # Make sure we're still on the right year
                picker_year = get_current_picker_year(driver)
                if picker_year and picker_year != year:
                    while picker_year > year:
                        click_year_arrow(driver, 'left')
                        time.sleep(1)
                        picker_year = get_current_picker_year(driver)

    return total_harvested, total_skipped


def main():
    print("=" * 60)
    print("TELEGRAPH BACKFILL HARVESTER")
    print(f"Date: {date.today().strftime('%A, %d %B %Y')}")
    print("=" * 60)

    creds = get_credentials()
    harvest_data = load_existing_harvest()

    existing_count = sum(len(v) for v in harvest_data.values())
    print(f"\nExisting harvest: {existing_count} API URLs")

    print("\nLaunching browser...")
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = uc.Chrome(options=options)

    try:
        login(driver, creds)

        harvested, skipped = harvest_all_puzzles(driver, harvest_data)

        print("\n" + "=" * 60)
        print("HARVEST COMPLETE")
        print("=" * 60)
        print(f"Harvested: {harvested}")
        print(f"Skipped (already had): {skipped}")
        print(f"\nTotal API URLs:")
        for ptype, urls in harvest_data.items():
            print(f"  {ptype}: {len(urls)}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        driver.save_screenshot(str(SCRIPT_DIR / "telegraph_error.png"))

    finally:
        driver.quit()
        print("\nBrowser closed.")


if __name__ == "__main__":
    main()