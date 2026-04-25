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
    total = sum(len(v) for v in data.values() if isinstance(v, dict))
    print(f"  [Saved {total} API URLs]")


def dismiss_cookie_consent(driver):
    """Dismiss the cookie consent popup if present."""
    import time as _time
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
        _time.sleep(1)
    except Exception:
        driver.switch_to.default_content()


def login(driver, creds):
    """Log in to Telegraph."""
    print("Navigating to login page...")
    driver.get(LOGIN_URL)
    dismiss_cookie_consent(driver)

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
                    api_url = f"https://puzzlesdata.telegraph.co.uk/puzzles/{folder}/{ptype}-{api_id}.json"
                    puzzles[key] = {
                        'type': ptype,
                        'puzzle_number': puzzle_num,
                        'api_id': api_id,
                        'folder': folder,
                        'url': api_url
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
    """Navigate directly to the puzzle type page."""
    url = f"https://www.telegraph.co.uk/puzzles/puzzle/{puzzle_type}/"
    print(f"  Navigating to {url}")
    driver.get(url)
    time.sleep(5)
    dismiss_overlays(driver)
    return True


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


def harvest_all_puzzles(driver, harvest_data, start_year=2017,
                        test_type=None, test_year=None, test_month=None):
    """Navigate through all dates and harvest puzzle URLs."""
    current_year = date.today().year
    current_month = date.today().month
    months = ['January', 'February', 'March', 'April', 'May', 'June',
              'July', 'August', 'September', 'October', 'November', 'December']

    total_harvested = 0
    total_skipped = 0

    # Track completed months so we can resume after a crash
    completed = set()
    if '_completed_months' in harvest_data:
        completed = set(harvest_data['_completed_months'])

    # In test mode, restrict to a single type/year/month
    types_to_process = [test_type] if test_type else PUZZLE_TYPES

    for puzzle_type in types_to_process:
        print(f"\n{'=' * 60}")
        print(f"HARVESTING: {puzzle_type}")
        print(f"{'=' * 60}")

        if puzzle_type not in harvest_data:
            harvest_data[puzzle_type] = {}

        # Select puzzle type filter
        navigate_to_puzzle_type(driver, puzzle_type)
        time.sleep(3)

        # Go through each year from current year back to start_year
        if test_year:
            years_to_process = [test_year]
        else:
            years_to_process = range(current_year, start_year - 1, -1)

        for year in years_to_process:
            # Only process up to current month for current year
            if test_month:
                months_to_process = [test_month]
            elif year == current_year:
                months_to_process = months[:current_month]
            else:
                months_to_process = months

            # Skip entire year if all its months are already completed
            all_done = all(
                f"{puzzle_type}:{year}:{m}" in completed
                for m in months_to_process
            )
            if all_done:
                print(f"\n  Year {year}: all months completed (skipping)")
                continue

            print(f"\n  Year {year}:")

            # Open date picker
            print(f"    [debug] opening date picker...")
            if not click_date_button(driver):
                print(f"    Could not open date picker for year {year}")
                continue
            print(f"    [debug] date picker opened")

            time.sleep(1)

            # Navigate to correct year by clicking left arrow
            print(f"    [debug] reading picker year...")
            picker_year = get_current_picker_year(driver)
            print(f"    [debug] picker year = {picker_year}")
            if picker_year:
                while picker_year > year:
                    print(f"    [debug] clicking left arrow ({picker_year} -> {year})...")
                    if not click_year_arrow(driver, 'left'):
                        print(f"    Could not navigate to year {year}")
                        break
                    time.sleep(0.5)
                    picker_year = get_current_picker_year(driver)
                    print(f"    [debug] picker year now = {picker_year}")

            # Now process each month in this year
            for month in months_to_process:
                month_key = f"{puzzle_type}:{year}:{month}"
                if month_key in completed:
                    print(f"    {month}: already completed (skipping)")
                    continue

                # Select the month
                print(f"    [debug] selecting {month}...")
                if not select_month(driver, month):
                    print(f"    {month}: could not select")
                    # Re-open picker for next month
                    print(f"    [debug] re-opening picker after failed select...")
                    click_date_button(driver)
                    time.sleep(2)
                    continue

                time.sleep(5)  # Wait for puzzles to load - needs several seconds

                # Extract puzzles from page
                puzzles = extract_puzzles_from_page(driver)

                new_count = 0
                month_nums = []
                for key, puzzle in puzzles.items():
                    if puzzle['type'] != puzzle_type:
                        continue

                    pnum = puzzle['puzzle_number']
                    month_nums.append(int(pnum))
                    if pnum not in harvest_data[puzzle_type]:
                        harvest_data[puzzle_type][pnum] = {
                            'api_id': puzzle['api_id'],
                            'folder': puzzle['folder'],
                            'url': puzzle['url']
                        }
                        new_count += 1
                        total_harvested += 1
                    else:
                        total_skipped += 1

                # Mark month as completed and save
                completed.add(month_key)
                harvest_data['_completed_months'] = sorted(completed)

                # Log puzzle numbers found this month
                if month_nums:
                    month_nums.sort()
                    print(f"    {month}: {len(month_nums)} found (#{month_nums[0]}-#{month_nums[-1]}), {new_count} new")
                    # Check for gaps in sequence
                    expected = set(range(month_nums[0], month_nums[-1] + 1))
                    missing = sorted(expected - set(month_nums))
                    if missing:
                        print(f"      GAPS: {missing}")
                else:
                    print(f"    {month}: 0 found")
                save_harvest(harvest_data)

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

    # Per-type summary: full range and gaps
    print(f"\n{'=' * 60}")
    print("SEQUENCE ANALYSIS")
    print(f"{'=' * 60}")
    for puzzle_type in types_to_process:
        if puzzle_type not in harvest_data or not harvest_data[puzzle_type]:
            continue
        all_nums = sorted(int(p) for p in harvest_data[puzzle_type].keys())
        expected = set(range(all_nums[0], all_nums[-1] + 1))
        missing = sorted(expected - set(all_nums))
        print(f"\n  {puzzle_type}: {len(all_nums)} puzzles, #{all_nums[0]}-#{all_nums[-1]}")
        if missing:
            print(f"    MISSING ({len(missing)}): {missing[:50]}")
            if len(missing) > 50:
                print(f"    ... and {len(missing) - 50} more")
        else:
            print(f"    No gaps in sequence")

    return total_harvested, total_skipped


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Telegraph Backfill Harvester")
    parser.add_argument("--test", action="store_true",
                        help="Test mode: harvest one puzzle type, one month only")
    parser.add_argument("--type", type=str, default=None,
                        help="Puzzle type to harvest (e.g. cryptic-crossword)")
    parser.add_argument("--year", type=int, default=None,
                        help="Single year to harvest")
    parser.add_argument("--month", type=str, default=None,
                        help="Single month to harvest (e.g. January)")
    args = parser.parse_args()

    # In --test mode, default to one recent month of cryptic
    if args.test:
        test_type = args.type or 'cryptic-crossword'
        test_year = args.year or 2026
        test_month = args.month or 'March'
        print("=" * 60)
        print(f"TEST MODE: {test_type}, {test_month} {test_year}")
        print("=" * 60)
    else:
        test_type = args.type
        test_year = args.year
        test_month = args.month
        print("=" * 60)
        print("TELEGRAPH BACKFILL HARVESTER")
        print(f"Date: {date.today().strftime('%A, %d %B %Y')}")
        print("=" * 60)

    creds = get_credentials()
    harvest_data = load_existing_harvest()

    existing_count = sum(len(v) for k, v in harvest_data.items()
                         if k != '_completed_months' and isinstance(v, dict))
    print(f"\nExisting harvest: {existing_count} API URLs")

    print("\nLaunching browser...")
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = uc.Chrome(options=options, version_main=146)

    try:
        login(driver, creds)

        harvested, skipped = harvest_all_puzzles(
            driver, harvest_data,
            test_type=test_type, test_year=test_year, test_month=test_month)

        print("\n" + "=" * 60)
        print("HARVEST COMPLETE")
        print("=" * 60)
        print(f"Harvested: {harvested}")
        print(f"Skipped (already had): {skipped}")
        print(f"\nTotal API URLs:")
        for ptype, urls in harvest_data.items():
            if ptype != '_completed_months' and isinstance(urls, dict):
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