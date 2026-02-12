#!/usr/bin/env python3
"""Telegraph Manual Harvest Helper

Opens browser, logs in, then waits for you to:
1. Navigate to the month you want
2. Press Enter to harvest the puzzles shown
3. Repeat until done
4. Type 'quit' to exit

Saves to telegraph_api_mapping.json
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

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
HARVEST_FILE = SCRIPT_DIR / "telegraph_api_mapping.json"

PUZZLES_URL = "https://www.telegraph.co.uk/puzzles/"
LOGIN_URL = "https://secure.telegraph.co.uk/customer/secure/login/"


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


def login(driver, creds):
    """Log in to Telegraph."""
    from selenium.webdriver.support.ui import WebDriverWait
    from selenium.webdriver.support import expected_conditions as EC

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
    print(f"Logged in!")


def main():
    print("=" * 60)
    print("TELEGRAPH MANUAL HARVEST HELPER")
    print(f"Date: {date.today().strftime('%A, %d %B %Y')}")
    print("=" * 60)

    creds = get_credentials()
    harvest_data = load_existing_harvest()

    # Count existing
    total_existing = sum(len(v) for v in harvest_data.values())
    print(f"\nExisting harvest: {total_existing} API URLs")
    for ptype, puzzles in harvest_data.items():
        print(f"  {ptype}: {len(puzzles)}")

    print("\nLaunching browser...")
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = uc.Chrome(options=options, version_main=144)

    try:
        login(driver, creds)

        print(f"\nNavigating to {PUZZLES_URL}")
        driver.get(PUZZLES_URL)
        time.sleep(5)

        print("\n" + "=" * 60)
        print("MANUAL HARVEST MODE")
        print("=" * 60)
        print("\nInstructions:")
        print("1. Use the browser to navigate to a month you want to harvest")
        print("2. Wait for puzzles to load")
        print("3. Come back here and press ENTER to harvest")
        print("4. Repeat for each month")
        print("5. Type 'quit' or 'q' to exit\n")

        session_harvested = 0

        while True:
            user_input = input(
                "\nPress ENTER to harvest current page (or 'quit' to exit): ").strip().lower()

            if user_input in ['quit', 'q', 'exit']:
                break

            # Extract puzzles from current page
            puzzles = extract_puzzles_from_page(driver)

            if not puzzles:
                print("  No puzzles found on this page. Make sure puzzles are loaded.")
                continue

            # Group by type and add to harvest
            new_count = 0
            skipped_count = 0

            for key, puzzle in puzzles.items():
                ptype = puzzle['type']
                pnum = puzzle['puzzle_number']

                if ptype not in harvest_data:
                    harvest_data[ptype] = {}

                if pnum not in harvest_data[ptype]:
                    harvest_data[ptype][pnum] = {
                        'api_id': puzzle['api_id'],
                        'folder': puzzle['folder']
                    }
                    new_count += 1
                else:
                    skipped_count += 1

            session_harvested += new_count

            # Save after each harvest
            save_harvest(harvest_data)

            # Report
            print(f"  Harvested: +{new_count} new, {skipped_count} already had")

            # Show totals by type
            print(f"  Totals:")
            for ptype, puzzles in sorted(harvest_data.items()):
                print(f"    {ptype}: {len(puzzles)}")

        print("\n" + "=" * 60)
        print("SESSION COMPLETE")
        print("=" * 60)
        print(f"Added this session: {session_harvested}")
        print(f"\nFinal totals:")
        for ptype, puzzles in sorted(harvest_data.items()):
            print(f"  {ptype}: {len(puzzles)}")
        total_final = sum(len(v) for v in harvest_data.values())
        print(f"  TOTAL: {total_final}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

    finally:
        driver.quit()
        print("\nBrowser closed.")


if __name__ == "__main__":
    main()