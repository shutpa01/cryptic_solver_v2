#!/usr/bin/env python3
"""Telegraph URL Harvester - uses undetected-chromedriver to evade bot detection.

First run: pip install undetected-chromedriver python-dotenv

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

# Load environment variables
load_dotenv()

# Store state in script directory
SCRIPT_DIR = Path(__file__).parent
STATE_FILE = SCRIPT_DIR / "telegraph_api_state.json"

PUZZLES_URL = "https://www.telegraph.co.uk/puzzles/"
LOGIN_URL = "https://secure.telegraph.co.uk/customer/secure/login/"


def get_credentials():
    """Get credentials from .env file."""
    email = os.getenv("TELEGRAPH_EMAIL")
    password = os.getenv("TELEGRAPH_PASSWORD")

    if not email or not password:
        raise ValueError("Missing TELEGRAPH_EMAIL or TELEGRAPH_PASSWORD in .env file")

    return {"email": email, "password": password}


def save_state(api_ids):
    """Save found API IDs to state file."""
    with open(STATE_FILE, 'w') as f:
        json.dump(api_ids, f, indent=2)


def main():
    today = date.today()
    print("=" * 60)
    print("TELEGRAPH URL HARVESTER (undetected-chromedriver)")
    print(f"Date: {today.strftime('%A, %d %B %Y')}")
    print("=" * 60)

    creds = get_credentials()

    print("\nLaunching undetected Chrome...")

    # Configure undetected Chrome
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    # Uncomment to run headless (no visible window):
    # options.add_argument("--headless=new")

    driver = uc.Chrome(options=options, version_main=144)

    try:
        # Go to login page
        print(f"Navigating to login page...")
        driver.get(LOGIN_URL)

        wait = WebDriverWait(driver, 15)

        # Enter email
        print("Entering email...")
        email_field = wait.until(EC.presence_of_element_located((By.ID, "email")))
        email_field.clear()
        email_field.send_keys(creds["email"])

        # Click continue
        print("Clicking continue...")
        continue_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.screen-cta")))
        continue_btn.click()

        # Wait for password field
        print("Entering password...")
        password_field = wait.until(EC.presence_of_element_located((By.ID, "password")))
        password_field.clear()
        password_field.send_keys(creds["password"])

        # Click login
        print("Clicking login...")
        login_btn = wait.until(
            EC.element_to_be_clickable((By.CSS_SELECTOR, "a.screen-cta")))
        login_btn.click()

        # Wait for redirect to complete
        print("Waiting for login to complete...")
        wait.until(lambda d: "login" not in d.current_url.lower())

        print(f"Logged in! Current URL: {driver.current_url}")

        # Navigate to puzzles page
        print(f"Navigating to puzzles page...")
        driver.get(PUZZLES_URL)

        # Wait for puzzle content to load - the app is JavaScript-rendered
        print("Waiting for puzzles to load (JS app)...")
        time.sleep(20)  # Give JS plenty of time to render

        # Get all links on the page
        all_links = driver.find_elements(By.TAG_NAME, "a")
        print(f"\nFound {len(all_links)} links on page")

        puzzles = {}
        today_str = f"{today.day} {today.strftime('%b')}, {today.year}"  # "25 Jan, 2026"

        print(f"Looking for puzzles dated: {today_str}")

        for link in all_links:
            try:
                href = link.get_attribute("href") or ""
                text = link.text.strip()

                # Only process cryptic/toughie links with dates (today's puzzles)
                if not href or "Archive" in text:
                    continue

                # Check if it's a puzzle we care about
                is_cryptic = 'cryptic' in href.lower()
                is_toughie = 'toughie' in href.lower()

                if not (is_cryptic or is_toughie):
                    continue

                # Debug: print what we found
                print(f"  Checking: {text[:30]}... -> {href[:80]}...")

                # Extract API ID from URL
                match = re.search(r'#crossword/([^/]+)/(.+)-(\d+)', href)
                if not match:
                    print(f"    No API ID match in URL")
                    continue

                folder, ptype, api_id = match.groups()

                # Extract date from link text
                date_match = re.search(r'(\d{1,2} \w{3}, \d{4})', text)
                link_date = date_match.group(1) if date_match else None

                # Extract puzzle number from URL
                num_match = re.search(r'number=(\d+)', href)
                puzzle_num = num_match.group(1) if num_match else None

                print(f"    FOUND: {ptype}: #{puzzle_num} -> API {api_id} ({link_date})")

                # Create unique key
                key = f"{ptype}-{api_id}"
                if key not in puzzles:
                    puzzles[key] = {
                        'name': text.split('\n')[0] if '\n' in text else text,
                        'folder': folder,
                        'type': ptype,
                        'api_id': api_id,
                        'puzzle_number': puzzle_num,
                        'date': link_date,
                        'url': href
                    }

            except Exception as e:
                print(f"  Error: {e}")

        print("\n" + "=" * 60)
        print("ALL CRYPTIC/TOUGHIE PUZZLES FOUND")
        print("=" * 60)

        for key, data in sorted(puzzles.items()):
            print(
                f"{data['type']}: #{data['puzzle_number']} -> API {data['api_id']} ({data['date']})")

        # Save full harvest to JSON
        if puzzles:
            harvest_file = SCRIPT_DIR / "telegraph_harvest.json"
            harvest_data = {
                'harvested_at': today.isoformat(),
                'puzzles': list(puzzles.values())
            }
            with open(harvest_file, 'w') as f:
                json.dump(harvest_data, f, indent=2)
            print(f"\nSaved {len(puzzles)} puzzles to {harvest_file}")

        return puzzles

    except Exception as e:
        print(f"Error: {e}")
        # Save screenshot for debugging
        screenshot_file = SCRIPT_DIR / "telegraph_error.png"
        driver.save_screenshot(str(screenshot_file))
        print(f"Saved screenshot to {screenshot_file}")
        raise

    finally:
        driver.quit()
        print("\nBrowser closed.")


if __name__ == "__main__":
    main()