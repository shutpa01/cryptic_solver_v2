#!/usr/bin/env python3
"""Times Crossword Backfill Harvester

Logs in, navigates archive page (year -> month -> week), collects all puzzle links,
visits each puzzle to capture API URL.

Requires .env file with:
TIMES_EMAIL=your_email
TIMES_PASSWORD=your_password
"""

import re
import os
import json
import time
import requests
from datetime import date
from pathlib import Path
from dotenv import load_dotenv

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By
from selenium.webdriver.support.ui import WebDriverWait
from selenium.webdriver.support import expected_conditions as EC

load_dotenv()

SCRIPT_DIR = Path(__file__).parent
HARVEST_FILE = SCRIPT_DIR / "times_api_mapping.json"
PROGRESS_FILE = SCRIPT_DIR / "times_harvest_progress.json"

LOGIN_URL = "https://login.thetimes.co.uk/"
ARCHIVE_URL = "https://www.thetimes.com/html-puzzles-sitemap"


def get_credentials():
    """Get credentials from .env file."""
    email = os.getenv("TIMES_EMAIL")
    password = os.getenv("TIMES_PASSWORD")
    if not email or not password:
        raise ValueError("Missing TIMES_EMAIL or TIMES_PASSWORD in .env file")
    return {"email": email, "password": password}


def load_existing_harvest():
    """Load existing harvest data if it exists."""
    if HARVEST_FILE.exists():
        with open(HARVEST_FILE, 'r') as f:
            data = json.load(f)
            # Remove quick-cryptic if present
            if 'quick-cryptic' in data:
                del data['quick-cryptic']
            return data
    return {'cryptic': {}, 'sunday-cryptic': {}}


def save_harvest(data):
    """Save harvest data to JSON file."""
    with open(HARVEST_FILE, 'w') as f:
        json.dump(data, f, indent=2)
    total = sum(len(v) for v in data.values())
    print(f"  [Saved {total} API URLs]")


def save_progress(year, month, week):
    """Save progress in case we need to resume."""
    progress = {'year': year, 'month': month, 'week': week}
    with open(PROGRESS_FILE, 'w') as f:
        json.dump(progress, f)


def login(driver, creds):
    """Log in to The Times."""
    print("Navigating to login page...")
    driver.get(LOGIN_URL)
    time.sleep(3)

    wait = WebDriverWait(driver, 15)

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
    email_field.send_keys(creds["email"])

    print("Entering password...")
    password_field = wait.until(EC.presence_of_element_located(
        (By.CSS_SELECTOR, "input.auth0-lock-input[name='password']")))
    password_field.clear()
    password_field.send_keys(creds["password"])

    print("Clicking login...")
    login_btn = wait.until(
        EC.element_to_be_clickable((By.CSS_SELECTOR, "button.auth0-lock-submit")))
    login_btn.click()

    print("Waiting for login to complete...")
    driver.switch_to.default_content()
    wait.until(lambda d: "login" not in d.current_url.lower())
    print(f"Logged in! URL: {driver.current_url}")


def extract_api_url(driver, puzzle_url):
    """Visit a puzzle page and extract the API URL from network logs."""
    driver.get(puzzle_url)
    time.sleep(5)

    # Check network logs
    api_url = None
    log_count = 0
    feeds_urls = []

    try:
        logs = driver.get_log('performance')
        log_count = len(logs)

        for entry in logs:
            try:
                log_data = json.loads(entry['message'])
                message = log_data.get('message', {})

                if message.get('method') in ['Network.requestWillBeSent',
                                             'Network.responseReceived']:
                    params = message.get('params', {})
                    request_url = params.get('request', {}).get('url', '')
                    if not request_url:
                        request_url = params.get('response', {}).get('url', '')

                    # Track any feeds.thetimes URLs for debugging
                    if 'feeds.thetimes' in request_url:
                        feeds_urls.append(request_url[:80])

                    if 'feeds.thetimes' in request_url and 'data.json' in request_url:
                        api_url = request_url
                        break
            except:
                continue
    except Exception as e:
        print(f"  [Log error: {e}]")

    if api_url:
        return api_url

    # Debug info
    print(f"  [Logs: {log_count}, feeds URLs: {len(feeds_urls)}]")
    if feeds_urls:
        for url in feeds_urls[:3]:
            print(f"    {url}...")

    # Fallback: search page source
    try:
        page_source = driver.page_source
        match = re.search(
            r'(https://feeds\.thetimes\.(?:com|co\.uk)/puzzles/[^"\']+/\d{8}/\d+/data\.json)',
            page_source
        )
        if match:
            print(f"  [Found in page source]")
            return match.group(1)

        # Try finding API ID in page source another way
        match = re.search(r'"apiId"\s*:\s*(\d+)', page_source)
        if match:
            print(f"  [Found apiId: {match.group(1)}]")
    except:
        pass

    return None


def collect_and_extract_from_archive(driver, harvest_data, creds, max_years=1,
                                     start_week=1, start_year=None):
    """Navigate archive, click each puzzle, capture API URL."""
    print("\n*** CLICK-THROUGH MODE - will click each puzzle ***")
    print(f"Navigating to archive: {ARCHIVE_URL}")
    driver.get(ARCHIVE_URL)
    time.sleep(5)

    # Find all week links - they have class "Sitemap-list-week"
    print("\nLooking for week links...")

    week_links = driver.find_elements(By.CSS_SELECTOR, "a.Sitemap-list-week")

    if not week_links:
        # Fallback: find by href pattern
        print("No elements with class 'Sitemap-list-week', trying href pattern...")
        all_links = driver.find_elements(By.TAG_NAME, "a")
        week_links = []
        for link in all_links:
            href = link.get_attribute("href") or ""
            if re.search(r'/html-puzzles-sitemap/\d{4}-\d{2}-\d', href):
                week_links.append(link)

    print(f"Found {len(week_links)} week links")

    if not week_links:
        print("ERROR: No week links found!")
        return 0

    # Extract info from each week link
    weeks_to_process = []
    current_year = date.today().year
    min_year = current_year - max_years

    for link in week_links:
        href = link.get_attribute("href") or ""
        text = link.text.strip()

        match = re.search(r'/html-puzzles-sitemap/(\d{4})-(\d{2})-(\d)', href)
        if match:
            year = int(match.group(1))
            month = int(match.group(2))
            week_num = int(match.group(3))

            if href.startswith('/'):
                href = f"https://www.thetimes.com{href}"

            if year >= min_year:
                weeks_to_process.append({
                    'year': year,
                    'month': month,
                    'week_num': week_num,
                    'href': href,
                    'text': text
                })

    weeks_to_process.sort(key=lambda x: (x['year'], x['month'], x['week_num']),
                          reverse=True)

    print(f"Will process {len(weeks_to_process)} weeks (from {min_year} onwards)")

    month_names = ['', 'January', 'February', 'March', 'April', 'May', 'June',
                   'July', 'August', 'September', 'October', 'November', 'December']

    total_harvested = 0
    total_skipped = 0
    total_errors = 0
    puzzles_since_restart = 0
    RESTART_EVERY = 50  # Restart browser every N puzzles

    for i, week in enumerate(weeks_to_process):
        # Skip weeks before start_week
        if i + 1 < start_week:
            continue

        # Skip years after start_year (list is reverse chronological)
        if start_year and week['year'] > start_year:
            continue

        year = week['year']
        month = week['month']
        month_name = month_names[month]
        week_text = week['text']
        week_href = week['href']

        print(
            f"\n[Week {i + 1}/{len(weeks_to_process)}] {month_name} {year}: {week_text}")
        print(f"  Navigating to week page: {week_href[:60]}...")

        # Navigate to week page
        driver.get(week_href)
        time.sleep(3)

        # Find puzzle links on this page
        print(f"  Scanning for puzzle links...")
        puzzle_links = []
        cryptic_found = 0
        sunday_found = 0

        for link in driver.find_elements(By.TAG_NAME, "a"):
            link_href = link.get_attribute("href") or ""
            link_text = link.text.strip()

            if link_href.startswith('/'):
                link_href = f"https://www.thetimes.com{link_href}"

            # Skip quick-cryptic
            if 'quick-cryptic' in link_href.lower():
                continue

            # Times Cryptic (not Sunday)
            if 'times-cryptic-no-' in link_href.lower() and 'sunday' not in link_href.lower():
                cryptic_found += 1
                match = re.search(r'times-cryptic-no-(\d+)', link_href, re.IGNORECASE)
                if match:
                    num = match.group(1)
                    # Skip if already harvested
                    if num not in harvest_data.get('cryptic', {}):
                        puzzle_links.append({
                            'type': 'cryptic',
                            'number': num,
                            'text': link_text
                        })
                    else:
                        total_skipped += 1
                continue

            # Sunday Times Cryptic
            if 'sunday-times-cryptic-no-' in link_href.lower():
                sunday_found += 1
                match = re.search(r'sunday-times-cryptic-no-(\d+)', link_href,
                                  re.IGNORECASE)
                if match:
                    num = match.group(1)
                    if num not in harvest_data.get('sunday-cryptic', {}):
                        puzzle_links.append({
                            'type': 'sunday-cryptic',
                            'number': num,
                            'text': link_text
                        })
                    else:
                        total_skipped += 1
                continue

        print(f"  Found {cryptic_found} cryptic, {sunday_found} sunday on page")

        if not puzzle_links:
            print(f"  No new puzzles to harvest (skipped {total_skipped})")
            continue

        print(
            f"  {len(puzzle_links)} puzzles to harvest: {[p['number'] for p in puzzle_links]}")

        # Click each puzzle and capture API URL
        print(f"  Starting to click puzzles...")
        for puzzle in puzzle_links:
            puzzle_type = puzzle['type']
            puzzle_num = puzzle['number']
            puzzle_text = puzzle['text']

            # Go back to week page
            driver.get(week_href)
            time.sleep(2)

            # Find and click the puzzle link
            print(f"    Looking for {puzzle_type} #{puzzle_num}...", end=" ", flush=True)

            clicked = False
            all_links = driver.find_elements(By.TAG_NAME, "a")

            for link in all_links:
                href = link.get_attribute("href") or ""
                text = link.text.strip()

                # Match by puzzle number in href
                if f"-no-{puzzle_num}" in href:
                    print(f"Found! Clicking...", end=" ", flush=True)

                    # Clear old logs before clicking
                    try:
                        driver.get_log('performance')
                    except:
                        pass

                    # Scroll element into view first
                    try:
                        driver.execute_script(
                            "arguments[0].scrollIntoView({block: 'center'});", link)
                        time.sleep(0.5)
                    except:
                        pass

                    # Try regular click
                    try:
                        link.click()
                        time.sleep(2)
                        # Verify we navigated
                        if f"-no-{puzzle_num}" in driver.current_url or 'cryptic' in driver.current_url.lower():
                            clicked = True
                            break
                    except:
                        pass

                    # Try JavaScript click if regular click didn't navigate
                    if not clicked:
                        try:
                            driver.execute_script("arguments[0].click();", link)
                            time.sleep(2)
                            if f"-no-{puzzle_num}" in driver.current_url or 'cryptic' in driver.current_url.lower():
                                clicked = True
                                break
                        except:
                            pass

                    # Last resort: navigate directly
                    if not clicked:
                        try:
                            # Clear logs before direct navigation
                            try:
                                driver.get_log('performance')
                            except:
                                pass
                            full_href = href if href.startswith(
                                'http') else f"https://www.thetimes.com{href}"
                            driver.get(full_href)
                            time.sleep(2)
                            clicked = True
                            break
                        except:
                            pass

            if not clicked:
                print(f"NOT FOUND")
                total_errors += 1
                continue

            time.sleep(3)  # Initial wait

            # Debug: show current URL
            current_url = driver.current_url
            if 'cryptic' not in current_url.lower():
                print(f"\n      WARNING: Not on puzzle page! URL: {current_url[:60]}...")

            # Check for Play button (older puzzles need this)
            play_clicked = False

            # Enable CDP network tracking BEFORE clicking play
            driver.execute_cdp_cmd('Network.enable', {})

            try:
                play_button = driver.find_element(By.ID, "puzzle-play")
                if play_button:
                    print(f"Play button found, clicking...", end=" ", flush=True)
                    play_button.click()
                    play_clicked = True
            except:
                pass  # No play button, puzzle already loaded

            # Wait for puzzle to load
            time.sleep(8)

            # Try to find the URL in the page source directly
            # The puzzle iframe/script should contain the API URL
            api_url = None
            page_source = driver.page_source

            # Look for the data.json URL pattern - simplified to catch any format
            match = re.search(
                r'(https://feeds\.thetimes\.com/puzzles/[^"\']+/\d{8}/\d+)/(?:data\.json|\?)',
                page_source
            )
            if match:
                api_url = match.group(1) + '/data.json'
                print(f"(found in page)...", end=" ", flush=True)

            # Also try looking for the puzzle ID and date to construct URL
            if not api_url:
                # Look for any puzzles path with date and ID
                match = re.search(
                    r'(feeds\.thetimes\.com/puzzles/[^"\']+)/(\d{8})/(\d+)',
                    page_source
                )
                if match:
                    base_path = match.group(1)
                    date_str = match.group(2)
                    api_id = match.group(3)
                    api_url = f"https://{base_path}/{date_str}/{api_id}/data.json"
                    print(f"(constructed: {api_id})...", end=" ", flush=True)

            # Try iframe src if present
            if not api_url:
                iframes = driver.find_elements(By.TAG_NAME, "iframe")
                for iframe in iframes:
                    src = iframe.get_attribute("src") or ""
                    match = re.search(
                        r'(https://feeds\.thetimes\.com/puzzles/[^"\']+/\d{8}/\d+)', src)
                    if match:
                        api_url = match.group(1) + '/data.json'
                        print(f"(from iframe)...", end=" ", flush=True)
                        break

            # Fallback: check page source for full data.json URL
            if not api_url:
                try:
                    page_source = driver.page_source
                    match = re.search(
                        r'(https://feeds\.thetimes\.(?:com|co\.uk)/puzzles/[^"\']+/\d{8}/\d+/data\.json)',
                        page_source
                    )
                    if match:
                        api_url = match.group(1)
                        print(f"(found in page source)...", end=" ", flush=True)
                except:
                    pass

            # If we have a URL, verify it works and save
            if api_url:
                try:
                    resp = requests.get(api_url, timeout=30)
                    if resp.status_code == 200:
                        # Just save the URL - we'll process data in a separate step
                        if puzzle_type not in harvest_data:
                            harvest_data[puzzle_type] = {}
                        harvest_data[puzzle_type][puzzle_num] = api_url
                        total_harvested += 1
                        save_harvest(harvest_data)
                        print(f"✓ Verified & saved")
                    else:
                        print(f"✗ URL returned {resp.status_code}")
                        total_errors += 1
                except Exception as e:
                    print(f"✗ Fetch error: {e}")
                    total_errors += 1
            else:
                print(f"✗ No API URL found")
                total_errors += 1

            puzzles_since_restart += 1

            # Restart browser periodically to prevent crashes
            if puzzles_since_restart >= RESTART_EVERY:
                print(f"\n  [Restarting browser to prevent crashes...]")
                try:
                    driver.quit()
                except:
                    pass
                time.sleep(3)
                try:
                    driver = create_driver()
                    time.sleep(2)
                    login(driver, creds)
                    puzzles_since_restart = 0
                except Exception as e:
                    print(f"  [Restart failed: {e}, continuing with current browser]")
                    # Try to create a fresh driver without login
                    try:
                        driver = create_driver()
                        time.sleep(2)
                        login(driver, creds)
                        puzzles_since_restart = 0
                    except:
                        print(f"  [Second restart attempt failed, stopping]")
                        break

        save_progress(year, month_name, week_text)

    return total_harvested, total_skipped, total_errors, driver


def create_driver():
    """Create a new browser instance with network logging."""
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    options.set_capability('goog:loggingPrefs', {'performance': 'ALL'})
    return uc.Chrome(options=options, version_main=144)


def main():
    print("=" * 60)
    print("TIMES CROSSWORD BACKFILL HARVESTER")
    print("(Cryptic + Sunday Times Cryptic only)")
    print(f"Date: {date.today().strftime('%A, %d %B %Y')}")
    print("=" * 60)

    creds = get_credentials()
    harvest_data = load_existing_harvest()

    existing_count = sum(len(v) for v in harvest_data.values())
    print(f"\nExisting harvest: {existing_count} API URLs")

    # How many years?
    import sys
    max_years = 1
    start_week = 1
    start_year = None
    for arg in sys.argv[1:]:
        if arg.startswith('--years='):
            max_years = int(arg.split('=')[1])
        elif arg.startswith('--start-week='):
            start_week = int(arg.split('=')[1])
        elif arg.startswith('--start-year='):
            start_year = int(arg.split('=')[1])
    print(f"Will process {max_years} year(s) of archives")
    if start_week > 1:
        print(f"Starting from week {start_week}")
    if start_year:
        print(f"Starting from year {start_year}")

    print("\nLaunching browser...")
    driver = create_driver()

    try:
        # Login
        login(driver, creds)

        # Collect and extract in one pass - clicking each puzzle from week pages
        harvested, skipped, errors, driver = collect_and_extract_from_archive(
            driver, harvest_data, creds, max_years=max_years, start_week=start_week,
            start_year=start_year
        )

        # Summary
        print("\n" + "=" * 60)
        print("HARVEST COMPLETE")
        print("=" * 60)
        print(f"Harvested: {harvested}")
        print(f"Skipped (already had): {skipped}")
        print(f"Errors: {errors}")
        print(f"\nTotal API URLs:")
        for ptype, urls in harvest_data.items():
            print(f"  {ptype}: {len(urls)}")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()

        try:
            driver.save_screenshot(str(SCRIPT_DIR / "times_harvest_error.png"))
        except:
            pass

    finally:
        driver.quit()
        print("\nBrowser closed.")


if __name__ == "__main__":
    main()