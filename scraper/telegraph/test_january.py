#!/usr/bin/env python3
"""Diagnostic: why does the Telegraph date picker fail on January?

Reproduces the exact state the harvest loop is in when it attempts January:
1. Navigate to cryptic-crossword page
2. Open picker, navigate to 2025
3. Click December (puzzles load, picker closes)
4. Re-open picker, navigate to 2024 (simulating the next year iteration)
5. Attempt January — this is where the real harvest fails

Then tries adding delays to see if timing is the issue.
"""

import time
from pathlib import Path
from dotenv import load_dotenv

import undetected_chromedriver as uc
from selenium.webdriver.common.by import By

load_dotenv()

SCRIPT_DIR = Path(__file__).parent

from telegraph_backfill import (
    get_credentials, login, dismiss_overlays,
    navigate_to_puzzle_type, click_date_button,
    click_year_arrow, get_current_picker_year,
    select_month, js_click
)


def dump_months(driver, label):
    """Print all month elements and their state."""
    print(f"\n  --- {label} ---")
    try:
        month_divs = driver.find_elements(By.CSS_SELECTOR, "div.month")
        print(f"  {len(month_divs)} div.month elements:")
        for m in month_divs:
            classes = m.get_attribute("class")
            text = m.text.strip()
            displayed = m.is_displayed()
            print(f"    '{text}' class='{classes}' displayed={displayed}")
    except Exception as e:
        print(f"  Error: {e}")


def try_january(driver, label):
    """Attempt to find and click January, report what happens."""
    print(f"\n  --- Try January: {label} ---")
    try:
        month_div = driver.find_element(By.XPATH,
            "//div[contains(@class, 'month') and text()='January']")
        classes = month_div.get_attribute("class")
        displayed = month_div.is_displayed()
        print(f"  Found: class='{classes}' displayed={displayed}")
        js_click(driver, month_div)
        print(f"  Click succeeded")
        return True
    except Exception as e:
        print(f"  FAILED: {e}")
        # Try partial match
        try:
            month_div = driver.find_element(By.XPATH,
                "//div[contains(@class, 'month') and contains(text(), 'Jan')]")
            print(f"  Partial match found: text='{month_div.text.strip()}' class='{month_div.get_attribute('class')}'")
        except:
            print(f"  Partial match also failed")
        return False


def main():
    print("Launching browser...")
    options = uc.ChromeOptions()
    options.add_argument("--start-maximized")
    driver = uc.Chrome(options=options, version_main=146)

    try:
        creds = get_credentials()
        login(driver, creds)

        print("\n=== Step 1: Navigate to cryptic-crossword ===")
        navigate_to_puzzle_type(driver, "cryptic-crossword")
        time.sleep(3)

        print("\n=== Step 2: Open picker, navigate to 2025 ===")
        click_date_button(driver)
        time.sleep(1)
        picker_year = get_current_picker_year(driver)
        print(f"  Picker year: {picker_year}")
        while picker_year and picker_year > 2025:
            click_year_arrow(driver, 'left')
            time.sleep(0.5)
            picker_year = get_current_picker_year(driver)
        print(f"  Navigated to: {picker_year}")

        print("\n=== Step 3: Click December (simulating end of year) ===")
        if select_month(driver, "December"):
            print("  December selected, waiting for puzzles to load...")
            time.sleep(5)
            print(f"  Page loaded")
        else:
            print("  FAILED to select December")
            return

        print("\n=== Step 4: Re-open picker, navigate to 2024 (next year iteration) ===")
        # This is exactly what the harvest loop does at lines 314-329
        # for the start of a new year
        click_date_button(driver)
        time.sleep(1)
        picker_year = get_current_picker_year(driver)
        print(f"  Picker re-opened at year: {picker_year}")
        while picker_year and picker_year > 2024:
            click_year_arrow(driver, 'left')
            time.sleep(0.5)
            picker_year = get_current_picker_year(driver)
        print(f"  Navigated to: {picker_year}")

        dump_months(driver, "Picker state before January attempt")

        print("\n=== Step 5: Attempt January (no extra delay) ===")
        result = try_january(driver, "no extra delay")

        if not result:
            # Try again with delays
            print("\n=== Step 6: Re-open picker, try with 2s delay ===")
            click_date_button(driver)
            time.sleep(1)
            picker_year = get_current_picker_year(driver)
            while picker_year and picker_year > 2024:
                click_year_arrow(driver, 'left')
                time.sleep(0.5)
                picker_year = get_current_picker_year(driver)
            time.sleep(2)
            dump_months(driver, "After 2s extra delay")
            try_january(driver, "with 2s delay")

    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
    finally:
        try:
            driver.quit()
        except:
            pass
        print("\nDone.")


if __name__ == "__main__":
    main()
