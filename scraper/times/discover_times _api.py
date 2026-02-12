#!/usr/bin/env python3
"""Find The Times puzzle API - with manual intervention for login/consent"""

import asyncio
import json
from playwright.async_api import async_playwright


async def main():
    puzzle_url = "https://www.thetimes.com/puzzles/crossword"

    print("=" * 60)
    print("THE TIMES API FINDER")
    print("=" * 60)
    print(f"\nTarget: {puzzle_url}")
    print("\nThis will open a browser. You can:")
    print("  - Log in with your trial account")
    print("  - Accept cookies")
    print("  - Navigate to the cryptic crossword")
    print("  - Click 'Reveal' to show answers")
    print("\nThe script will capture all network requests in the background.")
    print("=" * 60)

    async with async_playwright() as p:
        # Launch VISIBLE browser
        browser = await p.chromium.launch(headless=False)
        context = await browser.new_context(viewport={'width': 1400, 'height': 900})
        page = await context.new_page()

        # Collect all JSON responses
        api_responses = []

        async def handle_response(response):
            url = response.url
            try:
                content_type = response.headers.get('content-type', '')

                # Capture JSON responses
                if 'json' in content_type or '.json' in url:
                    try:
                        body = await response.json()
                        api_responses.append({
                            'url': url,
                            'status': response.status,
                            'body': body
                        })
                        # Print interesting ones
                        body_str = str(body)
                        if any(kw in body_str.lower() for kw in
                               ['clue', 'across', 'down', 'answer', 'solution',
                                'crossword']):
                            print(f"\n>>> PUZZLE DATA FOUND: {url[:80]}")
                            print(
                                f"    Keys: {list(body.keys()) if isinstance(body, dict) else 'list'}")
                    except:
                        pass

                # Also look for puzzle-related URLs
                if any(kw in url.lower() for kw in
                       ['puzzle', 'crossword', 'clue', 'grid', 'times']):
                    print(f">>> Interesting URL: {url[:100]}")
            except:
                pass

        page.on('response', handle_response)

        # Navigate to initial page
        print(f"\nOpening: {puzzle_url}")
        try:
            await page.goto(puzzle_url, wait_until='domcontentloaded', timeout=60000)
        except Exception as e:
            print(f"Initial navigation: {e}")
            print("Continuing anyway...")

        # Wait for user
        print("\n" + "=" * 60)
        print("BROWSER IS OPEN - DO YOUR THING")
        print("=" * 60)
        print("\nDo whatever you need to:")
        print("  1. Accept cookies/consent")
        print("  2. Log in with your Times trial account")
        print("  3. Navigate to the Cryptic Crossword")
        print("  4. Click 'Reveal' or 'Show solution'")
        print("\nAll network requests are being captured.")
        print("\n>>> Press ENTER here in the terminal when done <<<")

        await asyncio.get_event_loop().run_in_executor(None, input)

        # Save results
        print(f"\n\nCaptured {len(api_responses)} JSON responses")

        # Filter for puzzle-related
        puzzle_responses = []
        for r in api_responses:
            body_str = str(r['body'])
            if any(kw in body_str.lower() for kw in
                   ['clue', 'across', 'down', 'crossword', 'solution']):
                puzzle_responses.append(r)

        print(f"Puzzle-related responses: {len(puzzle_responses)}")

        # Save all responses
        with open('times_api_responses.json', 'w') as f:
            json.dump(api_responses, f, indent=2, default=str)
        print(f"Saved all responses to: times_api_responses.json")

        # Save puzzle responses separately
        if puzzle_responses:
            with open('times_puzzle_data.json', 'w') as f:
                json.dump(puzzle_responses, f, indent=2, default=str)
            print(f"Saved puzzle data to: times_puzzle_data.json")

            print("\nPuzzle API URLs found:")
            for r in puzzle_responses:
                print(f"  {r['url']}")

        await browser.close()

    print("\n" + "=" * 60)
    print("DONE - Check the JSON files")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())