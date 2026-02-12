#!/usr/bin/env python3
"""FT Crossword Scraper - Vision API version
1. Scrapes puzzle listing page for today's puzzle URL
2. Downloads the PDF
3. Uses Claude Vision to extract clues
4. Saves to database
"""

import requests
import json
import sqlite3
import os
import sys
import re
import base64
from datetime import datetime, date
from pathlib import Path
from dotenv import load_dotenv
from bs4 import BeautifulSoup
from anthropic import Anthropic

try:
    from pdf2image import convert_from_path
except ImportError:
    print("Please install: pip install pdf2image")
    print("Also need poppler: https://github.com/oschwarz10612/poppler-windows/releases")
    sys.exit(1)

load_dotenv()

DB_PATH = os.getenv('DB_PATH',
                    r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db")
PDF_DIR = Path("ft_pdfs")
POPPLER_PATH = r"C:\Program Files\Release-25.12.0-0\poppler-25.12.0\Library\bin"

client = Anthropic()


def get_todays_puzzle_url():
    """Scrape FT puzzles page to find today's crossword URL."""
    url = "https://www.ft.com/puzzles-games"
    print(f"Fetching puzzle listing: {url}")

    response = requests.get(url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    for link in soup.find_all('a', href=True):
        text = link.get_text()
        if 'FT Crossword' in text and 'Number' in text:
            href = link['href']
            match = re.search(r'Number\s*([\d,]+)', text)
            puzzle_number = int(match.group(1).replace(',', '')) if match else None
            full_url = f"https://www.ft.com{href}" if href.startswith('/') else href

            print(f"Found: {text.strip()}")
            print(f"URL: {full_url}")

            return full_url, puzzle_number

    print("Could not find FT Crossword link")
    return None, None


def get_pdf_url(puzzle_page_url):
    """Fetch puzzle page and find PDF download link and published date."""
    print(f"Fetching puzzle page: {puzzle_page_url}")

    response = requests.get(puzzle_page_url, timeout=30)
    response.raise_for_status()

    soup = BeautifulSoup(response.text, 'html.parser')

    pdf_url = None
    published_date = None

    # Find PDF link
    for link in soup.find_all('a', href=True):
        href = link['href']
        if '.pdf' in href.lower():
            pdf_url = href if href.startswith('http') else f"https://www.ft.com{href}"
            print(f"Found PDF: {pdf_url}")
            break

    if not pdf_url:
        for tag in soup.find_all(['embed', 'object', 'iframe']):
            src = tag.get('src') or tag.get('data')
            if src and '.pdf' in src.lower():
                pdf_url = src if src.startswith('http') else f"https://www.ft.com{src}"
                print(f"Found embedded PDF: {pdf_url}")
                break

    if not pdf_url:
        print("Could not find PDF link on page")

    # Find published date - look for "datePublished":"2026-01-24T00:00:01.399Z" in JSON-LD
    page_text = response.text  # Use raw HTML, not soup.get_text()
    date_match = re.search(r'"datePublished"\s*:\s*"(\d{4}-\d{2}-\d{2})', page_text)
    if date_match:
        published_date = date_match.group(1)
        print(f"Published: {published_date}")

    return pdf_url, published_date


def download_pdf(pdf_url, puzzle_number, force=False):
    """Download PDF to local folder."""
    PDF_DIR.mkdir(exist_ok=True)

    local_path = PDF_DIR / f"ft_{puzzle_number}.pdf"

    if local_path.exists() and not force:
        print(f"PDF already downloaded: {local_path}")
        return local_path

    print(f"Downloading PDF...")
    response = requests.get(pdf_url, timeout=60)
    response.raise_for_status()

    with open(local_path, 'wb') as f:
        f.write(response.content)

    print(f"Saved: {local_path}")
    return local_path


def extract_clues_with_vision(pdf_path):
    """Use Claude Vision to extract clues from PDF."""
    print(f"Converting PDF to image...")

    # Convert PDF to image
    try:
        print(f"Using Poppler at: {POPPLER_PATH}")
        images = convert_from_path(pdf_path, dpi=150, poppler_path=POPPLER_PATH)
    except Exception as e:
        print(f"PDF conversion failed: {e}")
        print("Make sure Poppler is installed and path is correct")
        return None

    if not images:
        print("Failed to convert PDF to image")
        return None

    # Save first page as PNG
    img_path = pdf_path.with_suffix('.png')
    images[0].save(img_path, 'PNG')
    print(f"Saved image: {img_path}")

    # Read and encode image
    with open(img_path, 'rb') as f:
        image_data = base64.standard_b64encode(f.read()).decode('utf-8')

    print(f"Extracting clues with Vision API...")

    prompt = """Extract all crossword clues from this image.

Return the data as JSON in exactly this format:
{
    "puzzle_number": 18275,
    "setter": "MUDD",
    "across": [
        {"number": 1, "clue": "Put down date and location for game", "enumeration": "6,5"},
        {"number": 7, "clue": "Lower allowance", "enumeration": "3"}
    ],
    "down": [
        {"number": 1, "clue": "Evergreen, happy â€" and tidy", "enumeration": "6,2"},
        {"number": 2, "clue": "Just rubbish bin used, article thrown in", "enumeration": "8"}
    ]
}

Rules:
- Include ALL clues from both ACROSS and DOWN sections
- Copy clue text exactly as shown (including punctuation)
- Enumeration should be just the numbers, e.g. "6,5" not "(6,5)"
- Ignore the solution grid for the previous puzzle
- Return ONLY valid JSON, no other text"""

    response = client.messages.create(
        model="claude-sonnet-4-20250514",
        max_tokens=4000,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {
                            "type": "base64",
                            "media_type": "image/png",
                            "data": image_data
                        }
                    },
                    {
                        "type": "text",
                        "text": prompt
                    }
                ]
            }
        ]
    )

    # Parse response
    response_text = response.content[0].text

    # Extract JSON from response (in case there's extra text)
    json_match = re.search(r'\{[\s\S]*\}', response_text)
    if not json_match:
        print(f"Failed to parse JSON from response: {response_text[:200]}")
        return None

    try:
        puzzle_data = json.loads(json_match.group())
    except json.JSONDecodeError as e:
        print(f"JSON parse error: {e}")
        print(f"Response: {response_text[:500]}")
        return None

    # Add answer field to each clue
    for direction in ['across', 'down']:
        for clue in puzzle_data.get(direction, []):
            clue['answer'] = None

    print(f"Puzzle #: {puzzle_data.get('puzzle_number')}")
    print(f"Setter: {puzzle_data.get('setter')}")
    print(
        f"Extracted: {len(puzzle_data.get('across', []))} across, {len(puzzle_data.get('down', []))} down")

    return puzzle_data


def puzzle_already_fetched(puzzle_number):
    """Check if puzzle is already in database."""
    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ft_clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_type TEXT,
            puzzle_number TEXT,
            puzzle_date TEXT,
            setter TEXT,
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
        SELECT COUNT(*) FROM ft_clues 
        WHERE puzzle_number = ?
    """, (str(puzzle_number),))

    count = cursor.fetchone()[0]
    conn.close()
    return count > 0


def save_to_database(puzzle_data, published_date=None):
    """Save puzzle clues to ft_clues table."""
    print(f"\nSaving to ft_clues table...")

    conn = sqlite3.connect(DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS ft_clues (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            puzzle_type TEXT,
            puzzle_number TEXT,
            puzzle_date TEXT,
            setter TEXT,
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
        CREATE INDEX IF NOT EXISTS idx_ft_puzzle 
        ON ft_clues(puzzle_number)
    """)

    puzzle_number = puzzle_data.get('puzzle_number', 0)
    setter = puzzle_data.get('setter', '')
    fetched_at = datetime.now().isoformat()

    clue_count = 0
    for direction in ['across', 'down']:
        for clue in puzzle_data.get(direction, []):
            cursor.execute("""
                INSERT INTO ft_clues 
                (puzzle_type, puzzle_number, puzzle_date, setter, clue_number, direction, 
                 clue_text, enumeration, answer, fetched_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                'cryptic',
                str(puzzle_number),
                published_date,
                setter,
                str(clue.get('number', '')),
                direction,
                clue.get('clue', ''),
                clue.get('enumeration', ''),
                clue.get('answer', ''),
                fetched_at
            ))
            clue_count += 1

    conn.commit()
    conn.close()

    print(f"Saved {clue_count} clues")
    return clue_count


def main():
    print("=" * 60)
    print("FT CROSSWORD SCRAPER (Vision API)")
    print("=" * 60)
    print(f"Date: {date.today().strftime('%A, %d %B %Y')}")

    force = '--force' in sys.argv

    puzzle_url, puzzle_number = get_todays_puzzle_url()
    if not puzzle_url:
        print("Failed to find today's puzzle")
        return

    if not force and puzzle_already_fetched(puzzle_number):
        print(f"\nPuzzle {puzzle_number} already in database - skipping")
        print("Use --force to re-fetch")
        return

    if force and puzzle_already_fetched(puzzle_number):
        print(f"\nForce mode: removing existing puzzle {puzzle_number} from database")
        conn = sqlite3.connect(DB_PATH)
        cursor = conn.cursor()
        cursor.execute("DELETE FROM ft_clues WHERE puzzle_number = ?",
                       (str(puzzle_number),))
        conn.commit()
        conn.close()

    pdf_url, published_date = get_pdf_url(puzzle_url)
    if not pdf_url:
        print("Failed to find PDF link")
        return

    pdf_path = download_pdf(pdf_url, puzzle_number, force)

    puzzle_data = extract_clues_with_vision(pdf_path)

    if not puzzle_data or (not puzzle_data.get('across') and not puzzle_data.get('down')):
        print("Failed to extract clues from PDF")
        return

    save_to_database(puzzle_data, published_date)

    json_path = f"ft_cryptic_{puzzle_number}.json"
    with open(json_path, 'w') as f:
        json.dump(puzzle_data, f, indent=2)
    print(f"JSON backup: {json_path}")

    print("\n" + "=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == "__main__":
    main()