"""One-off ingest of Everyman 4150 from the Observer PDF (2026-05-03).

Used because the Observer/SDWU JSON API does not yet have 4150; only the PDF
at https://cdn.slowdownwiseup.co.uk/media/documents/everyman-20260503-4150.pdf.

Saves clue text + enumeration to the clues table with answer NULL. Answers
will be filled later — either via Danword backfill or via next week's
Observer API previous-solution backfill.

Run: python scripts/everyman_4150_pdf_ingest.py [--dry-run]
"""

import argparse
import re
import sqlite3
from pathlib import Path

import pdfplumber

PROJECT_ROOT = Path(__file__).resolve().parent.parent
PDF_PATH = PROJECT_ROOT / "data" / "everyman_4150.pdf"
DB_PATH = PROJECT_ROOT / "data" / "clues_master.db"

PUZZLE_NUMBER = 4150
PUBLICATION_DATE = "2026-05-03"
SOURCE = "guardian"
SOURCE_URL = (
    "https://cdn.slowdownwiseup.co.uk/media/documents/"
    "everyman-20260503-4150.pdf"
)


def parse_section(text: str, direction: str) -> list[dict]:
    """Parse one column-text into a list of clues for the given direction.

    A clue starts with a number at line start, may wrap across lines, and
    ends with `(enum)` like `(7)`, `(4,6)`, or `(6-6)`.
    """
    clues: list[dict] = []
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    cur_num: int | None = None
    cur_buf: list[str] = []
    for line in lines:
        if line in ("Across", "Down"):
            continue
        m = re.match(r"^(\d+)\s+(.+)$", line)
        if m:
            if cur_num is not None:
                clues.append(_finish_clue(cur_num, " ".join(cur_buf), direction))
            cur_num = int(m.group(1))
            cur_buf = [m.group(2)]
        else:
            if cur_num is not None:
                cur_buf.append(line)
    if cur_num is not None:
        clues.append(_finish_clue(cur_num, " ".join(cur_buf), direction))
    return clues


def _finish_clue(number: int, full: str, direction: str) -> dict:
    full = full.strip()
    em = re.search(r"\(([^()]+)\)\s*$", full)
    enum = em.group(1) if em else ""
    text = re.sub(r"\(([^()]+)\)\s*$", "", full).strip()
    # Normalise curly apostrophes to ASCII for DB consistency with other clues
    text = text.replace("’", "'").replace("‘", "'").replace("—", "-").replace("–", "-")
    return {
        "number": number,
        "direction": direction,
        "text": text,
        "enum": enum,
    }


def extract_clues_from_pdf(path: Path) -> list[dict]:
    with pdfplumber.open(str(path)) as pdf:
        page = pdf.pages[0]
        W, H = page.width, page.height
        # Crop below the grid (~y=440) into three columns.
        col1 = page.crop((0, 440, W / 3, H)).extract_text() or ""
        col2 = page.crop((W / 3, 440, 2 * W / 3, H)).extract_text() or ""
        col3 = page.crop((2 * W / 3, 440, W, H)).extract_text() or ""

    # Column 1: across only.
    across = parse_section(col1, "across")
    # Column 2: across continued, then down (split on "Down" header).
    if "Down" in col2:
        ac_part, dn_part = col2.split("Down", 1)
        across.extend(parse_section(ac_part, "across"))
        down = parse_section(dn_part, "down")
    else:
        down = []
        across.extend(parse_section(col2, "across"))
    # Column 3: down continued.
    down.extend(parse_section(col3, "down"))

    return across + down


def write_clues(clues: list[dict], dry_run: bool = False) -> int:
    """Insert each clue into clues table with empty-string answer.

    Schema requires `answer NOT NULL`, so we follow the existing convention
    of using '' for puzzles not yet solved. Backfill (Danword or next week's
    Observer previous-solution) will populate them later.
    """
    if dry_run:
        return 0
    conn = sqlite3.connect(str(DB_PATH))
    n = 0
    try:
        for c in clues:
            cur = conn.execute(
                """
                INSERT OR IGNORE INTO clues
                (source, puzzle_number, publication_date, clue_number, direction,
                 clue_text, enumeration, answer, original_db)
                VALUES (?, ?, ?, ?, ?, ?, ?, '', ?)
                """,
                (
                    SOURCE,
                    str(PUZZLE_NUMBER),
                    PUBLICATION_DATE,
                    str(c["number"]),
                    c["direction"],
                    c["text"],
                    c["enum"],
                    f"observer-everyman-{PUZZLE_NUMBER}",
                ),
            )
            n += cur.rowcount
        conn.commit()
    finally:
        conn.close()
    return n


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--dry-run", action="store_true", help="Parse but don't write")
    args = ap.parse_args()

    if not PDF_PATH.exists():
        raise SystemExit(f"PDF not found at {PDF_PATH}")

    clues = extract_clues_from_pdf(PDF_PATH)
    print(f"Extracted {len(clues)} clues "
          f"({sum(1 for c in clues if c['direction']=='across')} across, "
          f"{sum(1 for c in clues if c['direction']=='down')} down)")

    print()
    for c in clues:
        d = c["direction"][0]
        print(f"  {c['number']:>2}{d}  ({c['enum']:>5})  {c['text']}")

    if args.dry_run:
        print("\n--dry-run set, not writing.")
        return

    print(f"\nWriting to {DB_PATH}...")
    write_clues(clues)
    # Verify
    conn = sqlite3.connect(str(DB_PATH))
    n = conn.execute(
        "SELECT COUNT(*) FROM clues WHERE source=? AND puzzle_number=?",
        (SOURCE, str(PUZZLE_NUMBER)),
    ).fetchone()[0]
    conn.close()
    print(f"Clues now in DB for {SOURCE} #{PUZZLE_NUMBER}: {n}")


if __name__ == "__main__":
    main()
