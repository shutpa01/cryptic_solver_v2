"""Ingest a Telegraph PRIZE TOUGHIE (Sunday, weekly) from its scraped JSON into the
main `clues` table.

Built 2026-07-15: the scraper (scraper/telegraph/telegraph_all.py) downloads the Prize
Toughie each Sunday to scraper/telegraph/telegraph_prize-toughie_*.json, but its
promote-to-`clues` step is broken for prize puzzles (it requires an `explanation` that is
never scraped, and drops publication_date). This ingests directly.

Prize puzzles are EMBARGOED: the JSON carries clues + enumerations but NO answers (only a
hashed solution). Rows are filed with answer NULL, exactly like the other prize puzzles —
the user enters the grid via the admin prize-puzzle flow, then cascades/solves.

Filed as source='telegraph', puzzle_number = the "No N" from the title (Prize Toughie No
233 -> 233). classify_puzzle maps telegraph 1-2999 -> Prize Toughie (no collision: prize
cryptics are 3xxx, cryptics 31xxx). Idempotent: skips a puzzle already present.

Usage:
    python -m scripts.ingest_prize_toughie                 # newest JSON, DRY RUN
    python -m scripts.ingest_prize_toughie --commit        # newest JSON, write
    python -m scripts.ingest_prize_toughie <path> --commit
"""
import glob
import json
import os
import re
import sqlite3
import sys
from datetime import datetime
from html import unescape

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "clues_master.db")
GLOB = os.path.join(ROOT, "scraper", "telegraph", "telegraph_prize-toughie_*.json")


_TAG = re.compile(r"</?[a-zA-Z][^>]*>")


def _clean_clue(s):
    """Decode HTML entities AND strip inline formatting tags (<i>, <b>, ...) that the
    Telegraph JSON carries. Tags render zero-width in a browser, so they strip to
    nothing (Godfather'</i>s -> Godfather's); whitespace is then normalised."""
    s = unescape(s or "")
    s = _TAG.sub("", s)
    return re.sub(r"\s+", " ", s).strip()


def _iso(s):
    try:
        return datetime.strptime(s, "%A, %d %B %Y").strftime("%Y-%m-%d")
    except (ValueError, TypeError):
        return s


def parse(path):
    copy = json.load(open(path, encoding="utf-8"))["json"]["copy"]
    title = copy.get("title", "")
    m = re.search(r"No\s*(\d+)", title)
    pnum = int(m.group(1)) if m else copy.get("id")
    date = _iso(copy.get("date-publish", ""))
    clues = []
    for group in copy.get("clues", []):
        direction = (group.get("title", "") or "").lower()   # 'across' / 'down'
        for c in group.get("clues", []):
            clues.append({
                "clue_number": str(c.get("number", "")),
                "direction": direction,
                "clue_text": _clean_clue(c.get("clue", "")),
                "enumeration": c.get("format", ""),
                "answer": c.get("answer") or "",              # embargoed -> '' (NOT NULL col)
            })
    return title, str(pnum), date, clues


def main():
    args = [a for a in sys.argv[1:] if a != "--commit"]
    commit = "--commit" in sys.argv
    path = args[0] if args else max(glob.glob(GLOB), key=os.path.getmtime)

    title, pnum, date, clues = parse(path)
    print("File:   %s" % os.path.basename(path))
    print("Puzzle: %s  (source=telegraph, puzzle_number=%s, date=%s)" % (title, pnum, date))
    print("Clues:  %d  (answers embargoed -> NULL)" % len(clues))

    conn = sqlite3.connect(DB)
    existing = conn.execute(
        "SELECT COUNT(*) FROM clues WHERE source='telegraph' AND puzzle_number=?",
        (pnum,)).fetchone()[0]
    if existing:
        print("ALREADY PRESENT: %d rows for telegraph #%s — nothing to do." % (existing, pnum))
        conn.close()
        return
    for c in clues[:3]:
        print("   e.g. %s %s | %s (%s)" % (c["direction"], c["clue_number"],
                                           c["clue_text"][:45], c["enumeration"]))
    if not commit:
        print("\nDRY RUN — re-run with --commit to write these %d rows." % len(clues))
        conn.close()
        return
    conn.executemany(
        "INSERT INTO clues (source, puzzle_number, publication_date, clue_number, "
        "direction, clue_text, enumeration, answer) "
        "VALUES ('telegraph', ?, ?, ?, ?, ?, ?, ?)",
        [(pnum, date, c["clue_number"], c["direction"], c["clue_text"],
          c["enumeration"], c["answer"]) for c in clues])
    conn.commit()
    n = conn.execute("SELECT COUNT(*) FROM clues WHERE source='telegraph' AND puzzle_number=?",
                     (pnum,)).fetchone()[0]
    conn.close()
    print("\nWROTE %d rows. telegraph #%s now has %d clues." % (len(clues), pnum, n))


if __name__ == "__main__":
    main()
