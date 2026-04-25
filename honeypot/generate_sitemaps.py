"""Generate sitemap XML files for all clues with answers.

Produces sitemap files with max 50,000 URLs each, plus a sitemap index.
Run: python generate_sitemaps.py --domain https://yourdomain.com

Memory-efficient: streams rows from DB and writes sitemaps incrementally.
"""

import argparse
import re
import sqlite3
from pathlib import Path

MAX_PER_SITEMAP = 50_000
BATCH_SIZE = 10_000

DB_PATH = Path(__file__).resolve().parent / "data" / "clues.db"
OUT_DIR = Path(__file__).resolve().parent / "static" / "sitemaps"


def make_slug(clue_text, answer, clue_id=None):
    """Generate URL slug from clue ID and clue text. Answer is NOT included."""
    if not clue_id:
        return None
    text = re.sub(r"[^a-z0-9]+", "-", clue_text.lower().strip()).strip("-")
    if not text:
        return None
    words = text.split("-")[:12]
    text = "-".join(words)
    return f"{clue_id}-{text}"


def generate(domain):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))

    # Count total (exclude Guardian non-cryptic: puzzle_number < 20000)
    total = conn.execute(
        """SELECT COUNT(*) FROM clues WHERE answer IS NOT NULL AND length(answer) > 0
           AND source IN ('telegraph', 'times', 'guardian', 'independent', 'dailymail')
           AND NOT (source = 'guardian' AND CAST(puzzle_number AS INTEGER) < 20000)"""
    ).fetchone()[0]
    print(f"Total clues with answers: {total}")

    # Stream rows in batches, deduplicate, write sitemaps incrementally
    seen = set()
    sitemap_files = []
    file_num = 1
    current_file = None
    urls_in_current = 0
    unique_count = 0

    cursor = conn.execute("""
        SELECT id, clue_text, answer, publication_date
        FROM clues
        WHERE answer IS NOT NULL AND length(answer) > 0
          AND source IN ('telegraph', 'times', 'guardian', 'independent', 'dailymail')
          AND NOT (source = 'guardian' AND CAST(puzzle_number AS INTEGER) < 20000)
        ORDER BY publication_date DESC
    """)

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break

        for clue_id, clue_text, answer, pub_date in rows:
            slug = make_slug(clue_text, answer, clue_id=clue_id)
            if not slug or slug in seen:
                continue
            seen.add(slug)
            unique_count += 1

            # Start a new sitemap file if needed
            if current_file is None or urls_in_current >= MAX_PER_SITEMAP:
                if current_file is not None:
                    current_file.write("</urlset>\n")
                    current_file.close()
                    print(f"  Wrote sitemap_{file_num - 1}.xml ({urls_in_current} URLs)")

                filename = f"sitemap_{file_num}.xml"
                sitemap_files.append(filename)
                filepath = OUT_DIR / filename
                current_file = open(filepath, "w", encoding="utf-8")
                current_file.write('<?xml version="1.0" encoding="UTF-8"?>\n')
                current_file.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
                file_num += 1
                urls_in_current = 0

            current_file.write("  <url>\n")
            current_file.write(f"    <loc>{domain}/clue/{slug}</loc>\n")
            if pub_date:
                current_file.write(f"    <lastmod>{pub_date}</lastmod>\n")
            current_file.write("    <changefreq>yearly</changefreq>\n")
            current_file.write("  </url>\n")
            urls_in_current += 1

        # Free memory: clear the slug text from seen, keep only hashes
        print(f"  Processed batch... {unique_count} unique so far")

    # Close last file
    if current_file is not None:
        current_file.write("</urlset>\n")
        current_file.close()
        print(f"  Wrote sitemap_{file_num - 1}.xml ({urls_in_current} URLs)")

    print(f"\nUnique slugs: {unique_count}")

    from datetime import date as _date
    today = _date.today().isoformat()

    # --- Puzzle sitemap ---
    puzzle_cursor = conn.execute("""
        SELECT source, puzzle_number, MAX(publication_date) as pub_date
        FROM clues
        WHERE source IN ('telegraph', 'times', 'guardian', 'independent', 'dailymail')
          AND puzzle_number IS NOT NULL
          AND answer IS NOT NULL AND length(answer) > 0
          AND NOT (source = 'guardian' AND CAST(puzzle_number AS INTEGER) < 20000)
        GROUP BY source, puzzle_number
        ORDER BY pub_date DESC
    """)

    puzzle_file = OUT_DIR / "sitemap_puzzles.xml"
    with open(puzzle_file, "w", encoding="utf-8") as pf:
        pf.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        pf.write('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        puzzle_count = 0
        for row in puzzle_cursor:
            source, pnum, pub_date = row
            pf.write("  <url>\n")
            pf.write(f"    <loc>{domain}/puzzle/{source}/{pnum}</loc>\n")
            if pub_date:
                pf.write(f"    <lastmod>{pub_date}</lastmod>\n")
            pf.write("    <changefreq>monthly</changefreq>\n")
            pf.write("  </url>\n")
            puzzle_count += 1

        # Future puzzle "coming soon" pages — pre-index before publication
        future_ranges = [
            ("telegraph", 31000, 31999),
            ("dailymail", 16000, 19999),
            ("times", 26000, 39999),
            ("guardian", 20000, 39999),
            ("independent", 1, 19999),
        ]
        for f_source, lo, hi in future_ranges:
            row = conn.execute(
                "SELECT MAX(CAST(puzzle_number AS INTEGER)) FROM clues "
                "WHERE source = ? AND CAST(puzzle_number AS INTEGER) BETWEEN ? AND ?",
                (f_source, lo, hi),
            ).fetchone()
            if not row or row[0] is None:
                continue
            latest = row[0]
            for i in range(1, 6):
                pf.write("  <url>\n")
                pf.write(f"    <loc>{domain}/puzzle/{f_source}/{latest + i}</loc>\n")
                pf.write(f"    <lastmod>{today}</lastmod>\n")
                pf.write("    <changefreq>daily</changefreq>\n")
                pf.write("  </url>\n")
                puzzle_count += 1

        pf.write("</urlset>\n")

    sitemap_files.append("sitemap_puzzles.xml")
    print(f"  Wrote sitemap_puzzles.xml ({puzzle_count} puzzles)")

    # Write sitemap index (with lastmod so Google knows when to re-read)
    index_path = OUT_DIR / "sitemap_index.xml"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        for fname in sitemap_files:
            f.write("  <sitemap>\n")
            f.write(f"    <loc>{domain}/static/sitemaps/{fname}</loc>\n")
            f.write(f"    <lastmod>{today}</lastmod>\n")
            f.write("  </sitemap>\n")
        f.write("</sitemapindex>\n")

    print(f"Sitemap index: {index_path} ({len(sitemap_files)} sitemaps)")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Site domain e.g. https://example.com")
    args = parser.parse_args()
    generate(args.domain.rstrip("/"))
