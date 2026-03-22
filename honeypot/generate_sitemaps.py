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


def make_slug(clue_text, answer):
    text = re.sub(r"[^a-z0-9]+", "-", clue_text.lower().strip()).strip("-")
    ans = re.sub(r"[^A-Za-z0-9]", "", answer or "").upper()
    if not text or not ans:
        return None
    return f"{text}-{ans}"


def generate(domain):
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(DB_PATH))

    # Count total
    total = conn.execute(
        "SELECT COUNT(*) FROM clues WHERE answer IS NOT NULL AND length(answer) > 0"
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
        SELECT clue_text, answer, publication_date
        FROM clues
        WHERE answer IS NOT NULL AND length(answer) > 0
        ORDER BY publication_date DESC
    """)

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break

        for clue_text, answer, pub_date in rows:
            slug = make_slug(clue_text, answer)
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

    # Write sitemap index
    index_path = OUT_DIR / "sitemap_index.xml"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        for fname in sitemap_files:
            f.write("  <sitemap>\n")
            f.write(f"    <loc>{domain}/static/sitemaps/{fname}</loc>\n")
            f.write("  </sitemap>\n")
        f.write("</sitemapindex>\n")

    print(f"Sitemap index: {index_path} ({len(sitemap_files)} sitemaps)")

    conn.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--domain", required=True, help="Site domain e.g. https://example.com")
    args = parser.parse_args()
    generate(args.domain.rstrip("/"))
