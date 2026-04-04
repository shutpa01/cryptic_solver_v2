"""Generate slugs for all clues and regenerate sitemaps.

Adds a 'slug' column to the clues table if it doesn't exist, then populates
it for any clue that has an answer but no slug. After slugs are generated,
regenerates the sitemap XML files.

Called by puzzle_scraper.py before the DB is uploaded to the droplet.
Can also be run standalone for testing.

Usage:
    python honeypot/generate_slugs.py                    # default domain
    python honeypot/generate_slugs.py --domain https://clairesclues.xyz
    python honeypot/generate_slugs.py --dry-run           # show counts, no writes
"""

import argparse
import re
import sqlite3
from pathlib import Path

DEFAULT_DOMAIN = "https://clairesclues.xyz"
MAX_PER_SITEMAP = 50_000
BATCH_SIZE = 10_000

# When called from puzzle_scraper, DB_PATH is clues_master.db (the source)
# When running on the droplet, it's honeypot/data/clues.db
# We accept it as a parameter, with a sensible default
PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_DB = PROJECT_ROOT / "data" / "clues_master.db"
SITEMAP_DIR = Path(__file__).resolve().parent / "static" / "sitemaps"


def make_slug(clue_text, answer):
    """Generate URL slug from clue text and answer."""
    text = re.sub(r"[^a-z0-9]+", "-", clue_text.lower().strip()).strip("-")
    ans = re.sub(r"[^A-Za-z0-9]", "", answer or "").upper()
    if not text or not ans:
        return None
    return f"{text}-{ans}"


def ensure_slug_column(conn):
    """Add slug column to clues table if it doesn't exist."""
    cols = [row[1] for row in conn.execute("PRAGMA table_info(clues)").fetchall()]
    if "slug" not in cols:
        conn.execute("ALTER TABLE clues ADD COLUMN slug TEXT")
        conn.commit()
        print("Added 'slug' column to clues table")
        return True
    return False


def generate_slugs(db_path, dry_run=False):
    """Populate slug column for all clues that have an answer but no slug."""
    conn = sqlite3.connect(str(db_path), timeout=30)

    # Check if slug column exists
    cols = [row[1] for row in conn.execute("PRAGMA table_info(clues)").fetchall()]
    has_slug_col = "slug" in cols

    if not has_slug_col and dry_run:
        total_with_answer = conn.execute(
            "SELECT COUNT(*) FROM clues WHERE answer IS NOT NULL AND answer != ''"
        ).fetchone()[0]
        print(f"Clues with answers: {total_with_answer:,}")
        print(f"Missing slugs: {total_with_answer:,} (slug column does not exist yet)")
        print("--dry-run: no writes")
        conn.close()
        return total_with_answer

    if not has_slug_col:
        ensure_slug_column(conn)

    # Count clues needing slugs
    total_with_answer = conn.execute(
        "SELECT COUNT(*) FROM clues WHERE answer IS NOT NULL AND answer != ''"
    ).fetchone()[0]

    missing_slug = conn.execute(
        "SELECT COUNT(*) FROM clues WHERE answer IS NOT NULL AND answer != '' AND (slug IS NULL OR slug = '')"
    ).fetchone()[0]

    print(f"Clues with answers: {total_with_answer:,}")
    print(f"Missing slugs: {missing_slug:,}")

    if dry_run:
        print("--dry-run: no writes")
        conn.close()
        return missing_slug

    if missing_slug == 0:
        print("All slugs up to date")
        conn.close()
        return 0

    # Generate slugs in batches
    cursor = conn.execute("""
        SELECT id, clue_text, answer FROM clues
        WHERE answer IS NOT NULL AND answer != ''
        AND (slug IS NULL OR slug = '')
    """)

    updated = 0
    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break

        updates = []
        for clue_id, clue_text, answer in rows:
            slug = make_slug(clue_text, answer)
            if slug:
                updates.append((slug, clue_id))

        if updates:
            conn.executemany("UPDATE clues SET slug = ? WHERE id = ?", updates)
            conn.commit()
            updated += len(updates)
            print(f"  Updated {updated:,} slugs...")

    conn.close()
    print(f"Generated {updated:,} slugs")
    return updated


def generate_sitemaps(db_path, domain):
    """Regenerate sitemap XML files from slugs in the DB."""
    SITEMAP_DIR.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(str(db_path), timeout=30)

    total = conn.execute(
        "SELECT COUNT(*) FROM clues WHERE slug IS NOT NULL AND slug != ''"
        " AND NOT (source = 'guardian' AND CAST(puzzle_number AS INTEGER) < 20000)"
    ).fetchone()[0]
    print(f"\nGenerating sitemaps for {total:,} slugs...")

    seen = set()
    sitemap_files = []
    file_num = 1
    current_file = None
    urls_in_current = 0
    unique_count = 0

    cursor = conn.execute("""
        SELECT slug, publication_date FROM clues
        WHERE slug IS NOT NULL AND slug != ''
          AND NOT (source = 'guardian' AND CAST(puzzle_number AS INTEGER) < 20000)
        ORDER BY publication_date DESC
    """)

    while True:
        rows = cursor.fetchmany(BATCH_SIZE)
        if not rows:
            break

        for slug, pub_date in rows:
            if slug in seen:
                continue
            seen.add(slug)
            unique_count += 1

            if current_file is None or urls_in_current >= MAX_PER_SITEMAP:
                if current_file is not None:
                    current_file.write("</urlset>\n")
                    current_file.close()
                    print(f"  Wrote sitemap_{file_num - 1}.xml ({urls_in_current:,} URLs)")

                filename = f"sitemap_{file_num}.xml"
                sitemap_files.append(filename)
                filepath = SITEMAP_DIR / filename
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

    if current_file is not None:
        current_file.write("</urlset>\n")
        current_file.close()
        print(f"  Wrote sitemap_{file_num - 1}.xml ({urls_in_current:,} URLs)")

    # Write sitemap index
    index_path = SITEMAP_DIR / "sitemap_index.xml"
    with open(index_path, "w", encoding="utf-8") as f:
        f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
        f.write('<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n')
        for fname in sitemap_files:
            f.write("  <sitemap>\n")
            f.write(f"    <loc>{domain}/static/sitemaps/{fname}</loc>\n")
            f.write("  </sitemap>\n")
        f.write("</sitemapindex>\n")

    print(f"Unique URLs: {unique_count:,}")
    print(f"Sitemap index: {index_path} ({len(sitemap_files)} sitemaps)")

    conn.close()


def run(db_path=None, domain=None, dry_run=False):
    """Main entry point — called by puzzle_scraper or standalone."""
    db_path = db_path or DEFAULT_DB
    domain = domain or DEFAULT_DOMAIN

    print("=" * 60)
    print("GENERATE SLUGS + SITEMAPS")
    print(f"  DB: {db_path}")
    print(f"  Domain: {domain}")
    print("=" * 60)

    generate_slugs(db_path, dry_run=dry_run)

    if not dry_run:
        generate_sitemaps(db_path, domain)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Generate slugs and sitemaps")
    parser.add_argument("--domain", default=DEFAULT_DOMAIN, help="Site domain")
    parser.add_argument("--db", default=None, help="Path to clues DB")
    parser.add_argument("--dry-run", action="store_true", help="Show counts only")
    args = parser.parse_args()

    db = Path(args.db) if args.db else DEFAULT_DB
    run(db_path=db, domain=args.domain, dry_run=args.dry_run)
