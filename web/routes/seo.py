"""SEO routes — sitemaps and robots.txt."""

from datetime import date, timedelta

from flask import Blueprint, Response, abort, request, current_app

from web.db import get_db
from web.models import clue_slug

bp = Blueprint("seo", __name__)

SITEMAP_PAGE_SIZE = 50000  # Google's limit per sitemap file
CANONICAL_HOST = "https://justcordelia.com"

# All sources to include in sitemaps
SITEMAP_SOURCES = ('telegraph', 'times', 'dailymail', 'guardian', 'independent')


def _clue_url_count():
    """Count of clue URLs eligible for the sitemap (clue + answer present)."""
    db = get_db()
    placeholders = ",".join("?" for _ in SITEMAP_SOURCES)
    row = db.execute(
        f"""SELECT COUNT(*) AS n FROM clues
           WHERE source IN ({placeholders})
             AND clue_text IS NOT NULL
             AND answer IS NOT NULL AND answer != ''""",
        SITEMAP_SOURCES,
    ).fetchone()
    return row["n"] or 0


def _clue_sitemap_page_count():
    n = _clue_url_count()
    return max(1, (n + SITEMAP_PAGE_SIZE - 1) // SITEMAP_PAGE_SIZE)


@bp.route("/robots.txt")
def robots_txt():
    """Serve robots.txt with sitemap location."""
    body = (
        "User-agent: GPTBot\n"
        "Disallow: /\n"
        "\n"
        "User-agent: ClaudeBot\n"
        "Disallow: /\n"
        "\n"
        "User-agent: Bytespider\n"
        "Disallow: /\n"
        "\n"
        "User-agent: CCBot\n"
        "Disallow: /\n"
        "\n"
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /admin/\n"
        "Disallow: /reveal\n"
        "Disallow: /explain\n"
        "\n"
        f"Sitemap: {CANONICAL_HOST}/sitemap.xml\n"
    )
    return Response(body, mimetype="text/plain")


@bp.route("/sitemap.xml")
def sitemap_index():
    """Sitemap index — paginated clue sitemaps, puzzle sitemap, news sitemap.

    Lists every clue-sitemap page so Google sees the full corpus
    (~580k URLs across ~12 pages of 50k each). The 7-day cutoff that
    was in place 2026-04-19 to 2026-04-25 told Google only ~1k URLs
    were canonical and is believed to have caused mass deindexing of
    discovered URLs.
    """
    today = date.today().isoformat()

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    # Paginated clue sitemaps
    n_pages = _clue_sitemap_page_count()
    for page in range(1, n_pages + 1):
        xml.append("  <sitemap>")
        xml.append(f"    <loc>{CANONICAL_HOST}/sitemap-clues-{page}.xml</loc>")
        xml.append(f"    <lastmod>{today}</lastmod>")
        xml.append("  </sitemap>")

    # Puzzle pages sitemap
    xml.append("  <sitemap>")
    xml.append(f"    <loc>{CANONICAL_HOST}/sitemap-puzzles.xml</loc>")
    xml.append(f"    <lastmod>{today}</lastmod>")
    xml.append("  </sitemap>")
    # News sitemap
    xml.append("  <sitemap>")
    xml.append(f"    <loc>{CANONICAL_HOST}/news-sitemap.xml</loc>")
    xml.append(f"    <lastmod>{today}</lastmod>")
    xml.append("  </sitemap>")
    xml.append("</sitemapindex>")

    return Response("\n".join(xml), mimetype="application/xml")


@bp.route("/sitemap-clues-<int:page>.xml")
def sitemap_clues_paged(page):
    """One page of the paginated clue sitemap.

    Pages are 1-indexed. Each page contains up to SITEMAP_PAGE_SIZE
    URLs (Google's 50k per-file limit). Ordered by clue id ASC for
    stable pagination — new clues append to the last page rather than
    shifting all earlier pages.
    """
    if page < 1:
        abort(404)
    n_pages = _clue_sitemap_page_count()
    if page > n_pages:
        abort(404)

    db = get_db()
    offset = (page - 1) * SITEMAP_PAGE_SIZE

    placeholders = ",".join("?" for _ in SITEMAP_SOURCES)
    rows = db.execute(
        f"""SELECT c.id, c.clue_text, c.publication_date,
                   se.updated_at AS enriched_at
           FROM clues c
           LEFT JOIN structured_explanations se ON se.clue_id = c.id
           WHERE c.source IN ({placeholders})
             AND c.clue_text IS NOT NULL
             AND c.answer IS NOT NULL AND c.answer != ''
           ORDER BY c.id
           LIMIT ? OFFSET ?""",
        (*SITEMAP_SOURCES, SITEMAP_PAGE_SIZE, offset),
    ).fetchall()

    from web.routes.clue import generate_clue_slug

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    for row in rows:
        slug = generate_clue_slug(row["clue_text"], clue_id=row["id"])
        if not slug:
            continue

        lastmod = row["publication_date"] or ""
        enriched = (row["enriched_at"] or "")[:10]
        if enriched > lastmod:
            lastmod = enriched

        xml.append("  <url>")
        xml.append(f"    <loc>{CANONICAL_HOST}/clue/{slug}</loc>")
        if lastmod:
            xml.append(f"    <lastmod>{lastmod}</lastmod>")
        xml.append("    <changefreq>weekly</changefreq>")
        xml.append("  </url>")

    xml.append("</urlset>")
    return Response("\n".join(xml), mimetype="application/xml")


@bp.route("/sitemap-clues.xml")
def sitemap_clues_legacy():
    """Backward-compat alias for the old single-file sitemap URL.

    Returns page 1 of the paginated sitemap. Kept so any external
    backlinks to /sitemap-clues.xml don't 404.
    """
    return sitemap_clues_paged(1)


@bp.route("/sitemap-puzzles.xml")
def sitemap_puzzles():
    """Puzzle-level sitemap for 'DT 31180' style searches."""
    db = get_db()

    placeholders = ",".join("?" for _ in SITEMAP_SOURCES)
    rows = db.execute(
        f"""SELECT source, puzzle_number, MAX(publication_date) as pub_date
           FROM clues
           WHERE source IN ({placeholders})
             AND puzzle_number IS NOT NULL
             AND clue_text IS NOT NULL
           GROUP BY source, puzzle_number
           ORDER BY pub_date DESC""",
        SITEMAP_SOURCES,
    ).fetchall()

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    for row in rows:
        source = row["source"]
        pnum = row["puzzle_number"]

        from web.models import classify_puzzle
        type_slug, _ = classify_puzzle(source, pnum, row["pub_date"])
        if not type_slug:
            continue

        xml.append("  <url>")
        xml.append(f"    <loc>{CANONICAL_HOST}/{source}/{type_slug}/{pnum}</loc>")
        if row["pub_date"]:
            xml.append(f"    <lastmod>{row['pub_date']}</lastmod>")
        xml.append("    <changefreq>monthly</changefreq>")
        xml.append("  </url>")

    # Static pages — tools, learn
    for static_path in ("/tools", "/tools/anagram", "/tools/pattern", "/tools/synonym", "/learn"):
        xml.append("  <url>")
        xml.append(f"    <loc>{CANONICAL_HOST}{static_path}</loc>")
        xml.append("    <changefreq>weekly</changefreq>")
        xml.append("  </url>")

    xml.append("</urlset>")
    return Response("\n".join(xml), mimetype="application/xml")


@bp.route("/news-sitemap.xml")
def news_sitemap():
    """Google News sitemap — puzzles published in the last 48 hours."""
    db = get_db()
    cutoff = (date.today() - timedelta(days=2)).isoformat()

    placeholders = ",".join("?" for _ in SITEMAP_SOURCES)
    rows = db.execute(
        f"""SELECT source, puzzle_number, publication_date
           FROM clues
           WHERE source IN ({placeholders})
             AND puzzle_number IS NOT NULL
             AND clue_text IS NOT NULL
             AND publication_date >= ?
           GROUP BY source, puzzle_number
           ORDER BY publication_date DESC""",
        (*SITEMAP_SOURCES, cutoff),
    ).fetchall()

    SOURCE_NAMES = {
        "telegraph": "The Daily Telegraph",
        "times": "The Times",
        "dailymail": "Daily Mail",
        "guardian": "The Guardian",
        "independent": "The Independent",
    }

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"')
    xml.append('        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">')

    for row in rows:
        source = row["source"]
        pnum = row["puzzle_number"]

        from web.models import classify_puzzle
        type_slug, type_label = classify_puzzle(source, pnum, row["publication_date"])
        if not type_slug:
            continue

        pub_name = SOURCE_NAMES.get(source, source.title())
        title = f"{pub_name} {type_label} #{pnum} — Answers and Explanations"

        xml.append("  <url>")
        xml.append(f"    <loc>{CANONICAL_HOST}/{source}/{type_slug}/{pnum}</loc>")
        xml.append("    <news:news>")
        xml.append(f"      <news:publication>")
        xml.append(f"        <news:name>Cordelia</news:name>")
        xml.append(f"        <news:language>en</news:language>")
        xml.append(f"      </news:publication>")
        xml.append(f"      <news:publication_date>{row['publication_date']}</news:publication_date>")
        xml.append(f"      <news:title>{title}</news:title>")
        xml.append("    </news:news>")
        xml.append("  </url>")

    xml.append("</urlset>")
    return Response("\n".join(xml), mimetype="application/xml")
