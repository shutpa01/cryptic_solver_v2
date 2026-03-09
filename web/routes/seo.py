"""SEO routes — sitemaps and robots.txt."""

from flask import Blueprint, Response, request, current_app

from web.db import get_db
from web.models import clue_slug

bp = Blueprint("seo", __name__)

SITEMAP_PAGE_SIZE = 50000  # Google's limit per sitemap file


@bp.route("/robots.txt")
def robots_txt():
    """Serve robots.txt with sitemap location."""
    host = request.host_url.rstrip("/")
    body = (
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /admin/\n"
        "Disallow: /reveal\n"
        "Disallow: /explain\n"
        "\n"
        f"Sitemap: {host}/sitemap.xml\n"
    )
    return Response(body, mimetype="text/plain")


@bp.route("/sitemap.xml")
def sitemap_index():
    """Sitemap index listing all sub-sitemaps."""
    db = get_db()
    host = request.host_url.rstrip("/")

    # Count total indexable clues (Telegraph + Times with answers)
    total = db.execute(
        """SELECT COUNT(*) FROM clues
           WHERE source IN ('telegraph', 'times')
             AND clue_text IS NOT NULL
             AND answer IS NOT NULL AND answer != ''"""
    ).fetchone()[0]

    num_pages = max(1, (total + SITEMAP_PAGE_SIZE - 1) // SITEMAP_PAGE_SIZE)

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<sitemapindex xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')
    for page in range(1, num_pages + 1):
        xml.append("  <sitemap>")
        xml.append(f"    <loc>{host}/sitemap-clues-{page}.xml</loc>")
        xml.append("  </sitemap>")
    # Puzzle pages sitemap
    xml.append("  <sitemap>")
    xml.append(f"    <loc>{host}/sitemap-puzzles.xml</loc>")
    xml.append("  </sitemap>")
    xml.append("</sitemapindex>")

    return Response("\n".join(xml), mimetype="application/xml")


@bp.route("/sitemap-clues-<int:page>.xml")
def sitemap_clues(page):
    """Individual clue sitemap — up to 50k URLs per page."""
    db = get_db()
    host = request.host_url.rstrip("/")
    offset = (page - 1) * SITEMAP_PAGE_SIZE

    rows = db.execute(
        """SELECT clue_text, enumeration, publication_date
           FROM clues
           WHERE source IN ('telegraph', 'times')
             AND clue_text IS NOT NULL
             AND answer IS NOT NULL AND answer != ''
           ORDER BY id
           LIMIT ? OFFSET ?""",
        (SITEMAP_PAGE_SIZE, offset),
    ).fetchall()

    if not rows:
        return Response("Not found", status=404)

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    seen_slugs = set()
    for row in rows:
        slug = clue_slug(row["clue_text"], row["enumeration"])
        if not slug or slug in seen_slugs:
            continue
        seen_slugs.add(slug)

        xml.append("  <url>")
        xml.append(f"    <loc>{host}/clue/{slug}</loc>")
        if row["publication_date"]:
            xml.append(f"    <lastmod>{row['publication_date']}</lastmod>")
        xml.append("    <changefreq>monthly</changefreq>")
        xml.append("  </url>")

    xml.append("</urlset>")
    return Response("\n".join(xml), mimetype="application/xml")


@bp.route("/sitemap-puzzles.xml")
def sitemap_puzzles():
    """Puzzle-level sitemap for 'DT 31180' style searches."""
    db = get_db()
    host = request.host_url.rstrip("/")

    rows = db.execute(
        """SELECT source, puzzle_number, MAX(publication_date) as pub_date
           FROM clues
           WHERE source IN ('telegraph', 'times')
             AND puzzle_number IS NOT NULL
             AND clue_text IS NOT NULL
           GROUP BY source, puzzle_number
           ORDER BY pub_date DESC"""
    ).fetchall()

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    for row in rows:
        source = row["source"]
        pnum = row["puzzle_number"]

        # Determine type slug for URL
        from web.models import classify_puzzle
        type_slug, _ = classify_puzzle(source, pnum, row["pub_date"])
        if not type_slug:
            continue

        xml.append("  <url>")
        xml.append(f"    <loc>{host}/{source}/{type_slug}/{pnum}</loc>")
        if row["pub_date"]:
            xml.append(f"    <lastmod>{row['pub_date']}</lastmod>")
        xml.append("    <changefreq>monthly</changefreq>")
        xml.append("  </url>")

    xml.append("</urlset>")
    return Response("\n".join(xml), mimetype="application/xml")
