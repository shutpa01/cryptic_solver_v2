"""SEO routes — sitemaps and robots.txt."""

import os
import tempfile
import threading
import time
from datetime import date, timedelta

from flask import Blueprint, Response, abort, request, current_app

from web.db import get_db
from web.indexnow import KEY as INDEXNOW_KEY
from web.models import clue_slug

bp = Blueprint("seo", __name__)

SITEMAP_PAGE_SIZE = 50000  # Google's limit per sitemap file
CANONICAL_HOST = "https://justcordelia.com"

# Candidate sources for the PUZZLE and NEWS sitemaps. Both sitemaps then filter
# to fully-served puzzles via web.serving.served_puzzle_numbers (puzzle-level
# display rule, user 2026-07-15) — which only ever returns served sources, so
# dailymail/independent puzzles drop out automatically.
SITEMAP_SOURCES = ('telegraph', 'times', 'dailymail', 'guardian', 'independent')


def _clue_url_count():
    """Count of clue URLs in the sitemap = count of clue pages that EXIST.

    Week-only, no legacy (user decision 2026-07-13): a clue page exists iff the
    clue has a WFW pass parse AND a served source — the same rule the clue route
    enforces with 410 (web/serving.py). The sitemap must list exactly those
    pages: listing URLs that answer 410 would be lying to the crawler.
    NOTE: this SQL count drives PAGINATION only; the page route additionally
    drops rows whose card cannot render (a pass with no atoms, or an INVALID
    with no comment), so it may run a few high — harmless against the 50k page
    size. INVALID-with-comment pages are served too (user 2026-07-14), so the
    count includes them; is_served is the final arbiter in the page route."""
    from web.serving import SERVED_SOURCES
    db = get_db()
    placeholders = ",".join("?" for _ in SERVED_SOURCES)
    row = db.execute(
        f"""SELECT COUNT(*) AS n FROM clues c
           JOIN wfw_solve w ON w.clue_id = c.id AND w.status IN ('pass', 'invalid')
           WHERE c.source IN ({placeholders})
             AND c.clue_text IS NOT NULL
             AND c.answer IS NOT NULL AND c.answer != ''""",
        SERVED_SOURCES,
    ).fetchone()
    return row["n"] or 0


def _clue_sitemap_page_count():
    n = _clue_url_count()
    return max(1, (n + SITEMAP_PAGE_SIZE - 1) // SITEMAP_PAGE_SIZE)


@bp.route("/robots.txt")
def robots_txt():
    """Serve robots.txt with sitemap location."""
    body = (
        # AI crawlers are deliberately NOT disallowed. The site's differentiator is
        # step-by-step wordplay reasoning, which AI assistants preferentially cite
        # (Bing AI Performance: 321 Copilot citations 21 Jul–6 Aug 2026, 50% citation
        # share on clue queries). Blocking them closed the one retrieval channel that
        # does not depend on this domain's demoted search ranking.
        "User-agent: *\n"
        "Allow: /\n"
        "Disallow: /admin/\n"
        "Disallow: /reveal\n"
        "Disallow: /explain\n"
        "\n"
        f"Sitemap: {CANONICAL_HOST}/sitemap.xml\n"
    )
    return Response(body, mimetype="text/plain")


@bp.route("/" + INDEXNOW_KEY + ".txt")
def indexnow_key():
    """IndexNow key file (public by design) — lets Bing/Yandex verify we own the domain
    before honouring URL submissions. Served from the origin; Cloudflare only intercepts
    /robots.txt, so this passes through. See web/indexnow.py."""
    return Response(INDEXNOW_KEY, mimetype="text/plain")


@bp.route("/sitemap.xml")
def sitemap_index():
    """Sitemap index — paginated clue sitemaps, puzzle sitemap, news sitemap.

    Clue sitemaps list exactly the clue pages that EXIST under the serving
    rule (WFW pass + served source — week-only/no-legacy, user decision
    2026-07-13); unserved clue URLs answer 410, and a sitemap must never list
    a 410. This is a different regime from the 2026-04-19..25 "7-day cutoff"
    (which listed a subset of pages that all still served 200 — the mismatch
    between sitemap and site is what that episode warns against; today the
    sitemap and the 410 gate share one rule in web/serving.py).
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


# --- clue-sitemap generation + cache -------------------------------------------------
# Building a clue-sitemap PAGE renders a WFW card per clue (is_served -> get_card):
# ~12s for 1599 clues. A cold 12s response can tip past a gateway timeout -> Google's
# sitemap "temporary processing error". The cache is ON DISK (not per-process memory):
# gunicorn runs several workers, so an in-memory cache leaves every worker to pay its own
# cold build and requests hitting a cold worker stay slow (observed on the droplet: a
# second fetch still took ~10s). A shared file means ONE worker builds, ALL workers serve
# it instantly. get_card stays the SOLE arbiter of which URLs are listed (set unchanged);
# we only avoid rebuilding on every fetch. Stale-while-revalidate: serve the last-built
# file now, rebuild off-thread when older than the TTL, so no fetch waits on the render
# (except the very first, before any file exists). A just-served clue is absent from the
# sitemap for at most the TTL — Google polls sitemaps far slower, and it is internally
# linked immediately regardless.
_CLUE_SITEMAP_TTL = 3600  # seconds
_SITEMAP_CACHE_DIR = os.path.join(tempfile.gettempdir(), "cordelia_sitemap")
_clue_sitemap_lock = threading.Lock()
_clue_sitemap_building = set()    # pages THIS worker already has a rebuild in flight for


def _clue_cache_path(page):
    return os.path.join(_SITEMAP_CACHE_DIR, "clues-%d.xml" % page)


def _clue_cache_read(page):
    """(mtime, xml) for the cached page, or (None, None) if not built yet."""
    try:
        path = _clue_cache_path(page)
        mtime = os.path.getmtime(path)
        with open(path, "r", encoding="utf-8") as fh:
            return mtime, fh.read()
    except OSError:
        return None, None


def _clue_cache_write(page, xml):
    """Atomically publish the cached page (temp + rename) so a reader never sees a
    partial file and all workers share one copy."""
    os.makedirs(_SITEMAP_CACHE_DIR, exist_ok=True)
    path = _clue_cache_path(page)
    tmp = "%s.tmp.%d" % (path, os.getpid())
    with open(tmp, "w", encoding="utf-8") as fh:
        fh.write(xml)
    os.replace(tmp, path)


def _build_clue_sitemap_page(page):
    """Render one clue-sitemap page's XML — the slow part (is_served/get_card per row).
    Pure read; identical output to the pre-cache route. Runs inside an app context."""
    db = get_db()
    offset = (page - 1) * SITEMAP_PAGE_SIZE

    # Served pages only — the SAME rule as the clue route's 410 gate (week-only,
    # no legacy; web/serving.is_served). The SQL narrows to served sources and to
    # pass OR reviewer-INVALID rows (an INVALID-with-comment clue is served too —
    # user 2026-07-14); is_served then drops the ones whose card cannot render (a
    # pass with no atoms, an INVALID with no comment — those pages 410, so they
    # must not be listed). Lastmod = the later of publication and the WFW solve.
    from web.serving import SERVED_SOURCES, is_served
    placeholders = ",".join("?" for _ in SERVED_SOURCES)
    rows = db.execute(
        f"""SELECT c.id, c.source, c.clue_text, c.publication_date,
                   w.created_at AS enriched_at
           FROM clues c
           JOIN wfw_solve w ON w.clue_id = c.id AND w.status IN ('pass', 'invalid')
           WHERE c.source IN ({placeholders})
             AND c.clue_text IS NOT NULL
             AND c.answer IS NOT NULL AND c.answer != ''
           ORDER BY c.id
           LIMIT ? OFFSET ?""",
        (*SERVED_SOURCES, SITEMAP_PAGE_SIZE, offset),
    ).fetchall()
    rows = [r for r in rows if is_served(r["source"], r["id"])]

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
    return "\n".join(xml)


def _refresh_clue_sitemap(app, page):
    """Background rebuild of one page (stale-while-revalidate): its own app context so
    get_db works off the request thread; writes the shared file; clears the flag."""
    try:
        with app.app_context():
            xml = _build_clue_sitemap_page(page)
        _clue_cache_write(page, xml)
    finally:
        with _clue_sitemap_lock:
            _clue_sitemap_building.discard(page)


@bp.route("/sitemap-clues-<int:page>.xml")
def sitemap_clues_paged(page):
    """One page of the paginated clue sitemap (disk-cached; see _build_clue_sitemap_page).

    Pages are 1-indexed, up to SITEMAP_PAGE_SIZE URLs (Google's 50k/file limit),
    ordered by clue id ASC so new clues append to the last page.
    """
    if page < 1:
        abort(404)
    n_pages = _clue_sitemap_page_count()
    if page > n_pages:
        abort(404)

    mtime, xml = _clue_cache_read(page)
    if xml is None:
        # No file yet (first fetch since deploy): build once, synchronously, share it.
        xml = _build_clue_sitemap_page(page)
        _clue_cache_write(page, xml)
        return Response(xml, mimetype="application/xml")

    if time.time() - mtime > _CLUE_SITEMAP_TTL:
        # Stale: serve the shared file now, rebuild off-thread (rewrites it for all workers).
        app = current_app._get_current_object()
        with _clue_sitemap_lock:
            if page not in _clue_sitemap_building:
                _clue_sitemap_building.add(page)
                threading.Thread(target=_refresh_clue_sitemap, args=(app, page),
                                 daemon=True).start()
    return Response(xml, mimetype="application/xml")


@bp.route("/sitemap-clues.xml")
def sitemap_clues_legacy():
    """Backward-compat alias for the old single-file sitemap URL.

    Returns page 1 of the paginated sitemap. Kept so any external
    backlinks to /sitemap-clues.xml don't 404.
    """
    return sitemap_clues_paged(1)


@bp.route("/sitemap-puzzles.xml")
def sitemap_puzzles():
    """Puzzle-level sitemap for 'DT 31180' style searches.

    Lists EXACTLY the puzzle pages that exist under the puzzle-level display
    rule (user 2026-07-15): a puzzle page is served only when EVERY one of its
    clues is served; otherwise it 410s. The sitemap must share that one rule
    with the page gate — listing a puzzle URL that 410s would lie to the crawler
    (the same scar tissue as the clue sitemap). served_puzzle_numbers() applies
    the identical test as web.serving.puzzle_is_served in one query.
    """
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

    from web.serving import served_puzzle_numbers
    served = served_puzzle_numbers()

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">')

    for row in rows:
        source = row["source"]
        pnum = row["puzzle_number"]

        if (source, str(pnum)) not in served:
            continue

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

    # Static pages — tools, learn (learn paths generated from route data so
    # every listed URL renders 200)
    from web.routes.learn import served_learn_paths
    static_paths = ["/tools", "/tools/anagram", "/tools/pattern", "/tools/synonym"]
    static_paths += served_learn_paths()
    for static_path in static_paths:
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

    # Same display rule as the puzzle page/sitemap: only fully-served puzzles
    # exist, so news must not advertise a puzzle URL that 410s.
    from web.serving import served_puzzle_numbers
    served = served_puzzle_numbers()

    xml = ['<?xml version="1.0" encoding="UTF-8"?>']
    xml.append('<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9"')
    xml.append('        xmlns:news="http://www.google.com/schemas/sitemap-news/0.9">')

    for row in rows:
        source = row["source"]
        pnum = row["puzzle_number"]

        if (source, str(pnum)) not in served:
            continue

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
