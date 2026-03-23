"""Crossword Answer Lookup — minimal site for measuring long-tail SEO traffic.

Serves 500k+ clue pages with answer + definition only.
No explanations, no solver tools, no API calls.
"""

import re
import sqlite3
from pathlib import Path

from flask import Flask, g, abort, render_template, request, Response

app = Flask(__name__)

DB_PATH = Path(__file__).resolve().parent / "data" / "clues.db"


def get_db():
    if "db" not in g:
        g.db = sqlite3.connect(f"file:{DB_PATH}?mode=ro", uri=True)
        g.db.row_factory = sqlite3.Row
    return g.db


@app.teardown_appcontext
def close_db(exc):
    db = g.pop("db", None)
    if db is not None:
        db.close()


# ---------------------------------------------------------------------------
# Slug helpers (same format as main site for consistency)
# ---------------------------------------------------------------------------

def make_slug(clue_text, answer):
    text = re.sub(r"[^a-z0-9]+", "-", clue_text.lower().strip()).strip("-")
    ans = re.sub(r"[^A-Za-z0-9]", "", answer or "").upper()
    if not text or not ans:
        return None
    return f"{text}-{ans}"


def parse_slug(slug):
    """Extract clue words and answer from slug. Answer is the trailing uppercase segment."""
    parts = slug.rsplit("-", 1)
    if len(parts) != 2:
        return None, None
    # Walk backwards to find where the uppercase answer starts
    segments = slug.split("-")
    # Find the split point: last contiguous uppercase segments
    answer_parts = []
    for seg in reversed(segments):
        if seg == seg.upper() and seg.isalpha():
            answer_parts.insert(0, seg)
        else:
            break
    if not answer_parts:
        return None, None
    answer = "".join(answer_parts)
    clue_words = segments[: len(segments) - len(answer_parts)]
    return " ".join(clue_words), answer


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    db = get_db()
    sources = db.execute("""
        SELECT source, COUNT(*) as cnt
        FROM clues
        WHERE answer IS NOT NULL AND length(answer) > 0
        GROUP BY source ORDER BY cnt DESC
    """).fetchall()
    return render_template("home.html", sources=sources)


@app.route("/source/<source>/")
def source_page(source):
    db = get_db()
    page = request.args.get("page", 1, type=int)
    per_page = 100
    offset = (page - 1) * per_page

    total = db.execute(
        "SELECT COUNT(*) FROM clues WHERE source = ? AND answer IS NOT NULL AND length(answer) > 0",
        (source,),
    ).fetchone()[0]

    if total == 0:
        abort(404)

    clues = db.execute("""
        SELECT clue_text, answer, definition, enumeration, publication_date
        FROM clues
        WHERE source = ? AND answer IS NOT NULL AND length(answer) > 0
        ORDER BY publication_date DESC, id DESC
        LIMIT ? OFFSET ?
    """, (source, per_page, offset)).fetchall()

    total_pages = (total + per_page - 1) // per_page

    return render_template(
        "source.html",
        source=source,
        clues=clues,
        page=page,
        total_pages=total_pages,
        total=total,
        make_slug=make_slug,
    )


@app.route("/clue/<slug>")
def clue_page(slug):
    clue_text_hint, answer = parse_slug(slug)
    if not answer:
        abort(404)

    db = get_db()

    # Try exact match first
    row = db.execute("""
        SELECT clue_text, answer, definition, enumeration, source,
               puzzle_number, clue_number, direction, publication_date
        FROM clues
        WHERE answer = ? AND clue_text LIKE ?
        LIMIT 1
    """, (answer, f"%{clue_text_hint.replace(' ', '%')}%")).fetchone()

    if row is None:
        # Fallback: just match on answer + first few words
        words = clue_text_hint.split()[:3]
        pattern = "%".join(words) + "%"
        row = db.execute("""
            SELECT clue_text, answer, definition, enumeration, source,
                   puzzle_number, clue_number, direction, publication_date
            FROM clues
            WHERE answer = ? AND lower(clue_text) LIKE ?
            LIMIT 1
        """, (answer, pattern.lower())).fetchone()

    if row is None:
        abort(404)

    # Find other appearances of the same clue text
    others = db.execute("""
        SELECT source, puzzle_number, publication_date
        FROM clues
        WHERE answer = ? AND clue_text = ? AND id != (
            SELECT id FROM clues WHERE answer = ? AND clue_text = ? LIMIT 1
        )
        ORDER BY publication_date DESC
        LIMIT 5
    """, (answer, row["clue_text"], answer, row["clue_text"])).fetchall()

    return render_template(
        "clue.html",
        clue=row,
        slug=slug,
        others=others,
    )


@app.route("/puzzle/<source>/<puzzle_number>")
def puzzle_page(source, puzzle_number):
    db = get_db()

    clues = db.execute("""
        SELECT clue_text, answer, definition, enumeration, clue_number, direction,
               publication_date
        FROM clues
        WHERE source = ? AND puzzle_number = ?
        ORDER BY CASE direction WHEN 'across' THEN 0 WHEN 'down' THEN 1 ELSE 2 END,
                 CAST(clue_number AS INTEGER)
    """, (source, puzzle_number)).fetchall()

    if not clues:
        abort(404)

    pub_date = clues[0]["publication_date"] if clues else None
    across = [c for c in clues if c["direction"] == "across"]
    down = [c for c in clues if c["direction"] == "down"]

    return render_template(
        "puzzle.html",
        source=source,
        puzzle_number=puzzle_number,
        publication_date=pub_date,
        across=across,
        down=down,
        total=len(clues),
        make_slug=make_slug,
    )


@app.route("/sitemap_index.xml")
def sitemap_index():
    path = Path(__file__).resolve().parent / "static" / "sitemaps" / "sitemap_index.xml"
    if not path.exists():
        abort(404)
    return Response(path.read_text(encoding="utf-8"), mimetype="application/xml")


@app.route("/robots.txt")
def robots():
    content = "User-agent: *\nAllow: /\nSitemap: /sitemap_index.xml\n"
    return Response(content, mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
