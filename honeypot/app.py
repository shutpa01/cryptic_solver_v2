"""Crossword Answer Lookup — SEO site for measuring long-tail search traffic.

Serves 500k+ clue pages with answer, definition, wordplay type, and
explanation — all visible on page load for search engine indexing.
No interactive features, no API calls, no HTMX.
"""

import json
import re
import sqlite3
from pathlib import Path

from flask import Flask, g, abort, render_template, request, Response, jsonify

app = Flask(__name__)

DB_PATH = Path(__file__).resolve().parent.parent / "data" / "clues_master.db"

ALLOWED_SOURCES = ('telegraph', 'times', 'guardian', 'independent', 'dailymail')
ALLOWED_SOURCES_SQL = "('telegraph', 'times', 'guardian', 'independent', 'dailymail')"

# --- Blog attribution for human explanations ---
# TFTT post index: puzzle_number -> {link, title, ...}
TFTT_INDEX_PATH = Path(__file__).resolve().parent.parent / "scraper" / "timesforthetimes" / "tftt_post_index.json"
_tftt_index = None


def _get_tftt_index():
    global _tftt_index
    if _tftt_index is None:
        try:
            with open(TFTT_INDEX_PATH, encoding="utf-8") as f:
                _tftt_index = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            _tftt_index = {}
    return _tftt_index


def get_blog_attribution(source, puzzle_number):
    """Return (blog_name, blog_url) for a human explanation, or (None, None)."""
    if source == "times":
        idx = _get_tftt_index()
        entry = idx.get(str(puzzle_number))
        if entry and entry.get("link"):
            return "Times for the Times", entry["link"]
        return "Times for the Times", None
    elif source == "guardian":
        return "Fifteensquared", f"https://fifteensquared.net/?s=guardian+{puzzle_number}"
    elif source == "independent":
        return "Fifteensquared", f"https://fifteensquared.net/?s=independent+{puzzle_number}"
    return None, None


def get_source_puzzle_url(source, puzzle_number):
    """Return the URL to the original puzzle on the newspaper's website, or None."""
    if source == "guardian":
        return f"https://www.theguardian.com/crosswords/cryptic/{puzzle_number}"
    elif source == "telegraph":
        return "https://www.telegraph.co.uk/puzzles/puzzle/crosswords/"
    elif source == "times":
        return "https://www.thetimes.com/puzzles/crossword"
    elif source == "independent":
        return "https://puzzles.independent.co.uk/games/cryptic-crossword-independent"
    elif source == "dailymail":
        return "https://www.dailymail.co.uk/puzzles/index.html"
    return None

WORDPLAY_LABELS = {
    "anagram": "Anagram",
    "charade": "Charade",
    "container": "Container",
    "hidden": "Hidden word",
    "reversal": "Reversal",
    "double_definition": "Double definition",
    "cryptic_definition": "Cryptic definition",
    "homophone": "Homophone",
    "deletion": "Deletion",
    "substitution": "Substitution",
    "spoonerism": "Spoonerism",
    "initial_letters": "Initial letters",
    "alternation": "Alternation",
}

SOURCE_NAMES = {"dailymail": "Daily Mail", "telegraph-toughie": "Telegraph Toughie"}


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


@app.template_filter("wordplay_label")
def wordplay_label_filter(value):
    if not value:
        return ""
    return WORDPLAY_LABELS.get(value, value.replace("_", " ").title())


@app.template_filter("source_name")
def source_name_filter(value):
    return SOURCE_NAMES.get(value, (value or "").title())


# ---------------------------------------------------------------------------
# Slug helpers
# ---------------------------------------------------------------------------

def make_slug(clue_text, answer):
    text = re.sub(r"[^a-z0-9]+", "-", clue_text.lower().strip()).strip("-")
    ans = re.sub(r"[^A-Za-z0-9]", "", answer or "").upper()
    if not text or not ans:
        return None
    return f"{text}-{ans}"


def parse_slug(slug):
    """Extract clue words and answer from slug."""
    segments = slug.split("-")
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
# Explanation builder (ported from web/models.py)
# ---------------------------------------------------------------------------

def _correct_mechanism(mech, word, letters):
    w = re.sub(r"[^A-Za-z]", "", word).upper()
    lt = letters.upper()
    if not w or len(w) < 2:
        return mech
    if mech == "first_letter" and len(lt) >= 2 and lt == w[0] + w[-1]:
        return "outer_letters"
    if mech == "first_letter" and len(lt) == 1 and lt != w[0] and lt == w[-1]:
        return "last_letter"
    return mech


def _describe_p_piece(p):
    mech = p.get("mechanism", "")
    word = p.get("clue_word", "")
    letters = p.get("letters", "")
    indicator = p.get("indicator", "")
    source = p.get("source", "")
    deleted = p.get("deleted", "")
    deleted_word = p.get("deleted_word", "")

    if not mech or not word or not letters:
        return None

    mech = _correct_mechanism(mech, word, letters)

    if mech == "synonym":
        return f'{letters} (synonym of "{word}")'
    elif mech == "abbreviation":
        return f'{letters} (abbreviation of "{word}")'
    elif mech == "anagram_fodder":
        return f'"{word}"'
    elif mech == "first_letter":
        if indicator:
            return f'{letters} ("{indicator}" of "{word}" = first letter(s))'
        return f'{letters} (first letter(s) of "{word}")'
    elif mech == "last_letter":
        if indicator:
            return f'{letters} ("{indicator}" of "{word}" = last letter(s))'
        return f'{letters} (last letter(s) of "{word}")'
    elif mech == "outer_letters":
        if indicator:
            return f'{letters} ("{indicator}" of "{word}" = outer letters)'
        return f'{letters} (outer letters of "{word}")'
    elif mech == "inner_letters":
        if indicator:
            return f'{letters} ("{indicator}" of "{word}" = inner letters)'
        return f'{letters} (inner letters of "{word}")'
    elif mech == "alternating":
        if indicator:
            return f'{letters} ("{indicator}" of "{word}" = alternating letters)'
        return f'{letters} (alternating letters of "{word}")'
    elif mech == "reversal":
        if indicator:
            return f'{letters} (reverse ["{indicator}"] of "{word}")'
        return f'{letters} (reverse of "{word}")'
    elif mech == "deletion":
        if source and deleted:
            if deleted_word:
                return f'{letters} ({source} minus {deleted} ["{deleted_word}"])'
            elif indicator:
                return f'{letters} ({source} minus {deleted} ["{indicator}"])'
            return f'{letters} ({source} minus {deleted})'
        if indicator:
            return f'{letters} ("{indicator}" of "{word}")'
        return f'{letters} (from "{word}")'
    elif mech == "homophone":
        if indicator:
            return f'{letters} (sounds like ["{indicator}"] "{word}")'
        return f'{letters} (sounds like "{word}")'
    elif mech == "hidden":
        return f'"{word}"'
    elif mech == "raw":
        return f'{letters} ("{word}")'
    else:
        return f'{letters} ("{word}", {mech})'


def _build_explanation(clue):
    ai_expl = clue["ai_explanation"] if "ai_explanation" in clue.keys() else None
    if ai_expl:
        return ai_expl

    comps_json = clue["components"] if "components" in clue.keys() else None
    if comps_json:
        try:
            comps = json.loads(comps_json)
            if isinstance(comps, list):
                return None
            pieces = comps.get("ai_pieces", [])
            wtype = comps.get("wordplay_type", "")
            if pieces:
                part_strs = []
                for p in pieces:
                    desc = _describe_p_piece(p)
                    if desc:
                        part_strs.append(desc)
                if not part_strs:
                    return None

                if wtype == "anagram":
                    ana_words = [p.get("clue_word", "") for p in pieces
                                 if p.get("mechanism") == "anagram_fodder"]
                    extras = [s for s, p in zip(part_strs, pieces)
                              if p.get("mechanism") != "anagram_fodder"]
                    fodder = " ".join(ana_words) if ana_words else ""
                    if fodder:
                        result = f'Anagram of "{fodder}"'
                        if extras:
                            result = f'{" + ".join(extras)} + {result}'
                        answer = clue["answer"] if "answer" in clue.keys() else ""
                        return f'{result} = {answer}'
                elif wtype == "hidden":
                    hid_words = [p.get("clue_word", "") for p in pieces
                                 if p.get("mechanism") == "hidden"]
                    answer = clue["answer"] if "answer" in clue.keys() else ""
                    if hid_words:
                        return f'Hidden in "{" ".join(hid_words)}" = {answer}'

                answer = clue["answer"] if "answer" in clue.keys() else ""
                return f'{" + ".join(part_strs)} = {answer}'
        except (json.JSONDecodeError, TypeError):
            pass
    return None


# ---------------------------------------------------------------------------
# SEO helpers (adapted from web/routes/clue_seo.py)
# ---------------------------------------------------------------------------

_WORDPLAY_LABELS_SEO = {
    "anagram": "an anagram",
    "charade": "a charade (building blocks)",
    "container": "a container (one word inside another)",
    "hidden": "a hidden word",
    "reversal": "a reversal",
    "double_definition": "a double definition",
    "cryptic_definition": "a cryptic definition",
    "homophone": "a homophone (sounds like)",
    "deletion": "a deletion",
    "substitution": "a substitution",
    "spoonerism": "a spoonerism",
    "initial_letters": "initial letters",
    "alternation": "alternating letters",
}


def _wordplay_label_seo(wordplay_type):
    return _WORDPLAY_LABELS_SEO.get(wordplay_type, wordplay_type.replace("_", " "))


def generate_meta_description(clue):
    clue_text = clue.get("clue_text", "")
    enum = clue.get("enumeration", "")
    source = source_name_filter(clue.get("source", ""))
    puzzle_number = clue.get("puzzle_number", "")

    core = clue_text
    if enum:
        core += f" ({enum})"

    origin = f"{source} #{puzzle_number}"

    definition = clue.get("definition")
    wordplay_type = clue.get("wordplay_type")
    has_explanation = clue.get("_has_explanation", False)

    if definition and wordplay_type and has_explanation:
        offer = "Full explanation with definition, wordplay type, and step-by-step breakdown."
    elif definition and wordplay_type:
        offer = "Answer with definition and wordplay type."
    elif definition:
        offer = "Answer with definition."
    else:
        offer = "Answer to this cryptic crossword clue."

    desc = f'Cryptic crossword clue: "{core}" from {origin}. {offer}'
    if len(desc) > 160:
        desc = desc[:157] + "..."
    return desc


def generate_faq_schema(clue):
    clue_text = clue.get("clue_text", "")
    enum = clue.get("enumeration", "")
    clue_display = clue_text
    if enum:
        clue_display += f" ({enum})"

    faq_entries = []

    # Q1: What does this clue mean?
    meaning_parts = []
    definition = clue.get("definition")
    wordplay_type = clue.get("wordplay_type")
    answer = clue.get("answer", "")
    explanation = clue.get("_explanation_text")

    if definition:
        meaning_parts.append(f'The definition part of the clue is "{definition}".')
    if wordplay_type:
        meaning_parts.append(f"The wordplay technique is {_wordplay_label_seo(wordplay_type)}.")
    if answer:
        meaning_parts.append(f"The answer is {answer}.")
    if explanation:
        meaning_parts.append(f"Explanation: {explanation}")

    faq_entries.append({
        "@type": "Question",
        "name": f'What does the cryptic crossword clue "{clue_display}" mean?',
        "acceptedAnswer": {
            "@type": "Answer",
            "text": " ".join(meaning_parts) if meaning_parts else "See the page for the full answer.",
        },
    })

    # Q2: What is the answer?
    if answer:
        faq_entries.append({
            "@type": "Question",
            "name": f'What is the answer to "{clue_display}"?',
            "acceptedAnswer": {
                "@type": "Answer",
                "text": f"The answer is {answer}.",
            },
        })

    # Q3: What type of wordplay is used?
    if wordplay_type:
        faq_entries.append({
            "@type": "Question",
            "name": f'What type of wordplay is used in "{clue_display}"?',
            "acceptedAnswer": {
                "@type": "Answer",
                "text": f"This clue uses {_wordplay_label_seo(wordplay_type)} as its wordplay technique.",
            },
        })

    schema = {
        "@context": "https://schema.org",
        "@type": "FAQPage",
        "mainEntity": faq_entries,
    }
    return json.dumps(schema, ensure_ascii=False)


def generate_breadcrumb_schema(clue, base_url="https://clairesclues.xyz"):
    source = source_name_filter(clue.get("source", ""))
    puzzle_number = clue.get("puzzle_number", "")
    clue_text = clue.get("clue_text", "")
    enum = clue.get("enumeration", "")

    clue_display = clue_text
    if enum:
        clue_display += f" ({enum})"
    if len(clue_display) > 60:
        clue_display = clue_display[:57] + "..."

    items = [
        {"@type": "ListItem", "position": 1, "name": "Home", "item": f"{base_url}/"},
    ]
    raw_source = clue.get("source", "")
    if raw_source:
        items.append({
            "@type": "ListItem", "position": 2,
            "name": source,
            "item": f"{base_url}/source/{raw_source}/",
        })
    if puzzle_number:
        items.append({
            "@type": "ListItem", "position": len(items) + 1,
            "name": f"#{puzzle_number}",
            "item": f"{base_url}/puzzle/{raw_source}/{puzzle_number}",
        })
    items.append({
        "@type": "ListItem", "position": len(items) + 1,
        "name": clue_display,
    })

    schema = {
        "@context": "https://schema.org",
        "@type": "BreadcrumbList",
        "itemListElement": items,
    }
    return json.dumps(schema, ensure_ascii=False)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route("/")
def home():
    db = get_db()
    sources = db.execute(f"""
        SELECT source, COUNT(*) as cnt
        FROM clues
        WHERE answer IS NOT NULL AND length(answer) > 0
          AND source IN {ALLOWED_SOURCES_SQL}
        GROUP BY source ORDER BY cnt DESC
    """).fetchall()
    return render_template("home.html", sources=sources)


@app.route("/search")
def search():
    q = request.args.get("q", "").strip()
    if not q or len(q) < 3:
        return render_template("search.html", q=q, results=[], too_short=len(q) > 0)

    db = get_db()
    words = q.split()[:6]
    conditions = []
    params = []
    for word in words:
        conditions.append("lower(c.clue_text) LIKE ?")
        params.append(f"%{word.lower()}%")

    where_clause = " AND ".join(conditions)

    results = db.execute(f"""
        SELECT c.clue_text, c.answer, c.definition, c.enumeration,
               c.source, c.puzzle_number, c.wordplay_type
        FROM clues c
        WHERE {where_clause}
          AND c.answer IS NOT NULL AND length(c.answer) > 0
          AND c.source IN {ALLOWED_SOURCES_SQL}
        ORDER BY c.publication_date DESC
        LIMIT 50
    """, params).fetchall()

    return render_template(
        "search.html",
        q=q,
        results=results,
        total=len(results),
        make_slug=make_slug,
    )


@app.route("/search/suggest")
def search_suggest():
    q = request.args.get("q", "").strip()
    if not q or len(q) < 3:
        return jsonify([])

    db = get_db()

    # Check if it looks like a puzzle number search (e.g. "DT 31180", "telegraph 31180", "29504")
    puzzle_match = re.match(r'^(?:dt|telegraph|times|guardian|independent|daily\s*mail)?\s*#?(\d{4,6})$', q, re.IGNORECASE)
    if puzzle_match:
        num = puzzle_match.group(1)
        rows = db.execute(f"""
            SELECT DISTINCT source, puzzle_number, publication_date
            FROM clues
            WHERE puzzle_number = ? AND source IN {ALLOWED_SOURCES_SQL}
            LIMIT 10
        """, (num,)).fetchall()
        results = []
        for r in rows:
            sname = source_name_filter(r["source"])
            results.append({
                "type": "puzzle",
                "text": f"{sname} #{r['puzzle_number']}",
                "date": r["publication_date"] or "",
                "url": f"/puzzle/{r['source']}/{r['puzzle_number']}",
            })
        if results:
            return jsonify(results)

    # Otherwise search clue text
    words = q.split()[:6]
    conditions = []
    params = []
    for word in words:
        conditions.append("lower(c.clue_text) LIKE ?")
        params.append(f"%{word.lower()}%")

    where_clause = " AND ".join(conditions)

    rows = db.execute(f"""
        SELECT c.clue_text, c.answer, c.enumeration, c.source
        FROM clues c
        WHERE {where_clause}
          AND c.answer IS NOT NULL AND length(c.answer) > 0
          AND c.source IN {ALLOWED_SOURCES_SQL}
        ORDER BY c.publication_date DESC
        LIMIT 8
    """, params).fetchall()

    results = []
    for r in rows:
        slug = make_slug(r["clue_text"], r["answer"])
        if slug:
            enum = f" ({r['enumeration']})" if r["enumeration"] else ""
            results.append({
                "type": "clue",
                "text": r["clue_text"] + enum,
                "answer": r["answer"],
                "source": source_name_filter(r["source"]),
                "url": f"/clue/{slug}",
            })
    return jsonify(results)


@app.route("/source/<source>/")
def source_page(source):
    if source not in ALLOWED_SOURCES:
        abort(404)

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

    row = db.execute(f"""
        SELECT c.clue_text, c.answer, c.definition, c.enumeration, c.source,
               c.puzzle_number, c.clue_number, c.direction, c.publication_date,
               c.wordplay_type, c.ai_explanation, c.explanation,
               se.components, se.confidence
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE c.answer = ? AND c.clue_text LIKE ?
          AND c.source IN {ALLOWED_SOURCES_SQL}
        LIMIT 1
    """, (answer, f"%{clue_text_hint.replace(' ', '%')}%")).fetchone()

    if row is None:
        words = clue_text_hint.split()[:3]
        pattern = "%".join(words) + "%"
        row = db.execute(f"""
            SELECT c.clue_text, c.answer, c.definition, c.enumeration, c.source,
                   c.puzzle_number, c.clue_number, c.direction, c.publication_date,
                   c.wordplay_type, c.ai_explanation, c.explanation,
                   se.components, se.confidence
            FROM clues c
            LEFT JOIN structured_explanations se ON se.clue_id = c.id
            WHERE c.answer = ? AND lower(c.clue_text) LIKE ?
              AND c.source IN {ALLOWED_SOURCES_SQL}
            LIMIT 1
        """, (answer, pattern.lower())).fetchone()

    if row is None:
        abort(404)

    # Build explanation — prefer AI, fall back to human with attribution
    explanation = _build_explanation(row)
    human_explanation = None
    blog_name = None
    blog_url = None

    if not explanation:
        # Use human explanation if available (not for telegraph — Big Dave excluded)
        human_raw = row["explanation"] if "explanation" in row.keys() else None
        source = row["source"]
        if human_raw and source != "telegraph":
            human_explanation = human_raw
            blog_name, blog_url = get_blog_attribution(source, row["puzzle_number"])

    # Build clue dict for SEO helpers
    clue_dict = dict(row)
    clue_dict["_has_explanation"] = (explanation or human_explanation) is not None
    clue_dict["_explanation_text"] = explanation or human_explanation

    meta_description = generate_meta_description(clue_dict)
    faq_schema = generate_faq_schema(clue_dict)
    breadcrumb_schema = generate_breadcrumb_schema(clue_dict)

    # Other appearances
    others = db.execute(f"""
        SELECT source, puzzle_number, publication_date
        FROM clues
        WHERE answer = ? AND clue_text = ?
          AND source IN {ALLOWED_SOURCES_SQL}
          AND id != (
            SELECT id FROM clues WHERE answer = ? AND clue_text = ? LIMIT 1
          )
        ORDER BY publication_date DESC
        LIMIT 5
    """, (answer, row["clue_text"], answer, row["clue_text"])).fetchall()

    source_puzzle_url = get_source_puzzle_url(row["source"], row["puzzle_number"])

    return render_template(
        "clue.html",
        clue=row,
        slug=slug,
        others=others,
        explanation=explanation,
        human_explanation=human_explanation,
        blog_name=blog_name,
        blog_url=blog_url,
        source_puzzle_url=source_puzzle_url,
        meta_description=meta_description,
        faq_schema=faq_schema,
        breadcrumb_schema=breadcrumb_schema,
    )


@app.route("/puzzle/<source>/<puzzle_number>")
def puzzle_page(source, puzzle_number):
    if source not in ALLOWED_SOURCES:
        abort(404)

    db = get_db()

    rows = db.execute("""
        SELECT c.clue_text, c.answer, c.definition, c.enumeration,
               c.clue_number, c.direction, c.publication_date,
               c.wordplay_type, c.ai_explanation, c.explanation,
               se.components, se.confidence
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE c.source = ? AND c.puzzle_number = ?
        ORDER BY CASE c.direction WHEN 'across' THEN 0 WHEN 'down' THEN 1 ELSE 2 END,
                 CAST(c.clue_number AS INTEGER)
    """, (source, puzzle_number)).fetchall()

    if not rows:
        abort(404)

    # Get blog attribution once for the whole puzzle
    blog_name, blog_url = get_blog_attribution(source, puzzle_number) if source != "telegraph" else (None, None)

    clue_dicts = []
    for row in rows:
        d = dict(row)
        ai_expl = _build_explanation(row)
        d["_explanation"] = ai_expl
        if not ai_expl and source != "telegraph":
            d["_human_explanation"] = row["explanation"] if row["explanation"] else None
        else:
            d["_human_explanation"] = None
        clue_dicts.append(d)

    pub_date = clue_dicts[0]["publication_date"] if clue_dicts else None
    across = [c for c in clue_dicts if c["direction"] == "across"]
    down = [c for c in clue_dicts if c["direction"] == "down"]

    source_puzzle_url = get_source_puzzle_url(source, puzzle_number)

    return render_template(
        "puzzle.html",
        source=source,
        puzzle_number=puzzle_number,
        publication_date=pub_date,
        across=across,
        down=down,
        total=len(clue_dicts),
        make_slug=make_slug,
        blog_name=blog_name,
        blog_url=blog_url,
        source_puzzle_url=source_puzzle_url,
    )


@app.route("/sitemap_index.xml")
def sitemap_index():
    path = Path(__file__).resolve().parent / "static" / "sitemaps" / "sitemap_index.xml"
    if not path.exists():
        abort(404)
    return Response(path.read_text(encoding="utf-8"), mimetype="application/xml")


@app.route("/robots.txt")
def robots():
    content = "User-agent: *\nAllow: /\nSitemap: https://clairesclues.xyz/sitemap_index.xml\n"
    return Response(content, mimetype="text/plain")


if __name__ == "__main__":
    app.run(debug=True, port=5001)
