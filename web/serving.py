"""THE serving rule for the public site (week-only, no legacy — user decision
2026-07-13).

A clue is SERVED if and only if:
  1. its source is a served publication (telegraph, times, guardian —
     scope decision 2026-07-04), AND
  2. its stored WFW pass parse renders a card (engine pass, frozen manual
     solve, or a prefill the user confirmed — all human-verified by the
     publish-first process).

Everything else stays in the database as data but gets NO public page: the
clue URL answers 410 Gone, and the page returns at the SAME URL if the clue
is later solved (resurrection). Sitemaps, cross-links and browse pages must
apply the same rule so we never link to a 410.

The card itself is core/wfw_card.stored_card — the SAME renderer the solver
uses (user 2026-07-13: "base it on what we have"), imported lazily so the
site boots without core. The import chain is light (store/screens/render
only, no engine wiring); the /solver mount already set the precedent of core
inside the site process.
"""

import re

from flask import g

SERVED_SOURCES = ("telegraph", "times", "guardian")

# The (source, type_slug) pairs a PUBLIC visitor can browse from the home /
# puzzles pages (user 2026-07-13). Admin sees the full BROWSE_SOURCES list.
# Everyman stays visible pending the user's Everyman discussion.
SERVED_BROWSE = {
    ("telegraph", "cryptic"), ("telegraph", "prize"),
    ("telegraph", "prize-toughie"),          # Sunday weekly prize puzzle (public, 2026-07-15)
    ("times", "cryptic"), ("times", "sunday"),
    ("guardian", "cryptic"), ("guardian", "everyman"),
}

# Solver-workflow chips stripped from the PUBLIC card (user 2026-07-13): the
# verdict badge (PASS), the solving-engine tag (manual/dd/...) and the
# provenance chip a manual piece wears. They are review-surface furniture; a
# public page only exists BECAUSE the parse passed review, so they say nothing.
# These spans contain plain text only (no nesting) — the regex is safe.
_INTERNAL_CHIPS = re.compile(
    r'<span class="wfw-(?:engine|verdict|prov)[^"]*"[^>]*>.*?</span>')


def get_card(clue_id):
    """The rendered WFW card HTML for this clue, or None. Memoised per request.

    Two ways a clue earns a card (and therefore a public page):
      1. a stored PASS parse — the normal case (core.wfw_card.stored_card);
      2. a reviewer INVALID verdict WITH an explanatory comment — the reviewer
         determined the clue uses an unsound cryptic mechanism and wrote why
         (user decision 2026-07-14). There is no wordplay to render, so the card
         shows the answer plus that comment. INVALID with NO comment stays 410
         (nothing to show)."""
    cache = getattr(g, "_wfw_card_cache", None)
    if cache is None:
        cache = g._wfw_card_cache = {}
    if clue_id not in cache:
        html = None
        try:
            from core.wfw_card import stored_card
            core_html = stored_card(clue_id)
            if core_html is not None:
                html = _INTERNAL_CHIPS.sub("", core_html)
        except Exception:
            html = None
        if html is None:
            html = _invalid_card(clue_id)
        cache[clue_id] = html
    return cache[clue_id]


def _invalid_card(clue_id):
    """A served card for a clue the reviewer marked INVALID (an unsound cryptic
    mechanism) AND explained in a comment. No wordplay parse exists, so the card
    shows the answer tiles plus the reviewer's comment — real content standing in
    for the breakdown. Reads wfw_solve + wfw_notes directly with raw SQL (no core
    import), the same discipline as web/wfw_read.py. Returns None when the clue is
    not INVALID or carries no comment (no comment => nothing to serve => 410)."""
    import sqlite3
    from html import escape
    from web.db import get_db
    try:
        db = get_db()
        row = db.execute(
            "SELECT status, answer_text FROM wfw_solve WHERE clue_id = ?",
            (clue_id,)).fetchone()
        if row is None or row["status"] != "invalid":
            return None
        note_row = db.execute(
            "SELECT note FROM wfw_notes WHERE clue_id = ?", (clue_id,)).fetchone()
    except sqlite3.OperationalError:
        return None
    note = (note_row["note"].strip() if note_row and note_row["note"] else "")
    if not note:
        return None
    answer = (row["answer_text"] or "").strip()
    if not answer:
        try:
            arow = db.execute("SELECT answer FROM clues WHERE id = ?",
                              (clue_id,)).fetchone()
            answer = (arow["answer"] if arow and arow["answer"] else "") or ""
        except sqlite3.OperationalError:
            answer = ""
    tiles = []
    for ch in answer:
        if ch.isalpha():
            tiles.append('<span class="wfw-tile" style="background:#f1f5f9;'
                         'border-color:#cbd5e1;color:#334155">%s</span>'
                         % escape(ch.upper()))
        elif ch in "-–—":
            tiles.append('<span class="wfw-tile-sep">&ndash;</span>')
        elif ch.isspace():
            tiles.append('<span class="wfw-tile-gap"></span>')
    tiles_html = "".join(tiles)
    note_html = escape(note).replace("\n", "<br>")
    return ('<div class="wfw-card">'
            '<div class="wfw-tiles">%s</div>'
            '<div class="wfw-banner wfw-banner-warn">'
            '<strong>This clue doesn’t work by the standard cryptic rules.'
            '</strong>'
            '<div style="margin-top:.4rem">%s</div></div>'
            '</div>' % (tiles_html, note_html))


def card_css():
    """The card's embeddable stylesheet (no page-shell rules)."""
    from core.wfw_render import CARD_CSS
    return CARD_CSS


def is_served(source, clue_id):
    """True when this clue gets a public page — ONE truth for the clue route,
    the sitemap and every internal link."""
    if source not in SERVED_SOURCES:
        return False
    return get_card(clue_id) is not None


# ---------------------------------------------------------------------------
# ARCHIVE PAGES — the answer, and nothing else (user decision 2026-08-29)
# ---------------------------------------------------------------------------
# MEASURED that day, origin logs, 15 days: of 44 Googlebot fetches of /clue/*,
# 42 returned 410 and 2 returned 200. Google's list of this site is the corpus
# it learned BEFORE the July relaunch, nearly all of which we then withdrew, and
# GSC reports our crawl purpose as 100% Refresh / <1% Discovery. So the crawl we
# get is spent knocking on doors we bricked up, while the live pages — which sit
# in no sitemap Google has read since 6 August — are never visited at all.
#
# The user's decision: reopen those URLs with the ANSWER ONLY. It uses crawl we
# already have instead of crawl we would have to earn, it costs nothing per page
# because the answers are already in the database, and thin is demonstrably not
# what excludes us — Danword ranks on the same clues with the answer and less
# (see web/related.py). Each page is also a doorway to the tools and to today's
# puzzle, which is the point: someone lands on a 2019 clue and meets a working
# solver.
#
# WHAT IT MUST NOT DO. No explanation of any kind — not the scraped blog text,
# not the 183k parsed explanations, not a definition, not a wordplay type. The
# page says the puzzle is from the archive and that Cordelia works through the
# current ones. It does not say "yet": we are not going to solve these, and
# implying otherwise would be a lie (user, 2026-08-29).
#
# THE SITEMAP IS NOT TOUCHED. `is_served` above is the sitemap's truth and stays
# exactly as it was, so nothing submitted to Google changes and there is no
# fourth swing in sitemap size. Google reaches these URLs from its own memory of
# the site, not from a file we publish.

# The sources the PRE-JULY sitemap carried — the set Google actually holds.
# Mirrors SITEMAP_SOURCES in web/routes/seo.py; the archive can only reopen a
# door Google already knows about.
ARCHIVE_SOURCES = ('telegraph', 'times', 'dailymail', 'guardian', 'independent')

def archive_answer(clue):
    """The answer to show on an archive page, or None if this clue gets no page.

    Called ONLY after the served check has failed, so a clue with a real parse
    always renders the full card and never falls back to this.

    NO DATE WINDOW (user, 2026-08-29). An earlier draft held anything published
    in the last 21 days back, to keep an embargoed prize puzzle out. The user's
    rule is simpler and covers strictly more: no answer, no page. An embargoed
    puzzle has no answer to show, so it is excluded by the answer test itself —
    a date guard adds nothing except an arbitrary line that withholds pages we
    could serve. (And it protected nothing in practice: today's prize puzzle
    31331 already serves its answers in full, with the wordplay.)
    """
    if clue is None:
        return None
    if (clue["source"] or "") not in ARCHIVE_SOURCES:
        return None
    if not (clue["clue_text"] or "").strip():
        return None
    return (clue["answer"] or "").strip() or None


def puzzle_is_served(source, puzzle_number):
    """True when EVERY clue of a puzzle is served — the puzzle-level display
    rule (user 2026-07-15): clues serve one by one; a PUZZLE is displayed only
    when all of its clues are served. One unserved clue hides the whole puzzle.

    Completeness keys off SOLVE STATUS — a pass, or an INVALID with a reviewer
    comment — NOT card renderability. A linked "See N" continuation stub earns a
    pass in the walk but has no standalone card (nothing of its own to render);
    it is still a solved clue, so it counts. This is the SAME rule the browse
    list uses (get_puzzle_list), so the page and the list never disagree.

    (The clue-level is_served above stays card-based: a stub's own clue page
    correctly 410s and a stub never surfaces in search or 'also seen in'.)"""
    if source not in SERVED_SOURCES:
        return False
    from web.db import get_db
    row = get_db().execute(
        """SELECT COUNT(*) AS total,
                  SUM(CASE WHEN w.clue_id IS NOT NULL
                        OR (wi.clue_id IS NOT NULL AND n.note IS NOT NULL
                            AND TRIM(n.note) != '') THEN 1 ELSE 0 END) AS served
           FROM clues c
           LEFT JOIN wfw_solve w  ON w.clue_id = c.id AND w.status = 'pass'
           LEFT JOIN wfw_solve wi ON wi.clue_id = c.id AND wi.status = 'invalid'
           LEFT JOIN wfw_notes  n ON n.clue_id = c.id
           WHERE c.source = ? AND c.puzzle_number = ?""",
        (source, str(puzzle_number)),
    ).fetchone()
    return bool(row and row["total"] and row["served"] == row["total"])


def served_puzzle_numbers(sources=None):
    """The set of (source, puzzle_number) pairs whose EVERY clue is served — the
    puzzle-level display rule in bulk, for the sitemap. SAME per-clue test as
    puzzle_is_served (pass OR INVALID-with-comment), computed in ONE query so the
    sitemap can filter thousands of puzzles cheaply and match the page gate
    exactly (the sitemap must never list a puzzle URL that 410s). Restricted to
    the served sources; non-served sources are dropped."""
    from web.db import get_db
    srcs = [s for s in (sources or SERVED_SOURCES) if s in SERVED_SOURCES]
    if not srcs:
        return set()
    placeholders = ",".join("?" for _ in srcs)
    rows = get_db().execute(
        f"""SELECT c.source, c.puzzle_number
           FROM clues c
           LEFT JOIN wfw_solve w  ON w.clue_id = c.id AND w.status = 'pass'
           LEFT JOIN wfw_solve wi ON wi.clue_id = c.id AND wi.status = 'invalid'
           LEFT JOIN wfw_notes  n ON n.clue_id = c.id
           WHERE c.source IN ({placeholders})
           GROUP BY c.source, c.puzzle_number
           HAVING COUNT(*) = SUM(CASE WHEN w.clue_id IS NOT NULL
               OR (wi.clue_id IS NOT NULL AND n.note IS NOT NULL
                   AND TRIM(n.note) != '') THEN 1 ELSE 0 END)""",
        srcs,
    ).fetchall()
    return {(r["source"], str(r["puzzle_number"])) for r in rows}
