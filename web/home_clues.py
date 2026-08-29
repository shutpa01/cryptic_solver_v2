"""Today's Telegraph clues, for the home page.

WHY THIS EXISTS (decided with the user 2026-08-29)
--------------------------------------------------
Google has ~18 of our pages indexed against 55,551 crawled-but-not-indexed, and
requesting indexing by hand on the puzzle pages has never got one in. So the
gate that fails is not discovery, it is selection: a NEW url has to win an
indexing decision, and ours do not.

The home page has already won it — it is indexed and it ranks first for
`site:justcordelia.com`. Putting the day's clue text ON that page therefore
skips the decision entirely: no new url, no new judgement, just fresh text on a
page Google already keeps. Googlebot fetched "/" on 5 of the 15 days to 29 Aug
(origin log), so this is a real, if unhurried, channel.

The clue TEXT is the point. It is the exact string a solver types into a search
box, which is why each card shows the clue and not the answer.

THE SELECTION RULE (user, 2026-08-29)
-------------------------------------
Telegraph only for now, to see whether it works at all.

    Mon-Sat   five clues from that day's cryptic (Mon-Fri) or prize cryptic (Sat)
    Sunday    three from the prize cryptic, two from the prize toughie

"Trickiest" is approximated by the number of wordplay pieces in the WFW card —
the richest breakdown, the same proxy `scripts/youtube_short.py` uses to choose
its clue. It is richness, not measured difficulty; we have no difficulty score.

THE SERVING RULE
----------------
Every clue offered here goes through `web.serving.is_served`, so the block can
never link to a 410. A puzzle is only considered once every one of its clues
has a stored pass.

NO NEW URLS. Every link points at a /clue/<slug> page that already exists.

PERFORMANCE
-----------
`clues` holds 606k rows and the wfw_* tables carry no indexes (see web/related.py
for the measurement). So nothing here joins clues to wfw_*: the small tables are
scanned once each, the intersection is done in Python, and clues are fetched via
idx_clues_source. The whole result is cached at module level and thrown away
when clues_master.db changes on disk — which is exactly when the dashboard
deploys a new one.
"""

import os

from web.db import get_db

# Telegraph only while we find out whether the idea works at all.
SOURCE = "telegraph"

# How many clues the block shows, and the Sunday split between the prize
# cryptic and the prize toughie.
BLOCK_SIZE = 5
SUNDAY_PRIZE = 3
SUNDAY_TOUGHIE = 2

# How far back to look for a fully-served puzzle. Three weeks is generous: if
# nothing inside it is served the block simply does not render.
LOOKBACK_DAYS = 21

# Over-fetch before the is_served pass, because a stored pass does not
# guarantee a renderable card.
CANDIDATE_POOL = 12

_cache = {"stamp": None, "clues": None}


def _db_stamp():
    """(mtime, size) of clues_master.db, or None if it cannot be read.

    The cache key. A dashboard deploy replaces this file, so the block picks up
    a new puzzle without a restart.
    """
    try:
        from flask import current_app
        path = current_app.config["CLUES_DB"]
    except Exception:
        return None
    try:
        s = os.stat(path)
        return (s.st_mtime, s.st_size)
    except OSError:
        return None


def _passed_ids():
    """Clue ids with a stored PASS — one scan of the small wfw_solve table."""
    return {r["clue_id"] for r in get_db().execute(
        "SELECT clue_id FROM wfw_solve WHERE status = 'pass'")}


def _piece_counts(ids):
    """clue_id -> number of wordplay pieces, for the ids we care about.

    One scan of wfw_piece (small, unindexed), counted in Python.
    """
    counts = {}
    for r in get_db().execute("SELECT clue_id FROM wfw_piece"):
        cid = r["clue_id"]
        if cid in ids:
            counts[cid] = counts.get(cid, 0) + 1
    return counts


def _recent_puzzles():
    """Telegraph puzzles inside the lookback, newest first.

    Returns a list of (puzzle_number, publication_date, clue_ids).
    """
    rows = get_db().execute(
        """SELECT id, puzzle_number, publication_date
             FROM clues
            WHERE source = ?
              AND publication_date >= date('now', ?)
            ORDER BY publication_date DESC""",
        (SOURCE, "-%d days" % LOOKBACK_DAYS),
    ).fetchall()

    order = []
    grouped = {}
    for r in rows:
        key = (r["puzzle_number"], r["publication_date"])
        if key not in grouped:
            grouped[key] = []
            order.append(key)
        grouped[key].append(r["id"])
    return [(num, date, grouped[(num, date)]) for num, date in order]


def _served_puzzles(passed):
    """Recent puzzles where EVERY clue has a pass, newest first, with type.

    Mirrors the puzzle-level display rule: a puzzle the visitor cannot fully
    reach is not a puzzle we advertise.
    """
    from web.models import classify_puzzle

    out = []
    for number, date, ids in _recent_puzzles():
        if not ids or any(i not in passed for i in ids):
            continue
        type_slug, type_label = classify_puzzle(SOURCE, number, date)
        if not type_slug:
            continue
        out.append({
            "puzzle_number": number,
            "publication_date": date,
            "type_slug": type_slug,
            "type_label": type_label,
            "clue_ids": ids,
        })
    return out


def _is_sunday(date_str):
    row = get_db().execute(
        "SELECT CAST(strftime('%w', ?) AS INTEGER)", (date_str,)).fetchone()
    return row is not None and row[0] == 0


def _pick(puzzle, counts, want):
    """The `want` richest served clues of one puzzle, as template dicts."""
    from web.models import get_source_puzzle_url  # noqa: F401  (kept for parity)
    from web.routes.clue import generate_clue_slug
    from web.serving import is_served

    ranked = sorted(puzzle["clue_ids"],
                    key=lambda i: (-counts.get(i, 0), i))[:CANDIDATE_POOL]
    if not ranked:
        return []

    rows = get_db().execute(
        "SELECT id, clue_number, direction, clue_text, enumeration"
        "  FROM clues WHERE id IN (%s)" % ",".join("?" * len(ranked)),
        tuple(ranked),
    ).fetchall()
    by_id = {r["id"]: r for r in rows}

    out = []
    for cid in ranked:
        if len(out) >= want:
            break
        r = by_id.get(cid)
        if r is None or not (r["clue_text"] or "").strip():
            continue
        if not is_served(SOURCE, cid):
            continue
        out.append({
            "clue_text": r["clue_text"],
            "enumeration": r["enumeration"] or "",
            "clue_number": r["clue_number"] or "",
            "direction": (r["direction"] or "").title(),
            "url": "/clue/%s" % generate_clue_slug(r["clue_text"], clue_id=cid),
            "puzzle_number": puzzle["puzzle_number"],
            "type_label": puzzle["type_label"],
            "publication_date": puzzle["publication_date"],
            "puzzle_url": "/%s/%s/%s" % (
                SOURCE, puzzle["type_slug"], puzzle["puzzle_number"]),
        })
    return out


def _build():
    passed = _passed_ids()
    if not passed:
        return []

    served = _served_puzzles(passed)
    if not served:
        return []

    # The lead puzzle: the newest cryptic or prize cryptic. The toughie is
    # never the lead — on a Sunday it supplies the last two slots only.
    lead = next((p for p in served
                 if p["type_slug"] in ("cryptic", "prize")), None)
    if lead is None:
        return []

    picks = [(lead, BLOCK_SIZE)]

    if _is_sunday(lead["publication_date"]):
        toughie = next((p for p in served
                        if p["type_slug"] == "prize-toughie"
                        and p["publication_date"] == lead["publication_date"]),
                       None)
        if toughie is not None:
            picks = [(lead, SUNDAY_PRIZE), (toughie, SUNDAY_TOUGHIE)]

    wanted_ids = set()
    for puzzle, _n in picks:
        wanted_ids.update(puzzle["clue_ids"])
    counts = _piece_counts(wanted_ids)

    out = []
    for puzzle, n in picks:
        out.extend(_pick(puzzle, counts, n))
    return out


def home_puzzles(clues):
    """The distinct puzzles the block draws from, in the order they appear.

    On a Sunday the block spans two (prize cryptic + prize toughie), so the
    heading cannot name a single puzzle and the template needs the list.
    """
    out = []
    for c in clues:
        key = (c["type_label"], c["puzzle_number"])
        if key not in [(p["type_label"], p["puzzle_number"]) for p in out]:
            out.append({
                "type_label": c["type_label"],
                "puzzle_number": c["puzzle_number"],
                "publication_date": c["publication_date"],
                "url": c["puzzle_url"],
            })
    return out


def get_home_clues():
    """The clues the home-page block should show. [] when there is nothing.

    Cached until clues_master.db changes on disk.
    """
    stamp = _db_stamp()
    if _cache["clues"] is not None and _cache["stamp"] == stamp:
        return _cache["clues"]
    try:
        clues = _build()
    except Exception:
        # The home page must render even if this block cannot.
        clues = []
    _cache["stamp"] = stamp
    _cache["clues"] = clues
    return clues
