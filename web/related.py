"""Lateral links between clue pages — same indicator, same answer, same shape.

WHY THIS EXISTS (measured 2026-08-29, do not re-derive)
-------------------------------------------------------
Google has 18 of our pages indexed and 55,551 crawled-but-not-indexed. Danword
carries the SAME clue with less information than us — the answer and nothing
else, no definition and no wordplay — and it is indexed while we are not. So
thin content is not the explanation.

The measurable difference on the page: their clue page links to about twenty
other clue pages, ours linked to two neighbours and its own puzzle. A dense
internal link graph is how a crawler reaches deep pages and how it decides to
come back. That is a mechanism, not a theory about quality.

Crucially these link SIDEWAYS across the archive, not within one puzzle. Every
clue is already one hop from every other clue in the same puzzle via the puzzle
page, so same-puzzle links add nothing a crawler can use — which is why all
three blocks below EXCLUDE the clue's own puzzle.

THE SERVING RULE
----------------
Every candidate goes through `web.serving.is_served`, which its own docstring
names as "ONE truth for the clue route, the sitemap and every internal link".
A link to an unserved clue would point at a 410.

NO NEW URLS. Every link here points at a /clue/<slug> page that already exists.

PERFORMANCE — WHY THIS IS NOT ONE SQL JOIN
------------------------------------------
`clues` holds 606,118 rows; `wfw_piece` (29,785) and `wfw_solve` (7,353) carry
NO indexes at all. Joining clues to wfw_solve therefore degenerates into a scan
of 606k rows probing an unindexed table, and one page took over two minutes.

So the work is done the other way round: scan the two small tables once each to
get a set of clue ids, intersect in Python, then fetch the surviving clues by
PRIMARY KEY. Each of those steps measures in hundredths of a second. Adding an
index would be the obvious fix but that is a schema change, which needs the
user's approval (CLAUDE.md).
"""

from flask import g

from web.db import get_db
from web.serving import SERVED_SOURCES, is_served

# How many links each block shows. Enough to be a real path through the archive,
# few enough that the block stays readable and the is_served pass stays cheap.
BLOCK_SIZE = 6

# How many clues to hydrate before the is_served filter. Over-fetch, because a
# stored pass does not guarantee a renderable card.
CANDIDATE_POOL = 24


def _passed_ids():
    """Clue ids with a stored PASS. One scan of 7,353 rows, memoised per request."""
    cache = getattr(g, "_related_passed_ids", None)
    if cache is None:
        cache = g._related_passed_ids = {
            r["clue_id"] for r in
            get_db().execute("SELECT clue_id FROM wfw_solve WHERE status = 'pass'")
        }
    return cache


def _hydrate(ids, clue_id, source, puzzle_number):
    """Ids -> served link dicts, newest first, capped at BLOCK_SIZE.

    The anchor text is the CLUE, never the answer: these pages are reached by
    people mid-solve and a sibling's answer in a link would be a spoiler.
    """
    ids = {i for i in ids if i != clue_id} & _passed_ids()
    if not ids:
        return []

    from web.models import classify_puzzle
    from web.routes.clue import generate_clue_slug

    db = get_db()
    marks = ",".join("?" * len(SERVED_SOURCES))
    rows = db.execute(
        """SELECT id, clue_text, source, puzzle_number, publication_date
             FROM clues
            WHERE id IN (%s)
              AND source IN (%s)
              AND NOT (source = ? AND puzzle_number = ?)
            ORDER BY publication_date DESC, id DESC
            LIMIT %d""" % (",".join("?" * len(ids)), marks, CANDIDATE_POOL),
        tuple(ids) + tuple(SERVED_SOURCES) + (source, str(puzzle_number)),
    ).fetchall()

    out = []
    for r in rows:
        if len(out) >= BLOCK_SIZE:
            break
        if not is_served(r["source"], r["id"]):
            continue
        _slug, type_label = classify_puzzle(
            r["source"], r["puzzle_number"], r["publication_date"])
        out.append({
            "clue_text": r["clue_text"],
            "url": "/clue/%s" % generate_clue_slug(r["clue_text"], clue_id=r["id"]),
            "source": r["source"].title(),
            "type_label": type_label or "",
            "puzzle_number": r["puzzle_number"],
        })
    return out


def _ids_with_indicator(text, note):
    return {r["clue_id"] for r in get_db().execute(
        """SELECT clue_id FROM wfw_piece
            WHERE role = 'indicator'
              AND LOWER(TRIM(text)) = LOWER(?)
              AND TRIM(note) = ?""", (text, note))}


def _ids_with_note(note):
    return {r["clue_id"] for r in get_db().execute(
        """SELECT clue_id FROM wfw_piece
            WHERE role = 'indicator' AND TRIM(note) = ?""", (note,))}


def _own_indicators(clue_id):
    """This clue's (text, note) indicator pairs, in clue order, deduplicated."""
    rows = get_db().execute(
        """SELECT text, note FROM wfw_piece
            WHERE clue_id = ? AND role = 'indicator'
              AND TRIM(COALESCE(text, '')) != ''
              AND TRIM(COALESCE(note, '')) != ''
            ORDER BY ord""", (clue_id,)).fetchall()
    seen, out = set(), []
    for r in rows:
        pair = ((r["text"] or "").strip(), (r["note"] or "").strip())
        if pair[0] and pair[1] and pair not in seen:
            seen.add(pair)
            out.append(pair)
    return out


def _role(note):
    """'anagram indicator' -> 'anagram'. The heading supplies the rest."""
    return note[:-len(" indicator")] if note.endswith(" indicator") else note


def indicator_blocks(clue_id, source, puzzle_number):
    """One block per indicator this clue uses: other clues using the same word
    in the same role.

    The role comes from the piece's `note`, which is what the card itself
    displays — so a block heading cannot claim a role the page does not show.
    Matching is on the note as well as the word: 'about' is a container
    indicator in one clue and a reversal indicator in another, and conflating
    them would assert something untrue.
    """
    blocks = []
    for text, note in _own_indicators(clue_id):
        links = _hydrate(_ids_with_indicator(text, note),
                         clue_id, source, puzzle_number)
        if links:
            blocks.append({"word": text, "role": _role(note), "links": links})
    return blocks


def same_answer(clue_id, answer, source, puzzle_number):
    """Other clues whose answer is this one's — the most-asked question in the
    niche, and a link the answer sites cannot make with any explanation behind it.

    `clues.answer` is indexed, so this one is a plain equality lookup.
    """
    answer = (answer or "").strip()
    if not answer:
        return []
    ids = {r["id"] for r in get_db().execute(
        "SELECT id FROM clues WHERE answer = ? COLLATE NOCASE", (answer,))}
    return _hydrate(ids, clue_id, source, puzzle_number)


def same_shape(clue_id, source, puzzle_number):
    """Other clues built from the same set of indicator roles.

    Deliberately keyed on the ROLE SET actually recorded on this clue — a clue
    with an anagram indicator and a container indicator finds other clues with
    both. It is NOT the canonical clue-type label, which web.wfw_read derives at
    render time and which is stored nowhere queryable; claiming that label here
    would assert something this query has not checked.
    """
    notes = []
    for _text, note in _own_indicators(clue_id):
        if note not in notes:
            notes.append(note)
    if not notes:
        return [], []
    ids = None
    for note in notes:
        got = _ids_with_note(note)
        ids = got if ids is None else (ids & got)
        if not ids:
            return [], []
    return (_hydrate(ids, clue_id, source, puzzle_number),
            [_role(n) for n in notes])


def related_blocks(clue_id, answer, source, puzzle_number):
    """Everything the template needs, or empty lists when there is nothing to say."""
    shape_links, shape_roles = same_shape(clue_id, source, puzzle_number)
    return {
        "indicators": indicator_blocks(clue_id, source, puzzle_number),
        "same_answer": same_answer(clue_id, answer, source, puzzle_number),
        "same_shape": shape_links,
        "same_shape_roles": shape_roles,
    }
