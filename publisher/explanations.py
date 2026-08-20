"""The four-step hint ladder, served.

Steps, each giving more away: definition, clue type, answer, full explanation.

**The full explanation IS the site's WFW breakdown.** `web.wfw_read` is imported
directly — the one deliberate crossing of the rule that this package imports
nothing from `web` (see `publisher_build_decisions`). Reimplementing it was
tried and thrown away: the format is the product, and a second copy of 950
lines of transform handling, selection-fodder highlighting and colour
assignment would fork from the page it is meant to reproduce the first time
either changed. The coupling is narrow — `wfw_read` is a leaf that deliberately
avoids `core/` imports and needs only `current_app.config["CLUES_DB"]` and
Flask's `g`, both of which this app has, so it travels if the package is ever
lifted out.

**This module serves RENDERED TEXT, never parse primitives.** No atom ids, no
mechanism codes, no roles or ordinals reach the browser. That line is the whole
commercial argument of the licensing design: the structured corpus is the one
genuinely unbuyable asset, and shipping it as data is what would let a customer
walk away with the capability rather than the output. Rendering server-side
also means the detailed clue type — the part that is new to the world — is
delivered as a phrase, not as a taxonomy anyone could learn from.

Joining a puzzle to its explanations goes through the DISPLAY number, not the
feed's filename: `scraper/telegraph/telegraph_toughie-crossword_93933.json` is
"Toughie Crossword No 3740", and clues_master.db knows it as 3740.
"""

import re
import sqlite3

# How a piece reads once it is a sentence rather than a row. Anything not
# listed falls back to a plain, honest phrasing — the alternative is inventing
# grammar for a mechanism nobody has described yet.
_OPERATION_LABELS = {
    "double_definition": "Double definition",
    "triple_definition": "Triple definition",
    "cd": "Cryptic definition",
    "anagram": "Anagram",
    "anagram_container": "Anagram inside a container",
    "charade": "Charade",
    "charade_deletion": "Charade with a deletion",
    "charade_homophone": "Charade with a homophone",
    "container": "Container",
    "container_charade": "Container and charade",
    "deletion": "Deletion",
    "hidden": "Hidden word",
    "hidden_reversed": "Hidden word reversed",
    "reversal": "Reversal",
    "reversal_charade": "Reversal of a charade",
    "spoonerism": "Spoonerism",
    "acrostic": "Initial letters",
    "lit": "&lit",
}


def _connect(clues_db):
    conn = sqlite3.connect(f"file:{clues_db}?mode=ro", uri=True)
    conn.row_factory = sqlite3.Row
    return conn


def _quote(text):
    return "‘" + (text or "").strip() + "’"


def load_clue_index(clues_db, source, display_number):
    """Map "<number><a|d>" -> clue row id for one puzzle.

    Returns {} when the puzzle is not in the database at all, which is the
    normal case for older feed files.
    """
    if not display_number:
        return {}
    db = _connect(clues_db)
    try:
        rows = db.execute(
            "SELECT id, clue_number, direction FROM clues "
            "WHERE source = ? AND puzzle_number = ?",
            (source, str(display_number)),
        ).fetchall()
    finally:
        db.close()

    index = {}
    for row in rows:
        number = re.sub(r"\D", "", row["clue_number"] or "")
        direction = (row["direction"] or "").lower()
        if not number or direction not in ("across", "down"):
            continue
        index[f"{'a' if direction == 'across' else 'd'}{number}"] = row["id"]
    return index


def steps_for(clues_db, clue_id):
    """The four hint steps for one clue, rendered.

    Any step with nothing solid behind it is returned as None and the widget
    simply does not offer it. A hint ladder that pads itself with a guess is
    worse than one with a missing rung.
    """
    db = _connect(clues_db)
    try:
        clue = db.execute(
            "SELECT id, clue_text, answer, enumeration, wordplay_type "
            "FROM clues WHERE id = ?",
            (clue_id,),
        ).fetchone()
        if clue is None:
            return None

        solve = db.execute(
            "SELECT operation, status FROM wfw_solve WHERE clue_id = ?",
            (clue_id,),
        ).fetchone()
        pieces = db.execute(
            "SELECT role, ord, text, value, mechanism, note FROM wfw_piece "
            "WHERE clue_id = ? ORDER BY ord",
            (clue_id,),
        ).fetchall()
    finally:
        db.close()

    passed = solve is not None and (solve["status"] or "").lower() == "pass"
    breakdown = _breakdown(clue_id) if passed else None

    return {
        "definition": _definition(pieces) if passed else None,
        # Prefer the site's own label: it names EVERY mechanism a clue uses,
        # which is the detailed clue type the licensing design treats as the
        # thing that is new to the world. The local map is only a fallback for
        # a pass with no stored atoms.
        "clue_type": (breakdown["label"] if breakdown else None)
                     or (_clue_type(solve, clue) if passed else None),
        "answer": (clue["answer"] or "").upper() or None,
        # The site's own card, as HTML. The widget draws no explanation of its
        # own any more — see card_html().
        "explanation": card_html(clue_id) if passed else None,
        "enumeration": clue["enumeration"] or None,
    }


def card_html(clue_id):
    """The site's OWN rendered WFW card for this clue, or None.

    This is the whole point: not a second renderer that resembles the card, but
    the card — `core.wfw_card.stored_card` via `web.serving.get_card`, the very
    call the live clue page makes, including its strip of the internal review
    chips (PASS / engine / provisional). Whatever the site shows, the widget
    shows, and it cannot drift.

    It replaces a hand-kept parallel that drifted three times in one day
    (2026-08-20): deletion rows dropped, the homophone partner never printed,
    a replacement letter labelled "unclued" while its clue words were thrown
    away. Each was invisible until a reader noticed. See
    `web/wfw_read.load_breakdown`, which now survives only for the clue-type
    label and is still guarded by `web/test_wfw_overlay_contract.py`.

    Returns None when the clue has no served card, which is the same condition
    that gives the clue no public page.
    """
    try:
        from web.serving import get_card
        return get_card(clue_id)
    except Exception:                       # noqa: BLE001 — a missing card is not an error
        return None


def card_stylesheet():
    """The card's embeddable stylesheet — no page-shell rules, by its own
    contract (`web.serving.card_css`). Served once with the widget shell."""
    try:
        from web.serving import card_css
        return card_css()
    except Exception:                       # noqa: BLE001
        return ""


def _breakdown(clue_id):
    """The WFW breakdown, with colours resolved, ready to render.

    Shape matches the overlay on the site: a clue-type pill, the answer as
    tiles coloured by the piece that placed each letter, the one-line assembly,
    and word-by-word rows in clue order. Returns None when the clue has a pass
    but no stored atoms, which is the one case the overlay cannot draw either.
    """
    from web.wfw_read import load_breakdown

    data = load_breakdown(clue_id)
    if data is None:
        return None

    fg_by_source = data.get("src_fg") or {}
    fill_by_source = data.get("src_fill") or {}

    tiles = []
    for tile in data.get("answer_tiles", []):
        if "sep" in tile:
            tiles.append({"sep": True})
            continue
        source = tile.get("source_index")
        tiles.append({
            "char": tile.get("char", ""),
            # A letter no piece accounts for is drawn plain rather than given a
            # colour it has not earned.
            "fg": fg_by_source.get(source),
            "fill": fill_by_source.get(source),
        })

    rows = []
    for row in data.get("rows", []):
        rows.append({
            "pill": row.get("pill", ""),
            "fg": row.get("fg"),
            "fill": row.get("fill"),
            # detail_html carries the site's fodder-letter highlighting; the
            # rest of the line is escaped where it is built.
            "html": row.get("detail_html"),
            "text": row.get("detail", ""),
        })

    return {
        "label": data.get("operation_label"),
        "summary": data.get("summary"),
        "tiles": tiles,
        "rows": rows,
    }


def _definition(pieces):
    words = [p["text"] for p in pieces
             if (p["role"] == "definition" or p["mechanism"] == "definition")
             and (p["text"] or "").strip()]
    if not words:
        return None
    if len(words) == 1:
        return f"The definition is {_quote(words[0])}."
    joined = " and ".join(_quote(w) for w in words)
    return f"This clue has more than one definition: {joined}."


def _clue_type(solve, clue):
    """The detailed type, as a phrase.

    'manual' is a filing state, not a clue type — it means a person entered the
    parse without the engine naming a mechanism. Saying "Manual" to a solver
    would be meaningless, so fall back to the stored wordplay type, and if
    there is none, say nothing rather than guess.
    """
    operation = (solve["operation"] or "").strip().lower()
    if operation and operation != "manual":
        label = _OPERATION_LABELS.get(operation)
        if label is None:
            label = operation.replace("_", " ").capitalize()
        return label
    fallback = (clue["wordplay_type"] or "").strip()
    if fallback:
        return fallback.replace("_", " ").capitalize()
    return None
