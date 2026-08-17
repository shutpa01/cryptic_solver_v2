"""The four-step hint ladder, served.

Steps, each giving more away: definition, clue type, answer, full explanation.

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
_MECHANISM_PHRASING = {
    "definition": "{text} is the definition",
    "synonym": "{text} gives {value}",
    "abbreviation": "{text} is short for {value}",
    "anagram_fodder": "{text} supplies the letters",
    "selection": "{text} contributes {value}",
    "raw": "{text} is used as it stands, giving {value}",
    "first_letter": "the first letter of {text} gives {value}",
    "hidden": "{value} is hidden inside {text}",
    "hidden_reversed": "{value} is hidden backwards inside {text}",
    "homophone": "{text} sounds like {value}",
    "deletion": "{value} is removed from {text}",
    "alternate": "alternate letters of {text} give {value}",
    "definition_by_example": "{text} is an example, giving {value}",
    "replacement_letter": "a letter of {text} is replaced, giving {value}",
}

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

    return {
        "definition": _definition(pieces) if passed else None,
        "clue_type": _clue_type(solve, clue) if passed else None,
        "answer": (clue["answer"] or "").upper() or None,
        "explanation": _explanation(pieces, clue["clue_text"]) if passed else None,
        "enumeration": clue["enumeration"] or None,
    }


def _in_clue_order(pieces, clue_text):
    """Sort pieces by where their words fall in the clue.

    Stored order is insertion order, which interleaves the definition among the
    letters and reads as a jumble. Clue order is the rule the public overlay
    already follows: read the clue left to right and see what each part does.
    Pieces whose text cannot be located keep their stored position, at the end,
    rather than being dropped.
    """
    haystack = (clue_text or "").lower()
    ordered = []
    for piece in pieces:
        text = (piece["text"] or "").strip().lower()
        at = haystack.find(text) if text else -1
        ordered.append(((0, at) if at >= 0 else (1, piece["ord"]), piece))
    ordered.sort(key=lambda pair: pair[0])
    return [piece for _key, piece in ordered]


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


def _explanation(pieces, clue_text):
    lines = []
    for piece in _in_clue_order(pieces, clue_text):
        line = _render_piece(piece)
        if line:
            lines.append(line)
    return lines or None


def _render_piece(piece):
    text = (piece["text"] or "").strip()
    value = (piece["value"] or "").strip().upper()
    mechanism = (piece["mechanism"] or "").strip()
    role = (piece["role"] or "").strip()
    note = (piece["note"] or "").strip()

    if not text:
        return None

    if role == "indicator":
        kind = mechanism.replace("_", " ") if mechanism else ""
        line = (f"{_quote(text)} is the {kind} indicator" if kind
                else f"{_quote(text)} is the indicator")
    elif role == "link":
        line = f"{_quote(text)} joins the parts"
    elif mechanism in _MECHANISM_PHRASING:
        line = _MECHANISM_PHRASING[mechanism].format(text=_quote(text), value=value)
    elif value:
        line = f"{_quote(text)} gives {value}"
    else:
        # No mechanism recorded and no value: say only what is known.
        line = f"{_quote(text)} is part of the wordplay"

    if note:
        line += f" ({note})"
    return line + "."
