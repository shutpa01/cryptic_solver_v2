"""Tuition zone — learn how cryptic crosswords work, type by type."""

import json
import random
import re
from pathlib import Path

from flask import Blueprint, render_template, abort, request

bp = Blueprint("learn", __name__)

_DATA = None
_DATA_PATH = Path(__file__).resolve().parent.parent.parent / "data" / "tuition_clues.json"

# Friendly names, short descriptions, and emoji for each type
TYPE_INFO = {
    "anagram": {
        "label": "Anagram",
        "short": "Letters rearranged to spell the answer",
        "icon": "🔀",
        "colour": "violet",
    },
    "charade": {
        "label": "Charade",
        "short": "Pieces joined end to end",
        "icon": "🔗",
        "colour": "blue",
    },
    "container": {
        "label": "Container",
        "short": "One word placed inside another",
        "icon": "📦",
        "colour": "amber",
    },
    "deletion": {
        "label": "Deletion",
        "short": "Letters removed from a word",
        "icon": "✂️",
        "colour": "red",
    },
    "double_definition": {
        "label": "Double Definition",
        "short": "Two meanings, one answer",
        "icon": "🪞",
        "colour": "teal",
    },
    "hidden": {
        "label": "Hidden Word",
        "short": "Answer hiding inside the clue",
        "icon": "🔍",
        "colour": "emerald",
    },
    "reversal": {
        "label": "Reversal",
        "short": "A word spelled backwards",
        "icon": "🔄",
        "colour": "orange",
    },
    "homophone": {
        "label": "Homophone",
        "short": "Answer sounds like another word",
        "icon": "👂",
        "colour": "rose",
    },
    "acrostic": {
        "label": "Acrostic",
        "short": "First letters of words spell the answer",
        "icon": "📝",
        "colour": "cyan",
    },
    "cryptic_definition": {
        "label": "Cryptic Definition",
        "short": "The whole clue is a tricky definition",
        "icon": "🧩",
        "colour": "indigo",
    },
}

# Display order
TYPE_ORDER = [
    "hidden", "anagram", "charade", "container", "reversal",
    "deletion", "double_definition", "homophone", "acrostic",
    "cryptic_definition",
]


def _load_data():
    global _DATA
    if _DATA is None:
        with open(_DATA_PATH, encoding="utf-8") as f:
            _DATA = json.load(f)
    return _DATA


def _get_type_data(wtype):
    data = _load_data()
    type_data = data.get("types", {}).get(wtype)
    if not type_data:
        return None
    return type_data


def _parse_components(clue):
    """Extract pieces from a clue's components field."""
    comps = clue.get("components")
    if not comps:
        return [], None
    if isinstance(comps, str):
        comps = json.loads(comps)
    pieces = comps.get("ai_pieces", [])
    assembly = comps.get("assembly")
    return pieces, assembly


def _highlight_hidden(clue_text, answer):
    """Highlight the hidden answer letters within the clue text.

    Returns HTML with hidden letters wrapped in <strong> tags.
    E.g. 'grim peloton' with answer IMPEL -> 'gr<strong>IM PEL</strong>oton'
    """
    if not answer or not clue_text:
        return clue_text

    answer_upper = re.sub(r"[^A-Z]", "", answer.upper())
    letters_only = []
    letter_positions = []
    for i, ch in enumerate(clue_text):
        if ch.isalpha():
            letters_only.append(ch.upper())
            letter_positions.append(i)

    letters_str = "".join(letters_only)
    idx = letters_str.find(answer_upper)
    if idx < 0:
        return clue_text

    # Build result with highlighting
    start_pos = letter_positions[idx]
    end_pos = letter_positions[idx + len(answer_upper) - 1]

    result = ""
    result += clue_text[:start_pos]
    result += '<strong class="text-emerald-700 bg-emerald-100 px-0.5 rounded">'
    result += clue_text[start_pos:end_pos + 1].upper()
    result += '</strong>'
    result += clue_text[end_pos + 1:]

    return result


def _build_colour_map(clue):
    """Build a word-to-role colour map for a clue.

    Returns dict mapping lowercase word -> (role, colour_class)
    Roles: definition, indicator, fodder, result
    """
    import re

    clue_text = clue.get("clue_text", "")
    definition = (clue.get("definition") or "").lower().strip()
    pieces, assembly = _parse_components(clue)

    colour_map = {}  # word_index -> (role, bg_class, text_class)

    words = re.findall(r"[A-Za-z''-]+", clue_text)
    words_lower = [w.lower() for w in words]

    # Mark definition words
    if definition:
        def_words = re.findall(r"[A-Za-z''-]+", definition)
        def_lower = [w.lower() for w in def_words]
        # Find contiguous match in clue
        for start in range(len(words_lower)):
            if words_lower[start:start + len(def_lower)] == def_lower:
                for i in range(start, start + len(def_lower)):
                    colour_map[i] = ("definition", "bg-purple-100", "text-purple-800")
                break

    # Mark pieces
    for p in pieces:
        clue_word = (p.get("clue_word") or "").lower().strip()
        mechanism = p.get("mechanism", "")
        if not clue_word:
            continue

        piece_words = re.findall(r"[A-Za-z''-]+", clue_word)
        piece_lower = [w.lower() for w in piece_words]

        # Determine role colour
        if mechanism in ("anagram_fodder", "literal"):
            role = ("fodder", "bg-blue-100", "text-blue-800")
        elif mechanism == "hidden":
            role = ("fodder", "bg-blue-100", "text-blue-800")
        elif mechanism in ("synonym", "abbreviation", "first_letter", "last_letter",
                           "reversal", "deletion", "alternate_letters", "sound_of"):
            role = ("wordplay", "bg-amber-100", "text-amber-800")
        else:
            role = ("wordplay", "bg-amber-100", "text-amber-800")

        # Find in clue
        for start in range(len(words_lower)):
            if words_lower[start:start + len(piece_lower)] == piece_lower:
                for i in range(start, start + len(piece_lower)):
                    if i not in colour_map:  # Don't overwrite definition
                        colour_map[i] = role
                break

    return words, colour_map


@bp.route("/learn")
def learn_index():
    """Landing page — shows all wordplay types as cards."""
    data = _load_data()
    types = []
    for wtype in TYPE_ORDER:
        info = TYPE_INFO.get(wtype, {})
        type_data = data.get("types", {}).get(wtype, {})
        types.append({
            "slug": wtype,
            "label": info.get("label", wtype),
            "short": info.get("short", ""),
            "icon": info.get("icon", ""),
            "colour": info.get("colour", "gray"),
            "count": type_data.get("count", 0),
            "description": type_data.get("description", ""),
        })

    return render_template("learn.html", types=types)


@bp.route("/learn/<wtype>")
def learn_type(wtype):
    """Type page — explains the type with visual clue breakdowns."""
    type_data = _get_type_data(wtype)
    if not type_data:
        abort(404)

    info = TYPE_INFO.get(wtype, {})
    clues = type_data.get("clues", [])

    # Build colour maps for each clue
    clue_cards = []
    for clue in clues:
        words, colour_map = _build_colour_map(clue)
        pieces, assembly = _parse_components(clue)
        # For hidden word clues, generate highlighted text
        hidden_highlight = None
        if wtype == "hidden" or (assembly and assembly.get("op") == "hidden"):
            hidden_highlight = _highlight_hidden(clue.get("clue_text", ""), clue.get("answer", ""))
        clue_cards.append({
            "clue": clue,
            "words": words,
            "colour_map": colour_map,
            "pieces": pieces,
            "assembly": assembly,
            "hidden_highlight": hidden_highlight,
        })

    return render_template(
        "learn_type.html",
        wtype=wtype,
        label=info.get("label", wtype),
        icon=info.get("icon", ""),
        colour=info.get("colour", "gray"),
        description=type_data.get("description", ""),
        clue_cards=clue_cards,
        total=len(clues),
    )


@bp.route("/learn/<wtype>/practice")
def learn_practice(wtype):
    """Practice mode — random clue, try to solve it."""
    type_data = _get_type_data(wtype)
    if not type_data:
        abort(404)

    info = TYPE_INFO.get(wtype, {})
    clues = type_data.get("clues", [])

    if not clues:
        abort(404)

    # Pick a random clue (or specific index if provided)
    idx = request.args.get("i", type=int)
    if idx is not None and 0 <= idx < len(clues):
        clue = clues[idx]
    else:
        clue = random.choice(clues)
        idx = clues.index(clue)

    words, colour_map = _build_colour_map(clue)
    pieces, assembly = _parse_components(clue)

    hidden_highlight = None
    if wtype == "hidden" or (assembly and assembly.get("op") == "hidden"):
        hidden_highlight = _highlight_hidden(clue.get("clue_text", ""), clue.get("answer", ""))

    return render_template(
        "learn_practice.html",
        wtype=wtype,
        label=info.get("label", wtype),
        icon=info.get("icon", ""),
        colour=info.get("colour", "gray"),
        clue=clue,
        words=words,
        colour_map=colour_map,
        pieces=pieces,
        assembly=assembly,
        hidden_highlight=hidden_highlight,
        idx=idx,
        total=len(clues),
    )
