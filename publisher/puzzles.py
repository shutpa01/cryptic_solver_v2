"""Build the widget's grid model from a publisher puzzle file.

Phase one reads the Telegraph puzzle JSONs already in ``scraper/telegraph/``.
Those files carry a complete per-square model — each square has ``Number``,
``Blank``, ``WordAcrossID``, ``WordDownID`` and ``Letter`` — so nothing has to
be reconstructed the way ``web/grid.py`` has to reconstruct it for the site.

Two things this module is careful about:

* **Solutions never travel with the model.** ``build_model`` returns the shape
  the browser needs (cells, entries, clue text, enumeration) and
  ``solutions_for`` returns the answers separately, for server-side Check and
  Reveal only. Nothing in the payload the widget receives contains a letter of
  the solution.
* **A length mismatch is reported, never papered over.** If an entry's stated
  answer does not fit its squares, the entry is marked ``unverified`` and its
  solution is dropped rather than truncated to fit.
"""

import html
import json
import re
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TELEGRAPH_DIR = PROJECT_ROOT / "scraper" / "telegraph"

SOURCES = {
    "telegraph": TELEGRAPH_DIR,
}


class PuzzleNotFound(Exception):
    pass


def _clean_text(s):
    """Un-escape the HTML entities the Telegraph feed embeds in clue text.

    Their clues arrive with `&ndash;` and `&#039;` inline. Left alone these
    render literally in the clue list.
    """
    if not s:
        return ""
    return html.unescape(s).strip()


def _clean_answer(s):
    return re.sub(r"[^A-Za-z]", "", s or "").upper()


def _split_filename(source, name):
    """Return (kind, number) for a puzzle filename, or None.

    Two naming vintages live side by side in scraper/telegraph:
    `telegraph_cryptic-crossword_80116.json` and the older
    `telegraph_cryptic-crossword-80116.json`. Both are read.
    """
    m = re.match(rf"{re.escape(source)}_(.+?)[_-](\d+)\.json$", name)
    return (m.group(1), m.group(2)) if m else None


def find_puzzle_file(source, number):
    """Locate the JSON for a puzzle. Returns a Path or None."""
    directory = SOURCES.get(source)
    if directory is None or not directory.is_dir():
        return None
    number = str(number)
    for path in sorted(directory.glob(f"{source}_*.json")):
        parts = _split_filename(source, path.name)
        if parts and parts[1] == number:
            return path
    return None


def list_puzzles(source, limit=50):
    """Return [{number, kind, title}] for the puzzles available locally.

    Phase one only: the real pipe is a publisher feed, not a directory scan.
    """
    directory = SOURCES.get(source)
    if directory is None or not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob(f"{source}_*.json"), reverse=True):
        parts = _split_filename(source, path.name)
        if not parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        copy = _copy_block(data)
        if not copy:
            continue
        out.append({
            "number": parts[1],
            "kind": parts[0],
            "title": _clean_text(copy.get("title")),
        })
        if len(out) >= limit:
            break
    return out


def summarise(source):
    """A light index of every local puzzle: number, title, display number, and
    whether the feed carries answers.

    Deliberately avoids `build_model` — this reads each file once and looks at
    a handful of fields, so listing every puzzle stays quick enough for a page
    load. Used by the development index to say which puzzles are worth opening.
    """
    directory = SOURCES.get(source)
    if directory is None or not directory.is_dir():
        return []
    out = []
    for path in sorted(directory.glob(f"{source}_*.json"), reverse=True):
        parts = _split_filename(source, path.name)
        if not parts:
            continue
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        copy = _copy_block(data)
        grid = _grid_block(data)
        if not copy or not grid:
            continue
        letters = 0
        whites = 0
        for row in grid:
            for square in row:
                if not isinstance(square, dict) or square.get("Blank") == "blank":
                    continue
                whites += 1
                if (square.get("Letter") or "").strip():
                    letters += 1
        title = _clean_text(copy.get("title"))
        out.append({
            "number": parts[1],
            "kind": parts[0],
            "title": title,
            "display_number": _display_number(title),
            "has_answers": whites > 0 and letters >= whites,
        })
    return out


def _copy_block(data):
    """The feed nests its payload under `json` or `data` depending on vintage."""
    inner = data.get("json") or data.get("data") or data
    return inner.get("copy") or {}


def _grid_block(data):
    inner = data.get("json") or data.get("data") or data
    return inner.get("grid")


def _entry_id(direction, number):
    return f"{'a' if direction == 'across' else 'd'}{number}"


def _display_number(title):
    """The printed puzzle number from a feed title, or None.

    'Toughie Crossword No 3740' -> '3740'. Prefer the number quoted after "No";
    a bare digit hunt would pick up years and other noise.
    """
    match = re.search(r"\bNo\.?\s*0*(\d+)", title or "")
    return match.group(1) if match else None


def build_model(source, number):
    """Return (model, solutions).

    model     — everything the browser gets. No solution letters.
    solutions — {entry_id: "ANSWER"} for entries whose answer is verified
                against the grid. Server-side only.
    """
    path = find_puzzle_file(source, number)
    if path is None:
        raise PuzzleNotFound(f"no local puzzle file for {source} {number}")

    data = json.loads(path.read_text(encoding="utf-8"))
    copy = _copy_block(data)
    grid_rows = _grid_block(data)
    if not copy or not grid_rows:
        raise PuzzleNotFound(f"{path.name} has no grid array")

    size = copy.get("gridsize", {})
    rows = int(size.get("rows", len(grid_rows)))
    cols = int(size.get("cols", len(grid_rows[0]) if grid_rows else 15))

    # --- word id -> the entry that owns those squares -----------------------
    # A linked ("See 1 Across") clue keeps its own list item, but its squares
    # belong to the main entry: typing across the join must not stop at it.
    word_owner = {}
    entries = {}
    entry_words = {}    # entry id -> [word id, ...] IN READING ORDER
    word_dir = {}       # word id -> 'across' | 'down'
    order = {"across": [], "down": []}
    stubs = {}          # linked entry id -> main entry id
    raw_answers = {}

    for section in copy.get("clues", []):
        direction = "down" if section.get("title", "").lower().startswith("down") \
            else "across"
        for clue in section.get("clues", []):
            num = int(clue["number"])
            eid = _entry_id(direction, num)
            word_id = clue.get("word")
            entries[eid] = {
                "id": eid,
                "number": num,
                "dir": direction,
                "clue": _clean_text(clue.get("clue")),
                "enum": _clean_text(clue.get("format")),
                "cells": [],
                "links": [],
            }
            order[direction].append(eid)
            entry_words[eid] = []
            if word_id is not None:
                word_owner[word_id] = eid
                word_dir[word_id] = direction
                entry_words[eid].append(word_id)
            raw_answers[eid] = _clean_answer(clue.get("answer"))

    # Second pass, once every entry exists: redirect linked words to the main.
    #
    # Reading order is main-word-then-links, NOT grid order. A backward link is
    # real and common: 24 Across "Writer having everyone inside…" (5,5,3) links
    # to 1 Across, and EDGAR sits at 24 while ALLAN POE continues at 1, near
    # the top of the grid. Sorting those thirteen squares row-major spells the
    # answer backwards, which is what the validation sweep caught.
    for section in copy.get("clues", []):
        direction = "down" if section.get("title", "").lower().startswith("down") \
            else "across"
        for clue in section.get("clues", []):
            links = clue.get("links") or []
            if not links:
                continue
            main_id = _entry_id(direction, int(clue["number"]))
            for link in links:
                link_dir = (link.get("direction") or "").lower()
                link_dir = "down" if link_dir.startswith("down") else "across"
                link_id = _entry_id(link_dir, int(link["number"]))
                if link_id == main_id or link_id in stubs:
                    continue
                stubs[link_id] = main_id
                entries[main_id]["links"].append(link_id)
                # Any square carrying the linked word now answers to the main
                # entry, so the two read as one run of cells.
                for wid, owner in list(word_owner.items()):
                    if owner == link_id:
                        word_owner[wid] = main_id
                        entry_words[main_id].append(wid)
                entry_words[link_id] = []

    # --- squares ------------------------------------------------------------
    cells = []
    word_cells = {}         # word id -> [[r, c], ...]
    solution_letters = {}   # (r, c) -> letter
    for r, row in enumerate(grid_rows):
        out_row = []
        for c, square in enumerate(row):
            if not isinstance(square, dict) or square.get("Blank") == "blank":
                out_row.append(None)
                continue
            across_word = square.get("WordAcrossID")
            down_word = square.get("WordDownID")
            across_id = word_owner.get(across_word) if across_word not in ("", None) else None
            down_id = word_owner.get(down_word) if down_word not in ("", None) else None
            # NOT `number` — that is the puzzle number, this function's
            # argument. Shadowing it here left every model carrying the last
            # square's label as its puzzle number, which is almost always "".
            # Downstream that made every puzzle share one browser storage key
            # AND one server-side solution cache, so Check and Reveal answered
            # with a different puzzle's letters.
            square_number = square.get("Number")
            out_row.append({
                "n": int(square_number) if str(square_number).isdigit() else None,
                "a": across_id,
                "d": down_id,
            })
            for wid in (across_word, down_word):
                if wid not in ("", None):
                    word_cells.setdefault(wid, []).append([r, c])
            letter = (square.get("Letter") or "").strip().upper()
            if len(letter) == 1 and letter.isalpha():
                solution_letters[(r, c)] = letter
        cells.append(out_row)

    # Each word's own squares run left-to-right or top-to-bottom; the entry is
    # those runs concatenated in link order.
    for wid, cell_list in word_cells.items():
        cell_list.sort(key=(lambda rc: (rc[1], rc[0])) if word_dir.get(wid) == "down"
                       else (lambda rc: (rc[0], rc[1])))
    for eid, entry in entries.items():
        run = []
        for wid in entry_words.get(eid, []):
            run.extend(word_cells.get(wid, []))
        entry["cells"] = run
        entry["len"] = len(run)

    # --- solutions, verified against the squares ----------------------------
    solutions = {}
    for eid, entry in entries.items():
        if eid in stubs or not entry["cells"]:
            continue
        from_grid = "".join(solution_letters.get(tuple(rc), "") for rc in entry["cells"])
        stated = raw_answers.get(eid, "")
        if len(from_grid) == entry["len"]:
            solutions[eid] = from_grid
            # A stated answer that disagrees with the grid is a data fault, not
            # something to silently prefer one way or the other.
            entry["unverified"] = bool(stated) and stated != from_grid
        elif stated and len(stated) == entry["len"]:
            solutions[eid] = stated
            entry["unverified"] = False
        else:
            entry["unverified"] = True

    # A stub keeps its list row ("See 1 Across") but selects the main entry.
    for stub_id, main_id in stubs.items():
        if stub_id in entries:
            entries[stub_id]["stub_of"] = main_id
            entries[stub_id]["cells"] = []
            entries[stub_id]["len"] = 0
            entries[stub_id].pop("unverified", None)

    model = {
        "source": source,
        "number": str(number),
        # The number the paper prints, dug out of the title. The feed files are
        # named by the Telegraph's internal id (93933) while everything
        # public — and every row in clues_master.db — uses the display number
        # (Toughie No 3740). Without this the two never join up.
        "display_number": _display_number(copy.get("title")),
        "title": _clean_text(copy.get("title")),
        "setter": _clean_text(copy.get("setter") or copy.get("byline")),
        "rows": rows,
        "cols": cols,
        "cells": cells,
        "entries": [entries[eid] for eid in order["across"] + order["down"]],
        "across": order["across"],
        "down": order["down"],
    }
    return model, solutions
