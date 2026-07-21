"""Crypticker converter: a verified clue -> the daily-puzzle JSON (spec 4.5).

Reuses the live WFW tile logic (crypticker.serve_data.atoms_for, which itself
reuses core.store.load_parse + core.wfw_render). This module adds only the
game layer on top: mode assignment, deterministic jumbles, the bonus question,
source attribution and a plain-text explanation. Read-only against the DB.

Tile colour is the WFW per-source palette, carried verbatim on each atom
(`colour`) plus its `source_index` — no role buckets, no recolouring.

Scope: assembly clues (charade / container / container_charade). Anagram is a
separate atom shape (single-letter fodder tiles) and is the next increment.
"""
import sqlite3, os, sys, json, math

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from core.store import load_parse
from crypticker.serve_data import atoms_for

DB = os.path.join(os.path.dirname(__file__), "..", "data", "clues_master.db")

TIME_LIMIT_S = 120
SKIP_PENALTY_S = 30

# Anagram foothold: pre-place ceil(length/3) tiles in their correct positions
# (7->3, 10->4, 15->5). Anagram answers run 7-15 letters.
ANAGRAM_MIN_LEN = 7
ANAGRAM_MAX_LEN = 15

from core.wfw_render import PALETTE


def anchor_count(n):
    return math.ceil(n / 3)


# ---- classification: which of the three assembly shapes is this? -------------
def classify(atoms, assembly_order):
    """Return (kind, jumble_eligible).

    charade  : tiles concatenate in clue order (assembly == identity)  -> jumble-safe
    reorder  : whole tiles, shuffled out of clue order
    split    : one clue word produced >1 tile (a word broken around another)
    """
    split = len({a["source_index"] for a in atoms}) < len(atoms)
    identity = assembly_order == list(range(len(atoms)))
    if split:
        kind = "split"
    elif identity:
        kind = "charade"
    else:
        kind = "reorder"
    return kind, (kind == "charade")


# ---- deterministic jumble: same for every player, never the true order -------
def jumble(text, seed):
    """A stable scramble of `text` driven by `seed` (so all players see the same
    one). Never applied to 1-2 letter tiles by the caller; guaranteed != text."""
    chars = list(text)
    s = seed & 0x7FFFFFFF
    for i in range(len(chars) - 1, 0, -1):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        j = s % (i + 1)
        chars[i], chars[j] = chars[j], chars[i]
    out = "".join(chars)
    if out == text:                       # degenerate shuffle -> rotate one place
        out = text[1:] + text[0]
    return out


# ---- anagram: single-letter fodder tiles + a positional foothold -------------
def build_anagram(parse, max_anchors=None):
    """Anagram tiles: one per fodder letter, in fodder (clue) order, coloured by
    which fodder word it came from (WFW per-source palette). Returns
    (atoms, assembly_order, anchors). assembly_order[j] = the tile index that
    belongs in answer position j; anchors are the positions revealed at start.
    max_anchors caps the foothold (the extreme tail strips it to 1-2)."""
    letters = parse.answer_letters()
    fodder = [(i, s) for i, s in enumerate(parse.sources)
              if s.mechanism == "anagram_fodder"]

    atoms = []                             # single-letter tiles in fodder order
    for si, s in fodder:
        for ch in (s.value or "").upper():
            if ch.isalpha():
                atoms.append({
                    "true": ch,
                    "colour": PALETTE[si % len(PALETTE)][0],
                    "source_index": si,
                    "source_word": s.text,
                    "mechanism": "anagram_fodder",
                })

    # canonical solution: greedily match each answer letter to an unused tile
    used = [False] * len(atoms)
    assembly_order = []
    for ch in letters:
        for i, a in enumerate(atoms):
            if not used[i] and a["true"] == ch:
                used[i] = True
                assembly_order.append(i)
                break
        else:
            assembly_order.append(None)    # fodder/answer mismatch -> caller checks

    # foothold: reveal ceil(n/3) evenly-spaced positions (deterministic, same for all)
    n = len(letters)
    k = anchor_count(n)
    if max_anchors is not None:
        k = min(k, max_anchors)          # extreme tail strips the foothold
    k = min(k, max(0, n - 1))
    positions = []
    for i in range(k):
        p = min(n - 1, int((i + 0.5) * n / k)) if k else 0
        while p in positions:              # guarantee k distinct positions
            p = (p + 1) % n
        positions.append(p)
    positions.sort()
    anchors = [{"pos": p, "atom_index": assembly_order[p], "letter": letters[p]}
               for p in positions if assembly_order[p] is not None]
    return atoms, assembly_order, anchors


# ---- bonus question: derived from the parse, deterministic -------------------
def bonus_question(parse, atoms):
    """One 3-option multiple-choice question about this clue. v1: the definition
    question (works for every clue; options = definition + 2 other clue words).
    Indicator / source questions are a later increment."""
    def_text = parse.definition.text if parse.definition else None
    words = [w for w in parse.clue_text.replace("–", " ").split() if w.strip(",.")]
    if not def_text:
        return None
    distractors = [w for w in words if w.lower() not in def_text.lower()][:2]
    options = [def_text] + distractors
    # deterministic placement of the correct answer: by clue_id parity
    correct_index = parse.clue_id % len(options) if hasattr(parse, "clue_id") else 0
    options = options[:]                  # keep order stable; put def at correct_index
    options.remove(def_text)
    options.insert(correct_index, def_text)
    return {"question": "What is the definition?",
            "options": options, "correct_index": correct_index}


# ---- plain-text explanation --------------------------------------------------
def explanation(parse, atoms, assembly_order, source):
    pieces = " + ".join("%s (%s)" % (atoms[i]["true"], atoms[i]["source_word"])
                        for i in assembly_order)
    d = parse.definition.text if parse.definition else "?"
    attr = ""
    if source.get("publication"):
        attr = " %s %s%s." % (source["publication"], source.get("puzzle", ""),
                              (" " + source["position"]) if source.get("position") else "")
    return "%s → %s. Definition: %s.%s" % (pieces, parse.answer_text, d, attr)


# ---- one clue -> the schema's clue object ------------------------------------
def convert_clue(conn, clue_id, mode=None, max_anchors=None):
    parse = load_parse(conn, clue_id)
    if parse is None:
        raise ValueError("no parse for clue %s" % clue_id)
    parse.clue_id = clue_id

    anchors = []
    if (parse.operation or "") == "anagram":
        mode = "anagram"
        highlighted = False               # spec open decision 6: no def highlight on anagram
        atoms, assembly_order, anchors = build_anagram(parse, max_anchors)
        kind = "anagram"
        out_atoms = [{"true": a["true"], "display": a["true"],
                      "colour": a["colour"], "source_index": a["source_index"],
                      "source_word": a["source_word"], "mechanism": a["mechanism"]}
                     for a in atoms]
    else:
        atoms, assembly_order = atoms_for(parse)
        kind, jumble_ok = classify(atoms, assembly_order)
        # mode: caller override, else auto (jumble-safe charade -> easy, else hard)
        if mode is None:
            mode = "easy" if jumble_ok else "hard"
        highlighted = (mode == "easy")
        # A charade's tiles sit in ANSWER order, so clean tiles would spell the
        # answer outright (OFF + SPRING = OFFSPRING). Jumble charade tiles in
        # EVERY mode so the answer is never on show. Reorder/split tiles stay
        # clean — for those the non-trivial order IS the puzzle. Mode then only
        # controls whether the definition is marked.
        out_atoms = []
        for i, a in enumerate(atoms):
            true = a["true"]
            display = true
            if jumble_ok and len(true) >= 3:
                display = jumble(true, clue_id * 100 + i)
            out_atoms.append({
                "true": true, "display": display,
                "colour": a["colour"], "source_index": a["source_index"],
                "source_word": a["source_word"], "mechanism": a["mechanism"],
            })

    row = conn.execute(
        "SELECT source, puzzle_number, clue_number, direction, publication_date "
        "FROM clues WHERE id=?", (clue_id,)).fetchone()
    source = {}
    if row:
        pos = (row[2] or "") + (row[3][:1].upper() if row[3] else "")
        source = {"publication": row[0], "puzzle": row[1],
                  "position": pos, "pub_date": row[4]}

    return {
        "clue_id": clue_id,
        "source": source,
        "clue": parse.clue_text,
        "enum": parse.enumeration(),
        "answer": parse.answer_text,
        "mode": mode,
        "kind": kind,                     # charade / reorder / split (curation aid)
        "definition": {"text": parse.definition.text if parse.definition else "",
                       "highlighted": highlighted},
        "atoms": out_atoms,
        "assembly_order": assembly_order,
        "anchors": anchors,               # pre-placed positions (anagram foothold); [] otherwise
        "jumbled": any(o["display"] != o["true"] for o in out_atoms),
        "bonus": bonus_question(parse, atoms),
        "explanation": explanation(parse, atoms, assembly_order, source),
    }


# ---- N clues -> a day file ---------------------------------------------------
def build_day(conn, clue_ids, puzzle_no, date, modes=None):
    modes = modes or {}
    clues = [convert_clue(conn, cid, modes.get(cid)) for cid in clue_ids]
    return {
        "puzzle_no": puzzle_no,
        "date": date,
        "time_limit_s": TIME_LIMIT_S,
        "skip_penalty_s": SKIP_PENALTY_S,
        "clues": clues,
    }


if __name__ == "__main__":
    conn = sqlite3.connect(DB)
    # smoke test: assembly clues + real anagrams (GRANITE, WRESTLE, TRAIPSE)
    day = build_day(conn, [1710439, 1740016, 1712378, 1710243, 1710250, 1711600],
                    puzzle_no=1, date="2026-09-01")
    out_dir = os.path.join(os.path.dirname(__file__), "days")
    os.makedirs(out_dir, exist_ok=True)
    out_path = os.path.join(out_dir, "sample_day.json")
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(day, f, indent=2, default=str, ensure_ascii=False)
    print("wrote", out_path)
    for c in day["clues"]:
        built = "".join(c["atoms"][i]["true"] for i in c["assembly_order"]
                        if i is not None)
        ans = "".join(ch for ch in c["answer"].upper() if ch.isalpha())
        ok = "OK" if built == ans else "MISMATCH"
        line = ("  %-9s mode=%-7s kind=%-8s tiles=%s rebuild=%s %s"
                % (c["answer"], c["mode"], c["kind"],
                   [a["display"] for a in c["atoms"]], built, ok))
        if c["anchors"]:
            line += "  anchors=" + str([(a["pos"], a["letter"]) for a in c["anchors"]])
        print(line)
