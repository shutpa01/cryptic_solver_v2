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


def scramble_perm(n, seed):
    """A deterministic NON-identity permutation of range(n). Used to shuffle a
    charade's tile order so the pieces no longer sit in answer order (tapping
    left-to-right must not spell the answer)."""
    if n < 2:
        return list(range(n))
    idx = list(range(n))
    s = seed & 0x7FFFFFFF
    for i in range(n - 1, 0, -1):
        s = (s * 1103515245 + 12345) & 0x7FFFFFFF
        j = s % (i + 1)
        idx[i], idx[j] = idx[j], idx[i]
    if idx == list(range(n)):             # never leave it in answer order
        idx = idx[1:] + idx[:1]
    return idx


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


# ---- hidden / lurker: atomise the clue, reveal the mechanism -----------------
def build_hidden(clue_text, answer, reversed_hidden):
    """A lurker's answer is a consecutive substring of the clue's letters
    (reversed for hidden_reversed). We atomise the SPANNING words (+/- one
    neighbour) into single-letter tiles coloured per clue word, and mark which
    consecutive run is the answer. The player selects that run among decoys.
    Returns None if the substring can't be located (caller falls back)."""
    a = "".join(ch for ch in (answer or "").upper() if ch.isalpha())
    target = a[::-1] if reversed_hidden else a
    words = (clue_text or "").split()
    seq = []                                   # (char, word_index) for every clue letter
    for wi, w in enumerate(words):
        for ch in w.upper():
            if ch.isalpha():
                seq.append((ch, wi))
    full = "".join(c for c, _ in seq)
    start = full.find(target)
    if start < 0 or not target:
        return None
    end = start + len(target)                  # exclusive
    w_start, w_end = seq[start][1], seq[end - 1][1]
    win_lo = max(0, w_start - 1)               # one decoy word each side
    win_hi = min(len(words) - 1, w_end + 1)
    letters, seq_to_win = [], {}
    for i, (ch, wi) in enumerate(seq):
        if win_lo <= wi <= win_hi:
            seq_to_win[i] = len(letters)
            letters.append({"ch": ch, "colour": PALETTE[(wi - win_lo) % len(PALETTE)][0]})
    span = [seq_to_win[start], seq_to_win[end - 1]]
    return {"letters": letters, "answer_span": span,
            "target": target, "reversed": bool(reversed_hidden)}


# ---- acrostic: show the words, take initials of a consecutive run ------------
def build_acrostic(clue_text, answer):
    """An acrostic answer is the first letters of a run of consecutive clue
    words. We show the words (+/- one neighbour as decoys), coloured per word,
    and mark which run's initials are the answer. The player selects that run
    of words. Returns None if the initials don't contain the answer."""
    a = "".join(ch for ch in (answer or "").upper() if ch.isalpha())
    words = []                                  # (display, letters) for words with letters
    for w in (clue_text or "").split():
        lw = "".join(ch for ch in w.upper() if ch.isalpha())
        if lw:
            words.append((w, lw))
    initials = "".join(lw[0] for _, lw in words)
    start = initials.find(a)
    if start < 0 or not a:
        return None
    end = start + len(a)                         # exclusive, in word-index space
    win_lo = max(0, start - 1)
    win_hi = min(len(words) - 1, end)            # last answer word is end-1; +1 decoy
    win_words, seq_to_win = [], {}
    for i in range(len(words)):
        if win_lo <= i <= win_hi:
            seq_to_win[i] = len(win_words)
            win_words.append({"text": words[i][0],
                              "colour": PALETTE[(i - win_lo) % len(PALETTE)][0]})
    span = [seq_to_win[start], seq_to_win[end - 1]]
    return {"words": win_words, "answer_span": span, "target": a}


# ---- jumble board: arrange the answer's own letters (DD + anagram-hybrid) ----
NEUTRAL = "#64748b"

def build_jumble_board(answer, foothold, seed_base):
    """Single-letter tiles of the ANSWER's own letters, jumbled (deterministic),
    played on the anagram slot board. Used for double_definition (no foothold —
    you unscramble from the two definitions) and, as an interim, anagram-hybrids
    (with the normal foothold). Returns (atoms, assembly_order, anchors)."""
    letters = [ch for ch in (answer or "").upper() if ch.isalpha()]
    n = len(letters)
    tray = list(jumble("".join(letters), seed_base))       # shuffle preserves the multiset
    atoms = [{"true": ch, "display": ch, "colour": NEUTRAL, "source_index": 0,
              "source_word": "", "mechanism": "jumble"} for ch in tray]
    used, assembly_order = [False] * n, []
    for ch in letters:                                      # answer pos -> a tray tile
        for i, a in enumerate(atoms):
            if not used[i] and a["true"] == ch:
                used[i] = True; assembly_order.append(i); break
        else:
            assembly_order.append(None)
    anchors = []
    if foothold and n > 1:
        k = min(anchor_count(n), n - 1)
        positions = []
        for i in range(k):
            p = min(n - 1, int((i + 0.5) * n / k)) if k else 0
            while p in positions:
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
    board = "assembly"
    hidden_data = None
    acrostic_data = None
    from crypticker.mech_kind import effective_op   # local: mech_kind imports this module
    op = effective_op(parse)              # /hs 'manual' clues: kind from the pieces
    if op in ("hidden", "hidden_reversed"):
        hidden_data = build_hidden(parse.clue_text, parse.answer_text,
                                   op == "hidden_reversed")
        if hidden_data is None:
            raise ValueError("hidden substring not found for %s" % clue_id)
        board, kind, mode, highlighted = "hidden", "hidden", "hard", False
        _sp = hidden_data["answer_span"]
        _run = hidden_data["letters"][_sp[0]:_sp[1] + 1]
        _seq = _run if not hidden_data["reversed"] else list(reversed(_run))
        atoms = out_atoms = [{"true": x["ch"], "display": x["ch"], "colour": x["colour"],
                              "source_index": 0, "source_word": "hidden",
                              "mechanism": "hidden"} for x in _seq]
        assembly_order = list(range(len(out_atoms)))
    elif op == "acrostic":
        acrostic_data = build_acrostic(parse.clue_text, parse.answer_text)
        if acrostic_data is None:
            raise ValueError("acrostic run not found for %s" % clue_id)
        board, kind, mode, highlighted = "acrostic", "acrostic", "hard", False
        _sp, _tg = acrostic_data["answer_span"], acrostic_data["target"]
        atoms = out_atoms = [{"true": _tg[j], "display": _tg[j],
                              "colour": acrostic_data["words"][_sp[0] + j]["colour"],
                              "source_index": 0, "source_word": "acrostic",
                              "mechanism": "acrostic"} for j in range(len(_tg))]
        assembly_order = list(range(len(out_atoms)))
    elif op == "double_definition":
        board = mode = "anagram"; kind = "dd"; highlighted = False
        atoms, assembly_order, anchors = build_jumble_board(
            parse.answer_text, foothold=False, seed_base=clue_id * 100 + 7)
        out_atoms = atoms
    elif op in ("anagram_charade", "anagram_container"):
        board = mode = "anagram"; kind = "anagram_hybrid"; highlighted = False
        atoms, assembly_order, anchors = build_jumble_board(
            parse.answer_text, foothold=True, seed_base=clue_id * 100 + 3)
        out_atoms = atoms
    elif op == "anagram":
        mode = "anagram"
        board = "anagram"
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
        if kind == "charade" and len(out_atoms) >= 2:
            # A charade's tiles sit in ANSWER order, so tapping them left-to-right
            # assembles the answer with no thought. Scramble the tile order (the
            # letters stay jumbled too) so the player must work out the order.
            perm = scramble_perm(len(out_atoms), clue_id * 7 + 1)
            out_atoms = [out_atoms[perm[i]] for i in range(len(perm))]
            assembly_order = [perm.index(p) for p in range(len(perm))]

    row = conn.execute(
        "SELECT source, puzzle_number, clue_number, direction, publication_date "
        "FROM clues WHERE id=?", (clue_id,)).fetchone()
    source = {}
    if row:
        pos = (row[2] or "") + (row[3][:1].upper() if row[3] else "")
        source = {"publication": row[0], "puzzle": row[1],
                  "position": pos, "pub_date": row[4]}

    expl = explanation(parse, atoms, assembly_order, source)
    if board == "hidden":
        d = parse.definition.text if parse.definition else "?"
        rev = "(reversed) " if hidden_data["reversed"] else ""
        expl = "%s is hidden %sin the surface. Definition: %s." % (
            parse.answer_text, rev, d)
    elif board == "acrostic":
        d = parse.definition.text if parse.definition else "?"
        expl = "%s = the initial letters of consecutive words. Definition: %s." % (
            parse.answer_text, d)
    elif kind == "dd":
        expl = "Double definition — the clue defines %s two ways." % parse.answer_text
    elif kind == "anagram_hybrid":
        d = parse.definition.text if parse.definition else "?"
        expl = "Anagram-based wordplay: the letters rearrange to %s. Definition: %s." % (
            parse.answer_text, d)

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
        "board": board,                   # assembly / anagram / hidden / acrostic — drives the UI
        "hidden": hidden_data,            # hidden-board data (letters/span); None otherwise
        "acrostic": acrostic_data,        # acrostic-board data (words/span); None otherwise
        "anchors": anchors,               # pre-placed positions (anagram foothold); [] otherwise
        "jumbled": any(o["display"] != o["true"] for o in out_atoms),
        "bonus": bonus_question(parse, atoms),
        "explanation": expl,
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
