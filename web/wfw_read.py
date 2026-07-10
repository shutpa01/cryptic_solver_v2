"""WFW read path for the live site (phase 1 of the live-site plumbing, 2026-07-10).

Reads a clue's word-for-word solve DIRECTLY from the wfw_* tables with raw SQL —
deliberately NO core/ imports (the site must not depend on the solver runtime).
Serves ONLY status='pass' parses; anything else returns None and the caller falls
back to the old-system hint data, untouched.

Feeds the 4-step hint ladder (web/models.py hooks):
  step 1 Definition     <- the definition piece(s)
  step 2 Wordplay type  <- wfw_solve.operation; a frozen MANUAL solve stores the
                           useless operation 'manual', so its label is derived from
                           the indicator notes the human assigned in the hand-solver
  step 3 Explanation    <- a ONE-LINE mechanical summary assembled from the pieces
                           ("RAG reversed + LAND -> GARLAND"); the full breakdown
                           overlay is phase 2
  step 4 Answer         <- unchanged (clues.answer)

Per-request memoised in flask.g — the puzzle page probes every clue.
"""

import sqlite3

from flask import g

from web.db import get_db

# Display vocabulary for wfw operations (data copied from the admin renderer's
# _TYPE_LABEL in core/wfw_render.py — keep in sync by hand; no core import).
_OP_LABEL = {
    "hidden": "Hidden word",
    "hidden_reversed": "Hidden word (reversed)",
    "dd": "Double definition",
    "double_definition": "Double definition",
    "cd": "Cryptic definition",
    "andlit": "All-in-one (&lit)",
    "charade": "Charade",
    "container": "Container",
    "anagram": "Anagram",
    "reversal": "Reversal",
    "deletion": "Deletion",
    "homophone": "Homophone",
    "acrostic": "Acrostic",
    "alternation": "Alternation",
    "palindrome": "Palindrome",
    "spoonerism": "Spoonerism",
}

# Mechanism words recognised inside compound engine operation names
# (anagram_charade, container_inner_deletion, ...) and inside the indicator notes
# of manual solves ("container indicator", "deletion/tail indicator", ...).
_MECH_WORDS = ("anagram", "hidden", "container", "reversal", "deletion",
               "charade", "homophone", "alternation", "selection",
               "palindrome", "spoonerism", "acrostic", "replacement",
               "cycling", "substitution")


def _load(clue_id):
    """The clue's wfw parse as a plain dict, or None (no row / not a pass /
    tables absent). Memoised per request."""
    cache = getattr(g, "_wfw_cache", None)
    if cache is None:
        cache = g._wfw_cache = {}
    if clue_id in cache:
        return cache[clue_id]
    parse = None
    try:
        db = get_db()
        row = db.execute(
            "SELECT operation, solved_by, status, answer_text "
            "FROM wfw_solve WHERE clue_id = ?", (clue_id,)).fetchone()
        if row is not None and row["status"] == "pass":
            pieces = db.execute(
                "SELECT role, ord, text, value, mechanism, note FROM wfw_piece "
                "WHERE clue_id = ? ORDER BY ord", (clue_id,)).fetchall()
            links = db.execute(
                "SELECT answer_pos, source_index, transform FROM wfw_link "
                "WHERE clue_id = ? ORDER BY answer_pos", (clue_id,)).fetchall()
            parse = {
                "operation": row["operation"] or "",
                "solved_by": row["solved_by"] or "",
                "answer_text": row["answer_text"] or "",
                "definitions": [p for p in map(dict, pieces)
                                if p["role"] == "definition"],
                "sources": sorted((p for p in map(dict, pieces)
                                   if p["role"] == "source"),
                                  key=lambda p: p["ord"]),
                "indicators": [p for p in map(dict, pieces)
                               if p["role"] == "indicator"],
                "links": [dict(l) for l in links],
            }
    except sqlite3.OperationalError:
        parse = None            # wfw tables absent in this DB — old behaviour
    cache[clue_id] = parse
    return parse


def has_wfw_pass(clue_id):
    """True when this clue has a human-checked / engine-verified WFW pass to serve."""
    return _load(clue_id) is not None


def wfw_hint(clue_id, step_type):
    """Hint content for 'definition' | 'wordplay_type' | 'explanation', or None."""
    parse = _load(clue_id)
    if parse is None:
        return None
    if step_type == "definition":
        return _definition(parse)
    if step_type == "wordplay_type":
        return _wordplay_label(parse)
    if step_type == "explanation":
        return _summary(parse)
    return None


# ---------------------------------------------------------------------------
# step 1 — definition
# ---------------------------------------------------------------------------

def _definition(parse):
    texts = [d["text"] for d in parse["definitions"] if (d["text"] or "").strip()]
    # A double definition may store the second definition as a source piece.
    texts += [s["text"] for s in parse["sources"]
              if s["mechanism"] == "definition" and (s["text"] or "").strip()
              and s["text"] not in texts]
    return " / ".join(texts) or None


# ---------------------------------------------------------------------------
# step 2 — wordplay-type label
# ---------------------------------------------------------------------------

def _wordplay_label(parse):
    op = parse["operation"]
    if op == "manual":
        return _manual_label(parse)
    if op in _OP_LABEL:
        return _OP_LABEL[op]
    # Compound engine name: keep the mechanism words, drop shape modifiers
    # (charade_multi_deletion -> "Charade + deletion").
    words = [w for w in op.split("_") if w in _MECH_WORDS]
    if words:
        seen = []
        for w in words:
            if w not in seen:
                seen.append(w)
        return (" + ".join(seen)).capitalize()
    return op.replace("_", " ").capitalize() if op else None


def _manual_label(parse):
    """Derive the clue type of a frozen manual solve from its indicator notes —
    the same information the solver's badge would carry had an engine solved it."""
    found = []
    for ind in parse["indicators"]:
        note = (ind["note"] or "").lower()
        if "definition by example" in note or "positional" in note:
            continue            # definition marker / charade glue, not the clue type
        for w in _MECH_WORDS:
            if w in note and w not in found:
                found.append(w)
    if found:
        return (" + ".join(found)).capitalize()
    n = len([s for s in parse["sources"] if s["mechanism"] != "definition"])
    if n >= 2:
        return "Charade"
    if n == 1:
        m = parse["sources"][0]["mechanism"]
        return {"synonym": "Synonym", "abbreviation": "Abbreviation",
                "hidden": "Hidden word"}.get(m, "Word building")
    return "Word building"


# ---------------------------------------------------------------------------
# step 3 — one-line mechanical summary
# ---------------------------------------------------------------------------

def _summary(parse):
    op = parse["operation"]
    answer = parse["answer_text"].upper()
    if op == "cd":
        return "Cryptic definition — the whole clue is a playful definition of %s." % answer
    if op in ("dd", "double_definition"):
        d = _definition(parse)
        if d:
            return "Double definition — %s each define %s." % (
                " and ".join('"%s"' % t for t in d.split(" / ")), answer)
        return "Double definition — two meanings of %s." % answer

    segments = _segments(parse)
    if not segments:
        return None
    line = " + ".join(segments) + " → " + answer
    if op == "andlit":
        line += " (and the whole clue defines it — &lit)"
    return line


def _segments(parse):
    """Ordered descriptor per contiguous letters-run of the answer, merged so a
    simple linear assembly reads piece by piece and a single split source reads
    'OUTER around INNER'."""
    letters = "".join(c for c in parse["answer_text"].upper() if c.isalpha())
    srcs = {s["ord"]: s for s in parse["sources"]}

    # contiguous runs of answer positions by source_index, in answer order
    runs = []           # [si, placed_letters, transforms]
    for l in parse["links"]:
        si, pos, tr = l["source_index"], l["answer_pos"], l["transform"]
        ch = letters[pos - 1] if 0 < pos <= len(letters) else ""
        if runs and runs[-1][0] == si:
            runs[-1][1] += ch
            if tr and tr not in runs[-1][2]:
                runs[-1][2].append(tr)
        else:
            runs.append([si, ch, [tr] if tr else []])

    # group consecutive anagram-fodder runs into one "anagram of ..." segment
    merged, i = [], 0
    while i < len(runs):
        si, placed, trs = runs[i]
        s = srcs.get(si)
        is_ana = (s is not None and (s["mechanism"] == "anagram_fodder"
                                     or "anagram_of" in trs))
        if is_ana:
            group_sis, all_placed = set(), ""
            while i < len(runs):
                s2 = srcs.get(runs[i][0])
                if s2 is None or not (s2["mechanism"] == "anagram_fodder"
                                      or "anagram_of" in runs[i][2]):
                    break
                group_sis.add(runs[i][0])
                all_placed += runs[i][1]
                i += 1
            # fodder texts in CLUE order (source ord), not answer-letter order
            texts = [srcs[si]["text"] for si in sorted(group_sis)
                     if srcs[si]["text"]]
            merged.append(("ana", texts, all_placed))
        else:
            merged.append(("src", si, placed, trs))
            i += 1

    # ALL letters each source places across the answer (a split container source
    # places two runs) — the honest basis for the deletion/reversal comparison,
    # so a split source is never misread as a deletion.
    placed_all = {}
    for si, placed, _ in runs:
        placed_all[si] = placed_all.get(si, "") + placed

    # a single source split into two runs with exactly one segment between them
    # reads as a container: OUTER around INNER
    if (len(merged) == 3 and merged[0][0] == "src" and merged[2][0] == "src"
            and merged[0][1] == merged[2][1]):
        outer = _describe(srcs.get(merged[0][1]),
                          placed_all[merged[0][1]], merged[0][3])
        inner = (("anagram of " + " ".join('"%s"' % t for t in merged[1][1]))
                 if merged[1][0] == "ana"
                 else _describe(srcs.get(merged[1][1]),
                                placed_all.get(merged[1][1], merged[1][2]),
                                merged[1][3]))
        return [outer + " around " + inner] if outer and inner else None

    out, described, ana_done = [], set(), set()
    for m in merged:
        if m[0] == "ana":
            sis = frozenset(t for t in m[1])
            if sis and sis <= ana_done:   # split anagram group seen again: letters
                out.append(m[2])
                continue
            ana_done |= sis
            out.append("anagram of " + " ".join('"%s"' % t for t in m[1]))
            continue
        si, placed, trs = m[1], m[2], m[3]
        s = srcs.get(si)
        if si in described:     # a split source seen again: just its letters
            out.append(placed)
            continue
        described.add(si)
        d = _describe(s, placed_all.get(si, placed), trs)
        if d is None:
            return None
        out.append(d)
    return out


def _describe(s, placed, transforms):
    """One piece as 'text→VALUE [reversed] [less X]' — mechanical, no prose."""
    if s is None:
        return None
    text = (s["text"] or "").strip()
    value = (s["value"] or "").strip().upper()
    mech = s["mechanism"] or ""
    reversed_ = "reversed" in (transforms or [])

    if mech == "hidden":
        return "hidden in \"%s\"" % text
    if mech == "replacement_letter":
        return "%s (new letter, unclued)" % (value or placed)
    if mech == "definition":
        return "\"%s\"" % text

    if not value:
        value = placed
    # A manual solve stores no transform — but a piece whose placed letters are
    # exactly its value reversed IS a reversal, knowable from the letters alone.
    if (not reversed_ and len(value) > 1 and placed
            and placed == value[::-1] and placed != value):
        reversed_ = True
    base = ("%s→%s" % (text, value)) if text and value != text.upper() \
        else (value or text)
    if reversed_:
        base += " reversed"
    # deletion shown honestly: the placed letters are the value minus something
    cmp_placed = placed[::-1] if reversed_ else placed
    if value and cmp_placed and cmp_placed != value:
        missing = _removed(value, cmp_placed)
        if missing:
            base += " less %s" % missing
    return base


def _removed(value, placed):
    """Letters of `value` not used by `placed` (in value order), when placed is an
    in-order sub-selection of value; else None (no claim made)."""
    out, j = [], 0
    for ch in value:
        if j < len(placed) and placed[j] == ch:
            j += 1
        else:
            out.append(ch)
    if j != len(placed) or not out:
        return None
    return "".join(out)


# ---------------------------------------------------------------------------
# Full breakdown (phase 2) — everything the overlay renders
# ---------------------------------------------------------------------------

# One stable (text, fill) colour per source index — copied from the admin
# renderer's palette (core/wfw_render.py) so both surfaces read alike.
PALETTE = [
    ("#1d6fb8", "#e3f0fb"), ("#2e8b57", "#e4f4ea"), ("#d2691e", "#fbeadf"),
    ("#c2185b", "#fbe4ee"), ("#0097a7", "#e0f5f7"), ("#b8860b", "#f8efd6"),
    ("#c62828", "#fbe4e4"), ("#6a1b9a", "#f0e4f7"),
]
ROLE_COLOURS = {                       # (text, fill) for the non-source roles
    "definition": ("#166534", "#dcfce7"),
    "indicator": ("#92400e", "#fef3c7"),
    "link": ("#475569", "#e2e8f0"),
}

# Friendly pill label per piece mechanism (data copied from the admin renderer's
# _MECH_LABEL — keep in sync by hand; no core import).
_MECH_LABEL = {
    "hidden": "Hidden in", "hidden_reversed": "Hidden in (rev.)",
    "synonym": "Synonym", "abbreviation": "Substitution", "raw": "Literal",
    "letters": "Letters", "replacement_letter": "New letter",
    "first_letter": "Initial", "last_letter": "Last letter",
    "outer": "Outer letters", "homophone": "Sounds like",
    "anagram_fodder": "Anagram of", "alternate": "Alternate letters",
    "selection": "Letters from", "deletion": "Deletion",
    "definition": "Definition",
}


def source_colour(i):
    return PALETTE[i % len(PALETTE)]


def load_breakdown(clue_id):
    """Everything the full-explanation overlay renders, as one plain dict —
    or None when the clue has no pass parse (or no stored atoms).

    {operation_label, summary, clue_tokens: [{text, role, source_index}],
     answer_tiles: [{char, source_index}|{sep}], rows: [{pill, fg, fill, detail}]}
    """
    import json

    parse = _load(clue_id)
    if parse is None:
        return None
    try:
        db = get_db()
        row = db.execute("SELECT atoms FROM wfw_solve WHERE clue_id = ?",
                         (clue_id,)).fetchone()
        pieces = db.execute(
            "SELECT role, ord, text, value, mechanism, note, atom_ids "
            "FROM wfw_piece WHERE clue_id = ? ORDER BY ord", (clue_id,)).fetchall()
    except sqlite3.OperationalError:
        return None
    if row is None or not row["atoms"]:
        return None
    try:
        atoms = json.loads(row["atoms"])
    except ValueError:
        return None

    # atom -> (role, source_index); precedence source > definition > indicator >
    # link (an &lit uses the same words twice — the wordplay colour wins).
    atom_role = {}
    for want in ("source", "definition", "indicator", "link"):
        for p in pieces:
            if p["role"] != want:
                continue
            try:
                aids = json.loads(p["atom_ids"] or "[]")
            except ValueError:
                aids = []
            for aid in aids:
                atom_role.setdefault(aid, (p["role"], p["ord"]))

    clue_tokens = []
    for t in atoms.get("clue_tokens", []):
        role, si = None, None
        for aid in t.get("atom_ids", []):
            if aid in atom_role:
                role, si = atom_role[aid]
                break
        clue_tokens.append({"text": t.get("text", ""),
                            "kind": t.get("kind", "word"),
                            "role": role, "source_index": si})

    # answer tiles: letters coloured by the source that placed them
    letters = "".join(c for c in parse["answer_text"].upper() if c.isalpha())
    pos_src = {l["answer_pos"]: l["source_index"] for l in parse["links"]}
    answer_tiles, pos = [], 0
    for ch in parse["answer_text"].upper():
        if ch.isalpha():
            pos += 1
            answer_tiles.append({"char": ch, "source_index": pos_src.get(pos)})
        else:
            answer_tiles.append({"sep": ch})

    # per-source placed letters + transforms (for the honest detail line)
    placed_all, trans = {}, {}
    for l in parse["links"]:
        si, p, tr = l["source_index"], l["answer_pos"], l["transform"]
        if 0 < p <= len(letters):
            placed_all[si] = placed_all.get(si, "") + letters[p - 1]
        if tr:
            trans.setdefault(si, []).append(tr)

    rows = []
    for d in parse["definitions"]:
        fg, fill = ROLE_COLOURS["definition"]
        rows.append({"pill": "Definition", "fg": fg, "fill": fill,
                     "detail": '"%s" → %s' % (d["text"], parse["answer_text"].upper())})
    for s in parse["sources"]:
        if s["mechanism"] == "definition":     # second definition of a DD
            fg, fill = ROLE_COLOURS["definition"]
            rows.append({"pill": "Definition", "fg": fg, "fill": fill,
                         "detail": '"%s" → %s'
                                   % (s["text"], parse["answer_text"].upper())})
            continue
        fg, fill = source_colour(s["ord"])
        detail = _describe(s, placed_all.get(s["ord"], ""), trans.get(s["ord"], []))
        rows.append({"pill": _MECH_LABEL.get(s["mechanism"],
                                             (s["mechanism"] or "Piece").title()),
                     "fg": fg, "fill": fill, "detail": detail or ""})
    for ind in parse["indicators"]:
        fg, fill = ROLE_COLOURS["indicator"]
        rows.append({"pill": "Indicator", "fg": fg, "fill": fill,
                     "detail": '"%s"%s' % (ind["text"],
                                           (" — " + ind["note"]) if ind["note"] else "")})
    for p in pieces:
        if p["role"] == "link":
            fg, fill = ROLE_COLOURS["link"]
            rows.append({"pill": "Link", "fg": fg, "fill": fill,
                         "detail": '"%s"' % p["text"]})

    n_src = max([s["ord"] for s in parse["sources"]], default=-1) + 1
    return {
        "operation_label": _wordplay_label(parse),
        "summary": _summary(parse),
        "clue_tokens": clue_tokens,
        "answer_tiles": answer_tiles,
        "rows": rows,
        "src_fg": {i: source_colour(i)[0] for i in range(n_src)},
        "src_fill": {i: source_colour(i)[1] for i in range(n_src)},
    }
