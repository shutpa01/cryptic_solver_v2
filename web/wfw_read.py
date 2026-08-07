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

import json
import re
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

# Canonical reading order for a composite clue type — outer operation first,
# inner transforms last. We name EVERY mechanism a clue uses (the full clue-type
# is a differentiator), so the order has to be deterministic. Mirrors
# core/wfw_render._MECH_ORDER — keep in sync by hand.
_MECH_ORDER = ("container", "charade", "anagram", "reversal", "deletion",
               "selection", "hidden", "acrostic", "alternation", "homophone",
               "spoonerism", "palindrome", "cycling", "letter_shift",
               "substitution", "replacement")

_UNPLACED = 10 ** 9   # clue position for a piece with no locatable atoms — sorts last


def _clue_pos(atom_ids_json):
    """The piece's position in the clue's READING order: the smallest clue
    character-index across its atom_ids. atom_ids look like 'clue_char_0015',
    so the numeric suffix is the character offset. Every explanation is ordered
    by this so it reads left-to-right like the clue; the colour coding (clue
    words + answer tiles) carries which piece lands where in the answer."""
    try:
        ids = json.loads(atom_ids_json or "[]")
    except (ValueError, TypeError):
        return _UNPLACED
    nums = [int(m.group(1)) for a in ids
            for m in [re.search(r"(\d+)", a or "")] if m]
    return min(nums) if nums else _UNPLACED


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
                "SELECT role, ord, text, value, mechanism, note, atom_ids "
                "FROM wfw_piece WHERE clue_id = ? ORDER BY ord",
                (clue_id,)).fetchall()
            links = db.execute(
                "SELECT answer_pos, source_index, transform FROM wfw_link "
                "WHERE clue_id = ? ORDER BY answer_pos", (clue_id,)).fetchall()
            parse = {
                "operation": row["operation"] or "",
                "solved_by": row["solved_by"] or "",
                "answer_text": row["answer_text"] or "",
                "definitions": [p for p in map(dict, pieces)
                                if p["role"] == "definition"],
                "sources": sorted(
                    ({**p, "clue_pos": _clue_pos(p.get("atom_ids"))}
                     for p in map(dict, pieces) if p["role"] == "source"),
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

# Ops whose curated label is authoritative and must NOT be enriched from
# notes/pieces: they either have no enumerable wordplay (double def / cryptic
# def / continuation) or the enrichment would drop a defining designation
# (&lit; a hidden word is never a charade). Mirrors core/wfw_render._ATOMIC_OPS.
_ATOMIC_OPS = frozenset((
    "dd", "double_definition", "cd", "andlit", "continuation",
    "hidden", "hidden_reversed"))


def _note_mech(note):
    """The single mechanism an indicator note denotes, or None to skip. Handles the
    note-vocabulary variants that don't spell the mechanism verbatim: insertion is
    container from the other end; 'first-letter indicator' / 'last-letter indicator'
    are selections; 'deleted letters' is deletion. 'positional' / 'definition by
    example' / 'surface' notes are glue or markers, not clue-type mechanisms.
    Mirrors core/wfw_render._note_mech."""
    n = (note or "").lower()
    if (not n or "definition by example" in n or "positional" in n
            or "surface" in n or n == "wordplay indicator"):
        return None
    if "insertion" in n or "container" in n:
        return "container"
    if ("selection" in n or "first-letter" in n or "last-letter" in n
            or "middle-letter" in n or "outer-letter" in n):
        return "selection"
    if "acrostic" in n:
        return "acrostic"
    if "letter_shift" in n:
        return "letter_shift"
    if "deletion" in n or "deleted" in n:
        return "deletion"
    for w in _MECH_WORDS:
        if w in n:
            return w
    return None


def _note_mechs(indicators):
    """The mechanism set named by a solve's indicator notes, plus the count of
    container/insertion joins. Charade is NOT included here — it carries no
    indicator; the caller adds it from the join count."""
    found = set()
    container_joins = 0
    for ind in indicators:
        m = _note_mech(ind["note"])
        if m is None:
            continue
        found.add(m)
        if m == "container":
            container_joins += 1
    return found, container_joins


# Ops whose extra sources are NOT charade pieces, so a charade must NOT be inferred
# when one is present (a false mechanism is worse than an incomplete one):
#  - gather ops assemble several source WORDS into one gestalt (anagram of a phrase,
#    acrostic of consecutive words, alternate letters of a run, a spoonerism /
#    homophone of a two-word phrase);
#  - substitution / replacement swap one source's letters into another, they don't
#    sit side by side.
# Mirrors core/wfw_render._CHARADE_SUPPRESS.
_CHARADE_SUPPRESS = frozenset((
    "anagram", "acrostic", "alternation", "spoonerism", "homophone",
    "hidden", "palindrome", "cycling", "substitution", "replacement"))


def _has_charade(placed_pieces, container_joins, mechs):
    """Charade (side-by-side concatenation) carries no indicator, so we infer it by
    counting joins: PLACED pieces (sources that actually contribute answer letters —
    a deleted/removed source places nothing) need placed-1 binary joins, and each
    container nesting is one join, so any leftover join is a charade. Suppressed when
    a _CHARADE_SUPPRESS op is present (that op accounts for the extra sources)."""
    if mechs & _CHARADE_SUPPRESS:
        return False
    return placed_pieces - 1 > container_joins


def _order_mechs(found):
    """Canonical outer→inner ordering of a mechanism set into 'Container + charade
    + selection'. Mirrors core/wfw_render._order_mechs."""
    ordered = [m for m in _MECH_ORDER if m in found]
    ordered += [m for m in found if m not in _MECH_ORDER]  # any stray note word
    return (" + ".join(ordered)).replace("_", " ").capitalize()


def _wordplay_label(parse):
    op = parse["operation"]
    if op == "manual":
        return _manual_label(parse)
    if op in _ATOMIC_OPS:
        return _OP_LABEL.get(op) or op.replace("_", " ").capitalize()
    # Engine solves carry the same rich indicator notes as manual solves, and the
    # notes are richer than the op name (op='container' can hide an inner acrostic
    # + a charade). Read the mechanisms from notes+pieces and, if that reveals MORE
    # than the op name already names, build the full clue-type; otherwise keep the
    # curated op-name label (nicer phrasing for single-mechanism clues).
    op_mechs = set(w for w in op.split("_") if w in _MECH_WORDS)
    if "insertion" in op_mechs:
        op_mechs.discard("insertion"); op_mechs.add("container")
    placed = len({l["source_index"] for l in parse["links"]})
    found, container_joins = _note_mechs(parse["indicators"])
    if "container" in op_mechs:
        container_joins = max(container_joins, 1)   # op name declares the nesting
    allm = op_mechs | found
    if _has_charade(placed, container_joins, allm):
        allm.add("charade")
    # A composite (2+ mechanisms) always renders in canonical order; a single
    # mechanism that discovered something new does too. Only a single-mechanism op
    # with nothing new keeps its curated (nicer) label. This keeps the badge and
    # hint identical regardless of which curated compound entries each map happens
    # to carry.
    if allm and (len(allm) >= 2 or allm != op_mechs):
        return _order_mechs(allm)
    if op in _OP_LABEL:
        return _OP_LABEL[op]
    # Compound engine name with no curated label: keep the mechanism words.
    if op_mechs:
        return _order_mechs(op_mechs)
    return op.replace("_", " ").capitalize() if op else None


def _manual_label(parse):
    """Derive the clue type of a frozen manual solve, naming EVERY mechanism it
    uses — the same information the solver's badge would carry had an engine
    solved it. Mirrors core/wfw_render._manual_type_label."""
    n = len([s for s in parse["sources"] if s["mechanism"] != "definition"])
    placed = len({l["source_index"] for l in parse["links"]})
    found, container_joins = _note_mechs(parse["indicators"])
    if _has_charade(placed, container_joins, found):
        found.add("charade")
    if found:
        return _order_mechs(found)
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

# A letter-shift re-orders letters WITHIN a piece rather than contributing its own
# letters, so it never surfaces as a segment in the assembly line — its direction has to
# be named separately or the one-liner reads as a plain charade. Label-only (no solving
# duty); mirrors core/wfw_render's letter-shift detail.
_LETTER_SHIFT_DETAIL = {
    "last_front": "move last letter to front",
    "first_end": "move first letter to end",
    "move_left": "move letter left",
    "move_right": "move letter right",
}


def _letter_shift_note(indicators):
    """The readable direction of a letter-shift indicator, or None if the clue has none."""
    for ind in indicators:
        n = (ind["note"] or "").lower()
        if (n.startswith("letter_shift") or n.startswith("letter-shift")
                or n.startswith("letter shift")):
            sub = n.split("/", 1)[1].replace("indicator", "").strip() if "/" in n else ""
            return _LETTER_SHIFT_DETAIL.get(sub, sub or "letter shift")
    return None


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
    # order pieces by clue reading position (not answer-assembly order)
    segments = sorted(segments, key=lambda ps: ps[0])
    line = " + ".join(seg for _, seg in segments) + " → " + answer
    ls = _letter_shift_note(parse["indicators"])
    if ls:
        line += " (%s)" % ls
    if op == "andlit":
        line += " (and the whole clue defines it — &lit)"
    return line


def _segments(parse):
    """(clue_pos, descriptor) per contiguous letters-run of the answer, merged so
    a simple linear assembly reads piece by piece and a single split source reads
    'OUTER around INNER'. The caller orders by clue_pos so the line reads in the
    clue's own word order; colour coding (clue words + answer tiles) shows where
    each piece lands in the answer."""
    letters = "".join(c for c in parse["answer_text"].upper() if c.isalpha())
    srcs = {s["ord"]: s for s in parse["sources"]}
    _found, _ = _note_mechs(parse["indicators"])
    has_ana, has_rev = "anagram" in _found, "reversal" in _found

    def cpos(si):
        s = srcs.get(si)
        return s.get("clue_pos", _UNPLACED) if s else _UNPLACED

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
            gpos = min((cpos(si) for si in group_sis), default=_UNPLACED)
            merged.append(("ana", texts, all_placed, gpos))
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
                          placed_all[merged[0][1]], merged[0][3], has_ana, has_rev)
        if merged[1][0] == "ana":
            inner = "anagram of " + " ".join('"%s"' % t for t in merged[1][1])
            inner_pos = merged[1][3]
        else:
            inner = _describe(srcs.get(merged[1][1]),
                              placed_all.get(merged[1][1], merged[1][2]),
                              merged[1][3], has_ana, has_rev)
            inner_pos = cpos(merged[1][1])
        if outer and inner:
            return [(min(cpos(merged[0][1]), inner_pos),
                     outer + " around " + inner)]
        return None

    out, described, ana_done = [], set(), set()
    for m in merged:
        if m[0] == "ana":
            sis = frozenset(t for t in m[1])
            if sis and sis <= ana_done:   # split anagram group seen again: letters
                out.append((m[3], m[2]))
                continue
            ana_done |= sis
            out.append((m[3], "anagram of " + " ".join('"%s"' % t for t in m[1])))
            continue
        si, placed, trs = m[1], m[2], m[3]
        s = srcs.get(si)
        if si in described:     # a split source seen again: just its letters
            out.append((cpos(si), placed))
            continue
        described.add(si)
        d = _describe(s, placed_all.get(si, placed), trs, has_ana, has_rev)
        if d is None:
            return None
        out.append((cpos(si), d))
    return out


def _anagram_desc(value, placed, has_ana, has_rev):
    """'anagram [less X]' when the clue names an anagram indicator and `placed` (the piece's
    answer letters, in answer order) is a RE-ORDERING of `value` — a sub-multiset of it whose
    order is not preserved. Lets a source roled selection/synonym still read as anagram fodder
    so the hint line agrees with the card. Mirrors core/wfw_render._anagram_note. An in-order
    survivor is a plain deletion (shown as 'less X' below); a real reversal keeps priority."""
    if not has_ana or not value or not placed or placed == value:
        return None
    from collections import Counter
    if Counter(placed) - Counter(value):                 # placed uses letters value lacks
        return None
    it = iter(value)
    if all(ch in it for ch in placed):                   # in-order survivor => a deletion
        return None
    if has_rev and len(value) > 1 and placed == value[::-1]:
        return None                                      # a real reversal keeps priority
    removed = "".join(sorted((Counter(value) - Counter(placed)).elements()))
    return "anagram less %s" % removed if removed else "anagram"


def _describe(s, placed, transforms, has_ana=False, has_rev=False):
    """One piece as 'text→VALUE [anagram|reversed] [less X]' — mechanical, no prose."""
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
    # an anagram indicator governs a re-ordered piece: show 'anagram' (not the reversal/
    # deletion the letters would otherwise be read as), matching the card.
    ana = _anagram_desc(value, placed, has_ana, has_rev)
    if ana:
        base = ("%s→%s" % (text, value)) if text and value != text.upper() else (value or text)
        return "%s %s" % (base, ana)
    # A manual solve stores no transform — but a piece whose placed letters are
    # exactly its value reversed IS a reversal, knowable from the letters alone.
    if (not reversed_ and len(value) > 1 and placed
            and placed == value[::-1] and placed != value):
        reversed_ = True
    # Reversal COMBINED with a deletion: the placed letters are NOT a forward
    # sub-selection of the value, but the REVERSED placed letters ARE (value
    # reversed then trimmed — e.g. LEMON reversed, less N -> OMEL). Without this
    # the summary detects neither the reversal nor the deletion and silently maps
    # the 5-letter value onto 4 tiles, dropping the removed letter. (clue 10081158
    # OMELETTE: fruit=LEMON reversed less N.)
    if (not reversed_ and len(value) > 1 and placed and placed != value
            and _removed(value, placed) is None
            and _removed(value, placed[::-1]) is not None):
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
# Definition / indicator / link render PLAIN (neutral grey pill). Only the
# source pieces carry colour, so colour maps directly to the answer tiles —
# nothing that doesn't build answer letters is colour-coded.
_PLAIN_ROLE = ("#475569", "#f1f5f9")
ROLE_COLOURS = {
    "definition": _PLAIN_ROLE,
    "indicator": _PLAIN_ROLE,
    "link": _PLAIN_ROLE,
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

    # EVERY row (definition / source / indicator / link) carries its clue
    # position and the whole list is sorted by it, so the breakdown reads
    # top-to-bottom in the clue's own word order — not grouped by role (which
    # buried the definition at the top and the first-word indicator at the
    # bottom). Matches the inline card renderer (core/wfw_render).
    ans_up = parse["answer_text"].upper()
    scored = []   # (clue_pos, row)
    for d in parse["definitions"]:
        fg, fill = ROLE_COLOURS["definition"]
        scored.append((_clue_pos(d.get("atom_ids")),
                       {"pill": "Definition", "fg": fg, "fill": fill,
                        "detail": '"%s" → %s' % (d["text"], ans_up)}))
    for s in parse["sources"]:
        pos = s.get("clue_pos", _UNPLACED)
        if s["mechanism"] == "definition":     # second definition of a DD
            fg, fill = ROLE_COLOURS["definition"]
            scored.append((pos, {"pill": "Definition", "fg": fg, "fill": fill,
                                 "detail": '"%s" → %s' % (s["text"], ans_up)}))
            continue
        fg, fill = source_colour(s["ord"])
        detail = _describe(s, placed_all.get(s["ord"], ""), trans.get(s["ord"], []))
        scored.append((pos, {"pill": _MECH_LABEL.get(s["mechanism"],
                                                     (s["mechanism"] or "Piece").title()),
                             "fg": fg, "fill": fill, "detail": detail or ""}))
    for ind in parse["indicators"]:
        fg, fill = ROLE_COLOURS["indicator"]
        scored.append((_clue_pos(ind.get("atom_ids")),
                       {"pill": "Indicator", "fg": fg, "fill": fill,
                        "detail": '"%s"%s' % (ind["text"],
                                              (" — " + ind["note"]) if ind["note"] else "")}))
    for p in pieces:
        if p["role"] == "link":
            fg, fill = ROLE_COLOURS["link"]
            scored.append((_clue_pos(p["atom_ids"]),
                           {"pill": "Link", "fg": fg, "fill": fill,
                            "detail": '"%s"' % p["text"]}))
    rows = [r for _, r in sorted(scored, key=lambda x: x[0])]

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
