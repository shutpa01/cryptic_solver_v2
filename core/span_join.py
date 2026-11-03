"""Split-enumeration (spanning) answers — a read-only helper lens (2026-07-12).

Some puzzles put ONE answer across TWO grid entries. The scraper stores that as:
  - the PRIMARY clue: real wordplay text, the FULL enumeration, but only its own
    entry's letters as the answer (telegraph 31289 3d: "Wild Ryder Cup row...",
    enumeration "5,6", answer CURRY),
  - a STUB clue on the partner entry: text "See 3 Down", empty enumeration,
    the remaining letters (5a: POWDER).

Both then fail the cascade: the primary's wordplay builds the whole phrase
(CURRY POWDER), and the stub has no wordplay at all. Wordplay pieces can even
cross the entry boundary (DINING ROOM = DIN + IN + GROOM — the G is the last
tile of the first entry), so the fix must present the JOINED answer, not the
per-entry one.

This module DETECTS the pair mechanically and returns the joined answer:
  - detection needs no AI and no new data: a primary is a clue whose
    enumeration counts MORE letters than its stored answer has, and whose
    puzzle contains a "See <n> <direction>" stub pointing back at it whose
    letters complete the count exactly;
  - the joined answer is primary letters + stub letters (the phrase reads from
    the primary's entry into the continuation entry — that is why the setter
    put the wordplay there).

Used by core/wfw_web._load_clue (one choke point: /hs tiles, Resolve, commit,
re-run all see the joined answer) and scripts/nightly_cascade.py (solve the
primary against the full phrase; file a status='continuation' marker on the
stub so it stops counting as a fail).

Never modifies clue rows or any engine. The ONLY write in this module is
mark_continuation(), which files the stub's wfw_solve marker row (and never
overwrites an existing row).
"""

import json
import re

# "See 3 Down" / "See 27 Across" / "See 3" (direction omitted); tolerant of a
# trailing full stop. Anything longer is a real clue that merely mentions a
# cross-reference, and must NOT match.
_STUB_RE = re.compile(r"^\s*see\s+(\d+)(?:\s+(across|down))?\s*\.?\s*$",
                      re.IGNORECASE)


def _letters(s):
    return "".join(c for c in (s or "").upper() if c.isalpha())


def _enum_total(enumeration):
    """Total letters the enumeration promises (0 when there is none)."""
    return sum(int(n) for n in re.findall(r"\d+", enumeration or ""))


def stub_ref(clue_text):
    """(clue_number:str, direction:str|None) when the text is a pure
    "See <n> [<direction>]" stub, else None."""
    m = _STUB_RE.match(clue_text or "")
    if not m:
        return None
    return m.group(1), (m.group(2) or "").lower() or None


def _find_stub(conn, source, puzzle_number, clue_number, direction, missing):
    """The stub clue pointing at (clue_number, direction) whose answer supplies
    exactly `missing` letters. Returns (stub_id, stub_letters) or None."""
    rows = conn.execute(
        "SELECT id, clue_text, answer, direction FROM clues "
        "WHERE source = ? AND puzzle_number = ? AND lower(clue_text) LIKE 'see %'",
        (source, puzzle_number)).fetchall()
    hits = []
    for sid, text, answer, _sdir in rows:
        ref = stub_ref(text)
        if ref is None:
            continue
        num, rdir = ref
        if num != str(clue_number):
            continue
        if rdir is not None and rdir != (direction or "").lower():
            continue
        letters = _letters(answer)
        if letters and len(letters) == missing:
            hits.append((sid, letters))
    # exactly one completing stub, or no claim at all (three-entry chains and
    # ambiguous references are left alone — they stay visible as fails)
    return hits[0] if len(hits) == 1 else None


def primary_join(conn, source, puzzle_number, clue_number, direction,
                 answer, enumeration):
    """When these clue fields describe a split-enumeration PRIMARY, return
    (joined_letters, stub_id); else None. Cheap: exits on the arithmetic check
    before ever querying for a stub."""
    own = _letters(answer)
    total = _enum_total(enumeration)
    if not own or total <= len(own):
        return None                      # normal clue (or no enumeration)
    found = _find_stub(conn, source, puzzle_number, clue_number, direction,
                       missing=total - len(own))
    if found is None:
        return None
    stub_id, stub_letters = found
    return own + stub_letters, stub_id


def classify_rows(conn, rows):
    """Batch lens for the cascade work list. `rows` are
    (id, clue_text, answer, source, pnum, direction, enumeration) tuples.
    Returns {clue_id: ("primary", joined_letters) | ("stub", primary_label)}
    — clues absent from the map are normal."""
    out = {}
    for cid, clue_text, answer, src, pnum, direction, enum in rows:
        joined = primary_join(conn, src, pnum, _clue_number(conn, cid),
                              direction, answer, enum)
        if joined is not None:
            out[cid] = ("primary", joined[0])
            continue
        ref = stub_ref(clue_text)
        if ref is None:
            continue
        num, rdir = ref
        primary = _find_primary(conn, src, pnum, num, rdir, _letters(answer))
        if primary is not None:
            out[cid] = ("stub", primary)
    return out


def _clue_number(conn, clue_id):
    row = conn.execute("SELECT clue_number FROM clues WHERE id = ?",
                       (clue_id,)).fetchone()
    return row[0] if row else None


def mark_continuation(conn, stub_id, clue_text, answer, primary_id,
                      primary_label):
    """File the stub's wfw_solve marker: status='continuation' — not a fail
    (there is nothing to solve) and not a pass (nothing was verified), and the
    distinct status drops the stub out of every fail/pending work list while
    the hint reader (pass-only) keeps its old fallback behaviour. Never
    overwrites an existing row. Returns True when a row was written."""
    have = conn.execute("SELECT 1 FROM wfw_solve WHERE clue_id = ?",
                        (stub_id,)).fetchone()
    if have:
        return False
    conn.execute(
        "INSERT INTO wfw_solve (clue_id, clue_text, answer_text, operation, "
        "solved_by, status, confidence, warnings, atoms) "
        "VALUES (?, ?, ?, 'continuation', 'span_join', 'continuation', "
        "NULL, ?, NULL)",
        (stub_id, clue_text, answer, json.dumps(
            ["Second entry of %s — the whole answer is clued and solved "
             "there (clue %d)." % (primary_label, primary_id)])))
    conn.commit()
    return True


# ---------------------------------------------------------------------------
# A "See N" stub FOLLOWS its parent (user, 2026-11-02): "where a clue uses the
# tiles from another clue, the secondary clue that just refers to the main clue
# should be set to pass when the parent is passed", and Cordelia does not narrate
# or film it.
#
# The stub's row becomes operation='continuation', solved_by='span_join', with NO
# pieces, and status 'pass' while the parent is served, 'continuation' while it
# is not. With no pieces, core/wfw_card.stored_card returns None. The stub then
# has no card or page of its own, so the video (scripts/youtube_capture.clue_list)
# and its narration skip it. But it is a PASS, so the puzzle-level gate counts it
# and the puzzle can go live.
#
# Called from core/store after every save_parse / set_status / delete_parse, so
# every route that passes or demotes a parent is covered without touching any of
# them. It never touches a human's own stub verdict: a frozen or manual stub row,
# or an INVALID one, is left exactly as it is.
# ---------------------------------------------------------------------------

def _stub_parent(conn, source, puzzle_number, num, rdir, stub_answer=None):
    """The clue id a "See <num> [<rdir>]" stub points at, or None when it is
    missing or still ambiguous.

    No direction given and both N across and N down real clues: the parent is the
    one whose answer CONTAINS the stub's answer (guardian 30154 "See 1" BIRDIE ->
    1 across WATCHTHEBIRDIE, not 1 down WHIFFY; left FAIL on 2026-11-03)."""
    params = [source, puzzle_number, str(num)]
    where = "source = ? AND puzzle_number = ? AND clue_number = ?"
    if rdir:
        where += " AND direction = ?"
        params.append(rdir)
    cands = [(cid, ans) for cid, text, ans in conn.execute(
        "SELECT id, clue_text, answer FROM clues WHERE %s" % where, params).fetchall()
        if stub_ref(text) is None]
    if len(cands) > 1 and _letters(stub_answer or ""):
        mine = _letters(stub_answer)
        cands = [c for c in cands if mine in _letters(c[1] or "")]
    return cands[0][0] if len(cands) == 1 else None


def _is_served(conn, clue_id):
    """The site's per-clue serving rule: a pass, or INVALID with a comment."""
    row = conn.execute("SELECT status FROM wfw_solve WHERE clue_id = ?",
                       (clue_id,)).fetchone()
    if not row:
        return False
    if row[0] == "pass":
        return True
    if row[0] == "invalid":
        n = conn.execute("SELECT note FROM wfw_notes WHERE clue_id = ?",
                         (clue_id,)).fetchone()
        return bool(n and (n[0] or "").strip())
    return False


def sync_stubs_for(conn, clue_id):
    """Bring every "See N" stub in this clue's puzzle into line with its parent.
    Returns the number of stub rows written. Commits only when it writes."""
    row = conn.execute("SELECT source, puzzle_number FROM clues WHERE id = ?",
                       (clue_id,)).fetchone()
    if not row:
        return 0
    source, pnum = row
    written = 0
    for sid, text, answer in conn.execute(
            "SELECT id, clue_text, answer FROM clues WHERE source = ? AND "
            "puzzle_number = ? AND lower(clue_text) LIKE 'see %'",
            (source, pnum)).fetchall():
        ref = stub_ref(text)
        if ref is None:
            continue
        parent = _stub_parent(conn, source, pnum, ref[0], ref[1], answer)
        if parent is None:
            continue
        cur = conn.execute("SELECT status, operation, solved_by FROM wfw_solve "
                           "WHERE clue_id = ?", (sid,)).fetchone()
        if cur and (cur[2] == "manual" or cur[0] == "invalid"):
            continue                          # the human's own verdict stands
        if conn.execute("SELECT 1 FROM wfw_frozen WHERE clue_id = ?",
                        (sid,)).fetchone():
            continue
        want = "pass" if _is_served(conn, parent) else "continuation"
        if cur and cur[1] == "continuation" and cur[0] == want:
            continue                          # already in line
        for table in ("wfw_solve", "wfw_piece", "wfw_link"):
            conn.execute("DELETE FROM %s WHERE clue_id = ?" % table, (sid,))
        plabel = conn.execute("SELECT clue_number || ' ' || direction FROM clues "
                              "WHERE id = ?", (parent,)).fetchone()[0]
        conn.execute(
            "INSERT INTO wfw_solve (clue_id, clue_text, answer_text, operation, "
            "solved_by, status, confidence, warnings, atoms) "
            "VALUES (?, ?, ?, 'continuation', 'span_join', ?, NULL, ?, NULL)",
            (sid, text, answer, want, json.dumps(
                ["Continuation of %s (clue %d) — the answer is clued and explained "
                 "there; this entry follows its verdict." % (plabel, parent)])))
        written += 1
    if written:
        conn.commit()
    return written


def _find_primary(conn, source, puzzle_number, clue_number, direction,
                  stub_letters):
    """The primary a stub points at — confirmed by the same arithmetic in the
    other direction. Returns (primary_id, label) or None."""
    params = [source, puzzle_number, str(clue_number)]
    where = "source = ? AND puzzle_number = ? AND clue_number = ?"
    if direction:
        where += " AND direction = ?"
        params.append(direction)
    for pid, answer, enum, pdir in conn.execute(
            "SELECT id, answer, enumeration, direction FROM clues WHERE %s"
            % where, params).fetchall():
        own = _letters(answer)
        if own and stub_letters and \
                _enum_total(enum) == len(own) + len(stub_letters):
            return pid, "%s %s" % (clue_number, pdir)
    return None
