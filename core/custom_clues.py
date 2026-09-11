"""Custom clues — clues written by other people that Cordelia answers.

The user answers clue-writing threads on Reddit (2026-09-10). Those clues have
no publication, so they are filed under a source of their own, `custom`, which
is invisible to the public site by construction: `custom` is in neither
`web.serving.SERVED_SOURCES` nor `SERVED_BROWSE`, so the Custom section, its
clues and any page under it are admin-only without a single new gate.

Nothing downstream changes. Each pasted clue is filed as its OWN one-clue
puzzle (puzzle_number allocated in sequence from 1), which is what lets the
existing per-puzzle machinery run on it untouched — INCLUDING the solving
tools, because a clue from a thread arrives without its answer (user,
2026-09-10: "I will not normally have the answer"). The one-clue puzzle page
is the whole workspace and it is the live puzzle page, not a copy of it:

    /custom/clues/N            the clue's own page — Solve mode, the answer
                               box, and the Tools overlay (anagram, pattern,
                               similar); "Save all to DB" writes the answer
                               through /admin/save-all-answers, the same route
                               a prize puzzle uses. No grid, and none needed.
    scripts/nightly_cascade.py --source custom --pnum N     (engines first)
    scripts/run_prefill.py     --source custom --pnum N     (Claude reading)
    /solver/?id=<clue_id>                                   (Confirm)
    /solver/hs?id=<clue_id>                                 (hand-solver)

The puzzle_number is plumbing and is never presented as one: the Custom section
is a flat list of clues, newest first.

Plain sqlite3, no Flask and no solver imports — the dashboard's Custom tab
imports this directly (same discipline as core/admin_db.py).
"""

import os
import re
import sqlite3

MASTER_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                         "data", "clues_master.db")

SOURCE = "custom"

# Pasted text arrives from a browser or a chat window: non-breaking spaces,
# smart quotes and en/em dashes all defeat a plain comparison. Normalised the
# same way the dashboard's single-clue box already normalises (pages/pipeline.py).
_SUBS = [
    (" ", " "),                       # NBSP
    ("‘", "'"), ("’", "'"),      # smart single quotes
    ("“", '"'), ("”", '"'),      # smart double quotes
    ("–", "-"), ("—", "-"),      # en / em dash
]

# A trailing enumeration: (7)  (3,4)  (5-2)  (6-2-4)  (4,3,2)
_ENUM_RE = re.compile(
    r"\s*[(\[{<](\d+(?:[\-,]\s*\d+)*)[)\]}>]\s*$")
# The BRACKET FAMILY, not just round ones. A newspaper always prints (4), but a
# clue written in a forum thread is typed by hand and arrives as [4], {4} or <4>
# just as often — and an unrecognised enumeration is not a cosmetic problem: it
# stays in the clue text, so the solver treats "[4]" as clue words to account
# for, and the clue has no length for the answer box or the pattern search.
# Found on a real paste, 2026-09-10: "How long I'm in the hotel, lost? [4]".
# Mismatched pairs ("(4]") are accepted deliberately — that is a typo, and
# refusing to read it would help nobody.


def normalise(text):
    """Clean pasted text — the punctuation a paste brings with it, nothing more.

    The clue's own words are never touched: a setter's wording is not ours to
    rewrite (memory: feedback_plain_cryptic_language / the narrate rule).
    """
    text = (text or "")
    for old, new in _SUBS:
        text = text.replace(old, new)
    return re.sub(r"\s+", " ", text).strip()


def split_enumeration(text):
    """('Sweet note well conveyed by blessing (6)') ->
       ('Sweet note well conveyed by blessing', '6').

    Returns (clue_text, enumeration_or_None). A clue with no trailing bracket
    keeps its text and gets None — the caller supplies the enumeration.
    """
    text = normalise(text)
    m = _ENUM_RE.search(text)
    if not m:
        return text, None
    enum = re.sub(r"\s+", "", m.group(1))
    return text[:m.start()].strip(), enum


def normalise_answer(answer):
    """'tin of soup' / 'TIN-OF-SOUP' -> 'TINOFSOUP'.

    Answers are stored letters-only and upper-case throughout the corpus (the
    enumeration carries the word split) — verified against telegraph/times/
    guardian rows. Accents are kept as typed; the solver normalises them.
    """
    return re.sub(r"[^A-Za-z]", "", answer or "").upper()


def enumeration_length(enumeration):
    """Total letters an enumeration accounts for, or None if unparseable."""
    if not enumeration:
        return None
    parts = re.findall(r"\d+", enumeration)
    if not parts:
        return None
    return sum(int(p) for p in parts)


def validate(clue_text, answer, enumeration):
    """Return (problems, warnings). Only the clue text is required.

    The ANSWER IS OPTIONAL (user, 2026-09-10: "I will not normally have the
    answer"). A clue arrives from a Reddit thread as a puzzle to work out, so
    it is filed answerless — `answer = ''`, the same shape a prize puzzle's
    clues have before the user solves the grid — and the answer is entered on
    the clue's own page with the solving tools, saved by the SAME route a prize
    puzzle uses (/admin/save-all-answers). Only then can it be cascaded: the
    cascade and the prefill both refuse a clue with no answer, which is why
    that is a WARNING here and a step in the flow, never a gate on filing.
    """
    problems, warnings = [], []
    if not (clue_text or "").strip():
        problems.append("No clue text.")
    if not answer:
        warnings.append("No answer yet — solve it on the clue's own page "
                        "(pattern, anagram and synonym tools are all there), "
                        "save it, then cascade.")
    if not enumeration:
        warnings.append("No enumeration — the answer box and the pattern "
                        "search have no length to work with. Add one if the "
                        "clue had brackets.")
    n = enumeration_length(enumeration)
    if answer and n is not None and n != len(answer):
        problems.append("Enumeration (%s) accounts for %d letters but the "
                        "answer %s has %d." % (enumeration, n, answer, len(answer)))
    return problems, warnings


def _connect():
    return sqlite3.connect(MASTER_DB, timeout=30)


def next_puzzle_number(con=None):
    """The next free custom puzzle_number (sequence from 1)."""
    own = con is None
    con = con or _connect()
    try:
        row = con.execute(
            "SELECT MAX(CAST(puzzle_number AS INTEGER)) FROM clues "
            "WHERE source = ?", (SOURCE,)).fetchone()
        return (row[0] or 0) + 1
    finally:
        if own:
            con.close()


def find_duplicate(clue_text, con=None):
    """An existing custom clue with the same text, or None — pasting the same
    clue twice is easy to do and would give it two entries.

    Matched on the CLUE TEXT alone: a custom clue is normally filed with no
    answer, so an answer could not tell two entries apart anyway.
    """
    own = con is None
    con = con or _connect()
    try:
        con.row_factory = sqlite3.Row
        return con.execute(
            "SELECT id, puzzle_number, publication_date, answer FROM clues "
            "WHERE source = ? AND clue_text = ? LIMIT 1",
            (SOURCE, clue_text)).fetchone()
    finally:
        if own:
            con.close()


def add_clue(clue_text, answer, enumeration=None, publication_date=None):
    """File one pasted clue as its own custom puzzle.

    The answer is OPTIONAL — see validate(). With none, the clue is filed with
    `answer = ''` (the corpus stores an unsolved clue that way; `clues.answer`
    is NOT NULL, so an empty string is the only shape available) and waits on
    its own page to be solved.

    Returns {"ok": True, "clue_id": ..., "puzzle_number": ..., "warnings": [...]}
            or {"ok": False, "problems": [...]} / {"ok": False, "duplicate": row}.

    Writes ONE row to `clues` and nothing else — no wfw_* row, no enrichment,
    no reference-DB write. The cascade is what gives the clue its first parse,
    exactly as it does for a scraped puzzle.
    """
    from datetime import date

    clue_text, parsed_enum = split_enumeration(clue_text)
    enumeration = (enumeration or parsed_enum or "").strip() or None
    answer = normalise_answer(answer)
    problems, warnings = validate(clue_text, answer, enumeration)
    if problems:
        return {"ok": False, "problems": problems}

    con = _connect()
    try:
        dup = find_duplicate(clue_text, con)
        if dup is not None:
            return {"ok": False, "duplicate": dict(dup)}
        pnum = next_puzzle_number(con)
        cur = con.execute(
            "INSERT INTO clues (source, puzzle_number, publication_date, "
            "clue_number, direction, clue_text, enumeration, answer, "
            "has_solution, reviewed) "
            "VALUES (?, ?, ?, '1', 'across', ?, ?, ?, NULL, NULL)",
            (SOURCE, str(pnum), publication_date or date.today().isoformat(),
             clue_text, enumeration, answer))
        con.commit()
        return {"ok": True, "clue_id": cur.lastrowid, "puzzle_number": pnum,
                "clue_text": clue_text, "enumeration": enumeration,
                "answer": answer, "warnings": warnings}
    finally:
        con.close()


def list_clues(limit=200):
    """Every custom clue, newest first, with its WFW verdict.

    ORDER: publication_date DESC then id DESC — the date is the day it was
    added, and two clues added the same day read newest-first by id.
    """
    con = _connect()
    try:
        con.row_factory = sqlite3.Row
        return con.execute(
            """SELECT c.id, c.puzzle_number, c.publication_date, c.clue_text,
                      c.enumeration, c.answer,
                      s.status, s.solved_by, s.operation
               FROM clues c
               LEFT JOIN wfw_solve s ON s.clue_id = c.id
               WHERE c.source = ?
               ORDER BY c.publication_date DESC, c.id DESC
               LIMIT ?""", (SOURCE, limit)).fetchall()
    finally:
        con.close()


def delete_clue(clue_id):
    """Remove one custom clue and every trace of a solve attempt.

    Only ever a custom clue — the WHERE clause pins source='custom', so this
    cannot touch a scraped puzzle even if called with the wrong id. Used by the
    dashboard's Delete button for a clue pasted in error.
    """
    con = _connect()
    try:
        row = con.execute("SELECT id FROM clues WHERE id = ? AND source = ?",
                          (clue_id, SOURCE)).fetchone()
        if row is None:
            return {"ok": False, "problems": ["Not a custom clue: %s" % clue_id]}
        for table in ("wfw_solve", "wfw_piece", "wfw_link", "wfw_notes",
                      "wfw_frozen", "wfw_hs_assignments", "wfw_forced_def",
                      "wfw_filler", "wfw_forced_indicator", "wfw_word_split",
                      "structured_explanations"):
            try:
                con.execute("DELETE FROM %s WHERE clue_id = ?" % table, (clue_id,))
            except sqlite3.OperationalError:
                pass          # table absent in this DB — nothing to clear
        con.execute("DELETE FROM clues WHERE id = ? AND source = ?",
                    (clue_id, SOURCE))
        con.commit()
        return {"ok": True, "clue_id": clue_id}
    finally:
        con.close()
