"""Durable WFW substrate — the persisted Parse, the substrate of RECORD.

SOLVER_REDESIGN.md §2: "preserve all letter-contributing evidence, even on
failure." §10: the persisted provenance "is the WFW substrate persisted: render
straight from it ... nothing may live only in a Python object that vanishes when
the call returns."

Every solve (PASS or FAIL) is persisted for its clue across three additive tables
in clues_master.db, and load_parse() reconstructs the exact wfw_model.Parse so the
screen renders straight from the DB:

  wfw_solve  — one row per clue: clue text/answer, operation, solved_by, the
               verdict (status) and its warnings.
  wfw_piece  — every piece: a wordplay source, the definition, or an annotation
               (indicator / link word) — text, value, mechanism, provenance flag,
               and the clue atom ids it covers.
  wfw_link   — one row per answer letter: which source produced it, the operation,
               the exact clue character (for letter-selection ops), any transform.
"""

import json
import os
import sqlite3

from core.wfw_model import Source, Link, Annotation, Parse
from core.wfw_atoms import context_from_dict

DEFAULT_DB = os.path.join(os.path.dirname(os.path.dirname(__file__)),
                          "data", "clues_master.db")

# Additive only — CREATE IF NOT EXISTS touches no existing table.
SCHEMA = """
CREATE TABLE IF NOT EXISTS wfw_solve (
    clue_id     INTEGER PRIMARY KEY,
    clue_text   TEXT,
    answer_text TEXT,
    operation   TEXT,
    solved_by   TEXT,
    template_id INTEGER,         -- FK to catalog_templates: WHICH signature solved
                                 --   this clue (the clue<->signature cross-reference,
                                 --   local because the catalog lives in this same DB).
                                 --   NULL when no catalog signature solved it.
    status      TEXT,            -- 'pass' | 'pending' | 'fail'
    confidence  INTEGER,
    warnings    TEXT,            -- JSON array of plain-English strings
    atoms       TEXT,            -- JSON of the PRESERVED atomisation (WFWAtomContext
                                 --   .as_dict): the exact atoms the provenance below
                                 --   references, so rendering never re-atomises
    created_at  TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE TABLE IF NOT EXISTS wfw_piece (
    clue_id   INTEGER NOT NULL,
    role      TEXT NOT NULL,     -- 'source' | 'definition' | 'indicator' | 'link'
    ord       INTEGER NOT NULL,  -- order within its role (source_index for sources)
    text      TEXT,
    value     TEXT,
    mechanism TEXT,
    source    TEXT,              -- 'db' | 'pending'
    note      TEXT,
    atom_ids  TEXT,              -- JSON array of clue atom ids this piece covers
    transform TEXT               -- SOURCE pieces: what happened to `value` on its
                                 --   way to the answer squares (cuts with the
                                 --   position each was taken from, letter shift,
                                 --   reversal) as a core.piece_transform JSON
                                 --   string. Recorded at authoring, never derived
                                 --   at render. '' / NULL = landed unchanged, or a
                                 --   row written before this column existed.
);
CREATE TABLE IF NOT EXISTS wfw_link (
    clue_id      INTEGER NOT NULL,
    answer_pos   INTEGER NOT NULL,
    source_index INTEGER,
    operation    TEXT,
    clue_atom_id TEXT,
    transform    TEXT
);
CREATE TABLE IF NOT EXISTS wfw_forced_def (
    clue_id INTEGER PRIMARY KEY,   -- a human override: SOLVE this clue with the
    text    TEXT                   --   definition pinned to exactly this edge phrase,
);                                 --   so the wordplay must account for the rest.
CREATE TABLE IF NOT EXISTS wfw_filler (
    clue_id INTEGER NOT NULL,      -- per-clue SURFACE FILLER: setter padding with no
    word    TEXT NOT NULL,         --   cryptic role, tagged here ONLY (never link_words);
    PRIMARY KEY (clue_id, word)    --   accounted in the solve like a link, this clue alone.
);
CREATE TABLE IF NOT EXISTS wfw_forced_indicator (
    clue_id INTEGER NOT NULL,      -- per-clue FORCED INDICATOR: a contiguous span the
    phrase  TEXT NOT NULL,         --   human pins as an indicator of `wptype`, this clue
    wptype  TEXT NOT NULL,         --   ONLY (never the shared indicators table). Looked up
    PRIMARY KEY (clue_id, phrase, wptype)  --   as a PHRASE; makes a multi-word indicator
);                                 --   (e.g. 'picked up') valid without per-word typing.
CREATE TABLE IF NOT EXISTS wfw_frozen (
    clue_id INTEGER PRIMARY KEY    -- FROZEN: a forced clue that reached a clean PASS. Its
);                                 --   stored parse is authoritative; save_parse refuses to
                                   --   DOWNGRADE it (never reverts to fail on a later run);
                                   --   only admin unforce clears it. Separate table so it
                                   --   survives save_parse's delete+reinsert of wfw_solve.
CREATE TABLE IF NOT EXISTS wfw_word_split (
    clue_id     INTEGER NOT NULL,  -- per-clue HUMAN SPLIT of a run-together clue word into
    token_index INTEGER NOT NULL,  --   parts that take their OWN roles: "fightback" =
    offset      INTEGER NOT NULL,  --   fight (synonym WAR) + back (reversal indicator).
    PRIMARY KEY (clue_id, token_index, offset)
);                                 --   token_index is the word's position in the clue's token
                                   --   list; offset is the character in that word where a new
                                   --   part BEGINS (fightback -> 5). One row per cut, so a word
                                   --   can be split more than once. Stored because every
                                   --   assignment refers to words by INDEX: re-deriving the
                                   --   split would renumber the words and corrupt a saved
                                   --   reading. Hyphens split without a row (that rule is in
                                   --   the text itself); this is for words written solid.
CREATE TABLE IF NOT EXISTS wfw_hs_assignments (
    clue_id INTEGER PRIMARY KEY,   -- the hand-solver's FULL assignment list (JSON), saved on
    payload TEXT                   --   each Assign AND on Resolve so a failed solve never
);                                 --   loses the work; restored into the grid on load.
CREATE TABLE IF NOT EXISTS wfw_notes (
    clue_id INTEGER PRIMARY KEY,   -- free-text note authored in the hand-solver and shown on
    note TEXT                      --   the clue page (extra info for the user). Per-clue,
);                                 --   never written to any reference DB.
"""


def connect(db_path=None):
    return sqlite3.connect(db_path or DEFAULT_DB, timeout=30)


def ensure_schema(conn):
    conn.executescript(SCHEMA)
    # Additive migration for a wfw_solve created before the atoms column existed:
    # ALTER ADD COLUMN is non-destructive (existing rows get NULL).
    cols = {r[1] for r in conn.execute("PRAGMA table_info(wfw_solve)")}
    if "atoms" not in cols:
        conn.execute("ALTER TABLE wfw_solve ADD COLUMN atoms TEXT")
    if "template_id" not in cols:
        conn.execute("ALTER TABLE wfw_solve ADD COLUMN template_id INTEGER")
    # Same additive migration for wfw_piece.transform (2026-08-17): existing rows
    # get NULL, which reads back as "nothing recorded" — no stored solve changes.
    pcols = {r[1] for r in conn.execute("PRAGMA table_info(wfw_piece)")}
    if "transform" not in pcols:
        conn.execute("ALTER TABLE wfw_piece ADD COLUMN transform TEXT")
    conn.commit()


def save_parse(conn, clue_id, parse, ctx=None):
    """Persist a solved clue's full Parse. Idempotent: a re-solve replaces this
    clue's prior rows rather than duplicating them. Pieces are preserved on FAIL
    too — the whole point is that the evidence survives the call.

    `ctx` is the WFWAtomContext this parse was built from; when given, the whole
    atomisation is preserved (serialised into wfw_solve.atoms) so the screen can
    render from the exact stored atoms instead of re-atomising the clue text.

    FREEZE GUARANTEE: a FROZEN clue (a forced clue that reached a clean PASS) is never
    DOWNGRADED — a later re-solve that does not pass is discarded, so the stored forced
    pass survives any subsequent run. Only admin unforce (clear_frozen) lifts it. And a
    clean PASS for a clue that carries any manual override auto-freezes it here."""
    ensure_schema(conn)
    if parse.status != "pass" and is_frozen(conn, clue_id):
        return                                   # never revert a frozen forced pass
    for table in ("wfw_solve", "wfw_piece", "wfw_link"):
        conn.execute("DELETE FROM %s WHERE clue_id = ?" % table, (clue_id,))

    conn.execute(
        "INSERT INTO wfw_solve (clue_id, clue_text, answer_text, operation, "
        "solved_by, template_id, status, confidence, warnings, atoms) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (clue_id, parse.clue_text, parse.answer_text, parse.operation,
         parse.solved_by, getattr(parse, "template_id", None), parse.status,
         parse.confidence, json.dumps(list(parse.warnings or [])),
         json.dumps(ctx.as_dict()) if ctx is not None else None))

    def _piece(role, ord_, text, value, mechanism, source, note, atom_ids,
               transform=""):
        conn.execute(
            "INSERT INTO wfw_piece (clue_id, role, ord, text, value, mechanism, "
            "source, note, atom_ids, transform) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (clue_id, role, ord_, text, value, mechanism, source, note,
             json.dumps(list(atom_ids or ())), transform or ""))

    for i, s in enumerate(parse.sources):
        _piece("source", i, s.text, s.value, s.mechanism, s.source, "",
               s.clue_atom_ids, getattr(s, "transform", "") or "")
    if parse.definition is not None:
        d = parse.definition
        _piece("definition", 0, d.text, d.value, d.mechanism, d.source, "",
               d.clue_atom_ids)
    for i, a in enumerate(parse.annotations):
        _piece(a.role, i, a.text, "", "", a.source, a.note, a.clue_atom_ids)

    for l in parse.links:
        conn.execute(
            "INSERT INTO wfw_link (clue_id, answer_pos, source_index, operation, "
            "clue_atom_id, transform) VALUES (?, ?, ?, ?, ?, ?)",
            (clue_id, l.answer_pos, l.source_index, l.operation,
             l.clue_atom_id, l.transform))
    # Auto-freeze: a clean PASS for a clue carrying any manual override is the human's
    # forced solution — freeze it so it can never later revert.
    if parse.status == "pass" and _has_overrides(conn, clue_id):
        set_frozen(conn, clue_id)
    conn.commit()


def delete_parse(conn, clue_id):
    """Remove a clue's stored solve (wfw_solve / wfw_piece / wfw_link) so it reverts to
    UNSOLVED. Does NOT touch the hand-solver assignment, notes or overrides. Use when the
    human's assignment has no valid reading (e.g. every wordplay word marked 'none' =
    deliberately unsolvable) and a stale or fabricated solve must stop being displayed.
    Ignores the freeze — the caller lifts it first when a stale freeze is protecting the
    solve being cleared."""
    ensure_schema(conn)
    for table in ("wfw_solve", "wfw_piece", "wfw_link"):
        conn.execute("DELETE FROM %s WHERE clue_id = ?" % table, (clue_id,))
    conn.commit()


def is_frozen(conn, clue_id):
    """True if this clue is frozen (a forced pass that must never be downgraded)."""
    ensure_schema(conn)
    return conn.execute("SELECT 1 FROM wfw_frozen WHERE clue_id = ?",
                        (clue_id,)).fetchone() is not None


def set_frozen(conn, clue_id):
    """Mark a clue frozen. Idempotent. (Commit handled by the caller / save_parse.)"""
    ensure_schema(conn)
    conn.execute("INSERT OR IGNORE INTO wfw_frozen (clue_id) VALUES (?)", (clue_id,))


def clear_frozen(conn, clue_id):
    """Lift the freeze (admin unforce), so the clue solves normally again."""
    ensure_schema(conn)
    conn.execute("DELETE FROM wfw_frozen WHERE clue_id = ?", (clue_id,))
    conn.commit()


def _has_overrides(conn, clue_id):
    """True if this clue carries any manual override (filler / forced def / forced
    indicator) — i.e. a human has forced something on it."""
    for q in ("SELECT 1 FROM wfw_filler WHERE clue_id = ?",
              "SELECT 1 FROM wfw_forced_def WHERE clue_id = ?",
              "SELECT 1 FROM wfw_forced_indicator WHERE clue_id = ?"):
        if conn.execute(q, (clue_id,)).fetchone() is not None:
            return True
    return False


def unforce(conn, clue_id):
    """Admin UNFORCE: drop ALL of this clue's manual overrides and lift the freeze, so the
    next solve evaluates it from scratch. The ONLY way a frozen verdict changes."""
    ensure_schema(conn)
    conn.execute("DELETE FROM wfw_filler WHERE clue_id = ?", (clue_id,))
    conn.execute("DELETE FROM wfw_forced_def WHERE clue_id = ?", (clue_id,))
    conn.execute("DELETE FROM wfw_forced_indicator WHERE clue_id = ?", (clue_id,))
    conn.execute("DELETE FROM wfw_frozen WHERE clue_id = ?", (clue_id,))
    conn.execute("DELETE FROM wfw_hs_assignments WHERE clue_id = ?", (clue_id,))
    conn.commit()


def load_parse(conn, clue_id):
    """Reconstruct the exact wfw_model.Parse persisted for this clue, or None if
    nothing is stored. The screen renders straight from this."""
    ensure_schema(conn)
    head = conn.execute(
        "SELECT clue_text, answer_text, operation, solved_by, status, "
        "confidence, warnings, template_id FROM wfw_solve WHERE clue_id = ?",
        (clue_id,)).fetchone()
    if head is None:
        return None
    (clue_text, answer_text, operation, solved_by, status, confidence,
     warnings, template_id) = head

    pieces = conn.execute(
        "SELECT role, ord, text, value, mechanism, source, note, atom_ids, "
        "transform FROM wfw_piece WHERE clue_id = ?", (clue_id,)).fetchall()

    def _atoms(js):
        return tuple(json.loads(js)) if js else ()

    sources = []
    definition = None
    annotations = []
    for role, ord_, text, value, mechanism, source, note, atom_ids, transform in \
            sorted(pieces, key=lambda p: p[1]):
        if role == "source":
            sources.append(Source(clue_atom_ids=_atoms(atom_ids), text=text,
                                  value=value, mechanism=mechanism, source=source,
                                  transform=transform or ""))
        elif role == "definition":
            definition = Source(clue_atom_ids=_atoms(atom_ids), text=text,
                                value=value, mechanism=mechanism, source=source)
        else:  # 'indicator' | 'link'
            annotations.append(Annotation(clue_atom_ids=_atoms(atom_ids),
                                          text=text, role=role, note=note,
                                          source=source))

    link_rows = conn.execute(
        "SELECT answer_pos, source_index, operation, clue_atom_id, transform "
        "FROM wfw_link WHERE clue_id = ? ORDER BY answer_pos", (clue_id,)).fetchall()
    links = [Link(answer_pos=r[0], source_index=r[1], operation=r[2],
                  clue_atom_id=r[3], transform=r[4]) for r in link_rows]

    parse = Parse(clue_text=clue_text, answer_text=answer_text, sources=sources,
                  links=links, annotations=annotations, definition=definition,
                  operation=operation, confidence=confidence or 0,
                  solved_by=solved_by, status=status or "pass",
                  warnings=json.loads(warnings) if warnings else [])
    parse.template_id = template_id      # the signature cross-reference (may be None)
    return parse


def set_status(conn, clue_id, status):
    """Manually override a clue's stored verdict (pass / pending / fail). Persists in
    wfw_solve; the screen renders it until the clue is re-solved."""
    ensure_schema(conn)
    conn.execute("UPDATE wfw_solve SET status = ? WHERE clue_id = ?",
                 (status, clue_id))
    conn.commit()


def set_manual_definition(conn, clue_id, text, answer):
    """Set a DISPLAY-ONLY definition (no reference-DB write, no checks) — for &lit
    clues where the whole clue is both the definition and the wordplay. Updates the
    stored definition piece (mechanism/source 'manual'), inserting one if absent. The
    atom span is left empty (the text is shown, the clue line is not re-highlighted)."""
    ensure_schema(conn)
    n = conn.execute(
        "UPDATE wfw_piece SET text = ?, value = ?, mechanism = 'manual', "
        "source = 'manual', atom_ids = '[]' WHERE clue_id = ? AND role = 'definition'",
        (text, answer, clue_id)).rowcount
    if n == 0:
        conn.execute(
            "INSERT INTO wfw_piece (clue_id, role, ord, text, value, mechanism, "
            "source, note, atom_ids) VALUES (?, 'definition', 0, ?, ?, 'manual', "
            "'manual', '', '[]')", (clue_id, text, answer))
    conn.commit()


def set_forced_definition(conn, clue_id, text):
    """Pin this clue's definition to exactly `text` for SOLVING (not display): the
    re-solve wraps `defines` to confirm only this edge phrase, so the wordplay engines
    must reconstruct the answer from every other word. Persists until cleared, so the
    override survives later re-runs. Unlike set_manual_definition (display only), this
    changes what the cascade actually does."""
    ensure_schema(conn)
    conn.execute(
        "INSERT INTO wfw_forced_def (clue_id, text) VALUES (?, ?) "
        "ON CONFLICT(clue_id) DO UPDATE SET text = excluded.text",
        (clue_id, text))
    conn.commit()


def get_forced_definition(conn, clue_id):
    """The human-pinned definition for this clue, or None."""
    ensure_schema(conn)
    row = conn.execute("SELECT text FROM wfw_forced_def WHERE clue_id = ?",
                       (clue_id,)).fetchone()
    return row[0] if row else None


def clear_forced_definition(conn, clue_id):
    """Drop the pinned definition so the clue solves with the normal definition stage."""
    ensure_schema(conn)
    conn.execute("DELETE FROM wfw_forced_def WHERE clue_id = ?", (clue_id,))
    conn.commit()


def set_hs_assignments(conn, clue_id, payload):
    """Persist the hand-solver's FULL assignment list (a JSON string) for this clue, so a
    failed Resolve — or leaving and coming back — never loses the work; the grid restores it
    on load. Saved on each Assign and on Resolve. An empty/`[]` payload clears the record."""
    ensure_schema(conn)
    payload = (payload or "").strip()
    if not payload or payload in ("[]", "null"):
        conn.execute("DELETE FROM wfw_hs_assignments WHERE clue_id = ?", (clue_id,))
    else:
        conn.execute(
            "INSERT INTO wfw_hs_assignments (clue_id, payload) VALUES (?, ?) "
            "ON CONFLICT(clue_id) DO UPDATE SET payload = excluded.payload",
            (clue_id, payload))
    conn.commit()


def get_hs_assignments(conn, clue_id):
    """The saved hand-solver assignment JSON for this clue, or '' if none."""
    ensure_schema(conn)
    row = conn.execute("SELECT payload FROM wfw_hs_assignments WHERE clue_id = ?",
                       (clue_id,)).fetchone()
    return row[0] if row and row[0] else ""


def set_note(conn, clue_id, note):
    """Save (or clear) a free-text note for this clue — authored in the hand-solver, shown on
    the clue page for the user. An empty note removes the record. Never touches a reference DB."""
    ensure_schema(conn)
    note = (note or "").strip()
    if not note:
        conn.execute("DELETE FROM wfw_notes WHERE clue_id = ?", (clue_id,))
    else:
        conn.execute("INSERT INTO wfw_notes (clue_id, note) VALUES (?, ?) "
                     "ON CONFLICT(clue_id) DO UPDATE SET note = excluded.note",
                     (clue_id, note))
    conn.commit()


def get_note(conn, clue_id):
    """The free-text note for this clue, or '' if none."""
    ensure_schema(conn)
    row = conn.execute("SELECT note FROM wfw_notes WHERE clue_id = ?", (clue_id,)).fetchone()
    return row[0] if row and row[0] else ""


def add_clue_filler(conn, clue_id, word):
    """Tag a word as SURFACE FILLER for THIS clue only — setter padding with no cryptic
    role. Accounted in the solve like a link, but stored per-clue here and NEVER written
    to the shared link_words table (so common words like 'get'/'will' can't pollute it)."""
    ensure_schema(conn)
    w = (word or "").strip().lower()
    if w:
        conn.execute("INSERT OR IGNORE INTO wfw_filler (clue_id, word) VALUES (?, ?)",
                     (clue_id, w))
        conn.commit()


def get_clue_filler(conn, clue_id):
    """The set of surface-filler words tagged for this clue (lower-cased)."""
    ensure_schema(conn)
    return {r[0] for r in conn.execute(
        "SELECT word FROM wfw_filler WHERE clue_id = ?", (clue_id,))}


def clear_clue_filler(conn, clue_id, word=None):
    """Untag one filler word, or all of this clue's filler when word is None."""
    ensure_schema(conn)
    if word is None:
        conn.execute("DELETE FROM wfw_filler WHERE clue_id = ?", (clue_id,))
    else:
        conn.execute("DELETE FROM wfw_filler WHERE clue_id = ? AND word = ?",
                     (clue_id, (word or "").strip().lower()))
    conn.commit()


def get_word_splits(conn, clue_id):
    """This clue's human word splits as {token_index: [offset, ...]} — the points where a
    solid clue word is broken into parts that take their own roles ("fightback" -> fight +
    back). Every surface that numbers the clue's words (the /hs grid AND the commit gate)
    must apply the SAME splits, or a saved assignment's word indices no longer line up."""
    ensure_schema(conn)
    out = {}
    for ti, off in conn.execute(
            "SELECT token_index, offset FROM wfw_word_split WHERE clue_id = ? "
            "ORDER BY token_index, offset", (clue_id,)):
        out.setdefault(ti, []).append(off)
    return out


def add_word_split(conn, clue_id, token_index, offset):
    """Split this clue's word `token_index` at `offset` (the character the SECOND part
    starts at). Idempotent."""
    ensure_schema(conn)
    conn.execute("INSERT OR IGNORE INTO wfw_word_split (clue_id, token_index, offset) "
                 "VALUES (?, ?, ?)", (clue_id, int(token_index), int(offset)))
    conn.commit()


def clear_word_split(conn, clue_id, token_index=None):
    """Undo one word's splits, or every split on this clue when token_index is None."""
    ensure_schema(conn)
    if token_index is None:
        conn.execute("DELETE FROM wfw_word_split WHERE clue_id = ?", (clue_id,))
    else:
        conn.execute("DELETE FROM wfw_word_split WHERE clue_id = ? AND token_index = ?",
                     (clue_id, int(token_index)))
    conn.commit()


def add_forced_indicator(conn, clue_id, phrase, wptype):
    """Pin a contiguous span as an indicator of `wptype` for THIS clue only — stored here,
    NEVER written to the shared indicators table. Looked up as a phrase, so a multi-word
    indicator (e.g. 'picked up') is valid even when a component word isn't typed that way."""
    ensure_schema(conn)
    p = (phrase or "").strip()
    t = (wptype or "").strip().lower()
    if p and t:
        conn.execute("INSERT OR IGNORE INTO wfw_forced_indicator "
                     "(clue_id, phrase, wptype) VALUES (?, ?, ?)", (clue_id, p, t))
        conn.commit()


def get_forced_indicators(conn, clue_id):
    """[(phrase, wptype), ...] forced indicators for this clue (possibly empty)."""
    ensure_schema(conn)
    return [(r[0], r[1]) for r in conn.execute(
        "SELECT phrase, wptype FROM wfw_forced_indicator WHERE clue_id = ?", (clue_id,))]


def clear_forced_indicator(conn, clue_id, phrase=None):
    """Drop one forced indicator (any type for that phrase), or all of this clue's when
    phrase is None."""
    ensure_schema(conn)
    if phrase is None:
        conn.execute("DELETE FROM wfw_forced_indicator WHERE clue_id = ?", (clue_id,))
    else:
        conn.execute("DELETE FROM wfw_forced_indicator WHERE clue_id = ? AND phrase = ?",
                     (clue_id, (phrase or "").strip()))
    conn.commit()


def load_atoms(conn, clue_id):
    """Reconstruct the PRESERVED atomisation for this clue (the exact atoms the
    stored provenance references), or None if none was stored (a row solved
    before atom preservation). Lets the screen render from the preserved atoms
    rather than re-running the atomiser."""
    ensure_schema(conn)
    row = conn.execute("SELECT atoms FROM wfw_solve WHERE clue_id = ?",
                       (clue_id,)).fetchone()
    if not row or not row[0]:
        return None
    return context_from_dict(json.loads(row[0]))


def persist(clue_id, parse, ctx=None, db_path=None):
    """Convenience: open a short-lived connection, save the Parse, close. Used by
    the solve path so every solve becomes durable the moment it is produced.
    `ctx` preserves the atomisation alongside the parse (see save_parse)."""
    conn = connect(db_path)
    try:
        save_parse(conn, clue_id, parse, ctx)
    finally:
        conn.close()
