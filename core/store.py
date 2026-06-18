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
    atom_ids  TEXT               -- JSON array of clue atom ids this piece covers
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
    conn.commit()


def save_parse(conn, clue_id, parse, ctx=None):
    """Persist a solved clue's full Parse. Idempotent: a re-solve replaces this
    clue's prior rows rather than duplicating them. Pieces are preserved on FAIL
    too — the whole point is that the evidence survives the call.

    `ctx` is the WFWAtomContext this parse was built from; when given, the whole
    atomisation is preserved (serialised into wfw_solve.atoms) so the screen can
    render from the exact stored atoms instead of re-atomising the clue text."""
    ensure_schema(conn)
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

    def _piece(role, ord_, text, value, mechanism, source, note, atom_ids):
        conn.execute(
            "INSERT INTO wfw_piece (clue_id, role, ord, text, value, mechanism, "
            "source, note, atom_ids) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (clue_id, role, ord_, text, value, mechanism, source, note,
             json.dumps(list(atom_ids or ()))))

    for i, s in enumerate(parse.sources):
        _piece("source", i, s.text, s.value, s.mechanism, s.source, "",
               s.clue_atom_ids)
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
        "SELECT role, ord, text, value, mechanism, source, note, atom_ids "
        "FROM wfw_piece WHERE clue_id = ?", (clue_id,)).fetchall()

    def _atoms(js):
        return tuple(json.loads(js)) if js else ()

    sources = []
    definition = None
    annotations = []
    for role, ord_, text, value, mechanism, source, note, atom_ids in \
            sorted(pieces, key=lambda p: p[1]):
        if role == "source":
            sources.append(Source(clue_atom_ids=_atoms(atom_ids), text=text,
                                  value=value, mechanism=mechanism, source=source))
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
