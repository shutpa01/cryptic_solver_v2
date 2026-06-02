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
    status      TEXT,            -- 'pass' | 'pending' | 'fail'
    confidence  INTEGER,
    warnings    TEXT,            -- JSON array of plain-English strings
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
"""


def connect(db_path=None):
    return sqlite3.connect(db_path or DEFAULT_DB, timeout=30)


def ensure_schema(conn):
    conn.executescript(SCHEMA)
    conn.commit()


def save_parse(conn, clue_id, parse):
    """Persist a solved clue's full Parse. Idempotent: a re-solve replaces this
    clue's prior rows rather than duplicating them. Pieces are preserved on FAIL
    too — the whole point is that the evidence survives the call."""
    ensure_schema(conn)
    for table in ("wfw_solve", "wfw_piece", "wfw_link"):
        conn.execute("DELETE FROM %s WHERE clue_id = ?" % table, (clue_id,))

    conn.execute(
        "INSERT INTO wfw_solve (clue_id, clue_text, answer_text, operation, "
        "solved_by, status, confidence, warnings) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (clue_id, parse.clue_text, parse.answer_text, parse.operation,
         parse.solved_by, parse.status, parse.confidence,
         json.dumps(list(parse.warnings or []))))

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
        "confidence, warnings FROM wfw_solve WHERE clue_id = ?",
        (clue_id,)).fetchone()
    if head is None:
        return None
    clue_text, answer_text, operation, solved_by, status, confidence, warnings = head

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

    return Parse(clue_text=clue_text, answer_text=answer_text, sources=sources,
                 links=links, annotations=annotations, definition=definition,
                 operation=operation, confidence=confidence or 0,
                 solved_by=solved_by, status=status or "pass",
                 warnings=json.loads(warnings) if warnings else [])


def persist(clue_id, parse, db_path=None):
    """Convenience: open a short-lived connection, save the Parse, close. Used by
    the solve path so every solve becomes durable the moment it is produced."""
    conn = connect(db_path)
    try:
        save_parse(conn, clue_id, parse)
    finally:
        conn.close()
