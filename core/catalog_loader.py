"""Catalog loader — reads the mined signature templates into memory.

The redesign's core is catalog-DRIVEN (design §4): an engine does not free-tile
the answer, it walks the mined signatures for its operation in priority order and
tries to instantiate each one. This loader is the single place that reads the
catalog_templates / catalog_template_slots tables (data/clues_master.db) into
small, immutable Template objects the engines consume. Read once at run start and
inject — the engines stay pure and DB-decoupled (same pattern as the RefDB wiring).

A signature like "SYN_F+ABR_F charade def:end" means: two fodder pieces read in
clue order — the first a synonym, the second an abbreviation — with the definition
at the end. A slot's n_words is how many consecutive clue words form that one
piece (SYN_F(2w) = a two-word synonym).
"""

import os
import sqlite3
from dataclasses import dataclass


@dataclass(frozen=True)
class Slot:
    position: int        # 0-based order within the signature (clue order)
    role: str            # SYN_F | ABR_F | ANA_F | ... (fodder/indicator role)
    n_words: int         # how many consecutive clue words form this one piece


@dataclass(frozen=True)
class Template:
    id: int
    operation: str       # 'charade' | 'container' | ...
    signature: str       # the human-readable signature string
    def_pos: str         # 'start' | 'end' | None
    count: int           # frequency in the mining corpus
    priority: int        # rank within the operation (1 = most frequent)
    slots: tuple         # tuple[Slot], in clue order

    @property
    def fodder_word_count(self) -> int:
        """Total clue words the fodder slots consume (definition excluded)."""
        return sum(s.n_words for s in self.slots)


def _default_db_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "clues_master.db")


def load_templates(operation=None, db_path=None, active_only=True):
    """All templates (optionally for one operation), priority order, slots attached.

    Returns list[Template] sorted by (operation, priority). `operation` filters to
    one op (e.g. 'charade'); None loads all. `active_only` honours the soft-disable
    flag so a retired template is skipped without being deleted.
    """
    path = db_path or _default_db_path()
    conn = sqlite3.connect(path, timeout=30)
    try:
        tq = ("SELECT id, operation, signature, def_pos, count, priority "
              "FROM catalog_templates")
        clauses, params = [], []
        if operation is not None:
            clauses.append("operation = ?")
            params.append(operation)
        if active_only:
            clauses.append("active = 1")
        if clauses:
            tq += " WHERE " + " AND ".join(clauses)
        tq += " ORDER BY operation, priority"
        rows = conn.execute(tq, params).fetchall()

        slot_rows = conn.execute(
            "SELECT template_id, position, role, n_words "
            "FROM catalog_template_slots ORDER BY template_id, position").fetchall()
    finally:
        conn.close()

    slots_by_template = {}
    for tid, position, role, n_words in slot_rows:
        slots_by_template.setdefault(tid, []).append(
            Slot(position=position, role=role, n_words=n_words or 1))

    templates = []
    for tid, op, sig, def_pos, count, priority in rows:
        slots = tuple(slots_by_template.get(tid, []))
        templates.append(Template(id=tid, operation=op, signature=sig,
                                   def_pos=def_pos, count=count or 0,
                                   priority=priority or 0, slots=slots))
    return templates


def load_charade_templates(db_path=None):
    """The charade signatures, priority order — what the charade engine consumes."""
    return load_templates(operation="charade", db_path=db_path)


def load_anagram_templates(db_path=None):
    """The anagram signatures, priority order — what the anagram engine consumes."""
    return load_templates(operation="anagram", db_path=db_path)


def load_anagram_charade_templates(db_path=None):
    """The anagram+charade signatures, priority order."""
    return load_templates(operation="anagram_charade", db_path=db_path)
