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

import json
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
    assembly: str = None      # 'single' | 'charade' | 'container' (operation/assembly schema)
    structure: dict = None    # nested pieces/operations referencing slot indices (parsed JSON)
    tier: str = 'pass'        # 'pass' (may PASS) | 'pending' (PENDING-only, human-reviewed)

    @property
    def fodder_word_count(self) -> int:
        """Total clue words the fodder slots consume (definition excluded)."""
        return sum(s.n_words for s in self.slots)


def _default_db_path():
    return os.path.join(os.path.dirname(os.path.dirname(__file__)),
                        "data", "clues_master.db")


def load_templates(operation=None, db_path=None, active_only=True, tier=None):
    """All templates (optionally for one operation), priority order, slots attached.

    Returns list[Template] sorted by (operation, priority). `operation` filters to
    one op (e.g. 'charade'); None loads all. `active_only` honours the soft-disable
    flag so a retired template is skipped without being deleted. `tier` filters to
    one signature tier ('pass' | 'pending'); None loads all tiers (current behaviour).
    """
    path = db_path or _default_db_path()
    conn = sqlite3.connect(path, timeout=30)
    try:
        # assembly/structure (operation/assembly migration) and tier (signature-tiers
        # migration) are read only if present, so the loader works against a
        # pre-migration catalog too.
        cols = {r[1] for r in conn.execute("PRAGMA table_info(catalog_templates)")}
        has_as = {"assembly", "structure"} <= cols
        has_tier = "tier" in cols
        extra_cols = (["assembly", "structure"] if has_as else []) + \
                     (["tier"] if has_tier else [])
        extra = (", " + ", ".join(extra_cols)) if extra_cols else ""
        tq = ("SELECT id, operation, signature, def_pos, count, priority" + extra +
              " FROM catalog_templates")
        clauses, params = [], []
        if operation is not None:
            clauses.append("operation = ?")
            params.append(operation)
        if active_only:
            clauses.append("active = 1")
        if tier is not None:
            if has_tier:
                clauses.append("tier = ?")
                params.append(tier)
            elif tier != 'pass':
                # pre-migration: no tier column => every template is implicitly 'pass',
                # so a request for any other tier matches nothing.
                clauses.append("1 = 0")
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
    for row in rows:
        tid, op, sig, def_pos, count, priority = row[:6]
        idx = 6
        assembly = structure = None
        tier_val = 'pass'
        if has_as:
            assembly = row[idx]
            structure = json.loads(row[idx + 1]) if row[idx + 1] else None
            idx += 2
        if has_tier:
            tier_val = row[idx] or 'pass'
            idx += 1
        slots = tuple(slots_by_template.get(tid, []))
        templates.append(Template(id=tid, operation=op, signature=sig,
                                   def_pos=def_pos, count=count or 0,
                                   priority=priority or 0, slots=slots,
                                   assembly=assembly, structure=structure,
                                   tier=tier_val))
    return templates


def load_template_tiers(db_path=None):
    """A {template_id: tier} map over ALL templates (active and retired, every operation).

    Feeds the defence-in-depth cap in engine_registry._finish: even if a pending-tier
    template somehow reaches the main cascade and produces a pass, _finish can look up its
    tier here and force the verdict to 'pending'. Returns {} against a pre-migration catalog
    (no tier column) — the cap then never fires, which is the correct pre-migration default
    (every template is implicitly 'pass')."""
    path = db_path or _default_db_path()
    conn = sqlite3.connect(path, timeout=30)
    try:
        cols = {r[1] for r in conn.execute("PRAGMA table_info(catalog_templates)")}
        if "tier" not in cols:
            return {}
        return {tid: (tier or 'pass')
                for tid, tier in conn.execute("SELECT id, tier FROM catalog_templates")}
    finally:
        conn.close()


def load_charade_templates(db_path=None, tier=None):
    """The charade signatures, priority order — what the charade engine consumes."""
    return load_templates(operation="charade", db_path=db_path, tier=tier)


def load_anagram_templates(db_path=None, tier=None):
    """The anagram signatures, priority order — what the anagram engine consumes."""
    return load_templates(operation="anagram", db_path=db_path, tier=tier)


def load_anagram_charade_templates(db_path=None, tier=None):
    """The anagram+charade signatures, priority order."""
    return load_templates(operation="anagram_charade", db_path=db_path, tier=tier)


def load_anagram_container_templates(db_path=None, tier=None):
    """The anagram+container signatures, priority order (seeded from working solves)."""
    return load_templates(operation="anagram_container", db_path=db_path, tier=tier)


def load_container_templates(db_path=None, tier=None):
    """The plain container signatures, priority order (seeded from working solves)."""
    return load_templates(operation="container", db_path=db_path, tier=tier)


def load_container_charade_templates(db_path=None, tier=None):
    """The container+charade signatures, priority order (seeded from working solves)."""
    return load_templates(operation="container_charade", db_path=db_path, tier=tier)


def load_reversal_templates(db_path=None, tier=None):
    """The plain reversal signatures, priority order (seeded from working solves)."""
    return load_templates(operation="reversal", db_path=db_path, tier=tier)


def load_reversal_charade_templates(db_path=None, tier=None):
    """The reversal+charade signatures, priority order (seeded from working solves)."""
    return load_templates(operation="reversal_charade", db_path=db_path, tier=tier)


def load_charade_homophone_templates(db_path=None, tier=None):
    """The charade+homophone signatures, priority order (authored from the clean
    batch). A charade whose pieces concatenate to the answer, one piece a HOM_F
    homophone (an answer span sounding like a clue word/synonym)."""
    return load_templates(operation="charade_homophone", db_path=db_path, tier=tier)


def load_deletion_templates(db_path=None, tier=None):
    """The plain-deletion signatures, priority order (operation 'deletion'). A recipe here
    records the full structure: a base slot (SYN_F/ABR_F), a DEL_I deletion-indicator slot,
    and — for a named deletion — a REM_F removed-letters source slot, plus the definition
    edge. The verifier reads the deletion op from the indicator's DB sub-type and executes
    it. (Supersedes the 4 thin 'del' rows, which recorded only the fodder shape.)"""
    return load_templates(operation="deletion", db_path=db_path, tier=tier)
