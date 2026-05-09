"""Shadow DB for the parallel-of-production system.

A separate sqlite file `data/shadow_blog_v0.db` holds:

  1. Vocabulary tables mirroring the live DB structure (synonyms_pairs,
     wordplay, indicators, definition_answers_augmented), with clue_id
     and solve_id back-provenance so every shadow row can be audited
     to the clue and solve attempt that justified it.

  2. `solves` — one row per (clue_id, signature) attempt, holding the
     full form tree as JSON, the verdict, per-check details, and
     enrichment candidates emitted by the verifier.

  3. `solve_assignments` — one row per word/span in the clue, for
     cheap "find all clues where word X plays role Y" queries. The
     JSON form blob in `solves` is the source of truth; this table
     is the queryable index.

Verification reads from a composite of (live ∪ shadow). The live DB
is never touched.
"""
from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Iterable, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
DEFAULT_SHADOW_PATH = PROJECT_ROOT / "data" / "shadow_blog_v0.db"


SHADOW_SCHEMA = """
-- Vocabulary tables (with clue_id / solve_id provenance) ------------------

CREATE TABLE IF NOT EXISTS synonyms_pairs (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    synonym TEXT NOT NULL,
    source TEXT DEFAULT 'shadow_blog',
    clue_id INTEGER,
    solve_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_sp_word ON synonyms_pairs(word);
CREATE INDEX IF NOT EXISTS idx_sp_syn ON synonyms_pairs(synonym);
CREATE INDEX IF NOT EXISTS idx_sp_clue ON synonyms_pairs(clue_id);

CREATE TABLE IF NOT EXISTS wordplay (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    indicator TEXT NOT NULL,
    substitution TEXT NOT NULL,
    category TEXT,
    confidence TEXT DEFAULT 'medium',
    notes TEXT,
    clue_id INTEGER,
    solve_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_wp_ind ON wordplay(indicator);
CREATE INDEX IF NOT EXISTS idx_wp_sub ON wordplay(substitution);
CREATE INDEX IF NOT EXISTS idx_wp_clue ON wordplay(clue_id);

CREATE TABLE IF NOT EXISTS indicators (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    word TEXT NOT NULL,
    wordplay_type TEXT NOT NULL,
    subtype TEXT,
    confidence TEXT DEFAULT 'medium',
    source TEXT DEFAULT 'shadow_blog',
    clue_id INTEGER,
    solve_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_ind_word ON indicators(word);
CREATE INDEX IF NOT EXISTS idx_ind_clue ON indicators(clue_id);

CREATE TABLE IF NOT EXISTS definition_answers_augmented (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    definition TEXT NOT NULL,
    answer TEXT NOT NULL,
    source TEXT DEFAULT 'shadow_blog',
    clue_id INTEGER,
    solve_id INTEGER
);
CREATE INDEX IF NOT EXISTS idx_da_ans ON definition_answers_augmented(answer);
CREATE INDEX IF NOT EXISTS idx_da_def ON definition_answers_augmented(definition);
CREATE INDEX IF NOT EXISTS idx_da_clue ON definition_answers_augmented(clue_id);

-- Per-clue solve records ---------------------------------------------------

CREATE TABLE IF NOT EXISTS solves (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clue_id INTEGER NOT NULL,
    signature TEXT NOT NULL,
    verdict TEXT NOT NULL,                 -- PASS | PENDING | FAIL
    answer TEXT NOT NULL,
    form_json TEXT NOT NULL,               -- recursive form tree (source of truth)
    checks_json TEXT,                      -- per-rule check details
    enrichments_json TEXT,                 -- enrichment candidates from verifier
    run_number INTEGER NOT NULL DEFAULT 1, -- 1=initial, 2=enriched, 3=review
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_solves_clue ON solves(clue_id);
CREATE INDEX IF NOT EXISTS idx_solves_signature ON solves(signature);
CREATE INDEX IF NOT EXISTS idx_solves_verdict ON solves(verdict);
-- idx_solves_run is created in _migrate_run_number after the column is added.

-- Per-clue word/span role assignments (queryable index over the form) -----

CREATE TABLE IF NOT EXISTS solve_assignments (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    solve_id INTEGER NOT NULL,
    clue_id INTEGER NOT NULL,
    span_start INTEGER NOT NULL,           -- 0-based word index
    span_end INTEGER NOT NULL,             -- exclusive
    surface_phrase TEXT NOT NULL,          -- lowercased phrase from clue
    role TEXT NOT NULL,                    -- definition | leaf | indicator | link
    mechanism TEXT,                        -- synonym | abbreviation | literal | positional | homophone
    op TEXT,                               -- for indicators: which operation
    op_kind TEXT,                          -- positional/deletion/acrostic kind
    value TEXT,                            -- contributed letters (leaves)
    qualifier TEXT,                        -- preserved blogger qualifier
    db_source TEXT,                        -- live | shadow | literal | na
    db_table TEXT,
    db_row_id INTEGER,
    FOREIGN KEY(solve_id) REFERENCES solves(id)
);
CREATE INDEX IF NOT EXISTS idx_assign_solve ON solve_assignments(solve_id);
CREATE INDEX IF NOT EXISTS idx_assign_clue ON solve_assignments(clue_id);
CREATE INDEX IF NOT EXISTS idx_assign_phrase ON solve_assignments(surface_phrase);
CREATE INDEX IF NOT EXISTS idx_assign_role ON solve_assignments(role);
CREATE INDEX IF NOT EXISTS idx_assign_mech ON solve_assignments(mechanism);

-- Seeder failure log -----------------------------------------------------
--
-- Records every clue the seeder processed but didn't end up writing as a
-- PASS solve. Two kinds:
--
--   translation_error — the translator couldn't build a Form (malformed
--     JSON, unrecognised op, missing required fields, blog text the parser
--     couldn't decode, etc.). No Form produced.
--   verifier_fail    — the translator built a Form but clipboard_verifier
--     rejected it. The translated_form_json is preserved for inspection.
--
-- Together with `solves`, this table covers every clue the seeder
-- touched. Presence in (solves ∪ seed_failures) means processed.

CREATE TABLE IF NOT EXISTS seed_failures (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clue_id INTEGER NOT NULL,
    seed_source TEXT NOT NULL,             -- 'structured_explanations' | 'blog'
    source TEXT,                           -- the publication: times | telegraph | ...
    puzzle_number TEXT,
    clue_number TEXT,
    direction TEXT,
    clue_text TEXT,
    answer TEXT,
    structured_explanation_id INTEGER,     -- the source SE row id, if from JSON path
    components_json TEXT,                  -- the original components JSON, if any
    blog_text TEXT,                        -- the original blog text, if from blog path
    translated_form_json TEXT,             -- the Form we built (if translation succeeded)
    failure_kind TEXT NOT NULL,            -- 'translation_error' | 'verifier_fail'
    failure_detail TEXT,                   -- error message or verifier's check details
    enrichments_json TEXT,                 -- if verifier_fail, enrichment candidates
    run_number INTEGER NOT NULL DEFAULT 1, -- 1=initial, 2=enriched, 3=review
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_seed_fail_clue ON seed_failures(clue_id);
CREATE INDEX IF NOT EXISTS idx_seed_fail_kind ON seed_failures(failure_kind);
CREATE INDEX IF NOT EXISTS idx_seed_fail_source ON seed_failures(seed_source);
-- idx_seed_fail_run is created in _migrate_run_number after the column is added.
"""


def ensure_shadow(path=None) -> sqlite3.Connection:
    """Create-or-open the shadow DB and ensure schema. Also runs
    light migrations for older shadow_db files that pre-date the
    `run_number` column on `solves` and `seed_failures`."""
    p = Path(path) if path else DEFAULT_SHADOW_PATH
    p.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(p))
    conn.executescript(SHADOW_SCHEMA)
    _migrate_run_number(conn)
    conn.commit()
    return conn


def _migrate_run_number(conn: sqlite3.Connection) -> None:
    """Add `run_number` to solves / seed_failures if absent. Existing
    rows get the default value 1 (treated as initial-run history)."""
    for table in ("solves", "seed_failures"):
        cols = [r[1] for r in conn.execute(
            f"PRAGMA table_info({table})").fetchall()]
        if "run_number" not in cols:
            conn.execute(
                f"ALTER TABLE {table} ADD COLUMN "
                f"run_number INTEGER NOT NULL DEFAULT 1")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_solves_run "
                  "ON solves(run_number)")
    conn.execute("CREATE INDEX IF NOT EXISTS idx_seed_fail_run "
                  "ON seed_failures(run_number)")


def reset_shadow(path=None) -> sqlite3.Connection:
    """Wipe and recreate the shadow DB. Used at the start of a run."""
    p = Path(path) if path else DEFAULT_SHADOW_PATH
    if p.exists():
        p.unlink()
    return ensure_shadow(path)


def write_candidates(candidates: Iterable[dict],
                      conn: sqlite3.Connection) -> dict:
    """Write candidate enrichment rows to the shadow DB.

    Each candidate has: kind, source_word, value, optional subtype,
    optional clue_id, optional solve_id. The clue_id / solve_id
    provide back-provenance so a reviewer can trace every shadow row
    to the clue and solve attempt that justified it.

    Returns counts of inserted rows by kind.
    """
    counts = {"synonym": 0, "abbreviation": 0,
              "indicator": 0, "definition": 0, "skipped": 0}
    for c in candidates:
        kind = c.get("kind")
        word = (c.get("source_word") or "").strip()
        value = (c.get("value") or "").strip()
        clue_id = c.get("clue_id")
        solve_id = c.get("solve_id")
        if not (kind and word and value):
            counts["skipped"] += 1
            continue
        if kind == "synonym":
            existing = conn.execute(
                "SELECT 1 FROM synonyms_pairs "
                "WHERE LOWER(word)=? AND UPPER(synonym)=?",
                (word.lower(), value.upper())).fetchone()
            if existing:
                counts["skipped"] += 1
                continue
            conn.execute(
                "INSERT INTO synonyms_pairs "
                "(word, synonym, source, clue_id, solve_id) "
                "VALUES (?, ?, 'shadow_blog', ?, ?)",
                (word, value.upper(), clue_id, solve_id))
            counts["synonym"] += 1
        elif kind == "abbreviation":
            existing = conn.execute(
                "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
                "AND UPPER(substitution)=? AND category='abbreviation'",
                (word.lower(), value.upper())).fetchone()
            if existing:
                counts["skipped"] += 1
                continue
            conn.execute(
                "INSERT INTO wordplay (indicator, substitution, "
                "category, confidence, clue_id, solve_id) "
                "VALUES (?, ?, 'abbreviation', 'medium', ?, ?)",
                (word, value.upper(), clue_id, solve_id))
            counts["abbreviation"] += 1
        elif kind == "indicator":
            sub = c.get("subtype")
            wp_type = c.get("wordplay_type") or c.get("operation") \
                or c.get("op")
            if not wp_type:
                counts["skipped"] += 1
                continue
            existing = conn.execute(
                "SELECT 1 FROM indicators WHERE LOWER(word)=? "
                "AND wordplay_type=? AND COALESCE(subtype,'')=?",
                (word.lower(), wp_type, sub or "")).fetchone()
            if existing:
                counts["skipped"] += 1
                continue
            conn.execute(
                "INSERT INTO indicators (word, wordplay_type, "
                "subtype, confidence, clue_id, solve_id) "
                "VALUES (?, ?, ?, 'medium', ?, ?)",
                (word, wp_type, sub, clue_id, solve_id))
            counts["indicator"] += 1
        elif kind == "definition":
            existing = conn.execute(
                "SELECT 1 FROM definition_answers_augmented "
                "WHERE LOWER(definition)=? AND UPPER(answer)=?",
                (word.lower(), value.upper())).fetchone()
            if existing:
                counts["skipped"] += 1
                continue
            conn.execute(
                "INSERT INTO definition_answers_augmented "
                "(definition, answer, clue_id, solve_id) "
                "VALUES (?, ?, ?, ?)",
                (word, value.upper(), clue_id, solve_id))
            counts["definition"] += 1
        else:
            counts["skipped"] += 1
    conn.commit()
    return counts


# --- Solve and assignment writers -----------------------------------------

def write_solve(conn: sqlite3.Connection,
                clue_id: int,
                signature: str,
                verdict: str,
                answer: str,
                form_dict: dict,
                checks: Optional[list] = None,
                enrichments: Optional[list] = None,
                run_number: int = 1) -> int:
    """Insert a single solve record. Returns the solve_id (lastrowid).

    `verdict` is one of 'PASS', 'PENDING', 'FAIL'. PENDING is used for
    &lit and cryptic_definition forms that mechanically verify but
    require human review before contributing to the catalog.
    `run_number` records which run produced this row: 1=initial cascade,
    2=cascade after enrichment, 3=human review."""
    cur = conn.execute(
        "INSERT INTO solves "
        "(clue_id, signature, verdict, answer, form_json, "
        "checks_json, enrichments_json, run_number) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
        (clue_id, signature, verdict, answer.upper(),
         json.dumps(form_dict, ensure_ascii=False),
         json.dumps(checks, ensure_ascii=False) if checks else None,
         json.dumps(enrichments, ensure_ascii=False) if enrichments else None,
         run_number))
    conn.commit()
    return cur.lastrowid


def write_assignments(conn: sqlite3.Connection,
                       solve_id: int,
                       clue_id: int,
                       assignments: Iterable[dict]) -> int:
    """Insert per-word/span role assignments for a solve.

    Each assignment dict carries:
      span_start, span_end, surface_phrase, role,
      optional: mechanism, op, op_kind, value, qualifier,
                db_source, db_table, db_row_id.

    Returns the count of rows inserted.
    """
    n = 0
    for a in assignments:
        conn.execute(
            "INSERT INTO solve_assignments "
            "(solve_id, clue_id, span_start, span_end, surface_phrase, "
            "role, mechanism, op, op_kind, value, qualifier, "
            "db_source, db_table, db_row_id) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (solve_id, clue_id,
             a["span_start"], a["span_end"],
             a["surface_phrase"], a["role"],
             a.get("mechanism"), a.get("op"), a.get("op_kind"),
             a.get("value"), a.get("qualifier"),
             a.get("db_source"), a.get("db_table"),
             a.get("db_row_id")))
        n += 1
    conn.commit()
    return n


def write_seed_failure(conn: sqlite3.Connection,
                        clue_id: int,
                        seed_source: str,
                        failure_kind: str,
                        failure_detail: str,
                        clue_meta: Optional[dict] = None,
                        structured_explanation_id: Optional[int] = None,
                        components_json: Optional[str] = None,
                        blog_text: Optional[str] = None,
                        translated_form: Optional[dict] = None,
                        enrichments: Optional[list] = None,
                        run_number: int = 1) -> int:
    """Record a seeder-failure row.

    `seed_source` is 'structured_explanations' or 'blog'.
    `failure_kind` is 'translation_error' or 'verifier_fail'.
    `clue_meta` carries `source` / `puzzle_number` / `clue_number` /
        `direction` / `clue_text` / `answer` for cheap reading later
        (denormalised on purpose so the failure row stands alone).
    `run_number`: 1=initial cascade, 2=cascade after enrichment,
        3=human review.

    Returns the lastrowid.
    """
    meta = clue_meta or {}
    cur = conn.execute(
        "INSERT INTO seed_failures "
        "(clue_id, seed_source, source, puzzle_number, clue_number, "
        "direction, clue_text, answer, structured_explanation_id, "
        "components_json, blog_text, translated_form_json, "
        "failure_kind, failure_detail, enrichments_json, run_number) "
        "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
        (clue_id, seed_source,
         meta.get("source"), meta.get("puzzle_number"),
         meta.get("clue_number"), meta.get("direction"),
         meta.get("clue_text"), meta.get("answer"),
         structured_explanation_id, components_json, blog_text,
         json.dumps(translated_form, ensure_ascii=False)
            if translated_form else None,
         failure_kind, failure_detail,
         json.dumps(enrichments, ensure_ascii=False)
            if enrichments else None,
         run_number))
    conn.commit()
    return cur.lastrowid


def is_clue_processed(conn: sqlite3.Connection, clue_id: int) -> bool:
    """True if the clue has any solve OR any seed_failure recorded —
    i.e. the seeder has already touched it."""
    row = conn.execute(
        "SELECT 1 FROM solves WHERE clue_id=? LIMIT 1",
        (clue_id,)).fetchone()
    if row:
        return True
    row = conn.execute(
        "SELECT 1 FROM seed_failures WHERE clue_id=? LIMIT 1",
        (clue_id,)).fetchone()
    return row is not None


# --- Composite lookup helpers (live ∪ shadow) ----------------------------

def has_synonym(word: str, value: str,
                 live_conn: sqlite3.Connection,
                 shadow_conn: sqlite3.Connection = None) -> bool:
    word_l = word.lower().strip()
    value_u = value.upper().strip()
    rows = live_conn.execute(
        "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? "
        "AND UPPER(synonym)=? COLLATE NOCASE LIMIT 1",
        (word_l, value_u)).fetchone()
    if rows:
        return True
    if shadow_conn is not None:
        rows = shadow_conn.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? "
            "AND UPPER(synonym)=? COLLATE NOCASE LIMIT 1",
            (word_l, value_u)).fetchone()
        if rows:
            return True
    return False


def has_abbreviation(word: str, value: str,
                      live_conn: sqlite3.Connection,
                      shadow_conn: sqlite3.Connection = None) -> bool:
    word_l = word.lower().strip()
    value_u = value.upper().strip()
    rows = live_conn.execute(
        "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
        "AND UPPER(substitution)=? AND category='abbreviation' "
        "COLLATE NOCASE LIMIT 1",
        (word_l, value_u)).fetchone()
    if rows:
        return True
    if shadow_conn is not None:
        rows = shadow_conn.execute(
            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? "
            "AND UPPER(substitution)=? AND category='abbreviation' "
            "COLLATE NOCASE LIMIT 1",
            (word_l, value_u)).fetchone()
        if rows:
            return True
    return False


def get_indicator_types(word: str,
                         live_conn: sqlite3.Connection,
                         shadow_conn: sqlite3.Connection = None) -> list:
    out = []
    for r in live_conn.execute(
            "SELECT wordplay_type, subtype, confidence FROM indicators "
            "WHERE LOWER(word)=? COLLATE NOCASE",
            (word.lower(),)):
        out.append(tuple(r))
    if shadow_conn is not None:
        for r in shadow_conn.execute(
                "SELECT wordplay_type, subtype, confidence FROM indicators "
                "WHERE LOWER(word)=? COLLATE NOCASE",
                (word.lower(),)):
            out.append(tuple(r))
    return out


def has_definition(definition: str, answer: str,
                    live_conn: sqlite3.Connection,
                    shadow_conn: sqlite3.Connection = None) -> bool:
    d = definition.lower().strip()
    a = answer.upper().strip()
    rows = live_conn.execute(
        "SELECT 1 FROM definition_answers_augmented "
        "WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
        (d, a)).fetchone()
    if rows:
        return True
    if shadow_conn is not None:
        rows = shadow_conn.execute(
            "SELECT 1 FROM definition_answers_augmented "
            "WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (d, a)).fetchone()
        if rows:
            return True
    return False
