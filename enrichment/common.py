"""
Shared infrastructure for enrichment scripts.

All enrichment scripts import from here for DB connections,
insert helpers, and dry-run support.
"""

import argparse
import re
import sqlite3
from pathlib import Path
from typing import Optional

# ============================================================
# DATABASE PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).parent.parent

CRYPTIC_DB = PROJECT_ROOT / 'data' / 'cryptic_new.db'
CLUES_MASTER_DB = PROJECT_ROOT / 'data' / 'clues_master.db'
PIPELINE_DB = PROJECT_ROOT / 'pipeline_stages.db'

# ============================================================
# GLOBALS
# ============================================================

DRY_RUN = False


# ============================================================
# CONNECTION HELPERS
# ============================================================

def get_cryptic_conn() -> sqlite3.Connection:
    """Connection to the reference database (read/write target)."""
    conn = sqlite3.connect(CRYPTIC_DB)
    conn.execute("PRAGMA journal_mode=WAL")
    return conn


def get_clues_conn() -> sqlite3.Connection:
    """Connection to clues_master.db (read-only source)."""
    return sqlite3.connect(CLUES_MASTER_DB)


def get_pipeline_conn() -> sqlite3.Connection:
    """Connection to pipeline_stages.db (read-only source)."""
    return sqlite3.connect(PIPELINE_DB)


# ============================================================
# TEXT HELPERS
# ============================================================

def norm_letters(s: str) -> str:
    """Lowercase, letters only, no spaces."""
    return re.sub(r"[^A-Za-z]", "", s or "").lower()


# ============================================================
# INSERT HELPERS
#
# Each returns True if a new row was inserted, False if duplicate.
# All respect the DRY_RUN flag.
# ============================================================

def insert_wordplay(conn: sqlite3.Connection,
                    indicator: str,
                    substitution: str,
                    category: str,
                    confidence: str = 'medium',
                    notes: str = '',
                    source_tag: str = 'enrichment') -> bool:
    """Insert into wordplay table. Returns True if new row inserted."""
    if not indicator or not substitution:
        return False

    if DRY_RUN:
        existing = conn.execute(
            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND LOWER(substitution)=?",
            (indicator.lower(), substitution.lower())
        ).fetchone()
        return existing is None

    try:
        cur = conn.execute(
            """INSERT INTO wordplay (indicator, substitution, category, frequency, confidence, notes)
               VALUES (?, ?, ?, 0, ?, ?)
               ON CONFLICT(indicator, substitution) DO NOTHING""",
            (indicator.lower(), substitution.upper(), category, confidence,
             f"{source_tag}: {notes}" if notes else source_tag)
        )
        return cur.rowcount > 0
    except sqlite3.IntegrityError:
        return False


def insert_indicator(conn: sqlite3.Connection,
                     word: str,
                     wordplay_type: str,
                     subtype: Optional[str] = None,
                     confidence: str = 'medium',
                     frequency: int = 0,
                     source: Optional[str] = None) -> bool:
    """Insert into indicators table. Returns True if new row inserted.

    Uses existence check instead of ON CONFLICT because SQLite treats
    NULL as distinct in unique constraints, so ON CONFLICT DO NOTHING
    never fires when subtype is NULL.
    """
    if not word or not wordplay_type:
        return False

    # Check existence: if subtype is NULL, also skip if ANY subtype exists
    # for this word+type (avoids redundant NULL entries alongside specific ones)
    if subtype is None:
        existing = conn.execute(
            "SELECT 1 FROM indicators WHERE LOWER(word)=? AND wordplay_type=?",
            (word.lower(), wordplay_type)
        ).fetchone()
    else:
        existing = conn.execute(
            "SELECT 1 FROM indicators WHERE LOWER(word)=? AND wordplay_type=? AND subtype=?",
            (word.lower(), wordplay_type, subtype)
        ).fetchone()
    if existing:
        return False

    if DRY_RUN:
        return True

    conn.execute(
        "INSERT INTO indicators (word, wordplay_type, subtype, confidence, frequency, source) VALUES (?, ?, ?, ?, ?, ?)",
        (word.lower(), wordplay_type, subtype, confidence, frequency, source)
    )
    return True


def insert_synonym_pair(conn: sqlite3.Connection,
                        word: str,
                        synonym: str,
                        source: str = 'enrichment') -> bool:
    """Insert into synonyms_pairs table. Checks for duplicates manually
    since the table has no unique constraint."""
    if not word or not synonym:
        return False

    existing = conn.execute(
        "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND LOWER(synonym)=? LIMIT 1",
        (word.lower(), synonym.lower())
    ).fetchone()
    if existing:
        return False

    if DRY_RUN:
        return True

    conn.execute(
        "INSERT INTO synonyms_pairs (word, synonym, source) VALUES (?, ?, ?)",
        (word.lower(), synonym.lower(), source)
    )
    return True


def insert_homophone(conn: sqlite3.Connection,
                     word: str,
                     sounds_like: str) -> bool:
    """Insert into homophones table. Returns True if new row inserted.

    Note: the homophones table has no source column so these inserts cannot
    be bulk-rolled-back by source tag. Track manually when needed:
      DELETE FROM homophones WHERE word='X' AND homophone='Y'
    """
    if not word or not sounds_like:
        return False

    existing = conn.execute(
        "SELECT 1 FROM homophones WHERE LOWER(word)=? AND LOWER(homophone)=? LIMIT 1",
        (word.lower(), sounds_like.lower())
    ).fetchone()
    if existing:
        return False

    if DRY_RUN:
        return True

    conn.execute(
        "INSERT INTO homophones (word, homophone) VALUES (?, ?)",
        (word.lower(), sounds_like.lower())
    )
    return True


def insert_definition_answer(conn: sqlite3.Connection,
                             definition: str,
                             answer: str,
                             source: str = 'enrichment') -> bool:
    """Insert into both definition_answers and definition_answers_augmented.

    Returns True if genuinely new (not in either table).
    The pipeline reads from definition_answers_augmented, so we must
    write there too. definition_answers tracks frequency for analysis.
    """
    if not definition or not answer:
        return False

    # Check both tables for existence
    in_main = conn.execute(
        "SELECT 1 FROM definition_answers WHERE LOWER(definition)=? AND LOWER(answer)=?",
        (definition.lower(), answer.lower())
    ).fetchone()

    in_augmented = conn.execute(
        "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND LOWER(answer)=?",
        (definition.lower(), answer.lower())
    ).fetchone()

    is_new = not in_main and not in_augmented

    if in_main:
        if not DRY_RUN:
            conn.execute(
                "UPDATE definition_answers SET frequency = frequency + 1 WHERE LOWER(definition)=? AND LOWER(answer)=?",
                (definition.lower(), answer.lower())
            )

    if DRY_RUN:
        return is_new

    if not in_main:
        conn.execute(
            "INSERT INTO definition_answers (definition, answer, source, frequency) VALUES (?, ?, ?, 1)",
            (definition.lower(), answer.lower(), source)
        )

    if not in_augmented:
        conn.execute(
            "INSERT INTO definition_answers_augmented (definition, answer, source) VALUES (?, ?, ?)",
            (definition.lower(), answer.lower(), source)
        )

    return is_new


# ============================================================
# COUNTER HELPER
# ============================================================

class InsertCounter:
    """Tracks inserts per table and logs samples."""

    def __init__(self, script_name: str):
        self.script_name = script_name
        self.counts = {}      # table -> new count
        self.skipped = {}     # table -> duplicate count
        self.samples = {}     # table -> list of sample strings

    def record(self, table: str, inserted: bool, sample: str = ''):
        if table not in self.counts:
            self.counts[table] = 0
            self.skipped[table] = 0
            self.samples[table] = []

        if inserted:
            self.counts[table] += 1
            if len(self.samples[table]) < 10:
                self.samples[table].append(sample)
        else:
            self.skipped[table] += 1

    def report(self):
        mode = "[DRY RUN] " if DRY_RUN else ""
        print(f"\n{'=' * 60}")
        print(f"{mode}{self.script_name} — Results")
        print(f"{'=' * 60}")
        for table in sorted(self.counts.keys()):
            new = self.counts[table]
            dup = self.skipped[table]
            print(f"\n  {table}: {new} new, {dup} duplicates")
            if self.samples[table]:
                print(f"  Samples:")
                for s in self.samples[table]:
                    print(f"    {s}")
        if not self.counts:
            print("  No inserts attempted.")
        print()


# ============================================================
# CLI HELPERS
# ============================================================

def add_common_args(parser: argparse.ArgumentParser):
    """Add --dry-run flag to any enrichment script's parser."""
    parser.add_argument('--dry-run', action='store_true',
                        help='Show what would be added without writing to DB')


def apply_common_args(args):
    """Apply common CLI args (call after parse_args)."""
    global DRY_RUN
    DRY_RUN = args.dry_run
    if DRY_RUN:
        print("[DRY RUN MODE — no writes will be made]")
