#!/usr/bin/env python3
"""Generate Cordelia's Daily Mash-up puzzle.

Takes a real published puzzle's grid and answers, replaces each clue with an
alternative from a different source that gives the same answer. Balances
wordplay types, prefers clues with existing explanations, and tracks history
to avoid repetition.

Usage:
    python scripts/generate_mashup.py                          # today, auto-select base
    python scripts/generate_mashup.py --date 2026-04-14        # specific date
    python scripts/generate_mashup.py --base-source telegraph --base-puzzle 31200
    python scripts/generate_mashup.py --dry-run                # preview without DB writes
"""

import argparse
import hashlib
import random
import sqlite3
import sys
from datetime import date
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DB_PATH = PROJECT_ROOT / "data" / "clues_master.db"

ELIGIBLE_SOURCES = ["telegraph", "guardian", "times", "independent", "dailymail"]

# Soft targets for wordplay diversity (not enforced, just scoring bonuses/penalties)
TARGET_RATIOS = {
    "charade": 0.30,
    "anagram": 0.25,
    "container": 0.10,
    "double_definition": 0.08,
    "hidden": 0.07,
    "reversal": 0.06,
    "deletion": 0.05,
    "homophone": 0.03,
    "initial_letters": 0.03,
    "cryptic_definition": 0.03,
}


def _ensure_tables(conn):
    """Create mashup tracking tables if they don't exist."""
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mashup_puzzles (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mashup_number TEXT NOT NULL UNIQUE,
            base_source TEXT NOT NULL,
            base_puzzle_number TEXT NOT NULL,
            publication_date TEXT NOT NULL,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
            seed INTEGER NOT NULL,
            clue_count INTEGER,
            replaced_count INTEGER,
            UNIQUE(base_source, base_puzzle_number)
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS mashup_clue_selections (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            mashup_number TEXT NOT NULL,
            slot_clue_number TEXT NOT NULL,
            slot_direction TEXT NOT NULL,
            answer TEXT NOT NULL,
            original_clue_id INTEGER NOT NULL,
            selected_clue_id INTEGER,
            selected_source TEXT,
            wordplay_type TEXT,
            had_explanation INTEGER DEFAULT 0,
            candidate_count INTEGER DEFAULT 0,
            UNIQUE(mashup_number, slot_clue_number, slot_direction)
        )
    """)
    conn.commit()


def _next_mashup_number(conn):
    """Get the next mashup number (starts at 1001)."""
    row = conn.execute(
        "SELECT MAX(CAST(mashup_number AS INTEGER)) FROM mashup_puzzles"
    ).fetchone()
    current_max = row[0] if row[0] is not None else 1000
    return str(current_max + 1)


def _select_base_puzzle(conn, rng, preferred_source=None):
    """Auto-select a base puzzle that has a grid and hasn't been used."""
    used = set()
    for row in conn.execute(
        "SELECT base_source, base_puzzle_number FROM mashup_puzzles"
    ).fetchall():
        used.add((row[0], row[1]))

    source = preferred_source
    if not source:
        source = ELIGIBLE_SOURCES[rng.randint(0, len(ELIGIBLE_SOURCES) - 1)]

    # Find puzzles with grids, all answers filled, all directions set, not previously used
    candidates = conn.execute("""
        SELECT pg.source, pg.puzzle_number,
               COUNT(c.id) AS total,
               SUM(CASE WHEN c.answer IS NOT NULL AND c.answer != '' THEN 1 ELSE 0 END) AS with_answer,
               SUM(CASE WHEN c.direction IS NULL THEN 1 ELSE 0 END) AS null_dir
        FROM puzzle_grids pg
        JOIN clues c ON c.source = pg.source AND c.puzzle_number = pg.puzzle_number
        WHERE pg.source = ?
          AND c.source != 'cordelia'
        GROUP BY pg.source, pg.puzzle_number
        HAVING with_answer = total AND total >= 20 AND null_dir = 0
        ORDER BY c.publication_date DESC
        LIMIT 200
    """, (source,)).fetchall()

    # Filter out previously used
    candidates = [(s, p) for s, p, _t, _a, _d in candidates if (s, p) not in used]

    if not candidates:
        # Try other sources
        for fallback in ELIGIBLE_SOURCES:
            if fallback == source:
                continue
            candidates = conn.execute("""
                SELECT pg.source, pg.puzzle_number,
                       COUNT(c.id) AS total,
                       SUM(CASE WHEN c.answer IS NOT NULL AND c.answer != '' THEN 1 ELSE 0 END) AS with_answer,
                       SUM(CASE WHEN c.direction IS NULL THEN 1 ELSE 0 END) AS null_dir
                FROM puzzle_grids pg
                JOIN clues c ON c.source = pg.source AND c.puzzle_number = pg.puzzle_number
                WHERE pg.source = ?
                  AND c.source != 'cordelia'
                GROUP BY pg.source, pg.puzzle_number
                HAVING with_answer = total AND total >= 20 AND null_dir = 0
                ORDER BY c.publication_date DESC
                LIMIT 200
            """, (fallback,)).fetchall()
            candidates = [(s, p) for s, p, _t, _a, _d in candidates if (s, p) not in used]
            if candidates:
                break

    if not candidates:
        return None, None

    return rng.choice(candidates)


def _normalise_answer(answer):
    """Normalise answer for matching: uppercase, no spaces."""
    if not answer:
        return ""
    return answer.upper().replace(" ", "")


def _primary_type(wordplay_type):
    """Extract primary wordplay type from potentially compound value."""
    if not wordplay_type:
        return "unknown"
    return wordplay_type.split(",")[0].split("/")[0].strip().lower()


def _diversity_modifier(primary_type, type_counts, total_selected):
    """Score modifier based on how over/underrepresented this type is."""
    if total_selected == 0:
        return 0
    target = TARGET_RATIOS.get(primary_type, 0.02)
    current_ratio = type_counts.get(primary_type, 0) / total_selected
    if current_ratio < target:
        return 20  # underrepresented, boost
    elif current_ratio > target * 2:
        return -40  # overrepresented, penalise
    return 0


def _score_candidate(candidate, type_counts, total_selected, recent_sources, rng):
    """Score a candidate clue for selection."""
    score = 0.0

    # Explanation tier
    has_human = (candidate["explanation"] is not None
                 and candidate["explanation"] != ""
                 and candidate["source"] != "telegraph")
    has_structured = candidate["has_structured"]

    if has_human:
        score += 100
    elif has_structured:
        score += 70
    else:
        score += 20

    # Wordplay diversity
    pt = _primary_type(candidate["wordplay_type"])
    score += _diversity_modifier(pt, type_counts, total_selected)

    # Source variety — bonus if different from last 3
    if candidate["source"] not in recent_sources:
        score += 10

    # Jitter
    score += rng.uniform(-5, 5)

    return score, has_human or has_structured


def generate_mashup(target_date, base_source=None, base_puzzle_number=None, dry_run=False):
    """Generate a mash-up puzzle for the given date."""
    conn = sqlite3.connect(str(DB_PATH), timeout=30)
    conn.row_factory = sqlite3.Row

    _ensure_tables(conn)

    # Seed from date
    seed = int(hashlib.sha256(target_date.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    # Check if mashup already exists for this date
    existing = conn.execute(
        "SELECT mashup_number FROM mashup_puzzles WHERE publication_date = ?",
        (target_date,)
    ).fetchone()
    if existing:
        print(f"Mashup already exists for {target_date}: #{existing['mashup_number']}")
        conn.close()
        return None

    # Select base puzzle
    if base_source and base_puzzle_number:
        bp_source, bp_number = base_source, base_puzzle_number
    else:
        bp_source, bp_number = _select_base_puzzle(conn, rng, base_source)

    if not bp_source:
        print("ERROR: No eligible base puzzle found.")
        conn.close()
        return None

    print(f"Base puzzle: {bp_source} #{bp_number}")

    # Load base clues
    base_clues = conn.execute("""
        SELECT id, clue_number, direction, clue_text, answer, enumeration,
               definition, wordplay_type, ai_explanation, explanation
        FROM clues
        WHERE source = ? AND puzzle_number = ?
        ORDER BY CASE direction WHEN 'across' THEN 0 ELSE 1 END,
                 CAST(clue_number AS INTEGER)
    """, (bp_source, bp_number)).fetchall()

    if not base_clues:
        print(f"ERROR: No clues found for {bp_source} #{bp_number}")
        conn.close()
        return None

    print(f"Base puzzle has {len(base_clues)} clues")

    # Load history of previously used clue IDs
    used_clue_ids = set()
    for row in conn.execute(
        "SELECT selected_clue_id FROM mashup_clue_selections WHERE selected_clue_id IS NOT NULL"
    ).fetchall():
        used_clue_ids.add(row[0])

    # Selection loop
    selections = []
    type_counts = {}
    total_selected = 0
    recent_sources = []
    replaced = 0
    already_explained = 0

    for clue in base_clues:
        answer = clue["answer"]
        norm_answer = _normalise_answer(answer)

        # Skip spanning clues
        if clue["clue_text"] and clue["clue_text"].strip().startswith("See "):
            selections.append({
                "clue": clue,
                "selected": None,
                "kept_original": True,
                "reason": "spanning",
                "candidate_count": 0,
            })
            continue

        # Query candidates
        candidates = conn.execute("""
            SELECT c.id, c.source, c.puzzle_number, c.clue_text, c.wordplay_type,
                   c.definition, c.ai_explanation, c.explanation,
                   se.components IS NOT NULL AS has_structured,
                   se.confidence
            FROM clues c
            LEFT JOIN structured_explanations se ON se.clue_id = c.id
            WHERE UPPER(REPLACE(c.answer, ' ', '')) = ?
              AND c.id != ?
              AND c.source != 'cordelia'
              AND c.clue_text IS NOT NULL
              AND LENGTH(c.clue_text) > 5
              AND c.clue_text NOT LIKE 'See %'
              AND c.clue_text NOT GLOB '*[0-9]*'
        """, (norm_answer, clue["id"])).fetchall()

        # Exclude history
        candidates = [c for c in candidates if c["id"] not in used_clue_ids]

        if not candidates:
            selections.append({
                "clue": clue,
                "selected": None,
                "kept_original": True,
                "reason": "no alternatives",
                "candidate_count": 0,
            })
            continue

        # Score and select
        best_score = -999
        best_candidate = None
        best_has_expl = False

        for cand in candidates:
            sc, has_expl = _score_candidate(
                cand, type_counts, total_selected, recent_sources[-3:], rng
            )
            if sc > best_score:
                best_score = sc
                best_candidate = cand
                best_has_expl = has_expl

        # Record selection
        pt = _primary_type(best_candidate["wordplay_type"])
        type_counts[pt] = type_counts.get(pt, 0) + 1
        total_selected += 1
        recent_sources.append(best_candidate["source"])
        replaced += 1
        if best_has_expl:
            already_explained += 1

        selections.append({
            "clue": clue,
            "selected": best_candidate,
            "kept_original": False,
            "reason": None,
            "candidate_count": len(candidates),
        })

    # Report
    kept = len(selections) - replaced
    needs_pipeline = replaced - already_explained

    print(f"\n{'=' * 60}")
    print(f"MASHUP SUMMARY for {target_date}")
    print(f"{'=' * 60}")
    print(f"  Base: {bp_source} #{bp_number}")
    print(f"  Clues: {len(selections)}")
    print(f"  Replaced: {replaced} ({kept} kept original)")
    print(f"  Already explained: {already_explained} (free)")
    print(f"  Needs pipeline: {needs_pipeline}")
    print(f"\n  Wordplay distribution:")
    for wt, count in sorted(type_counts.items(), key=lambda x: -x[1]):
        print(f"    {wt:<25} {count:>3}")

    # Show each selection
    print(f"\n{'=' * 60}")
    print(f"{'#':>4} {'Answer':<20} {'Source':<12} {'Type':<18} {'Cands':>5} {'Expl':>4}")
    print(f"{'-' * 70}")
    for sel in selections:
        c = sel["clue"]
        d = "a" if c["direction"] == "across" else "d"
        if sel["kept_original"]:
            reason = sel["reason"] or "kept"
            print(f"{c['clue_number']:>3}{d} {c['answer'] or '???':<20} {'(original)':<12} {reason:<18} {sel['candidate_count']:>5}")
        else:
            s = sel["selected"]
            has_e = "Y" if (s["explanation"] and s["source"] != "telegraph") or s["has_structured"] else "N"
            print(f"{c['clue_number']:>3}{d} {c['answer'] or '???':<20} {s['source']:<12} {_primary_type(s['wordplay_type']):<18} {sel['candidate_count']:>5} {has_e:>4}")

    if dry_run:
        print(f"\n  DRY RUN — no changes written to DB")
        conn.close()
        return None

    # Write to DB
    mashup_number = _next_mashup_number(conn)
    print(f"\n  Writing mashup #{mashup_number} to DB...")

    # Insert clues
    for sel in selections:
        base = sel["clue"]
        if sel["kept_original"]:
            # Copy original clue
            conn.execute("""
                INSERT INTO clues (source, puzzle_number, publication_date,
                    clue_number, direction, clue_text, enumeration, answer,
                    definition, wordplay_type, ai_explanation, explanation)
                VALUES ('cordelia', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mashup_number, target_date,
                base["clue_number"], base["direction"], base["clue_text"],
                base["enumeration"], base["answer"],
                base["definition"], base["wordplay_type"],
                base["ai_explanation"], base["explanation"],
            ))
        else:
            alt = sel["selected"]
            conn.execute("""
                INSERT INTO clues (source, puzzle_number, publication_date,
                    clue_number, direction, clue_text, enumeration, answer,
                    definition, wordplay_type, ai_explanation, explanation)
                VALUES ('cordelia', ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                mashup_number, target_date,
                base["clue_number"], base["direction"], alt["clue_text"],
                base["enumeration"], base["answer"],
                alt["definition"], alt["wordplay_type"],
                alt["ai_explanation"], alt["explanation"],
            ))

    # Copy structured_explanations for selected clues that have them
    for sel in selections:
        if sel["kept_original"]:
            src_clue_id = sel["clue"]["id"]
        elif sel["selected"]:
            src_clue_id = sel["selected"]["id"]
        else:
            continue

        # Get the new clue ID we just inserted
        base = sel["clue"]
        new_row = conn.execute(
            "SELECT id FROM clues WHERE source = 'cordelia' AND puzzle_number = ? "
            "AND clue_number = ? AND direction = ?",
            (mashup_number, base["clue_number"], base["direction"])
        ).fetchone()
        if not new_row:
            continue
        new_clue_id = new_row["id"]

        # Copy structured_explanation if it exists
        se = conn.execute(
            "SELECT * FROM structured_explanations WHERE clue_id = ?",
            (src_clue_id,)
        ).fetchone()
        if se:
            conn.execute("""
                INSERT OR IGNORE INTO structured_explanations
                    (clue_id, definition_text, definition_start, definition_end,
                     wordplay_types, components, model_version, confidence,
                     source, puzzle_number, clue_number)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, 'cordelia', ?, ?)
            """, (
                new_clue_id,
                se["definition_text"], se["definition_start"], se["definition_end"],
                se["wordplay_types"], se["components"], se["model_version"],
                se["confidence"], mashup_number, base["clue_number"],
            ))

    # Copy grid
    conn.execute("""
        INSERT OR IGNORE INTO puzzle_grids (source, puzzle_number, solution, grid_rows, grid_cols)
        SELECT 'cordelia', ?, solution, grid_rows, grid_cols
        FROM puzzle_grids WHERE source = ? AND puzzle_number = ?
    """, (mashup_number, bp_source, bp_number))

    # Record mashup metadata
    conn.execute("""
        INSERT INTO mashup_puzzles
            (mashup_number, base_source, base_puzzle_number, publication_date, seed,
             clue_count, replaced_count)
        VALUES (?, ?, ?, ?, ?, ?, ?)
    """, (mashup_number, bp_source, bp_number, target_date, seed,
          len(selections), replaced))

    # Record clue selections
    for sel in selections:
        base = sel["clue"]
        alt = sel["selected"]
        had_expl = 0
        if alt:
            has_human = (alt["explanation"] and alt["source"] != "telegraph")
            had_expl = 1 if (has_human or alt["has_structured"]) else 0
        conn.execute("""
            INSERT INTO mashup_clue_selections
                (mashup_number, slot_clue_number, slot_direction, answer,
                 original_clue_id, selected_clue_id, selected_source,
                 wordplay_type, had_explanation, candidate_count)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            mashup_number, base["clue_number"], base["direction"],
            base["answer"], base["id"],
            alt["id"] if alt else None,
            alt["source"] if alt else None,
            alt["wordplay_type"] if alt else None,
            had_expl, sel["candidate_count"],
        ))

    conn.commit()
    conn.close()

    print(f"  Done! Mashup #{mashup_number} written to DB.")
    print(f"  View at: /cordelia/daily-mashup/{mashup_number}")
    if needs_pipeline > 0:
        print(f"  Run pipeline: --source cordelia {mashup_number}")

    return mashup_number


def main():
    parser = argparse.ArgumentParser(description="Generate Cordelia's Daily Mash-up")
    parser.add_argument("--date", default=date.today().isoformat(),
                        help="Target date (YYYY-MM-DD, default: today)")
    parser.add_argument("--base-source", default=None, help="Force base puzzle source")
    parser.add_argument("--base-puzzle", default=None, help="Force base puzzle number")
    parser.add_argument("--dry-run", action="store_true", help="Preview without writing to DB")
    args = parser.parse_args()

    generate_mashup(args.date, args.base_source, args.base_puzzle, args.dry_run)


if __name__ == "__main__":
    main()
