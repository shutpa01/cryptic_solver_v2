"""Test the signature solver against a full puzzle from the database."""

import sqlite3
import os
import re
import sys


def extract_wordplay_window(clue_text, definition):
    """Extract the wordplay portion by removing the definition from the clue.

    The definition is typically at the start or end of the clue.
    Returns list of words in the wordplay window.
    """
    clue_clean = clue_text.strip()
    def_clean = definition.strip()

    # Try removing definition from start
    if clue_clean.lower().startswith(def_clean.lower()):
        wordplay = clue_clean[len(def_clean):].strip()
    # Try removing from end
    elif clue_clean.lower().endswith(def_clean.lower()):
        wordplay = clue_clean[:-len(def_clean)].strip()
    else:
        # Definition might be embedded or slightly different — try fuzzy
        # For now, just use the full clue minus first/last word
        wordplay = clue_clean

    # Strip leading/trailing punctuation
    wordplay = wordplay.strip(" ,;:!?'\"")

    if not wordplay:
        return None

    # Split into words, preserving hyphenated words
    words = wordplay.split()
    return words if words else None


def run_puzzle_test(puzzle_number, source='telegraph'):
    from .db import RefDB
    from .solver import solve_clue

    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "clues_master.db"
    )
    conn = sqlite3.connect(db_path, timeout=30)

    rows = conn.execute('''
        SELECT clue_text, answer, definition, wordplay_type,
               clue_number, direction
        FROM clues
        WHERE source = ? AND puzzle_number = ?
        AND answer IS NOT NULL
        ORDER BY clue_number, direction
    ''', (source, puzzle_number)).fetchall()
    conn.close()

    if not rows:
        print(f"No clues found for {source} puzzle {puzzle_number}")
        return

    print(f"Loading reference database...")
    ref_db = RefDB()
    print(f"\nTesting {source} puzzle {puzzle_number}: {len(rows)} clues\n")

    high_conf = 0
    med_conf = 0
    low_conf = 0
    unsolved = 0
    results_list = []

    for clue_text, answer, definition, wtype, clue_num, direction in rows:
        label = f"{clue_num}{direction[0].upper()}"

        answer_clean = answer.upper().replace(" ", "").replace("-", "")
        sr = solve_clue(clue_text, answer_clean, ref_db)

        defn = getattr(sr, 'definition', None) or '?'
        if sr.solved:
            r = sr.result
            conf = sr.confidence
            tier = "HIGH" if sr.high_confidence else "MED " if sr.medium_confidence else "LOW "
            print(f"{tier} [{conf:3d}]  {label}: {answer}")
            print(f"        Def: {defn}")
            print(f"        Sig: {r.signature_str()}")
            print(f"        {r.explanation_parts[0]}")
            for reason, delta in sr.confidence_reasons:
                sign = "+" if delta >= 0 else ""
                print(f"        {sign}{delta} {reason}")
            if wtype:
                print(f"        Expected type: {wtype}")
            print()

            if sr.high_confidence:
                high_conf += 1
            elif sr.medium_confidence:
                med_conf += 1
            else:
                low_conf += 1
        else:
            print(f"NONE [{sr.confidence:3d}]  {label}: {answer} ({wtype or '?'})")
            print(f"        Clue: {clue_text}")
            print(f"        Def candidates: {defn}")
            print()
            unsolved += 1

        results_list.append((label, answer, wtype, sr))

    total = high_conf + med_conf + low_conf + unsolved
    print(f"{'='*60}")
    print(f"Total: {total} clues")
    print(f"  HIGH confidence (80+): {high_conf} — serve directly")
    print(f"  MED  confidence (50-79): {med_conf} — strong API evidence")
    print(f"  LOW  confidence (<50): {low_conf} — weak/discard")
    print(f"  UNSOLVED: {unsolved} — need API")
    print(f"\nMechanical solve rate: {high_conf}/{total} ({100*high_conf/total:.0f}%) high-confidence")
    print(f"Total found: {high_conf+med_conf+low_conf}/{total} ({100*(high_conf+med_conf+low_conf)/total:.0f}%)")


if __name__ == "__main__":
    puzzle = int(sys.argv[1]) if len(sys.argv) > 1 else 31174
    source = sys.argv[2] if len(sys.argv) > 2 else 'telegraph'
    run_puzzle_test(puzzle, source)
