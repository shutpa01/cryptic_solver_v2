"""Test the full signature solver pipeline: mechanical + API fallback.

Runs mechanical solver first, then calls API only for non-HIGH results.
Shows what the API receives and returns.
"""

import sqlite3
import os
import sys
import time

from .db import RefDB
from .solver import solve
from .api_solver import api_solve, validate_api_result
from .test_puzzle import extract_wordplay_window


def run_api_test(puzzle_number, source='telegraph', show_evidence=False):
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
        AND answer IS NOT NULL AND definition IS NOT NULL
        ORDER BY clue_number, direction
    ''', (source, puzzle_number)).fetchall()
    conn.close()

    if not rows:
        print(f"No clues found for {source} puzzle {puzzle_number}")
        return

    print("Loading reference database...")
    ref_db = RefDB()
    print(f"\nFull pipeline test: {source} puzzle {puzzle_number} ({len(rows)} clues)\n")

    stats = {
        'mechanical_high': 0,
        'api_solved': 0,
        'api_failed': 0,
        'skipped': 0,
        'total_api_in': 0,
        'total_api_out': 0,
    }

    for clue_text, answer, definition, wtype, clue_num, direction in rows:
        label = f"{clue_num}{direction[0].upper()}"
        answer_clean = answer.upper().replace(" ", "").replace("-", "")

        wordplay_words = extract_wordplay_window(clue_text, definition)
        if not wordplay_words:
            print(f"SKIP    {label}: Could not extract wordplay window")
            print()
            stats['skipped'] += 1
            continue

        # Step 1: Mechanical solve
        sr = solve(wordplay_words, answer_clean, ref_db)

        if sr.high_confidence:
            r = sr.result
            print(f"MECH [{sr.confidence:3d}]  {label}: {answer}")
            print(f"        {r.signature_str()}")
            print(f"        {r.explanation_parts[0]}")
            print()
            stats['mechanical_high'] += 1
            continue

        # Step 2: API fallback
        print(f"API  [{sr.confidence:3d}]  {label}: {answer} ({wtype or '?'})")
        print(f"        Clue: {clue_text}")

        if show_evidence:
            from .evidence import format_evidence
            evidence = format_evidence(
                sr.analyses, sr.phrases, answer_clean,
                clue_text=clue_text, definition=definition,
            )
            print(f"        --- Evidence sent to API ---")
            for line in evidence.split('\n'):
                print(f"        {line}")
            print(f"        --- End evidence ---")

        try:
            parsed, tok_in, tok_out = api_solve(
                clue_text, answer, definition, sr, ref_db
            )
            stats['total_api_in'] += tok_in
            stats['total_api_out'] += tok_out

            if parsed:
                is_valid, reason = validate_api_result(parsed, answer, db=ref_db)
                wt = parsed.get('wordplay_type', '?')
                assembly = parsed.get('assembly', '')
                conf = parsed.get('confidence', '?')
                pieces_str = _format_pieces(parsed.get('pieces', []))

                status = "OK" if is_valid else "BAD"
                # Sanitize unicode for Windows console
                assembly = assembly.encode('ascii', 'replace').decode('ascii')
                print(f"        API [{status}] type={wt} conf={conf}")
                print(f"        {assembly}")
                print(f"        {pieces_str}")
                if not is_valid:
                    reason_safe = reason.encode('ascii', 'replace').decode('ascii')
                    print(f"        Validation: {reason_safe}")
                print(f"        Tokens: {tok_in}in/{tok_out}out")
                print()

                if is_valid:
                    stats['api_solved'] += 1
                else:
                    stats['api_failed'] += 1
            else:
                print(f"        API returned no valid JSON")
                print(f"        Tokens: {tok_in}in/{tok_out}out")
                print()
                stats['api_failed'] += 1

        except Exception as e:
            print(f"        API ERROR: {e}")
            print()
            stats['api_failed'] += 1

        # Small delay to avoid rate limiting
        time.sleep(0.3)

    total = stats['mechanical_high'] + stats['api_solved'] + stats['api_failed']
    api_calls = stats['api_solved'] + stats['api_failed']
    print(f"{'='*60}")
    print(f"Results ({total} clues, {stats['skipped']} skipped):")
    print(f"  Mechanical HIGH: {stats['mechanical_high']} (zero cost)")
    print(f"  API solved:      {stats['api_solved']}/{api_calls} calls")
    print(f"  API failed:      {stats['api_failed']}/{api_calls} calls")
    print(f"  Total solved:    {stats['mechanical_high'] + stats['api_solved']}/{total}"
          f" ({100*(stats['mechanical_high'] + stats['api_solved'])/total:.0f}%)")
    print(f"\nAPI token usage: {stats['total_api_in']} in / {stats['total_api_out']} out")
    if api_calls:
        print(f"Average per call: {stats['total_api_in']//api_calls} in / "
              f"{stats['total_api_out']//api_calls} out")


def _format_pieces(pieces):
    """Format pieces array as a compact string."""
    parts = []
    for p in pieces:
        word = p.get('word', '?')
        letters = p.get('letters', '')
        role = p.get('role', '?')
        if letters:
            parts.append(f"{word}={letters}({role})")
        else:
            parts.append(f"{word}({role})")
    return " + ".join(parts)


if __name__ == "__main__":
    puzzle = int(sys.argv[1]) if len(sys.argv) > 1 else 31171
    source = sys.argv[2] if len(sys.argv) > 2 else 'telegraph'
    show_ev = '--evidence' in sys.argv
    run_api_test(puzzle, source, show_evidence=show_ev)
