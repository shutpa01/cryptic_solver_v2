"""Stage 6: Test the signature solver against parsed explanations.

For each verified parse from the Times explanations, check if the solver
can mechanically produce the answer. This tells us the TRUE solve rate
on real clues, not just unit tests.

We test on clues from clues_master.db that also have verified explanations
in times_explanations.db — this gives us the full clue text + definition
+ answer + known decomposition.
"""

import sqlite3
import sys
import time
from collections import Counter

sys.path.insert(0, '.')


def load_test_set(limit=None):
    """Load clues that exist in both clues_master and times_explanations.

    Returns list of (clue_text, answer, definition, explanation, puzzle_number)
    """
    # Get explanations with verified parses
    expl_conn = sqlite3.connect('data/times_explanations.db')
    clue_conn = sqlite3.connect('data/clues_master.db')

    # Get all times clues from clues_master that have explanations
    # Match on answer (uppercase) — not perfect but good enough
    query = """
        SELECT cm.clue, cm.answer, cm.definition, te.explanation, te.puzzle_number
        FROM (SELECT clue_text, answer, definition FROM clues
              WHERE source = 'times' AND definition IS NOT NULL
              AND definition != '' AND clue_text IS NOT NULL) cm
        JOIN (SELECT answer, explanation, puzzle_number FROM clues
              WHERE explanation IS NOT NULL AND explanation != '') te
        ON UPPER(cm.answer) = UPPER(te.answer)
    """

    # This cross-DB join won't work directly. Do it in Python.
    # Get times clues from clues_master
    master_rows = clue_conn.execute("""
        SELECT clue_text, answer, definition FROM clues
        WHERE source = 'times' AND definition IS NOT NULL
        AND definition != '' AND clue_text IS NOT NULL
    """).fetchall()
    clue_conn.close()

    # Index by answer
    master_by_answer = {}
    for clue, answer, defn in master_rows:
        key = answer.upper().strip()
        if key not in master_by_answer:
            master_by_answer[key] = (clue, answer, defn)

    # Get verified explanations
    expl_rows = expl_conn.execute("""
        SELECT answer, explanation, puzzle_number FROM clues
        WHERE explanation IS NOT NULL AND explanation != ''
    """).fetchall()
    expl_conn.close()

    # Join
    test_set = []
    for answer, explanation, puzzle_number in expl_rows:
        key = answer.upper().strip()
        if key in master_by_answer:
            clue, ans, defn = master_by_answer[key]
            test_set.append((clue, ans, defn, explanation, puzzle_number))

    if limit:
        # Take a representative sample
        import random
        random.seed(42)
        random.shuffle(test_set)
        test_set = test_set[:limit]

    return test_set


def extract_wordplay_window(clue_text, definition):
    """Extract wordplay words from clue by removing the definition."""
    if not definition or not clue_text:
        return None

    clue_lower = clue_text.lower().strip()
    defn_lower = definition.lower().strip()

    # Try removing definition from start or end
    if clue_lower.startswith(defn_lower):
        remainder = clue_text[len(definition):].strip()
    elif clue_lower.endswith(defn_lower):
        remainder = clue_text[:len(clue_text) - len(definition)].strip()
    else:
        return None

    # Clean and split
    import re
    remainder = remainder.strip('.,;:!? ')
    if not remainder:
        return None

    words = remainder.split()
    return [w for w in words if w]


def run_test(limit=2000):
    """Run the solver against test clues and measure performance."""
    from signature_solver.db import RefDB
    from signature_solver.solver import solve

    print("Loading test set...")
    test_set = load_test_set(limit=limit)
    print(f"Test set: {len(test_set)} clues")

    print("Loading reference database...")
    db = RefDB()

    # Parse explanations to know expected operations
    from cryptic_taxonomy.analysis.notation_parser import parse_explanation

    results = Counter()
    by_op = Counter()  # expected operation → (solved, total)
    by_op_solved = Counter()
    total_time = 0
    solved_examples = []
    failed_examples = []

    for i, (clue, answer, definition, explanation, puzzle_num) in enumerate(test_set):
        if i % 200 == 0 and i > 0:
            print(f"  ...{i}/{len(test_set)} ({100*results['solved']/i:.1f}% solved)")

        # Get expected operation from parser
        clean_answer = answer.upper().replace(' ', '').replace('-', '')
        parse_result = parse_explanation(explanation, clean_answer)
        expected_op = parse_result.operation

        # Extract wordplay window
        wp_words = extract_wordplay_window(clue, definition)
        if not wp_words:
            results['skip_no_window'] += 1
            continue

        by_op[expected_op] += 1

        # Run solver
        t0 = time.time()
        try:
            solve_result = solve(wp_words, clean_answer, db)
            elapsed = time.time() - t0
            total_time += elapsed

            if solve_result.solved:
                results['solved'] += 1
                by_op_solved[expected_op] += 1
                if solve_result.high_confidence:
                    results['high'] += 1
                elif solve_result.medium_confidence:
                    results['medium'] += 1
                else:
                    results['low'] += 1

                if len(solved_examples) < 5:
                    solved_examples.append((
                        answer, clue, solve_result.confidence,
                        expected_op, solve_result.result.signature_label
                        if hasattr(solve_result.result, 'signature_label') else '?'
                    ))
            else:
                results['unsolved'] += 1
                if len(failed_examples) < 10 and parse_result.verified:
                    failed_examples.append((
                        answer, clue, expected_op, explanation[:80]
                    ))
        except Exception as e:
            results['error'] += 1
            if results['error'] <= 3:
                print(f"  ERROR on {answer}: {e}")
            total_time += time.time() - t0

    # Report
    tested = sum(by_op.values())
    solved = results['solved']
    print(f"\n{'='*60}")
    print(f"SOLVER TEST RESULTS")
    print(f"{'='*60}")
    print(f"Tested: {tested} clues")
    print(f"Solved: {solved} ({100*solved/tested:.1f}%)")
    print(f"  HIGH (80+): {results.get('high', 0)}")
    print(f"  MED (50-79): {results.get('medium', 0)}")
    print(f"  LOW (<50): {results.get('low', 0)}")
    print(f"Unsolved: {results.get('unsolved', 0)}")
    print(f"Errors: {results.get('error', 0)}")
    print(f"Skipped (no window): {results.get('skip_no_window', 0)}")
    print(f"Avg time: {1000*total_time/tested:.0f}ms/clue")

    print(f"\nSolve rate by expected operation:")
    print(f"{'Operation':25s} {'Solved':>7s} {'Total':>7s} {'Rate':>7s}")
    print('-' * 50)
    for op in sorted(by_op.keys(), key=lambda x: -by_op[x]):
        t = by_op[op]
        s = by_op_solved.get(op, 0)
        rate = 100 * s / t if t else 0
        print(f"  {op:23s} {s:7d} {t:7d} {rate:5.1f}%")

    if failed_examples:
        print(f"\nSample failures (verified explanations the solver missed):")
        for answer, clue, expected_op, explanation in failed_examples:
            print(f"  {answer:15s} [{expected_op:20s}] {clue[:60]}")
            print(f"    Explanation: {explanation}")


if __name__ == '__main__':
    import sys
    limit = int(sys.argv[1]) if len(sys.argv) > 1 else 2000
    run_test(limit=limit)
