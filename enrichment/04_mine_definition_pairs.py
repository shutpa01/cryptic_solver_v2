"""
04_mine_definition_pairs.py — Extract definition-answer pairs from full clue corpus.

In cryptic crosswords, the definition is typically the first or last 1-3 words
of the clue. By counting how often the same (definition_window, answer) pair
appears across the full corpus, we can identify high-confidence definition pairs.

A pair that appears 3+ times across different clues is very likely correct.

Source: 500k+ clues in clues_master.db + 327k in cryptic_new.db.
Writes to: definition_answers + definition_answers_augmented in cryptic_new.db.
"""

import argparse
import re
import sqlite3
from collections import Counter

from enrichment.common import (
    get_cryptic_conn, get_clues_conn,
    InsertCounter, add_common_args, apply_common_args, DRY_RUN,
)

SOURCE_TAG = 'corpus_definition'

# Words too common to be useful as single-word definitions
SKIP_SINGLE = {
    'the', 'a', 'an', 'is', 'in', 'of', 'to', 'for', 'and', 'or',
    'it', 'on', 'at', 'by', 'be', 'as', 'if', 'so', 'no', 'do',
    'up', 'he', 'she', 'we', 'me', 'my', 'his', 'her', 'its',
    'not', 'but', 'all', 'are', 'was', 'has', 'had', 'who', 'how',
    'this', 'that', 'with', 'from', 'have',
}

# Minimum frequency to record a pair
MIN_FREQ = 3


def clean_clue_text(clue_text: str) -> str:
    """Remove enumeration and clean up clue text."""
    text = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text)
    # Remove trailing/leading punctuation and whitespace
    text = re.sub(r'[^\w\s]', ' ', text)  # Replace punctuation with spaces
    text = ' '.join(text.split())  # Normalize whitespace
    return text


def extract_windows(clue_text: str, max_words: int = 3) -> list:
    """Extract 1-N word windows from start and end of clue text."""
    text = clean_clue_text(clue_text)
    if not text:
        return []

    words = text.split()
    if len(words) < 2:
        return []

    windows = []
    for n in range(1, min(max_words + 1, len(words))):
        start_window = ' '.join(words[:n]).lower()
        if n == 1 and start_window in SKIP_SINGLE:
            continue
        windows.append(start_window)

        end_window = ' '.join(words[-n:]).lower()
        if n == 1 and end_window in SKIP_SINGLE:
            continue
        windows.append(end_window)

    return windows


def load_existing_pairs(conn: sqlite3.Connection) -> set:
    """Load all existing (definition, answer) pairs into memory for fast lookup."""
    existing = set()

    print("  Loading definition_answers...")
    for row in conn.execute("SELECT LOWER(definition), LOWER(answer) FROM definition_answers"):
        existing.add(row)

    print("  Loading definition_answers_augmented...")
    for row in conn.execute("SELECT LOWER(definition), LOWER(answer) FROM definition_answers_augmented"):
        existing.add(row)

    print(f"  Total existing pairs in memory: {len(existing)}")
    return existing


def main():
    parser = argparse.ArgumentParser(description='Mine definition-answer pairs from clue corpus')
    add_common_args(parser)
    parser.add_argument('--min-freq', type=int, default=MIN_FREQ,
                        help=f'Minimum frequency to record (default: {MIN_FREQ})')
    parser.add_argument('--max-words', type=int, default=3,
                        help='Maximum definition window size in words (default: 3)')
    args = parser.parse_args()
    apply_common_args(args)

    clues_conn = get_clues_conn()
    cryptic_conn = get_cryptic_conn()
    counter = InsertCounter('04_mine_definition_pairs')

    # Load all clues with answers from both databases
    print("Loading clues...")
    rows = clues_conn.execute("""
        SELECT clue_text, answer FROM clues
        WHERE answer IS NOT NULL AND answer != ''
          AND clue_text IS NOT NULL AND clue_text != ''
    """).fetchall()
    print(f"  clues_master.db: {len(rows)} clues")

    rows2 = cryptic_conn.execute("""
        SELECT clue_text, answer FROM clues
        WHERE answer IS NOT NULL AND answer != ''
          AND clue_text IS NOT NULL AND clue_text != ''
    """).fetchall()
    print(f"  cryptic_new.db: {len(rows2)} clues")

    # Combine and deduplicate
    all_clues = set()
    for clue_text, answer in rows + rows2:
        answer_clean = answer.upper().replace(' ', '')
        if len(answer_clean) >= 2:
            all_clues.add((clue_text, answer_clean))
    print(f"  Unique (clue, answer) pairs: {len(all_clues)}")

    # Generate candidate pairs and count frequencies
    print("\nGenerating definition windows...")
    pair_counts = Counter()
    for clue_text, answer in all_clues:
        windows = extract_windows(clue_text, args.max_words)
        for window in windows:
            if len(window) >= 2:
                pair_counts[(window, answer.lower())] += 1

    valid_pairs = {k: v for k, v in pair_counts.items() if v >= args.min_freq}
    print(f"Candidate pairs (freq >= {args.min_freq}): {len(valid_pairs)}")

    # Load existing pairs into memory for fast deduplication
    print("\nLoading existing pairs for deduplication...")
    existing = load_existing_pairs(cryptic_conn)

    # Filter to only new pairs
    new_pairs = {k: v for k, v in valid_pairs.items() if k not in existing}
    print(f"New pairs to insert: {len(new_pairs)}")

    if args.dry_run:
        # Show samples and exit
        print("\n[DRY RUN] Sample new pairs:")
        for (defn, answer), freq in sorted(new_pairs.items(), key=lambda x: -x[1])[:30]:
            print(f"  '{defn}' -> {answer.upper()} ({freq}x)")
        counter.counts['definition_answers'] = len(new_pairs)
        counter.skipped['definition_answers'] = len(valid_pairs) - len(new_pairs)
        counter.samples['definition_answers'] = [
            f"'{d}' -> {a.upper()} ({f}x)"
            for (d, a), f in sorted(new_pairs.items(), key=lambda x: -x[1])[:10]
        ]
        counter.report()
    else:
        # Batch insert
        print("\nInserting new pairs...")
        batch_main = []
        batch_aug = []
        for (defn, answer), freq in new_pairs.items():
            batch_main.append((defn, answer, SOURCE_TAG, 1))
            batch_aug.append((defn, answer, SOURCE_TAG))

        cryptic_conn.executemany(
            "INSERT INTO definition_answers (definition, answer, source, frequency) VALUES (?, ?, ?, ?)",
            batch_main
        )
        cryptic_conn.executemany(
            "INSERT INTO definition_answers_augmented (definition, answer, source) VALUES (?, ?, ?)",
            batch_aug
        )
        cryptic_conn.commit()
        print(f"Inserted {len(new_pairs)} new pairs into both tables")

        counter.counts['definition_answers'] = len(new_pairs)
        counter.skipped['definition_answers'] = len(valid_pairs) - len(new_pairs)
        counter.samples['definition_answers'] = [
            f"'{d}' -> {a.upper()} ({f}x)"
            for (d, a), f in sorted(new_pairs.items(), key=lambda x: -x[1])[:10]
        ]
        counter.report()

    clues_conn.close()
    cryptic_conn.close()


if __name__ == '__main__':
    main()
