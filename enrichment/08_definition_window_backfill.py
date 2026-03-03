#!/usr/bin/env python3
"""08_definition_window_backfill.py — Assign definitions to clues via window matching.

For each clue in clues_master.db that has an answer but no definition:
  1. Generate prefix/suffix windows of the clue text
  2. Check if any window is a known definition or synonym of the answer
     (using definition_answers_augmented + synonyms_pairs from cryptic_new.db)
  3. If a match is found, write the shortest matching window as the definition

This is safe: only UPDATES rows where definition IS NULL or empty.
No new rows created. Fully reversible via --revert.

Usage:
    python 08_definition_window_backfill.py                # full run
    python 08_definition_window_backfill.py --dry-run      # preview only
    python 08_definition_window_backfill.py --limit 100    # process 100 clues
    python 08_definition_window_backfill.py --revert       # NULL out definitions set by this script
"""

import argparse
import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
CLUES_MASTER_DB = PROJECT_ROOT / 'data' / 'clues_master.db'
CRYPTIC_DB = PROJECT_ROOT / 'data' / 'cryptic_new.db'

SOURCE_TAG = 'window_backfill'

ARTICLES = ('a ', 'an ', 'the ')


def clean_key(text: str) -> str:
    """Normalise text for lookup: lowercase, strip outer punctuation, collapse spaces."""
    text = text.lower().strip()
    text = text.replace('\u2019', "'").replace('\u2018', "'")
    text = re.sub(r'^[^a-z0-9]+', '', text)
    text = re.sub(r'[^a-z0-9]+$', '', text)
    text = re.sub(r'\s+', ' ', text)
    return text


def build_lookup(cryptic_conn: sqlite3.Connection) -> dict[str, set[str]]:
    """Build word -> set(answers) lookup from both tables.

    Keys are clean_key(word), values are sets of clean_key(answer).
    """
    lookup = {}

    def add(word: str | None, answer: str | None):
        if not word or not answer:
            return
        k = clean_key(word)
        a = clean_key(answer)
        if k and a and len(k) >= 2:
            lookup.setdefault(k, set()).add(a)

    # definition_answers_augmented
    cur = cryptic_conn.execute('SELECT definition, answer FROM definition_answers_augmented')
    da_count = 0
    for row in cur:
        add(row[0], row[1])
        da_count += 1

    # synonyms_pairs
    cur = cryptic_conn.execute('SELECT word, synonym FROM synonyms_pairs')
    sp_count = 0
    for row in cur:
        add(row[0], row[1])
        add(row[1], row[0])  # bidirectional
        sp_count += 1

    print(f'Lookup built: {len(lookup):,} keys '
          f'(from {da_count:,} def_answers + {sp_count:,} synonym_pairs)')
    return lookup


def generate_windows(clue_text: str) -> list[str]:
    """Generate prefix and suffix windows from clue text."""
    clue_text = clue_text.replace('\u2019', "'").replace('\u2018', "'")
    words = clue_text.split()
    windows = set()

    for i in range(len(words)):
        prefix = ' '.join(words[:i + 1])
        suffix = ' '.join(words[-(i + 1):])
        if prefix:
            windows.add(prefix.strip())
        if suffix:
            windows.add(suffix.strip())

    # Apostrophe bifurcation
    expanded = set(windows)
    for w in windows:
        if "'s" in w:
            expanded.add(w.replace("'s", 's'))

    return list(expanded)


def find_definition(clue_text: str, answer: str, lookup: dict) -> str | None:
    """Find the best definition window for a clue/answer pair.

    Returns the shortest matching window, or None if no match.
    """
    answer_key = clean_key(answer.replace(' ', '').replace('-', ''))
    if not answer_key:
        return None

    windows = generate_windows(clue_text)
    matches = []

    for w in windows:
        key = clean_key(w)
        if not key:
            continue

        # Direct match
        if key in lookup and answer_key in lookup[key]:
            matches.append(w)
            continue

        # Article variant: try "a X", "an X", "the X"
        for art in ARTICLES:
            art_key = clean_key(art + key)
            if art_key in lookup and answer_key in lookup[art_key]:
                matches.append(w)
                break

    if not matches:
        return None

    # Return shortest match (most specific definition)
    return min(matches, key=len)


def main():
    parser = argparse.ArgumentParser(
        description='Backfill definitions via window matching against known pairs'
    )
    parser.add_argument('--dry-run', action='store_true',
                        help='Preview matches without writing to DB')
    parser.add_argument('--limit', type=int, default=0,
                        help='Process at most N clues (0 = all)')
    parser.add_argument('--revert', action='store_true',
                        help='NULL out definitions set by this script')
    parser.add_argument('--source', default='',
                        help='Only process clues from this source (guardian, telegraph, etc.)')
    args = parser.parse_args()

    clues_conn = sqlite3.connect(CLUES_MASTER_DB)
    clues_conn.execute('PRAGMA journal_mode=WAL')

    if args.revert:
        cur = clues_conn.execute(
            "SELECT COUNT(*) FROM clues WHERE definition LIKE '%[window_backfill]%'"
        )
        count = cur.fetchone()[0]
        print(f'Reverting {count:,} definitions tagged with [window_backfill]...')
        if not args.dry_run:
            clues_conn.execute(
                "UPDATE clues SET definition = NULL WHERE definition LIKE '%[window_backfill]%'"
            )
            clues_conn.commit()
            print('Done.')
        else:
            print('DRY RUN - no changes made.')
        clues_conn.close()
        return

    # Build lookup
    cryptic_conn = sqlite3.connect(CRYPTIC_DB)
    lookup = build_lookup(cryptic_conn)
    cryptic_conn.close()

    # Fetch clues needing definitions
    query = '''
        SELECT id, clue_text, answer FROM clues
        WHERE answer IS NOT NULL AND answer != ''
          AND (definition IS NULL OR definition = '')
    '''
    params = []
    if args.source:
        query += ' AND source = ?'
        params.append(args.source)

    if args.limit:
        query += f' LIMIT {args.limit}'

    rows = clues_conn.execute(query, params).fetchall()
    print(f'Processing {len(rows):,} clues without definitions...')

    matched = 0
    batch = []
    batch_size = 500

    for i, (clue_id, clue_text, answer) in enumerate(rows):
        if not clue_text or not answer:
            continue

        defn = find_definition(clue_text, answer, lookup)
        if defn:
            matched += 1
            batch.append((defn, clue_id))

            if args.dry_run and matched <= 20:
                print(f'  [{matched}] "{clue_text[:60]}" -> {answer}')
                print(f'        definition: "{defn}"')

            if not args.dry_run and len(batch) >= batch_size:
                clues_conn.executemany(
                    'UPDATE clues SET definition = ? WHERE id = ?',
                    batch
                )
                clues_conn.commit()
                batch.clear()

        if (i + 1) % 10000 == 0:
            pct = 100 * matched / (i + 1)
            print(f'  ...{i + 1:,} processed, {matched:,} matched ({pct:.1f}%)')

    # Flush remaining batch
    if batch and not args.dry_run:
        clues_conn.executemany(
            'UPDATE clues SET definition = ? WHERE id = ?',
            batch
        )
        clues_conn.commit()

    clues_conn.close()

    pct = 100 * matched / len(rows) if rows else 0
    print(f'\nDone. Matched: {matched:,} / {len(rows):,} ({pct:.1f}%)')
    if args.dry_run:
        print('DRY RUN - no changes written.')


if __name__ == '__main__':
    main()
