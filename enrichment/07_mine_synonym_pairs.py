"""
07_mine_synonym_pairs.py — Mine WORD(meaning) synonym pairs from crossword explanations.

Crossword explanation bloggers use a consistent notation:

  OX(beefy type) CAME(turned up) inside BRA(underwear) → BOX CAMERA

The pattern WORD(meaning) tells us the setter used 'meaning' in the clue
as a synonym for the letter sequence WORD. Every parenthetical is a direct,
human-validated synonym pair.

We extract ALL such pairs and insert into synonyms_pairs as:
  word    = meaning  (what the clue says — the lookup key)
  synonym = WORD     (what letters it maps to — the answer component)

Sources: all explanations in cryptic_new.db (Times, Telegraph, FT, Toughie, etc.)
Writes to: synonyms_pairs in cryptic_new.db

Usage:
    python -m enrichment.07_mine_synonym_pairs --dry-run
    python -m enrichment.07_mine_synonym_pairs --dry-run --min-freq 2
    python -m enrichment.07_mine_synonym_pairs
"""

import argparse
import re
from collections import Counter

from enrichment.common import (
    get_cryptic_conn,
    insert_synonym_pair,
    InsertCounter,
    add_common_args,
    apply_common_args,
)

SOURCE_TAG = 'explanation_synonym'

# Core pattern: 2+ uppercase letters followed by (content)
# The lookbehind prevents matching mid-word (e.g. the R in PETER(name))
PAIR_PAT = re.compile(r'(?<![A-Z])([A-Z]{2,})\(([^)]{2,100})\)')

# Noise filters
DIGITS_ONLY  = re.compile(r'^[\d,\s\-]+$')
CLUE_REF     = re.compile(r'^\d{1,2}\s*[aAdD]?(\s*(across|down))?$', re.I)


def build_word_set(conn) -> set:
    """Lowercase word set from all clue texts — used for fragment detection."""
    words = set()
    for (text,) in conn.execute(
        "SELECT clue_text FROM clues WHERE clue_text IS NOT NULL"
    ):
        for w in re.findall(r'[a-zA-Z]+', text or ''):
            if len(w) >= 3:
                words.add(w.lower())
    return words


def is_valid_pair(word: str, meaning: str, word_set: set) -> bool:
    """
    Return True if (word, meaning) is a genuine synonym pair, not noise.

    word    = the uppercase letter sequence  e.g. CAME
    meaning = the text inside parens          e.g. "turned up"
    """
    meaning = meaning.strip()

    # Must contain at least one letter
    if not re.search(r'[a-zA-Z]', meaning):
        return False

    # Reject enumerations: (7), (3,4), (2-4)
    if DIGITS_ONLY.match(meaning):
        return False

    # Reject clue cross-references: (1a), (10d), (3 across)
    if CLUE_REF.match(meaning):
        return False

    # Reject anagram markers
    if '*' in meaning:
        return False

    # Reject letter-substitution notation: -a,+o or +b style
    if meaning.startswith('-') or meaning.startswith('+'):
        return False

    # Reject letter-extraction notation: [crow]n or [a]d[d]e[r] style
    if meaning.startswith('['):
        return False

    # Reject truncated bracket fragments: (gripping tool
    if meaning.startswith('('):
        return False

    # Reject very long strings — these are prose commentary, not synonyms
    if len(meaning) > 50:
        return False

    # Fragment detection: WORD.lower() + meaning forms a known word
    # e.g. REV(erend) → "reverend" is in word_set, so "erend" is a suffix not a synonym
    # Only applies to short single-word meanings with no spaces
    if ' ' not in meaning and len(meaning) <= 7:
        concat = word.lower() + meaning.lower()
        if concat in word_set:
            return False

    return True


def extract_pairs(explanations: list, word_set: set) -> Counter:
    """
    Scan explanations for WORD(meaning) and return a Counter of
    (meaning_lower, WORD_upper) → frequency.
    """
    counts = Counter()
    for expl, _answer in explanations:
        if not expl:
            continue
        for word, meaning in PAIR_PAT.findall(expl):
            if is_valid_pair(word, meaning.strip(), word_set):
                counts[(meaning.strip().lower(), word.upper())] += 1
    return counts


def main():
    parser = argparse.ArgumentParser(
        description='Mine WORD(meaning) synonym pairs from crossword explanations'
    )
    add_common_args(parser)
    parser.add_argument(
        '--min-freq', type=int, default=1,
        help='Minimum occurrences to insert (default: 1 = all valid pairs)'
    )
    args = parser.parse_args()
    apply_common_args(args)

    conn = get_cryptic_conn()
    counter = InsertCounter('07_mine_synonym_pairs')

    # Load all explanations from every source
    print("Loading explanations...")
    rows = conn.execute("""
        SELECT explanation, answer FROM clues
        WHERE explanation IS NOT NULL AND TRIM(explanation) != ''
    """).fetchall()
    print(f"  {len(rows):,} explanations")

    print("Building word set for fragment detection...")
    word_set = build_word_set(conn)
    print(f"  {len(word_set):,} words")

    print("Extracting WORD(meaning) pairs...")
    pair_counts = extract_pairs(rows, word_set)
    total_pairs = len(pair_counts)
    print(f"  {total_pairs:,} unique pairs found")

    freq_1    = sum(1 for c in pair_counts.values() if c == 1)
    freq_2    = sum(1 for c in pair_counts.values() if c >= 2)
    freq_5    = sum(1 for c in pair_counts.values() if c >= 5)
    freq_10   = sum(1 for c in pair_counts.values() if c >= 10)
    print(f"  freq=1: {freq_1:,}  >=2: {freq_2:,}  >=5: {freq_5:,}  >=10: {freq_10:,}")
    print()

    # Always show top 30 pairs for review
    print("Top 30 pairs by frequency:")
    for (meaning, word), freq in pair_counts.most_common(30):
        print(f"  {meaning!r:35} -> {word!r:20} ({freq}x)")
    print()

    # Load existing pairs into memory for fast dedup — avoids per-row DB queries
    print("Loading existing synonyms_pairs into memory...")
    existing_pairs = set(conn.execute(
        "SELECT LOWER(word), LOWER(synonym) FROM synonyms_pairs"
    ).fetchall())
    print(f"  {len(existing_pairs):,} existing pairs loaded")
    print()

    # Insert
    new_count = 0
    dup_count = 0
    candidates = [(m, w, f) for (m, w), f in pair_counts.most_common()
                  if f >= args.min_freq]
    print(f"Inserting {len(candidates):,} pairs with freq >= {args.min_freq}...")

    for meaning, word, freq in candidates:
        is_new = (meaning.lower(), word.lower()) not in existing_pairs
        if is_new:
            new_count += 1
            counter.record('synonyms_pairs', True,
                           f"{meaning!r:35} -> {word!r} ({freq}x)")
            if not args.dry_run:
                conn.execute(
                    "INSERT INTO synonyms_pairs (word, synonym, source) VALUES (?, ?, ?)",
                    (meaning.lower(), word.lower(), SOURCE_TAG)
                )
                existing_pairs.add((meaning.lower(), word.lower()))
        else:
            dup_count += 1
            counter.record('synonyms_pairs', False)

    print(f"  New: {new_count:,}  Already exist: {dup_count:,}")

    if not args.dry_run:
        conn.commit()

    counter.report()
    conn.close()


if __name__ == '__main__':
    main()
