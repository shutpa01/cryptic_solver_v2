"""
03_mine_indicators_from_tags.py — Discover indicators from 77k tagged clues.

Method: Find words that appear disproportionately in clues of one wordplay type.
A word is likely an indicator if:
  - It appears >= 5 times in clues of a specific type
  - Its specificity ratio (count_in_type / count_across_all_types) >= 0.6
  - It's not a common stop word or too short

Source: 77k clues with wordplay_type tags in cryptic_new.db.
Writes to: indicators table.
"""

import argparse
import re
import sqlite3
from collections import Counter, defaultdict

from enrichment.common import (
    get_cryptic_conn, insert_indicator,
    InsertCounter, add_common_args, apply_common_args,
)

SOURCE_TAG = 'tag_mining'

# Wordplay types from tagged clues → indicator table wordplay_type
TYPE_MAP = {
    'anagram': 'anagram',
    'container': 'container',
    'reversal': 'reversal',
    'deletion': 'deletion',
    'hidden': 'hidden',
    'acrostic': 'acrostic',
    'homophone': 'homophone',
}

# Types where indicators don't apply
SKIP_TYPES = {
    'charade', 'double_definition', 'cryptic_definition',
    'straight_definition', 'and_lit', 'spoonerism', 'palindrome',
}

# Common words that appear in clues but are NOT indicators
STOP_WORDS = {
    'the', 'a', 'an', 'is', 'in', 'of', 'to', 'for', 'and', 'or',
    'it', 'on', 'at', 'by', 'be', 'as', 'if', 'so', 'no', 'do',
    'up', 'he', 'she', 'we', 'me', 'my', 'his', 'her', 'its',
    'not', 'but', 'all', 'are', 'was', 'has', 'had', 'who', 'how',
    'may', 'can', 'one', 'two', 'new', 'old', 'big', 'see', 'get',
    'got', 'let', 'say', 'set', 'own', 'too', 'few', 'any', 'man',
    'men', 'way', 'did', 'end', 'put', 'run', 'use', 'try', 'ask',
    'yet', 'far', 'per', 'bit', 'lot', 'off', 'out', 'than',
    'that', 'this', 'with', 'from', 'have', 'been', 'were', 'them',
    'they', 'will', 'what', 'when', 'each', 'very', 'more', 'some',
    'also', 'most', 'just', 'over', 'only', 'such', 'make', 'like',
    'much', 'even', 'does', 'well', 'many', 'then', 'here', 'take',
    'come', 'made', 'find', 'long', 'look', 'want', 'give', 'else',
    'need', 'last', 'keep', 'good', 'best', 'part', 'show', 'work',
    'call', 'said', 'goes', 'down',
    # Common crossword words that are definitions, not indicators
    'king', 'queen', 'jack', 'ace', 'duke', 'earl',
    'love', 'hate', 'fear', 'hope', 'wish',
    'home', 'house', 'room', 'door', 'road', 'street',
    'fish', 'bird', 'tree', 'flower', 'plant', 'animal',
    'food', 'drink', 'wine', 'beer', 'water',
    'town', 'city', 'country', 'island', 'river', 'lake',
    'head', 'hand', 'foot', 'face', 'body', 'heart',
    'time', 'year', 'day', 'month', 'week', 'hour',
    'side', 'line', 'point', 'mark', 'sign',
}

# Minimum thresholds
MIN_FREQ = 5           # Must appear >= 5 times in the specific type
MIN_SPECIFICITY = 0.5  # Must have >= 50% of occurrences in that one type
MIN_WORD_LEN = 3       # Skip words shorter than 3 chars


def extract_clue_words(clue_text: str, answer: str) -> list:
    """Extract content words from clue text, excluding the answer."""
    words = re.findall(r'[a-zA-Z]+', clue_text.lower())
    answer_lower = answer.lower().replace(' ', '')
    # Filter out stop words, short words, and the answer itself
    return [
        w for w in words
        if len(w) >= MIN_WORD_LEN
        and w not in STOP_WORDS
        and w != answer_lower
    ]


def main():
    parser = argparse.ArgumentParser(description='Mine indicators from tagged clues')
    add_common_args(parser)
    parser.add_argument('--min-freq', type=int, default=MIN_FREQ,
                        help=f'Minimum frequency in type (default: {MIN_FREQ})')
    parser.add_argument('--min-spec', type=float, default=MIN_SPECIFICITY,
                        help=f'Minimum specificity ratio (default: {MIN_SPECIFICITY})')
    args = parser.parse_args()
    apply_common_args(args)

    conn = get_cryptic_conn()
    counter = InsertCounter('03_mine_indicators_from_tags')

    # Load tagged clues
    rows = conn.execute("""
        SELECT clue_text, answer, wordplay_type FROM clues
        WHERE wordplay_type IS NOT NULL AND wordplay_type != ''
          AND clue_text IS NOT NULL AND answer IS NOT NULL
    """).fetchall()
    print(f"Loaded {len(rows)} tagged clues")

    # Count word frequencies per wordplay type AND track (word, answer) pairs
    type_word_counts = defaultdict(Counter)  # type -> {word: count}
    total_word_counts = Counter()             # word -> total count
    # For content-word filtering: track which answers each word co-occurs with
    word_answers = defaultdict(list)  # (word, type) -> [answer, ...]

    for clue_text, answer, wtype in rows:
        if wtype in SKIP_TYPES or '+' in wtype:
            continue
        mapped_type = TYPE_MAP.get(wtype)
        if not mapped_type:
            continue

        words = extract_clue_words(clue_text, answer)
        for word in set(words):  # Use set to count once per clue
            type_word_counts[mapped_type][word] += 1
            total_word_counts[word] += 1
            word_answers[(word, mapped_type)].append(answer.upper().replace(' ', ''))

    # Find indicator candidates with content-word filtering
    print("\nDiscovering indicator candidates...")
    candidates = []
    content_filtered = 0

    for wtype, word_counts in type_word_counts.items():
        for word, type_freq in word_counts.items():
            if type_freq < args.min_freq:
                continue

            total_freq = total_word_counts[word]
            specificity = type_freq / total_freq

            if specificity < args.min_spec:
                continue

            # Content-word filter: check if this word typically contributes
            # letters to the answer (suggesting it's fodder, not an indicator)
            answers = word_answers[(word, wtype)]
            word_upper = word.upper()
            contributes_count = 0
            for ans in answers:
                # Check if word's letters are largely contained in the answer
                # (allowing for multi-word clue contributions)
                word_letters = sorted(word_upper)
                ans_letters = list(ans)
                matched = 0
                for ch in word_letters:
                    if ch in ans_letters:
                        ans_letters.remove(ch)
                        matched += 1
                # If >= 70% of word's letters appear in the answer, it's likely content
                if matched >= len(word_upper) * 0.7:
                    contributes_count += 1

            content_ratio = contributes_count / len(answers) if answers else 0
            if content_ratio > 0.5:
                content_filtered += 1
                continue

            candidates.append({
                'word': word,
                'type': wtype,
                'type_freq': type_freq,
                'total_freq': total_freq,
                'specificity': specificity,
            })

    candidates.sort(key=lambda c: (-c['specificity'], -c['type_freq']))
    print(f"Found {len(candidates)} indicator candidates (filtered {content_filtered} content words)")

    # Show summary by type
    type_summary = Counter()
    for c in candidates:
        type_summary[c['type']] += 1
    print("\nCandidates by type:")
    for wtype, count in type_summary.most_common():
        print(f"  {wtype:15s}: {count}")

    # Insert candidates
    for c in candidates:
        # Determine confidence based on specificity and frequency
        if c['specificity'] >= 0.8 and c['type_freq'] >= 20:
            confidence = 'high'
        elif c['specificity'] >= 0.6 and c['type_freq'] >= 10:
            confidence = 'medium'
        else:
            confidence = 'low'

        inserted = insert_indicator(
            conn,
            word=c['word'],
            wordplay_type=c['type'],
            subtype=None,
            confidence=confidence,
            frequency=c['type_freq'],
        )
        sample = f"{c['word']} [{c['type']}] spec={c['specificity']:.2f} freq={c['type_freq']}"
        counter.record('indicators', inserted, sample)

    if not args.dry_run:
        conn.commit()
    counter.report()

    # Show top new indicators per type
    print("\nTop candidates by type:")
    for wtype in sorted(type_summary.keys()):
        type_cands = [c for c in candidates if c['type'] == wtype]
        print(f"\n  {wtype}:")
        for c in type_cands[:10]:
            existing = conn.execute(
                "SELECT 1 FROM indicators WHERE word=? AND wordplay_type=?",
                (c['word'], c['type'])
            ).fetchone()
            status = "EXISTS" if existing else "NEW"
            print(f"    {c['word']:20s} spec={c['specificity']:.2f} freq={c['type_freq']:3d} [{status}]")

    conn.close()


if __name__ == '__main__':
    main()
