"""
06_reverse_engineer_charades.py — Discover substitutions from charade clues.

For 9k tagged charade clues where we know the answer, try to account for
the answer letters using known substitutions. When only a small gap remains
(1-4 unaccounted letters), map the gap to the remaining clue word.

Example:
  SLIGHT = S(mall) + ??? -> "insignificant" maps to LIGHT
  This gives us: insignificant -> LIGHT (new wordplay entry)

Source: 9k charade-tagged clues in cryptic_new.db.
Writes to: wordplay, synonyms_pairs.
"""

import argparse
import re
import sqlite3
from collections import Counter

from enrichment.common import (
    get_cryptic_conn, norm_letters,
    insert_wordplay, insert_synonym_pair,
    InsertCounter, add_common_args, apply_common_args,
)

SOURCE_TAG = 'charade_reverse'


def load_substitution_map(conn: sqlite3.Connection) -> dict:
    """Load all known substitutions: indicator -> [(substitution, category)]."""
    subs = {}
    for row in conn.execute("SELECT LOWER(indicator), UPPER(substitution), category FROM wordplay"):
        indicator, subst, cat = row
        if indicator not in subs:
            subs[indicator] = []
        subs[indicator].append((subst, cat or ''))
    return subs


def load_synonym_map(conn: sqlite3.Connection) -> dict:
    """Load synonyms: word -> [synonym_upper, ...]."""
    syns = {}
    rows = conn.execute("SELECT LOWER(word), UPPER(synonym) FROM synonyms_pairs").fetchall()
    for word, synonym in rows:
        if word not in syns:
            syns[word] = []
        syns[word].append(synonym)
    return syns


def get_clue_words(clue_text: str) -> list:
    """Extract content words from clue text (remove enumeration, lowercase)."""
    text = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text)
    words = re.findall(r'[a-zA-Z]+', text.lower())
    return words


def try_account_answer(answer: str, clue_words: list, sub_map: dict) -> list:
    """Try to account for answer letters using known substitutions.

    Returns list of (clue_word, substitution, remaining_answer) tuples
    for successful partial matches, or empty list if no progress.
    """
    answer_upper = answer.upper()
    results = []

    # Try each clue word against the answer
    for word in clue_words:
        if word in sub_map:
            for subst, cat in sub_map[word]:
                # Check if this substitution appears in the answer
                if subst in answer_upper:
                    results.append((word, subst))

    return results


def find_new_substitutions(answer: str, clue_words: list, sub_map: dict,
                           syn_map: dict) -> list:
    """Try to find new word->letters mappings from charade clues.

    Strategy: try to build the answer left-to-right using known substitutions.
    When a gap remains, try to match it to an unmatched clue word.
    """
    answer_upper = answer.upper().replace(' ', '')
    if len(answer_upper) < 3:
        return []

    # Build a list of possible substitutions for each clue word
    word_options = {}  # word -> [(substitution, source), ...]
    stop_words = {'the', 'a', 'an', 'is', 'in', 'of', 'to', 'for', 'and', 'or',
                  'with', 'from', 'by', 'on', 'at', 'its', 'this', 'that'}

    for word in clue_words:
        if word in stop_words or len(word) < 2:
            continue
        options = []

        # Known substitutions from wordplay table
        if word in sub_map:
            for subst, cat in sub_map[word]:
                options.append((subst, 'wordplay'))

        # First letter
        options.append((word[0].upper(), 'first_letter'))

        word_options[word] = options

    # Try greedy left-to-right matching
    discoveries = []
    remaining = answer_upper
    matched_words = set()

    for word in clue_words:
        if word in stop_words or word in matched_words:
            continue
        if not remaining:
            break

        if word in word_options:
            for subst, source in word_options[word]:
                if remaining.startswith(subst):
                    remaining = remaining[len(subst):]
                    matched_words.add(word)
                    break

    # If we have remaining letters and unmatched clue words, try to map
    # Only for short gaps (1-3 letters) — these are abbreviation-type substitutions
    # which are the highest-value discoveries. Longer gaps are often definitions.
    matched_len = len(answer_upper) - len(remaining)
    if 1 <= len(remaining) <= 3 and matched_len >= 2:
        unmatched = [w for w in clue_words if w not in matched_words
                     and w not in stop_words and len(w) >= 2]
        for word in unmatched:
            # Skip if this word is likely the definition (synonym of full answer)
            # by checking if the word appears in definition_answers for this answer
            if word == answer_upper.lower():
                continue
            discoveries.append((word, remaining))
            break  # Only one discovery per clue

    return discoveries


def main():
    parser = argparse.ArgumentParser(description='Reverse-engineer charade clues for substitutions')
    add_common_args(parser)
    args = parser.parse_args()
    apply_common_args(args)

    conn = get_cryptic_conn()
    counter = InsertCounter('06_reverse_engineer_charades')

    # Load reference data
    print("Loading substitution map...")
    sub_map = load_substitution_map(conn)
    print(f"  {len(sub_map)} indicators with substitutions")

    print("Loading synonym map...")
    syn_map = load_synonym_map(conn)
    print(f"  {len(syn_map)} words with synonyms")

    # Load charade clues
    rows = conn.execute("""
        SELECT clue_text, answer FROM clues
        WHERE wordplay_type = 'charade'
          AND answer IS NOT NULL AND clue_text IS NOT NULL
    """).fetchall()
    print(f"\nLoaded {len(rows)} charade clues")

    # Discover new substitutions
    print("Reverse-engineering charade clues...")
    discovery_counts = Counter()

    for clue_text, answer in rows:
        clue_words = get_clue_words(clue_text)
        discoveries = find_new_substitutions(answer, clue_words, sub_map, syn_map)

        for word, letters in discoveries:
            discovery_counts[(word, letters)] += 1

    # Filter to pairs that appear multiple times (higher confidence)
    valid = {k: v for k, v in discovery_counts.items() if v >= 2}
    print(f"Discovered {len(valid)} substitution pairs (freq >= 2)")

    # Insert
    for (word, letters), freq in sorted(valid.items(), key=lambda x: -x[1]):
        # Determine category
        if len(letters) == 1:
            category = 'abbreviation'
        elif len(letters) <= 3:
            category = 'abbreviation'
        else:
            category = 'synonym'

        confidence = 'medium' if freq >= 3 else 'low'

        inserted = insert_wordplay(
            conn, indicator=word, substitution=letters,
            category=category, confidence=confidence,
            notes=f'charade freq={freq}', source_tag=SOURCE_TAG
        )
        counter.record('wordplay', inserted, f"{word} -> {letters} ({freq}x)")

        # For longer substitutions, also add as synonym pair
        if len(letters) >= 3:
            syn_inserted = insert_synonym_pair(
                conn, word=word, synonym=letters.lower(),
                source=SOURCE_TAG
            )
            if syn_inserted:
                counter.record('synonyms_pairs', syn_inserted, f"{word} -> {letters.lower()}")

    if not args.dry_run:
        conn.commit()
    counter.report()
    conn.close()


if __name__ == '__main__':
    main()
