"""
01_mine_times_notation.py — Mine substitutions and indicators from Times explanations.

Times explanations use a semi-structured notation:
  - T(time), O(ld), RE(about)   → letter substitutions
  - GREATLY*, (BATH ONCE I)*    → anagram fodder marked with *
  - {e}JECTED, {dagge}R         → deletion (curly braces = removed letters)
  - WORD(meaning)               → synonym/definition pairs
  - hidden in faSTER News       → hidden word (caps mark the answer)
  - double definition           → DD tag

Source: ~35k Times explanations in cryptic_new.db clues table.
Writes to: wordplay, indicators, definition_answers.
"""

import argparse
import re
import sqlite3
from collections import Counter

from enrichment.common import (
    get_cryptic_conn, norm_letters,
    insert_wordplay, insert_indicator, insert_definition_answer,
    InsertCounter, add_common_args, apply_common_args,
)

SOURCE_TAG = 'times_notation'


# ============================================================
# WORD SET — for distinguishing fragments from full words
# ============================================================

def build_word_set(conn: sqlite3.Connection) -> set:
    """Build a set of lowercase words from all clue texts for validation."""
    words = set()
    rows = conn.execute("SELECT clue_text FROM clues WHERE clue_text IS NOT NULL").fetchall()
    for (text,) in rows:
        for w in re.findall(r"[a-zA-Z]+", text):
            if len(w) >= 2:
                words.add(w.lower())
    return words


# ============================================================
# PATTERN 1: LETTER(word) substitutions
# ============================================================

def extract_substitutions(explanations, word_set):
    """Extract UPPER(lower) substitution patterns.

    Two conventions exist:
      T(time)  — full word in parens, letter is abbreviation
      O(ld)    — tail in parens, letter is prefix of the full word

    We distinguish by checking if concatenation forms a known word.
    """
    results = []  # (indicator_word, substitution_letters, count)
    pair_counts = Counter()

    for expl, answer in explanations:
        # Match 1-4 uppercase letters followed by (lowercase text)
        # Lookbehind ensures we don't match mid-word (e.g., TIGHT → IGHT)
        matches = re.findall(r'(?<![A-Za-z])([A-Z]{1,4})\(([a-z][a-z\s]*?)\)', expl)
        for letters, text in matches:
            text = text.strip()
            if not text or len(text) < 2:
                continue

            # Substitution letters should be shorter than the indicator word
            if len(letters) > len(text) and ' ' not in text:
                continue

            # Try concatenation: letters.lower() + text
            concat = letters.lower() + text

            if ' ' not in text and concat in word_set and text not in word_set:
                # Concatenation is a known word but text alone isn't → fragment
                full_word = concat
            elif ' ' not in text and concat in word_set and text in word_set:
                # Both are words — prefer the longer one (more specific)
                # e.g., S(mall) → "small" is better than "mall"
                full_word = concat if len(concat) > len(text) else text
            else:
                # Text is the full meaning (e.g., "one", "about", "old boy")
                full_word = text

            pair_counts[(full_word, letters)] += 1

    return pair_counts


def categorise_substitution(word, letters):
    """Guess a category for the substitution based on patterns."""
    word_lower = word.lower()
    letters_upper = letters.upper()

    # Roman numerals
    if letters_upper in ('I', 'V', 'X', 'L', 'C', 'D', 'M'):
        num_words = {
            'one': 'I', 'two': 'II', 'three': 'III', 'four': 'IV',
            'five': 'V', 'six': 'VI', 'ten': 'X', 'fifty': 'L',
            'hundred': 'C', 'thousand': 'M', 'five hundred': 'D',
            'nothing': 'O', 'love': 'O', 'zero': 'O', 'nil': 'O',
        }
        if word_lower in num_words:
            return 'roman_numeral'

    # Single-letter abbreviations
    if len(letters_upper) == 1:
        return 'abbreviation'

    # Multi-letter abbreviations
    if len(letters_upper) <= 3:
        return 'abbreviation'

    return 'substitution'


# ============================================================
# PATTERN 2: Anagram fodder (WORD* or (WORDS)*)
# ============================================================

def extract_anagram_indicators(explanations):
    """Find words adjacent to anagram fodder marked with *.

    We can't easily extract the indicator from the explanation alone,
    but we CAN confirm that the clue involves an anagram and extract
    definition-answer pairs from the non-fodder portion.
    """
    # For now, just count anagram clues — indicator extraction is
    # handled by script 05 (reverse-engineer from tagged clues).
    count = 0
    for expl, answer in explanations:
        if re.search(r'[A-Z]{2,}\*|\([A-Z\s]+\)\*', expl):
            count += 1
    return count


# ============================================================
# PATTERN 3: FULLWORD(meaning) — synonym/definition pairs
# ============================================================

def extract_synonym_definitions(explanations, word_set):
    """Extract FULLWORD(meaning) patterns where FULLWORD is a known word.

    e.g., MATTERED(was important), CUTE(clever), SWEAT(hard work)
    These give us definition→answer pairs.
    """
    results = Counter()

    for expl, answer in explanations:
        # Match 3+ uppercase letters followed by (text)
        matches = re.findall(r'([A-Z]{3,})\(([^)]+)\)', expl)
        for word, meaning in matches:
            meaning = meaning.strip()
            if not meaning:
                continue
            # Skip if it looks like a letter substitution (meaning is very short)
            if len(meaning) <= 2 and ' ' not in meaning:
                continue
            # Skip anagram fodder markers
            if meaning.endswith('*'):
                continue
            # The WORD must be a recognizable word (not just random uppercase)
            if word.lower() not in word_set:
                continue
            # Skip if meaning looks like a word fragment (no spaces, and
            # concatenation with WORD forms a known word — e.g., REV(erend))
            if ' ' not in meaning and (word.lower() + meaning) in word_set:
                continue
            results[(meaning.lower(), word.upper())] += 1

    return results


# ============================================================
# PATTERN 4: Deletion notation {removed}KEPT
# ============================================================

def extract_deletion_examples(explanations):
    """Extract deletion patterns using {curly braces}.

    {e}JECTED = "expelled" minus first letter
    {dagge}R = last letter of "dagger"
    MATTERED minus M = deletion of M from MATTERED
    """
    deletions = []
    for expl, answer in explanations:
        # Curly brace notation
        matches = re.findall(r'\{([a-z]+)\}([A-Z]+)', expl)
        for removed, kept in matches:
            full_word = removed + kept.lower()
            deletions.append({
                'full_word': full_word,
                'removed': removed,
                'kept': kept.lower(),
                'position': 'head' if len(removed) <= 2 else 'middle',
            })

        # Also match KEPT{removed} (tail deletion)
        matches2 = re.findall(r'([A-Z]+)\{([a-z]+)\}', expl)
        for kept, removed in matches2:
            full_word = kept.lower() + removed
            deletions.append({
                'full_word': full_word,
                'removed': removed,
                'kept': kept.lower(),
                'position': 'tail',
            })

    return deletions


# ============================================================
# PATTERN 5: "minus" / "without" notation
# ============================================================

def extract_minus_substitutions(explanations):
    """Extract 'WORD minus LETTER(meaning)' patterns.

    e.g., MATTERED minus M(month) → "month" maps to M
    These confirm existing substitutions with high confidence.
    """
    results = Counter()
    for expl, answer in explanations:
        # Match patterns like "minus M(word)" or "without T(word)"
        matches = re.findall(
            r'(?:minus|without|less|losing|dropping)\s+([A-Z]{1,3})\(([a-z][a-z\s]*?)\)',
            expl, re.I
        )
        for letters, word in matches:
            results[(word.strip(), letters.upper())] += 1
    return results


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Mine Times notation for substitutions and indicators')
    add_common_args(parser)
    args = parser.parse_args()
    apply_common_args(args)

    conn = get_cryptic_conn()
    counter = InsertCounter('01_mine_times_notation')

    # Load explanations from all sources that use LETTER(word) notation
    rows = conn.execute("""
        SELECT explanation, answer FROM clues
        WHERE source IN ('times', 'toughie', 'sunday_telegraph', 'telegraph')
          AND explanation IS NOT NULL AND TRIM(explanation) != ''
    """).fetchall()
    print(f"Loaded {len(rows)} explanations")

    # Build word set for validation
    print("Building word set from clue corpus...")
    word_set = build_word_set(conn)
    print(f"Word set: {len(word_set)} unique words")

    # ----- SUBSTITUTIONS -----
    print("\nExtracting LETTER(word) substitutions...")
    subst_pairs = extract_substitutions(rows, word_set)
    print(f"Found {len(subst_pairs)} unique substitution pairs")

    for (word, letters), freq in subst_pairs.most_common():
        if freq < 2:
            continue  # Skip hapax — likely noise
        category = categorise_substitution(word, letters)
        inserted = insert_wordplay(
            conn, indicator=word, substitution=letters,
            category=category, confidence='medium' if freq >= 3 else 'low',
            notes=f'freq={freq}', source_tag=SOURCE_TAG
        )
        counter.record('wordplay', inserted, f"{word} -> {letters} ({freq}x)")

    # ----- MINUS/WITHOUT SUBSTITUTIONS -----
    print("Extracting minus/without substitutions...")
    minus_pairs = extract_minus_substitutions(rows)
    for (word, letters), freq in minus_pairs.most_common():
        inserted = insert_wordplay(
            conn, indicator=word, substitution=letters,
            category=categorise_substitution(word, letters),
            confidence='high',  # "minus X(word)" is very explicit
            notes=f'minus_pattern freq={freq}', source_tag=SOURCE_TAG
        )
        counter.record('wordplay', inserted, f"{word} -> {letters} (minus, {freq}x)")

    # ----- SYNONYM/DEFINITION PAIRS -----
    print("Extracting WORD(meaning) definition pairs...")
    syn_pairs = extract_synonym_definitions(rows, word_set)
    print(f"Found {len(syn_pairs)} unique definition-answer pairs")

    for (definition, answer), freq in syn_pairs.most_common():
        if freq < 2:
            continue
        inserted = insert_definition_answer(
            conn, definition=definition, answer=answer,
            source=SOURCE_TAG
        )
        counter.record('definition_answers', inserted, f"{definition} -> {answer} ({freq}x)")

    # ----- COMMIT AND REPORT -----
    if not args.dry_run:
        conn.commit()
    counter.report()
    conn.close()


if __name__ == '__main__':
    main()
