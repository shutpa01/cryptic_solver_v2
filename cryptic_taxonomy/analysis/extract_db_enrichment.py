"""Extract synonym and abbreviation pairs from verified parsed explanations.

The parser tells us: piece X with letters Y came from source Z.
Combined with clue text, we can extract new synonym/abbreviation pairs
that are missing from the reference DB.
"""

import sqlite3
import sys
import re
from collections import Counter

sys.path.insert(0, '.')
from cryptic_taxonomy.analysis.notation_parser import parse_explanation


def extract_pairs():
    """Extract (word, value, type) triples from verified parses."""

    conn = sqlite3.connect('data/times_explanations.db')
    rows = conn.execute(
        'SELECT answer, explanation FROM clues '
        'WHERE explanation IS NOT NULL AND explanation != ""'
    ).fetchall()
    conn.close()

    # Collect pairs: (source_word, letters, source_type)
    pairs = Counter()

    for answer, expl in rows:
        clean_answer = re.sub(r'[\s\-]', '', answer.upper())
        result = parse_explanation(expl, clean_answer)
        if not result.verified:
            continue

        for piece in result.pieces:
            if not piece.gloss or not piece.letters:
                continue

            # piece.gloss = the meaning (e.g. "male", "knock", "hesitation")
            # piece.letters = the contributed letters (e.g. "GENT", "LAM", "ER")
            gloss = piece.gloss.strip().lower()
            letters = piece.letters.upper()

            # Skip very long glosses (likely commentary, not a word)
            if len(gloss) > 30 or ' ' in gloss:
                continue
            # Skip if gloss equals the letters (trivial)
            if gloss.upper() == letters:
                continue

            ptype = 'SYN' if len(letters) > 2 else 'ABR'
            pairs[(gloss, letters, ptype)] += 1

    return pairs


def check_against_db(pairs):
    """Check which pairs are already in the DB."""
    from signature_solver.db import RefDB
    db = RefDB()

    new_syns = []
    new_abbrs = []

    for (gloss, letters, ptype), count in pairs.most_common():
        if ptype == 'ABR':
            existing = db.get_abbreviations(gloss)
            if letters not in [a.upper() for a in existing]:
                new_abbrs.append((gloss, letters, count))
        else:
            existing = db.get_synonyms(gloss, max_len=len(letters) + 2)
            if letters not in [s.upper() for s in existing]:
                new_syns.append((gloss, letters, count))

    return new_syns, new_abbrs


def main():
    print("Extracting pairs from verified explanations...")
    pairs = extract_pairs()
    print(f"Total unique pairs: {len(pairs)}")
    print(f"Total occurrences: {sum(pairs.values())}")

    print("\nChecking against reference DB...")
    new_syns, new_abbrs = check_against_db(pairs)

    print(f"\nNew synonyms not in DB: {len(new_syns)}")
    print(f"New abbreviations not in DB: {len(new_abbrs)}")

    print(f"\nTop 30 new synonyms (by frequency):")
    for gloss, letters, count in sorted(new_syns, key=lambda x: -x[2])[:30]:
        print(f"  {gloss:25s} -> {letters:15s} ({count}x)")

    print(f"\nTop 30 new abbreviations (by frequency):")
    for gloss, letters, count in sorted(new_abbrs, key=lambda x: -x[2])[:30]:
        print(f"  {gloss:25s} -> {letters:15s} ({count}x)")

    # Save to CSV for review
    import csv
    with open('cryptic_taxonomy/outputs/new_synonyms.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['word', 'synonym', 'frequency'])
        for gloss, letters, count in sorted(new_syns, key=lambda x: -x[2]):
            if count >= 2:
                w.writerow([gloss, letters, count])

    with open('cryptic_taxonomy/outputs/new_abbreviations.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['word', 'abbreviation', 'frequency'])
        for gloss, letters, count in sorted(new_abbrs, key=lambda x: -x[2]):
            if count >= 2:
                w.writerow([gloss, letters, count])

    print(f"\nSaved to cryptic_taxonomy/outputs/new_synonyms.csv and new_abbreviations.csv")


if __name__ == '__main__':
    main()
