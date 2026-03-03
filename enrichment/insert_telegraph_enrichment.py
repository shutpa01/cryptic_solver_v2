"""
insert_telegraph_enrichment.py — Read telegraph_enrichment_review.txt
and insert approved entries into cryptic_new.db.

Reads the output from parse_telegraph_explanations.py and inserts:
  - Substitutions (1-2 letters) → wordplay table
  - Synonyms (3+ letters) → synonyms_pairs table
"""

import re
import sys
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enrichment.common import (
    get_cryptic_conn, insert_wordplay, insert_synonym_pair, InsertCounter
)
REVIEW_FILE = PROJECT_ROOT / 'documents' / 'telegraph_enrichment_review.txt'


def parse_review_file():
    """Parse the review file and return (substitutions, synonyms)."""
    substitutions = []  # [(indicator, sub, freq), ...]
    synonyms = []       # [(word, synonym, freq), ...]

    with open(REVIEW_FILE, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    current_section = None
    for line in lines:
        line = line.rstrip()

        if '=== TABLE: wordplay' in line:
            current_section = 'wordplay'
            continue
        elif '=== TABLE: synonyms_pairs' in line:
            current_section = 'synonyms'
            continue
        elif line.startswith('Count:') or line.startswith('TOTALS:'):
            current_section = None
            continue

        if not current_section or not line.strip():
            continue

        # Parse entry: "indicator -> SUB (freq)" or "word = SYN (freq)"
        if current_section == 'wordplay':
            m = re.match(r'^(.+?)\s+->\s+([A-Z]+)\s+\((\d+)\)', line)
            if m:
                indicator = m.group(1).strip()
                sub = m.group(2).strip()
                freq = int(m.group(3))
                substitutions.append((indicator, sub, freq))

        elif current_section == 'synonyms':
            m = re.match(r'^(.+?)\s+=\s+([A-Z]+)\s+\((\d+)\)', line)
            if m:
                word = m.group(1).strip()
                synonym = m.group(2).strip()
                freq = int(m.group(3))
                synonyms.append((word, synonym, freq))

    return substitutions, synonyms


def main():
    print("=" * 60)
    print("INSERT TELEGRAPH ENRICHMENT")
    print("=" * 60)

    if not REVIEW_FILE.exists():
        print(f"ERROR: Review file not found: {REVIEW_FILE}")
        return

    print(f"Reading from: {REVIEW_FILE}")
    substitutions, synonyms = parse_review_file()

    print(f"  {len(substitutions)} substitutions")
    print(f"  {len(synonyms)} synonyms")

    conn = get_cryptic_conn()
    counter = InsertCounter('Telegraph Enrichment')

    # Insert substitutions → wordplay table
    for indicator, sub, freq in substitutions:
        inserted = insert_wordplay(
            conn, indicator, sub,
            category='abbreviation' if len(sub) <= 2 else 'synonym',
            confidence='medium',
            notes=f'freq={freq}',
            source_tag='telegraph'
        )
        counter.record('wordplay', inserted, f'{indicator} → {sub}')

    # Insert synonyms → synonyms_pairs table
    for word, synonym, freq in synonyms:
        inserted = insert_synonym_pair(conn, word, synonym, source='telegraph')
        counter.record('synonyms_pairs', inserted, f'{word} = {synonym}')

    conn.commit()
    conn.close()

    counter.report()

    print("=" * 60)
    print("DONE")
    print("=" * 60)


if __name__ == '__main__':
    main()
