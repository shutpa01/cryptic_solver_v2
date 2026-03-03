"""Simple bulk insert for Telegraph enrichment — no fancy helpers."""

import re
import sqlite3
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
REVIEW_FILE = PROJECT_ROOT / 'documents' / 'telegraph_enrichment_review.txt'
DB_PATH = PROJECT_ROOT / 'data' / 'cryptic_new.db'


def parse_review_file():
    substitutions = []
    synonyms = []

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
    print("INSERT TELEGRAPH ENRICHMENT (simple)")
    print("=" * 60)

    subs, syns = parse_review_file()
    print(f"Parsed: {len(subs)} substitutions, {len(syns)} synonyms")

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    c = conn.cursor()

    # Wordplay inserts
    print("\nInserting into wordplay table...")
    new_wordplay = 0
    for indicator, sub, freq in subs:
        try:
            c.execute("""
                INSERT INTO wordplay (indicator, substitution, category, frequency, confidence, notes)
                VALUES (?, ?, ?, 0, 'medium', ?)
                ON CONFLICT(indicator, substitution) DO NOTHING
            """, (indicator.lower(), sub.upper(), 'abbreviation', f'telegraph freq={freq}'))
            if c.rowcount > 0:
                new_wordplay += 1
        except Exception as e:
            print(f"  Error: {e} for {indicator} -> {sub}")

    print(f"  Inserted {new_wordplay} new entries (skipped {len(subs) - new_wordplay} duplicates)")

    # Synonym inserts — batch into chunks
    print("\nInserting into synonyms_pairs table...")
    new_syns = 0
    batch_size = 1000

    for i in range(0, len(syns), batch_size):
        batch = syns[i:i+batch_size]
        for word, synonym, freq in batch:
            # Check if exists
            c.execute("SELECT 1 FROM synonyms_pairs WHERE word=? AND synonym=? LIMIT 1",
                     (word.lower(), synonym.lower()))
            if c.fetchone():
                continue
            c.execute("INSERT INTO synonyms_pairs (word, synonym, source) VALUES (?, ?, ?)",
                     (word.lower(), synonym.lower(), 'telegraph'))
            new_syns += 1

        if (i // batch_size + 1) % 3 == 0:
            print(f"  Processed {min(i + batch_size, len(syns))}/{len(syns)}...")

    print(f"  Inserted {new_syns} new entries (skipped {len(syns) - new_syns} duplicates)")

    conn.commit()
    conn.close()

    print("\n" + "=" * 60)
    print(f"DONE: {new_wordplay} wordplay + {new_syns} synonyms")
    print("=" * 60)


if __name__ == '__main__':
    main()
