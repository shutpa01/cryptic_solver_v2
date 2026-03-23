"""
Insert verified explanation mining discoveries into cryptic_new.db synonyms_pairs.

- Reads from data/explanation_mining_discoveries.jsonl
- Filters: count >= 3, not in blacklist
- Dedup guard: skips if (word, synonym) already exists regardless of source
- Source: 'explanation_mining'
"""

import json
import sqlite3
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent
REF_DB = BASE / "data" / "cryptic_new.db"
INPUT_JSONL = BASE / "data" / "explanation_mining_discoveries.jsonl"

SOURCE = "explanation_mining"
MIN_COUNT = 3

# Known-bad entries identified by manual review of all 1,573 entries
BLACKLIST = {
    # Abbreviation artifacts — word not a real abbreviation source
    ('the', 'T'), ('this', 'TH'), ('oring', 'O'), ('tincan', 'TIN'),
    ('dirt', 'D'), ('horse', 'H'), ('cape', 'C'), ('goo', 'G'),
    ('eager', 'E'), ('eat', 'E'), ('bee', 'B'), ('roo', 'R'),
    ('tor', 'T'), ('emporium', 'E'), ('upset', 'U'), ('camp', 'C'),
    ('doubt', 'D'), ('escalope', 'E'), ('heman', 'HE'), ('redhead', 'R'),
    ('steve', 'S'), ('star', 'S'), ('trophy', 'T'), ('dday', 'D'),
    ('some', 'S'), ('one', 'O'), ('plant', 'P'), ('show', 'S'),
    ('script', 'S'), ('arts', 'A'),
    # Synonym artifacts — word fragments from mis-parsed notation
    ('ailwa', 'R'), ('allons', 'G'), ('cstacy', 'E'), ('danger', 'REAT'),
    ('ember', 'DEC'), ('ertory', 'REP'), ('people', 'ATION'),
    ('idence', 'RES'), ('ar', 'R'), ('ha', 'T'), ('eorgi', 'G'),
    ('eall', 'R'), ('learl', 'C'), ('pper class', 'U'), ('motio', 'E'),
    ('xces', 'E'), ('hat', 'RICORN'), ('an', 'AM'),
    ('terrible', 'RIGHTFUL'), ('form the clue', 'IN'),
    ('first clue', 'ACROSS'), ('our', 'C'),
    # left -> LE is wrong (L is standard), read -> RE is not standard
    ('left', 'LE'), ('read', 'RE'),
}


def main():
    # Load discoveries
    with open(INPUT_JSONL, encoding='utf-8') as f:
        entries = [json.loads(line) for line in f]

    print(f"Loaded {len(entries)} discoveries from JSONL")

    # Filter
    filtered = []
    skipped_count = 0
    skipped_blacklist = 0
    for e in entries:
        if e['count'] < MIN_COUNT:
            skipped_count += 1
            continue
        key = (e['word'].lower(), e['value'].upper())
        if key in BLACKLIST:
            skipped_blacklist += 1
            continue
        filtered.append(e)

    print(f"After filtering: {len(filtered)} entries")
    print(f"  Skipped (count < {MIN_COUNT}): {skipped_count}")
    print(f"  Skipped (blacklisted): {skipped_blacklist}")

    # Load all existing pairs into a set for fast dedup
    conn = sqlite3.connect(REF_DB)
    print("Loading existing synonym pairs for dedup...")
    existing = set()
    for word, syn in conn.execute("SELECT LOWER(word), UPPER(synonym) FROM synonyms_pairs"):
        existing.add((word, syn))
    print(f"  {len(existing)} existing pairs loaded")

    inserted = 0
    already_exists = 0

    for e in filtered:
        word = e['word'].lower()
        synonym = e['value'].upper()

        if (word, synonym) in existing:
            already_exists += 1
            continue

        conn.execute(
            "INSERT INTO synonyms_pairs (word, synonym, source) VALUES (?, ?, ?)",
            (word, synonym, SOURCE)
        )
        existing.add((word, synonym))  # prevent self-duplicates
        inserted += 1

    conn.commit()

    # Verify
    total_source = conn.execute(
        "SELECT COUNT(*) FROM synonyms_pairs WHERE source = ?", (SOURCE,)
    ).fetchone()[0]

    conn.close()

    print(f"\nResults:")
    print(f"  Inserted: {inserted}")
    print(f"  Already in DB: {already_exists}")
    print(f"  Total '{SOURCE}' entries now in DB: {total_source}")


if __name__ == '__main__':
    main()
