"""
apply_candidates.py — Safe DB insertion for API-approved self-learning candidates.

Reads documents/self_learning_audit.txt (written by audit_candidates.py) and
inserts every entry with verdict=YES into cryptic_new.db.

Every insert is logged with its exact rollback DELETE statement to
documents/self_learning_inserts.txt so any change can be reversed manually.

Usage:
  python -m enrichment.apply_candidates                  # dry-run (default)
  python -m enrichment.apply_candidates --apply          # write to DB
  python -m enrichment.apply_candidates --audit documents/self_learning_audit.txt --apply
"""

import argparse
import json
import sys
from datetime import datetime
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enrichment.common import (
    get_cryptic_conn,
    insert_definition_answer, insert_indicator, insert_wordplay, insert_synonym_pair,
    InsertCounter, DRY_RUN,
)

SOURCE_TAG = 'self_learning_approved'


# ============================================================
# ROLLBACK STATEMENT BUILDERS
# ============================================================

def rollback_definition(phrase: str, answer: str) -> list[str]:
    """Return the DELETE statements that undo a definition_pair insert."""
    return [
        f"DELETE FROM definition_answers WHERE LOWER(definition)='{phrase.lower()}' "
        f"AND LOWER(answer)='{answer.lower()}';",
        f"DELETE FROM definition_answers_augmented WHERE LOWER(definition)='{phrase.lower()}' "
        f"AND LOWER(answer)='{answer.lower()}';",
    ]


def rollback_indicator(phrase: str, wordplay_type: str, subtype: str | None) -> list[str]:
    if subtype:
        return [
            f"DELETE FROM indicators WHERE LOWER(word)='{phrase.lower()}' "
            f"AND wordplay_type='{wordplay_type}' AND subtype='{subtype}';"
        ]
    return [
        f"DELETE FROM indicators WHERE LOWER(word)='{phrase.lower()}' "
        f"AND wordplay_type='{wordplay_type}';"
    ]


def rollback_wordplay(phrase: str, letters: str) -> list[str]:
    return [
        f"DELETE FROM wordplay WHERE LOWER(indicator)='{phrase.lower()}' "
        f"AND LOWER(substitution)='{letters.lower()}';"
    ]


def rollback_synonym(phrase: str, letters: str) -> list[str]:
    return [
        f"DELETE FROM synonyms_pairs WHERE LOWER(word)='{phrase.lower()}' "
        f"AND LOWER(synonym)='{letters.lower()}';"
    ]


# ============================================================
# APPLY LOGIC
# ============================================================

def apply_entry(candidate: dict, conn, counter: InsertCounter,
                log_lines: list) -> bool:
    """Insert one approved candidate.  Returns True if a new row was written."""
    ctype = candidate.get('type', '')
    phrase = candidate.get('phrase', '').strip()
    answer = candidate.get('answer', '').strip()
    pattern = candidate.get('pattern', '?')
    clue = candidate.get('clue', '')
    verdict = candidate.get('verdict', '')

    if verdict != 'YES':
        return False
    if not phrase:
        return False

    inserted = False
    rollback_stmts = []

    if ctype == 'definition_pair':
        inserted = insert_definition_answer(conn, phrase, answer, SOURCE_TAG)
        rollback_stmts = rollback_definition(phrase, answer)

    elif ctype == 'indicator':
        wordplay_type = candidate.get('wordplay_type', 'parts')
        subtype = candidate.get('subtype')
        inserted = insert_indicator(conn, phrase, wordplay_type,
                                    subtype=subtype, confidence='medium')
        rollback_stmts = rollback_indicator(phrase, wordplay_type, subtype)

    elif ctype == 'wordplay':
        letters = candidate.get('letters', answer)
        inserted = insert_wordplay(conn, phrase, letters, 'abbreviation',
                                   confidence='medium',
                                   notes=f'self_learning:{answer}',
                                   source_tag=SOURCE_TAG)
        rollback_stmts = rollback_wordplay(phrase, letters)

    elif ctype == 'synonym':
        letters = candidate.get('letters', answer)
        inserted = insert_synonym_pair(conn, phrase, letters, SOURCE_TAG)
        rollback_stmts = rollback_synonym(phrase, letters)

    else:
        print(f"  WARNING: unknown type '{ctype}' — skipped")
        return False

    label = f"pattern-{pattern} {ctype}: '{phrase}' -> {answer}"
    counter.record(ctype, inserted, label)

    if inserted:
        ts = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        log_lines.append(f"# [{ts}] {label}")
        log_lines.append(f"# clue: {clue}")
        log_lines.append(f"# reason: {candidate.get('reason', '')[:120]}")
        for stmt in rollback_stmts:
            log_lines.append(stmt)
        log_lines.append("")

    return inserted


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(
        description='Apply API-approved self-learning candidates to the DB')
    parser.add_argument(
        '--audit',
        default=str(PROJECT_ROOT / 'documents' / 'self_learning_audit.txt'),
        help='Audit file to read (default: documents/self_learning_audit.txt)'
    )
    parser.add_argument(
        '--log',
        default=str(PROJECT_ROOT / 'documents' / 'self_learning_inserts.txt'),
        help='Insert log file (default: documents/self_learning_inserts.txt)'
    )
    parser.add_argument(
        '--apply', action='store_true',
        help='Write to DB (default: dry-run only)'
    )
    args = parser.parse_args()

    # Set DRY_RUN globally before any insert helpers are called
    if not args.apply:
        import enrichment.common as _common
        _common.DRY_RUN = True
        print("[DRY RUN MODE — pass --apply to write to DB]")

    audit_path = Path(args.audit)
    if not audit_path.exists():
        print(f"Audit file not found: {audit_path}")
        sys.exit(1)

    lines = [l.strip() for l in audit_path.read_text(encoding='utf-8').splitlines()
             if l.strip()]
    candidates = []
    for i, line in enumerate(lines, 1):
        try:
            candidates.append(json.loads(line))
        except json.JSONDecodeError as e:
            print(f"  Warning: skipping line {i} (JSON error: {e})")

    approved = [c for c in candidates if c.get('verdict') == 'YES']
    skipped  = len(candidates) - len(approved)

    print(f"\nTotal candidates : {len(candidates)}")
    print(f"Approved (YES)   : {len(approved)}")
    print(f"Skipped (NO/UNSURE/other): {skipped}\n")

    if not approved:
        print("Nothing to apply.")
        return

    conn = get_cryptic_conn()
    counter = InsertCounter("apply_candidates")
    log_lines = [
        f"# self_learning_inserts.txt — generated {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
        f"# Each block shows the INSERT that was made and the DELETE to roll it back.",
        f"# To undo a specific insert, run the DELETE statement against cryptic_new.db.",
        "",
    ]

    for candidate in approved:
        phrase = candidate.get('phrase', '')
        answer = candidate.get('answer', '')
        ctype  = candidate.get('type', '')
        print(f"  Applying: {ctype} '{phrase}' -> {answer}")
        apply_entry(candidate, conn, counter, log_lines)

    conn.commit()
    conn.close()

    counter.report()

    # Write log file (even in dry-run, so the user can preview rollback statements)
    log_path = Path(args.log)
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open('w', encoding='utf-8') as f:
        f.write('\n'.join(log_lines) + '\n')

    mode = "DRY RUN" if not args.apply else "APPLIED"
    print(f"[{mode}] Insert log written to: {log_path}")


if __name__ == '__main__':
    main()
