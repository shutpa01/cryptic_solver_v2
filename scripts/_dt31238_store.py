"""Store DT 31238 leftover parses + queue genuine DB-miss enrichments.

Mirrors the workflow in scripts/_times29540_store.py:
1. Coverage check each enrichment against DB; queue only genuine misses
2. Verify each parse and capture verdict/score
3. INSERT OR REPLACE structured_explanations (model_version='claude_review')
4. UPDATE clues table
5. Queue genuine misses to pending_enrichments

Run with --dry-run to preview without writing.
"""
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / 'data' / 'clues_master.db'
REF_DB = ROOT / 'data' / 'cryptic_new.db'
sys.path.insert(0, str(ROOT))
from sonnet_pipeline.verify_explanation import ExplanationVerifier
from scripts._dt31238_parses import CLUES

# Enrichment candidates. Each (type, word, letters/wordplay_type, answer)
# Coverage check below filters to genuine misses only.
ENRICHMENTS = [
    # ===== Definitions (CDs and unusual phrases) =====
    ('definition', 'that sucks flying at night', 'VAMPIREBAT', 'vampire bat'),
    ('definition', 'off', 'BELOWPAR', 'below par'),
    ('definition', 'line of Americans', 'QUEUE', 'QUEUE'),
    ('definition', 'marketing material', 'LEAFLET', 'LEAFLET'),
    ('definition', 'range of letters', 'ALPHABET', 'ALPHABET'),
    ('definition', 'Provider of interbank liquidity in Egypt', 'NILE', 'Nile'),
    ('definition', 'plain English', 'ANGLOSAXON', 'Anglo-Saxon'),
    ('definition', 'Attractiveness', 'MAGNETISM', 'MAGNETISM'),
    ('definition', 'officer of the navy', 'ADMIRAL', 'ADMIRAL'),
    ('definition', 'choice cut', 'TBONESTEAK', 'T-bone steak'),
    ('definition', 'Extemporised notes', 'JAMSESSION', 'jam session'),
    ('definition', 'where the envelope is pushed', 'LETTERBOX', 'LETTERBOX'),
    ('definition', 'Confused', 'INASPIN', 'in a spin'),

    # ===== Synonyms =====
    ("synonym", "Manuel's gag", 'QUE', 'QUEUE'),
    ("synonym", "Perrins's saucy partner", 'LEA', 'LEAFLET'),
    ('synonym', 'generously proportioned', 'AMPLE', 'EXAMPLE'),
    ('synonym', 'hardwood', 'TEAK', 'T-bone steak'),

    # ===== Indicators =====
    ('indicator', 'evacuating', 'deletion', 'IMPERIL'),
]


def has_in_db(ref, etype, word, letters):
    """Return True if this enrichment row is already in DB."""
    w = word.lower()
    if etype == 'synonym':
        r = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE "
            "(LOWER(word)=? AND UPPER(synonym)=?) OR "
            "(LOWER(word)=? AND UPPER(synonym)=?) LIMIT 1",
            (w, letters.upper(), letters.lower(), word.upper())
        ).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE "
            "LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (w, letters.upper())
        ).fetchone()
        return bool(r or r2)
    if etype == 'abbreviation':
        r = ref.execute(
            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND UPPER(substitution)=? LIMIT 1",
            (w, letters.upper())
        ).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
            (w, letters.upper())
        ).fetchone()
        return bool(r or r2)
    if etype == 'definition':
        r = ref.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (w, letters.upper())
        ).fetchone()
        return bool(r)
    if etype == 'indicator':
        # letters carries wordplay_type for indicators
        r = ref.execute(
            "SELECT 1 FROM indicators WHERE LOWER(word)=? AND LOWER(wordplay_type)=? LIMIT 1",
            (w, letters.lower())
        ).fetchone()
        return bool(r)
    return False


def main():
    dry_run = '--dry-run' in sys.argv
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    ref = sqlite3.connect(str(REF_DB))
    verifier = ExplanationVerifier()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"{'DRY RUN' if dry_run else 'WRITING'}: {len(CLUES)} clues; "
          f"{len(ENRICHMENTS)} candidate enrichments\n")

    # === Phase 1: Coverage check enrichments ===
    to_queue = []
    skipped_in_db = []
    for etype, word, letters, answer in ENRICHMENTS:
        if has_in_db(ref, etype, word, letters):
            skipped_in_db.append((etype, word, letters))
        else:
            to_queue.append((etype, word, letters, answer))
    print(f"Enrichments: {len(to_queue)} genuine misses, "
          f"{len(skipped_in_db)} already in DB (skipped)")
    for et, w, L in skipped_in_db:
        print(f"  SKIP (in DB): {et:13} {w!r} -> {L!r}")
    print()

    # === Phase 2: Verify and store each clue ===
    tiers = {'HIGH': [], 'MEDIUM': [], 'LOW': [], 'FAIL': []}
    for clue_id, wtype, definition, expl in CLUES:
        row = conn.execute("SELECT * FROM clues WHERE id=?", (clue_id,)).fetchone()
        if not row:
            print(f"  SKIP {clue_id}: not found"); continue
        se = conn.execute(
            "SELECT model_version FROM structured_explanations WHERE clue_id=?",
            (clue_id,)
        ).fetchone()
        if se and se['model_version'] in ('manual_edit', 'manual_approve'):
            print(f"  SKIP {clue_id}: protected as {se['model_version']}")
            continue
        if wtype == 'unparsed':
            verdict, score = 'LOW', 25
        else:
            v = verifier.verify(
                clue_text=row['clue_text'], answer=row['answer'],
                wordplay_type=wtype, definition=definition,
                ai_explanation=expl,
            )
            score = v.get('score', 0)
            verdict = v.get('verdict', 'FAIL')
        confidence = score / 100.0
        label = f"{row['clue_number']}{row['direction'][0]}"
        print(f"  [{verdict:6} {score:3}] {label:5} {row['answer']:14} (wtype={wtype})")
        tiers[verdict].append((label, row['answer'], score))
        if dry_run:
            continue
        components = json.dumps({
            "ai_pieces": [], "assembly": {"op": wtype},
            "wordplay_type": wtype, "source": "claude_review",
        })
        conn.execute(
            "UPDATE clues SET definition=?, wordplay_type=?, ai_explanation=?, "
            "has_solution=1, reviewed=1 WHERE id=?",
            (definition, wtype, expl, clue_id))
        conn.execute(
            "INSERT OR REPLACE INTO structured_explanations "
            "(clue_id, definition_text, wordplay_types, components, "
            " model_version, confidence, created_at, updated_at, "
            " source, puzzle_number, clue_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (clue_id, definition, json.dumps([wtype]), components,
             'claude_review', confidence, now, now,
             row['source'], row['puzzle_number'], row['clue_number']))

    # === Phase 3: Queue enrichments ===
    if not dry_run:
        for etype, word, letters, answer in to_queue:
            conn.execute(
                "INSERT OR IGNORE INTO pending_enrichments "
                "(type, word, letters, answer, source, puzzle_number, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (etype, word, letters, answer, 'telegraph', '31238', now))
        conn.commit()

    print(f"\nTier summary:")
    for t in ['HIGH', 'MEDIUM', 'LOW', 'FAIL']:
        print(f"  {t:6}: {len(tiers[t])}")
    print(f"\nEnrichments queued: {0 if dry_run else len(to_queue)}")
    conn.close()


if __name__ == '__main__':
    main()
