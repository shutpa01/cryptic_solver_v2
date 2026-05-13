"""Store Times 29540 leftover parses + queue genuine DB-miss enrichments.

Workflow:
1. For each parse: run verifier, capture verdict/score
2. INSERT OR REPLACE into structured_explanations (model_version='claude_review')
3. UPDATE clues table (definition / wordplay_type / ai_explanation / flags)
4. Queue genuine enrichment misses to pending_enrichments
5. Print tier summary

Manual-edit rows are skipped (none expected in Times 29540, but checked).
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
from scripts._times29540_parses import CLUES

# Enrichments to queue. Each tuple: (type, word, letters/answer, optional answer override)
# These were validated against the DB on 2026-05-12 and are genuine misses.
# type values: 'synonym' / 'abbreviation' / 'definition' / 'homophone' / 'indicator'
# For 'indicator' rows, the wordplay type (anagram, reversal, container, ...) goes in `letters`.
ENRICHMENTS = [
    # ===== Synonyms =====
    ('synonym', 'funny show', 'SITCOM', 'SITCOM'),
    ('synonym', 'something to play', 'GAME', 'GAMELAN'),
    ('synonym', 'girl, possibly', 'ISSUE', 'REISSUE'),
    ('synonym', 'one taking a part', 'ACTOR', 'EXACTOR'),
    ('synonym', 'Raised mark', 'WELT', 'DWELT'),
    ('synonym', '1', 'I', 'IMMERSE'),
    ('synonym', 'this writer', 'I', 'IMAGINE'),
    ('synonym', 'witnessing', 'AT', 'ATHEIST'),
    ('synonym', 'gruesome', 'GRISLY', 'GRIZZLYBEAR'),
    ('synonym', 'display', 'BARE', 'GRIZZLYBEAR'),
    ('synonym', 'I\'m relieved', 'PHEW', 'FEW'),
    # ===== Abbreviations =====
    ('abbreviation', 'millimetre', 'MM', 'IMMERSE'),
    # ===== Definitions =====
    ('definition', 'Split', 'GRASS', 'GRASS'),
    ('definition', 'related to nutrition', 'TROPHIC', 'TROPHIC'),
    ('definition', 'mixed type', 'PIE', 'PIE'),
    ('definition', 'nothing special', 'ADIMEADOZEN', 'ADIMEADOZEN'),
    ('definition', 'Only a handful', 'FEW', 'FEW'),
    ('definition', 'a handful', 'FEW', 'FEW'),
    ('definition', 'secret language', 'ARGOT', 'ARGOT'),
    ('definition', 'Newsworthy story', 'MANBITESDOG', 'MANBITESDOG'),
    ('definition', 'Sponge', 'SOT', 'SOT'),
    ('definition', "I'm disgusted", 'GROSS', 'GROSS'),
    ('definition', 'Infidel', 'ATHEIST', 'ATHEIST'),
    ('definition', 'as ceilings, perhaps', 'PLASTERED', 'PLASTERED'),
    ('definition', 'Complex pattern', 'FRACTAL', 'FRACTAL'),
    ('definition', 'Hat', 'DERBY', 'DERBY'),
    ('definition', 'Drink', 'GINGERALE', 'GINGERALE'),
    # ===== Indicators =====
    ('indicator', 'opening', 'container', 'ADIMEADOZEN'),
    ('indicator', 'delivered', 'hidden', 'ETH'),
    ('indicator', 'delivered in', 'hidden', 'ETH'),
    # ===== Homophone pairs =====
    ('homophone', 'few', 'phew', 'FEW'),
    ('homophone', 'grizzly', 'grisly', 'GRIZZLYBEAR'),
]


def has_in_db(ref, etype, word, letters):
    """Return True if this enrichment row is already in DB (so we should NOT queue)."""
    w, L = word.lower(), letters.upper() if etype != 'homophone' and etype != 'indicator' and not etype.startswith('indicator') else letters
    if etype == 'synonym':
        r = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE "
            "(LOWER(word)=? AND UPPER(synonym)=?) OR "
            "(LOWER(word)=? AND UPPER(synonym)=?) LIMIT 1",
            (w, L, L.lower(), w.upper())
        ).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE "
            "LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (w, L)
        ).fetchone()
        return bool(r or r2)
    if etype == 'abbreviation':
        r = ref.execute(
            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND UPPER(substitution)=? LIMIT 1",
            (w, L)
        ).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
            (w, L)
        ).fetchone()
        return bool(r or r2)
    if etype == 'definition':
        r = ref.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (w, L)
        ).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
            (w, L)
        ).fetchone()
        return bool(r or r2)
    if etype == 'indicator':
        # `letters` carries the wordplay_type for indicator enrichments
        # (matches dashboard convention; type column stays 'indicator').
        r = ref.execute(
            "SELECT 1 FROM indicators WHERE LOWER(word)=? AND LOWER(wordplay_type)=? LIMIT 1",
            (word.lower(), letters.lower())
        ).fetchone()
        return bool(r)
    if etype == 'homophone':
        r = ref.execute(
            "SELECT 1 FROM homophones WHERE "
            "(LOWER(word)=? AND LOWER(homophone)=?) OR "
            "(LOWER(word)=? AND LOWER(homophone)=?) LIMIT 1",
            (w, letters.lower(), letters.lower(), w)
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

    # === Phase 1: Coverage check enrichments — keep only genuine misses ===
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
        print(f"  SKIP (in DB): {et:25} {w!r} -> {L!r}")
    print()

    # === Phase 2: Verify and store each clue ===
    results = []
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
                wordplay_type=wtype, definition=definition, ai_explanation=expl,
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
            # type field: synonym / abbreviation / definition / homophone / indicator
            qtype = etype
            conn.execute(
                "INSERT INTO pending_enrichments "
                "(type, word, letters, answer, source, puzzle_number, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (qtype, word, letters, answer, 'times', 29540, now))
        conn.commit()

    print(f"\nTier summary:")
    for t in ['HIGH', 'MEDIUM', 'LOW', 'FAIL']:
        print(f"  {t}: {len(tiers[t])}")
    print(f"\nEnrichments queued: {0 if dry_run else len(to_queue)}")
    conn.close()


if __name__ == '__main__':
    main()
