"""Store DT 31239 leftover parses + queue genuine DB-miss enrichments.

Mirrors scripts/_dt31238_store.py.
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
from scripts._dt31239_parses import CLUES
from scripts.leftover_validation import validate_definition

# Enrichment candidates. Each (type, word, letters/wordplay_type, answer)
# Coverage-check filters to genuine misses only.
ENRICHMENTS = [
    # ===== Definitions =====
    ('definition', 'writers', 'BALLPOINTPENS', 'ballpoint pens'),
    ('definition', 'old Italians', 'ETRUSCANS', 'Etruscans'),
    ('definition', 'Goods', 'CARGO', 'CARGO'),
    ('definition', "that's a bloomer", 'IRIS', 'IRIS'),
    ('definition', 'food shop', 'DELI', 'DELI'),
    ('definition', 'bony frame', 'RIBCAGE', 'RIBCAGE'),
    ('definition', 'Areas for e.g. football', 'GROUNDS', 'GROUNDS'),
    ('definition', "One's in a sticky situation up here", 'GUMTREE', 'gum tree'),
    ('definition', 'Meat', 'VENISON', 'VENISON'),
    ('definition', 'Perhaps trunk', 'NOSE', 'NOSE'),
    ('definition', 'Shrub', 'ACER', 'Acer'),
    ('definition', 'Change', 'ALTER', 'ALTER'),
    ('definition', 'vulnerable', 'FRAIL', 'FRAIL'),
    ('definition', 'These people', 'PRISONERS', 'PRISONERS'),
    ('definition', 'eatery', 'TRANSPORTCAFE', 'transport cafe'),
    ('definition', 'another dish', 'BEEFSTROGANOFF', 'beef Stroganoff'),
    ('definition', 'Actress', 'LOREN', 'Loren'),
    ('definition', 'vehicles for small charges', 'PUSHCHAIRS', 'PUSHCHAIRS'),
    ('definition', 'fancy', 'IMAGINE', 'IMAGINE'),
    ('definition', 'Sampling', 'TASTING', 'TASTING'),
    ('definition', 'Cut', 'ETCH', 'ETCH'),
    ('definition', 'Officers', 'SERGEANTS', 'SERGEANTS'),
    ('definition', 'One meeting Friday in work', 'ROBINSONCRUSOE', 'Robinson Crusoe'),
    ('definition', 'Fairground Attraction', 'ROUNDABOUT', 'ROUNDABOUT'),
    ('definition', 'car needing this', 'BUMPSTART', 'bump-start'),
    ('definition', 'Bolts', 'ESCAPES', 'ESCAPES'),
    ('definition', 'Picture', 'VERTIGO', 'VERTIGO'),
    ('definition', 'character from Athens', 'THETA', 'THETA'),
    ('definition', 'Small jumper', 'FLEA', 'FLEA'),

    # ===== Synonyms (multi-word phrases that won't be in synonyms_pairs) =====
    ('synonym', 'Stalls', 'PENS', 'BALLPOINTPENS'),
    ('synonym', 'PM Liz', 'TRUSS', 'ETRUSCANS'),
    ('synonym', 'from Cork perhaps', 'IRISH', 'IRIS'),
    ('synonym', 'was in front', 'LED', 'DELI'),
    ('synonym', 'Oscar winner', 'CAGE', 'RIBCAGE'),
    ('synonym', 'games of golf', 'ROUNDS', 'GROUNDS'),
    ('synonym', 'Strong emotion', 'TRANSPORT', 'TRANSPORTCAFE'),
    ('synonym', 'unavailable', 'OFF', 'BEEFSTROGANOFF'),
    ('synonym', "president's", 'CHAIRS', 'PUSHCHAIRS'),
    ('synonym', 'Police singer', 'STING', 'TASTING'),
    ('synonym', 'water bird', 'RAIL', 'FRAIL'),
    ('synonym', 'Two articles', 'THE', 'THETA'),
    ('synonym', 'Two articles', 'A', 'THETA'),

    # ===== Indicators =====
    ('indicator', 'carried', 'hidden', 'CARGO'),
    ('indicator', 'looking back', 'reversal', 'DELI'),
    ('indicator', 'now and then', 'parts', 'TASTING'),
    ('indicator', 'having lost skin', 'deletion', 'ETCH'),
    ('indicator', 'to be too short', 'deletion', 'FLEA'),
    ('indicator', 'seen by', 'parts', 'ACER'),
    ('indicator', 'occasionally', 'parts', 'PRISONERS'),
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
        # Cheat-blocker: definition must be a contiguous substring of the
        # clue and must sit at one of the clue's edges (start or end).
        ok, reason = validate_definition(row['clue_text'], definition)
        if not ok:
            print(f"  BLOCK {clue_id} ({row['answer']}): {reason}")
            continue
        if wtype == 'unparsed':
            verdict, score = 'LOW', 25
        else:
            v = verifier.verify(
                clue_text=row['clue_text'], answer=row['answer'],
                wordplay_type=wtype, definition=definition,
                ai_explanation=expl,
                clue_id=clue_id, db_conn=conn,
            )
            score = v.get('score', 0)
            verdict = v.get('verdict', 'FAIL')
        confidence = score / 100.0
        label = f"{row['clue_number']}{row['direction'][0]}"
        print(f"  [{verdict:6} {score:3}] {label:5} {row['answer']:18} (wtype={wtype})")
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
                (etype, word, letters, answer, 'telegraph', '31239', now))
        conn.commit()

    print(f"\nTier summary:")
    for t in ['HIGH', 'MEDIUM', 'LOW', 'FAIL']:
        print(f"  {t:6}: {len(tiers[t])}")
    print(f"\nEnrichments queued: {0 if dry_run else len(to_queue)}")
    conn.close()


if __name__ == '__main__':
    main()
