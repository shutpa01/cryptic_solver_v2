"""Store DT 31240 leftover parses + queue genuine DB-miss enrichments."""
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
from scripts._dt31240_parses import CLUES
from scripts.leftover_validation import validate_definition

ENRICHMENTS = [
    # Definitions
    ('definition', 'Swindle Geordie vocalist', 'STING', 'STING'),
    ('definition', 'Swindle', 'STING', 'STING'),
    ('definition', 'Geordie vocalist', 'STING', 'STING'),
    ('definition', 'Bill, able to sing well', 'INVOICE', 'INVOICE'),
    ('definition', 'able to sing well', 'INVOICE', 'INVOICE'),
    ('definition', 'Country retail store', 'ICELAND', 'ICELAND'),
    ('definition', 'Disown', 'REPUDIATE', 'REPUDIATE'),
    ('definition', 'Disown and reject', 'REPUDIATE', 'REPUDIATE'),
    ('definition', 'hymn tune', 'CHORALE', 'CHORALE'),
    ('definition', 'final victory', 'LASTLAUGH', 'last laugh'),
    ('definition', 'empty', 'INANE', 'INANE'),
    ('definition', 'to be empty', 'INANE', 'INANE'),
    ('definition', 'towel perhaps', 'DRYER', 'DRYER'),
    ('definition', 'Consider', 'ENTERTAIN', 'ENTERTAIN'),
    ('definition', "Trump's break", 'RECESS', 'RECESS'),
    ('definition', 'Reservation', 'PROVISO', 'PROVISO'),
    ('definition', 'Maybe one refusing beer', 'DRAUGHTEXCLUDER', 'draught excluder'),
    ('definition', 'Puzzling', 'ENIGMATIC', 'ENIGMATIC'),
    ('definition', 'dog', 'GOLDENRETRIEVER', 'golden retriever'),
    ('definition', 'retail store', 'ICELAND', 'ICELAND'),
    ('definition', 'Mess up', 'DISHEVEL', 'DISHEVEL'),
    ('definition', 'wealth', 'AFFLUENCE', 'AFFLUENCE'),
    ('definition', 'greatly admired', 'IDOLISED', 'IDOLISED'),
    ('definition', 'Italian region', 'TUSCANY', 'TUSCANY'),
    ('definition', 'musical instrument', 'OCARINA', 'OCARINA'),
    ('definition', 'Banker', 'SEVERN', 'SEVERN'),
    ('definition', 'Raise', 'HOIST', 'HOIST'),
    ('definition', 'cut short', 'CONTRACT', 'CONTRACT'),
    ('definition', 'dental work', 'BRIDGE', 'BRIDGE'),
    ('definition', 'Substitute', 'STANDIN', 'stand-in'),
    ('definition', 'Card game', 'CONTRACTBRIDGE', 'contract bridge'),

    # Synonyms
    ('synonym', 'charity', 'AID', 'REPUDIATE'),
    ('synonym', 'celebrity', 'REPUTE', 'REPUDIATE'),
    ('synonym', 'Surrey town', 'GUILDFORD', 'GUILD'),
    ('synonym', 'pen', 'CORAL', 'CHORALE'),
    ('synonym', 'encountered', 'MET', 'TEMPERATE'),
    ('synonym', 'salesperson', 'REP', 'TEMPERATE'),
    ('synonym', 'Greek character', 'ETA', 'TEMPERATE'),
    ('synonym', 'cut short', 'CONTRACT', 'CONTRACTBRIDGE'),
    ('synonym', 'dental work', 'BRIDGE', 'CONTRACTBRIDGE'),
    ('synonym', 'plus', 'AND', 'STANDIN'),
    ('synonym', 'one', 'I', 'STANDIN'),
    ('synonym', 'able to sing well', 'INVOICE', 'INVOICE'),
    ('synonym', 'one staining', 'DYER', 'DRYER'),
    ('synonym', 'record', 'ENTER', 'ENTERTAIN'),
    ('synonym', 'on', 'RE', 'RECESS'),
    ('synonym', 'Geordie vocalist', 'STING', 'STING'),
    ('synonym', 'First Lady', 'EVE', 'DISHEVEL'),
    ('synonym', 'food course', 'DISH', 'DISHEVEL'),
    ('synonym', 'always', 'EVER', 'SEVERN'),
    ('synonym', 'army', 'HOST', 'HOIST'),
    ('synonym', 'is able to', 'CAN', 'TUSCANY'),
    ('synonym', 'American', 'US', 'TUSCANY'),
    ('synonym', 'elderly', 'OLD', 'GOLDENRETRIEVER'),
    ('synonym', 'information', 'GEN', 'GOLDENRETRIEVER'),

    # Abbreviations
    ('abbreviation', 'daughter', 'D', 'GUILD'),
    ('abbreviation', 'ship', 'SS', 'RECESS'),
    ('abbreviation', 'right', 'R', 'DRYER'),
    ('abbreviation', 'left', 'L', 'DISHEVEL'),
    ('abbreviation', 'Poles', 'S', 'SEVERN'),
    ('abbreviation', 'Poles', 'N', 'SEVERN'),

    # Indicators
    ('indicator', 'missing', 'deletion', 'GUILD'),
    ('indicator', 'reportedly', 'homophone', 'CHORALE'),
    ('indicator', 'travelling west', 'reversal', 'TEMPERATE'),
    ('indicator', 'Battling', 'anagram', 'LASTLAUGH'),
    ('indicator', 'Regularly', 'alternating', 'OCARINA'),
    ('indicator', 'to stop', 'container', 'DRYER'),
    ('indicator', 'to defend', 'container', 'HOIST'),
    ('indicator', 'composed', 'anagram', 'AFFLUENCE'),
    ('indicator', 'Wasting', 'deletion', 'AFFLUENCE'),
    ('indicator', 'stripped', 'deletion', 'INANE'),
    ('indicator', 'wingers', 'parts', 'STANDIN'),
    ('indicator', 'Suspect', 'anagram', 'IDOLISED'),
    ('indicator', 'needing vacation', 'deletion', 'RECESS'),
    ('indicator', 'inspired', 'anagram', 'REPUDIATE'),
    ('indicator', 'adopting', 'container', 'GOLDENRETRIEVER'),
]


def has_in_db(ref, etype, word, letters):
    w = word.lower()
    if etype == 'synonym':
        r = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
            (w, letters.upper())).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (w, letters.upper())).fetchone()
        return bool(r or r2)
    if etype == 'abbreviation':
        r = ref.execute(
            "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND UPPER(substitution)=? LIMIT 1",
            (w, letters.upper())).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=? LIMIT 1",
            (w, letters.upper())).fetchone()
        return bool(r or r2)
    if etype == 'definition':
        r = ref.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (w, letters.upper())).fetchone()
        return bool(r)
    if etype == 'indicator':
        r = ref.execute(
            "SELECT 1 FROM indicators WHERE LOWER(word)=? AND LOWER(wordplay_type)=? LIMIT 1",
            (w, letters.lower())).fetchone()
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

    to_queue = []
    skipped_in_db = []
    for etype, word, letters, answer in ENRICHMENTS:
        if has_in_db(ref, etype, word, letters):
            skipped_in_db.append((etype, word, letters))
        else:
            to_queue.append((etype, word, letters, answer))
    print(f"Enrichments: {len(to_queue)} genuine misses, "
          f"{len(skipped_in_db)} already in DB (skipped)\n")

    tiers = {'HIGH': [], 'MEDIUM': [], 'LOW': [], 'FAIL': []}
    for clue_id, wtype, definition, expl in CLUES:
        row = conn.execute("SELECT * FROM clues WHERE id=?", (clue_id,)).fetchone()
        if not row:
            print(f"  SKIP {clue_id}: not found"); continue
        se = conn.execute(
            "SELECT model_version FROM structured_explanations WHERE clue_id=?",
            (clue_id,)).fetchone()
        if se and se['model_version'] in ('manual_edit', 'manual_approve'):
            print(f"  SKIP {clue_id}: protected as {se['model_version']}")
            continue
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
        print(f"  [{verdict:6} {score:3}] {label:5} {row['answer']:20} (wtype={wtype})")
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
            """INSERT OR REPLACE INTO structured_explanations
               (clue_id, definition_text, wordplay_types, components,
                model_version, confidence, created_at, updated_at)
               VALUES (?,?,?,?,?,?,?,?)""",
            (clue_id, definition, wtype, components,
             'claude_review', confidence, now, now))

    if not dry_run:
        # Queue genuine enrichments
        queued = 0
        for etype, word, letters, answer in to_queue:
            conn.execute(
                """INSERT OR IGNORE INTO pending_enrichments
                   (type, word, letters, answer, source, puzzle_number)
                   VALUES (?,?,?,?,?,?)""",
                (etype, word, letters, answer, 'telegraph', '31240'))
            queued += 1
        conn.commit()
        print(f"\nQueued {queued} enrichments to pending_enrichments")

    print(f"\n=== SUMMARY ===")
    for tier in ['HIGH', 'MEDIUM', 'LOW', 'FAIL']:
        if tiers[tier]:
            names = ', '.join(f"{l}({s})" for l, a, s in tiers[tier])
            print(f"  {tier:6}: {len(tiers[tier]):2}  {names}")

    if dry_run:
        print(f"\nEnrichments that would be queued ({len(to_queue)}):")
        for etype, word, letters, answer in to_queue:
            print(f"  {etype:12} {word!r:35} -> {letters}")


if __name__ == '__main__':
    main()
