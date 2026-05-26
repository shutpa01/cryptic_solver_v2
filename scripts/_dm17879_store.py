"""Store DM 17879 leftover parses + queue genuine DB-miss enrichments."""
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
from scripts._dm17879_parses import CLUES
from scripts.leftover_validation import validate_definition

ENRICHMENTS = [
    # ===== Definitions =====
    ('definition', 'somewhere in Northern Ireland', 'LISBURN', 'LISBURN'),
    ('definition', 'Old coins', 'DUCATS', 'DUCATS'),
    ('definition', 'in final part of event?', 'TAILEND', 'tail end'),
    ('definition', 'Show liveliness in flash of light', 'SPARKLE', 'SPARKLE'),
    ('definition', 'regulation', 'LAW', 'LAW'),
    ('definition', 'in place to stop en route', 'STAGINGPOST', 'staging post'),
    ('definition', 'area adjoining a house', 'PATIO', 'PATIO'),
    ('definition', 'part of a residential block', 'STAIRWELL', 'STAIRWELL'),
    ('definition', 'typifying some errors?', 'ALTERABLE', 'ALTERABLE'),
    ('definition', 'Chap', 'CECII', 'CECII'),
    ('definition', 'Agree', 'SEEEYETOEYE', 'see eye to eye'),
    ('definition', 'another tree', 'ELM', 'ELM'),
    ('definition', 'Irish female', 'CAITLIN', 'CAITLIN'),
    ('definition', 'source of eucalyptus', 'GUMTREE', 'gum tree'),
    ('definition', 'governing body', 'SENATE', 'SENATE'),
    ('definition', 'Steals apples', 'SCRUMPS', 'SCRUMPS'),
    ('definition', 'Inadvertently disclose', 'LETSLIP', 'let slip'),
    ('definition', 'Conform to prevailing opinion', 'SWIMWITHTHETIDE', 'swim with the tide'),
    ('definition', 'Employ', 'USE', 'USE'),
    ('definition', 'get to vanish', 'DISSIPATE', 'DISSIPATE'),
    ('definition', 'Scottish male', 'CRAIG', 'CRAIG'),
    ('definition', 'Follow the example of', 'TAKEONESCUEFROM', 'take one\'s cue from'),
    ('definition', 'Relating to canines?', 'DENTAL', 'DENTAL'),
    ('definition', 'Comb', 'SCOUR', 'SCOUR'),
    ('definition', 'matter?', 'SUBSTANCE', 'SUBSTANCE'),
    ('definition', 'French region', 'ALSACE', 'ALSACE'),
    ('definition', 'Plods', 'LUMBERS', 'LUMBERS'),
    ('definition', 'port in the Black Sea', 'YALTA', 'YALTA'),
    ('definition', 'Damage', 'MAR', 'MAR'),

    # ===== Synonyms =====
    ('synonym', 'brand', 'BURN', 'LISBURN'),
    ('synonym', 'pipes', 'DUCTS', 'DUCATS'),
    ('synonym', 'learning', 'LORE', 'LAW'),
    ('synonym', 'group', 'TRIO', 'PATIO'),
    ('synonym', "look that's fixed", 'STARE', 'STAIRWELL'),
    ('synonym', 'properly', 'WELL', 'STAIRWELL'),
    ('synonym', 'additionally', 'ALSO', 'ALSACE'),
    ('synonym', 'expert', 'ACE', 'ALSACE'),
    ('synonym', 'constituency', 'SEAT', 'SENATE'),
    ('synonym', 'newspaper employee', 'SUB', 'SUBSTANCE'),
    ('synonym', 'opinion', 'STANCE', 'SUBSTANCE'),
    ('synonym', 'ordinary', 'LAY', 'YALTA'),
    ('synonym', 'wood in the U.S.', 'LUMBER', 'LUMBERS'),
    ('synonym', 'area for selling goods', 'MARKET', 'MAR'),
    ('synonym', 'type of service', 'LET', 'LETSLIP'),
    ('synonym', 'one in a field', 'SLIP', 'LETSLIP'),
    ('synonym', 'catalogue', 'LIST', 'LISBURN'),
    ('synonym', 'Show liveliness', 'SPARKLE', 'SPARKLE'),
    ('synonym', 'flash of light', 'SPARKLE', 'SPARKLE'),

    # ===== Abbreviations =====
    ('abbreviation', 'pressure', 'P', 'PATIO'),
    ('abbreviation', 'right', 'R', 'PATIO'),
    ('abbreviation', 'area', 'A', 'DUCATS'),
    ('abbreviation', 'name', 'N', 'SENATE'),
    ('abbreviation', 'son', 'S', 'LUMBERS'),

    # ===== Indicators =====
    ('indicator', 'reduced', 'deletion', 'LISBURN'),
    ('indicator', 'surrounded by', 'container', 'DUCATS'),
    ('indicator', 'out', 'anagram', 'TAILEND'),
    ('indicator', 'needing repair', 'anagram', 'STAGINGPOST'),
    ('indicator', 'away', 'deletion', 'PATIO'),
    ('indicator', 'Reportedly', 'homophone', 'STAIRWELL'),
    ('indicator', 'off', 'anagram', 'ALTERABLE'),
    ('indicator', 'some', 'hidden', 'CECII'),
    ('indicator', 'bringing back', 'reversal', 'CECII'),
    ('indicator', 'Intermittently', 'alternating', 'ELM'),
    ('indicator', 'replaced', 'anagram', 'CAITLIN'),
    ('indicator', 'possibly', 'anagram', 'GUMTREE'),
    ('indicator', 'beginning to', 'first letter', 'SENATE'),
    ('indicator', 'on and off', 'alternating', 'USE'),
    ('indicator', 'in trouble', 'anagram', 'DISSIPATE'),
    ('indicator', 'crashing', 'anagram', 'CRAIG'),
    ('indicator', 'initially', 'first letter', 'CRAIG'),
    ('indicator', 'in part', 'hidden', 'SCOUR'),
    ('indicator', 'snubbed', 'deletion', 'ALSACE'),
    ('indicator', 'Revolutionary', 'anagram', 'YALTA'),
    ('indicator', 'not half', 'deletion', 'MAR'),
    ('indicator', 'speaker', 'homophone', 'LAW'),
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
                ai_explanation=expl, clue_id=clue_id, db_conn=conn,
            )
            score = v.get('score', 0)
            verdict = v.get('verdict', 'FAIL')
        confidence = score / 100.0
        label = f"{row['clue_number']}{row['direction'][0]}"
        print(f"  [{verdict:6} {score:3}] {label:5} {row['answer']:22} (wtype={wtype})")
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
        queued = 0
        for etype, word, letters, answer in to_queue:
            conn.execute(
                """INSERT OR IGNORE INTO pending_enrichments
                   (type, word, letters, answer, source, puzzle_number)
                   VALUES (?,?,?,?,?,?)""",
                (etype, word, letters, answer, 'dailymail', '17879'))
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
            print(f"  {etype:12} {word!r:40} -> {letters}")


if __name__ == '__main__':
    main()
