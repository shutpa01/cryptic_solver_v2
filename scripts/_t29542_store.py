"""Store Times 29542 leftover parses + queue genuine DB-miss enrichments."""
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
from scripts._t29542_parses import CLUES
from scripts.leftover_validation import validate_definition

ENRICHMENTS = [
    # ===== Definitions =====
    ('definition', 'licentious jokes', 'RIBALDRY', 'RIBALDRY'),
    ('definition', 'Always a cool head', 'ICECAP', 'ice cap'),
    ('definition', 'famous picture', 'GONEWITHTHEWIND', 'Gone with the Wind'),
    ('definition', 'Ideal', 'UTOPIAN', 'UTOPIAN'),
    ('definition', 'Goddess', 'NEMESIS', 'NEMESIS'),
    ('definition', 'Religious rite', 'ASPERGES', 'ASPERGES'),
    ('definition', 'side', 'RAITA', 'RAITA'),
    ('definition', 'artist', 'ERNST', 'ERNST'),
    ('definition', 'Thunder', 'RESONATE', 'RESONATE'),
    ('definition', 'dish', 'LASAGNE', 'LASAGNE'),
    ('definition', 'holiday island', 'MINORCA', 'MINORCA'),
    ('definition', 'are in control', 'WEARTHETROUSERS', 'wear the trousers'),
    ('definition', 'spicy food', 'RAGOUT', 'RAGOUT'),
    ('definition', 'dramatist', 'BENJONSON', 'Ben Jonson'),
    ('definition', 'criminal?', 'LOWLIFE', 'LOWLIFE'),
    ('definition', 'shop', 'RATON', 'rat on'),
    ('definition', 'milk substitute', 'CREAMER', 'CREAMER'),
    ('definition', 'Carpenter', 'CHIPS', 'CHIPS'),
    ('definition', 'base', 'PEDESTAL', 'PEDESTAL'),
    ('definition', 'Constriction', 'STENOSIS', 'STENOSIS'),
    ('definition', 'Worker at Versailles possibly', 'GARDENER', 'GARDENER'),
    ('definition', 'Barge in', 'INTERFERE', 'INTERFERE'),
    ('definition', 'Bull', 'BELLOWER', 'BELLOWER'),
    ('definition', 'Faster', 'TIGHTER', 'TIGHTER'),
    ('definition', 'Previously unrivalled thing', 'NONSUCH', 'NONSUCH'),
    ('definition', 'Upright', 'SHAFT', 'SHAFT'),
    ('definition', 'seas', 'MARIA', 'MARIA'),

    # ===== Synonyms =====
    ('synonym', 'boring', 'DRY', 'RIBALDRY'),
    ('synonym', 'step', 'PACE', 'ICECAP'),
    ('synonym', 'best', 'TOP', 'UTOPIAN'),
    ('synonym', 'Magi', 'WISEMEN', 'NEMESIS'),
    ('synonym', 'snake', 'ASP', 'ASPERGES'),
    ('synonym', 'encourages', 'URGES', 'ASPERGES'),
    ('synonym', 'inferior cricketer', 'RABBIT', 'RAITA'),
    ('synonym', 'beak', 'NOSE', 'RESONATE'),
    ('synonym', 'judge', 'RATE', 'RESONATE'),
    ('synonym', 'way', 'LANE', 'LASAGNE'),
    ('synonym', 'spinach', 'SAG', 'LASAGNE'),
    ('synonym', 'popular', 'IN', 'MINORCA'),
    ('synonym', 'killer', 'ORCA', 'MINORCA'),
    ('synonym', 'pathetic', 'WET', 'WEARTHETROUSERS'),
    ('synonym', 'causing stir', 'ROUSERS', 'WEARTHETROUSERS'),
    ('synonym', 'Globe', 'EARTH', 'WEARTHETROUSERS'),
    ('synonym', 'kid', 'RAG', 'RAGOUT'),
    ('synonym', 'not allowed', 'OUT', 'RAGOUT'),
    ('synonym', 'good French', 'BON', 'BENJONSON'),
    ('synonym', 'orders', 'ENJOINS', 'BENJONSON'),
    ('synonym', 'one', 'I', 'BENJONSON'),
    ('synonym', 'prison sentence', 'LIFE', 'LOWLIFE'),
    ('synonym', 'bird', 'OWL', 'LOWLIFE'),
    ('synonym', 'hurried', 'RAN', 'RATON'),
    ('synonym', 'one shrieking', 'SCREAMER', 'CREAMER'),
    ('synonym', 'lives', 'IS', 'CHIPS'),
    ('synonym', 'operate bike', 'PEDAL', 'PEDESTAL'),
    ('synonym', 'forest', 'ARDEN', 'GARDENER'),
    ('synonym', 'bury', 'INTER', 'INTERFERE'),
    ('synonym', 'hit', 'BELT', 'BELLOWER'),
    ('synonym', 'cow', 'LOWER', 'BELLOWER'),
    ('synonym', 'animal', 'TIGER', 'TIGHTER'),
    ('synonym', 'not in Rome', 'NON', 'NONSUCH'),
    ('synonym', 'so great', 'SUCH', 'NONSUCH'),
    ('synonym', 'fedora', 'HAT', 'SHAFT'),
    ('synonym', 'yacht sanctuary', 'MARINA', 'MARIA'),

    # ===== Abbreviations =====
    ('abbreviation', 'in charge', 'IC', 'ICECAP'),
    ('abbreviation', 'area', 'A', 'UTOPIAN'),
    ('abbreviation', 'saint', 'S', 'NEMESIS'),
    ('abbreviation', 'maidens', 'M', 'MINORCA'),
    ('abbreviation', 'son', 'S', 'CREAMER'),
    ('abbreviation', 'church', 'CH', 'CHIPS'),
    ('abbreviation', 'power', 'P', 'CHIPS'),
    ('abbreviation', 'established', 'EST', 'PEDESTAL'),
    ('abbreviation', 'Germany', 'GER', 'GARDENER'),
    ('abbreviation', 'hard', 'H', 'TIGHTER'),
    ('abbreviation', 'time', 'T', 'TIGHTER'),
    ('abbreviation', 'small', 'S', 'SHAFT'),
    ('abbreviation', 'female', 'F', 'SHAFT'),
    ('abbreviation', 'November', 'N', 'MARIA'),

    # ===== Indicators =====
    ('indicator', 'cracked', 'anagram', 'RIBALDRY'),
    ('indicator', 'swimming', 'anagram', 'GONEWITHTHEWIND'),
    ('indicator', 'surrounding', 'container', 'UTOPIAN'),
    ('indicator', 'from the East', 'reversal', 'NEMESIS'),
    ('indicator', 'not West', 'deletion', 'NEMESIS'),
    ('indicator', 'vocally', 'homophone', 'ASPERGES'),
    ('indicator', 'leaves', 'deletion', 'RAITA'),
    ('indicator', 'periodically', 'parts', 'ERNST'),
    ('indicator', 'returning', 'reversal', 'RESONATE'),
    ('indicator', 'welcomes', 'container', 'RESONATE'),
    ('indicator', 'to get to grips with', 'container', 'LASAGNE'),
    ('indicator', 'outside', 'container', 'WEARTHETROUSERS'),
    ('indicator', 'taken by', 'container', 'BENJONSON'),
    ('indicator', 'missed', 'deletion', 'BENJONSON'),
    ('indicator', 'involving', 'container', 'LOWLIFE'),
    ('indicator', 'round', 'container', 'RATON'),
    ('indicator', 'denied', 'deletion', 'CREAMER'),
    ('indicator', 'going round', 'container', 'PEDESTAL'),
    ('indicator', 'badly', 'anagram', 'STENOSIS'),
    ('indicator', 'planting', 'container', 'GARDENER'),
    ('indicator', 'rent', 'anagram', 'INTERFERE'),
    ('indicator', 'almost', 'deletion', 'BELLOWER'),
    ('indicator', 'embracing', 'container', 'TIGHTER'),
    ('indicator', 'keeps out', 'deletion', 'MARIA'),
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
            "INSERT OR REPLACE INTO structured_explanations "
            "(clue_id, definition_text, wordplay_types, components, "
            " model_version, confidence, created_at, updated_at, "
            " source, puzzle_number, clue_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (clue_id, definition, json.dumps([wtype]), components,
             'claude_review', confidence, now, now,
             row['source'], row['puzzle_number'], row['clue_number']))

    if not dry_run:
        for etype, word, letters, answer in to_queue:
            conn.execute(
                "INSERT OR IGNORE INTO pending_enrichments "
                "(type, word, letters, answer, source, puzzle_number, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (etype, word, letters, answer, 'times', '29542', now))
        conn.commit()

    print(f"\nTier summary:")
    for t in ['HIGH', 'MEDIUM', 'LOW', 'FAIL']:
        print(f"  {t:6}: {len(tiers[t])}")
        if tiers[t]:
            for label, answer, score in tiers[t]:
                print(f"    {label} {answer} ({score})")
    print(f"\nEnrichments queued: {0 if dry_run else len(to_queue)}")
    conn.close()


if __name__ == '__main__':
    main()
