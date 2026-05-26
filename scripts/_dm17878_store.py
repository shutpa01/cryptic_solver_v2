"""Store DM 17878 leftover parses + queue genuine DB-miss enrichments."""
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
from scripts._dm17878_parses import CLUES
from scripts.leftover_validation import validate_definition

ENRICHMENTS = [
    # ===== Definitions (multi-word / proper / non-standard) =====
    ('definition', 'West African country', 'BURKINAFASO', 'Burkina Faso'),
    ('definition', 'Goes by', 'ELAPSES', 'ELAPSES'),
    ('definition', 'Do better than', 'OUTSELL', 'OUTSELL'),
    ('definition', 'Thing to hold flowers', 'URN', 'URN'),
    ('definition', 'Frill', 'FLOUNCE', 'FLOUNCE'),
    ('definition', 'part of the day', 'EVENING', 'EVENING'),
    ('definition', 'Sign of positivity', 'YES', 'YES'),
    ('definition', 'Terrible', 'AWFUL', 'AWFUL'),
    ('definition', 'spectacle', 'SIGHT', 'SIGHT'),
    ('definition', 'titled figures', 'EARLS', 'EARLS'),
    ('definition', 'More intense', 'RAWER', 'RAWER'),
    ('definition', 'Musical aptitude', 'EAR', 'EAR'),
    ('definition', 'woman', 'CECILIA', 'CECILIA'),
    ('definition', 'most weak', 'LIMPEST', 'LIMPEST'),
    ('definition', 'Sound from Hereford', 'MOO', 'MOO'),
    ('definition', 'Supply', 'PROVIDE', 'PROVIDE'),
    ('definition', 'Discharge', 'GIVEOFF', 'give off'),
    ('definition', 'in a doubtful way', 'SCEPTICALLY', 'SCEPTICALLY'),
    ('definition', 'heavy-bodied snakes', 'BOACONSTRICTORS', 'boa constrictors'),
    ('definition', 'female', 'ROSANNA', 'Rosanna'),
    ('definition', 'a female', 'ROSANNA', 'Rosanna'),
    ('definition', 'contentious matter', 'ISSUE', 'ISSUE'),
    ('definition', 'in single operation', 'ATONEBLOW', 'at one blow'),
    ('definition', 'feature of deer', 'ANTLERS', 'ANTLERS'),
    ('definition', 'event', 'OPENINGCEREMONY', 'opening ceremony'),
    ('definition', 'Source of rings', 'BELFRY', 'BELFRY'),
    ('definition', 'Minimal', 'SLIGHT', 'SLIGHT'),
    ('definition', 'insult', 'SLIGHT', 'SLIGHT'),
    ('definition', 'field of activity', 'FIRMAMENT', 'FIRMAMENT'),
    ('definition', 'Omit', 'EXCEPT', 'EXCEPT'),
    ('definition', 'cross', 'SALTIRE', 'SALTIRE'),
    ('definition', 'Withdrawing', 'REMOVAL', 'REMOVAL'),
    ('definition', 'Approve', 'RATIFY', 'RATIFY'),
    ('definition', 'Sound reasoning', 'LOGIC', 'LOGIC'),

    # ===== Synonyms (multi-word / proper / non-standard) =====
    ('synonym', 'expensive', 'DEAR', 'EARLS'),
    ('synonym', 'chap', 'GEOFF', 'GIVEOFF'),
    ('abbreviation', 'four', 'IV', 'GIVEOFF'),
    ('abbreviation', 'diamonds', 'D', 'EARLS'),
    ('synonym', 'gap', 'LAPSE', 'ELAPSES'),
    ('synonym', 'rival', 'OUT', 'OUTSELL'),
    ('synonym', 'publicised period', 'SPELL', 'OUTSELL'),
    ('synonym', 'little weight', 'OUNCE', 'FLOUNCE'),
    ('synonym', 'Figure', 'SEVEN', 'EVENING'),
    ('synonym', 'location', 'SITE', 'SIGHT'),
    ('synonym', 'bishop', 'RR', 'RAWER'),
    ('synonym', 'wonderment', 'AWE', 'RAWER'),
    ('synonym', 'In France, this', 'CECI', 'CECILIA'),
    ('synonym', 'trouble', 'AIL', 'CECILIA'),
    ('synonym', 'Rude types', 'BOORS', 'BOACONSTRICTORS'),
    ('synonym', 'a pair of names', 'ANNA', 'ROSANNA'),
    ('synonym', 'First person', 'I', 'ISSUE'),
    ('synonym', 'take legal action', 'SUE', 'ISSUE'),
    ('synonym', 'last word', 'AMEN', 'FIRMAMENT'),
    ('synonym', 'former', 'EX', 'EXCEPT'),
    ('synonym', 'Seasoned sailor', 'SALT', 'SALTIRE'),
    ('synonym', 'fury', 'IRE', 'SALTIRE'),
    ('synonym', 'supply', 'FIT', 'RATIFY'),
    ('synonym', 'part of canal', 'LOCK', 'LOGIC'),

    # ===== Abbreviations =====
    ('abbreviation', 'work', 'OP', 'OPENINGCEREMONY'),
    ('abbreviation', 'sun', 'S', 'EVENING'),
    ('abbreviation', 'pressure', 'P', 'OUTSELL'),
    ('abbreviation', 'point', 'PT', 'EXCEPT'),
    ('abbreviation', 'railway', 'RY', 'RATIFY'),
    ('abbreviation', 'soldier', 'GI', 'LOGIC'),

    # ===== Indicators =====
    ('indicator', 'stolen', 'deletion', 'EARLS'),
    ('indicator', 'outwardly', 'parts', 'EARLS'),
    ('indicator', 'ordered', 'anagram', 'OPENINGCEREMONY'),
    ('definition', 'start of major event', 'OPENINGCEREMONY', 'opening ceremony'),
    ('indicator', 'Mention', 'homophone', 'SIGHT'),
    ('indicator', 'given treatment', 'anagram', 'LIMPEST'),
    ('indicator', 'liquid', 'anagram', 'PROVIDE'),
    ('indicator', 'no end of', 'deletion', 'REMOVAL'),
    ('indicator', 'with no end of', 'deletion', 'REMOVAL'),
    ('indicator', 'on the rise', 'reversal', 'RATIFY'),
    ('indicator', 'origin', 'first letter', 'ANTLERS'),
    ('indicator', 'full of', 'container', 'RAWER'),
    ('indicator', 'Sound', 'homophone', 'LOGIC'),
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
        # Cheat-blocker: definition must be a contiguous substring of the
        # clue and must sit at one of the clue's edges (start or end).
        # Stops definition substitution and definition-extension cheats.
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

    if not dry_run:
        for etype, word, letters, answer in to_queue:
            conn.execute(
                "INSERT OR IGNORE INTO pending_enrichments "
                "(type, word, letters, answer, source, puzzle_number, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (etype, word, letters, answer, 'dailymail', '17878', now))
        conn.commit()

    print(f"\nTier summary:")
    for t in ['HIGH', 'MEDIUM', 'LOW', 'FAIL']:
        print(f"  {t:6}: {len(tiers[t])}")
    print(f"\nEnrichments queued: {0 if dry_run else len(to_queue)}")
    conn.close()


if __name__ == '__main__':
    main()
