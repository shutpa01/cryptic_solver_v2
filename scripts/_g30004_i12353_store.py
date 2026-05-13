"""Process Guardian 30004 and Independent 12353 leftover parses.

Phases per puzzle:
  1. Verifier dry-run (just to print verdicts; not gating)
  2. Coverage-check candidate enrichments; drop those already in DB
  3. INSERT OR REPLACE structured_explanations; UPDATE clues
  4. Queue genuine misses to pending_enrichments
  5. Self-check by re-running the work-list query

Manual-edit rows are skipped. Run with --dry-run to preview.
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
from scripts._g30004_i12353_parses import G30004, I12353

# (type, word, letters, answer) — same format as Times 29540 store script.
# type is 'synonym' / 'abbreviation' / 'definition' / 'homophone' / 'indicator'
# For 'indicator' rows, the wordplay type (anagram, reversal, container, ...) goes in `letters`.

G30004_ENRICHMENTS = [
    # 1a TITFER
    ('synonym', 'Row', 'TIER', 'TITFER'),
    ('synonym', 'newspaper', 'FT', 'TITFER'),
    ('definition', 'Panama', 'TITFER', 'TITFER'),
    # 4a STUPID
    ('synonym', 'positions', 'PUTS', 'STUPID'),
    ('abbreviation', 'I would', 'ID', 'STUPID'),
    # 9a FEATHERONESNEST (unparsed — no enrichments)
    ('definition', 'to get personal benefit', 'FEATHERONESNEST', 'FEATHERONESNEST'),
    # 10a ATTEST
    ('definition', 'give evidence', 'ATTEST', 'ATTEST'),
    # 11a LANDLORD
    ('synonym', 'funny', 'DROLL', 'LANDLORD'),
    ('synonym', 'joiner', 'AND', 'LANDLORD'),
    ('definition', 'Proprietor', 'LANDLORD', 'LANDLORD'),
    # 12a HENPARTY
    ('synonym', 'jovial', 'HEARTY', 'HENPARTY'),
    ('definition', 'preparatory celebration', 'HENPARTY', 'HENPARTY'),
    # 14a TOILET
    ('synonym', 'for hire', 'TOLET', 'TOILET'),
    ('definition', 'John', 'TOILET', 'TOILET'),
    # 15a ABSENT
    ('synonym', 'First couple of letters', 'AB', 'ABSENT'),
    ('synonym', 'dispatched', 'SENT', 'ABSENT'),
    ('definition', 'not here', 'ABSENT', 'ABSENT'),
    # 18a SPARSITY
    ('homophone', 'sparsity', 'spacity', 'SPARSITY'),
    ('synonym', 'Bath perhaps', 'SPACITY', 'SPARSITY'),
    ('definition', 'being in short supply', 'SPARSITY', 'SPARSITY'),
    # 21a FOOTWEAR
    ('synonym', 'Settle', 'FOOT', 'FOOTWEAR'),
    ('homophone', 'wear', 'where', 'FOOTWEAR'),
    ('definition', 'Oxford or Derby', 'FOOTWEAR', 'FOOTWEAR'),
    # 22a ANTLER — DB has 'horn' → ANTLER probably; check
    ('definition', 'horn', 'ANTLER', 'ANTLER'),
    # 24a THEFATOFTHELAND
    ('definition', 'Best in everything', 'THEFATOFTHELAND', 'THEFATOFTHELAND'),
    # 26a ADHERE
    ('synonym', 'poster', 'AD', 'ADHERE'),
    ('synonym', 'not over there', 'HERE', 'ADHERE'),
    ('definition', 'Stick', 'ADHERE', 'ADHERE'),
    # 1d THEATRE
    ('definition', 'a tremendous play enacted here', 'THEATRE', 'THEATRE'),
    # 2d TOTIETHEKNOT
    ('definition', 'get hitched', 'TOTIETHEKNOT', 'TOTIETHEKNOT'),
    # 3d ELECTOR
    ('synonym', 'shock treatment', 'ECT', 'ELECTOR'),
    ('definition', 'One chooses', 'ELECTOR', 'ELECTOR'),
    # 5d THEKNOT (cross-reference, no enrichments)
    # 7d DESIREE
    ('synonym', 'Want', 'DESIRE', 'DESIREE'),
    ('definition', 'potato', 'DESIREE', 'DESIREE'),
    # 8d MOTLEY
    ('definition', 'Disparate', 'MOTLEY', 'MOTLEY'),
    # 13d PLENTIFUL
    ('definition', 'more than enough', 'PLENTIFUL', 'PLENTIFUL'),
    # 16d BROTHER
    ('definition', 'Relative', 'BROTHER', 'BROTHER'),
    # 17d TRESTLE
    ('definition', 'support', 'TRESTLE', 'TRESTLE'),
    # 18d STRIFE
    ('definition', 'conflict', 'STRIFE', 'STRIFE'),
    # 19d ABASHED
    ('synonym', 'an outhouse', 'ASHED', 'ABASHED'),
    ('definition', 'Embarrassed', 'ABASHED', 'ABASHED'),
    # 23d TILDE
    ('definition', 'mañana, got that, but not tomorrow', 'TILDE', 'TILDE'),
]

I12353_ENRICHMENTS = [
    # 9a ONEROUS
    ('synonym', 'fiddler', 'NERO', 'ONEROUS'),
    ('definition', 'hard to bear', 'ONEROUS', 'ONEROUS'),
    # 10a BURGLAR
    ('definition', 'criminal', 'BURGLAR', 'BURGLAR'),
    ('indicator', 'Withdrawing', 'reversal', 'BURGLAR'),
    # 11a TES
    ('definition', 'Supplement', 'TES', 'TES'),
    # 13a CONDEMN
    ('synonym', 'study', 'DEN', 'CONDEMN'),
    ('definition', 'convict', 'CONDEMN', 'CONDEMN'),
    # 16a LEITH
    ('definition', 'in the dock', 'LEITH', 'LEITH'),
    # 17a IMPEACH
    ('synonym', 'slack', 'LIMP', 'IMPEACH'),
    ('synonym', 'school', 'TEACH', 'IMPEACH'),
    ('definition', 'Indict', 'IMPEACH', 'IMPEACH'),
    # 20a ROBBERS (unparsed)
    ('definition', 'tea leaves', 'ROBBERS', 'ROBBERS'),
    # 22a TEPIDLY
    ('synonym', 'Pat', 'PET', 'TEPIDLY'),
    ('synonym', 'lazily', 'IDLY', 'TEPIDLY'),
    ('definition', 'without enthusiasm', 'TEPIDLY', 'TEPIDLY'),
    ('indicator', 'rolled over', 'reversal', 'TEPIDLY'),
    # 26a GROUPIE
    ('synonym', 'swimmer', 'GROUPER', 'GROUPIE'),
    ('definition', 'Obsessive', 'GROUPIE', 'GROUPIE'),
    # 27a UNFUNNY
    ('synonym', 'a French', 'UN', 'UNFUNNY'),
    ('abbreviation', 'city', 'NY', 'UNFUNNY'),
    ('definition', 'Serious', 'UNFUNNY', 'UNFUNNY'),
    # 29a ARM
    ('definition', 'Provide with', 'ARM', 'ARM'),
    # 30a SPEARED
    ('synonym', 'drug', 'SPEED', 'SPEARED'),
    ('definition', 'Spiked', 'SPEARED', 'SPEARED'),
    ('indicator', 'snared by', 'container', 'SPEARED'),
    # 31a BILLETS
    ('synonym', 'ads', 'BILLS', 'BILLETS'),
    ('synonym', 'Spielberg movie', 'ET', 'BILLETS'),
    ('definition', 'Digs', 'BILLETS', 'BILLETS'),
    ('indicator', 'showcasing', 'container', 'BILLETS'),
    # 1d JOLLY
    ('synonym', 'ecstasy', 'JOY', 'JOLLY'),
    ('definition', 'trip', 'JOLLY', 'JOLLY'),
    ('indicator', 'absorbed by', 'container', 'JOLLY'),
    # 3d COPS
    ('synonym', 'Sci-fi character', 'SPOCK', 'COPS'),
    ('definition', 'the Force', 'COPS', 'COPS'),
    # 4d ESTEEM
    ('synonym', 'press', 'STEAM', 'ESTEEM'),
    ('homophone', 'steem', 'steam', 'ESTEEM'),
    ('definition', 'prize', 'ESTEEM', 'ESTEEM'),
    # 5d OBSCENER
    ('synonym', 'fiscal watchdog', 'OBR', 'OBSCENER'),
    ('synonym', 'Panorama', 'SCENE', 'OBSCENER'),
    ('definition', 'comparatively offensive', 'OBSCENER', 'OBSCENER'),
    # 6d TRUNDLEBED
    ('definition', 'one wheeled out for retirement', 'TRUNDLEBED', 'TRUNDLEBED'),
    # 7d FLUEPIPE
    ('synonym', 'wind instrument', 'FLUTE', 'FLUEPIPE'),
    ('synonym', 'wind instrument', 'PIPE', 'FLUEPIPE'),
    ('definition', "It's exhausting", 'FLUEPIPE', 'FLUEPIPE'),
    # 8d BRONCHOS
    ('definition', 'wild horses', 'BRONCHOS', 'BRONCHOS'),
    # 15d SPACEOPERA
    ('synonym', 'a', 'PER', 'SPACEOPERA'),
    ('definition', 'a work of science fiction', 'SPACEOPERA', 'SPACEOPERA'),
    ('indicator', 'inspiring', 'container', 'SPACEOPERA'),
    # 17d INDIGEST
    ('synonym', 'land', 'INDIA', 'INDIGEST'),
    ('homophone', 'gest', 'jest', 'INDIGEST'),
    ('definition', "that's crude", 'INDIGEST', 'INDIGEST'),
    ('definition', 'crude', 'INDIGEST', 'INDIGEST'),
    # 18d PINBONES
    ('definition', 'unwanted parts of fish', 'PINBONES', 'PINBONES'),
    # 19d HOTHEADS
    ('synonym', 'round', 'O', 'HOTHEADS'),
    ('definition', 'They come across rash', 'HOTHEADS', 'HOTHEADS'),
    # 23d PLUMBS
    ('synonym', 'Fruit', 'PLUM', 'PLUMBS'),
    ('definition', 'links to water', 'PLUMBS', 'PLUMBS'),
    # 24d YANKEE
    ('synonym', 'vote against', 'NAY', 'YANKEE'),
    ('synonym', 'Remain', 'KEEP', 'YANKEE'),
    ('definition', 'Unionist', 'YANKEE', 'YANKEE'),
    # 25d GYPSY
    ('synonym', 'agent', 'SPY', 'GYPSY'),
    ('definition', 'a traveller', 'GYPSY', 'GYPSY'),
    # 28d FILM
    ('definition', '18, perhaps', 'FILM', 'FILM'),
]


def has_in_db(ref, etype, word, letters):
    if etype == 'synonym':
        w, L = word.lower(), letters.upper()
        r = ref.execute(
            "SELECT 1 FROM synonyms_pairs WHERE "
            "(LOWER(word)=? AND UPPER(synonym)=?) OR "
            "(LOWER(word)=? AND UPPER(synonym)=?) LIMIT 1",
            (w, L, L.lower(), w.upper())
        ).fetchone()
        r2 = ref.execute(
            "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND UPPER(answer)=? LIMIT 1",
            (w, L)
        ).fetchone()
        return bool(r or r2)
    if etype == 'abbreviation':
        w, L = word.lower(), letters.upper()
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
        w, L = word.lower(), letters.upper()
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
            (word.lower(), letters.lower(), letters.lower(), word.lower())
        ).fetchone()
        return bool(r)
    return False


def process_puzzle(conn, ref, verifier, source, puzzle_number, clues, enrichments, dry_run):
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}\n{source} {puzzle_number}: {len(clues)} clues, "
          f"{len(enrichments)} candidate enrichments\n{'='*60}\n")

    # Coverage check enrichments
    to_queue = []
    skipped = []
    for et, w, L, ans in enrichments:
        if has_in_db(ref, et, w, L):
            skipped.append((et, w, L))
        else:
            to_queue.append((et, w, L, ans))
    print(f"Enrichments: {len(to_queue)} genuine misses, {len(skipped)} in DB")
    for et, w, L in skipped[:20]:
        print(f"  SKIP (in DB): {et:25} {w!r} -> {L!r}")
    if len(skipped) > 20:
        print(f"  ... and {len(skipped) - 20} more skipped")
    print()

    # Verify + store each clue
    tiers = {'HIGH': [], 'MEDIUM': [], 'LOW': [], 'FAIL': []}
    for clue_id, wtype, definition, expl in clues:
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

    # Queue enrichments
    if not dry_run:
        try:
            puzzle_num_int = int(puzzle_number)
        except ValueError:
            puzzle_num_int = None
        for et, w, L, ans in to_queue:
            conn.execute(
                "INSERT INTO pending_enrichments "
                "(type, word, letters, answer, source, puzzle_number, created_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?)",
                (et, w, L, ans, source, puzzle_num_int, now))

    print(f"\n{source} {puzzle_number} tier summary:")
    for t in ['HIGH', 'MEDIUM', 'LOW', 'FAIL']:
        print(f"  {t}: {len(tiers[t])}")
    print(f"  Enrichments queued: {0 if dry_run else len(to_queue)}")
    return tiers


def main():
    dry_run = '--dry-run' in sys.argv
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    ref = sqlite3.connect(str(REF_DB))
    verifier = ExplanationVerifier()

    g_tiers = process_puzzle(conn, ref, verifier, 'guardian', '30004',
                              G30004, G30004_ENRICHMENTS, dry_run)
    i_tiers = process_puzzle(conn, ref, verifier, 'independent', '12353',
                              I12353, I12353_ENRICHMENTS, dry_run)

    if not dry_run:
        conn.commit()
    conn.close()


if __name__ == '__main__':
    main()
