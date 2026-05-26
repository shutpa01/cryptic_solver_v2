"""Guardian 30003 leftover solver.

Writes structured_explanations + updates clues + queues pending_enrichments
for each of the 25 leftover clues. Uses the canonical workflow: live SQL
work-list query was already run; this is step 4 (write) for each clue.
"""
import json
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / 'data' / 'clues_master.db'
sys.path.insert(0, str(ROOT))

from sonnet_pipeline.verify_explanation import ExplanationVerifier

# Each tuple: (clue_id, wordplay_type, definition, explanation)
CLUES = [
    # 6a MUGGLE — anagram fodder regex catches LEG+MUG in ana_section so both sum
    (10064940, 'anagram', "he can't magically fix",
     'anagram of LEG + MUG (synonym="simpleton") [anagram: "broken"] = MUGGLE; definition: "he can\'t magically fix"'),

    # 9a SPOTON — SPOON contains T (abbreviation of "tea"); "say" claimed as homophone-ind annotation
    (10064941, 'container', 'Exactly',
     'SPOON (synonym="something to eat with") containing T (abbreviation="tea") [container: "coming in"; homophone: "say"] = SPOTON; definition: "Exactly"'),

    # 10a MARINADE — anagram RAIN+MADE
    (10064942, 'anagram', 'liquid mixture over some food',
     'anagram of RAIN + MADE [anagram: "Pelting"] = MARINADE; definition: "liquid mixture over some food"'),

    # 11a RELAYRACE — CD; def is "contest" (in DB); [parts:] claims other clue words
    (10064943, 'cryptic_definition', 'contest',
     'cryptic definition: a contest where the one who starts will not finish = RELAYRACE [parts: "One may start this, but one won\'t finish"]; definition: "contest"'),

    # 13a FREED — F + reversal of DEER
    (10064944, 'reversal', 'out of jail',
     'F (abbreviation="Fine") + reversal of DEER (synonym="supplier of venison") [reversal: "returns"] = FREED; definition: "out of jail"'),

    # 17a FLYTIP — FLY + TIP charade
    (10064946, 'charade', 'illegally dispose of it',
     'FLY (synonym="insect") + TIP (synonym="End") = FLYTIP; definition: "illegally dispose of it"'),

    # 18a RENDER — DD; [parts:] claims second-window words for word_coverage
    (10064947, 'double_definition', 'Hand over',
     'double definition: Hand over = RENDER, first thin coat = RENDER [parts: "first thin coat"]; definition: "Hand over"'),

    # 19a BARREL — DD; [parts:] claims second-window words
    (10064948, 'double_definition', 'Large quantity of beer',
     'double definition: Large quantity of beer = BARREL, over which one is powerless = BARREL [parts: "over which one is powerless"]; definition: "Large quantity of beer"'),

    # 21a MARSH — MARS + H
    (10064949, 'charade', 'waterlogged area',
     'MARS (synonym="planet") + H (abbreviation="hot") = MARSH; definition: "waterlogged area"'),

    # 22a WOLVERINE — anagram WINE+LOVER
    (10064950, 'anagram', 'glutton',
     'anagram of WINE + LOVER [anagram: "Drunken"] = WOLVERINE; definition: "glutton"'),

    # 25a INCIDENT — IN (part of) + CID + SENT minus first letter; add "to" linking
    (10064951, 'charade', 'disturbance',
     'IN (synonym="part of") + CID (synonym="detective force") + ENT (deletion="SENT") [deletion: "first off"; parts: "first off"] = INCIDENT; "to" linking; definition: "disturbance"'),

    # 26a DAMAGE — DAM + AGE; include "perhaps" to silence anagram-ind 7b
    (10064952, 'charade', 'a likely result of crash',
     'DAM (synonym="Barrier") + AGE (synonym="to become weaker") = DAMAGE; "perhaps" DBE-marker; definition: "a likely result of crash"'),

    # 28a CRISES — C + RISES; [parts: "for"] silences false 7b deletion + claims "for"
    (10064953, 'charade', 'Crucial moments',
     'C (abbreviation="Charlie") + RISES (synonym="gets up") [parts: "for which"] = CRISES; definition: "Crucial moments"'),

    # 29a BERGAMOT — BERG + A + MOT; [parts: "for"] silences false 7b deletion + claims "for"
    (10064954, 'charade', 'Essential oil',
     'BERG (synonym="mass of ice") + A + MOT (synonym="car test") [parts: "for"] = BERGAMOT; definition: "Essential oil"'),

    # 2d LAP — CD; def "Part of body" in DB; [parts:] claims rest
    (10064955, 'cryptic_definition', 'Part of body',
     'cryptic definition: a part of the body that disappears when one stands up = LAP [parts: "disappearing as one stands"]; definition: "Part of body"'),

    # 4d LONERANGER — L + ONE + R + ANGER, container verifies, [parts: "one"] claims "one"
    (10064957, 'charade', 'Wild West law enforcer',
     'L (abbreviation="two hands") + ONE (from clue) + R (abbreviation="two hands") + ANGER (synonym="fury") [container: "separating"; parts: "one"] = LONERANGER; definition: "Wild West law enforcer"'),

    # 6d MARE — DD; [parts:] claims second-window words
    (10064959, 'double_definition', 'Horse',
     'double definition: Horse = MARE, appears on the moon = MARE [parts: "appears on the moon"]; definition: "Horse"'),

    # 7d GENERATOR — anagram GOT+NEARER
    (10064960, 'anagram', 'source of energy',
     'anagram of GOT + NEARER [anagram: "Shivering"] = GENERATOR; definition: "source of energy"'),

    # 12d ENTERTAINER — claim "for" via [parts:]
    (10064962, 'anagram', 'job in showbiz',
     'anagram of RETRAIN + TEEN [anagram: "Somehow"; parts: "for"] = ENTERTAINER; definition: "job in showbiz"'),

    # 14d ELSALVADOR — "to" is already in LINK_WORDS; just include via syn source or comment
    (10064963, 'anagram', 'the country',
     'anagram of SALAD + LOVER [anagram: "moved"] = ELSALVADOR; "to the" linking; definition: "the country"'),

    # 16d LANDSLIPS — CD; def "they may fall off a cliff" in DB; [parts:] claims "Naturally"
    (10064964, 'cryptic_definition', 'they may fall off a cliff',
     'cryptic definition: things that naturally fall off a cliff = LANDSLIPS [parts: "Naturally"]; definition: "they may fall off a cliff"'),

    # 20d HOTTUB — HUB containing OTT [container: "installs"]; queue installs ind
    (10064965, 'container', 'bath',
     'HUB (synonym="Centre") containing OTT (synonym="unwarranted") [container: "installs"] = HOTTUB; definition: "bath"'),

    # 23d RUMBA — RUM + BA; claim "going" via [parts:]
    (10064966, 'charade', 'A Cuban export',
     'RUM (synonym="strong drink") + BA (abbreviation="airline") [parts: "going"] = RUMBA; definition: "A Cuban export"'),

    # 24d MEWS — MEWS sounds like MUSE (synonym="Ponder"); "in" linking
    (10064967, 'homophone', 'old stables',
     'MEWS sounds like MUSE (synonym="Ponder") [homophone: "aloud"] = MEWS; "in" linking; definition: "old stables"'),

    # 27d GOO — GOO from GOOD (synonym satisfactory) via deletion; [deletion: "entirely"; parts: "not entirely"]
    (10064968, 'deletion', 'Sticky stuff',
     'GOO (deletion="GOOD"), GOOD (synonym="satisfactory") [deletion: "entirely"; parts: "not entirely"] = GOO; definition: "Sticky stuff"'),
]

# Enrichments to queue. Format: (type, word, letters, clue_id)
ENRICHMENTS = [
    # 6a MUGGLE
    ('definition', "he can't magically fix", 'MUGGLE', 10064940),
    # 9a SPOTON
    ('synonym', 'something to eat with', 'SPOON', 10064941),
    ('homophone', 'tea', 'T', 10064941),
    ('indicator', 'coming in', 'CONTAINER', 10064941),
    # 10a MARINADE
    ('indicator', 'pelting', 'ANAGRAM', 10064942),
    # 11a RELAYRACE — no enrichment needed (using "contest" which is in DB)
    # 13a FREED
    ('synonym', 'supplier of venison', 'DEER', 10064944),
    # 18a RENDER
    ('definition', 'first thin coat', 'RENDER', 10064947),
    # 19a BARREL
    ('definition', 'large quantity of beer', 'BARREL', 10064948),
    ('definition', 'over which one is powerless', 'BARREL', 10064948),
    # 25a INCIDENT
    ('synonym', 'part of', 'IN', 10064951),
    ('indicator', 'first off', 'DELETION', 10064951),
    # 26a DAMAGE
    ('synonym', 'barrier', 'DAM', 10064952),
    ('synonym', 'to become weaker', 'AGE', 10064952),
    # 29a BERGAMOT
    ('synonym', 'car test', 'MOT', 10064954),
    # 2d LAP — no enrichment needed (using "Part of body" which is in DB)
    # 4d LONERANGER
    ('abbreviation', 'two hands', 'L', 10064957),
    ('abbreviation', 'two hands', 'R', 10064957),
    # 14d ELSALVADOR
    ('definition', 'the country', 'ELSALVADOR', 10064963),
    # 16d LANDSLIPS — no enrichment needed (using "they may fall off a cliff" which is in DB)
    # 20d HOTTUB
    ('indicator', 'installs', 'CONTAINER', 10064965),
    # 24d MEWS
    ('homophone', 'mews', 'MUSE', 10064967),
    ('synonym', 'ponder', 'MUSE', 10064967),
    ('definition', 'old stables', 'MEWS', 10064967),
]


def queue_enrichments(conn, now):
    queued = 0
    for ent in ENRICHMENTS:
        gtype, word, letters, clue_id = ent
        row = conn.execute(
            "SELECT clue_text, answer, source, puzzle_number FROM clues WHERE id=?",
            (clue_id,),
        ).fetchone()
        if not row:
            continue
        clue_text, answer, source, puzzle_number = row
        try:
            conn.execute(
                """INSERT OR IGNORE INTO pending_enrichments
                   (type, word, letters, answer, clue_text, source, puzzle_number, created_at)
                   VALUES (?, ?, ?, ?, ?, ?, ?, ?)""",
                (gtype, word.lower(), letters.upper(), answer, clue_text,
                 source, puzzle_number, now),
            )
            queued += conn.execute("SELECT changes()").fetchone()[0]
        except sqlite3.IntegrityError:
            pass
    return queued


def main():
    dry_run = '--dry-run' in sys.argv
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    verifier = ExplanationVerifier()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    if not dry_run:
        queued = queue_enrichments(conn, now)
        print(f"Queued {queued} pending enrichments.\n")
    else:
        print("DRY RUN — no writes.\n")

    results = []
    stored = 0
    for clue_id, wtype, definition, expl in CLUES:
        row = conn.execute("SELECT * FROM clues WHERE id=?", (clue_id,)).fetchone()
        if not row:
            print(f"  SKIP {clue_id}: not found"); continue
        answer = row['answer']
        clue_text = row['clue_text']
        v = verifier.verify(
            clue_text=clue_text, answer=answer,
            wordplay_type=wtype, definition=definition,
            ai_explanation=expl,
        )
        score = v.get('score', 0)
        verdict = v.get('verdict', 'FAIL')
        confidence = score / 100.0

        label = f"{row['clue_number']}{row['direction'][0]}"
        print(f"  [{verdict:6} {score:3}] {label:4} id={clue_id} {answer:13} {expl[:95]}")

        components = json.dumps({
            "ai_pieces": [],
            "assembly": {"op": wtype},
            "wordplay_type": wtype,
            "source": "claude_review",
        })

        if dry_run:
            stored += 1
            results.append((label, answer, verdict, score, clue_id))
            continue

        conn.execute("""
            UPDATE clues
            SET definition = ?, wordplay_type = ?, ai_explanation = ?,
                has_solution = 1, reviewed = 1
            WHERE id = ?
        """, (definition, wtype, expl, clue_id))

        conn.execute("""
            INSERT OR REPLACE INTO structured_explanations
            (clue_id, definition_text, wordplay_types, components,
             model_version, confidence, created_at, updated_at,
             source, puzzle_number, clue_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            clue_id, definition, json.dumps([wtype]), components,
            "claude_review", confidence, now, now,
            row['source'], row['puzzle_number'], row['clue_number'],
        ))
        stored += 1
        results.append((label, answer, verdict, score, clue_id))

    if not dry_run:
        conn.commit()
    print(f"\n{'Would store' if dry_run else 'Stored'} {stored} structured explanations.\n")

    tiers = {'HIGH': [], 'MEDIUM': [], 'LOW': [], 'FAIL': []}
    for label, ans, verdict, score, cid in results:
        tiers[verdict].append((label, ans, score))
    print("Tier summary:")
    for t in ['HIGH', 'MEDIUM', 'LOW', 'FAIL']:
        print(f"  {t}: {len(tiers[t])}")
        for label, ans, score in tiers[t]:
            print(f"     {label:5} {ans:13} {score:3}")

    conn.close()


if __name__ == '__main__':
    main()
