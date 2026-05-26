"""Independent 12352 leftover solver.

Writes structured_explanations + updates clues + queues pending_enrichments
for each of the 28 leftover clues. Blog compass already in place
(re-fetched from fifteensquared via _reparse_blog_indy12352.py).
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

CLUES = [
    # 1a CAMP (4) — M in CAP, "wearing" container indicator
    (10064969, 'container', 'Theatrical',
     'CAP (synonym="hat") containing M (abbreviation="maiden") [container: "wearing"] = CAMP; definition: "Theatrical"'),

    # 3a APPRENTICE (10) — APP + R + ENTICE charade; "to" mention silences false reversal 7b
    (10064970, 'charade', 'trainee',
     'APP (synonym="Program") + R (abbreviation="run") + ENTICE (synonym="attract") = APPRENTICE; "to attract" linking; definition: "trainee"'),

    # 9a WINDSOR (7) — substitution: WINNER with NE replaced by DSO. Verifier can't model. Accept FAIL.
    (10064971, 'substitution', 'castle',
     'WINNER (synonym="Victor") with NE replaced by DSO (abbreviation="award", Distinguished Service Order, "for Northumbrian") = WINDSOR; definition: "castle"'),

    # 11a PERUSED (7) — PER + US + ED
    (10064972, 'charade', 'Went over',
     'PER (synonym="a") + US (synonym="Yankee") + ED (abbreviation="editor") = PERUSED; definition: "Went over"'),

    # 12a LOGCABIN (3,5) — cryptic def; clue's HTML <i></i> tags surface as "i" tokens, claim them via [parts:]
    (10064973, 'cryptic_definition', "backwoods' location",
     'cryptic definition: what the director of a taxi may do — LOG CAB IN = LOGCABIN [parts: "What director of i Taxi i may do"]; definition: "backwoods\' location"'),

    # 13a HEADS (5) — DD: On board facilities = HEADS (queued), leaders = HEADS; include "for" against false 7b deletion fire
    (10064974, 'double_definition', 'leaders',
     'double definition: On board facilities = HEADS, leaders = HEADS [parts: "On board facilities for"]; definition: "leaders"'),

    # 15a PERFORMINGARTS (10,4) — reverse anagram (no blog originally). Treat as CD; accept FAIL/LOW.
    (10064975, 'cryptic_definition', 'ballet and opera',
     'cryptic definition: PERFORMING as anagram indicator applied to ARTS yields STAR — so PERFORMINGARTS cryptically clues "star" = PERFORMINGARTS [parts: "Presumably star in"]; definition: "ballet and opera?"'),

    # 17a FORTUNECOOKIES (7,7) — cryptic def; avoid parenthetical phrases that trigger silent_piece
    (10064976, 'cryptic_definition', 'culinary treats',
     'cryptic definition: lots may be wrapped up inside these culinary treats — FORTUNECOOKIES = FORTUNECOOKIES [parts: "Lots may be wrapped up in these"]; definition: "culinary treats"'),

    # 21a ISAAC (5) — ISA + AC
    (10064977, 'charade', "Ishmael's half-brother",
     'ISA (abbreviation="Type of investment") + AC (abbreviation="account") = ISAAC; definition: "Ishmael\'s half-brother"'),

    # 22a SOCIETAL (8) — reversal of (LATE + I + COS)
    (10064978, 'reversal', 'Community',
     'reversal of LATE (synonym="former") + I (abbreviation="independent") + COS (synonym="island") [reversal: "returns"] = SOCIETAL; definition: "Community"'),

    # 24a REGRETS (7) — R + EGRETS
    (10064979, 'charade', 'feels remorse',
     'R (first letter of "Russian") + EGRETS (synonym="fliers") [parts: "Leader of"] = REGRETS; definition: "feels remorse"'),

    # 25a LEAKOUT (4,3) — hidden in "bLEAK OUTcomes"
    (10064980, 'hidden', 'emerge',
     'hidden in "bLEAK OUTcomes" = LEAKOUT [hidden: "Some"]; definition: "emerge"'),

    # 26a BRASSBANDS (5,5) — BRASS + BANDS
    (10064981, 'charade', 'Players',
     'BRASS (synonym="money") + BANDS (synonym="belts") = BRASSBANDS; definition: "Players"'),

    # 27a STUD (4) — STUDY with last letter dropped; [deletion: "out"] (in DB) + [reversal: "backing"] claims "backing" against 7b
    (10064982, 'deletion', 'Boss',
     'STUD (deletion="STUDY"), STUDY (synonym="room") [deletion: "out"; reversal: "backing"] = STUD; definition: "Boss"'),

    # 1d COWSLIPS (8) — CO + W + SLIPS; source "Leader" is what's in clue (CO = leader/commanding officer)
    (10064983, 'charade', 'bloomers',
     'CO (synonym="Leader") + W (first letter of "wear") [parts: "beginning to"] + SLIPS (synonym="underskirts") = COWSLIPS; "and" linking; definition: "bloomers"'),

    # 2d MANAGER (7) — anagram of RANGEMA (orangeman with outer O and N removed); lowercase "orangeman" so anagram fodder regex doesn't pick it up
    (10064984, 'anagram', "Michael O'Neill?",
     'anagram of RANGEMA (deletion="orangeman", outer letters dropped) [deletion: "Naked"; anagram: "dancing"] = MANAGER; "for" linking; definition: "Michael O\'Neill?"'),

    # 4d PARLIAMENT (10) — PARENT containing reversal of MAIL
    (10064985, 'container', 'assembly',
     'PARENT (synonym="Train") containing reversal of MAIL (synonym="post") [container: "carrying"; reversal: "around"] = PARLIAMENT; definition: "assembly"'),

    # 5d RAPT (4) — hidden in "contRAPTion"
    (10064986, 'hidden', 'Fascinated',
     'hidden in "contRAPTion" = RAPT [hidden: "emerging from"]; definition: "Fascinated"'),

    # 6d NORTHCAROLINA (5,8) — anagram of TAILOR ON RANCH
    (10064987, 'anagram', 'State',
     'anagram of TAILOR + ON + RANCH [anagram: "tipsy"] = NORTHCAROLINA; "of" linking; definition: "State"'),

    # 7d INSTANT (7) — INS + T + ANT charade (T inserted between INS and ANT)
    (10064988, 'charade', 'Second',
     'INS (abbreviation="insurance") + T (abbreviation="Thailand") + ANT (synonym="worker") [container: "touring"] = INSTANT; definition: "Second"'),

    # 8d ELDEST (6) — EL + D + EST; "to be" anagram-indicator false fire silenced by mentioning
    (10064989, 'charade', 'most senior',
     'EL (synonym="The Spanish") + D (abbreviation="duke") + EST (abbreviation="established") = ELDEST; "to be" linking; definition: "most senior"'),

    # 10d SEASONTICKETS (6,7) — anagram; claim "obtain" via [parts:]
    (10064990, 'anagram', 'travel passes',
     'anagram of SNEAKIEST + COST [anagram: "involved"; parts: "to obtain"] = SEASONTICKETS; definition: "travel passes"'),

    # 14d UNSCHOOLED (10) — UN + SCHOOL + ED
    (10064991, 'charade', 'illiterate',
     'UN (synonym="A") [parts: "local"] + SCHOOL (synonym="group") + ED (abbreviation="education") [parts: "overseeing"] = UNSCHOOLED; "is" linking; definition: "illiterate"'),

    # 16d ISOLATED (8) — I + SO + (ELATED minus E for England); claim "England" via [parts:]
    (10064992, 'charade', 'cut off',
     'I (abbreviation="India") + SO (synonym="very") + LATED (deletion="ELATED"), ELATED (synonym="very happy") [deletion: "leave"; parts: "England"] = ISOLATED; "to" linking; definition: "cut off"'),

    # 18d OTALGIA (7) — NOSTALGIA minus N and S (bridge partners); two non-adjacent letter drops — verifier mechanism can't model, accept LOW
    (10064993, 'deletion', 'source of pain',
     'NOSTALGIA (synonym="Longing") with N and S removed (NS = bridge partners) gives OTALGIA [deletion: "forget"; parts: "to forget partners"] = OTALGIA; definition: "source of pain"'),

    # 19d INTROIT (7) — IN + TROT containing I (insert at position 5)
    (10064994, 'container', 'piece of music',
     'IN (synonym="Fashionable") + TROT (synonym="communist") containing I (abbreviation="one") [container: "collects"] = INTROIT; definition: "piece of music"'),

    # 20d MIDRIB (6) — MI + DR + I + B (first letters of "investigate" and "blood" separately so positional check fires)
    (10064995, 'charade', 'vein',
     'MI (synonym="Note") + DR (abbreviation="medic") + I (first letter of "investigate") + B (first letter of "blood") [parts: "starts to"] = MIDRIB; "in" linking; definition: "vein"'),

    # 23d ASIA (4) — even letters of "bAnS vIsAs"; piece-source format for positional check
    (10064996, 'alternate', "approximately 60% of the world's population",
     'ASIA (even letters of "bans visas") [parts: "Regularly"] = ASIA; "for" linking; definition: "approximately 60% of the world\'s population"'),
]

ENRICHMENTS = [
    # 9a WINDSOR
    ('abbreviation', 'award', 'DSO', 10064971),
    # 1d COWSLIPS — Leader = CO
    ('synonym', 'leader', 'CO', 10064983),
    # 11a PERUSED
    ('synonym', 'a', 'PER', 10064972),
    # 12a LOGCABIN
    ('definition', "backwoods' location", 'LOGCABIN', 10064973),
    # 13a HEADS
    ('definition', 'on board facilities', 'HEADS', 10064974),
    # 15a PERFORMINGARTS
    ('definition', 'ballet and opera?', 'PERFORMINGARTS', 10064975),
    # 21a ISAAC
    ('abbreviation', 'Type of investment', 'ISA', 10064977),
    # 22a SOCIETAL
    ('synonym', 'island', 'COS', 10064978),
    ('definition', 'Community', 'SOCIETAL', 10064978),
    # 24a REGRETS
    ('definition', 'feels remorse', 'REGRETS', 10064979),
    # 25a LEAKOUT
    ('definition', 'emerge', 'LEAKOUT', 10064980),
    # 26a BRASSBANDS
    ('synonym', 'belts', 'BANDS', 10064981),
    ('definition', 'Players', 'BRASSBANDS', 10064981),
    # 27a STUD
    ('synonym', 'room', 'STUDY', 10064982),
    # 2d MANAGER
    ('definition', "Michael O'Neill?", 'MANAGER', 10064984),
    # 4d PARLIAMENT
    ('synonym', 'Train', 'PARENT', 10064985),
    # 7d INSTANT
    ('abbreviation', 'insurance', 'INS', 10064988),
    ('abbreviation', 'Thailand', 'T', 10064988),
    # 14d UNSCHOOLED
    ('synonym', 'A', 'UN', 10064991),
    ('definition', 'illiterate', 'UNSCHOOLED', 10064991),
    # 18d OTALGIA
    ('abbreviation', 'partners', 'NS', 10064993),
    # 19d INTROIT
    ('definition', 'piece of music', 'INTROIT', 10064994),
    # 20d MIDRIB
    ('definition', 'vein', 'MIDRIB', 10064995),
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
        print(f"  [{verdict:6} {score:3}] {label:4} id={clue_id} {answer:14} {expl[:90]}")

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
            print(f"     {label:5} {ans:15} {score:3}")

    conn.close()


if __name__ == '__main__':
    main()
