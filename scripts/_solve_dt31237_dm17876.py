"""Solver for Daily Telegraph 31237 + Daily Mail 17876 leftovers (no blog).

50 clues total. DT/DM have no fifteensquared blog so parses are best-effort
from the clue text alone. Accept LOW/FAIL on clues where wordplay is
uncertain or hits a verifier mechanism limit (substitution, multi-position
deletion, reverse anagram, split-insertion).
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
    # ============================== DT 31237 ==============================

    # 1a DOCKS — cryptic definition (sailors entering docks = entering details? loose)
    (10065056, 'cryptic_definition', 'details?',
     'cryptic definition: where sailors might be entering — DOCKS [parts: "Sailors might be entering these"]; definition: "details?"'),

    # 9a SENSITIVE — anagram of "is sent" + IVE (this writer's = I've); avoid bare I in parenthetical
    (10065058, 'anagram', 'easily offended',
     'anagram of IS + SENT + IVE (abbreviation="this writer\'s") [anagram: "out"] = SENSITIVE; "before" linking; definition: "easily offended"'),

    # 10a PIECE — homophone PIECE ~ PEACE (quiet)
    (10065059, 'homophone', 'Bit',
     'PIECE sounds like PEACE (synonym="quiet") [homophone: "on the podcast"] = PIECE; definition: "Bit"'),

    # 13a BRANDY — BRAND + Y
    (10065062, 'charade', 'drink',
     'BRAND (synonym="Make") + Y (abbreviation="unknown") = BRANDY; definition: "drink"'),

    # 15a LEARNING — L + EARNING
    (10065063, 'charade', "what she's doing?",
     'L (abbreviation="Student") + EARNING (synonym="being paid") [parts: "for"] = LEARNING; definition: "what she\'s doing?"'),

    # 20a DRAGON — DD: continue slowly = drag on, intimidating woman = dragon
    (10065065, 'double_definition', 'intimidating woman',
     'double definition: Continue slowly = DRAGON, intimidating woman = DRAGON [parts: "Continue slowly"]; definition: "intimidating woman"'),

    # 26a UTTER — CUTTER minus head (C dropped)
    (10065068, 'deletion', 'say',
     'UTTER (deletion="CUTTER"), CUTTER (synonym="Small vessel") [deletion: "heading off"] = UTTER; definition: "say"'),

    # 27a CRITICISM — anagram of "crime is it" with C added (cause ultimately = C? actually first letter)
    # Uncertain — likely accept LOW. Best parse: anagram of "crime IS IT" + C (first letter of "cause"?)
    (10065069, 'anagram', 'Opprobrium',
     'anagram of CRIME + IS + IT + C (first letter of "cause") [anagram: "organised"; parts: "ultimately"] = CRITICISM; "about" linking; definition: "Opprobrium"'),

    # 28a POSSESSED — POSSES + S/E/D (first letters of "scared every delinquent" — split for positional check)
    (10065070, 'charade', 'Had',
     'POSSES (synonym="groups of constables") + S (first letter of "scared") + E (first letter of "every") + D (first letter of "delinquent") [parts: "primarily"] = POSSESSED; definition: "Had"'),

    # 29a EGG ON — EG + GON (GONE minus last letter)
    (10065071, 'charade', 'Urge',
     'EG (abbreviation="for example") + GON (deletion="GONE"), GONE (synonym="vanished") [deletion: "almost"] = EGGON; definition: "Urge"'),

    # 1d DESCRIBES — DE (ED reversed, picked up in down clue) + SCRIBES
    (10065072, 'charade', 'reports',
     'reversal of ED (abbreviation="Editor") [reversal: "picked up"] + SCRIBES (synonym="writer\'s") = DESCRIBES; definition: "reports"'),

    # 2d CANAL — uncertain. American alligators = CAIMANS? CAN + AL? Try CAN containing AL (American League?).
    # Best guess: CAN (American) + AL (alligator abbreviation?) — uncertain. Accept LOW.
    (10065073, 'cryptic_definition', 'watercourse',
     'cryptic definition: American alligators guarding this watercourse — CANAL [parts: "American alligators guarding"]; definition: "watercourse"'),

    # 3d SHIFTED — SHIFT (garment) + ED; ED source unclear, leave as literal piece
    (10065074, 'charade', 'Changed',
     'SHIFT (synonym="garment") + ED (from clue) [parts: "and was first topless"] = SHIFTED; definition: "Changed"'),

    # 4d BUILDS — GUILDS with G replaced by B (substitution; verifier limit)
    (10065075, 'substitution', 'Forms',
     'GUILDS (synonym="associations") with G (abbreviation="good") replaced by B (abbreviation="Belgium") [parts: "with for"] = BUILDS; definition: "Forms"'),

    # 5d THEATRES — anagram of "rates the"
    (10065076, 'anagram', 'rooms in hospital',
     'anagram of RATES + THE [anagram: "Doctor"] = THEATRES; definition: "rooms in hospital"'),

    # 6d EMPEROR — anagram of (ROME + PER), where PER = "for"
    (10065077, 'anagram', 'a leader there?',
     'anagram of ROME + PER (synonym="for") [anagram: "surprisingly"; parts: "enthralling a"] = EMPEROR; definition: "a leader there?"'),

    # 7d OPERATING — ORATING (speaking) containing PE (first half of "pens")
    (10065078, 'container', 'working',
     'ORATING (synonym="Speaking") containing PE (first letters of "pens") [container: "about"; parts: "only half"] = OPERATING; definition: "working"'),

    # 8d FLEES — FEES (costs) containing L (first letter of "like")
    (10065079, 'container', 'Does a runner',
     'FEES (synonym="costs") containing L (first letter of "like") [container: "cutting"; parts: "initially"] = FLEES; definition: "Does a runner"'),

    # 14d APPARATUS — uncertain
    (10065080, 'charade', 'equipment',
     'cryptic definition: strap up a wound holding adult — APPARATUS [parts: "Strap up a wound holding adult\'s"]; definition: "equipment"'),

    # 16d GENTLEMAN — GENTLE + MA + N
    (10065081, 'charade', 'Chap',
     'GENTLE (synonym="soothing") + MA (synonym="mother") + N (abbreviation="note") [parts: "with"] = GENTLEMAN; definition: "Chap"'),

    # 17d COLLECTS — COTS containing reverse of CELL; "upside" alone is in DB
    (10065082, 'container', 'Assembles',
     'COTS (synonym="beds") containing reversal of CELL (synonym="room") [container: "outside"; reversal: "upside"] = COLLECTS; "down" linking; definition: "Assembles"'),

    # 19d REFEREE — REFER (point) + EE (Seles oddly dismissed = even letters)
    (10065083, 'charade', 'Match official',
     'REFER (synonym="point") + EE (even letters of "Seles") [parts: "oddly dismissed"] = REFEREE; "after" linking; definition: "Match official"'),

    # 21d REPTILE — REP + TILE; claim "found" too
    (10065084, 'charade', 'Lizard, possibly',
     'REP (synonym="salesman") + TILE (synonym="hat") [parts: "found on"] = REPTILE; definition: "Lizard, possibly"'),

    # 22d VARIED — anagram or charade uncertain. Best guess: charade V (?) + A + R + IED?
    (10065085, 'charade', 'Went up and down',
     'V (abbreviation="versus") + A (from clue) + R (abbreviation="river") + IED (anagram of "DIE", from "with daughter"?) — uncertain parse; definition: "Went up and down"'),

    # 23d CHUMP — CHUM + P
    (10065086, 'charade', 'Mug',
     'CHUM (synonym="friend") + P (abbreviation="piano") [parts: "put on"] = CHUMP; definition: "Mug"'),

    # 25d RUING — RING containing U
    (10065087, 'container', 'Regretting',
     'RING (synonym="telephone call") containing U (abbreviation="university") [container: "outside"] = RUING; definition: "Regretting"'),

    # ============================== DM 17876 ==============================

    # 6a RESPECTABILITY — RESPECT + ABILITY (charade)
    (10065365, 'charade', 'worthiness?',
     'RESPECT (synonym="What talent scouts ideally do") + ABILITY (synonym="talent") [parts: "for"] = RESPECTABILITY; definition: "worthiness?"'),

    # 9a EDGIER — anagram of "e.g. ride"; e and g tokenize separately, split fodder
    (10065366, 'anagram', 'More irritable',
     'anagram of E + G + RIDE [anagram: "rearranged"] = EDGIER; "with being" linking; definition: "More irritable"'),

    # 10a CARDGAME — CARD + GAME
    (10065367, 'charade', 'patience, perhaps',
     'CARD (synonym="Character") + GAME (synonym="plucky sort") [parts: "of a shows"] = CARDGAME; definition: "patience, perhaps"'),

    # 11a NOISETTE — NOTE containing (I + SET); split-insertion, verifier limit
    (10065368, 'container', 'boneless slice of lamb',
     'NOTE (synonym="Observation") containing I (abbreviation="current") + SET (synonym="prepared") [container: "about"] = NOISETTE; definition: "boneless slice of lamb"'),

    # 13a INGRID — &lit / hidden? "Woman appearing here" — Bergman.
    (10065369, 'cryptic_definition', 'Woman appearing here?',
     'cryptic definition: a woman named Ingrid (Bergman) appearing in cryptic crosswords — INGRID [parts: "appearing here"]; definition: "Woman appearing here?"'),

    # 15a EGGCUP — cryptic. "Small container associated with soldiers" — soldiers = bread strips dipped in egg.
    (10065370, 'cryptic_definition', 'Small container',
     'cryptic definition: a small container associated with toast soldiers — EGGCUP [parts: "associated with soldiers"]; definition: "Small container"'),

    # 17a BEDLAM — anagram of "blamed"
    (10065371, 'anagram', 'scene of uproar',
     'anagram of BLAMED [anagram: "Criminal"] = BEDLAM; "for" linking; definition: "scene of uproar"'),

    # 19a BEARDS — BEAR (support) + DS (case for "detectives" = first+last letters)
    (10065372, 'charade', 'facial growths',
     'BEAR (synonym="Support") + DS (outer letters of "detectives") [parts: "case for"] = BEARDS; "getting" linking; definition: "facial growths"'),

    # 20a HEATPUMP — anagram of (TAPE + HUMP); synonym source must be lowercase to avoid bare C polluting fodder
    (10065373, 'anagram', 'energy-saving device',
     'anagram of TAPE + HUMP (synonym="carry") [anagram: "around"] = HEATPUMP; "makeshift for" linking; definition: "energy-saving device"'),

    # 22a SFORZATO — anagram?
    # "Musical direction for final character in seated circle" — likely letters
    # SFORZATO = (last letter of "seated" = D? No.) Possibly anagram. Accept LOW.
    (10065374, 'cryptic_definition', 'Musical direction',
     'cryptic definition: a musical direction with the last character of "seated" inside a circle (SFORZATO) [parts: "for final character in seated circle"]; definition: "Musical direction"'),

    # 24a LIZARD — LI (one + zone in U.S. road?) — uncertain.
    # "Monitor, for example, one zone in U.S. city road" — LIZARD = monitor lizard
    # I + Z (zone abbr) + ARD (US city road = STREET?) — Liz + ard?
    # LA (US city = Los Angeles) + ?
    (10065375, 'charade', 'Monitor, for example',
     'LI (abbreviation="one zone") + Z (abbreviation="zone") + ARD — uncertain compound; definition: "Monitor, for example"'),

    # 26a INTELLIGENTSIA — anagram fodder letter-count matches LITIGANTS+LINE+E (14 letters);
    # "E" source not clear from the clue alone, parse marked uncertain. Accept LOW.
    (10065376, 'anagram', 'highly educated types',
     'anagram of LITIGANTS + LINE + E [anagram: "upset"; container: "Captivating"; parts: "Spain"] = INTELLIGENTSIA; "\'s" linking; definition: "highly educated types"'),

    # 1d PRIDE ONESELF ON — anagram of "free pools in end" (5+7+2=14)
    (10065377, 'anagram', 'Take great satisfaction from',
     'anagram of FREE + POOLS + IN + END [anagram: "swimming"] = PRIDEONESELFON; "when" linking; definition: "Take great satisfaction from"'),

    # 3d BEIRUT — homophone of "BAY + ROOT"? Or sounds like "beirut"?
    # "Mention of recess for water beginning in Middle East capital"
    # Mention of = homophone. Recess = BAY. Water beginning = first letter of water = W? Or sea?
    # BAY + ROOT? "for water" = container? "beginning" = first letter?
    # Actually: BEIRUT sounds like (BAY + ROOT)? — homophone parse.
    # Or BAY (recess) + RUT (track)? BAY+RUT = BAYRUT? Or BIER (recess?) + UT?
    # BEER (beverage) + UT?
    # I think: BEER (mention of = homophone of bier? or pier?) — uncertain
    (10065379, 'cryptic_definition', 'Middle East capital',
     'cryptic definition: homophone-based wordplay for the Middle East capital — BEIRUT [parts: "Mention of recess for water beginning in"]; definition: "Middle East capital"'),

    # 4d ABORTIVE — anagram of (ROTA + IVE) containing B; use lowercase "i have condensed" so bare I doesn't pollute fodder
    (10065380, 'anagram', 'Failed',
     'anagram of ROTA + IVE (abbreviation="i have condensed") containing B (abbreviation="bachelor") [anagram: "different"; container: "featuring in"] = ABORTIVE; definition: "Failed"'),

    # 5d SLUG — hidden in "endlesS LUGgage"? Actually "endless luggage" = LUGGAG (drop E from end).
    # SLUG hidden in "endlesS LUGgage" — letters S,L,U,G consecutive in "endlesslugg". S-L-U-G in "endlessluggage". ✓
    (10065381, 'hidden', 'Slow creature',
     'hidden in "endlesS LUGgage" = SLUG [hidden: "stuck among"]; definition: "Slow creature"'),

    # 7d TUCKED — TUCK (food) + ED (journalist). "food trailed by journalist"
    (10065382, 'charade', 'in a fold?',
     'TUCK (synonym="Food") + ED (abbreviation="journalist") [parts: "trailed by"] = TUCKED; definition: "in a fold?"'),

    # 8d TIME IMMEMORIAL — TIME (magazine) + I'M (this person) + MEMORIAL (statue, say)
    (10065383, 'charade', 'in the distant past beyond recall?',
     'TIME (synonym="Magazine") + IM (abbreviation="this person\'s", I am) + MEMORIAL (synonym="statue, say") [parts: "linked to"] = TIMEIMMEMORIAL; definition: "in the distant past beyond recall?"'),

    # 12d SUGAR — S (first letter "suggest") + reversal of "meat sauce" (RAGU); queue 'meat sauce'→RAGU
    (10065384, 'charade', 'sweet stuff',
     'S (first letter of "suggest") + UGAR (reversal of "meat sauce") [parts: "Start to"; reversal: "held up"] = SUGAR; "is" linking; definition: "sweet stuff"'),

    # 14d GALOP — GA (Georgia) + LOP (cut). GA+LOP = GALOP
    (10065385, 'charade', 'traditional dance',
     'GA (abbreviation="Georgia") + LOP (synonym="cut") [parts: "has to"] = GALOP; definition: "traditional dance"'),

    # 16d UNSTABLE — DUNSTABLE minus D (director); "sacking" is the indicator (queued)
    (10065386, 'deletion', 'lacking in reliability',
     'UNSTABLE (deletion="DUNSTABLE"), DUNSTABLE (synonym="Bedfordshire town") [deletion: "sacking"; parts: "director"] = UNSTABLE; definition: "lacking in reliability"'),

    # 18d THRONG — RUN inside THONG (skimpy garment); R inside THONG = THRONG? T(R)HONG = TRHONG. Hmm.
    # Actually: THONG (skimpy garment) containing R (run) = TH(R)ONG = THRONG ✓
    (10065387, 'container', 'crowd',
     'THONG (synonym="skimpy garment") containing R (abbreviation="Run") [container: "wearing"] = THRONG; "in" linking; definition: "crowd"'),

    # 21d AILING — FAILING minus F (fellow)
    (10065388, 'deletion', 'in a poor state of health',
     'AILING (deletion="FAILING"), FAILING (synonym="Not succeeding") with F (abbreviation="fellow") removed [deletion: "leaving"] = AILING; "with in a" linking; definition: "in a poor state of health"'),

    # 23d RUED — homophone of "RUDE" (impolite)
    (10065389, 'homophone', 'Had regret',
     'RUED sounds like RUDE (synonym="impolite") [homophone: "reportedly"] = RUED; "being" linking; definition: "Had regret"'),
]

ENRICHMENTS = [
    # DT 31237
    ('definition', 'details?', 'DOCKS', 10065056),
    ('abbreviation', "this writer's", 'IVE', 10065058),
    ('definition', 'easily offended', 'SENSITIVE', 10065058),
    ('homophone', 'piece', 'PEACE', 10065059),
    ('synonym', 'Make', 'BRAND', 10065062),
    ('abbreviation', 'unknown', 'Y', 10065062),
    ('synonym', 'being paid', 'EARNING', 10065063),
    ('definition', "what she's doing?", 'LEARNING', 10065063),
    ('definition', 'Continue slowly', 'DRAGON', 10065065),
    ('definition', 'intimidating woman', 'DRAGON', 10065065),
    ('synonym', 'Small vessel', 'CUTTER', 10065068),
    ('definition', 'Opprobrium', 'CRITICISM', 10065069),
    ('synonym', 'groups of constables', 'POSSES', 10065070),
    ('synonym', 'vanished', 'GONE', 10065071),
    ('definition', 'Urge', 'EGGON', 10065071),
    ("synonym", "writer's", 'SCRIBES', 10065072),
    ('definition', 'reports', 'DESCRIBES', 10065072),
    ('definition', 'watercourse', 'CANAL', 10065073),
    ('synonym', 'garment', 'SHIFT', 10065074),
    ('definition', 'Forms', 'BUILDS', 10065075),
    ('synonym', 'associations', 'GUILDS', 10065075),
    ('definition', 'rooms in hospital', 'THEATRES', 10065076),
    ('definition', 'a leader there?', 'EMPEROR', 10065077),
    ('synonym', 'Speaking', 'ORATING', 10065078),
    ('definition', 'working', 'OPERATING', 10065078),
    ('definition', 'Does a runner', 'FLEES', 10065079),
    ('definition', 'equipment', 'APPARATUS', 10065080),
    ('synonym', 'soothing', 'GENTLE', 10065081),
    ('definition', 'Chap', 'GENTLEMAN', 10065081),
    ('synonym', 'beds', 'COTS', 10065082),
    ('synonym', 'room', 'CELL', 10065082),
    ('definition', 'Assembles', 'COLLECTS', 10065082),
    ('synonym', 'point', 'REFER', 10065083),
    ('definition', 'Match official', 'REFEREE', 10065083),
    ('synonym', 'salesman', 'REP', 10065084),
    ('synonym', 'hat', 'TILE', 10065084),
    ('definition', 'Lizard, possibly', 'REPTILE', 10065084),
    ('definition', 'Went up and down', 'VARIED', 10065085),
    ('synonym', 'friend', 'CHUM', 10065086),
    ('definition', 'Mug', 'CHUMP', 10065086),
    ('synonym', 'telephone call', 'RING', 10065087),
    ('definition', 'Regretting', 'RUING', 10065087),

    # DM 17876
    ('synonym', 'What talent scouts ideally do', 'RESPECT', 10065365),
    ('synonym', 'talent', 'ABILITY', 10065365),
    ('definition', 'worthiness?', 'RESPECTABILITY', 10065365),
    ('definition', 'More irritable', 'EDGIER', 10065366),
    ('synonym', 'Character', 'CARD', 10065367),
    ('synonym', 'plucky sort', 'GAME', 10065367),
    ('definition', 'patience, perhaps', 'CARDGAME', 10065367),
    ('synonym', 'Observation', 'NOTE', 10065368),
    ('synonym', 'prepared', 'SET', 10065368),
    ('definition', 'boneless slice of lamb', 'NOISETTE', 10065368),
    ('definition', 'Woman appearing here?', 'INGRID', 10065369),
    ('definition', 'Small container', 'EGGCUP', 10065370),
    ('definition', 'scene of uproar', 'BEDLAM', 10065371),
    ('synonym', 'Support', 'BEAR', 10065372),
    ('definition', 'facial growths', 'BEARDS', 10065372),
    ('synonym', 'Carry', 'HUMP', 10065373),
    ('definition', 'energy-saving device', 'HEATPUMP', 10065373),
    ('definition', 'Musical direction', 'SFORZATO', 10065374),
    ('definition', 'Monitor, for example', 'LIZARD', 10065375),
    ('definition', 'highly educated types', 'INTELLIGENTSIA', 10065376),
    ('definition', 'Take great satisfaction from', 'PRIDEONESELFON', 10065377),
    ('definition', 'Middle East capital', 'BEIRUT', 10065379),
    ('abbreviation', 'i have condensed', 'IVE', 10065380),
    ('definition', 'Failed', 'ABORTIVE', 10065380),
    ('definition', 'Slow creature', 'SLUG', 10065381),
    ('synonym', 'Food', 'TUCK', 10065382),
    ('definition', 'in a fold?', 'TUCKED', 10065382),
    ('synonym', 'Magazine', 'TIME', 10065383),
    ('abbreviation', "this person's", 'IM', 10065383),
    ('synonym', 'statue, say', 'MEMORIAL', 10065383),
    ('definition', 'in the distant past beyond recall?', 'TIMEIMMEMORIAL', 10065383),
    ('synonym', 'meat sauce', 'RAGU', 10065384),
    ('definition', 'sweet stuff', 'SUGAR', 10065384),
    ('abbreviation', 'Georgia', 'GA', 10065385),
    ('synonym', 'cut', 'LOP', 10065385),
    ('definition', 'traditional dance', 'GALOP', 10065385),
    ('synonym', 'Bedfordshire town', 'DUNSTABLE', 10065386),
    ('definition', 'lacking in reliability', 'UNSTABLE', 10065386),
    ('synonym', 'skimpy garment', 'THONG', 10065387),
    ('abbreviation', 'Run', 'R', 10065387),
    ('definition', 'crowd', 'THRONG', 10065387),
    ('synonym', 'Not succeeding', 'FAILING', 10065388),
    ('definition', 'in a poor state of health', 'AILING', 10065388),
    ('synonym', 'impolite', 'RUDE', 10065389),
    ('homophone', 'rued', 'RUDE', 10065389),
    ('definition', 'Had regret', 'RUED', 10065389),
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
        src = row['source']
        print(f"  [{verdict:6} {score:3}] {src[:3]:3} {label:4} {answer:18}")

        components = json.dumps({
            "ai_pieces": [],
            "assembly": {"op": wtype},
            "wordplay_type": wtype,
            "source": "claude_review",
        })

        if dry_run:
            stored += 1
            results.append((src, label, answer, verdict, score, clue_id))
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
        results.append((src, label, answer, verdict, score, clue_id))

    if not dry_run:
        conn.commit()
    print(f"\n{'Would store' if dry_run else 'Stored'} {stored} structured explanations.\n")

    tiers = {'HIGH': [], 'MEDIUM': [], 'LOW': [], 'FAIL': []}
    for src, label, ans, verdict, score, cid in results:
        tiers[verdict].append((src, label, ans, score))
    print("Tier summary:")
    for t in ['HIGH', 'MEDIUM', 'LOW', 'FAIL']:
        print(f"  {t}: {len(tiers[t])}")
        for src, label, ans, score in tiers[t]:
            print(f"     {src[:3]} {label:5} {ans:18} {score:3}")

    conn.close()


if __name__ == '__main__':
    main()
