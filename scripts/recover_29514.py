"""One-off recovery script for Times 29514.

Restores ai_explanations from puzzle report and relinks orphaned
structured_explanations rows to new clue IDs.
"""
import sqlite3
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "clues_master.db"

# Explanations extracted from documents/puzzle_report_times_29514.txt
REPORT_EXPLANATIONS = {
    # Pipeline (signature_solver, haiku_sonnet_tiered)
    ("14", "across"): {"def": "Country", "wtype": "hidden, reversal", "expl": 'hidden reversed in "region of sp AINABLA ze" [reversal: "to"]; definition: "Country"'},
    ("12", "across"): {"def": "Bar", "wtype": "charade", "expl": 'SOAP (synonym of "EastEnders?"); definition: "Bar"'},
    ("6", "down"): {"def": "Asian", "wtype": "homophone", "expl": 'THAI (sounds like "moor"); definition: "Asian"'},
    ("16", "across"): {"def": "Bugs", "wtype": "charade", "expl": 'B(first letter of "Bunny") + A(first letter of "ate") + C(first letter of "carrots") + ILLI("green"); definition: "Bugs"'},
    ("19", "across"): {"def": "Tender-hearted", "wtype": "charade", "expl": 'A + DO(abbr. "party") + RING(synonym of "cabal"); definition: "Tender-hearted"'},
    ("19", "down"): {"def": "port", "wtype": "charade", "expl": 'A (synonym of "American") + N (synonym of "northern") + TWER (synonym of "gull seen") + P (synonym of "European"); definition: "port"'},
    ("4", "down"): {"def": "Mexican dish", "wtype": "charade", "expl": 'ST (synonym of "Good man") + TOAD (synonym of "animal") + A (abbreviation of "a"); definition: "Mexican dish"'},
    ("8", "down"): {"def": "Arcade", "wtype": "charade", "expl": 'SHOPPING (synonym of "looking lively in Cologne,") + CENTRE (synonym of "residence empty"); definition: "Arcade"'},
    # Human/mechanical explanations
    ("1", "down"): {"def": "nuisance along the way?", "wtype": "charade", "expl": 'BACKS (synonym="Champions") + EAT (synonym="scoff") + DRIVER (synonym="club") = BACKSEATDRIVER; definition: "nuisance along the way?"'},
    ("13", "down"): {"def": "Educational", "wtype": "anagram", "expl": 'anagram of CLASSIC + HOT = SCHOLASTIC; definition: "Educational"'},
    ("17", "across"): {"def": "opens up", "wtype": "reversal", "expl": 'DIL (LID="cover", reversed) + A (from clue) + TES (SET="Ready,", reversed) = DILATES; definition: "opens up"'},
    ("2", "down"): {"def": "end of tax year?", "wtype": "container", "expl": 'ARIL (reversal="currency") + P (abbreviation="Penny") = APRIL; definition: "end of tax year?"'},
    ("21", "across"): {"def": "Fleshy tissue", "wtype": "anagram", "expl": 'anagram of FAT + APOSTLE = SOFTPALATE; definition: "Fleshy tissue"'},
    ("23", "down"): {"def": "completely collapsed", "wtype": "deletion", "expl": 'TWENTY (synonym="Score,") with deletion = WENT; definition: "completely collapsed"'},
    # Sonnet explanations
    ("9", "across"): {"def": "Indian?", "wtype": "charade", "expl": 'CUR(synonym of "Despicable") + RY("tracks"); definition: "Indian?"'},
    ("11", "across"): {"def": "get niggly", "wtype": "charade", "expl": 'SPLIT(synonym of "fracture") + H("boxers") + AIRS(synonym of "communication"); definition: "get niggly"'},
    ("15", "down"): {"def": "still water", "wtype": "charade", "expl": 'BILL(synonym of "Price") + A + BONG(synonym of "toll"); definition: "still water"'},
    ("22", "down"): {"def": "leave", "wtype": "charade", "expl": "A + D(core of \"Stop\") + I(first letter of \"I'm\") + E(core of \"taking\") + U(abbr. \"uniform\"); definition: \"leave\""},
    ("24", "across"): {"def": "pardon", "wtype": "charade", "expl": 'EX(abbr. "Old") + ONE(synonym of "I") + RATE(synonym of "fancy") = EXONERATE; definition: "pardon"'},
    ("3", "down"): {"def": "Bane of comic character", "wtype": "container", "expl": 'K(last letter of "shtick") + YPTONIT("pity not") inside RE(deletion from "routine"); definition: "Bane of comic character"'},
    ("5", "down"): {"def": "Chest bone", "wtype": "charade", "expl": 'TRUE(synonym of "perfectly") + RIB(synonym of "bone"); definition: "Chest bone"'},
}

# Direction overrides for ambiguous clue numbers
DIRECTION_OVERRIDES = {
    10052633: "across",  # clue 1 manual_approve -> 1a BLANKETSTITCH
    10052658: "down",    # clue 19 (def=port) -> 19d ANTWERP
    10052641: "across",  # clue 19 (def=Tender-hearted) -> 19a ADORING
}


def main():
    conn = sqlite3.connect(str(DB))
    conn.row_factory = sqlite3.Row

    # Build new clue ID map
    new_clues = conn.execute(
        "SELECT id, clue_number, direction FROM clues "
        "WHERE source = 'times' AND puzzle_number = '29514'"
    ).fetchall()
    new_map = {}
    for c in new_clues:
        new_map[(c["clue_number"], c["direction"])] = c["id"]

    # Step 1: Restore ai_explanations from puzzle report
    restored = 0
    for (clue_num, direction), data in REPORT_EXPLANATIONS.items():
        new_id = new_map.get((clue_num, direction))
        if not new_id:
            print(f"  SKIP {clue_num}{direction[0]}: no matching new clue row")
            continue
        conn.execute(
            "UPDATE clues SET definition=?, wordplay_type=?, ai_explanation=?, has_solution=1 WHERE id=?",
            (data["def"], data["wtype"], data["expl"], new_id),
        )
        restored += 1

    # Step 2: Relink orphaned SE rows
    orphaned = conn.execute(
        "SELECT * FROM structured_explanations "
        "WHERE source = 'times' AND puzzle_number = '29514'"
    ).fetchall()

    relinked = 0
    for o in orphaned:
        old_id = o["clue_id"]
        clue_num = o["clue_number"]

        if old_id in DIRECTION_OVERRIDES:
            direction = DIRECTION_OVERRIDES[old_id]
        else:
            candidates = [(k, v) for k, v in new_map.items() if k[0] == clue_num]
            if len(candidates) == 1:
                direction = candidates[0][0][1]
            else:
                print(f"  SKIP SE old={old_id} clue={clue_num}: ambiguous direction")
                continue

        new_id = new_map.get((clue_num, direction))
        if not new_id:
            continue

        # Don't overwrite if already relinked
        existing = conn.execute(
            "SELECT 1 FROM structured_explanations WHERE clue_id=?", (new_id,)
        ).fetchone()
        if existing:
            continue

        conn.execute(
            "INSERT INTO structured_explanations "
            "(clue_id, definition_text, wordplay_types, components, model_version, confidence, "
            "created_at, updated_at, source, puzzle_number, clue_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (new_id, o["definition_text"], o["wordplay_types"], o["components"],
             o["model_version"], o["confidence"], o["created_at"], o["updated_at"],
             o["source"], o["puzzle_number"], o["clue_number"]),
        )
        relinked += 1

    conn.commit()
    print(f"Explanations restored: {restored}")
    print(f"SE rows relinked: {relinked}")

    # Verify
    print("\nVerification:")
    rows = conn.execute(
        "SELECT c.clue_number, c.direction, "
        "  c.ai_explanation IS NOT NULL AND c.ai_explanation != '' as has_expl, "
        "  se.confidence, se.model_version "
        "FROM clues c "
        "LEFT JOIN structured_explanations se ON se.clue_id = c.id "
        "WHERE c.source = 'times' AND c.puzzle_number = '29514' "
        "ORDER BY CASE c.direction WHEN 'across' THEN 0 ELSE 1 END, "
        "  CAST(c.clue_number AS INTEGER)",
    ).fetchall()
    for r in rows:
        conf = r["confidence"]
        score = int(conf * 100) if conf and conf <= 1 else (int(conf) if conf else 0)
        model = (r["model_version"] or "")[:25]
        tier = "HIGH" if score >= 80 else ("MED" if score >= 40 else ("LOW" if score > 0 else "PEND"))
        print(f"  {r['clue_number']:>3}{r['direction'][0]} expl={r['has_expl']} {tier:4} {score:3} {model}")

    conn.close()


if __name__ == "__main__":
    main()
