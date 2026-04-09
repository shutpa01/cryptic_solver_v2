"""Insert Cordelia's tutorial puzzle into the database."""
import sqlite3
import json
from datetime import datetime

conn = sqlite3.connect("data/clues_master.db")
cur = conn.cursor()

# Check if already exists
existing = cur.execute("SELECT COUNT(*) FROM clues WHERE source='cordelia'").fetchone()[0]
if existing:
    print(f"Already have {existing} cordelia clues — deleting and re-inserting")
    ids = [r[0] for r in cur.execute("SELECT id FROM clues WHERE source='cordelia'").fetchall()]
    for cid in ids:
        cur.execute("DELETE FROM structured_explanations WHERE clue_id=?", (cid,))
    cur.execute("DELETE FROM clues WHERE source='cordelia'")
    cur.execute("DELETE FROM puzzle_grids WHERE source='cordelia'")
    conn.commit()

now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
pub_date = "2026-04-09"

# (clue_num, direction, clue_text, enum, answer, definition, wtype,
#  ai_expl, explanation, components_dict, wtype_json, confidence)
clues = [
    # === ACROSS ===
    ("1", "across",
     "Twist of fate's causing blow-out", "5", "FEAST",
     "blow-out", "anagram",
     "anagram of FATE'S = FEAST; definition: \"blow-out\"",
     "More food. Lots of it can be found by making an anagram (twist of) of FATE'S",
     {"ai_pieces": [{"clue_word": "fate's", "letters": "FEAST", "mechanism": "anagram_fodder"}],
      "assembly": {"op": "anagram", "fodder": ["FATES"], "gives": "FEAST"},
      "wordplay_type": "anagram"},
     '["anagram"]', 0.95),

    ("2", "across",
     "Hide a broken heart", "5", "EARTH",
     "Hide", "anagram",
     "anagram of HEART = EARTH; definition: \"Hide\"",
     None,
     {"ai_pieces": [{"clue_word": "heart", "letters": "EARTH", "mechanism": "anagram_fodder"}],
      "assembly": {"op": "anagram", "fodder": ["HEART"], "gives": "EARTH"},
      "wordplay_type": "anagram"},
     '["anagram"]', 0.95),

    ("3", "across",
     "Popular guest admitting row", "5", "ARGUE",
     "row", "hidden",
     'hidden in "populAR GUEst"; definition: "row"',
     None,
     {"ai_pieces": [{"clue_word": "popular guest", "letters": "ARGUE", "mechanism": "hidden"}],
      "assembly": {"op": "hidden", "words": "popular guest"},
      "wordplay_type": "hidden"},
     '["hidden"]', 0.95),

    ("4", "across",
     "Pack things", "5", "STUFF",
     None, "double_definition",
     "Double definition: pack = STUFF, things = STUFF",
     None,
     {"ai_pieces": [],
      "assembly": {"op": "double_definition", "left_def": "Pack", "right_def": "things"},
      "wordplay_type": "double_definition"},
     '["double_definition"]', 0.90),

    ("5", "across",
     "Stealing the newspaper", "5", "THEFT",
     "Stealing", "charade",
     'THE (literal) + FT (newspaper abbreviation) = THEFT; definition: "Stealing"',
     None,
     {"ai_pieces": [{"clue_word": "the", "letters": "THE", "mechanism": "synonym"},
                     {"clue_word": "newspaper", "letters": "FT", "mechanism": "abbreviation"}],
      "assembly": {"op": "charade", "order": ["THE", "FT"]},
      "wordplay_type": "charade"},
     '["charade"]', 0.95),

    # === DOWN ===
    ("1", "down",
     "Provided in cafe, a stonking big meal", "5", "FEAST",
     "meal", "hidden",
     'hidden in "caFE A STonking"; definition: "meal"',
     None,
     {"ai_pieces": [{"clue_word": "cafe, a stonking", "letters": "FEAST", "mechanism": "hidden"}],
      "assembly": {"op": "hidden", "words": "cafe, a stonking"},
      "wordplay_type": "hidden"},
     '["hidden"]', 0.95),

    ("2", "down",
     "Discovered near the ground", "5", "EARTH",
     "ground", "hidden",
     'hidden in "nEAR THe"; definition: "ground"',
     None,
     {"ai_pieces": [{"clue_word": "near the", "letters": "EARTH", "mechanism": "hidden"}],
      "assembly": {"op": "hidden", "words": "near the"},
      "wordplay_type": "hidden"},
     '["hidden"]', 0.95),

    ("3", "down",
     "A wild urge to quarrel", "5", "ARGUE",
     "quarrel", "anagram",
     'A + anagram of URGE = ARGUE; definition: "quarrel"',
     None,
     {"ai_pieces": [{"clue_word": "A", "letters": "A", "mechanism": "synonym"},
                     {"clue_word": "urge", "letters": "RGUE", "mechanism": "anagram_fodder"}],
      "assembly": {"op": "anagram", "fodder": ["A", "URGE"], "gives": "ARGUE"},
      "wordplay_type": "anagram"},
     '["anagram"]', 0.95),

    ("4", "down",
     "Material things", "5", "STUFF",
     None, "double_definition",
     "Double definition: material = STUFF, things = STUFF",
     None,
     {"ai_pieces": [],
      "assembly": {"op": "double_definition", "left_def": "Material", "right_def": "things"},
      "wordplay_type": "double_definition"},
     '["double_definition"]', 0.90),

    ("5", "down",
     "Taking offence", "5", "THEFT",
     None, "double_definition",
     "Double definition: taking = THEFT, offence = THEFT",
     None,
     {"ai_pieces": [],
      "assembly": {"op": "double_definition", "left_def": "Taking", "right_def": "offence"},
      "wordplay_type": "double_definition"},
     '["double_definition"]', 0.90),
]

for c in clues:
    (clue_num, direction, clue_text, enum, answer, definition, wtype,
     ai_expl, explanation, components_dict, wtype_json, confidence) = c

    components_json = json.dumps(components_dict)

    cur.execute("""
        INSERT INTO clues (source, puzzle_number, publication_date, clue_number, direction,
                          clue_text, enumeration, answer, definition, explanation, ai_explanation,
                          wordplay_type, has_solution, reviewed)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 1, 1)
    """, ("cordelia", "1", pub_date, clue_num, direction, clue_text, enum, answer,
          definition, explanation, ai_expl, wtype))

    clue_id = cur.lastrowid

    cur.execute("""
        INSERT INTO structured_explanations (clue_id, definition_text, wordplay_types, components,
                                            model_version, confidence, created_at, updated_at,
                                            source, puzzle_number, clue_number)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (clue_id, definition, wtype_json, components_json, "tutorial", confidence, now, now,
          "cordelia", "1", clue_num))

# Insert the grid
grid_solution = "FEASTEARTHARGUESTUFFTHEFT"
cur.execute("""
    INSERT OR REPLACE INTO puzzle_grids (source, puzzle_number, solution, grid_rows, grid_cols)
    VALUES (?, ?, ?, ?, ?)
""", ("cordelia", "1", grid_solution, 5, 5))

conn.commit()

# Verify
count = cur.execute("SELECT COUNT(*) FROM clues WHERE source='cordelia'").fetchone()[0]
print(f"Cordelia clues in DB: {count}")
se_count = cur.execute("SELECT COUNT(*) FROM structured_explanations WHERE source='cordelia'").fetchone()[0]
print(f"Structured explanations: {se_count}")
grid = cur.execute(
    "SELECT grid_rows, grid_cols, LENGTH(solution) FROM puzzle_grids WHERE source='cordelia'"
).fetchone()
print(f"Grid: {grid[0]}x{grid[1]}, solution length: {grid[2]}")

# Show what we inserted
print("\nClues:")
for r in cur.execute(
    "SELECT clue_number, direction, clue_text, answer, wordplay_type FROM clues WHERE source='cordelia' ORDER BY direction, CAST(clue_number AS INTEGER)"
).fetchall():
    print(f"  {r[0]}{r[1][0]}: {r[2]} -> {r[3]} [{r[4]}]")

conn.close()
