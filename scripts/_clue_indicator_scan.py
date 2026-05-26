"""Scan each leftover clue for words that are DB indicators.
Helps anticipate 7b mechanism-hidden false positives.
"""
import sqlite3, sys, re
sys.path.insert(0, '.')
sys.stdout.reconfigure(encoding='utf-8')

clues_db = sqlite3.connect('data/clues_master.db')
clues_db.row_factory = sqlite3.Row
ref = sqlite3.connect('data/cryptic_new.db')
ref.row_factory = sqlite3.Row

ids = [10064940,10064941,10064942,10064943,10064944,10064946,10064947,10064948,10064949,10064950,10064951,10064952,10064953,10064954,10064955,10064957,10064959,10064960,10064962,10064963,10064964,10064965,10064966,10064967,10064968]

# Load indicators
indicators_by_word = {}
for row in ref.execute("SELECT word, wordplay_type FROM indicators"):
    w = row[0].lower().strip()
    indicators_by_word.setdefault(w, set()).add(row[1])

for cid in ids:
    r = clues_db.execute("SELECT clue_number, direction, clue_text, answer FROM clues WHERE id=?", (cid,)).fetchone()
    clue = r['clue_text'].lower().replace('’', "'")
    words = re.findall(r"[a-zA-Z]+(?:'[a-zA-Z]+)?", clue)
    phrases = [words[i]+' '+words[i+1] for i in range(len(words)-1)]
    hits = []
    for tok in words + phrases:
        if tok in indicators_by_word:
            hits.append((tok, indicators_by_word[tok]))
    print(f"{r['clue_number']}{r['direction'][0]:1} {r['answer']:13}: {r['clue_text']}")
    for tok, types in hits:
        print(f"     {tok!r}: {sorted(types)}")
    print()
