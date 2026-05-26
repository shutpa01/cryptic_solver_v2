"""Fetch fifteensquared blog for Indy 12352 and overwrite the answer-only stub
explanations with real wordplay text. No Haiku parsing — just the raw blog
text saved into clues.explanation so we can use it as compass."""
import re
import sqlite3
import sys
from pathlib import Path

sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from sonnet_pipeline.fifteensquared_pipeline import fetch_fifteensquared

SOURCE = 'independent'
PUZZLE = 12352
PUB_DATE = '2026-05-11'

fs_clues = fetch_fifteensquared(PUZZLE, SOURCE, PUB_DATE)
if not fs_clues:
    print("No fifteensquared blog found.")
    sys.exit(1)

print(f"Fetched {len(fs_clues)} clues.\n")

conn = sqlite3.connect(str(ROOT / 'data' / 'clues_master.db'))
updated = 0
no_match = []
for fc in fs_clues:
    if not fc.get('explanation'):
        continue
    answer_clean = re.sub(r'[^A-Z]', '', fc['answer'].upper())
    # OVERWRITE the answer-only stub. (The normal pipeline only writes when explanation is empty;
    # here we have stubs that need replacing.)
    result = conn.execute(
        """UPDATE clues SET explanation = ?
           WHERE source = ? AND puzzle_number = ?
             AND UPPER(REPLACE(REPLACE(answer, ' ', ''), '-', '')) = ?""",
        (fc['explanation'], SOURCE, str(PUZZLE), answer_clean),
    )
    if result.rowcount > 0:
        updated += 1
        print(f"  {fc.get('clue_number','?')}{fc.get('direction','?')[:1]} {fc['answer']:20} (len={len(fc['explanation'])})")
    else:
        no_match.append(fc['answer'])
conn.commit()
conn.close()
print(f"\nUpdated {updated} blog explanations.")
if no_match:
    print(f"No DB match for: {no_match}")
