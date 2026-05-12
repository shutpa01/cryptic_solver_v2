"""Dry-run verifier pass for Times 29540 parses.

No DB writes. Shows verdict / score for each clue + the failing checks,
so the parses can be tuned (or honestly accepted as LOW) before storage.
"""
import sqlite3, sys
from pathlib import Path
sys.stdout.reconfigure(encoding='utf-8')
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from sonnet_pipeline.verify_explanation import ExplanationVerifier
from scripts._times29540_parses import CLUES

CLUES_DB = ROOT / 'data' / 'clues_master.db'
conn = sqlite3.connect(str(CLUES_DB), timeout=30)
conn.row_factory = sqlite3.Row
verifier = ExplanationVerifier()

tiers = {'HIGH': [], 'MEDIUM': [], 'LOW': [], 'FAIL': []}
for clue_id, wtype, definition, expl in CLUES:
    row = conn.execute("SELECT * FROM clues WHERE id=?", (clue_id,)).fetchone()
    if not row: continue
    if wtype == 'unparsed':
        verdict, score = 'LOW', 25
        failing = []
    else:
        v = verifier.verify(
            clue_text=row['clue_text'], answer=row['answer'],
            wordplay_type=wtype, definition=definition, ai_explanation=expl,
        )
        score = v.get('score', 0)
        verdict = v.get('verdict', 'FAIL')
        failing = [c for c in v.get('checks', [])
                   if c.get('status') in ('wrong', 'unverifiable')]
    label = f"{row['clue_number']}{row['direction'][0]}"
    tiers[verdict].append((label, row['answer'], score, failing))
    print(f"[{verdict:6} {score:3}] {label:5} {row['answer']:14} ({wtype})")
    for f in failing[:8]:
        print(f"    {f['check']:25} {f['status']:13} {f.get('detail','')[:120]}")

print(f"\nTier summary:")
for t in ['HIGH', 'MEDIUM', 'LOW', 'FAIL']:
    print(f"  {t}: {len(tiers[t])}")
