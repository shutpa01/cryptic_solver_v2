"""Inspect the 5 PASSes from the first_letter test in detail."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from signature_solver.db import RefDB
from prototypes.universal_form_v2.json_translator import translate_components
from prototypes.universal_form_v2.clipboard_verifier import verify


# Answer-text patterns of the 5 PASS clues
TARGETS = [
    ("Initially obstructive wife to enable young head-turner", "OWLET"),
    ("Despicable blokes initially tenanted bottom floor", "Basement"),
    ("Writing introduction to perennial beauty", "PROSE"),
    ("Balcony initially terrified the Queen and family", "TERRACE"),
    ("Breed's first bird dog", "BEAGLE"),
]


def main():
    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    for clue_pat, ans in TARGETS:
        rows = master.execute(
            """
            SELECT se.id AS se_id, se.clue_id, se.components, se.wordplay_types,
                   se.definition_text, se.confidence, se.model_version,
                   c.source, c.puzzle_number, c.clue_number, c.direction,
                   c.clue_text, c.answer
            FROM structured_explanations se
            JOIN clues c ON c.id = se.clue_id
            WHERE c.clue_text LIKE ? AND c.answer = ?
            LIMIT 1
            """,
            (f"{clue_pat}%", ans.upper()),
        ).fetchall()
        if not rows:
            print(f"NOT FOUND: {clue_pat!r} / {ans}")
            continue
        r = rows[0]
        print("=" * 70)
        print(f"clue:           {r['clue_text']}")
        print(f"answer:         {r['answer']}")
        print(f"definition:     {r['definition_text']!r}")
        print(f"model_version:  {r['model_version']}")
        comp = json.loads(r["components"])
        print("pieces:")
        for p in comp.get("ai_pieces", []):
            print(f"  {json.dumps(p, ensure_ascii=False)}")

        form, err = translate_components(r, db)
        if err:
            print(f"  TRANSLATION_ERROR: {err}")
            continue
        v = verify(form, r["clue_text"], db)
        print(f"verifier verdict: {v.verdict}")
        print("translator built tree:")
        print(json.dumps(form.to_dict(), indent=2, ensure_ascii=False))
        print()


if __name__ == "__main__":
    main()
