"""Targeted test: feed a sample of first_letter charade clues through
the translator+verifier and report results. Throwaway script — used
to verify the step-2a translator extension before running a full
seed batch."""
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


def main():
    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    rows = master.execute(
        """
        SELECT se.id AS se_id, se.clue_id, se.components, se.wordplay_types,
               se.definition_text, se.confidence, se.model_version,
               c.source, c.puzzle_number, c.clue_number, c.direction,
               c.clue_text, c.answer
        FROM structured_explanations se
        JOIN clues c ON c.id = se.clue_id
        WHERE c.answer IS NOT NULL AND c.answer != ''
          AND se.wordplay_types = '["charade"]'
          AND se.confidence >= 0.85
          AND se.components LIKE '%"mechanism": "first_letter"%'
        ORDER BY RANDOM()
        LIMIT 60
        """,
    ).fetchall()

    print(f"Testing {len(rows)} first_letter charade clues")
    print("=" * 70)

    pass_count = 0
    trans_err = 0
    verif_fail = 0

    for r in rows:
        print(f"\nclue:   {r['clue_text']}")
        print(f"answer: {r['answer']}")
        comp = json.loads(r["components"])
        fl_pieces = [p for p in comp.get("ai_pieces", [])
                      if (p.get("mechanism") or "").lower() == "first_letter"]
        for p in fl_pieces:
            print(f"  fl piece: {json.dumps(p, ensure_ascii=False)}")
        form, err = translate_components(r, db)
        if err:
            trans_err += 1
            print(f"  >> TRANSLATION_ERROR: {err['detail'][:140]}")
            continue
        print(f"  >> translated. tree: "
              f"{json.dumps(form.tree.to_dict(), ensure_ascii=False)[:200]}")
        v = verify(form, r["clue_text"], db)
        print(f"  >> VERIFIER: {v.verdict}")
        if v.verdict == "PASS":
            pass_count += 1
        else:
            verif_fail += 1
            for c in v.checks:
                if c.status == "fail":
                    print(f"     FAIL {c.name}: {c.detail[:140]}")

    print()
    print("=" * 70)
    print(f"PASS: {pass_count}   translation_error: {trans_err}   "
          f"verifier_fail: {verif_fail}")


if __name__ == "__main__":
    main()
