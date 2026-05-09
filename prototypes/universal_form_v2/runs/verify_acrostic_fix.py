"""Verify the acrostic indicator-strictness fix:
 1. MOONS should NOT translate as an acrostic merge (regularly→parts only)
 2. The 7 clean acrostic PASSes should still PASS
 3. The 4 simple positional PASSes should still PASS
"""
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


TARGETS = [
    # BARBS — pure 5-piece acrostic (covers abbr 'British→B' folded in)
    ("Leaders of British archaeological research body support digs", "BARBS",
     "BARBS — should be a pure acrostic of all 5 words"),
    # OCTET — 5-piece acrostic with abbr 'team→T' bridged
    ("Group or classical team, eight then first of all", "OCTET",
     "OCTET — should be a pure 5-piece acrostic"),
    # MOONS — should NOT acrostic-merge (regularly only typed `parts`)
    ("Low nest regularly seen in pines", "MOONS",
     "MOONS — should NOT acrostic-merge (regularly is not acrostic-typed)"),
    # OWLET — only 1 fl piece, no anchor
    ("Initially obstructive wife to enable young head-turner", "OWLET",
     "OWLET — charade of 3 pieces, no acrostic"),
    # AWASH — only 1 fl piece, no anchor
    ("A waterway initially remains flooded", "AWASH",
     "AWASH — charade of 3 pieces, no acrostic (A is literal)"),
    # PROSE / TERRACE / BEAGLE — single fl piece, charade
    ("Writing introduction to perennial beauty", "PROSE",
     "PROSE — charade with single positional[first], should PASS"),
    ("Balcony initially terrified the Queen and family", "TERRACE",
     "TERRACE — charade with single positional[first], should PASS"),
    ("Breed's first bird dog", "BEAGLE",
     "BEAGLE — charade with single positional[first], should PASS"),
    # FOSSE / ISM / GROUPER / ROBUST — clean acrostics
    ("Ditch openings of foreign operas", "FOSSE",
     "FOSSE — pure 5-piece acrostic"),
    ("Doctrine leads to intolerance, spite and malice", "ISM",
     "ISM — pure 3-piece acrostic"),
    ("Fish category initially excludes rays", "GROUPER",
     "GROUPER — charade(synonym, acrostic)"),
    ("Leaders for reasons obscure bankrupt firm", "ROBUST",
     "ROBUST — charade(acrostic, synonym)"),
]


def main():
    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    for clue_pat, ans, label in TARGETS:
        rows = master.execute(
            """
            SELECT se.id AS se_id, se.clue_id, se.components, se.wordplay_types,
                   se.definition_text, c.clue_text, c.answer
            FROM structured_explanations se
            JOIN clues c ON c.id = se.clue_id
            WHERE c.clue_text LIKE ? AND UPPER(c.answer) = UPPER(?)
            LIMIT 1
            """,
            (f"{clue_pat}%", ans),
        ).fetchall()
        if not rows:
            print(f"[NOT FOUND] {label}")
            continue
        r = rows[0]
        form, err = translate_components(r, db)
        if err:
            outcome = f"TRANSLATION_ERROR: {err['detail'][:80]}"
            has_acrostic = False
        else:
            v = verify(form, r["clue_text"], db)
            tree_str = json.dumps(form.tree.to_dict(), ensure_ascii=False)
            has_acrostic = '"acrostic"' in tree_str
            outcome = f"{v.verdict}  acrostic_in_tree={has_acrostic}"
        print(f"[{outcome}] {label}")
        print(f"   clue: {r['clue_text']}")


if __name__ == "__main__":
    main()
