"""Re-run try_charade (with the gate-on-licensing-indicator fix in
place) on the Category C 'single first_letter, no findable indicator'
failures. Classify each by outcome.

Outcomes:
  - try_charade returns a valid assembly (potentially with the
    correct mechanism this time)
  - try_charade returns None (assembly couldn't be built — same
    shape as FAR EAST)
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from signature_solver.db import RefDB
from backfill_ai_exp.batch_v1_solver import try_charade
from prototypes.universal_form_v2.shadow_db import ensure_shadow


def main():
    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    s = ensure_shadow()
    s.row_factory = sqlite3.Row
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    rows = s.execute(
        """
        SELECT clue_id, clue_text, answer, components_json, failure_detail
        FROM seed_failures
        WHERE failure_kind='translation_error'
          AND failure_detail LIKE '%no matching indicator found%'
        """,
    ).fetchall()

    # Filter to those with EXACTLY 1 first_letter piece (Category C shape)
    cat_c = []
    for r in rows:
        try:
            comp = json.loads(r["components_json"])
        except Exception:
            continue
        fl_count = sum(1 for p in comp.get("ai_pieces", [])
                        if (p.get("mechanism") or "").lower() == "first_letter")
        if fl_count == 1:
            cat_c.append(r)
    print(f"Category C clues to test: {len(cat_c)}")
    print()

    outcome_counts = Counter()
    examples = {"new_pieces": [], "none": []}

    for r in cat_c:
        # Need to recompute remaining_words. Use definition_text from SE row.
        se = master.execute(
            "SELECT definition_text, components FROM structured_explanations "
            "WHERE clue_id = ? LIMIT 1",
            (r["clue_id"],),
        ).fetchone()
        def_text = (se["definition_text"] if se else "") or ""

        # Strip enumeration and split clue
        import re
        clue_clean = re.sub(r"\s*\([0-9,\-\s]+\)\s*$", "", r["clue_text"]).strip()
        words = clue_clean.split()
        # Remove definition words from the end (find_definition logic
        # is more complex, but for our purposes this approximates)
        def_tokens = def_text.split()
        if def_tokens and len(words) >= len(def_tokens):
            if [w.lower() for w in words[-len(def_tokens):]] == [t.lower() for t in def_tokens]:
                remaining = words[:-len(def_tokens)]
            elif [w.lower() for w in words[:len(def_tokens)]] == [t.lower() for t in def_tokens]:
                remaining = words[len(def_tokens):]
            else:
                remaining = words
        else:
            remaining = words

        result = try_charade(remaining, r["answer"], db)

        if result is None:
            outcome_counts["None"] += 1
            if len(examples["none"]) < 8:
                examples["none"].append((r["answer"], r["clue_text"], remaining))
        else:
            # Inspect the new pieces to see if they differ from old (which had first_letter)
            new_mechs = [p.get("mechanism", "") for p in result.get("pieces", [])]
            has_first_letter = any("first_letter" in m for m in new_mechs)
            if has_first_letter:
                outcome_counts["assembly_now_with_licensed_first_letter"] += 1
            else:
                outcome_counts["assembly_now_without_first_letter"] += 1
            if len(examples["new_pieces"]) < 8:
                examples["new_pieces"].append(
                    (r["answer"], r["clue_text"], result.get("pieces", []))
                )

    print("Outcomes:")
    for k, v in outcome_counts.most_common():
        print(f"  {v:3d}  {k}")
    print()

    print(f"=== Examples: assembly produced (no longer None) ===")
    for ans, clue, pieces in examples["new_pieces"][:8]:
        print(f"  {ans:18s}  {clue}")
        for p in pieces:
            print(f"    {p}")
        print()

    print(f"=== Examples: returns None (FAR-EAST shape) ===")
    for ans, clue, remaining in examples["none"][:8]:
        print(f"  {ans:18s}  {clue}")
        print(f"    remaining_words: {remaining}")


if __name__ == "__main__":
    main()
