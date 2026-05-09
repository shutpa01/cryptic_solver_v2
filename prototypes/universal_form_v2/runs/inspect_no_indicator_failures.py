"""Pull and categorise the 'no matching indicator found' failures
from the most recent first_letter batch."""
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

from prototypes.universal_form_v2.shadow_db import ensure_shadow


def main():
    s = ensure_shadow()
    s.row_factory = sqlite3.Row
    rows = s.execute(
        """
        SELECT clue_text, answer, components_json, failure_detail
        FROM seed_failures
        WHERE failure_kind='translation_error'
          AND failure_detail LIKE '%no matching indicator found%'
          AND components_json LIKE '%"mechanism": "first_letter"%'
        """,
    ).fetchall()
    print(f"Total no-indicator failures: {len(rows)}")
    print()

    bucket_acrostic = []      # 2+ first_letter pieces in components
    bucket_single = []        # exactly 1 first_letter piece
    bucket_other = []

    for r in rows:
        try:
            comp = json.loads(r["components_json"])
        except Exception:
            bucket_other.append((r["clue_text"], r["answer"], "json_parse"))
            continue
        fl = [p for p in comp.get("ai_pieces", [])
              if (p.get("mechanism") or "").lower() == "first_letter"]
        if len(fl) >= 2:
            bucket_acrostic.append((r["clue_text"], r["answer"],
                                    [p.get("clue_word") for p in fl],
                                    [p.get("letters") for p in fl]))
        elif len(fl) == 1:
            bucket_single.append((r["clue_text"], r["answer"],
                                   fl[0].get("clue_word"),
                                   fl[0].get("letters"),
                                   r["failure_detail"]))
        else:
            bucket_other.append((r["clue_text"], r["answer"], "no fl pieces"))

    print(f"=== ACROSTIC pattern (2+ first_letter pieces): "
          f"{len(bucket_acrostic)} ===")
    for clue, ans, sources, letters in bucket_acrostic[:25]:
        joined = " + ".join(f"{s}->{l}" for s, l in zip(sources, letters))
        print(f"  {ans:18s}  {clue[:75]}")
        print(f"    fl pieces: {joined}")
    if len(bucket_acrostic) > 25:
        print(f"  ... {len(bucket_acrostic) - 25} more not shown")
    print()

    print(f"=== SINGLE first_letter piece (1 piece, no indicator found): "
          f"{len(bucket_single)} ===")
    for clue, ans, src, lett, detail in bucket_single:
        print(f"  {ans:18s}  {clue[:75]}")
        print(f"    fl piece: {src!r} -> {lett}")
    print()

    print(f"=== OTHER ({len(bucket_other)}) ===")
    for x in bucket_other:
        print(f"  {x}")

    print()
    print("=" * 70)
    print(f"Acrostic pattern: {len(bucket_acrostic)}/{len(rows)} "
          f"({100*len(bucket_acrostic)//len(rows)}%)")
    print(f"Single-piece     : {len(bucket_single)}/{len(rows)} "
          f"({100*len(bucket_single)//len(rows)}%)")
    print(f"Other            : {len(bucket_other)}/{len(rows)}")


if __name__ == "__main__":
    main()
