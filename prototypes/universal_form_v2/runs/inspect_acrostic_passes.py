"""Inspect the new acrostic-involving PASSes from shadow_db so we can
verify each explanation is cryptically faithful."""
from __future__ import annotations

import json
import sqlite3
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from prototypes.universal_form_v2.shadow_db import ensure_shadow


def render_node(node: dict, depth: int = 0) -> list:
    """Render a form-tree node as a list of indented strings."""
    pad = "  " * depth
    op = node.get("operation")
    ind = node.get("indicator")
    src = node.get("source_word")
    val = node.get("value")
    kind = node.get("positional_kind") or node.get("acrostic_kind")
    head = f"{pad}{op}"
    if kind:
        head += f"[{kind}]"
    if ind:
        head += f"  indicator={ind!r}"
    if val:
        head += f"  value={val!r}"
    if src:
        head += f"  src={src!r}"
    out = [head]
    for c in node.get("sources", []):
        out.extend(render_node(c, depth + 1))
    return out


def main():
    s = ensure_shadow()
    s.row_factory = sqlite3.Row
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    # Pull all acrostic-involving solves from this session
    rows = s.execute(
        """
        SELECT clue_id, signature, answer, form_json
        FROM solves
        WHERE signature LIKE '%acrostic%'
        ORDER BY id DESC
        LIMIT 8
        """,
    ).fetchall()

    print(f"=== Inspecting {len(rows)} most-recent acrostic PASSes ===\n")

    for r in rows:
        c = master.execute(
            "SELECT clue_text, source, puzzle_number, clue_number, direction "
            "FROM clues WHERE id = ?",
            (r["clue_id"],),
        ).fetchone()
        form = json.loads(r["form_json"])
        print("-" * 70)
        print(f"clue:      {c['clue_text']}")
        print(f"answer:    {r['answer']}")
        print(f"source:    {c['source']}/{c['puzzle_number']}/"
              f"{c['clue_number']} {c['direction']}")
        print(f"signature: {r['signature']}")
        print(f"definition: {form['definition']['phrase']!r}")
        print(f"link_words: {form['link_words']}")
        print("tree:")
        for line in render_node(form["tree"]):
            print("  " + line)
        print()


if __name__ == "__main__":
    main()
