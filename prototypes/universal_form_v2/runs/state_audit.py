"""Audit the state of shadow_db so I can propose the next action with
evidence:

  1. Re-verify every existing PASS against the current translator
     and verifier. Any case where the stored form no longer verifies
     under the current rules is a stale PASS that needs cleaning up.

  2. For the 65 'no matching indicator found' failures from the
     earlier first_letter batches, re-run translation with the
     CURRENT translator (which has the anchor+extend acrostic rule).
     Count how many would now PASS — this is the empirical lift the
     anchor+extend rule gives us on previously-failed clues without
     having to run a fresh batch.

  3. Break down the remaining seed_failures by failure_detail to
     surface what's actually blocking us next.
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
from prototypes.universal_form_v2.shadow_db import ensure_shadow
from prototypes.universal_form_v2.json_translator import translate_components
from prototypes.universal_form_v2.clipboard_verifier import verify
from prototypes.universal_form_v2.extract_catalog import signature
from prototypes.universal_form_v2.schema import Form, Definition, Node


def _node_from_dict(d):
    """Reconstruct a Node from its dict representation."""
    n = Node(
        operation=d["operation"],
        indicator=d.get("indicator"),
        value=d.get("value"),
        source_word=d.get("source_word"),
        positional_kind=d.get("positional_kind"),
        deletion_kind=d.get("deletion_kind"),
        acrostic_kind=d.get("acrostic_kind"),
        flags=d.get("flags", []),
    )
    n.sources = [_node_from_dict(c) for c in d.get("sources", [])]
    return n


def main():
    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    s = ensure_shadow()
    s.row_factory = sqlite3.Row
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    # ----- Audit 1: re-verify every existing PASS -----
    print("=" * 70)
    print("Audit 1 — re-verify every stored PASS under current rules")
    print("=" * 70)
    pass_rows = s.execute(
        "SELECT id, clue_id, signature, answer, form_json FROM solves"
    ).fetchall()
    print(f"Total stored PASSes: {len(pass_rows)}")

    still_pass = 0
    new_fail = 0
    sig_changed = 0
    fail_examples = []

    for r in pass_rows:
        c = master.execute(
            "SELECT clue_text FROM clues WHERE id = ?",
            (r["clue_id"],),
        ).fetchone()
        if not c:
            continue
        # Re-verify the STORED form against current verifier rules.
        form_dict = json.loads(r["form_json"])
        tree = _node_from_dict(form_dict["tree"])
        defn = Definition(
            phrase=form_dict["definition"]["phrase"],
            answer=form_dict["definition"]["answer"],
        )
        form = Form(
            tree=tree, definition=defn,
            link_words=form_dict.get("link_words", []),
            is_and_lit=form_dict.get("is_and_lit", False),
            flags=form_dict.get("flags", []),
        )
        v = verify(form, c["clue_text"], db)
        if v.verdict == "PASS":
            still_pass += 1
        else:
            new_fail += 1
            fail_examples.append((r["answer"], c["clue_text"], v))

    print(f"  Still PASS:    {still_pass}")
    print(f"  Now FAIL:      {new_fail}")
    if fail_examples:
        print("  Examples of stored PASSes that now FAIL:")
        for ans, clue, v in fail_examples[:5]:
            print(f"    {ans}: {clue}")
            for c in v.checks:
                if c.status == "fail":
                    print(f"      FAIL {c.name}: {c.detail[:100]}")

    # ----- Audit 2: re-translate previous 'no indicator' failures -----
    print()
    print("=" * 70)
    print("Audit 2 — re-translate previous 'no indicator' failures with "
          "current rules")
    print("=" * 70)
    fail_rows = s.execute(
        """
        SELECT clue_id, clue_text, answer, components_json, failure_detail
        FROM seed_failures
        WHERE failure_kind='translation_error'
          AND failure_detail LIKE '%no matching indicator found%'
        """
    ).fetchall()
    print(f"Stored 'no indicator' failures to retest: {len(fail_rows)}")

    now_pass = 0
    still_fail = 0
    new_signatures = Counter()

    for fr in fail_rows:
        c = master.execute(
            """
            SELECT se.id AS se_id, se.clue_id, se.components, se.wordplay_types,
                   se.definition_text, c.clue_text, c.answer
            FROM structured_explanations se
            JOIN clues c ON c.id = se.clue_id
            WHERE c.id = ?
            LIMIT 1
            """,
            (fr["clue_id"],),
        ).fetchone()
        if not c:
            still_fail += 1
            continue
        form, err = translate_components(c, db)
        if err:
            still_fail += 1
            continue
        v = verify(form, c["clue_text"], db)
        if v.verdict == "PASS":
            now_pass += 1
            new_signatures[signature(form.tree)] += 1
        else:
            still_fail += 1

    print(f"  Now PASS:  {now_pass}")
    print(f"  Still FAIL: {still_fail}")
    if now_pass:
        print(f"  Signatures unlocked:")
        for sig, n in new_signatures.most_common():
            print(f"    {n:3d}  {sig}")

    # ----- Audit 3: current seed_failures breakdown -----
    print()
    print("=" * 70)
    print("Audit 3 — current seed_failures breakdown")
    print("=" * 70)
    by_pattern = Counter()
    for r in s.execute(
        "SELECT failure_detail FROM seed_failures "
        "WHERE failure_kind='translation_error'"
    ):
        d = r[0] or ""
        if "no matching indicator found" in d:
            by_pattern["no matching indicator found"] += 1
        elif "leftover clue word(s) not on LINK_WORDS" in d:
            by_pattern["leftover words not on LINK_WORDS"] += 1
        elif "pieces concat to" in d:
            by_pattern["charade pieces concat mismatch"] += 1
        elif "without left_def/right_def" in d:
            by_pattern["DD missing left/right def"] += 1
        elif "no anagram indicator" in d:
            by_pattern["anagram indicator missing"] += 1
        elif "no hidden indicator" in d:
            by_pattern["hidden indicator missing"] += 1
        elif "anagram with no fodder" in d:
            by_pattern["anagram with no fodder"] += 1
        elif "first_letter:" in d:
            by_pattern["first_letter data malformed"] += 1
        elif "not yet supported" in d:
            by_pattern["unsupported mechanism"] += 1
        else:
            by_pattern["other: " + d[:60]] += 1

    for pat, n in by_pattern.most_common():
        print(f"  {n:4d}  {pat}")


if __name__ == "__main__":
    main()
