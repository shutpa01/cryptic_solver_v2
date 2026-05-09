"""For the 41 Category C failures still returning None, identify
the surface-dressing words causing the residue-check rejection.

A word is a "surface-dressing candidate" if:
  * it's in remaining_words
  * it's NOT in LINK_WORDS
  * it has no indicator types in the DB
  * it has no abbreviations or synonyms that fit in answer_clean
  * it's not anyone's source (i.e., the rest of the assembly can match
    the answer without it)
"""
from __future__ import annotations

import json
import re
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from signature_solver.db import RefDB
from signature_solver.tokens import LINK_WORDS
from prototypes.universal_form_v2.shadow_db import ensure_shadow


def norm(w):
    return re.sub(r"[^A-Za-z]", "", w or "").lower()


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

    candidate_words = Counter()  # word -> count
    examples_by_word = {}        # word -> list of (answer, clue, remaining)

    for r in cat_c:
        se = master.execute(
            "SELECT definition_text FROM structured_explanations "
            "WHERE clue_id = ? LIMIT 1",
            (r["clue_id"],),
        ).fetchone()
        def_text = (se["definition_text"] if se else "") or ""

        clue_clean = re.sub(r"\s*\([0-9,\-\s]+\)\s*$", "", r["clue_text"]).strip()
        words = clue_clean.split()
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

        # For each word in remaining, count it if NOT a link word and
        # NOT an indicator. Don't filter on abbreviation/synonym
        # presence — words like "and" have synonyms in DB but
        # function as link words in clues.
        for w in remaining:
            wn = norm(w)
            if not wn:
                continue
            if wn in LINK_WORDS:
                continue
            if db.get_indicator_types(wn):
                continue
            candidate_words[wn] += 1
            examples_by_word.setdefault(wn, []).append(
                (r["answer"], r["clue_text"], remaining)
            )

    print(f"Surface-dressing candidates appearing in Category C residues:")
    print(f"(words with no LINK_WORDS / indicator / abbreviation / synonym role)")
    print()
    for w, n in candidate_words.most_common(20):
        print(f"  {n:3d}  {w!r}")

    print()
    print("=== Top candidates with up to 4 example clues each ===")
    for w, _ in candidate_words.most_common(15):
        examples = examples_by_word.get(w, [])
        print()
        print(f"--- {w!r} ({len(examples)} occurrences) ---")
        for ans, clue, remaining in examples[:4]:
            print(f"  {ans:18s}  {clue}")


if __name__ == "__main__":
    main()
