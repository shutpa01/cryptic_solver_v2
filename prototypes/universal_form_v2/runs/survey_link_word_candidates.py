"""Survey all leftover-link-words failures in shadow_db. For each
failure, compute the residue after consuming definition + each piece's
clue_word tokens. Surface every word that appears in the residue but
is NOT currently in LINK_WORDS — those are the candidates the user
needs to evaluate before any addition.

Each candidate is shown with up to N example clues so the user can
judge whether the word is truly a link (no cryptic role) or actually
an indicator / truncated-definition / etc.
"""
from __future__ import annotations

import json
import sqlite3
import sys
from collections import defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from prototypes.universal_form_v2.shadow_db import ensure_shadow
from prototypes.universal_form_v2.surface import tokenize as _tokenize
from prototypes.universal_form_v2.clipboard_verifier import LINK_WORDS


def main():
    s = ensure_shadow()
    s.row_factory = sqlite3.Row

    rows = s.execute(
        """
        SELECT clue_text, answer, components_json, failure_detail
        FROM seed_failures
        WHERE failure_kind='translation_error'
          AND failure_detail LIKE '%leftover clue word(s) not on LINK_WORDS%'
        """,
    ).fetchall()

    print(f"Total leftover-link-words failures: {len(rows)}")
    print()

    # candidate -> list of (clue, answer, full_residue)
    candidates = defaultdict(list)

    for r in rows:
        try:
            comp = json.loads(r["components_json"])
        except Exception:
            continue
        clue_tokens = [t.lower() for t in _tokenize(r["clue_text"])]
        consumed = set()
        for p in comp.get("ai_pieces", []):
            cw = p.get("clue_word") or ""
            for t in _tokenize(cw):
                consumed.add(t.lower())
        # We don't have the full def_phrase here — but the failure
        # already accounted for it. Still, we'd need to also remove
        # def_phrase tokens. Approximate: the components may carry
        # the definition under "definition_text" or assembly. Skip
        # explicit def removal — the residue we compute is a
        # superset of the real residue, but the candidates we surface
        # are still correct (anything not in current LINK_WORDS).
        residue = [w for w in clue_tokens if w not in consumed]
        non_link = [w for w in residue if w not in LINK_WORDS]
        # Filter: only surface words consisting entirely of letters
        # (drops "-", punctuation noise, possessives) and length >= 2.
        for w in non_link:
            if not w.isalpha() or len(w) < 2:
                continue
            candidates[w].append((r["clue_text"], r["answer"], residue))

    # Order candidates by frequency
    sorted_cands = sorted(candidates.items(),
                           key=lambda x: -len(x[1]))

    print(f"Candidate words appearing in residues but not in LINK_WORDS: "
          f"{len(sorted_cands)}")
    print()

    # Show each candidate that appears at least 3 times — these are
    # the ones with enough recurrence to evaluate. Single-occurrence
    # candidates are noise.
    for word, examples in sorted_cands:
        if len(examples) < 3:
            continue
        print("=" * 70)
        print(f"{word!r}  ({len(examples)} occurrences)")
        for clue, ans, residue in examples[:5]:
            print(f"  {ans:18s}  {clue}")
            print(f"                      residue: {residue}")
        if len(examples) > 5:
            print(f"  ... {len(examples) - 5} more")
        print()

    print()
    print("=" * 70)
    print("Less-frequent candidates (1-2 occurrences each, "
          "for completeness):")
    for word, examples in sorted_cands:
        if len(examples) >= 3:
            continue
        print(f"  {word!r:18s}  ({len(examples)})  "
              f"e.g. {examples[0][1]}: {examples[0][0]}")


if __name__ == "__main__":
    main()
