"""Audit how often grammar_triage solves a clue and produces a
verifier-accepted Form, across a random sample.

For each clue:
  1. Tokenise.
  2. Extract definition candidates.
  3. For each (def, wp) pair, run grammar_triage.
  4. If it returns a high-confidence solve, push through the
     conversion chain (sig_adapter -> json_translator) to a Form.
  5. Run the clipboard verifier on the Form.

Buckets each clue into one of:
  GT_PASS         — grammar_triage solved AND form verifier-passed.
  GT_REJECT       — grammar_triage solved but form rejected (translator
                    or verifier).
  GT_MISS         — grammar_triage didn't solve (or no def candidate).
  NO_DEF          — no definition candidate found, can't run gt at all.

Usage: python -m prototypes.universal_form_v2.runs.audit_grammar_triage [N]
N defaults to 200.
"""
from __future__ import annotations

import json
import sqlite3
import sys
import time
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

sys.stdout.reconfigure(encoding="utf-8")

from signature_solver.db import RefDB
from signature_solver.solver import extract_definition_candidates
from signature_solver.grammar_triage import grammar_triage
from sonnet_pipeline.sig_adapter import (
    build_ai_pieces, build_assembly_dict,
)
from sonnet_pipeline.solver import clean
from prototypes.universal_form_v2.surface import tokenize
from prototypes.universal_form_v2.json_translator import (
    translate_components,
)
from prototypes.universal_form_v2.clipboard_verifier import verify


def audit_clue(clue_text, answer_clean, db):
    """Bucket a single clue. Returns (label, detail)."""
    try:
        tokens = tokenize(clue_text)
        def_candidates = extract_definition_candidates(
            tokens, answer_clean, db)
    except Exception as e:
        return "NO_DEF", f"def-extract error: {e}"

    if not def_candidates:
        return "NO_DEF", "no definition candidate"

    for def_phrase, wp_words in def_candidates:
        if not wp_words:
            continue
        try:
            gt = grammar_triage(clue_text, answer_clean, db,
                                 def_phrase=def_phrase,
                                 wp_words=list(wp_words))
        except Exception:
            continue
        if not gt or gt.result is None:
            continue

        # Conversion chain.
        try:
            ai_pieces = build_ai_pieces(gt)
            assembly = build_assembly_dict(gt)
        except Exception as e:
            return "GT_REJECT", f"sig_adapter error: {e}"
        if not ai_pieces or not assembly:
            return "GT_REJECT", "sig_adapter empty"

        components = json.dumps({
            "ai_pieces": ai_pieces,
            "assembly": assembly,
            "wordplay_type": assembly.get("op"),
        })
        row = {
            "clue_text": clue_text,
            "answer": answer_clean,
            "components": components,
            "definition_text": def_phrase,
        }
        try:
            form, err = translate_components(row, db)
        except Exception as e:
            return "GT_REJECT", f"translator error: {e}"
        if form is None:
            return "GT_REJECT", f"translator: {err}"

        try:
            v = verify(form, clue_text, db)
        except Exception as e:
            return "GT_REJECT", f"verifier error: {e}"
        if v.verdict == "PASS":
            return "GT_PASS", f"sig={assembly.get('op')}"
        return "GT_REJECT", f"verifier: {v.verdict}"

    return "GT_MISS", "no triage produced a solve"


def main():
    # Args: [N] or [source puzzle_number]
    source = None
    puzzle = None
    n = 200
    if len(sys.argv) == 2:
        n = int(sys.argv[1])
    elif len(sys.argv) >= 3:
        source = sys.argv[1]
        puzzle = sys.argv[2]
        n = int(sys.argv[3]) if len(sys.argv) > 3 else 100000

    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    master = sqlite3.connect(
        str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    if source and puzzle:
        print(f"Pulling all clues for {source} {puzzle} ...", flush=True)
        rows = master.execute(
            """
            SELECT id, clue_text, answer, source, puzzle_number,
                   clue_number, direction
            FROM clues
            WHERE source = ? AND puzzle_number = ?
              AND answer IS NOT NULL AND answer != ''
              AND clue_text IS NOT NULL AND clue_text != ''
            ORDER BY clue_number
            LIMIT ?
            """, (source, puzzle, n)).fetchall()
    else:
        print(f"Sampling {n} random clues with answers ...", flush=True)
        rows = master.execute(
            """
            SELECT id, clue_text, answer, source, puzzle_number,
                   clue_number
            FROM clues
            WHERE answer IS NOT NULL AND answer != ''
              AND clue_text IS NOT NULL AND clue_text != ''
              AND source NOT IN ('telegraph-toughie', 'cordelia')
            ORDER BY RANDOM()
            LIMIT ?
            """, (n,)).fetchall()
    print(f"Got {len(rows)} clues", flush=True)

    counts = Counter()
    t0 = time.time()
    examples = {}  # one example per bucket

    for i, r in enumerate(rows, 1):
        ans_clean = clean(r["answer"])
        if not ans_clean or len(ans_clean) < 3:
            counts["SKIP_SHORT"] += 1
            continue
        label, detail = audit_clue(r["clue_text"], ans_clean, db)
        counts[label] += 1
        if label not in examples:
            examples[label] = (r["clue_text"], r["answer"], detail)

        if i % 25 == 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(rows)} processed "
                  f"({elapsed:.0f}s, {elapsed/i:.2f}s/clue) "
                  f"GT_PASS={counts['GT_PASS']} "
                  f"GT_REJECT={counts['GT_REJECT']} "
                  f"GT_MISS={counts['GT_MISS']} "
                  f"NO_DEF={counts['NO_DEF']}",
                  flush=True)

    elapsed = time.time() - t0
    print()
    print(f"Finished {sum(counts.values())} clues in {elapsed:.0f}s",
          flush=True)
    print()
    total = sum(counts.values())
    for label in ("GT_PASS", "GT_REJECT", "GT_MISS", "NO_DEF",
                   "SKIP_SHORT"):
        n_ = counts[label]
        pct = 100 * n_ / total if total else 0
        print(f"  {label:12s} {n_:5d}  {pct:5.1f}%", flush=True)
    print()
    print("Examples (one per bucket):", flush=True)
    for label in ("GT_PASS", "GT_REJECT", "GT_MISS", "NO_DEF"):
        if label in examples:
            ct, ans, det = examples[label]
            print(f"  {label}: {ct[:80]} -> {ans}  ({det})",
                  flush=True)


if __name__ == "__main__":
    main()
