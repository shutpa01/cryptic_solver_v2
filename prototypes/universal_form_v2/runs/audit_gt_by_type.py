"""Measure grammar_triage performance AGAINST KNOWN wordplay types.

For a balanced sample of clues per known clues.wordplay_type, run the live
grammar_triage path (extract_definition_candidates -> wp_words -> grammar_triage)
and record, per known type:
  - tested
  - fired   : grammar_triage returned an answer-verified solve
  - op_match: of those that fired, the predicted mechanism matches the known type
  - the predicted-op distribution (the confusion / "noise")

This tests the redesign's "thin and noisy" claim:
  thin  = low fired% (grammar rarely produces a solve)
  noisy = low op_match% when it does fire / false-fires on DD/hidden it can't do

Ground-truth caveat: clues.wordplay_type is the system's own label, not gold; a
large sample still shows whether the grammar signal is informative.

Usage: python -m prototypes.universal_form_v2.runs.audit_gt_by_type [N_per_type]
"""
from __future__ import annotations

import sqlite3
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
sys.stdout.reconfigure(encoding="utf-8")

from signature_solver.db import RefDB
from signature_solver.solver import extract_definition_candidates
from signature_solver.grammar_triage import grammar_triage
from sonnet_pipeline.sig_adapter import build_assembly_dict
from sonnet_pipeline.solver import clean
from prototypes.universal_form_v2.surface import tokenize

TYPES = ["anagram", "charade", "container", "reversal", "deletion",
         "double_definition", "hidden", "homophone", "acrostic"]
HANDLED = {"anagram", "charade", "container", "reversal", "deletion"}  # gt's remit


def predicted_op(clue_text, answer_clean, db):
    """Run the live path; return (fired, op) — op is the mechanism gt assigned,
    or None if it produced nothing."""
    try:
        tokens = tokenize(clue_text)
        cands = extract_definition_candidates(tokens, answer_clean, db)
    except Exception:
        return False, None
    for def_phrase, wp_words in (cands or []):
        if not wp_words:
            continue
        try:
            gt = grammar_triage(clue_text, answer_clean, db,
                                def_phrase=def_phrase, wp_words=list(wp_words))
        except Exception:
            continue
        if not gt or getattr(gt, "result", None) is None:
            continue
        try:
            op = (build_assembly_dict(gt) or {}).get("op")
        except Exception:
            op = "?"
        return True, op
    return False, None


def main():
    n_per = int(sys.argv[1]) if len(sys.argv) > 1 else 60
    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row

    print(f"\n{n_per} clues per type\n" + "=" * 64, flush=True)
    t0 = time.time()
    for wtype in TYPES:
        rows = master.execute(
            "SELECT clue_text, answer FROM clues WHERE wordplay_type = ? "
            "AND answer IS NOT NULL AND answer != '' AND clue_text IS NOT NULL "
            "ORDER BY RANDOM() LIMIT ?", (wtype, n_per)).fetchall()
        tested = fired = match = 0
        ops = Counter()
        for r in rows:
            ans = clean(r["answer"])
            if not ans or len(ans) < 3:
                continue
            tested += 1
            f, op = predicted_op(r["clue_text"], ans, db)
            if f:
                fired += 1
                ops[op] += 1
                if op and wtype in str(op):
                    match += 1
        fpct = 100 * fired / tested if tested else 0
        mpct = 100 * match / fired if fired else 0
        tag = "(gt remit)" if wtype in HANDLED else "(control - gt should NOT fire)"
        print(f"\n{wtype:18s} {tag}", flush=True)
        print(f"   tested={tested}  fired={fired} ({fpct:.0f}%)  "
              f"op_match={match} ({mpct:.0f}% of fired)", flush=True)
        if ops:
            top = ", ".join(f"{o}:{c}" for o, c in ops.most_common(6))
            print(f"   predicted ops: {top}", flush=True)
    print(f"\nDone in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
