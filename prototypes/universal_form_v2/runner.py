"""Run v0 adapter on a puzzle, tabulate results, print summary.

Usage:
    python -m prototypes.universal_form_v2.runner --source telegraph --puzzle 31132

The runner reads from clues_master.db (read-only), runs the adapter +
verifier, and writes a simple text report to stdout.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import traceback
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signature_solver.db import RefDB

from .adapter import solve_to_form, AdapterResult
from .verifier import verify, Verdict


def fetch_clues(source: str, puzzle: str):
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    conn.row_factory = sqlite3.Row
    rows = conn.execute("""
        SELECT id, clue_number, direction, clue_text, answer, enumeration
        FROM clues
        WHERE source = ? AND puzzle_number = ?
          AND answer IS NOT NULL AND answer != ''
        ORDER BY direction, CAST(clue_number AS INTEGER)
    """, (source, str(puzzle))).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_one(clue: dict, db: RefDB):
    try:
        ar = solve_to_form(clue["clue_text"], clue["answer"], db)
    except Exception as e:
        return ar_failure(e)
    if ar.form is None:
        return {
            "form": None,
            "verdict": None,
            "ar": ar,
            "exception": None,
        }
    try:
        verdict = verify(ar.form, clue["clue_text"], db)
    except Exception as e:
        return {
            "form": ar.form,
            "verdict": None,
            "ar": ar,
            "exception": traceback.format_exc(),
        }
    return {
        "form": ar.form,
        "verdict": verdict,
        "ar": ar,
        "exception": None,
    }


def ar_failure(exc):
    return {
        "form": None,
        "verdict": None,
        "ar": AdapterResult(form=None,
                            flags=[f"adapter_exception:{type(exc).__name__}"],
                            notes=str(exc),
                            sr_present=False,
                            confidence=0),
        "exception": traceback.format_exc(),
    }


def short_status(rec) -> str:
    if rec["form"] is None:
        return "NO_FORM"
    return rec["verdict"].verdict if rec["verdict"] else "VERIFY_ERR"


def print_clue(c, rec):
    label = f"{c['clue_number']:>3s}{c['direction'][:1]}"
    ans = c["answer"]
    status = short_status(rec)
    flags = rec["ar"].flags if rec["ar"] else []
    flag_str = ",".join(sorted(set(flags))[:5])
    print(f"  {label}  {ans:<18s}  {status:<8s}  flags=[{flag_str}]")
    if rec["form"] is not None and rec["verdict"]:
        for chk in rec["verdict"].checks:
            mark = "+" if chk.status == "pass" else "X"
            detail = (chk.detail[:80] + "...") if len(chk.detail) > 83 \
                else chk.detail
            print(f"      {mark} {chk.name:<22s} {detail}")
    elif rec["form"] is None:
        print(f"      notes: {rec['ar'].notes}")
    if rec["exception"]:
        print("      EXCEPTION trace (truncated):")
        for ln in rec["exception"].splitlines()[-3:]:
            print(f"        {ln}")


def run(source: str, puzzle: str, db: RefDB, verbose: bool = True):
    clues = fetch_clues(source, puzzle)
    if not clues:
        print(f"No clues for {source}/{puzzle}")
        return None

    results = []
    for c in clues:
        rec = run_one(c, db)
        results.append((c, rec))
        if verbose:
            print_clue(c, rec)

    # Summary
    statuses = Counter()
    flag_counts = Counter()
    for c, rec in results:
        statuses[short_status(rec)] += 1
        for f in (rec["ar"].flags if rec["ar"] else []):
            flag_counts[f] += 1

    print()
    print(f"=== Summary {source}/{puzzle} ({len(clues)} clues) ===")
    for k in ("PASS", "FAIL", "NO_FORM", "VERIFY_ERR"):
        if statuses[k]:
            print(f"  {k:<10s} {statuses[k]:>3d}")
    print()
    print("  flag counts (top 10):")
    for flag, count in flag_counts.most_common(10):
        print(f"    {count:>3d}  {flag}")

    return {
        "source": source,
        "puzzle": puzzle,
        "n_clues": len(clues),
        "statuses": dict(statuses),
        "flag_counts": dict(flag_counts),
        "per_clue": [
            {
                "clue_number": c["clue_number"],
                "direction": c["direction"],
                "clue_text": c["clue_text"],
                "answer": c["answer"],
                "status": short_status(rec),
                "flags": rec["ar"].flags if rec["ar"] else [],
                "verdict": rec["verdict"].to_dict() if rec["verdict"] else None,
                "form": rec["form"].to_dict() if rec["form"] else None,
                "notes": rec["ar"].notes if rec["ar"] else "",
            }
            for c, rec in results
        ],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", required=True)
    ap.add_argument("--puzzle", required=True)
    ap.add_argument("--out", default=None,
                    help="Optional path to write JSON results to")
    args = ap.parse_args()

    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    summary = run(args.source, args.puzzle, db)

    if args.out and summary:
        Path(args.out).write_text(json.dumps(summary, indent=2))
        print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
