"""Two-pass blog pipeline with shadow DB.

Pass 1: build forms from blogs, collect vocabulary candidates that the
        live DB doesn't have. Write them to shadow_blog_v0.db.
Pass 2: re-verify the forms with the shadow DB available. Many of the
        FAILs from pass 1 should now PASS (the bridge checks find the
        candidate entries in shadow).

Output: a single JSON with both pass-1 and pass-2 verdicts per clue.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from signature_solver.db import RefDB

from .db_anchored_mapper import map_clue_words_db
from .assembly_enumerator import assemble
from .verifier import verify
from .renderer import wordplay_type
from .shadow_db import (
    reset_shadow, write_candidates, ensure_shadow,
)


def fetch_clues(source: str, limit: int, puzzle_range=None):
    conn = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    conn.row_factory = sqlite3.Row
    where = ("source = ? AND answer != '' AND answer IS NOT NULL "
             "AND explanation IS NOT NULL AND explanation != ''")
    params = [source]
    if puzzle_range:
        lo, hi = puzzle_range
        where += " AND CAST(puzzle_number AS INTEGER) BETWEEN ? AND ?"
        params.extend([lo, hi])
    rows = conn.execute(
        f"""SELECT id, source, puzzle_number, clue_number, direction,
                   clue_text, answer, explanation, definition
            FROM clues WHERE {where}
            ORDER BY id LIMIT ?""",
        params + [limit]).fetchall()
    conn.close()
    return [dict(r) for r in rows]


def run_pass1(clues, db, ref_conn):
    """Build forms, verify against live DB only, collect candidates."""
    results = []
    all_candidates = []
    for c in clues:
        m = map_clue_words_db(
            c["clue_text"], c["answer"], c["explanation"], ref_conn,
            clue_definition=c.get("definition"))
        a = assemble(m)
        verdict_p1 = None
        if a.form is not None:
            try:
                verdict_p1 = verify(a.form, c["clue_text"], db,
                                      shadow_conn=None)
            except Exception:
                verdict_p1 = None
        cands = list(getattr(m, "shadow_candidates", []) or [])
        all_candidates.extend(cands)
        results.append({
            "clue_id": c["id"], "source": c["source"],
            "puzzle_number": c["puzzle_number"],
            "clue_number": c["clue_number"], "direction": c["direction"],
            "clue_text": c["clue_text"], "answer": c["answer"],
            "blog": c["explanation"],
            "definition_db": c.get("definition"),
            "mapping": m.to_dict(),
            "form": a.form.to_dict() if a.form else None,
            "wordplay_type": wordplay_type(a.form) if a.form else None,
            "pattern": a.pattern,
            "verdict_pass1": verdict_p1.to_dict() if verdict_p1 else None,
            "shadow_candidates": cands,
        })
    return results, all_candidates


def run_pass2(results, db, shadow_conn):
    """Re-verify each form with shadow_conn available."""
    for r in results:
        if not r["form"]:
            r["verdict_pass2"] = None
            continue
        # Reconstruct the form
        from .schema import Form, Definition
        from .html_report import _form_from_dict
        form = _form_from_dict(r["form"])
        try:
            v = verify(form, r["clue_text"], db, shadow_conn=shadow_conn)
            r["verdict_pass2"] = v.to_dict()
        except Exception as e:
            r["verdict_pass2"] = None
    return results


def summarise(results, label):
    statuses = Counter()
    for r in results:
        v = r.get(f"verdict_{label}")
        if v is None:
            if r["form"] is None:
                statuses["NO_FORM"] += 1
            else:
                statuses["VERIFY_ERR"] += 1
        else:
            statuses[v["verdict"]] += 1
    return dict(statuses)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="times")
    ap.add_argument("--limit", type=int, default=300)
    ap.add_argument("--puzzle-range", default=None)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    pr = None
    if args.puzzle_range:
        lo, hi = args.puzzle_range.split("-")
        pr = (int(lo), int(hi))

    print(f"[shadow] preparing fresh shadow DB...")
    shadow_conn = reset_shadow()

    clues = fetch_clues(args.source, args.limit, pr)
    print(f"[shadow] fetched {len(clues)} clues")

    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    ref_conn = sqlite3.connect(
        str(PROJECT_ROOT / "data" / "cryptic_new.db"))

    print(f"[shadow] pass 1 — build forms, collect candidates")
    results, candidates = run_pass1(clues, db, ref_conn)
    p1 = summarise(results, "pass1")
    print(f"  pass-1 verdicts: {p1}")

    print(f"[shadow] writing {len(candidates)} candidate rows to shadow DB")
    counts = write_candidates(candidates, shadow_conn)
    print(f"  written: {counts}")

    print(f"[shadow] pass 2 — re-verify with shadow DB")
    results = run_pass2(results, db, shadow_conn)
    p2 = summarise(results, "pass2")
    print(f"  pass-2 verdicts: {p2}")

    # Compute pass1 -> pass2 transitions
    transitions = Counter()
    for r in results:
        v1 = r.get("verdict_pass1")
        v2 = r.get("verdict_pass2")
        s1 = v1["verdict"] if v1 else ("NO_FORM" if not r["form"] else "?")
        s2 = v2["verdict"] if v2 else ("NO_FORM" if not r["form"] else "?")
        transitions[f"{s1} -> {s2}"] += 1

    print(f"[shadow] transitions:")
    for k, v in transitions.most_common():
        print(f"  {v:>4d}  {k}")

    summary = {
        "n": len(clues),
        "pass1": p1,
        "pass2": p2,
        "transitions": dict(transitions),
        "shadow_writes": counts,
        "per_clue": results,
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(summary, indent=2),
                              encoding="utf-8")
    print(f"\nWrote {args.out}")


if __name__ == "__main__":
    main()
