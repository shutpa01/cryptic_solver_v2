"""worklist_probe — a READ-ONLY window onto the live cascade.

Purpose (2026-07-20): ground every engine_worklist decision in what the REAL
production cascade actually does with a clue, not in a model of it. Nothing here
writes to a DB or deploys anything.

Read-only guarantees (all three cascade write-paths are closed):
  1. clue_id=None passed to solve_clue_text  -> store.persist() never runs
     (engine_registry.py:1747 gates persist on clue_id is not None).
  2. wiring["store"] set to None            -> _finalize_provisional() early-returns
     (engine_registry.py:1756), so NO pending_enrichments / indicator / piece / dd
     queueing. make_db_wiring() otherwise sets store=PendingStore() which WOULD write.
  3. auto_signature never set in the wiring  -> the auto-discover/auto-file branches
     (engine_registry.py:1739,1779) are skipped, so no catalog signature is filed.
  db_only() additionally nulls every AI touch-point, so no AI/billable call is made.

Usage:
    python -m scripts.worklist_probe 1710874 10077940      # probe specific clues
    python -m scripts.worklist_probe --selfcheck 200        # harness-fidelity check
    python -m scripts.worklist_probe --run 111 222 333      # terse status line per clue

The selfcheck re-runs a sample of clues that wfw_solve currently records as PASS and
reports how many the live cascade still passes right now. With no code change in flight
it should be ~100%; the same run_set() diffed before/after a proposed change is the
zero-regression gate.
"""

import argparse
import random
import sqlite3
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"


def get_wiring():
    """A READ-ONLY view of the live DB wiring. See module docstring, guarantee #2."""
    from core import engine_registry
    w = engine_registry.db_only(engine_registry.make_db_wiring())
    w["store"] = None                      # close the enrichment-queue write path
    w.pop("auto_signature", None)          # belt-and-braces: never file signatures
    w.pop("auto_signature_queue", None)
    return w


def _connect():
    con = sqlite3.connect(f"file:{CLUES_DB}?mode=ro", uri=True)
    con.row_factory = sqlite3.Row
    return con


def fetch_clue(con, clue_id):
    return con.execute(
        "SELECT id, source, clue_text, enumeration, answer, direction "
        "FROM clues WHERE id=?", (clue_id,)).fetchone()


def solve(clue_id, wiring, con=None):
    """Run one clue through the live cascade, read-only. Returns a record dict."""
    from core.engine_registry import solve_clue_text
    from core.wfw_web import enum_space
    own = con is None
    if own:
        con = _connect()
    try:
        r = fetch_clue(con, clue_id)
    finally:
        if own:
            con.close()
    if r is None:
        return {"clue_id": clue_id, "error": "clue id not found"}
    if not r["answer"]:
        return {"clue_id": clue_id, "error": "clue has no answer", "clue_text": r["clue_text"]}
    spaced = enum_space(r["answer"], r["enumeration"])
    ctx, parse, name = solve_clue_text(
        r["clue_text"], spaced, wiring,
        source=r["source"], clue_id=None, direction=r["direction"])
    rec = {
        "clue_id": clue_id,
        "source": r["source"],
        "clue_text": r["clue_text"],
        "answer": r["answer"],
        "engine": name,
        "status": None if parse is None else parse.status,
        "operation": None if parse is None else parse.operation,
        "solved_by": None if parse is None else parse.solved_by,
        "definition": None if (parse is None or not parse.definition) else parse.definition.text,
        "sources": [] if parse is None else [
            {"text": s.text, "value": s.value, "mechanism": s.mechanism} for s in parse.sources],
        "warnings": [] if parse is None else list(parse.warnings),
    }
    return rec


def probe(ids):
    """Human-readable dump of the true cascade result for each clue."""
    wiring = get_wiring()
    con = _connect()
    try:
        for cid in ids:
            rec = solve(cid, wiring, con)
            print("=" * 88)
            if rec.get("error"):
                print(f"[{cid}] ERROR: {rec['error']}")
                continue
            print(f"[{cid}] {rec['source']}   ANSWER = {rec['answer']}")
            print(f"   clue: {rec['clue_text']}")
            print(f"   -> engine={rec['engine']}  status={rec['status']}  operation={rec['operation']}")
            print(f"   definition: {rec['definition']}")
            for s in rec["sources"]:
                print(f"     piece: {s['text']!r} -> {s['value']}  ({s['mechanism']})")
            if rec["warnings"]:
                for wn in rec["warnings"]:
                    print(f"     warning: {wn}")
    finally:
        con.close()


def run_set(ids, wiring=None):
    """Terse {clue_id: (status, engine, operation)} for diffing two cascade runs."""
    wiring = wiring or get_wiring()
    con = _connect()
    out = {}
    try:
        for cid in ids:
            rec = solve(cid, wiring, con)
            out[cid] = (rec.get("status"), rec.get("engine"), rec.get("operation"))
    finally:
        con.close()
    return out


def current_pass_ids(limit=None):
    """clue_ids that wfw_solve currently records as PASS (the regression baseline)."""
    con = _connect()
    try:
        q = "SELECT clue_id FROM wfw_solve WHERE status='pass' ORDER BY clue_id"
        rows = [r[0] for r in con.execute(q)]
    finally:
        con.close()
    return rows[:limit] if limit else rows


def selfcheck(n):
    """Re-run a random sample of currently-passing clues; report how many still pass.
    Deterministic sample (fixed seed) so the same clues are used across runs."""
    ids = current_pass_ids()
    random.seed(20260720)
    sample = sorted(random.sample(ids, min(n, len(ids))))
    print(f"harness-fidelity selfcheck: {len(sample)} clues that wfw_solve marks PASS")
    res = run_set(sample)
    still = sum(1 for v in res.values() if v[0] == "pass")
    other = [cid for cid, v in res.items() if v[0] != "pass"]
    print(f"  live cascade still PASS: {still}/{len(sample)}")
    if other:
        print(f"  NOT pass now ({len(other)}): showing up to 15")
        wiring = get_wiring()
        con = _connect()
        try:
            for cid in other[:15]:
                rec = solve(cid, wiring, con)
                print(f"    [{cid}] now status={rec.get('status')} engine={rec.get('engine')}"
                      f"  {rec.get('clue_text')!r} = {rec.get('answer')}")
        finally:
            con.close()
    return res


def main(argv=None):
    ap = argparse.ArgumentParser(description="Read-only cascade probe for the engine worklist")
    ap.add_argument("ids", nargs="*", type=int, help="clue ids to probe (human-readable)")
    ap.add_argument("--selfcheck", type=int, metavar="N", help="re-run N currently-passing clues")
    ap.add_argument("--run", nargs="+", type=int, metavar="ID", help="terse status line per clue")
    args = ap.parse_args(argv)

    if args.selfcheck:
        selfcheck(args.selfcheck)
    elif args.run:
        for cid, v in run_set(args.run).items():
            print(f"[{cid}] status={v[0]} engine={v[1]} operation={v[2]}")
    elif args.ids:
        probe(args.ids)
    else:
        ap.print_help()
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
