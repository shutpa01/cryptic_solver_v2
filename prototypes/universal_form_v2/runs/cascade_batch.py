"""Cascade batch runner — drives tree_matcher across cold clues and
writes results to shadow_db.

Parallel of seed_batch.py (which uses the JSON translator). Where
seed_batch.py converts blog-derived components JSON into a Form, this
runner has no input form: it asks the catalog to produce one. For each
clue with a known answer it calls cascade.solve_clue_parallel against
catalog_v1.json:

  - PASS    — write to shadow_db.solves with the verified form
  - PENDING — write to shadow_db.solves (e.g. &lit form awaiting
              human review)
  - FAIL    — write to shadow_db.seed_failures with the dedup'd
              enrichment candidates the failed attempts emitted

Skips clues already processed (any solves or seed_failures row).

Usage:
    python -m prototypes.universal_form_v2.runs.cascade_batch [N]

N defaults to 50.
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
from backfill_ai_exp.backfill_dd_hidden import build_graph as _build_dd_graph
from prototypes.universal_form_v2.cascade import solve_clue_parallel
from prototypes.universal_form_v2.shadow_db import (
    ensure_shadow, write_solve, write_seed_failure, is_clue_processed,
)


CATALOG_PATH = (PROJECT_ROOT / "prototypes" / "universal_form_v2"
                / "runs" / "catalog_v1.json")


def load_catalog() -> list:
    """Catalog entries sorted by frequency descending. cascade.py walks
    them in the order given."""
    with open(CATALOG_PATH, encoding="utf-8") as f:
        cat = json.load(f)
    return sorted(cat["entries"], key=lambda e: -e.get("frequency", 0))


def run_batch(n_clues: int) -> dict:
    db = RefDB(str(PROJECT_ROOT / "data" / "cryptic_new.db"))
    master = sqlite3.connect(str(PROJECT_ROOT / "data" / "clues_master.db"))
    master.row_factory = sqlite3.Row
    shadow = ensure_shadow()

    catalog = load_catalog()
    print(f"Building dd_graph for hidden / DD detectors...")
    dd_graph = _build_dd_graph(db)

    # Over-fetch — some pulls will be already-processed.
    rows = master.execute(
        """
        SELECT c.id AS clue_id, c.source, c.puzzle_number, c.clue_number,
               c.direction, c.clue_text, c.answer
        FROM clues c
        WHERE c.answer IS NOT NULL AND c.answer != ''
          AND c.clue_text IS NOT NULL AND c.clue_text != ''
          AND c.source NOT IN ('telegraph-toughie', 'cordelia')
        ORDER BY RANDOM()
        LIMIT ?
        """,
        (n_clues * 3,),
    ).fetchall()

    stats = Counter()
    sig_counts = Counter()       # signature -> PASS count this batch
    pending_sigs = Counter()     # signature -> PENDING count this batch
    enrich_counter = Counter()   # kind -> total enrichment candidates emitted
    processed = 0

    for r in rows:
        if processed >= n_clues:
            break
        if is_clue_processed(shadow, r["clue_id"]):
            stats["skipped_already_processed"] += 1
            continue

        meta = {
            "source": r["source"], "puzzle_number": r["puzzle_number"],
            "clue_number": r["clue_number"], "direction": r["direction"],
            "clue_text": r["clue_text"], "answer": r["answer"],
        }

        result = solve_clue_parallel(
            clue_id=r["clue_id"],
            clue_text=r["clue_text"],
            answer=r["answer"],
            ref_db=db,
            dd_graph=dd_graph,
            catalog_entries=catalog,
            shadow_conn=shadow,
        )

        if result.verdict == "PASS":
            stats["pass"] += 1
            sig_counts[result.signature] += 1
            write_solve(
                shadow, clue_id=r["clue_id"],
                signature=result.signature, verdict="PASS",
                answer=r["answer"], form_dict=result.form.to_dict())
        elif result.verdict == "PENDING":
            stats["pending"] += 1
            pending_sigs[result.signature] += 1
            write_solve(
                shadow, clue_id=r["clue_id"],
                signature=result.signature, verdict="PENDING",
                answer=r["answer"], form_dict=result.form.to_dict())
        else:
            stats["fail"] += 1
            enrichments = [c.to_dict() for c in result.enrichment_candidates]
            for c in result.enrichment_candidates:
                enrich_counter[c.kind] += 1
            fail_detail = (f"no template PASSed; "
                           f"{len(enrichments)} enrichment candidate(s)")
            write_seed_failure(
                shadow, clue_id=r["clue_id"],
                seed_source="cascade",
                failure_kind="cascade_fail",
                failure_detail=fail_detail,
                clue_meta=meta,
                enrichments=enrichments)

        processed += 1

    return {
        "processed": processed,
        "stats": stats,
        "sig_counts": sig_counts,
        "pending_sigs": pending_sigs,
        "enrich_counter": enrich_counter,
    }


def report(result: dict) -> str:
    lines = []
    lines.append("# Cascade batch report\n")
    lines.append("## Summary")
    lines.append(f"- Processed: **{result['processed']}**")
    s = result["stats"]
    lines.append(f"- PASS:    **{s['pass']}**")
    lines.append(f"- PENDING: {s['pending']}")
    lines.append(f"- FAIL:    {s['fail']}")
    if s.get("skipped_already_processed"):
        lines.append(f"- Skipped (already processed): "
                     f"{s['skipped_already_processed']}")
    lines.append("")
    if result["sig_counts"]:
        lines.append("## PASS signatures (top 10)")
        lines.append("| signature | count |")
        lines.append("|---|---|")
        for sig, n in result["sig_counts"].most_common(10):
            lines.append(f"| `{sig}` | {n} |")
        lines.append("")
    if result["pending_sigs"]:
        lines.append("## PENDING signatures (top 10)")
        lines.append("| signature | count |")
        lines.append("|---|---|")
        for sig, n in result["pending_sigs"].most_common(10):
            lines.append(f"| `{sig}` | {n} |")
        lines.append("")
    if result["enrich_counter"]:
        lines.append("## Enrichment candidates from FAILs")
        lines.append("| kind | total emitted |")
        lines.append("|---|---|")
        for kind, n in result["enrich_counter"].most_common():
            lines.append(f"| {kind} | {n} |")
    return "\n".join(lines)


def main():
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 50
    print(f"Running cascade batch of {n} clues...")
    result = run_batch(n)
    print()
    print(report(result))


if __name__ == "__main__":
    main()
