"""Run the live additive atomic parser path for one puzzle.

This is the production prototype path: it appends one atomic artifact per
clue, plus review items for clues that do not yet have a complete WFW parse.
It does not update legacy clue rows, structured explanations, scores, or
manual verification state.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
from collections import Counter
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from signature_solver.atomic_parse_store import (  # noqa: E402
    artifact_from_solve_result,
    finish_atomic_run,
    start_atomic_run,
    write_atomic_artifact,
    write_review_item,
)
from signature_solver.clue_context import build_clue_context  # noqa: E402
from signature_solver.db import RefDB  # noqa: E402
from signature_solver.solver import solve_clue  # noqa: E402
from signature_solver.token_parse_assembler import assemble_token_parses  # noqa: E402
from signature_solver.wfw_formatter import format_token_parse_for_wfw  # noqa: E402


CLUES_DB = ROOT / "data" / "clues_master.db"
SOLVER_VERSION = "atomic_prototype_live:v1"


def main():
    parser = argparse.ArgumentParser(
        description="Append live atomic parse artifacts for a puzzle.")
    parser.add_argument("--source", default="telegraph")
    parser.add_argument("--puzzle-number", required=True)
    parser.add_argument(
        "--json", action="store_true",
        help="Emit a JSON summary instead of the compact text report.")
    args = parser.parse_args()

    summary = run_atomic_parse(args.source, args.puzzle_number)
    if args.json:
        print(json.dumps(summary, indent=2, sort_keys=True))
    else:
        _print_summary(summary)
    return 0


def run_atomic_parse(source, puzzle_number):
    """Append atomic artifacts for one puzzle and return a run summary."""
    rows = _fetch_puzzle(source, puzzle_number)
    if not rows:
        raise ValueError("No clues found for %s %s" % (source, puzzle_number))

    db = RefDB()
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    run_id = None
    results = []
    try:
        run_id = start_atomic_run(
            source, puzzle_number, len(rows), conn=conn,
            solver_version=SOLVER_VERSION)
        for row in rows:
            result = _process_row(row, db, conn, run_id)
            results.append(result)
        complete = sum(1 for result in results if result["complete"])
        review = len(results) - complete
        finish_atomic_run(run_id, complete, review, conn=conn)
        conn.commit()
    except Exception:
        conn.rollback()
        if run_id is not None:
            try:
                finish_atomic_run(run_id, 0, 0, conn=conn, status="failed")
                conn.commit()
            except Exception:
                conn.rollback()
        raise
    finally:
        conn.close()

    summary = {
        "run_id": run_id,
        "source": source,
        "puzzle_number": puzzle_number,
        "total": len(results),
        "complete": sum(1 for result in results if result["complete"]),
        "review": sum(1 for result in results if not result["complete"]),
        "operations": dict(Counter(
            result["operation"] or "(none)" for result in results)),
        "review_types": dict(Counter(
            result["review_type"] for result in results
            if result["review_type"])),
        "clues": results,
    }
    return summary


def _process_row(row, db, conn, run_id):
    sr = solve_clue(row["clue_text"], row["answer"], db)
    _ensure_atomic_parse(sr, row["clue_text"], row["answer"], db)
    artifact = artifact_from_solve_result(sr, solver_version=SOLVER_VERSION)
    artifact_id = write_atomic_artifact(
        row["id"], row["clue_text"], row["answer"], artifact, conn=conn)

    complete = bool(artifact.get("wfw"))
    operation = ""
    token_parses = artifact.get("token_parses") or []
    if token_parses:
        operation = token_parses[0].get("operation") or ""

    review_type = ""
    review_id = None
    if not complete:
        review_type, summary, payload = _review_payload(row, sr, artifact)
        review_id = write_review_item(
            run_id, artifact_id, row["id"], review_type, summary, payload,
            conn=conn)

    return {
        "clue_id": row["id"],
        "artifact_id": artifact_id,
        "review_id": review_id,
        "ref": "%s%s" % (
            row["clue_number"], (row["direction"] or "?")[0].upper()),
        "answer": row["answer"],
        "complete": complete,
        "operation": operation,
        "review_type": review_type,
        "clue": row["clue_text"],
    }


def _ensure_atomic_parse(sr, clue_text, answer, db):
    if sr is None or getattr(sr, "token_parses", None):
        return
    context = getattr(sr, "clue_context", None)
    if context is None:
        context = build_clue_context(clue_text, answer, db, annotate=True)
        sr.clue_context = context
    token_parses = assemble_token_parses(context)
    if not token_parses:
        return
    sr.token_parses = token_parses
    sr.wfw_token_parses = [
        format_token_parse_for_wfw(context, token_parse)
        for token_parse in token_parses
    ]


def _review_payload(row, sr, artifact):
    context = getattr(sr, "clue_context", None)
    reason = _classify_gap(row, sr, context)
    annotations = artifact.get("annotations") or []
    definition_candidates = []
    pos_model_status = None
    tokens = []
    if context is not None:
        definition_candidates = [
            candidate.as_dict()
            for candidate in context.definition_candidates
        ]
        pos_model_status = context.pos_model_status
        tokens = [token.as_dict() for token in context.tokens]

    payload = {
        "reason": reason,
        "solver_confidence": getattr(sr, "confidence", None),
        "solver_high": bool(getattr(sr, "high_confidence", False)),
        "definition_candidates": definition_candidates,
        "pos_model_status": pos_model_status,
        "tokens": tokens,
        "annotation_summary": _annotation_summary(annotations),
        "gt2_bundle_count": len(artifact.get("gt2_bundles") or []),
        "token_parse_count": len(artifact.get("token_parses") or []),
        "wfw_count": len(artifact.get("wfw") or []),
    }
    summary = "%s: %s needs review before atomic WFW can publish" % (
        row["answer"], reason)
    return reason, summary, payload


def _annotation_summary(annotations):
    counts = Counter(annotation.get("token") for annotation in annotations)
    examples = {}
    for annotation in annotations:
        token = annotation.get("token")
        if token in examples:
            continue
        examples[token] = {
            "text": annotation.get("text"),
            "values": (annotation.get("values") or [])[:5],
            "source": annotation.get("source"),
        }
    return {
        "counts": dict(counts),
        "examples": examples,
    }


def _classify_gap(row, solve_result, context):
    if context is None:
        return "no_context"
    if not context.definition_candidates:
        return "missing_definition_span"
    if not getattr(solve_result, "result", None):
        return "solver_no_candidate"
    if getattr(solve_result, "high_confidence", False):
        return "old_solved_no_atomic_parse"
    annotations = context.annotations
    if not any(ann.token.endswith("_I") or ann.token.startswith("POS_I_")
               for ann in annotations):
        return "missing_indicator_or_whole_clue_type"
    if not any(ann.token in ("SYN_F", "ABR_F", "HOM_F", "POS_F")
               for ann in annotations):
        return "missing_source_value"
    return "assembly_or_coverage_rejected"


def _fetch_puzzle(source, puzzle_number):
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        return list(conn.execute(
            """SELECT id, clue_number, direction, clue_text, answer
               FROM clues
               WHERE source = ? AND puzzle_number = ?
               ORDER BY direction, CAST(clue_number AS INTEGER), clue_number""",
            (source, puzzle_number),
        ))
    finally:
        conn.close()


def _print_summary(summary):
    print("Atomic live run: %s %s (run_id=%s)" % (
        summary["source"], summary["puzzle_number"], summary["run_id"]))
    print("Complete WFW: %d/%d" % (summary["complete"], summary["total"]))
    print("Review queue: %d/%d" % (summary["review"], summary["total"]))
    print()
    print("Review types:")
    for review_type, count in sorted(summary["review_types"].items()):
        print("  %-32s %2d" % (review_type, count))
    print()
    print("%-4s %-16s %-8s %-24s %s" % (
        "Ref", "Answer", "Status", "Operation/Review", "Clue"))
    for clue in summary["clues"]:
        status = "WFW" if clue["complete"] else "REVIEW"
        op = clue["operation"] or clue["review_type"] or "-"
        print("%-4s %-16s %-8s %-24s %s" % (
            clue["ref"], clue["answer"][:16], status, op, clue["clue"]))


if __name__ == "__main__":
    raise SystemExit(main())
