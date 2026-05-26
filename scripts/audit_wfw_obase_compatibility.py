"""Compare fresh WFW-native proof coverage against stored obase results.

This is a structural integrity audit.  It does not write to the database and
does not treat any clue as a one-off target.  The contract is simple: WFW must
be a superset of obase, so every stored obase solve that does not fresh-prove
through WFW is a compatibility gap to investigate by capability class.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
import sqlite3
import sys
from collections import Counter, defaultdict
from pathlib import Path


ROOT = Path(__file__).resolve().parent.parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from signature_solver.db import RefDB
from signature_solver.wfw_unified_solver import solve_wfw_unified
from signature_solver.wfw_unified_proof import build_wfw_proof_from_unified_result


CLUES_DB = ROOT / "data" / "clues_master.db"


def audit(source, puzzle, db_path=CLUES_DB):
    conn = sqlite3.connect(str(db_path), timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT c.id, c.clue_number, c.direction, c.clue_text,
                      c.answer, c.has_solution, c.wordplay_type,
                      c.definition, c.ai_explanation, se.components,
                      se.model_version, se.confidence AS structured_confidence
               FROM clues c
               LEFT JOIN structured_explanations se ON se.clue_id = c.id
               WHERE c.source = ? AND c.puzzle_number = ?
               ORDER BY
                 CASE WHEN c.direction = 'across' THEN 0 ELSE 1 END,
                 CAST(c.clue_number AS INTEGER),
                 c.clue_number""",
            (source, str(puzzle)),
        ).fetchall()
    finally:
        conn.close()

    with contextlib.redirect_stdout(io.StringIO()):
        ref_db = RefDB()

    findings = []
    for row in rows:
        obase_solved = bool(row["has_solution"])
        res = solve_wfw_unified(row["clue_text"], row["answer"], db=ref_db)
        proof = build_wfw_proof_from_unified_result(res)
        token_parse = proof.get("token_parse") if proof else None
        fresh_status = proof.get("status") if proof else "none"
        operation = token_parse.get("operation") if token_parse else ""
        confidence = token_parse.get("confidence") if token_parse else ""
        finding = {
            "clue_id": row["id"],
            "label": "%s%s" % (
                row["clue_number"],
                (row["direction"] or "?")[0].lower()),
            "clue_text": row["clue_text"],
            "answer": row["answer"],
            "obase_solved": obase_solved,
            "obase_type": row["wordplay_type"] or "",
            "wfw_status": fresh_status,
            "wfw_operation": operation,
            "wfw_confidence": confidence,
        }
        if row["model_version"] == "manual_approve":
            finding["compatibility"] = "covered"
        elif obase_solved and fresh_status != "wfw_proven":
            objections = tuple(proof.get("objections") or ()) if proof else ()
            if _looks_like_unsafe_obase(row) or _proof_rejects_unsafe_parse(objections):
                finding["compatibility"] = "obase_unsafe"
            elif token_parse and objections:
                finding["compatibility"] = "mechanical_review"
            elif confidence in (
                    "mechanically_verified_inferred_definition",
                    "mechanically_verified_surface_gaps",
                    "mechanically_verified_definition_gaps"):
                finding["compatibility"] = "mechanical_review"
            else:
                finding["compatibility"] = "regression"
        elif obase_solved:
            finding["compatibility"] = "covered"
        elif fresh_status == "wfw_proven":
            finding["compatibility"] = "wfw_extra"
        else:
            finding["compatibility"] = "both_unsolved"
        findings.append(finding)
    return findings


def print_report(findings, source, puzzle):
    counts = Counter(item["compatibility"] for item in findings)
    by_type = defaultdict(Counter)
    for item in findings:
        by_type[item["obase_type"] or "unsolved"][item["compatibility"]] += 1

    print("WFW/obase compatibility audit: %s #%s" % (source, puzzle))
    for key in ("covered", "mechanical_review", "obase_unsafe",
                "regression", "wfw_extra", "both_unsolved"):
        print("%s: %d" % (key, counts[key]))
    print("")
    print("By stored obase type:")
    for typ in sorted(by_type):
        bits = ", ".join(
            "%s=%d" % (k, v)
            for k, v in sorted(by_type[typ].items())
        )
        print("  %s: %s" % (typ, bits))
    print("")
    print("Regressions:")
    for item in findings:
        if item["compatibility"] != "regression":
            continue
        print(
            "  {label} {answer}: obase={obase_type} | WFW={wfw_status} "
            "| {clue_text}".format(**item)
        )
    print("")
    print("Obase solved but WFW rejected as unsafe/under-evidenced:")
    for item in findings:
        if item["compatibility"] != "obase_unsafe":
            continue
        print(
            "  {label} {answer}: obase={obase_type} | {clue_text}".format(
                **item)
        )
    print("")
    print("Mechanical WFW assemblies still gated for missing WFW evidence:")
    for item in findings:
        if item["compatibility"] != "mechanical_review":
            continue
        print(
            "  {label} {answer}: obase={obase_type} | WFW={wfw_operation} "
            "| {clue_text}".format(**item)
        )


def _looks_like_unsafe_obase(row):
    wordplay_type = (row["wordplay_type"] or "").lower()
    definition = (row["definition"] or "").strip().lower()
    explanation = (row["ai_explanation"] or "").strip().lower()
    try:
        confidence = float(row["structured_confidence"])
    except (TypeError, ValueError):
        confidence = None
    if confidence is not None and confidence <= 0:
        return True
    if confidence is not None and confidence < 0.2:
        return True
    try:
        components = json.loads(row["components"] or "{}")
    except (TypeError, ValueError):
        components = {}
    if not components:
        return True
    assembly = components.get("assembly") or {}
    pieces = components.get("ai_pieces") or []
    if wordplay_type in ("", "unknown", "unparsed", "cryptic_definition"):
        return True
    if components.get("source") == "claude_review" and not pieces:
        return True
    if assembly.get("op") == "charade" and not assembly.get("order") and not pieces:
        return True
    if wordplay_type == "double_definition" and (
            not definition or definition == "double definition"):
        return True
    if wordplay_type == "double_definition":
        if not (assembly.get("left_def") and assembly.get("right_def")):
            if len(pieces) < 2:
                return True
    if not definition and wordplay_type in ("deletion", "charade"):
        return True
    if "definition: none" in explanation:
        return True
    for piece in pieces:
        mechanism = (piece.get("mechanism") or "").lower()
        letters = _clean_letters(piece.get("letters") or "")
        if mechanism in ("last_letter", "first_letter") and len(letters) != 1:
            return True
        if mechanism == "indicator" and letters:
            return True
    return False


def _proof_rejects_unsafe_parse(objections):
    return any(
        objection.startswith("v4_source_definition_overlap:")
        for objection in objections
    )


def _clean_letters(value):
    return "".join(char.upper() for char in (value or "") if char.isalpha())


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--puzzle", required=True)
    args = parser.parse_args(argv)
    findings = audit(args.source, args.puzzle)
    print_report(findings, args.source, args.puzzle)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
