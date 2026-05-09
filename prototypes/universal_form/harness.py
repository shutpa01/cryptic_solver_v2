"""End-to-end simulation harness.

Pulls clues from data/clues_master.db, translates each row's flat components
to the universal form, runs both the new (form-based) verifier and the live
(prose-based) verifier, and produces a side-by-side comparison report.

Read-only: opens the live DBs but never writes.
"""
from __future__ import annotations

import argparse
import json
import sqlite3
import sys
import traceback
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))  # so we can import the live verifier

from prototypes.universal_form.translator import translate
from prototypes.universal_form.verifier import FormVerifier, Verdict
from prototypes.universal_form.renderer import render
from prototypes.universal_form.enrichment_sim import (
    collect_proposals, dedupe, counterfactual_verify,
)


CLUES_DB = str(PROJECT_ROOT / "data" / "clues_master.db")


@dataclass
class ClueResult:
    clue_id: int
    clue_number: str
    direction: str
    clue_text: str
    answer: str
    definition_text: Optional[str]
    wordplay_types: Optional[str]
    model_version: Optional[str]
    confidence: Optional[float]
    ai_explanation: Optional[str]
    translation_flag: str
    translation_notes: list
    new_verdict: Optional[str]   # PASS / FAIL / None
    new_checks: list             # full list of {check, status, detail}
    old_verdict: Optional[str]
    old_score: Optional[int]
    old_checks: list             # full list of {check, status, detail}
    rendered_explanation: Optional[str]
    # Production-cycle simulation
    proposals: list              # list of Proposal dicts
    counterfactual_verdict: Optional[str]   # PASS / FAIL / None
    counterfactual_checks: list


def _live_verifier_verdict(clue, ref_db_path: str) -> tuple[Optional[str],
                                                            Optional[int],
                                                            list]:
    """Run the LIVE prose-based verifier and return (verdict, score, checks).

    Errors fall through to (None, None, []) — the live verifier is
    sometimes fragile on legacy rows; we don't let it crash the whole
    simulation.
    """
    try:
        from sonnet_pipeline.verify_explanation import ExplanationVerifier
    except Exception:
        return None, None, []
    try:
        verifier = ExplanationVerifier(ref_db=ref_db_path)
        wt = clue.get("wordplay_types") or ""
        # Live verifier expects wordplay_type as a comma-separated string,
        # not a JSON list. Normalise:
        try:
            wt_parsed = json.loads(wt)
            if isinstance(wt_parsed, list):
                wt = ",".join(str(x) for x in wt_parsed)
        except Exception:
            pass
        result = verifier.verify(
            clue["clue_text"], clue["answer"], clue.get("definition_text"),
            wt, clue.get("ai_explanation"))
        return result["verdict"], result["score"], list(result.get("checks", []))
    except Exception:
        return None, None, []


def run_query(sql: str, params: tuple = ()) -> list[ClueResult]:
    conn = sqlite3.connect(CLUES_DB)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(sql, params).fetchall()
    out: list[ClueResult] = []
    new_verifier = FormVerifier()
    try:
        for r in rows:
            clue_dict = dict(r)
            translation = translate(
                r["components"], r["wordplay_types"],
                r["definition_text"], r["answer"] or "",
                r["ai_explanation"], clue_text=r["clue_text"])
            new_verdict_str: Optional[str] = None
            new_checks: list = []
            rendered: Optional[str] = None
            proposals_dicts: list = []
            cf_verdict: Optional[str] = None
            cf_checks: list = []
            if translation.form is not None:
                try:
                    v: Verdict = new_verifier.verify(
                        translation.form, r["clue_text"] or "")
                    new_verdict_str = v.verdict
                    new_checks = [c.to_dict() for c in v.checks]
                    rendered = render(translation.form,
                                      r["clue_text"] or "")
                    # Production cycle: collect proposals, dedupe via gate,
                    # then counterfactual re-verification.
                    if v.verdict == "FAIL":
                        proposals = dedupe(collect_proposals(v))
                        proposals_dicts = [p.to_dict() for p in proposals]
                        cf = counterfactual_verify(
                            translation.form, r["clue_text"] or "",
                            new_verifier, proposals)
                        cf_verdict = cf.verdict
                        cf_checks = [c.to_dict() for c in cf.checks]
                    else:
                        # No FAIL → no enrichments needed; counterfactual = same
                        cf_verdict = v.verdict
                        cf_checks = new_checks
                except Exception:
                    new_verdict_str = "ERR"
                    new_checks = [{"check": "error", "status": "fail",
                                   "detail": traceback.format_exc(limit=2)}]
            old_verdict, old_score, old_checks = _live_verifier_verdict(
                clue_dict, str(PROJECT_ROOT / "data" / "cryptic_new.db"))
            out.append(ClueResult(
                clue_id=r["id"],
                clue_number=r["clue_number"] or "",
                direction=r["direction"] or "",
                clue_text=r["clue_text"] or "",
                answer=r["answer"] or "",
                definition_text=r["definition_text"],
                wordplay_types=r["wordplay_types"],
                model_version=r["model_version"],
                confidence=r["confidence"],
                ai_explanation=r["ai_explanation"],
                translation_flag=translation.report.flag,
                translation_notes=translation.report.notes,
                new_verdict=new_verdict_str,
                new_checks=new_checks,
                old_verdict=old_verdict,
                old_score=old_score,
                old_checks=old_checks,
                rendered_explanation=rendered,
                proposals=proposals_dicts,
                counterfactual_verdict=cf_verdict,
                counterfactual_checks=cf_checks,
            ))
    finally:
        new_verifier.close()
        conn.close()
    return out


def query_for_puzzle(source: str, puzzle_number: str) -> str:
    """SQL for all clues+structured rows in a single puzzle."""
    return ("""
        SELECT c.id, c.clue_number, c.direction, c.clue_text, c.answer,
               c.ai_explanation,
               se.components, se.wordplay_types, se.definition_text,
               se.model_version, se.confidence
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE c.source = ? AND c.puzzle_number = ?
        ORDER BY c.direction, CAST(c.clue_number AS INTEGER)
    """, (source, puzzle_number))


def query_random_sample(limit: int = 50) -> tuple[str, tuple]:
    return ("""
        SELECT c.id, c.clue_number, c.direction, c.clue_text, c.answer,
               c.ai_explanation,
               se.components, se.wordplay_types, se.definition_text,
               se.model_version, se.confidence
        FROM clues c
        JOIN structured_explanations se ON se.clue_id = c.id
        WHERE se.components IS NOT NULL
        ORDER BY RANDOM() LIMIT ?
    """, (limit,))


# --- Reporting -------------------------------------------------------------

def print_clue_block(r: ClueResult, verbose: bool = False) -> None:
    head = (f"{r.clue_number:>3s}{r.direction[:1]}  "
            f"{r.answer:<14s}  "
            f"NEW=[{(r.new_verdict or '?'):<6s}]  "
            f"OLD=[{(r.old_verdict or '?'):<6s} "
            f"{r.old_score if r.old_score is not None else '-':>3}]  "
            f"flag={r.translation_flag:<22s}  "
            f"v={r.model_version}")
    print(head)
    print(f"     clue: {r.clue_text}")
    if verbose:
        if r.translation_notes:
            print(f"     translate notes: {r.translation_notes}")
        failures = [c for c in r.new_checks
                    if c.get("status") == "fail"]
        for f in failures[:6]:
            print(f"       - {f.get('check')}: {f.get('detail')}")
        if r.rendered_explanation:
            print("     RENDERED:")
            for line in r.rendered_explanation.split("\n"):
                print(f"        {line}")
        if r.ai_explanation:
            print(f"     ORIGINAL ai_explanation: "
                  f"{r.ai_explanation[:200]}")
        print()


def print_summary(results: list[ClueResult]) -> None:
    flag_counts: Counter = Counter(r.translation_flag for r in results)
    new_verdicts: Counter = Counter(r.new_verdict or "?" for r in results)
    old_verdicts: Counter = Counter(r.old_verdict or "?" for r in results)

    # Old verifier is graded; bucket as "old-pass" (HIGH) vs "old-fail"
    # (everything else) for a like-for-like comparison.
    def _old_bucket(v: Optional[str]) -> str:
        if v is None:
            return "?"
        return "old-pass" if v == "HIGH" else "old-fail"

    agreement: Counter = Counter()
    for r in results:
        if r.new_verdict in ("PASS", "FAIL") and r.old_verdict:
            new_p = (r.new_verdict == "PASS")
            old_p = (r.old_verdict == "HIGH")
            if new_p and old_p:
                agreement["both_pass"] += 1
            elif not new_p and not old_p:
                agreement["both_fail"] += 1
            elif new_p and not old_p:
                agreement["new_only"] += 1
            else:
                agreement["old_only"] += 1
        else:
            agreement["incomparable"] += 1

    print()
    print("=" * 78)
    print(f"  SUMMARY  ({len(results)} clues)")
    print("=" * 78)
    print(f"  Translation flags:     {dict(flag_counts)}")
    print(f"  NEW verdicts:          {dict(new_verdicts)}")
    print(f"  OLD verdicts:          {dict(old_verdicts)}  "
          f"(HIGH counts as 'pass')")
    print(f"  Comparison:            {dict(agreement)}")


# --- CLI -------------------------------------------------------------------

def main() -> None:
    ap = argparse.ArgumentParser(
        description=("Universal-form prototype harness — "
                     "translate, verify, compare against live verifier."))
    ap.add_argument("--source", help="puzzle source (e.g. telegraph)")
    ap.add_argument("--puzzle", help="puzzle number (e.g. 31229)")
    ap.add_argument("--random", type=int, default=0,
                    help="sample N random clues with components")
    ap.add_argument("--verbose", "-v", action="store_true")
    ap.add_argument("--max-clues", type=int, default=200,
                    help="maximum clues to process per run")
    ap.add_argument("--html", metavar="FILE",
                    help="write a self-contained HTML report to FILE "
                         "(opens in browser, no server needed)")
    ap.add_argument("--no-terminal", action="store_true",
                    help="suppress terminal per-clue output (useful with "
                         "--html when you only want the HTML)")
    args = ap.parse_args()

    if args.source and args.puzzle:
        sql, params = query_for_puzzle(args.source, args.puzzle)
        results = run_query(sql, params)
    elif args.random:
        sql, params = query_random_sample(args.random)
        results = run_query(sql, params)
    else:
        ap.error("provide either --source X --puzzle N, or --random N")

    if len(results) > args.max_clues:
        results = results[: args.max_clues]

    if not args.no_terminal:
        for r in results:
            print_clue_block(r, verbose=args.verbose)
        print_summary(results)

    if args.html:
        from prototypes.universal_form.report import write_html_report
        title_bits = []
        if args.source and args.puzzle:
            title_bits.append(f"{args.source} #{args.puzzle}")
        elif args.random:
            title_bits.append(f"random sample n={args.random}")
        title_bits.append(f"{len(results)} clues")
        write_html_report(results, args.html,
                          title="Universal form sim — " +
                                ", ".join(title_bits))
        print(f"\nHTML report written to: {args.html}")
        print("Open it in your browser — no server needed.")


if __name__ == "__main__":
    main()
