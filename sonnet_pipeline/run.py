"""Sonnet pipeline orchestrator — single entry point.

Usage:
    python -m sonnet_pipeline.run 29939
    python -m sonnet_pipeline.run 29939 29926 --source guardian
    python -m sonnet_pipeline.run 29939 --write-db
    python -m sonnet_pipeline.run 29939 --output-dir reports/
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time

# Allow running as both `python sonnet_pipeline/run.py` and `python -m sonnet_pipeline.run`
if __name__ == "__main__" and __package__ is None:
    sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    __package__ = "sonnet_pipeline"

from .enricher import ClueEnricher
from .solver import (
    HomophoneEngine, build_example_messages, clean,
    resolve_cross_references, solve_clue, store_result,
)
from .report import generate_report, _describe_assembly

if sys.stdout.encoding and sys.stdout.encoding.lower() != "utf-8":
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")

# ============================================================
# USER CONFIGURATION
# ============================================================
CLUES_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db"
CRYPTIC_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\cryptic_new.db"
OUTPUT_DIR = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\documents"

# ============================================================
# RUN CRITERIA (edit these or override via CLI args)
# ============================================================
SOURCE = "guardian"            # telegraph, guardian, times, independentclaude
PUZZLE_NUMBER = "29935"             # puzzle number to solve
WRITE_DB = False                # write results to clues_master.db
FORCE_API = False              # True = fresh API calls for all clues (ignore cached)
SINGLE_CLUE_MATCH = ""




def run_puzzle(source, puzzle, enricher, homo_engine, example_messages,
               write_db=False, output_dir=OUTPUT_DIR, single_clue="", force=False):
    """Run the Sonnet pipeline on a single puzzle. Returns (results, stats)."""
    db_path = CLUES_DB
    conn = sqlite3.connect(db_path)
    rows = conn.execute("""
        SELECT id, clue_number, direction, clue_text, answer, enumeration, explanation
        FROM clues WHERE source = ? AND puzzle_number = ?
        ORDER BY clue_number
    """, (source, puzzle)).fetchall()

    # Build cross-reference map from full puzzle (before filtering)
    puzzle_answers = {}
    for _, cnum, _, _, answer, _, _ in rows:
        if answer:
            num_only = re.sub(r'[^0-9]', '', str(cnum))
            puzzle_answers[num_only] = answer

    if single_clue:
        rows = [r for r in rows if single_clue.lower() in r[3].lower()]

    if not rows:
        conn.close()
        print("No clues found for %s puzzle %s" % (source, puzzle))
        return [], {}

    print("=" * 80)
    print("SONNET PIPELINE: Sonnet -> Assembler -> Fallback")
    print("Puzzle: %s %s (%d clues)" % (source, puzzle, len(rows)))
    print("=" * 80)

    sonnet_tokens_in = 0
    sonnet_tokens_out = 0
    results = []

    for row in rows:
        cid, cnum, direction, clue, answer, enum, explanation = row
        target = clean(answer)

        # Check for existing solved result — skip API call but re-run assembler
        cached_ai = None
        if not force:
            existing = conn.execute("""
                SELECT se.confidence, se.wordplay_types, se.components, se.definition_text
                FROM structured_explanations se
                JOIN clues c ON se.clue_id = c.id
                WHERE se.clue_id = ? AND c.has_solution = 1
            """, (cid,)).fetchone()
            if existing:
                solved_conf, solved_wptypes, solved_comps, solved_def = existing
                try:
                    wp_list = json.loads(solved_wptypes) if solved_wptypes else []
                except (json.JSONDecodeError, TypeError):
                    wp_list = []
                try:
                    comps = json.loads(solved_comps) if solved_comps else {}
                except (json.JSONDecodeError, TypeError):
                    comps = {}

                # Remap stored piece keys to standard format if needed
                raw_pieces = comps.get("ai_pieces", []) if isinstance(comps, dict) else []
                for p in raw_pieces:
                    if "fodder" in p and "clue_word" not in p:
                        p["clue_word"] = p.pop("fodder")
                    if "yields" in p and "letters" not in p:
                        p["letters"] = p.pop("yields")
                    if "type" in p and "mechanism" not in p:
                        p["mechanism"] = p.pop("type")
                cached_ai = {
                    "definition": solved_def,
                    "wordplay_type": wp_list[0] if wp_list else "unknown",
                    "pieces": raw_pieces,
                }

        # Resolve cross-references
        xrefs = resolve_cross_references(clue, puzzle_answers)
        xref_note = ""
        if xrefs:
            xref_note = "\nCross-references:\n"
            for num, ref_answer in xrefs.items():
                xref_note += "  Clue %s answer = %s\n" % (num, ref_answer)

        # Enrichment
        enrichment = enricher.enrich(clue, answer)
        if xref_note:
            enrichment += xref_note

        # Solve (skip API call if we have cached AI output from prior solve)
        try:
            result = solve_clue(
                clue, answer, enrichment, enricher, homo_engine,
                example_messages, cached_ai=cached_ai
            )
        except Exception as e:
            print("\n%s. %s = %s" % (cnum, clue, answer))
            print("   SONNET ERROR: %s" % e)
            results.append({
                "status": "error", "tier": None,
                "clue_number": cnum, "direction": direction,
                "enumeration": enum, "clue": clue, "answer": answer,
                "explanation": explanation,
            })
            continue

        sonnet_tokens_in += result["tokens_in"]
        sonnet_tokens_out += result["tokens_out"]

        assembly = result["assembly"]
        tier = result["tier"]
        if cached_ai and tier:
            tier = "Cached+" + tier  # e.g. "Cached+Sonnet", "Cached+Fallback"
        validation = result["validation"]
        sonnet_out = result["ai_output"]
        sonnet_pieces = result["sonnet_pieces"]
        sonnet_wtype = result["sonnet_wtype"]
        sonnet_def = result["sonnet_def"]
        fallback_method = result["fallback_method"]

        # Print progress
        print("\n%s. %s = %s" % (cnum, clue, answer))
        if xrefs:
            print("   Cross-refs: %s" % xrefs)
        if cached_ai:
            print("   [CACHED] re-running assembler on stored AI output")
        print("   Sonnet: type=%s, def=%s" % (sonnet_wtype, repr(sonnet_def)))
        print("   Pieces: %s -> %s (target=%s)" % (
            sonnet_pieces, "".join(sonnet_pieces), target))
        if assembly:
            desc = _describe_assembly(assembly, sonnet_out.get("pieces", []) if sonnet_out else [])
            if desc:
                print("   Assembly: %s — %s" % (assembly.get("op", "?"), desc))
            else:
                print("   Assembly: %s" % assembly)
            if fallback_method and fallback_method not in ("direct", None):
                print("   Fallback: %s" % fallback_method)
            print("   Confidence: %s (%d/100) %s" % (
                validation["confidence"].upper(), validation["score"],
                " | ".join("%s=%s" % (k, v) for k, v in validation["checks"].items())))
            print("   Status: ASSEMBLED (%s)" % tier)
            if write_db:
                store_result(conn, cid, sonnet_out, assembly, validation, tier)
        else:
            print("   Confidence: NONE (0/100)")
            print("   Status: FAILED")
            if write_db:
                conn.execute("UPDATE clues SET has_solution = 0 WHERE id = ?", (cid,))
        if explanation:
            print("   Human:  %s" % explanation[:90])

        results.append({
            "status": "ASSEMBLED" if assembly else "FAILED",
            "tier": tier,
            "clue_number": cnum,
            "direction": direction,
            "enumeration": enum,
            "clue": clue,
            "answer": answer,
            "explanation": explanation,
            "enrichment": enrichment,
            "confidence": validation.get("confidence", "none") if assembly else "none",
            "score": validation.get("score", 0) if assembly else 0,
            "checks": validation.get("checks", {}),
            "ai_output": sonnet_out,
            "assembly": assembly,
        })

        time.sleep(0.2)

    # Commit DB writes
    if write_db:
        conn.commit()
        print("\nDB writes committed to %s" % db_path)
    conn.close()

    # Compute stats
    total = len(results)
    assembled = sum(1 for r in results if r.get("status") == "ASSEMBLED")
    failed = sum(1 for r in results if r.get("status") == "FAILED")
    errors = sum(1 for r in results if r.get("status") == "error")
    t1 = sum(1 for r in results if r.get("tier") in ("Sonnet", "Cached+Sonnet"))
    t3 = sum(1 for r in results if r.get("tier") in ("Fallback", "Cached+Fallback"))
    t_cached = sum(1 for r in results if (r.get("tier") or "").startswith("Cached+"))

    assembled_results = [r for r in results if r.get("status") == "ASSEMBLED"]
    high = sum(1 for r in assembled_results if r.get("confidence") == "high")
    medium = sum(1 for r in assembled_results if r.get("confidence") == "medium")
    low = sum(1 for r in assembled_results if r.get("confidence") == "low")
    avg_score = sum(r.get("score", 0) for r in assembled_results) / max(len(assembled_results), 1)

    sonnet_cost = sonnet_tokens_in / 1e6 * 3.0 + sonnet_tokens_out / 1e6 * 15.0

    stats = {
        "total": total, "assembled": assembled, "failed": failed, "errors": errors,
        "sonnet": t1, "fallback": t3, "cached": t_cached,
        "high": high, "medium": medium, "low": low, "avg_score": avg_score,
        "total_cost": sonnet_cost,
        "sonnet_tokens_in": sonnet_tokens_in,
        "sonnet_tokens_out": sonnet_tokens_out,
    }

    # Print summary
    print("\n" + "=" * 80)
    print("SONNET PIPELINE SUMMARY")
    print("=" * 80)
    print("Total clues:      %d" % total)
    print("ASSEMBLED:        %d/%d (%d%%)" % (assembled, total, 100 * assembled // max(total, 1)))
    print("  Sonnet:         %d" % t1)
    print("  DB Fallback:    %d" % t3)
    if t_cached:
        print("  Cached (no API): %d" % t_cached)
    print("FAILED:           %d/%d (%d%%)" % (failed, total, 100 * failed // max(total, 1)))
    if errors:
        print("Errors:           %d" % errors)
    print("\nConfidence breakdown:")
    print("  High:   %d/%d" % (high, assembled))
    print("  Medium: %d/%d" % (medium, assembled))
    print("  Low:    %d/%d" % (low, assembled))
    print("  Avg score: %.0f/100" % avg_score)
    api_calls = total - errors - t_cached
    print("\nCost: $%.4f (%d API calls, %d cached, %d+%d tokens)" % (
        sonnet_cost, api_calls, t_cached, sonnet_tokens_in, sonnet_tokens_out))

    if write_db:
        print("\nResults written to DB.")
    else:
        print("\nDry run (no DB writes). Use --write-db to persist results.")

    # Generate report
    report = generate_report(results, source, puzzle, stats)
    report_path = "%s/puzzle_report_%s_%s.txt" % (output_dir, source, puzzle)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\nReport saved to %s" % report_path)

    return results, stats


def main():
    parser = argparse.ArgumentParser(
        description="Sonnet pipeline: Sonnet -> Assembler -> Fallback"
    )
    parser.add_argument("puzzles", nargs="*", default=[PUZZLE_NUMBER] if PUZZLE_NUMBER else None,
                        help="Puzzle number(s) (e.g. 29939 29926)")
    parser.add_argument("--source", default=SOURCE,
                        help="Puzzle source (default: %s)" % SOURCE)
    parser.add_argument("--write-db", action="store_true", default=WRITE_DB,
                        help="Write results to clues_master.db (default: dry run)")
    parser.add_argument("--output-dir", default=OUTPUT_DIR,
                        help="Directory for report files (default: %s)" % OUTPUT_DIR)
    parser.add_argument("--single-clue", type=str, default=SINGLE_CLUE_MATCH,
                        help="Filter to single clue matching this text (overrides puzzle selection)")
    parser.add_argument("--force", action="store_true", default=FORCE_API,
                        help="Fresh API calls for all clues (ignore cached results)")
    args = parser.parse_args()

    # Single-clue mode: if puzzle provided, just filter within it;
    # if no puzzle, search the whole DB to find it
    if args.single_clue and not args.puzzles:
        conn = sqlite3.connect(CLUES_DB)
        match = conn.execute(
            "SELECT source, puzzle_number, clue_text FROM clues WHERE clue_text LIKE ? LIMIT 1",
            ("%" + args.single_clue + "%",)
        ).fetchone()
        conn.close()
        if not match:
            print("No clue found matching: %s" % args.single_clue)
            sys.exit(1)
        args.source = match[0]
        args.puzzles = [str(match[1])]
        print("Single-clue mode: matched '%s' in %s #%s" % (
            match[2][:60], match[0], match[1]))

    if not args.puzzles:
        parser.error("No puzzle number(s) provided. Set PUZZLE_NUMBER at the top of run.py or pass on the command line.")

    # Load shared resources once
    print("Loading enricher...")
    enricher = ClueEnricher()
    print("Loading homophone engine...")
    homo_engine = HomophoneEngine(db_path=CRYPTIC_DB)
    example_messages = build_example_messages()

    all_stats = []
    for puzzle in args.puzzles:
        results, stats = run_puzzle(
            args.source, puzzle, enricher, homo_engine, example_messages,
            write_db=args.write_db, output_dir=args.output_dir,
            single_clue=args.single_clue, force=args.force,
        )
        if stats:
            all_stats.append((puzzle, stats))

    # Cross-puzzle summary
    if len(all_stats) > 1:
        print("\n" + "=" * 80)
        print("CROSS-PUZZLE SUMMARY (%d puzzles)" % len(all_stats))
        print("=" * 80)
        grand_total = sum(s["total"] for _, s in all_stats)
        grand_assembled = sum(s["assembled"] for _, s in all_stats)
        grand_failed = sum(s["failed"] for _, s in all_stats)
        grand_cost = sum(s["total_cost"] for _, s in all_stats)
        grand_high = sum(s["high"] for _, s in all_stats)
        grand_medium = sum(s["medium"] for _, s in all_stats)
        grand_low = sum(s["low"] for _, s in all_stats)
        for puzzle, stats in all_stats:
            print("  %s: %d/%d assembled (%d%%), avg score %.0f" % (
                puzzle, stats["assembled"], stats["total"],
                100 * stats["assembled"] // max(stats["total"], 1),
                stats["avg_score"]))
        print("-" * 40)
        print("  Total:  %d/%d assembled (%d%%)" % (
            grand_assembled, grand_total,
            100 * grand_assembled // max(grand_total, 1)))
        print("  High: %d | Medium: %d | Low: %d" % (
            grand_high, grand_medium, grand_low))
        print("  Cost:   $%.4f" % grand_cost)

    enricher.close()


if __name__ == "__main__":
    main()
