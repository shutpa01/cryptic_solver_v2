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
    try_hidden, try_spoonerism_v2, try_double_definition,
)
from .report import generate_report, _describe_assembly
from .sig_adapter import (
    build_result_dict as sig_build_result_dict,
    store_signature_result,
    SIG_OP_TO_TYPE,
)
from .sig_enrichment import (
    collect_gaps_from_results, enrich_refdb,
    collect_signatures_from_results, enrich_catalog,
    collect_indicators_from_results, enrich_indicators,
)

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
SOURCE = "telegraph"            # telegraph, guardian, times, independentclaude
PUZZLE_NUMBER = "3359"             # puzzle number to solve
WRITE_DB = True                # write results to clues_master.db
FORCE_API = True               # Always make fresh API calls (captures reasoning, uses latest prompt)
PARTIALS = False              # True = re-run partial solves (has_solution=2)
SINGLE_CLUE_MATCH = ""





def run_puzzle(source, puzzle, enricher, homo_engine, example_messages,
               write_db=False, output_dir=OUTPUT_DIR, single_clue="", force=False,
               partials=False, ref_db=None):
    """Run the combined S+P pipeline on a single puzzle. Returns (results, stats, gaps).

    Flow:
      Phase 1: Signature solver (S) on all clues — zero API cost
      Phase 2: Production solver (P) on remaining clues — API calls
      Phase 3: Collect P's discoveries → enrich RefDB → S re-runs on failures
    """
    db_path = CLUES_DB
    conn = sqlite3.connect(db_path, timeout=30)
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
        return [], {}, []

    # Skip clues that already have explanations (protect manual corrections)
    # Only force=True overrides this safety check
    if not force:
        total_before = len(rows)
        already_done_ids = set()
        for r in rows:
            cid = r[0]
            existing = conn.execute(
                "SELECT has_solution FROM clues WHERE id = ? AND has_solution IS NOT NULL",
                (cid,)
            ).fetchone()
            if existing:
                already_done_ids.add(cid)
        if already_done_ids:
            rows = [r for r in rows if r[0] not in already_done_ids]
            print("Skipping %d clue(s) with existing explanations (use --force to override)"
                  % len(already_done_ids))

    if not rows:
        conn.close()
        print("All clues already have explanations for %s puzzle %s" % (source, puzzle))
        return [], {}, []

    print("=" * 80)
    print("COMBINED PIPELINE: Signature -> Sonnet -> Enrichment -> Signature")
    print("Puzzle: %s %s (%d clues)" % (source, puzzle, len(rows)))
    print("=" * 80)

    sonnet_tokens_in = 0
    sonnet_tokens_out = 0
    results = []

    # ================================================================
    # PHASE 0: Mechanical hidden word check (zero API cost, guaranteed)
    # If the answer is contiguously hidden in the clue text, it's a hidden word.
    # This should NEVER be missed.
    # ================================================================
    hidden_solved_ids = set()
    hidden_count = 0

    print("\n--- Phase 0: Mechanical hidden word check ---")
    for row in rows:
        cid, cnum, direction, clue, answer, enum, explanation = row
        answer_clean = clean(answer)
        if not answer_clean or len(answer_clean) < 3:
            continue

        hidden_result = try_hidden(clue, answer_clean)
        if hidden_result:
            hidden_solved_ids.add(cid)
            hidden_count += 1

            # Build a result dict compatible with the rest of the pipeline
            wtype = "hidden" if hidden_result["op"] == "hidden" else "hidden"
            is_reversed = "reversed" in hidden_result.get("op", "")
            hiding_words = hidden_result.get("words", "")

            # Build explanation text with highlighted hidden letters
            from sonnet_pipeline.report import _highlight_hidden
            if is_reversed:
                highlighted = _highlight_hidden(hiding_words, answer_clean[::-1])
                expl_text = 'hidden reversed in "%s"' % highlighted
            else:
                highlighted = _highlight_hidden(hiding_words, answer_clean)
                expl_text = 'hidden in "%s"' % highlighted

            results.append({
                "status": "ASSEMBLED",
                "tier": "Hidden",
                "confidence": "high",
                "score": 100,
                "clue_number": cnum,
                "direction": direction,
                "enumeration": enum,
                "clue": clue,
                "answer": answer,
                "explanation": expl_text,
            })

            if write_db:
                # Store as a structured explanation
                import json as _json
                pieces_data = [{
                    "clue_word": hiding_words,
                    "letters": answer_clean,
                    "mechanism": "hidden",
                }]
                components = _json.dumps({
                    "ai_pieces": pieces_data,
                    "assembly": hidden_result,
                    "wordplay_type": hidden_result["op"],
                })
                conn.execute("""
                    INSERT OR REPLACE INTO structured_explanations
                    (clue_id, components, wordplay_types, definition_text, confidence, model_version, source)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (cid, components, _json.dumps([hidden_result["op"]]),
                      None, 1.0, "mechanical_hidden", source))
                conn.execute("""
                    UPDATE clues SET
                        wordplay_type = COALESCE(NULLIF(wordplay_type, ''), ?),
                        ai_explanation = COALESCE(NULLIF(ai_explanation, ''), ?),
                        has_solution = 1
                    WHERE id = ?
                """, (hidden_result["op"], expl_text, cid))
                conn.commit()

            print("  [HIDDEN 100] %s. %s = %s  %s" % (cnum, clue[:50], answer, expl_text))

    if hidden_count:
        print("  Phase 0a: %d hidden words found" % hidden_count)
    else:
        print("  Phase 0a: no hidden words")

    # ================================================================
    # PHASE 0b: Mechanical spoonerism check
    # If "Spooner" appears in clue and the answer splits into two words
    # whose swapped initials are also valid words, it's solved.
    # ================================================================
    spoonerism_count = 0

    if ref_db is not None:
        for row in rows:
            cid, cnum, direction, clue, answer, enum, explanation = row
            if cid in hidden_solved_ids:
                continue

            answer_clean = clean(answer)
            if not answer_clean or len(answer_clean) < 4:
                continue

            # Only try if "Spooner" appears in the clue
            if "spooner" not in clue.lower():
                continue

            spoon_result = try_spoonerism_v2(answer_clean, ref_db.is_real_word,
                                              clue_text=clue, ref_db=ref_db)
            if spoon_result:
                hidden_solved_ids.add(cid)  # reuse the set to skip in later phases
                spoonerism_count += 1

                w1 = spoon_result["word1"]
                w2 = spoon_result["word2"]
                sw1 = spoon_result["swapped1"]
                sw2 = spoon_result["swapped2"]
                cw1 = spoon_result.get("clue_word1")
                cw2 = spoon_result.get("clue_word2")

                # Build explanation with clue word mappings where found
                parts = []
                if cw1:
                    parts.append('"%s" = %s' % (cw1, sw1))
                else:
                    parts.append(sw1)
                if cw2:
                    parts.append('"%s" = %s' % (cw2, sw2))
                else:
                    parts.append(sw2)
                expl_text = 'Spoonerism: %s -> swap initials -> %s %s' % (' + '.join(parts), w1, w2)

                results.append({
                    "status": "ASSEMBLED",
                    "tier": "Spoonerism",
                    "confidence": "high",
                    "score": 100,
                    "clue_number": cnum,
                    "direction": direction,
                    "enumeration": enum,
                    "clue": clue,
                    "answer": answer,
                    "explanation": expl_text,
                })

                if write_db:
                    import json as _json
                    pieces_data = [
                        {"clue_word": "%s %s" % (sw1, sw2), "letters": answer_clean, "mechanism": "spoonerism"},
                    ]
                    components = _json.dumps({
                        "ai_pieces": pieces_data,
                        "assembly": spoon_result,
                        "wordplay_type": "spoonerism",
                    })
                    conn.execute("""
                        INSERT OR REPLACE INTO structured_explanations
                        (clue_id, components, wordplay_types, definition_text, confidence, model_version, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (cid, components, _json.dumps(["spoonerism"]),
                          None, 1.0, "mechanical_spoonerism", source))
                    conn.execute("""
                        UPDATE clues SET
                            wordplay_type = COALESCE(NULLIF(wordplay_type, ''), 'spoonerism'),
                            ai_explanation = COALESCE(NULLIF(ai_explanation, ''), ?),
                            has_solution = 1
                        WHERE id = ?
                    """, (expl_text, cid))
                    conn.commit()

                print("  [SPOONERISM 100] %s. %s = %s  %s" % (cnum, clue[:50], answer, expl_text))

    if spoonerism_count:
        print("  Phase 0b: %d spoonerisms found" % spoonerism_count)
    else:
        print("  Phase 0b: no spoonerisms")

    # ================================================================
    # PHASE 0c: Mechanical double definition check
    # Split clue at every point, check if both halves independently
    # define the known answer via definition_answers or synonyms.
    # ================================================================
    dd_solved_ids = set()
    dd_count = 0

    if ref_db is not None:
        for row in rows:
            cid, cnum, direction, clue, answer, enum, explanation = row
            if cid in hidden_solved_ids:
                continue

            answer_clean = clean(answer)
            if not answer_clean or len(answer_clean) < 2:
                continue

            dd_result = try_double_definition(clue, answer_clean, ref_db)
            if dd_result:
                dd_solved_ids.add(cid)
                dd_count += 1

                left_def = dd_result["left_def"]
                right_def = dd_result["right_def"]
                expl_text = 'Double definition: "%s" and "%s" both mean %s' % (
                    left_def, right_def, answer)

                results.append({
                    "status": "ASSEMBLED",
                    "tier": "DD",
                    "confidence": "high",
                    "score": 100,
                    "clue_number": cnum,
                    "direction": direction,
                    "enumeration": enum,
                    "clue": clue,
                    "answer": answer,
                    "explanation": expl_text,
                })

                if write_db:
                    import json as _json
                    components = _json.dumps({
                        "ai_pieces": [],
                        "assembly": dd_result,
                        "wordplay_type": "double_definition",
                    })
                    conn.execute("""
                        INSERT OR REPLACE INTO structured_explanations
                        (clue_id, components, wordplay_types, definition_text, confidence, model_version, source)
                        VALUES (?, ?, ?, ?, ?, ?, ?)
                    """, (cid, components, _json.dumps(["double_definition"]),
                          left_def, 1.0, "mechanical_dd", source))
                    conn.execute("""
                        UPDATE clues SET
                            wordplay_type = COALESCE(NULLIF(wordplay_type, ''), 'double_definition'),
                            definition = COALESCE(NULLIF(definition, ''), ?),
                            ai_explanation = COALESCE(NULLIF(ai_explanation, ''), ?),
                            has_solution = 1
                        WHERE id = ?
                    """, (left_def, expl_text, cid))
                    conn.commit()

                print("  [DD 100] %s. %s = %s  %s" % (cnum, clue[:50], answer, expl_text))

    if dd_count:
        print("  Phase 0c: %d double definitions found" % dd_count)
    else:
        print("  Phase 0c: no double definitions")

    # ================================================================
    # PHASE 1: Signature solver on all clues (zero API cost)
    # ================================================================
    sig_solved_ids = set()    # clue IDs solved by S with HIGH confidence
    sig_results = {}          # clue_id -> result dict (for all S attempts)
    sig_high = 0
    sig_medium = 0

    if ref_db is not None:
        from signature_solver.solver import solve_clue as sig_solve_clue

        print("\n--- Phase 1: Signature solver (mechanical, zero API cost) ---")
        t0 = time.time()

        for row in rows:
            cid, cnum, direction, clue, answer, enum, explanation = row
            answer_clean = clean(answer)

            # Skip clues already solved by Phase 0 (hidden word, spoonerism, DD)
            if cid in hidden_solved_ids or cid in dd_solved_ids:
                continue

            # Skip cross-reference clues — S can't resolve them
            if re.search(r'\b\d+\s*(?:across|down|ac|dn)\b', clue, re.IGNORECASE):
                continue

            try:
                sr = sig_solve_clue(clue, answer_clean, ref_db)
            except Exception as e:
                print("  S error on %s: %s" % (cnum, e))
                continue

            if sr.high_confidence:
                sig_solved_ids.add(cid)
                sig_high += 1
                result_dict = sig_build_result_dict(
                    sr, clue, answer, cnum, direction, enum, explanation
                )
                sig_results[cid] = result_dict
                results.append(result_dict)

                # Store to DB
                if write_db:
                    store_signature_result(conn, cid, sr, clue, answer)

                print("  [S HIGH %3d] %s. %s = %s" % (sr.confidence, cnum, clue[:50], answer))
            elif sr.solved:
                sig_medium += 1
                # Store medium result for potential Phase 3 upgrade
                sig_results[cid] = sig_build_result_dict(
                    sr, clue, answer, cnum, direction, enum, explanation
                )

        elapsed = time.time() - t0
        print("  Phase 1: %d HIGH, %d medium in %.1fs — %d clues skip API" % (
            sig_high, sig_medium, elapsed, sig_high))

    # ================================================================
    # PHASE 1.5: TFTT — parse human explanations with Haiku (Times only)
    # ================================================================
    tftt_solved_ids = set()
    tftt_high = 0

    if source == "times" and not single_clue:
        try:
            from .tftt_pipeline import fetch_tftt, parse_with_haiku, score_parse, store_tftt_result
            import anthropic as _anthropic

            print("\n--- Phase 1.5: TFTT blog explanations (Haiku parsing) ---")
            t0 = time.time()

            tftt_clues = fetch_tftt(int(puzzle))
            if tftt_clues:
                print("  Fetched %d clues from TFTT" % len(tftt_clues))

                # Build lookup: clean_answer -> tftt_clue
                tftt_by_answer = {}
                for tc in tftt_clues:
                    key = clean(tc["answer"])
                    tftt_by_answer[key] = tc

                haiku_client = _anthropic.Anthropic()

                for row in rows:
                    cid, cnum, direction, clue, answer, enum, explanation = row

                    # Skip clues already solved by S
                    if cid in sig_solved_ids:
                        continue

                    answer_clean = clean(answer)
                    tc = tftt_by_answer.get(answer_clean)
                    if not tc or not tc.get("explanation"):
                        continue

                    # Parse with Haiku
                    parsed, usage = parse_with_haiku(
                        haiku_client, clue, answer, tc["explanation"]
                    )
                    if not parsed:
                        continue

                    # Score
                    score, reasons = score_parse(parsed, answer, ref_db)

                    if score >= 70:
                        tftt_solved_ids.add(cid)
                        tftt_high += 1

                        # Store to DB
                        if write_db:
                            store_tftt_result(
                                conn, cid, parsed, score,
                                tc.get("definition", ""),
                                raw_explanation=tc.get("explanation", "")
                            )

                        # Add to results for report + gap collection
                        conf_label = "high" if score >= 80 else "medium"
                        results.append({
                            "status": "ASSEMBLED",
                            "tier": "TFTT",
                            "confidence": conf_label,
                            "score": score,
                            "clue_number": cnum,
                            "direction": direction,
                            "enumeration": enum,
                            "clue": clue,
                            "answer": answer,
                            "explanation": explanation,
                            "ai_output": parsed,  # include pieces for gap collection
                        })

                        reason_str = ", ".join(
                            "%s(%d)" % (r, d) for r, d in reasons
                        ) if reasons else "clean"
                        print("  [TFTT %3d] %s. %s = %s  %s" % (
                            score, cnum, clue[:40], answer, reason_str))
                    else:
                        reason_str = ", ".join(
                            "%s(%d)" % (r, d) for r, d in reasons
                        )
                        print("  [TFTT LOW %3d] %s. %s = %s  %s" % (
                            score, cnum, clue[:40], answer, reason_str))

                elapsed = time.time() - t0
                remaining = len(rows) - len(sig_solved_ids) - len(tftt_solved_ids)
                print("  Phase 1.5: %d TFTT solved in %.1fs — %d clues remain for API" % (
                    tftt_high, elapsed, remaining))
            else:
                print("\n--- Phase 1.5: No TFTT page found for puzzle %s ---" % puzzle)
        except Exception as e:
            print("\n--- Phase 1.5: TFTT failed: %s ---" % e)

    # ================================================================
    # PHASE 1.5b: fifteensquared — Guardian/Independent human explanations
    # ================================================================
    fs_solved_ids = set()
    fs_high = 0

    if source in ("guardian", "independent") and not single_clue:
        try:
            from .fifteensquared_pipeline import fetch_fifteensquared, store_fifteensquared_result
            from .tftt_pipeline import parse_with_haiku, score_parse
            import anthropic as _anthropic

            print("\n--- Phase 1.5b: fifteensquared blog explanations (Haiku parsing) ---")
            t0 = time.time()

            # Get publication date for URL discovery
            pub_date = conn.execute(
                "SELECT publication_date FROM clues WHERE source = ? AND puzzle_number = ? LIMIT 1",
                (source, puzzle),
            ).fetchone()
            pub_date = pub_date[0] if pub_date else None

            fs_clues = fetch_fifteensquared(int(puzzle), source, pub_date)
            if fs_clues:
                print("  Fetched %d clues from fifteensquared" % len(fs_clues))

                # Build lookup: clean_answer -> blog_clue
                fs_by_answer = {}
                for fc in fs_clues:
                    key = clean(fc["answer"])
                    fs_by_answer[key] = fc

                haiku_client = _anthropic.Anthropic()

                for row in rows:
                    cid, cnum, direction, clue, answer, enum, explanation = row

                    # Skip clues already solved by S
                    if cid in sig_solved_ids:
                        continue

                    answer_clean = clean(answer)
                    fc = fs_by_answer.get(answer_clean)
                    if not fc or not fc.get("explanation"):
                        continue

                    # Parse with Haiku
                    parsed, usage = parse_with_haiku(
                        haiku_client, clue, answer, fc["explanation"]
                    )
                    if not parsed:
                        continue

                    # Score
                    score, reasons = score_parse(parsed, answer, ref_db)

                    if score >= 70:
                        fs_solved_ids.add(cid)
                        fs_high += 1

                        # Store to DB
                        if write_db:
                            store_fifteensquared_result(
                                conn, cid, parsed, score,
                                fc.get("definition", ""),
                                raw_explanation=fc.get("explanation", ""),
                                source_name=source,
                            )

                        # Add to results for report + gap collection
                        conf_label = "high" if score >= 80 else "medium"
                        results.append({
                            "status": "ASSEMBLED",
                            "tier": "FS",
                            "confidence": conf_label,
                            "score": score,
                            "clue_number": cnum,
                            "direction": direction,
                            "enumeration": enum,
                            "clue": clue,
                            "answer": answer,
                            "explanation": explanation,
                            "ai_output": parsed,  # include pieces for gap collection
                        })

                        reason_str = ", ".join(
                            "%s(%d)" % (r, d) for r, d in reasons
                        ) if reasons else "clean"
                        print("  [FS %3d] %s. %s = %s  %s" % (
                            score, cnum, clue[:40], answer, reason_str))
                    else:
                        reason_str = ", ".join(
                            "%s(%d)" % (r, d) for r, d in reasons
                        )
                        print("  [FS LOW %3d] %s. %s = %s  %s" % (
                            score, cnum, clue[:40], answer, reason_str))

                elapsed = time.time() - t0
                remaining = len(rows) - len(sig_solved_ids) - len(fs_solved_ids)
                print("  Phase 1.5b: %d fifteensquared solved in %.1fs — %d clues remain for API" % (
                    fs_high, elapsed, remaining))
            else:
                print("\n--- Phase 1.5b: No fifteensquared page found for %s %s ---" % (source, puzzle))
        except Exception as e:
            import traceback
            traceback.print_exc()
            print("\n--- Phase 1.5b: fifteensquared failed: %s ---" % e)

    # ================================================================
    # PHASE 2: Production solver on remaining clues (API calls)
    # ================================================================
    all_solved = hidden_solved_ids | dd_solved_ids | sig_solved_ids | tftt_solved_ids | fs_solved_ids
    remaining_for_api = [r for r in rows if r[0] not in all_solved]
    print("\n--- Phase 2: Production solver (API calls) — %d clues ---" % len(remaining_for_api))

    for row in remaining_for_api:
        cid, cnum, direction, clue, answer, enum, explanation = row
        target = clean(answer)

        # Check for existing solved result — skip API call but re-run assembler
        # has_solution: 1=solved, 2=partial, 0=failed, NULL=untried
        # Normal run: cache 1+2. Partials mode: cache 1 only (re-run 2s).
        cached_ai = None
        if not force:
            if partials:
                cache_sql = "WHERE se.clue_id = ? AND c.has_solution = 1"
            else:
                cache_sql = "WHERE se.clue_id = ? AND c.has_solution IN (1, 2)"
            existing = conn.execute("""
                SELECT se.confidence, se.wordplay_types, se.components, se.definition_text
                FROM structured_explanations se
                JOIN clues c ON se.clue_id = c.id
                %s
            """ % cache_sql, (cid,)).fetchone()
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
                example_messages, cached_ai=cached_ai, ref_db=ref_db
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
        # Show reasoning from two-pass solver if available
        reasoning = sonnet_out.get("_reasoning", "") if sonnet_out else ""
        if reasoning and not cached_ai:
            # Show first 120 chars of reasoning for quick debugging
            preview = reasoning.replace("\n", " | ")[:120]
            print("   Reasoning: %s" % preview)
        print("   Sonnet: type=%s, def=%s" % (sonnet_wtype, repr(sonnet_def)))
        print("   Pieces: %s -> %s (target=%s)" % (
            sonnet_pieces, "".join(sonnet_pieces), target))
        if assembly:
            desc = _describe_assembly(assembly, sonnet_out.get("pieces", []) if sonnet_out else [], answer=answer)
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
                conn.commit()
        else:
            print("   Confidence: NONE (0/100)")
            print("   Status: FAILED")
            if write_db:
                # Only set reviewed=0 if not already manually reviewed
                cur_rev = conn.execute("SELECT reviewed FROM clues WHERE id = ?", (cid,)).fetchone()
                if cur_rev and cur_rev[0] in (1, 2):
                    conn.execute("UPDATE clues SET has_solution = 0 WHERE id = ?", (cid,))
                else:
                    conn.execute("UPDATE clues SET has_solution = 0, reviewed = 0 WHERE id = ?", (cid,))
                conn.commit()

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

    # ================================================================
    # PHASE 3: Collect P's discoveries → enrich RefDB + catalog → S re-runs
    # ================================================================
    sig_re_solved = 0
    if ref_db is not None:
        from signature_solver.solver import solve_clue as sig_solve_clue
        from signature_solver.catalog import CATALOG

        # Collect synonym/abbreviation/definition gaps from ALL results (P, TFTT, FS)
        non_sig_results = [r for r in results if r.get("tier") not in ("Signature",)]
        gaps_for_enrichment = collect_gaps_from_results(non_sig_results)

        # Collect new signature patterns from P's results
        new_sigs = collect_signatures_from_results(p_results)
        extra_catalog, n_cat_added = enrich_catalog(CATALOG, new_sigs)

        enriched_db = ref_db
        n_db_injected = 0
        if gaps_for_enrichment:
            enriched_db, injected = enrich_refdb(ref_db, gaps_for_enrichment)
            n_db_injected = len(injected)

        # Inject inferred indicator words into the enriched DB
        new_indicators = collect_indicators_from_results(p_results, ref_db=ref_db)
        n_ind_injected = 0
        if new_indicators:
            # enrich_indicators mutates enriched_db.indicators in place
            # (already a cloned copy from enrich_refdb, or the original if no gaps)
            if enriched_db is ref_db:
                # No synonym/abbreviation gaps — need to clone before mutating
                import copy
                enriched_db = copy.copy(ref_db)
                enriched_db.indicators = dict(ref_db.indicators)
            elif not hasattr(enriched_db, '_indicators_cloned'):
                # enrich_refdb cloned synonyms/abbreviations but not indicators
                enriched_db.indicators = dict(enriched_db.indicators)
            ind_injected = enrich_indicators(enriched_db, new_indicators)
            n_ind_injected = len(ind_injected)

        if n_db_injected or n_cat_added or n_ind_injected:
            print("\n--- Phase 3: Signature re-solve with enriched DB + catalog ---")
            if n_db_injected:
                print("  Injected %d DB entries (synonyms/abbreviations)" % n_db_injected)
            if n_ind_injected:
                print("  Injected %d inferred indicators" % n_ind_injected)
            if n_cat_added:
                print("  Injected %d new catalog signatures from P's solves:" % n_cat_added)
                for entry in extra_catalog[:10]:
                    print("    %s [%s]" % (entry.label, entry.operation))
                if n_cat_added > 10:
                    print("    ... and %d more" % (n_cat_added - 10))

            # Re-run S on all clues not already solved by S in Phase 1
            failed_or_weak = [
                row for row in rows
                if row[0] not in sig_solved_ids  # not already S HIGH
            ]

            for row in failed_or_weak:
                cid, cnum, direction, clue, answer, enum, explanation = row
                answer_clean = clean(answer)

                # Skip cross-reference clues
                if re.search(r'\b\d+\s*(?:across|down|ac|dn)\b', clue, re.IGNORECASE):
                    continue

                try:
                    sr = sig_solve_clue(clue, answer_clean, enriched_db,
                                        extra_catalog=extra_catalog or None)
                except Exception as e:
                    print("  [S+E ERROR] %s. %s: %s" % (cnum, clue[:40], e))
                    continue

                if not sr.solved:
                    print("  [S+E MISS] %s. %s — no solve" % (cnum, clue[:50]))
                elif not sr.high_confidence:
                    print("  [S+E LOW] %s. %s — conf=%d" % (cnum, clue[:50], sr.confidence))
                    if hasattr(sr, 'confidence_reasons'):
                        for reason, delta in sr.confidence_reasons:
                            print("    %+d  %s" % (delta, reason))

                if sr.high_confidence:
                    sig_re_solved += 1
                    new_result = sig_build_result_dict(
                        sr, clue, answer, cnum, direction, enum, explanation
                    )
                    new_result["tier"] = "Signature+Enriched"

                    # Replace existing result for this clue in results[]
                    replaced = False
                    for i, r in enumerate(results):
                        if r.get("clue_number") == cnum and r.get("direction") == direction:
                            results[i] = new_result
                            replaced = True
                            break
                    if not replaced:
                        results.append(new_result)

                    # Overwrite DB
                    if write_db:
                        store_signature_result(conn, cid, sr, clue, answer, enriched=True)

                    print("  [S+E HIGH %3d] %s. %s = %s" % (
                        sr.confidence, cnum, clue[:50], answer))

            if sig_re_solved:
                print("  Phase 3: %d clues upgraded to signature quality" % sig_re_solved)
            else:
                print("  Phase 3: no additional solves from enrichment")

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
    t_sig = sum(1 for r in results if r.get("tier") == "Signature")
    t_sig_e = sum(1 for r in results if r.get("tier") == "Signature+Enriched")

    assembled_results = [r for r in results if r.get("status") == "ASSEMBLED"]
    high = sum(1 for r in assembled_results if r.get("confidence") == "high")
    medium = sum(1 for r in assembled_results if r.get("confidence") == "medium")
    low = sum(1 for r in assembled_results if r.get("confidence") == "low")
    avg_score = sum(r.get("score", 0) for r in assembled_results) / max(len(assembled_results), 1)
    t_tftt = sum(1 for r in results if r.get("tier") == "TFTT")
    t_fs = sum(1 for r in results if r.get("tier") == "FS")

    sonnet_cost = sonnet_tokens_in / 1e6 * 3.0 + sonnet_tokens_out / 1e6 * 15.0

    stats = {
        "total": total, "assembled": assembled, "failed": failed, "errors": errors,
        "sonnet": t1, "fallback": t3, "cached": t_cached,
        "signature": t_sig, "signature_enriched": t_sig_e,
        "tftt": t_tftt, "fifteensquared": t_fs,
        "high": high, "medium": medium, "low": low, "avg_score": avg_score,
        "total_cost": sonnet_cost,
        "sonnet_tokens_in": sonnet_tokens_in,
        "sonnet_tokens_out": sonnet_tokens_out,
    }

    # Print summary
    print("\n" + "=" * 80)
    print("COMBINED PIPELINE SUMMARY")
    print("=" * 80)
    print("Total clues:      %d" % total)
    print("ASSEMBLED:        %d/%d (%d%%)" % (assembled, total, 100 * assembled // max(total, 1)))
    if t_sig:
        print("  Signature:      %d  (zero API cost)" % t_sig)
    if t_sig_e:
        print("  Sig+Enriched:   %d  (P-discovered, S-explained)" % t_sig_e)
    if t_tftt:
        print("  TFTT+Haiku:     %d  (human explanation, Haiku parsed)" % t_tftt)
    if t_fs:
        print("  15²+Haiku:      %d  (human explanation, Haiku parsed)" % t_fs)
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
    api_calls = total - errors - t_cached - t_sig - t_sig_e - t_tftt - t_fs
    sig_saved = t_sig + t_sig_e
    blog_saved = t_tftt + t_fs
    print("\nCost: $%.4f (%d API calls, %d signature, %d blog+Haiku, %d cached, %d+%d tokens)" % (
        sonnet_cost, api_calls, sig_saved, blog_saved, t_cached, sonnet_tokens_in, sonnet_tokens_out))

    if write_db:
        print("\nResults written to DB.")
    else:
        print("\nDry run (no DB writes). Use --write-db to persist results.")

    # Generate report (DB gaps shown in the actionable quality section)
    report, gaps = generate_report(results, source, puzzle, stats)
    os.makedirs(output_dir, exist_ok=True)
    report_path = "%s/puzzle_report_%s_%s.txt" % (output_dir, source, puzzle)
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report)
    print("\nReport saved to %s" % report_path)

    if gaps:
        gaps_path = "%s/pending_gaps_%s_%s.json" % (output_dir, source, puzzle)
        gaps_data = {
            "source": source,
            "puzzle": puzzle,
            "stats": {
                "total": stats["total"],
                "assembled": stats["assembled"],
                "high": stats["high"],
                "medium": stats["medium"],
                "low": stats["low"],
                "failed": stats["failed"],
                "avg_score": stats["avg_score"],
            },
            "gaps": gaps,
        }
        with open(gaps_path, "w", encoding="utf-8") as f:
            json.dump(gaps_data, f, indent=2, ensure_ascii=False)
        print("Pending DB gaps saved to %s (%d entries)" % (gaps_path, len(gaps)))

        # Store gaps in DB for dashboard review — skip items already in reference DB
        gap_conn = sqlite3.connect(CLUES_DB, timeout=30)
        ref_conn = sqlite3.connect(CRYPTIC_DB, timeout=10)
        for g in gaps:
            gtype = g.get("type", "")
            word = g.get("word") or g.get("definition") or ""
            letters = g.get("letters") or g.get("answer") or ""
            # Reclassify: "abbreviation" with 3+ letter result is really a synonym
            if gtype == "abbreviation" and len(re.sub(r"[^A-Z]", "", letters.upper())) >= 3:
                gtype = "synonym"
            answer = g.get("answer", "")
            clue = g.get("clue", "")

            # Check if already in reference DB (either table)
            already = False
            if gtype in ("synonym", "abbreviation"):
                already = ref_conn.execute(
                    "SELECT 1 FROM synonyms_pairs WHERE LOWER(word)=? AND UPPER(synonym)=?",
                    (word.lower(), letters.upper())
                ).fetchone() is not None
                if not already:
                    already = ref_conn.execute(
                        "SELECT 1 FROM wordplay WHERE LOWER(indicator)=? AND UPPER(substitution)=?",
                        (word.lower(), letters.upper())
                    ).fetchone() is not None
            elif gtype == "definition":
                already = ref_conn.execute(
                    "SELECT 1 FROM definition_answers_augmented WHERE LOWER(definition)=? AND LOWER(answer)=?",
                    (word.lower(), letters.lower())
                ).fetchone() is not None

            if not already:
                gap_conn.execute(
                    "INSERT OR IGNORE INTO pending_enrichments "
                    "(type, word, letters, answer, clue_text, source, puzzle_number) "
                    "VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (gtype, word, letters, answer, clue, source, puzzle),
                )
        gap_conn.commit()
        gap_conn.close()
        ref_conn.close()

    return results, stats, gaps


def _show_puzzle_summary(source, puzzle):
    """Show current solve status for a puzzle."""
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    rows = conn.execute("""
        SELECT COUNT(*) AS total,
               SUM(CASE WHEN has_solution = 1 THEN 1 ELSE 0 END) AS solved,
               SUM(CASE WHEN has_solution = 2 THEN 1 ELSE 0 END) AS partial,
               SUM(CASE WHEN has_solution = 0 THEN 1 ELSE 0 END) AS failed,
               SUM(CASE WHEN has_solution IS NULL THEN 1 ELSE 0 END) AS untried,
               SUM(CASE WHEN definition IS NOT NULL AND wordplay_type IS NOT NULL
                         AND ai_explanation IS NOT NULL THEN 1 ELSE 0 END) AS fully_annotated
        FROM clues WHERE source = ? AND puzzle_number = ?
    """, (source, puzzle)).fetchone()
    conn.close()
    total, solved, partial, failed, untried, annotated = rows
    print("  %s #%s: %d clues — %d solved, %d partial, %d failed, %d untried, %d fully annotated" % (
        source, puzzle, total, solved or 0, partial or 0, failed or 0, untried or 0, annotated or 0))


def _run_full_pipeline(args):
    """Mode 1: Run full pipeline (S → P → enrich → S re-run → gaps → manual)."""
    # Load shared resources once
    print("Loading enricher...")
    enricher = ClueEnricher()
    print("Loading homophone engine...")
    homo_engine = HomophoneEngine(db_path=CRYPTIC_DB)
    example_messages = build_example_messages()

    # Load signature solver's RefDB once (~2s)
    print("Loading signature solver reference DB...")
    from signature_solver.db import RefDB
    ref_db = RefDB()

    all_stats = []
    any_gaps = False
    for puzzle in args.puzzles:
        results, stats, gaps = run_puzzle(
            args.source, puzzle, enricher, homo_engine, example_messages,
            write_db=args.write_db, output_dir=args.output_dir,
            single_clue=args.single_clue, force=args.force,
            partials=args.partials, ref_db=ref_db,
        )
        if stats:
            all_stats.append((puzzle, stats))
        if gaps:
            any_gaps = True

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

    # After pipeline: DB gaps → re-run → manual entry
    if any_gaps and args.write_db and not args.no_review:
        from .review_gaps import main as review_main
        print("\n" + "-" * 80)
        print("DB GAPS DETECTED — launching gap review...")
        print("-" * 80)
        # Find the gaps file for this puzzle and set sys.argv so review_main picks it up
        gaps_path = os.path.join(args.output_dir, "pending_gaps_%s_%s.json" % (args.source, args.puzzles[0]))
        if os.path.exists(gaps_path):
            sys.argv = [sys.argv[0], gaps_path]
        else:
            sys.argv = [sys.argv[0]]
        try:
            review_main()
        except (EOFError, KeyboardInterrupt):
            pass
    elif any_gaps and args.no_review:
        print("\nDB gaps detected but --no-review set. Review from the dashboard.")


def _run_db_additions(args):
    """Mode 2: Review and approve DB gap inserts (synonyms, abbreviations, definitions)."""
    from .review_gaps import main as review_main
    # review_gaps finds the latest gaps file automatically
    review_args = [args.puzzles[0]] if args.puzzles else []
    # Pass the gaps file path if it exists for this puzzle
    gaps_path = os.path.join(args.output_dir, "pending_gaps_%s_%s.json" % (args.source, args.puzzles[0]))
    if os.path.exists(gaps_path):
        sys.argv = [sys.argv[0], gaps_path]
    else:
        sys.argv = [sys.argv[0]]
    review_main()


def _run_manual_explanations(args):
    """Mode 3: Manually enter definitions, types, and explanations for weak clues."""
    from .review_gaps import manual_entry_phase
    for puzzle in args.puzzles:
        _show_puzzle_summary(args.source, puzzle)
        manual_entry_phase(args.source, puzzle, {})


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
    parser.add_argument("--partials", action="store_true", default=PARTIALS,
                        help="Re-run partial solves (has_solution=2) with fresh API calls")
    parser.add_argument("--mode", type=int, choices=[1, 2, 3], default=None,
                        help="Skip menu: 1=Full Pipeline, 2=DB Additions, 3=Manual Explanations")
    parser.add_argument("--no-review", action="store_true", default=False,
                        help="Skip interactive gap review (for non-interactive/subprocess use)")
    args = parser.parse_args()

    # Single-clue mode: auto-detect source and puzzle from DB.
    # Always overrides source/puzzle — the clue text is the primary selector.
    if args.single_clue:
        conn = sqlite3.connect(CLUES_DB, timeout=30)
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

    mode = args.mode
    if mode is None:
        # Show current status and menu
        print()
        print("=" * 60)
        print("SONNET PIPELINE — %s %s" % (args.source, ", ".join(args.puzzles)))
        print("=" * 60)
        for puzzle in args.puzzles:
            _show_puzzle_summary(args.source, puzzle)
        print()
        print("  1. Run Full Pipeline   (solve → DB additions → re-run → manual)")
        print("  2. DB Additions only   (review/approve reference DB inserts)")
        print("  3. Manual Explanations (enter definition/type/explanation)")
        print()
        try:
            choice = input("Choose [1/2/3]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            sys.exit(0)
        if choice not in ("1", "2", "3"):
            print("Invalid choice.")
            sys.exit(1)
        mode = int(choice)

    if mode == 1:
        _run_full_pipeline(args)
    elif mode == 2:
        _run_db_additions(args)
    elif mode == 3:
        _run_manual_explanations(args)


if __name__ == "__main__":
    main()
