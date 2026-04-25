"""Signature solver waterfall diagnostic — traces the cascade of eliminations.

DRY RUN ONLY. No DB writes, no side effects.

Waterfall stages (each clue is eliminated at the first stage it fails):
  1. DEFINITION  — is any phrase from either end a known definition of the answer?
  2. INDICATOR   — does any catalog pattern find its required indicator in the words?
  3. PLACEMENT   — does any pattern+indicator produce a valid word-to-slot assignment?
  4. LOOKUP      — does any placement find values for all fodder slots from RefDB?
  5. VERIFY      — does any set of values assemble into the answer string?
  6. CONFIDENCE  — does any verified result score >= 80?

Caps flag: if the placement cap (20) or combo cap (50) was reached, the clue
may have been eliminated at the wrong stage.

Cohort support:
    python scripts/sig_diagnostic.py --limit 500 --save-cohort baseline_500
    python scripts/sig_diagnostic.py --load-cohort baseline_500
"""

import argparse
import json
import sqlite3
import sys
import os
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
COHORT_DIR = os.path.join(ROOT, "data", "cohorts")


def _load_cohort(name):
    path = os.path.join(COHORT_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def _save_cohort(name, clue_ids):
    os.makedirs(COHORT_DIR, exist_ok=True)
    path = os.path.join(COHORT_DIR, f"{name}.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(clue_ids, f)
    print(f"Cohort saved: {path} ({len(clue_ids)} clues)")


def _trace_waterfall(wp_words, answer, ref_db, def_pos):
    """Trace a single clue through stages 2-5 of the base matcher.

    Returns: (deepest_stage, caps_hit)
        deepest_stage: 'indicator', 'placement', 'lookup', 'verify'
        caps_hit: True if any cap was reached
    """
    from signature_solver.word_analyzer import analyze_phrases, clean_word
    from signature_solver.base_catalog import (
        BASE_CATALOG, OPERATION_INDICATOR_TYPE, OPERATION_FODDER_TYPES,
    )
    from signature_solver.base_matcher import (
        _generate_span_assignments, _place_spans, _letter_budget_ok,
        MAX_PLACEMENTS_PER_ENTRY, MAX_FODDER_COMBOS, _fodder_type_combos,
        _MockEntry,
    )
    from signature_solver.matcher import (
        _lookup_slot, _verify_combo,
    )
    from signature_solver.positional_catalog import INDICATOR_TOKENS_SET

    answer_clean = answer.upper().replace(" ", "").replace("-", "")
    answer_len = len(answer_clean)
    analyses, phrases = analyze_phrases(wp_words, answer_clean, ref_db)
    n = len(wp_words)

    word_possible = [set(wa.roles.keys()) for wa in analyses]

    word_indicator_map = {}
    phrase_indicator_map = {}
    for i, wa in enumerate(analyses):
        for tok in wa.roles:
            if tok in INDICATOR_TOKENS_SET:
                word_indicator_map.setdefault(tok, []).append(i)
    for (pi, pj), pwa in phrases.items():
        for tok in pwa.roles:
            if tok in INDICATOR_TOKENS_SET:
                phrase_indicator_map.setdefault(tok, []).append((pi, pj))

    # Track deepest stage reached and whether caps were hit
    reached_indicator = False
    reached_placement = False
    reached_lookup = False
    reached_verify = False
    any_caps_hit = False

    for entry in BASE_CATALOG:
        # Quick filters
        if n < entry.min_words:
            continue
        max_words = entry.n_fodder * 4 + entry.n_indicator * 2 + 5
        if n > max_words:
            continue
        if not _letter_budget_ok(entry, answer_len):
            continue

        op = entry.operation
        is_reversal = op in ("reversal", "reversal_charade", "container_reversal")

        # Stage 2: Indicator check
        ind_type_raw = OPERATION_INDICATOR_TYPE.get(op)
        if isinstance(ind_type_raw, list):
            ind_types_to_try = [
                t for t in ind_type_raw
                if word_indicator_map.get(t) or phrase_indicator_map.get(t)
            ]
            if entry.n_indicator > 0 and not ind_types_to_try:
                continue
        else:
            ind_types_to_try = [ind_type_raw] if ind_type_raw else [None]
            if entry.n_indicator > 0 and ind_type_raw:
                if not word_indicator_map.get(ind_type_raw) and not phrase_indicator_map.get(ind_type_raw):
                    continue

        reached_indicator = True

        # Stage 3: Placement
        placements_tried = 0
        for ind_type in ind_types_to_try:
            for spans in _generate_span_assignments(entry.pattern, n):
                if placements_tried >= MAX_PLACEMENTS_PER_ENTRY:
                    any_caps_hit = True
                    break

                for placement in _place_spans(
                    entry.pattern, spans, n, word_possible, phrases, ind_type
                ):
                    placements_tried += 1
                    if placements_tried > MAX_PLACEMENTS_PER_ENTRY:
                        any_caps_hit = True
                        break

                    reached_placement = True

                    # Stage 4: Lookup
                    f_slots = []
                    i_slots = []
                    used = set()
                    for idx, ((start, span), tok_type) in enumerate(zip(placement, entry.pattern)):
                        indices = list(range(start, start + span))
                        used.update(indices)
                        if tok_type == 'F':
                            f_slots.append((idx, start, span))
                        else:
                            i_slots.append((idx, start, span))

                    ind_assignment = {}
                    if ind_type and i_slots:
                        for _, start, span in i_slots:
                            if span == 1:
                                ind_assignment[ind_type] = start
                            else:
                                ind_assignment[ind_type] = (start, start + span)

                    leftover = [i for i in range(n) if i not in used]
                    fodder_types = OPERATION_FODDER_TYPES.get(op, ['SYN_F', 'ABR_F'])
                    combined = set(ind_assignment.keys())
                    mock_entry = _MockEntry(op, combined)

                    slot_type_values = []
                    all_slots_have_values = True
                    for _, start, span in f_slots:
                        indices = list(range(start, start + span))
                        type_vals = []
                        for ftype in fodder_types:
                            vals = _lookup_slot(
                                indices, ftype, span, wp_words, analyses,
                                answer_clean, answer_len, is_reversal, ref_db,
                                clean_word, mock_entry
                            )
                            if vals:
                                type_vals.append((ftype, vals))
                        if not type_vals:
                            all_slots_have_values = False
                            break
                        slot_type_values.append(type_vals)

                    if not all_slots_have_values:
                        continue

                    reached_lookup = True

                    # Stage 5: Verify
                    combo_count = 0
                    for type_combo in _fodder_type_combos(slot_type_values):
                        combo_count += 1
                        if combo_count > MAX_FODDER_COMBOS:
                            any_caps_hit = True
                            break

                        slot_values = []
                        slot_word_groups = []
                        for i, (ftype, vals) in enumerate(type_combo):
                            _, start, span = f_slots[i]
                            indices = list(range(start, start + span))
                            slot_values.append(vals)
                            slot_word_groups.append((indices, ftype, span))

                        result = _verify_combo(op, mock_entry, slot_values,
                                               slot_word_groups, ind_assignment,
                                               leftover, answer_clean, wp_words)
                        if result is not None:
                            reached_verify = True
                            return 'verify', False  # shouldn't happen (solver would have solved)

    # Return deepest stage reached
    if reached_lookup:
        return 'verify', any_caps_hit     # got to lookup, failed at verify
    if reached_placement:
        return 'lookup', any_caps_hit     # got to placement, failed at lookup
    if reached_indicator:
        return 'placement', any_caps_hit  # got to indicator, failed at placement
    return 'indicator', any_caps_hit      # never passed indicator check


def main():
    parser = argparse.ArgumentParser(description="Signature solver waterfall diagnostic (dry run)")
    parser.add_argument("--source", type=str, default=None)
    parser.add_argument("--since", type=str, default="2026-04-01")
    parser.add_argument("--limit", type=int, default=500)
    parser.add_argument("--save-cohort", type=str, default=None)
    parser.add_argument("--load-cohort", type=str, default=None)
    args = parser.parse_args()

    print("Loading RefDB...")
    from signature_solver.db import RefDB
    ref_db = RefDB()

    from signature_solver.solver import (
        extract_definition_candidates, solve, _normalize_clue
    )
    from signature_solver.confidence import score_result

    conn = sqlite3.connect(CLUES_DB, timeout=30)

    if args.load_cohort:
        clue_ids = _load_cohort(args.load_cohort)
        placeholders = ",".join("?" for _ in clue_ids)
        rows = conn.execute(f"""
            SELECT id, source, puzzle_number, clue_number, clue_text, answer
            FROM clues WHERE id IN ({placeholders})
            ORDER BY source, publication_date DESC, CAST(clue_number AS INTEGER)
        """, clue_ids).fetchall()
        print(f"Loaded cohort '{args.load_cohort}': {len(rows)} clues")
    else:
        sources = (args.source,) if args.source else ('telegraph', 'times')
        placeholders = ",".join("?" for _ in sources)
        query = f"""
            SELECT id, source, puzzle_number, clue_number, clue_text, answer
            FROM clues
            WHERE source IN ({placeholders})
              AND answer IS NOT NULL AND answer != ''
              AND clue_text IS NOT NULL
              AND publication_date >= ?
            ORDER BY publication_date DESC, source, CAST(clue_number AS INTEGER)
            LIMIT ?
        """
        params = list(sources) + [args.since, args.limit]
        rows = conn.execute(query, params).fetchall()

        if args.save_cohort:
            _save_cohort(args.save_cohort, [r[0] for r in rows])

    conn.close()

    total = len(rows)
    print(f"Processing {total} clues...\n")

    # Waterfall counters
    stage_eliminated = {
        "definition": 0,
        "indicator": 0,
        "placement": 0,
        "lookup": 0,
        "verify": 0,
        "confidence": 0,
    }
    solved_high = 0
    caps_hit_at = {s: 0 for s in stage_eliminated}  # how many had caps hit per stage

    examples = {s: [] for s in stage_eliminated}
    examples["solved"] = []
    MAX_EX = 5

    t0 = time.time()

    for i, (cid, source, pnum, cnum, clue_text, answer) in enumerate(rows):
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{total} ({time.time()-t0:.1f}s)")

        answer_clean = answer.upper().replace(" ", "").replace("-", "")
        clue_words = _normalize_clue(clue_text).strip().split()
        label = f"{source} #{pnum} {cnum}: {clue_text} = {answer}"

        # Stage 1: Definition
        candidates = extract_definition_candidates(clue_words, answer_clean, ref_db)
        if not candidates:
            stage_eliminated["definition"] += 1
            if len(examples["definition"]) < MAX_EX:
                examples["definition"].append(f"  {label}")
            continue

        # Try the full solver first — if it solves, record and move on
        best_confidence = -1
        solved = False
        for def_phrase, wp_words in candidates:
            if clue_words[:len(clue_words) - len(wp_words)] == clue_words[:len(def_phrase.split())]:
                def_pos = 'start'
            else:
                def_pos = 'end'
            sr = solve(wp_words, answer_clean, ref_db, min_confidence=0, def_pos=def_pos)
            if sr.solved and sr.confidence > best_confidence:
                best_confidence = sr.confidence
                solved = True
            if sr.solved and sr.confidence >= 80:
                break

        if solved and best_confidence >= 80:
            solved_high += 1
            continue

        if solved and best_confidence < 80:
            stage_eliminated["confidence"] += 1
            if len(examples["confidence"]) < MAX_EX:
                examples["confidence"].append(f"  {label} (conf={best_confidence})")
            continue

        # Clue failed — trace through stages 2-5 to find where
        def_phrase, wp_words = candidates[0]
        if clue_words[:len(clue_words) - len(wp_words)] == clue_words[:len(def_phrase.split())]:
            def_pos = 'start'
        else:
            def_pos = 'end'

        failed_at, caps = _trace_waterfall(wp_words, answer_clean, ref_db, def_pos)
        stage_eliminated[failed_at] += 1
        if caps:
            caps_hit_at[failed_at] += 1
        if len(examples[failed_at]) < MAX_EX:
            cap_flag = " [CAPS]" if caps else ""
            examples[failed_at].append(f"  {label}{cap_flag}")

    elapsed = time.time() - t0

    # Waterfall report
    print("\n" + "=" * 70)
    print(f"SIGNATURE SOLVER WATERFALL — {total} clues in {elapsed:.1f}s")
    print("=" * 70)

    remaining = total
    print(f"\n  {'Stage':<14} {'Eliminated':>10} {'Caps hit':>10} {'Continuing':>10}")
    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'Start':<14} {'':>10} {'':>10} {remaining:>10}")

    for stage in ["definition", "indicator", "placement", "lookup", "verify", "confidence"]:
        elim = stage_eliminated[stage]
        caps = caps_hit_at[stage]
        remaining -= elim
        caps_str = str(caps) if caps > 0 else ""
        print(f"  {stage.upper():<14} {elim:>10} {caps_str:>10} {remaining:>10}")

    print(f"  {'-'*14} {'-'*10} {'-'*10} {'-'*10}")
    print(f"  {'SOLVED HIGH':<14} {'':>10} {'':>10} {solved_high:>10}")

    # Percentages
    print(f"\n  Solve rate: {solved_high}/{total} = {100*solved_high//total}%")
    total_caps = sum(caps_hit_at.values())
    if total_caps:
        print(f"  Caps hit: {total_caps} clues (may have been eliminated at wrong stage)")

    # Examples per stage
    for stage in ["definition", "indicator", "placement", "lookup", "verify", "confidence"]:
        if examples[stage]:
            print(f"\n--- Eliminated at {stage.upper()} ({stage_eliminated[stage]} clues) ---")
            for ex in examples[stage]:
                print(ex)

    print()


if __name__ == "__main__":
    main()
