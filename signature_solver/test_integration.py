"""Integration test: signature solver + API enrichment pipeline.

Flow:
  Phase 1: Run signature solver (zero API calls)
  Phase 2: For unsolved clues, call API to discover pieces,
           inject them into a cloned RefDB, re-run signature solver
  Phase 3: Report what still needs pure AI explanation

This is READ-ONLY — no database writes.
"""

import copy
import json
import os
import sqlite3
import sys
import time


def load_puzzle(source, puzzle_number):
    """Load clues from clues_master.db (read-only)."""
    db_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        "data", "clues_master.db"
    )
    conn = sqlite3.connect(db_path, timeout=30)
    rows = conn.execute('''
        SELECT clue_text, answer, definition, wordplay_type,
               clue_number, direction
        FROM clues
        WHERE source = ? AND puzzle_number = ?
        AND answer IS NOT NULL
        ORDER BY clue_number, direction
    ''', (source, puzzle_number)).fetchall()
    conn.close()
    return rows


def enrich_refdb(ref_db, api_pieces, definition=None, answer=None):
    """Clone RefDB and inject API-discovered pieces. Returns new RefDB.

    Only injects synonyms, abbreviations, and definitions —
    the three things the signature solver needs from the DB.
    Does NOT modify the original RefDB.
    """
    enriched = copy.copy(ref_db)
    # Deep-copy only the dicts we'll mutate
    enriched.synonyms = dict(ref_db.synonyms)
    enriched.abbreviations = dict(ref_db.abbreviations)

    injected = []

    for piece in api_pieces:
        word = piece.get("word", "").lower().strip(".,;:!?\"'()-")
        letters = piece.get("letters", "").upper().replace(" ", "").replace("-", "")
        role = piece.get("role", "")

        if not word or not letters:
            continue

        if role == "synonym":
            if word not in enriched.synonyms:
                enriched.synonyms[word] = []
            else:
                enriched.synonyms[word] = list(enriched.synonyms[word])
            if letters not in enriched.synonyms[word]:
                enriched.synonyms[word].append(letters)
                injected.append(f"  SYN: {word} -> {letters}")

        elif role == "abbreviation":
            if word not in enriched.abbreviations:
                enriched.abbreviations[word] = []
            else:
                enriched.abbreviations[word] = list(enriched.abbreviations[word])
            if letters not in enriched.abbreviations[word]:
                enriched.abbreviations[word].append(letters)
                injected.append(f"  ABR: {word} -> {letters}")

        elif role == "homophone":
            # Inject as synonym — the signature solver checks synonyms for fodder
            if word not in enriched.synonyms:
                enriched.synonyms[word] = []
            else:
                enriched.synonyms[word] = list(enriched.synonyms[word])
            if letters not in enriched.synonyms[word]:
                enriched.synonyms[word].append(letters)
                injected.append(f"  HOM->SYN: {word} -> {letters}")

    # Inject definition -> answer mapping if provided
    if definition and answer:
        def_lower = definition.lower().strip(".,;:!?\"'()-")
        ans_upper = answer.upper().replace(" ", "").replace("-", "")
        if def_lower not in enriched.synonyms:
            enriched.synonyms[def_lower] = []
        else:
            enriched.synonyms[def_lower] = list(enriched.synonyms[def_lower])
        if ans_upper not in enriched.synonyms[def_lower]:
            enriched.synonyms[def_lower].append(ans_upper)
            injected.append(f"  DEF: {def_lower} -> {ans_upper}")

    return enriched, injected


def run_integration_test(puzzle_number, source='telegraph'):
    from .db import RefDB
    from .solver import solve_clue
    from .api_solver import api_solve

    rows = load_puzzle(source, puzzle_number)
    if not rows:
        print(f"No clues found for {source} puzzle {puzzle_number}")
        return

    print("Loading reference database...")
    ref_db = RefDB()
    print(f"\nIntegration test: {source} puzzle {puzzle_number} ({len(rows)} clues)")
    print(f"{'='*70}\n")

    # ============================================================
    # Phase 1: Signature solver alone (zero API calls)
    # ============================================================
    print("PHASE 1: Signature solver (mechanical, no API)")
    print("-" * 50)

    tier1_solved = []   # (row, sr) — HIGH confidence
    tier1_unsolved = [] # (row, sr) — needs help

    for row in rows:
        clue_text, answer, definition, wtype, clue_num, direction = row
        label = f"{clue_num}{direction[0].upper()}"
        answer_clean = answer.upper().replace(" ", "").replace("-", "")

        sr = solve_clue(clue_text, answer_clean, ref_db)

        if sr.high_confidence:
            tier1_solved.append((row, sr))
            r = sr.result
            print(f"  HIGH [{sr.confidence:3d}]  {label}: {answer}")
            print(f"        {r.explanation_parts[0]}")
        else:
            tier1_unsolved.append((row, sr))

    print(f"\n  Phase 1 result: {len(tier1_solved)} HIGH / {len(rows)} total")
    print(f"  Remaining: {len(tier1_unsolved)} clues need API help\n")

    if not tier1_unsolved:
        print("All clues solved mechanically!")
        return

    # ============================================================
    # Phase 2: API enrichment + re-solve
    # ============================================================
    print("PHASE 2: API enrichment + signature re-solve")
    print("-" * 50)

    tier2_solved = []    # (row, sr, api_parsed, injected) — solved after enrichment
    tier3_unsolved = []  # (row, sr, api_parsed) — still unsolved
    total_tokens_in = 0
    total_tokens_out = 0

    for row, sr in tier1_unsolved:
        clue_text, answer, definition, wtype, clue_num, direction = row
        label = f"{clue_num}{direction[0].upper()}"
        answer_clean = answer.upper().replace(" ", "").replace("-", "")

        # Get definition from mechanical solver if available
        mech_def = getattr(sr, 'definition', None)

        print(f"\n  {label}: {answer} ({wtype or '?'})")
        print(f"        Clue: {clue_text}")

        # Call API with evidence from mechanical solver
        try:
            parsed, tok_in, tok_out = api_solve(
                clue_text, answer_clean,
                mech_def or definition or '',
                sr, ref_db
            )
            total_tokens_in += tok_in
            total_tokens_out += tok_out
        except Exception as e:
            print(f"        API error: {e}")
            tier3_unsolved.append((row, sr, None))
            continue

        if not parsed:
            print(f"        API returned no valid response")
            tier3_unsolved.append((row, sr, None))
            continue

        api_type = parsed.get("wordplay_type", "?")
        api_conf = parsed.get("confidence", "?")
        pieces = parsed.get("pieces", [])
        assembly = parsed.get("assembly", "")

        print(f"        API says: {api_type} ({api_conf})")
        print(f"        Assembly: {assembly}")

        # Extract the definition the API identified (if any)
        # Use the definition from DB or the one the API would identify
        api_def = mech_def or definition

        # Inject API discoveries into a cloned RefDB
        enriched_db, injected = enrich_refdb(
            ref_db, pieces,
            definition=api_def,
            answer=answer_clean
        )

        if injected:
            print(f"        Injected into DB:")
            for inj in injected:
                print(f"          {inj}")
        else:
            print(f"        Nothing new to inject")

        # Re-run signature solver with enriched DB
        sr2 = solve_clue(clue_text, answer_clean, enriched_db)

        if sr2.high_confidence:
            tier2_solved.append((row, sr2, parsed, injected))
            r = sr2.result
            print(f"        -> SIGNATURE SOLVED [{sr2.confidence:3d}]")
            print(f"           {r.explanation_parts[0]}")
            for reason, delta in sr2.confidence_reasons:
                sign = "+" if delta >= 0 else ""
                print(f"           {sign}{delta} {reason}")
        else:
            tier3_unsolved.append((row, sr2, parsed))
            if sr2.solved:
                print(f"        -> Signature found match [{sr2.confidence:3d}] but not HIGH")
            else:
                print(f"        -> Signature still cannot solve")

        time.sleep(0.3)  # Rate limit

    # ============================================================
    # Report
    # ============================================================
    print(f"\n{'='*70}")
    print(f"INTEGRATION TEST RESULTS: {source} puzzle {puzzle_number}")
    print(f"{'='*70}")

    total = len(rows)
    n1 = len(tier1_solved)
    n2 = len(tier2_solved)
    n3 = len(tier3_unsolved)

    print(f"\n  Total clues:                    {total}")
    print(f"  Phase 1 (signature alone):      {n1:3d}  ({100*n1/total:4.0f}%)  — zero API calls")
    print(f"  Phase 2 (API + signature):      {n2:3d}  ({100*n2/total:4.0f}%)  — {len(tier1_unsolved)} API calls")
    print(f"  Phase 3 (needs AI explanation):  {n3:3d}  ({100*n3/total:4.0f}%)  — use production solver")
    print(f"\n  Signature explanation coverage:  {n1+n2}/{total} ({100*(n1+n2)/total:.0f}%)")
    print(f"  API calls used:                 {len(tier1_unsolved)} (saved {n1} vs calling all)")
    print(f"  API tokens: {total_tokens_in} in / {total_tokens_out} out")

    # Show tier 2 clues — these are the wins
    if tier2_solved:
        print(f"\n  PHASE 2 WINS — clues now with signature-quality explanations:")
        for row, sr, parsed, injected in tier2_solved:
            clue_text, answer, definition, wtype, clue_num, direction = row
            label = f"{clue_num}{direction[0].upper()}"
            r = sr.result
            print(f"    {label}: {answer} [{sr.confidence}]")
            print(f"      {r.explanation_parts[0]}")
            print(f"      DB additions needed: {len(injected)}")
            for inj in injected:
                print(f"        {inj}")

    # Show tier 3 — still need AI
    if tier3_unsolved:
        print(f"\n  PHASE 3 — still need AI explanation:")
        for row, sr, parsed in tier3_unsolved:
            clue_text, answer, definition, wtype, clue_num, direction = row
            label = f"{clue_num}{direction[0].upper()}"
            reason = ""
            if parsed:
                reason = f" (API: {parsed.get('wordplay_type', '?')})"
            print(f"    {label}: {answer} ({wtype or '?'}){reason}")


if __name__ == "__main__":
    puzzle = int(sys.argv[1]) if len(sys.argv) > 1 else 31185
    source = sys.argv[2] if len(sys.argv) > 2 else 'telegraph'
    run_integration_test(puzzle, source)
