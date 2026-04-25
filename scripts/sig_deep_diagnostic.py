"""Deep signature solver diagnostic — explains exactly why each clue fails.

DRY RUN ONLY. No DB writes.

For each failing clue that has a structured explanation, determines
the specific reason it fails by checking:

  1. Does the required definition exist in RefDB?
  2. Does a catalog pattern exist for this wordplay structure?
  3. For each piece: is the synonym/abbreviation findable by _lookup_slot?
  4. Can the pieces actually assemble into the answer?

Reports concrete, actionable failure reasons.

Usage:
    python scripts/sig_deep_diagnostic.py --load-cohort baseline_500
"""

import argparse
import json
import sqlite3
import sys
import os
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
COHORT_DIR = os.path.join(ROOT, "data", "cohorts")


def _load_cohort(name):
    path = os.path.join(COHORT_DIR, f"{name}.json")
    with open(path, encoding="utf-8") as f:
        return json.load(f)


# Map structured_explanations wordplay_type to catalog operation names
WTYPE_TO_OPS = {
    "charade": ["charade"],
    "container": ["container", "container_charade"],
    "anagram": ["anagram", "anagram_charade", "anagram_plus"],
    "reversal": ["reversal", "reversal_charade"],
    "hidden": ["hidden"],
    "hidden_reversed": ["hidden_reversed"],
    "deletion": ["deletion", "trim", "trim_charade"],
    "homophone": ["homophone"],
    "double_definition": ["synonym"],  # DD uses synonym catalog entry
    "cryptic_definition": [],  # no catalog pattern
    "acrostic": ["acrostic", "positional_charade"],
    "alternate_letters": ["alternate", "positional_charade"],
    "spoonerism": [],  # handled separately
}


def _count_pieces(pieces):
    """Count fodder pieces and indicator pieces from a structured explanation."""
    n_fodder = 0
    n_mechanical = 0  # first_letter, last_letter, alternate, etc.
    mechanisms = []
    for p in pieces:
        mech = p.get("type", p.get("mechanism", ""))
        mechanisms.append(mech)
        if mech in ("synonym", "abbreviation", "sound_of"):
            n_fodder += 1
        elif mech in ("first_letter", "last_letter", "alternate_letters",
                       "core_letters", "even_letters", "odd_letters"):
            n_mechanical += 1
        elif mech in ("literal", "anagram_fodder", "hidden", "reversal", "deletion"):
            n_fodder += 1
    return n_fodder, n_mechanical, mechanisms


def _catalog_has_pattern(wtype, n_pieces, catalog_entries):
    """Check if a catalog pattern exists for this wordplay type with this many pieces."""
    ops = WTYPE_TO_OPS.get(wtype, [])
    for entry in catalog_entries:
        if entry["operation"] in ops:
            if entry["n_fodder"] == n_pieces or entry["n_fodder"] >= n_pieces:
                return True
    return False


def _check_piece_in_refdb(piece, ref_db):
    """Check if a piece's data exists in RefDB.
    Returns (found, gap_type, detail)
    """
    ptype = piece.get("type", piece.get("mechanism", ""))
    fodder = (piece.get("fodder") or piece.get("clue_word") or "").lower().strip()
    yields = (piece.get("yields") or piece.get("letters") or "").upper().strip()

    if not fodder or not yields:
        return True, None, "empty"

    if ptype == "synonym":
        for v in ref_db._word_variants(fodder):
            if yields in ref_db.get_synonyms(v):
                return True, None, f"'{fodder}'={yields} found"
        return False, "synonym", f"'{fodder}'={yields} MISSING"

    elif ptype == "abbreviation":
        for v in ref_db._word_variants(fodder):
            if yields in ref_db.get_abbreviations(v):
                return True, None, f"'{fodder}'={yields} found"
        return False, "abbreviation", f"'{fodder}'={yields} MISSING"

    elif ptype == "sound_of":
        if yields in ref_db.get_homophones(fodder):
            return True, None, f"'{fodder}'={yields} found"
        return False, "homophone", f"'{fodder}'={yields} MISSING"

    # Mechanical types don't need DB lookups
    return True, None, f"{ptype} (mechanical)"


def main():
    parser = argparse.ArgumentParser(description="Deep sig diagnostic (dry run)")
    parser.add_argument("--load-cohort", type=str, required=True)
    args = parser.parse_args()

    print("Loading RefDB...")
    from signature_solver.db import RefDB
    ref_db = RefDB()

    from signature_solver.solver import (
        extract_definition_candidates, solve, _normalize_clue
    )

    # Load catalog entries as dicts for pattern checking
    with open(os.path.join(ROOT, "data", "base_catalog.json")) as f:
        catalog_raw = json.load(f)
    catalog_entries = []
    for item in catalog_raw:
        item["n_fodder"] = sum(1 for t in item["pattern"] if t == "F")
        item["n_indicator"] = sum(1 for t in item["pattern"] if t == "I")
        catalog_entries.append(item)

    conn = sqlite3.connect(CLUES_DB, timeout=30)
    clue_ids = _load_cohort(args.load_cohort)
    placeholders = ",".join("?" for _ in clue_ids)
    rows = conn.execute(f"""
        SELECT c.id, c.source, c.puzzle_number, c.clue_number, c.clue_text,
               c.answer, se.definition_text, se.wordplay_types, se.components
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE c.id IN ({placeholders})
        ORDER BY c.source, c.publication_date DESC, CAST(c.clue_number AS INTEGER)
    """, clue_ids).fetchall()
    conn.close()

    total = len(rows)
    print(f"Loaded {total} clues\n")

    # Pass 1: identify failures
    print("--- Pass 1: Running solver ---")
    failures = []
    solved = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        cid = row[0]
        clue_text, answer = row[4], row[5]
        if (i + 1) % 100 == 0:
            print(f"  ... {i+1}/{total} ({time.time()-t0:.1f}s)")

        answer_clean = answer.upper().replace(" ", "").replace("-", "")
        clue_words = _normalize_clue(clue_text).strip().split()
        candidates = extract_definition_candidates(clue_words, answer_clean, ref_db)

        is_solved = False
        if candidates:
            for def_phrase, wp_words in candidates:
                dp = 'start' if clue_words[:len(def_phrase.split())] == clue_words[:len(clue_words) - len(wp_words)] else 'end'
                sr = solve(wp_words, answer_clean, ref_db, min_confidence=0, def_pos=dp)
                if sr.solved and sr.confidence >= 80:
                    is_solved = True
                    break

        if is_solved:
            solved += 1
        else:
            failures.append(row)

    print(f"  Solved: {solved}, Failed: {len(failures)}\n")

    # Pass 2: diagnose each failure
    print(f"--- Pass 2: Diagnosing {len(failures)} failures ---\n")

    reasons = Counter()
    reason_examples = {}
    missing_synonyms = Counter()
    missing_abbreviations = Counter()
    missing_definitions = Counter()
    MAX_EX = 5

    for row in failures:
        cid, source, pnum, cnum, clue_text, answer = row[:6]
        se_def, se_types, se_components = row[6], row[7], row[8]
        label = f"{source} #{pnum} {cnum}: {clue_text} = {answer}"
        answer_clean = answer.upper().replace(" ", "").replace("-", "")
        clue_words = _normalize_clue(clue_text).strip().split()

        # No definition found by solver?
        candidates = extract_definition_candidates(clue_words, answer_clean, ref_db)
        if not candidates:
            reason = "1. Definition not in RefDB"
            reasons[reason] += 1
            reason_examples.setdefault(reason, [])
            if len(reason_examples[reason]) < MAX_EX:
                if se_def:
                    reason_examples[reason].append(f"  {label}\n    need: '{se_def}' = {answer_clean}")
                    missing_definitions[(se_def.lower(), answer_clean)] += 1
                else:
                    reason_examples[reason].append(f"  {label}")
            continue

        # Has structured explanation to check against?
        if not se_components:
            reason = "7. No structured explanation (can't diagnose)"
            reasons[reason] += 1
            reason_examples.setdefault(reason, [])
            if len(reason_examples[reason]) < MAX_EX:
                reason_examples[reason].append(f"  {label}")
            continue

        try:
            comp = json.loads(se_components)
        except (json.JSONDecodeError, TypeError):
            reasons["7. No structured explanation (can't diagnose)"] += 1
            continue

        pieces = comp.get("ai_pieces", comp) if isinstance(comp, dict) else comp
        if not isinstance(pieces, list):
            reasons["7. No structured explanation (can't diagnose)"] += 1
            continue

        try:
            wtypes = json.loads(se_types) if se_types else []
        except (json.JSONDecodeError, TypeError):
            wtypes = []
        wtype = wtypes[0] if wtypes else "unknown"

        # Check 1: is this a wordplay type the solver doesn't handle?
        if wtype in ("cryptic_definition", "spoonerism"):
            reason = f"2. Wordplay type not supported: {wtype}"
            reasons[reason] += 1
            reason_examples.setdefault(reason, [])
            if len(reason_examples[reason]) < MAX_EX:
                reason_examples[reason].append(f"  {label}")
            continue

        if wtype == "double_definition":
            reason = "3. Double definition (solver has limited DD support)"
            reasons[reason] += 1
            reason_examples.setdefault(reason, [])
            if len(reason_examples[reason]) < MAX_EX:
                reason_examples[reason].append(f"  {label}")
            continue

        # Check 2: are any pieces missing from RefDB?
        missing = []
        for p in pieces:
            found, gap_type, detail = _check_piece_in_refdb(p, ref_db)
            if not found:
                missing.append((gap_type, detail, p))

        if missing:
            gap_types = set(g for g, _, _ in missing)
            reason = f"4. Missing from RefDB: {', '.join(sorted(gap_types))}"
            reasons[reason] += 1
            reason_examples.setdefault(reason, [])
            if len(reason_examples[reason]) < MAX_EX:
                details = "; ".join(d for _, d, _ in missing)
                reason_examples[reason].append(f"  {label}\n    {details}")
            for gap_type, detail, p in missing:
                fodder = (p.get("fodder") or p.get("clue_word") or "").lower().strip()
                yields = (p.get("yields") or p.get("letters") or "").upper().strip()
                if gap_type == "synonym":
                    missing_synonyms[(fodder, yields)] += 1
                elif gap_type == "abbreviation":
                    missing_abbreviations[(fodder, yields)] += 1
            continue

        # Check 3: does the catalog have a matching pattern?
        n_fodder, n_mechanical, mechanisms = _count_pieces(pieces)
        n_total_pieces = n_fodder + n_mechanical
        has_positional = n_mechanical > 0
        has_anagram = any(m in ("anagram_fodder",) for m in mechanisms)

        # Determine what operation the solver needs
        if has_positional and wtype == "charade":
            needed_ops = ["positional_charade", "charade"]
        elif has_anagram and wtype in ("charade", "anagram"):
            needed_ops = ["anagram_charade", "anagram_plus", "anagram"]
        else:
            needed_ops = WTYPE_TO_OPS.get(wtype, [])

        pattern_exists = False
        for entry in catalog_entries:
            if entry["operation"] in needed_ops:
                # Check if piece count is compatible
                if entry["n_fodder"] >= n_total_pieces or entry["n_fodder"] == n_total_pieces:
                    pattern_exists = True
                    break

        if not pattern_exists:
            reason = f"5. No catalog pattern for {wtype} with {n_total_pieces} pieces"
            if has_positional:
                reason += f" (includes positional: {[m for m in mechanisms if m not in ('synonym', 'abbreviation', 'literal')]})"
            reasons[reason] += 1
            reason_examples.setdefault(reason, [])
            if len(reason_examples[reason]) < MAX_EX:
                reason_examples[reason].append(f"  {label}\n    pieces: {mechanisms}")
            continue

        # Everything looks present: definition found, data in DB, catalog pattern exists
        # Yet the solver still fails. This is caps/search-order/assembly issue.
        reason = f"6. All data present, pattern exists - caps or assembly issue [{wtype}]"
        reasons[reason] += 1
        reason_examples.setdefault(reason, [])
        if len(reason_examples[reason]) < MAX_EX:
            reason_examples[reason].append(f"  {label}\n    {n_total_pieces} pieces: {mechanisms}")

    # Report
    print("=" * 70)
    print(f"WHY DO {len(failures)} CLUES FAIL? (of {total} total, {solved} solved)")
    print("=" * 70)

    print(f"\n  {'Reason':<65} {'Count':>5} {'%':>5}")
    print(f"  {'-'*65} {'-'*5} {'-'*5}")
    analysable = len(failures)
    for reason, count in sorted(reasons.items()):
        pct = 100 * count // analysable if analysable else 0
        print(f"  {reason:<65} {count:>5} {pct:>4}%")

    print(f"  {'-'*65} {'-'*5} {'-'*5}")
    print(f"  {'Total':<65} {analysable:>5}")

    # Examples
    for reason in sorted(reasons.keys()):
        if reason_examples.get(reason):
            print(f"\n--- {reason} ---")
            for ex in reason_examples[reason]:
                print(ex)

    # Missing data lists
    if missing_synonyms:
        print(f"\n--- Missing Synonyms ({len(missing_synonyms)} unique) ---")
        for (fodder, yields), count in missing_synonyms.most_common(15):
            print(f"  '{fodder}' = {yields}" + (f" (x{count})" if count > 1 else ""))

    if missing_abbreviations:
        print(f"\n--- Missing Abbreviations ({len(missing_abbreviations)} unique) ---")
        for (fodder, yields), count in missing_abbreviations.most_common(15):
            print(f"  '{fodder}' = {yields}" + (f" (x{count})" if count > 1 else ""))

    if missing_definitions:
        print(f"\n--- Missing Definitions ({len(missing_definitions)} unique) ---")
        for (defn, ans), count in missing_definitions.most_common(15):
            print(f"  '{defn}' = {ans}" + (f" (x{count})" if count > 1 else ""))

    print()


if __name__ == "__main__":
    main()
