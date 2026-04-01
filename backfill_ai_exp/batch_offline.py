"""Offline batch solver — writes results to JSONL without locking the DB.

Reads clue list once at startup, loads RefDB into memory, then closes all
DB connections. Processes clues entirely in-memory, writing results to
data/batch_results.jsonl one line at a time (append-safe, killable).

After completion, runs Haiku validation on enrichment proposals.

Upload results later with batch_upload.py.

Usage:
    python scripts/batch_offline.py --limit 1000                # 1k test batch
    python scripts/batch_offline.py --limit 100000 --skip 100000  # skip first 100k
    python scripts/batch_offline.py --limit 100000 --skip 100000 --source telegraph
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time
from collections import Counter

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from signature_solver.solver import solve_clue, extract_definition_candidates, _normalize_clue
from signature_solver.db import RefDB
from signature_solver.word_analyzer import analyze_phrases
from sonnet_pipeline.solver import try_hidden, try_double_definition, clean
from sonnet_pipeline.sig_adapter import build_ai_pieces, build_assembly_dict, sig_explain
from sonnet_pipeline.report import _highlight_hidden

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
CRYPTIC_DB = os.path.join(ROOT, "data", "cryptic_new.db")
RESULTS_PATH = os.path.join(ROOT, "data", "batch_results.jsonl")
PROPOSALS_PATH = os.path.join(ROOT, "data", "batch_enrichment_proposals.jsonl")
VALIDATED_PATH = os.path.join(ROOT, "data", "batch_enrichment_validated.jsonl")


def load_clues(limit, skip, source_filter):
    """Load clue list from DB, then close connection immediately."""
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    conn.row_factory = sqlite3.Row

    where = "answer IS NOT NULL AND answer != '' AND clue_text IS NOT NULL AND clue_text != ''"
    where += " AND (has_solution IS NULL OR has_solution = 0)"
    where += " AND publication_date < date('now')"
    params = []
    if source_filter:
        where += " AND source = ?"
        params.append(source_filter)

    # Use LIMIT + OFFSET to skip already-processed clues
    rows = conn.execute("""
        SELECT id, source, puzzle_number, clue_number, clue_text, answer, enumeration
        FROM clues WHERE %s
        ORDER BY publication_date DESC, source, CAST(puzzle_number AS INTEGER) DESC
        LIMIT ? OFFSET ?
    """ % where, params + [limit, skip]).fetchall()

    # Convert to plain dicts so we don't need the connection
    clues = []
    for r in rows:
        clues.append({
            "id": r["id"],
            "source": r["source"],
            "puzzle_number": r["puzzle_number"],
            "clue_number": r["clue_number"],
            "clue_text": r["clue_text"],
            "answer": r["answer"],
            "enumeration": r["enumeration"],
        })

    conn.close()
    return clues


def extract_enrichment(clue_text, answer_clean, ref_db):
    """Extract proposed synonym pairs from a single clue.

    Copied from batch_enrichment.py — same logic, no DB writes.
    """
    proposals = []
    answer_up = answer_clean.upper()

    clue_words = _normalize_clue(clue_text).strip().split()
    def_candidates = extract_definition_candidates(clue_words, answer_up, ref_db)

    # 1. Definition enrichment: definition word -> full answer
    for def_phrase, wp_words in def_candidates:
        def_clean = def_phrase.lower().strip(".,;:!?\"'()-").strip()
        existing = answer_up in [s.upper().replace(" ", "").replace("-", "")
                                 for s in ref_db.get_synonyms(def_clean)]
        if not existing:
            proposals.append(("definition", def_clean, answer_up, "def->answer"))

    # 2. Residual enrichment from partial piece matching
    if not def_candidates:
        return proposals

    def_phrase, wp_words = def_candidates[0]
    analyses, phrases = analyze_phrases(wp_words, answer_up, ref_db)

    confirmed = []
    unmatched_indices = []  # track position for contiguity check

    for i, wa in enumerate(analyses):
        best_match = None
        best_len = 0
        for tok, vals in wa.roles.items():
            if tok in ("SYN_F", "ABR_F") and vals:
                for v in vals:
                    if isinstance(v, str):
                        vup = v.upper().replace(" ", "").replace("-", "")
                        if vup and vup in answer_up and len(vup) > best_len:
                            best_match = vup
                            best_len = len(vup)
        if best_match:
            confirmed.append((wa.text, best_match))
        else:
            is_ind = any(t.endswith("_I") or t.startswith("POS_I") for t in wa.roles)
            is_lnk = "LNK" in wa.roles
            if not is_ind and not is_lnk:
                unmatched_indices.append(i)

    if not confirmed or not unmatched_indices:
        return proposals

    residual = answer_up
    for word, val in confirmed:
        if val in residual:
            residual = residual.replace(val, "", 1)

    if not residual:
        return proposals

    # Check unmatched words are contiguous
    is_contiguous = all(
        unmatched_indices[j + 1] == unmatched_indices[j] + 1
        for j in range(len(unmatched_indices) - 1)
    )
    if not is_contiguous:
        return proposals

    # Build the phrase from contiguous unmatched words
    uw = " ".join(analyses[i].text for i in unmatched_indices)
    uw = uw.lower().strip(".,;:!?\"'()-").strip()
    if not uw or len(uw) < 2:
        return proposals

    # Check not already in DB
    existing_syn = residual in [s.upper().replace(" ", "")
                                for s in ref_db.get_synonyms(uw)]
    existing_abr = residual in [a.upper()
                                for a in ref_db.abbreviations.get(uw, [])]
    if not existing_syn and not existing_abr:
        pieces_desc = ", ".join(f"{w}->{v}" for w, v in confirmed)
        proposals.append(("residual", uw, residual,
                          f"answer={answer_up}, confirmed=[{pieces_desc}]"))

    return proposals


def serialize_solve_result(clue_row, sr, answer):
    """Serialize a SolveResult into a dict matching what store_signature_result writes.

    This captures everything needed to replay the DB writes later.
    """
    ai_pieces = build_ai_pieces(sr)
    assembly = build_assembly_dict(sr)
    definition = getattr(sr, "definition", None)

    sig_expl = sig_explain(sr, answer)
    explanation_text = sig_expl["explanation"]
    wordplay_types = sig_expl["wordplay_types"]
    wordplay_type = ", ".join(wordplay_types) if wordplay_types else "unknown"

    components = {
        "ai_pieces": ai_pieces,
        "assembly": assembly,
        "wordplay_types": wordplay_types,
        "sig_explanation": explanation_text,
    }

    confidence = sr.confidence / 100.0

    # Definition position in clue text
    def_start = None
    def_end = None
    if definition:
        idx = clue_row["clue_text"].lower().find(definition.lower())
        if idx >= 0:
            def_start = idx
            def_end = idx + len(definition)

    # Determine has_solution: full (1) or partial (2)
    has_def = bool(definition)
    has_type = bool(wordplay_type and wordplay_type != "unknown")
    has_expl = bool(explanation_text) or bool(ai_pieces)

    if has_def and has_type and has_expl:
        has_solution = 1
    elif has_def or has_type:
        has_solution = 2
    else:
        has_solution = 2  # solved but minimal data

    auto_reviewed = 1 if sr.confidence >= 80 else 0

    return {
        "definition": definition,
        "wordplay_type": wordplay_type,
        "ai_explanation": explanation_text,
        "has_solution": has_solution,
        "reviewed": auto_reviewed,
        "confidence": confidence,
        "components": components,
        "wordplay_types": wordplay_types,
        "definition_start": def_start,
        "definition_end": def_end,
        "model_version": "signature_solver_v1",
        "source": clue_row["source"],
        "puzzle_number": clue_row["puzzle_number"],
        "clue_number": clue_row["clue_number"],
    }


def run_batch(clues, ref_db, results_path):
    """Process clues, write results to JSONL. No DB writes."""

    all_proposals = []
    enrichment_clue_ids = []
    solved = 0
    skipped = 0
    attempted = 0
    t0 = time.time()

    with open(results_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(clues):
            cid = row["id"]
            clue = row["clue_text"]
            answer = row["answer"]
            answer_clean = clean(answer)
            if not answer_clean or len(answer_clean) < 2:
                skipped += 1
                continue

            # Skip cross-references
            if re.search(r"\b\d+\s*(?:across|down|ac|dn)\b", clue, re.IGNORECASE):
                f.write(json.dumps({"clue_id": cid, "action": "skipped_crossref"}) + "\n")
                skipped += 1
                continue

            # Check hidden words — store result
            hidden_result = try_hidden(clue, answer_clean)
            if hidden_result:
                is_reversed = "reversed" in hidden_result.get("op", "")
                hiding_words = hidden_result.get("words", hidden_result.get("word", ""))

                if is_reversed:
                    highlighted = _highlight_hidden(hiding_words, answer_clean[::-1])
                    expl_text = 'hidden reversed in "%s"' % highlighted
                else:
                    highlighted = _highlight_hidden(hiding_words, answer_clean)
                    expl_text = 'hidden in "%s"' % highlighted

                pieces_data = [{
                    "clue_word": hiding_words,
                    "letters": answer_clean,
                    "mechanism": "hidden",
                }]
                payload = {
                    "definition": None,
                    "wordplay_type": hidden_result["op"],
                    "ai_explanation": expl_text,
                    "has_solution": 1,
                    "reviewed": 1,
                    "confidence": 1.0,
                    "components": {
                        "ai_pieces": pieces_data,
                        "assembly": hidden_result,
                        "wordplay_type": hidden_result["op"],
                    },
                    "wordplay_types": [hidden_result["op"]],
                    "definition_start": None,
                    "definition_end": None,
                    "model_version": "mechanical_hidden",
                    "source": row["source"],
                    "puzzle_number": row["puzzle_number"],
                    "clue_number": row["clue_number"],
                }
                f.write(json.dumps({
                    "clue_id": cid,
                    "action": "hidden_solve",
                    "payload": payload,
                }) + "\n")
                solved += 1
                continue

            # Check double definitions — store result
            dd_result = try_double_definition(clue, answer_clean, ref_db)
            if dd_result:
                left_def = dd_result["left_def"]
                right_def = dd_result["right_def"]
                expl_text = 'Double definition: "%s" and "%s" both mean %s' % (
                    left_def, right_def, answer)

                payload = {
                    "definition": left_def,
                    "wordplay_type": "double_definition",
                    "ai_explanation": expl_text,
                    "has_solution": 1,
                    "reviewed": 1,
                    "confidence": 1.0,
                    "components": {
                        "ai_pieces": [],
                        "assembly": dd_result,
                        "wordplay_type": "double_definition",
                    },
                    "wordplay_types": ["double_definition"],
                    "definition_start": None,
                    "definition_end": None,
                    "model_version": "mechanical_dd",
                    "source": row["source"],
                    "puzzle_number": row["puzzle_number"],
                    "clue_number": row["clue_number"],
                }
                f.write(json.dumps({
                    "clue_id": cid,
                    "action": "dd_solve",
                    "payload": payload,
                }) + "\n")
                solved += 1
                continue

            # Run S
            try:
                sr = solve_clue(clue, answer_clean, ref_db)
            except Exception:
                f.write(json.dumps({"clue_id": cid, "action": "error"}) + "\n")
                continue

            if sr.solved:
                solved += 1
                if sr.high_confidence:
                    payload = serialize_solve_result(row, sr, answer)
                    f.write(json.dumps({
                        "clue_id": cid,
                        "action": "high_solve",
                        "payload": payload,
                    }) + "\n")
                else:
                    payload = serialize_solve_result(row, sr, answer)
                    f.write(json.dumps({
                        "clue_id": cid,
                        "action": "medium_solve",
                        "confidence": sr.confidence,
                        "payload": payload,
                    }) + "\n")

                # Enrichment from solved clues (definition pairs only, same as original)
                proposals = extract_enrichment(clue, answer_clean, ref_db)
                proposals = [p for p in proposals if p[0] == "definition"]
                if proposals:
                    all_proposals.extend(proposals)
                    enrichment_clue_ids.append(cid)
            else:
                attempted += 1
                f.write(json.dumps({"clue_id": cid, "action": "attempted"}) + "\n")

                # Enrichment from unsolved clues (all types)
                proposals = extract_enrichment(clue, answer_clean, ref_db)
                if proposals:
                    all_proposals.extend(proposals)
                    enrichment_clue_ids.append(cid)

            if (i + 1) % 200 == 0:
                elapsed = time.time() - t0
                n_def = sum(1 for p in all_proposals if p[0] == "definition")
                n_res = sum(1 for p in all_proposals if p[0] == "residual")
                print(f"  {i+1}/{len(clues)} ({elapsed:.0f}s) - {solved} solved, "
                      f"{attempted} attempted, {n_def} def pairs, {n_res} residual pairs")

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s")
    print(f"  Solved: {solved}")
    print(f"  Attempted (not solved): {attempted}")
    print(f"  Skipped: {skipped}")
    print(f"  Enrichment proposals: {len(all_proposals)}")
    print(f"  Results written to: {results_path}")

    return all_proposals, enrichment_clue_ids


def deduplicate(proposals):
    """Deduplicate and count proposals."""
    counts = Counter()
    info = {}
    for ptype, word, letters, source in proposals:
        key = (word, letters)
        counts[key] += 1
        if key not in info:
            info[key] = (ptype, source)

    results = []
    for (word, letters), count in counts.most_common():
        ptype, source = info[(word, letters)]
        results.append((word, letters, count, ptype, source))
    return results


def verify_with_haiku(pairs):
    """Send proposed pairs to Haiku for validation."""
    import anthropic

    client = anthropic.Anthropic()
    chunk_size = 100
    all_results = []

    for i in range(0, len(pairs), chunk_size):
        chunk = pairs[i:i + chunk_size]

        pair_lines = []
        for word, letters, count, ptype, source in chunk:
            pair_lines.append(f"{word} -> {letters}")

        prompt = f"""You are a crossword expert. For each pair below, determine if the word on the left
is a valid synonym or recognised abbreviation of the letters on the right, as commonly used in
cryptic crossword clues.

IMPORTANT: Both sides must be real English words or widely recognised abbreviations.
Reject any pair where the right side is not a real word or standard abbreviation.
For example: "crowd -> RABLE" is INVALID because RABLE is not a word.
"painting -> OIL" is VALID because OIL is a real word and a synonym of painting.
"doctor -> DR" is VALID because DR is a recognised abbreviation.

Reply with ONLY a JSON array of objects, each with "word", "letters", and "valid" (true/false).

Pairs to check:
{chr(10).join(pair_lines)}"""

        print(f"\nVerifying batch {i // chunk_size + 1} ({len(chunk)} pairs)...")

        response = client.messages.create(
            model="claude-haiku-4-5-20251001",
            max_tokens=4096,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )

        try:
            text = response.content[0].text
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group())
                all_results.extend(results)
                valid_count = sum(1 for r in results if r.get("valid"))
                print(f"  {valid_count}/{len(results)} valid")
        except Exception as e:
            print(f"  Parse error: {e}")

    valid = [r for r in all_results if r.get("valid")]
    invalid = [r for r in all_results if not r.get("valid")]

    print(f"\n{'='*60}")
    print(f"HAIKU VALIDATION: {len(valid)}/{len(all_results)} valid")
    print(f"{'='*60}")

    print(f"\n--- VALID (add to DB) ---")
    for r in valid[:50]:
        print(f"  {r['word']:25} -> {r['letters']}")

    print(f"\n--- REJECTED ---")
    for r in invalid[:20]:
        print(f"  {r['word']:25} -> {r['letters']}")

    # Save validated pairs
    with open(VALIDATED_PATH, "w", encoding="utf-8") as f:
        for r in valid:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(valid)} validated pairs to {VALIDATED_PATH}")

    return valid


def main():
    parser = argparse.ArgumentParser(description="Offline batch solver — no DB locking")
    parser.add_argument("--limit", type=int, default=1000, help="Number of clues to process")
    parser.add_argument("--skip", type=int, default=0, help="Skip first N eligible clues (OFFSET)")
    parser.add_argument("--source", type=str, default=None, help="Filter by source")
    parser.add_argument("--no-haiku", action="store_true", help="Skip Haiku validation")
    args = parser.parse_args()

    print("=" * 60)
    print("OFFLINE BATCH SOLVER")
    print(f"  Limit: {args.limit}, Skip: {args.skip}")
    if args.source:
        print(f"  Source filter: {args.source}")
    print("=" * 60)

    # Step 1: Load clues (brief DB read, then close)
    print("\nLoading clues from DB...")
    clues = load_clues(args.limit, args.skip, args.source)
    print(f"Loaded {len(clues)} clues. DB connection closed.")

    if not clues:
        print("No clues to process.")
        return

    # Step 2: Load RefDB into memory (brief DB read, then close)
    print("Loading RefDB into memory...")
    ref_db = RefDB()
    print("RefDB loaded. All DB connections closed — DB is free.\n")

    # Step 3: Process clues (no DB access)
    proposals, enrichment_ids = run_batch(clues, ref_db, RESULTS_PATH)

    # Step 4: Deduplicate and save enrichment proposals
    if proposals:
        deduped = deduplicate(proposals)
        n_def = sum(1 for _, _, _, t, _ in deduped if t == "definition")
        n_res = sum(1 for _, _, _, t, _ in deduped if t == "residual")

        print(f"\n{'='*60}")
        print(f"UNIQUE PAIRS: {len(deduped)} ({n_def} definition, {n_res} residual)")
        print(f"{'='*60}")

        with open(PROPOSALS_PATH, "w", encoding="utf-8") as f:
            for word, letters, count, ptype, source in deduped:
                f.write(json.dumps({
                    "word": word, "letters": letters, "count": count,
                    "type": ptype, "source": source
                }) + "\n")
        print(f"Saved {len(deduped)} pairs to {PROPOSALS_PATH}")

        # Step 5: Haiku validation
        if not args.no_haiku:
            verify_with_haiku(deduped)
        else:
            print("Skipping Haiku validation (--no-haiku)")
    else:
        print("\nNo enrichment proposals generated.")

    # Save enrichment clue IDs for round 2
    if enrichment_ids:
        r2_path = os.path.join(ROOT, "data", "batch_offline_enrichment_ids.json")
        with open(r2_path, "w") as f:
            json.dump(enrichment_ids, f)
        print(f"Saved {len(enrichment_ids)} enrichment clue IDs to {r2_path}")


if __name__ == "__main__":
    main()
