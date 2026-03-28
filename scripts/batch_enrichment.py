"""Batch enrichment capture — extract synonym proposals from S's partial matches.

Two sources of enrichment:
1. Definition → Answer: when S identifies a definition, that word is a synonym of the answer
2. Partial match residuals: when S finds some pieces but not all, the unused word
   maps to the residual answer letters

Usage:
    python scripts/batch_enrichment.py                    # Run on 1000 random clues
    python scripts/batch_enrichment.py --limit 5000       # More clues
    python scripts/batch_enrichment.py --source telegraph  # One source
    python scripts/batch_enrichment.py --verify            # Send to Haiku for validation
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
from sonnet_pipeline.sig_adapter import store_signature_result

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
CRYPTIC_DB = os.path.join(ROOT, "data", "cryptic_new.db")


def extract_enrichment(clue_text, answer_clean, ref_db):
    """Extract proposed synonym pairs from a single clue.

    Returns list of (type, word, letters, source) tuples where:
      type = 'definition' or 'residual'
      word = the clue word (lowercase)
      letters = the proposed synonym/value (uppercase)
      source = brief description of how it was found
    """
    proposals = []
    answer_up = answer_clean.upper()

    clue_words = _normalize_clue(clue_text).strip().split()
    def_candidates = extract_definition_candidates(clue_words, answer_up, ref_db)

    # 1. Definition enrichment: definition word → full answer
    for def_phrase, wp_words in def_candidates:
        def_clean = def_phrase.lower().strip(".,;:!?\"'()-").strip()
        # Check it's not already in DB
        existing = answer_up in [s.upper().replace(" ", "").replace("-", "")
                                 for s in ref_db.get_synonyms(def_clean)]
        if not existing:
            proposals.append(("definition", def_clean, answer_up, "def->answer"))

    # 2. Residual enrichment from partial piece matching
    if not def_candidates:
        return proposals

    # Use first (best) definition candidate
    def_phrase, wp_words = def_candidates[0]
    analyses, phrases = analyze_phrases(wp_words, answer_up, ref_db)

    # Find pieces that ARE substrings of the answer (confirmed pieces)
    confirmed = []  # (word_text, value_letters)
    unmatched = []  # word texts with no confirmed piece

    for wa in analyses:
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
            # Is it an indicator or link word? Those don't contribute letters
            is_ind = any(t.endswith("_I") or t.startswith("POS_I") for t in wa.roles)
            is_lnk = "LNK" in wa.roles
            if not is_ind and not is_lnk:
                unmatched.append(wa.text)

    if not confirmed or not unmatched:
        return proposals

    # Subtract confirmed pieces from answer to get residual
    residual = answer_up
    for word, val in confirmed:
        if val in residual:
            residual = residual.replace(val, "", 1)

    if not residual or len(residual) > 6:
        return proposals

    # If exactly one unmatched word, the pair is unambiguous
    if len(unmatched) == 1:
        uw = unmatched[0].lower().strip(".,;:!?\"'()-").strip()
        if uw and len(uw) >= 2:
            # Residual must be a plausible word or known abbreviation (1-2 letters)
            # Reject random letter fragments
            is_real = ref_db.is_real_word(residual.lower()) if len(residual) >= 3 else True
            is_short = len(residual) <= 2  # 1-2 letter abbreviations are always plausible
            if not is_real and not is_short:
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


def run_batch(ref_db, limit=1000, source_filter=None, batch_ids=None, solves_only=False):
    """Run enrichment capture on a batch of unsolved clues."""

    conn = sqlite3.connect(CLUES_DB, timeout=60)
    conn.row_factory = sqlite3.Row
    write_conn = conn  # Use same connection for reads and writes

    where = "answer IS NOT NULL AND answer != '' AND clue_text IS NOT NULL AND clue_text != ''"
    where += " AND (has_solution IS NULL OR has_solution = 0)"  # excludes 1 (solved), 2 (partial), 3 (S attempted)
    where += " AND publication_date < date('now')"
    params = []
    if source_filter:
        where += " AND source = ?"
        params.append(source_filter)

    if batch_ids is not None:
        # Re-run a saved batch
        placeholders = ",".join("?" * len(batch_ids))
        rows = conn.execute("""
            SELECT id, source, clue_text, answer
            FROM clues WHERE id IN (%s)
            ORDER BY publication_date DESC, source, CAST(puzzle_number AS INTEGER) DESC
        """ % placeholders, batch_ids).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, source, clue_text, answer
            FROM clues WHERE %s
            ORDER BY publication_date DESC, source, CAST(puzzle_number AS INTEGER) DESC
            LIMIT ?
        """ % where, params + [limit]).fetchall()

    # Save batch IDs for re-running
    row_ids = [row["id"] for row in rows]

    print(f"Processing {len(rows)} clues...")

    all_proposals = []
    enrichment_clue_ids = []  # IDs of clues that produced proposals
    solved = 0
    t0 = time.time()

    for i, row in enumerate(rows):
        cid = row["id"]
        clue = row["clue_text"]
        answer = row["answer"]
        answer_clean = clean(answer)
        if not answer_clean or len(answer_clean) < 2:
            continue

        # Skip hidden/DD/cross-refs
        if try_hidden(clue, answer_clean):
            solved += 1
            continue
        if try_double_definition(clue, answer_clean, ref_db):
            solved += 1
            continue
        if re.search(r"\b\d+\s*(?:across|down|ac|dn)\b", clue, re.IGNORECASE):
            continue

        # Run S
        try:
            sr = solve_clue(clue, answer_clean, ref_db)
        except Exception:
            continue

        if sr.solved:
            solved += 1
            # Store HIGH confidence solves to DB
            if sr.high_confidence:
                try:
                    store_signature_result(write_conn, cid, sr, clue, answer)
                    write_conn.commit()
                except Exception:
                    pass
            else:
                # Solved but not HIGH — mark as attempted
                write_conn.execute("UPDATE clues SET has_solution = 3 WHERE id = ? AND (has_solution IS NULL OR has_solution = 0)", (cid,))
                write_conn.commit()
            if not solves_only:
                proposals = extract_enrichment(clue, answer_clean, ref_db)
                proposals = [p for p in proposals if p[0] == "definition"]
                if proposals:
                    all_proposals.extend(proposals)
                    enrichment_clue_ids.append(cid)
        else:
            # Not solved — mark as attempted
            write_conn.execute("UPDATE clues SET has_solution = 3 WHERE id = ? AND (has_solution IS NULL OR has_solution = 0)", (cid,))
            write_conn.commit()
            if not solves_only:
                proposals = extract_enrichment(clue, answer_clean, ref_db)
                if proposals:
                    all_proposals.extend(proposals)
                    enrichment_clue_ids.append(cid)

        if (i + 1) % 200 == 0:
            elapsed = time.time() - t0
            n_def = sum(1 for p in all_proposals if p[0] == "definition")
            n_res = sum(1 for p in all_proposals if p[0] == "residual")
            print(f"  {i+1}/{len(rows)} ({elapsed:.0f}s) — {solved} solved, "
                  f"{n_def} def pairs, {n_res} residual pairs")

    conn.close()
    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.0f}s — {solved} solved")

    # Save enrichment clue IDs for targeted round 2
    if enrichment_clue_ids and not solves_only:
        r2_path = os.path.join(ROOT, "data", "enrichment_round2_ids.json")
        with open(r2_path, "w") as f:
            json.dump(enrichment_clue_ids, f)
        print(f"Saved {len(enrichment_clue_ids)} clue IDs for round 2 to {r2_path}")

    return all_proposals, row_ids


def deduplicate(proposals):
    """Deduplicate and count proposals. Returns sorted list of (word, letters, count, type, source)."""
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



def main():
    parser = argparse.ArgumentParser(description="Batch enrichment capture")
    parser.add_argument("--limit", type=int, default=1000, help="Number of clues to process")
    parser.add_argument("--source", type=str, default=None, help="Filter by source")
    parser.add_argument("--rerun", type=str, default=None, help="Re-run a saved batch file")
    parser.add_argument("--solves-only", action="store_true", help="Just solve, skip enrichment capture and Haiku")
    args = parser.parse_args()

    print("=" * 60)
    print("BATCH ENRICHMENT CAPTURE")
    print("=" * 60)

    batch_ids = None
    if args.rerun:
        with open(args.rerun) as f:
            batch_ids = json.loads(f.read())
        print(f"Re-running saved batch: {len(batch_ids)} clue IDs")

    ref_db = RefDB()
    proposals, row_ids = run_batch(ref_db, limit=args.limit,
                                   source_filter=args.source,
                                   batch_ids=batch_ids,
                                   solves_only=args.solves_only)

    # Save batch IDs for re-running
    batch_path = os.path.join(ROOT, "data", "enrichment_batch_ids.json")
    if batch_ids is None:
        with open(batch_path, "w") as f:
            json.dump(row_ids, f)
        print(f"Saved {len(row_ids)} batch IDs to {batch_path}")

    if args.solves_only:
        return

    # Deduplicate
    deduped = deduplicate(proposals)
    n_def = sum(1 for _, _, _, t, _ in deduped if t == "definition")
    n_res = sum(1 for _, _, _, t, _ in deduped if t == "residual")

    print(f"\n{'='*60}")
    print(f"UNIQUE PAIRS: {len(deduped)} ({n_def} definition, {n_res} residual)")
    print(f"{'='*60}")

    print(f"\n--- DEFINITION PAIRS (word -> answer) ---")
    def_pairs = [(w, l, c, s) for w, l, c, t, s in deduped if t == "definition"]
    for word, letters, count, source in def_pairs[:30]:
        print(f"  {word:25} -> {letters:15} x{count}")
    if len(def_pairs) > 30:
        print(f"  ... and {len(def_pairs) - 30} more")

    print(f"\n--- RESIDUAL PAIRS (unused word -> leftover letters) ---")
    res_pairs = [(w, l, c, s) for w, l, c, t, s in deduped if t == "residual"]
    for word, letters, count, source in res_pairs[:30]:
        print(f"  {word:25} -> {letters:15} x{count}  [{source}]")
    if len(res_pairs) > 30:
        print(f"  ... and {len(res_pairs) - 30} more")

    # Save proposals to file
    output_path = os.path.join(ROOT, "data", "enrichment_proposals.jsonl")
    with open(output_path, "w") as f:
        for word, letters, count, ptype, source in deduped:
            f.write(json.dumps({
                "word": word, "letters": letters, "count": count,
                "type": ptype, "source": source
            }) + "\n")
    print(f"\nSaved {len(deduped)} pairs to {output_path}")

    # Always validate with Haiku
    verify_with_haiku(deduped)


def verify_with_haiku(pairs):
    """Send proposed pairs to Haiku for validation."""
    import anthropic

    client = anthropic.Anthropic()

    # Batch into chunks of 100
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
            # Extract JSON from response
            json_match = re.search(r"\[.*\]", text, re.DOTALL)
            if json_match:
                results = json.loads(json_match.group())
                all_results.extend(results)
                valid_count = sum(1 for r in results if r.get("valid"))
                print(f"  {valid_count}/{len(results)} valid")
        except Exception as e:
            print(f"  Parse error: {e}")

    # Summary
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
    output_path = os.path.join(ROOT, "data", "enrichment_validated.jsonl")
    with open(output_path, "w") as f:
        for r in valid:
            f.write(json.dumps(r) + "\n")
    print(f"\nSaved {len(valid)} validated pairs to {output_path}")

    return valid


if __name__ == "__main__":
    main()
