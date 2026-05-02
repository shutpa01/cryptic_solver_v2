"""Ingest Claude review responses and write to database via the verifier.

Parses the structured output from Claude's clue solutions, runs each through
the piece verifier for confidence scoring, then writes to the database.

Usage:
    python scripts/ingest_claude_review.py data/claude_review_2026-04-10_response.txt
    python scripts/ingest_claude_review.py data/claude_review_2026-04-10_response.txt --dry-run
"""

import argparse
import json
import re
import sqlite3
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = ROOT / "data" / "clues_master.db"
sys.path.insert(0, str(ROOT))


def parse_response(text):
    """Parse Claude's structured response into a list of clue solutions.

    Expected format per clue:
    ---
    ID: 12345
    WORDPLAY: charade
    DEFINITION: having three parties
    EXPLANATION: TRIAL (suffering) contains LATE (delayed) + R (fourth) = TRILATERAL
    ---
    """
    results = []
    # Split on --- delimiters
    blocks = re.split(r'^---\s*$', text, flags=re.MULTILINE)

    for block in blocks:
        block = block.strip()
        if not block:
            continue

        entry = {}
        for line in block.splitlines():
            line = line.strip()
            if line.upper().startswith("ID:"):
                try:
                    entry["id"] = int(line.split(":", 1)[1].strip())
                except ValueError:
                    pass
            elif line.upper().startswith("WORDPLAY:"):
                entry["wordplay_type"] = line.split(":", 1)[1].strip().lower()
            elif line.upper().startswith("DEFINITION:"):
                entry["definition"] = line.split(":", 1)[1].strip()
            elif line.upper().startswith("EXPLANATION:"):
                entry["explanation"] = line.split(":", 1)[1].strip()

        if "id" in entry and "explanation" in entry:
            results.append(entry)

    return results


def _is_indicator_candidate(s):
    """Does this quoted string look like an indicator candidate rather than a
    hidden-span (which has internal uppercase letters) or a long phrase?"""
    s_stripped = s.strip()
    if not s_stripped or len(s_stripped) > 30:
        return False
    # Spans like "bacheLORISaw" have uppercase letters after position 0
    if any(c.isupper() for c in s_stripped[1:]):
        return False
    return True


def _extract_gaps(checks, explanation=""):
    """Extract DB gaps from verifier checks for the enrichment queue.

    Parses unverifiable/wrong checks to find missing synonym, definition,
    abbreviation, indicator, and homophone mappings that should be reviewed
    for addition to the DB.
    """
    gaps = []
    indicator_types_needed = set()

    for check in checks:
        status = check.get("status", "")
        detail = check.get("detail", "")
        check_type = check.get("check", "")

        if status not in ("unverifiable", "wrong"):
            continue

        if check_type == "definition" and "not in DB" in detail:
            m = re.match(r"'(.+?)'\s*->\s*(\w+):", detail)
            if m:
                gaps.append(("definition", m.group(1).lower(), m.group(2).upper()))

        elif check_type == "synonym" and "not in DB" in detail:
            m = re.match(r"'(.+?)'\s*=\s*(\w+):", detail)
            if m:
                gaps.append(("synonym", m.group(1).lower(), m.group(2).upper()))

        elif check_type == "abbreviation" and "NOT KNOWN" in detail:
            m = re.match(r"'(.+?)'\s*->\s*(\w+):", detail)
            if m:
                gaps.append(("abbreviation", m.group(1).lower(), m.group(2).upper()))

        elif check_type == "homophone" and "not in DB" in detail:
            m = re.match(r"'(.+?)' sounds like '(.+?)': not in DB", detail)
            if m:
                gaps.append(("homophone", m.group(1).lower(), m.group(2).upper()))

        elif check_type == "dd":
            # Extract the window(s) that didn't map to the answer.
            # Windows are delimited by backticks (`) to avoid apostrophe
            # collisions ("Elon Musk's baby").
            # Partial format:   DD partial: `X`=True, `Y`=False (answer ANS)[, ...]
            # Both-wrong format: DD: neither `X` nor `Y` maps to ANS in DB
            ans_m = re.search(r"\(answer (\w+)\)|maps to (\w+) in DB", detail)
            answer_letters = ""
            if ans_m:
                answer_letters = (ans_m.group(1) or ans_m.group(2) or "").upper()
            # Partial format
            for m in re.finditer(r"`([^`]+)`=False", detail):
                window = m.group(1).strip().lower()
                if window and answer_letters:
                    gaps.append(("definition", window, answer_letters))
            # Both-wrong format
            if "neither" in detail:
                m = re.match(r"DD: neither `([^`]+)` nor `([^`]+)`", detail)
                if m and answer_letters:
                    for w in (m.group(1), m.group(2)):
                        window = w.strip().lower()
                        if window:
                            gaps.append(("definition", window, answer_letters))

        elif check_type == "indicator" and status == "wrong":
            m = re.search(r"no '([^']+)' indicator", detail)
            if m:
                indicator_types_needed.add(m.group(1))

    # For each missing indicator type, extract candidate indicator words from
    # the explanation's quoted annotations. To avoid noise, strip out known
    # annotation contexts first (synonym/abbreviation/positional/deletion
    # sources, definition, hidden spans) so only "orphan" quoted words remain
    # — those are likely the indicator candidates.
    if indicator_types_needed and explanation:
        cleaned = explanation
        cleaned = re.sub(r'\(\s*synonym\s*[=:]?\s*(?:of\s+)?["\'][^"\']*["\']\s*[^)]*\)',
                         '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(\s*abbreviation\s*=\s*["\'][^"\']*["\']\s*[^)]*\)',
                         '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(\s*abbr\.?\s*[^)]*\)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(
            r'\(\s*(?:first|last|middle|initial|final|odd|even|alternat(?:e|ing))'
            r'\s+letters?\s+(?:of|in|from)\s+[^)]*\)', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'\(\s*deletion\s*=\s*["\'][^"\']*["\']\s*[^)]*\)',
                         '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'definition:\s*["\'][^"\']*["\']', '', cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r'hidden(?:\s+reversed)?\s+in\s+["\'][^"\']*["\']',
                         '', cleaned, flags=re.IGNORECASE)

        quoted = re.findall(r'"([^"]+)"', cleaned)
        seen = set()
        for req_type in indicator_types_needed:
            for q in quoted:
                if not _is_indicator_candidate(q):
                    continue
                key = (q.strip().lower(), req_type.upper())
                if key in seen:
                    continue
                seen.add(key)
                gaps.append(("indicator", key[0], key[1]))

    return gaps


def verify_and_store(results, dry_run=False):
    """Run each result through the verifier and store in the database."""
    conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    conn.row_factory = sqlite3.Row

    # Import verifier
    from sonnet_pipeline.verify_explanation import ExplanationVerifier
    verifier = ExplanationVerifier()

    stored = 0
    failed = 0
    enrichments_queued = 0
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    for r in results:
        clue_id = r["id"]

        # Fetch the clue
        row = conn.execute(
            "SELECT * FROM clues WHERE id = ?", (clue_id,)
        ).fetchone()
        if not row:
            print(f"  SKIP {clue_id}: clue not found")
            failed += 1
            continue

        answer = row["answer"]
        clue_text = row["clue_text"]
        wordplay_type = r.get("wordplay_type", "")
        definition = r.get("definition", "")
        explanation = r.get("explanation", "")

        # Run through verifier
        try:
            v_result = verifier.verify(
                clue_text=clue_text,
                answer=answer,
                wordplay_type=wordplay_type,
                definition=definition,
                ai_explanation=explanation,
            )
            score = v_result.get("score", 0)
            verdict = v_result.get("verdict", "FAIL")
            confidence = score / 100.0
        except Exception as e:
            print(f"  VERIFY ERROR {clue_id}: {e}")
            score = 50
            verdict = "MEDIUM"
            confidence = 0.5
            v_result = {"checks": []}

        tier = verdict

        label = f"{row['clue_number']}{row['direction'][0]}"
        print(f"  [{tier:6} {score:3}] {label:4} {answer:15} {explanation[:60]}")

        if dry_run:
            continue

        # Build components JSON
        components = json.dumps({
            "ai_pieces": [],
            "assembly": {"op": wordplay_type},
            "wordplay_type": wordplay_type,
            "source": "claude_review",
        })

        # Update clues table — set reviewed=1 to protect from re-verify clearing
        conn.execute("""
            UPDATE clues
            SET definition = ?, wordplay_type = ?, ai_explanation = ?,
                has_solution = 1, reviewed = 1
            WHERE id = ?
        """, (definition, wordplay_type, explanation, clue_id))

        # Insert/replace structured_explanations
        conn.execute("""
            INSERT OR REPLACE INTO structured_explanations
            (clue_id, definition_text, wordplay_types, components,
             model_version, confidence, created_at, updated_at,
             source, puzzle_number, clue_number)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            clue_id, definition, json.dumps([wordplay_type]), components,
            "claude_review", confidence, now, now,
            row["source"], row["puzzle_number"], row["clue_number"],
        ))

        stored += 1

        # Queue DB gaps for enrichment review
        gaps = _extract_gaps(v_result.get("checks", []), explanation=explanation)
        from sonnet_pipeline.enrichment_gate import already_in_reference_db
        for gap_type, word, letters in gaps:
            # Skip if already known in the reference DB. The verifier
            # sometimes flags a "missing" lookup that's actually present
            # via a different table or matching path; queueing those
            # wastes review time.
            if already_in_reference_db(gap_type, word, letters):
                continue
            # Skip if previously rejected
            rejected = conn.execute(
                "SELECT 1 FROM rejected_enrichments WHERE type=? AND LOWER(word)=? AND UPPER(letters)=?",
                (gap_type, word, letters),
            ).fetchone()
            if rejected:
                continue
            try:
                conn.execute("""
                    INSERT OR IGNORE INTO pending_enrichments
                    (type, word, letters, answer, clue_text, source, puzzle_number, created_at)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """, (gap_type, word, letters, answer, clue_text,
                      row["source"], row["puzzle_number"], now))
                enrichments_queued += 1
            except sqlite3.IntegrityError:
                pass  # Already queued

    if not dry_run:
        conn.commit()
    conn.close()

    print(f"\n{'Would store' if dry_run else 'Stored'}: {stored}, Failed: {failed}, Enrichments queued: {enrichments_queued}")
    return stored, failed


def main():
    parser = argparse.ArgumentParser(description="Ingest Claude review responses")
    parser.add_argument("response_file", help="Path to Claude's response file")
    parser.add_argument("--dry-run", action="store_true", help="Show what would be stored")
    args = parser.parse_args()

    path = Path(args.response_file)
    if not path.exists():
        print(f"File not found: {path}")
        sys.exit(1)

    text = path.read_text(encoding="utf-8")
    results = parse_response(text)
    print(f"Parsed {len(results)} clue solutions from {path.name}")

    if not results:
        print("No results to ingest.")
        sys.exit(0)

    verify_and_store(results, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
