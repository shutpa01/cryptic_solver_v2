"""Targeted explanation fixer — uses AI to correct specific verified errors.

Flow:
1. Run mechanical verifier to find WRONG claims
2. Send only the flagged issues to AI for targeted correction
3. Re-assemble with corrected pieces
4. Re-verify the result

The AI gets an easy job: "this first_letter claim is wrong, fix it"
rather than "explain this clue from scratch".
"""

import json
import os
import re
import sqlite3
import sys
import time
from pathlib import Path

from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent
load_dotenv(PROJECT_ROOT / ".env")

CLUES_DB = str(PROJECT_ROOT / "data" / "clues_master.db")
SONNET_MODEL = "claude-sonnet-4-20250514"

FIX_PROMPT = """You are fixing errors in a cryptic crossword explanation. The mechanical verifier has found specific wrong claims.

Clue: {clue_text}
Answer: {answer}
Current explanation pieces (JSON): {pieces_json}
Human explanation (if available): {human_explanation}

ERRORS FOUND:
{errors}

Fix ONLY the errors listed above. Keep all correct pieces unchanged. Return the full corrected pieces array as JSON.

Rules:
- Each piece has: clue_word, letters, mechanism
- "clue_word" must be the actual word(s) from the clue text
- "letters" must be uppercase letters that piece contributes
- All pieces' letters must combine to spell {answer}
- For first_letter: check which clue word actually starts with that letter
- For abbreviations: use the correct clue word that maps to those letters
- For synonyms: use the correct clue word
- Do NOT change pieces that aren't flagged as errors

Return ONLY the corrected JSON array, no other text."""


def fix_explanation(clue_text, answer, pieces, human_explanation, errors):
    """Send targeted fix request to Sonnet. Returns corrected pieces or None."""
    import anthropic

    client = anthropic.Anthropic(api_key=os.environ.get("ANTHROPIC_API_KEY"))

    errors_text = "\n".join(f"- {e}" for e in errors)
    pieces_json = json.dumps(pieces, indent=2)

    prompt = FIX_PROMPT.format(
        clue_text=clue_text,
        answer=answer,
        pieces_json=pieces_json,
        human_explanation=human_explanation or "(not available)",
        errors=errors_text,
    )

    try:
        response = client.messages.create(
            model=SONNET_MODEL,
            max_tokens=1000,
            temperature=0,
            messages=[{"role": "user", "content": prompt}],
        )
        text = response.content[0].text.strip()

        # Parse JSON from response (may be wrapped in markdown)
        text = re.sub(r"^```json\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        fixed_pieces = json.loads(text)
        tokens = response.usage.input_tokens + response.usage.output_tokens
        return fixed_pieces, tokens
    except Exception as e:
        print(f"  Fix error: {e}")
        return None, 0


def fix_puzzle(source, puzzle_number, write_db=False):
    """Find and fix wrong explanations for a puzzle."""
    from .verify_explanation import ExplanationVerifier
    from .solver import assemble, clean
    from .report import _describe_assembly

    conn = sqlite3.connect(CLUES_DB)
    conn.row_factory = sqlite3.Row

    rows = conn.execute("""
        SELECT c.id, c.clue_number, c.direction, c.clue_text, c.answer,
               c.definition, c.wordplay_type, c.ai_explanation, c.explanation,
               se.components, se.id as se_id
        FROM clues c
        LEFT JOIN structured_explanations se ON se.clue_id = c.id
        WHERE c.source = ? AND c.puzzle_number = ?
        AND c.ai_explanation IS NOT NULL AND length(c.ai_explanation) > 10
        AND c.direction IS NOT NULL
        ORDER BY c.direction, CAST(c.clue_number AS INTEGER)
    """, (source, str(puzzle_number))).fetchall()

    verifier = ExplanationVerifier()
    total_tokens = 0
    fixed = 0
    skipped = 0

    print(f"\n{'='*60}")
    print(f"FIX EXPLANATIONS: {source} #{puzzle_number} ({len(rows)} clues)")
    print(f"{'='*60}\n")

    for r in rows:
        d = (r["direction"] or "?")[0]
        label = f"{r['clue_number']}{d}"

        # Verify current explanation
        result = verifier.verify(
            r["clue_text"], r["answer"], r["definition"],
            r["wordplay_type"], r["ai_explanation"],
        )

        if result is None or result["wrong"] == 0:
            print(f"  {label} {r['answer']:15} OK ({result['verdict'] if result else '?'})")
            skipped += 1
            continue

        # Collect error descriptions
        errors = []
        for c in result["checks"]:
            if c["status"] == "wrong":
                errors.append(f"{c['check']}: {c['detail']}")

        print(f"  {label} {r['answer']:15} FIXING ({len(errors)} errors): {'; '.join(errors)}")

        # Get current pieces
        comps = json.loads(r["components"] or "{}") if r["components"] else {}
        pieces = comps.get("ai_pieces", [])
        if not pieces:
            print(f"    No pieces to fix, skipping")
            continue

        # Send to AI for targeted fix
        fixed_pieces, tokens = fix_explanation(
            r["clue_text"], r["answer"], pieces,
            r["explanation"] or "",
            errors,
        )
        total_tokens += tokens

        if not fixed_pieces:
            print(f"    Fix failed")
            continue

        # Re-assemble with corrected pieces
        piece_letters = [clean(p.get("letters", "")) for p in fixed_pieces if p.get("letters")]
        asm = assemble(r["clue_text"], r["answer"], piece_letters, ai_wtype=r["wordplay_type"])

        if not asm:
            # Try hidden word
            from .solver import try_hidden
            asm = try_hidden(r["clue_text"], clean(r["answer"]))

        if asm:
            expl = _describe_assembly(asm, fixed_pieces, answer=r["answer"])
            if expl and r["definition"]:
                expl += '; definition: "%s"' % r["definition"]
        else:
            # Fallback: just format the fixed pieces
            parts = []
            for p in fixed_pieces:
                cw = p.get("clue_word", "?")
                lt = p.get("letters", "?")
                mech = p.get("mechanism", "?")
                parts.append(f"{lt}({mech} \"{cw}\")")
            expl = " + ".join(parts)
            if r["definition"]:
                expl += '; definition: "%s"' % r["definition"]

        # Re-verify
        new_result = verifier.verify(
            r["clue_text"], r["answer"], r["definition"],
            r["wordplay_type"], expl,
        )

        old_score = result["score"]
        new_score = new_result["score"] if new_result else 0
        # Reject trivial fixes (just restating the definition)
        has_trivial = any(c["check"] == "trivial" for c in (new_result or {}).get("checks", []))
        improved = new_score > old_score and not has_trivial

        print(f"    {old_score} -> {new_score} ({new_result['verdict'] if new_result else '?'}) {'IMPROVED' if improved else 'no improvement'}")
        print(f"    New: {expl[:100]}")

        if improved and write_db:
            conn.execute("UPDATE clues SET ai_explanation = ? WHERE id = ?", (expl, r["id"]))
            # Update components
            comps["ai_pieces"] = fixed_pieces
            comps["assembly"] = asm
            if r["se_id"]:
                new_conf = new_score / 100.0
                conn.execute(
                    "UPDATE structured_explanations SET components = ?, confidence = ? WHERE id = ?",
                    (json.dumps(comps), new_conf, r["se_id"]),
                )
            fixed += 1

    conn.commit()
    conn.close()

    cost = total_tokens * 3.0 / 1_000_000  # Approximate Sonnet cost
    print(f"\n{'='*60}")
    print(f"DONE: {fixed} fixed, {skipped} already OK, {total_tokens} tokens (~${cost:.4f})")
    print(f"{'='*60}")


if __name__ == "__main__":
    if len(sys.argv) >= 3:
        source = sys.argv[1]
        puzzle = sys.argv[2]
        write = "--write-db" in sys.argv
        fix_puzzle(source, puzzle, write_db=write)
    else:
        print("Usage: python -m sonnet_pipeline.fix_explanations <source> <puzzle_number> [--write-db]")
