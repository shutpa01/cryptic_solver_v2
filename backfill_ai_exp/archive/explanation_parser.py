"""Parse human explanations into structured solver output.

Notation-aware parser following the grammar in documents/explanation_notation_grammar.md.
Extracts letter pieces from explanation text and verifies they assemble to the known answer.

All output goes to JSONL — NEVER writes to the database.

Usage:
    python scripts/explanation_parser.py --limit 1000       # test run
    python scripts/explanation_parser.py --source times      # one source
    python scripts/explanation_parser.py                     # all unsolved with explanations
"""

import argparse
import json
import os
import re
import sqlite3
import sys
import time

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
RESULTS_PATH = os.path.join(ROOT, "data", "parsed_explanations_v2.jsonl")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def norm_letters(s):
    return re.sub(r"[^A-Za-z]", "", s or "").lower()


def strip_enumeration(text):
    return re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', text).strip()


# ---------------------------------------------------------------------------
# Piece extraction from notation
# ---------------------------------------------------------------------------

def extract_uppercase_piece(token):
    """Extract contributing letters from an uppercase token.

    Handles:
    - Plain CAPS: HEAD -> HEAD
    - Deletion [lower]CAPS or CAPS[lower]: [pr]EVENT -> EVENT
    - Partial deletion {lower}CAPS or CAPS{lower}: {r}EEL -> EEL
    - Abbreviation source CAPS(lower): S(econd) -> S

    Returns (letters, annotation) or None if not a contributing piece.
    """
    # Skip pure lowercase or empty
    if not token or not any(c.isupper() for c in token):
        return None

    letters = ""
    annotation = ""

    # Remove [] deletions: [lower]CAPS or CAPS[lower]
    # The brackets show what was removed — keep the CAPS part
    cleaned = re.sub(r'\[[a-z]+\]', '', token)

    # Remove {} partial deletions: {lower}CAPS or CAPS{lower}
    cleaned = re.sub(r'\{[a-z]+\}', '', cleaned)

    # Handle CAPS(lowercase) — gloss or abbreviation source
    # If it's LETTER(rest-of-word) where LETTER is 1-2 uppercase chars, it's abbreviation source
    # If it's WORD(gloss), the parens content is just annotation
    m = re.match(r'^([A-Z][A-Z.]*)\(([^)]*)\)(.*)$', cleaned)
    if m:
        letters = re.sub(r'[^A-Z]', '', m.group(1))
        annotation = m.group(2)
        rest = m.group(3)
        if rest:
            letters += re.sub(r'[^A-Z]', '', rest)
        return (letters, annotation) if letters else None

    # Extract just the uppercase letters
    letters = re.sub(r'[^A-Z]', '', cleaned)
    return (letters, "") if letters else None


def split_on_plus(explanation):
    """Split explanation on + separator, respecting parentheses."""
    parts = []
    depth = 0
    current = ""
    for ch in explanation:
        if ch == '(':
            depth += 1
            current += ch
        elif ch == ')':
            depth -= 1
            current += ch
        elif ch == '+' and depth == 0:
            parts.append(current.strip())
            current = ""
        else:
            current += ch
    if current.strip():
        parts.append(current.strip())
    return parts


# ---------------------------------------------------------------------------
# Pattern 1: Charade with + separator
# ---------------------------------------------------------------------------

def try_parse_charade_plus(explanation, answer):
    """Parse charade notation: PIECE + PIECE + ... = ANSWER

    Examples:
        HEAD (top) + BUTT (bottom) -> HEAD + BUTT = HEADBUTT
        S(econd) + O.M. + ME -> S + OM + ME = SOMME
    """
    answer_norm = norm_letters(answer)

    # Must contain +
    if '+' not in explanation:
        return None

    # Strip everything after = if present
    expl = explanation.split('=')[0].strip() if '=' in explanation else explanation

    # Strip trailing prose after semicolon or dash
    expl = re.split(r'\s*[;–—]\s*', expl)[0].strip()

    parts = split_on_plus(expl)
    if len(parts) < 2:
        return None

    pieces = []
    for part in parts:
        part = part.strip()
        if not part:
            continue

        extracted = extract_uppercase_piece(part)
        if extracted:
            letters, annotation = extracted
            if letters:
                pieces.append({
                    "letters": letters,
                    "source": part,
                    "annotation": annotation,
                })

    if not pieces:
        return None

    # Verification: pieces must concatenate to the answer
    assembled = "".join(p["letters"] for p in pieces)
    if norm_letters(assembled) != answer_norm:
        return None

    return {
        "wordplay_type": "charade",
        "pieces": pieces,
    }


# ---------------------------------------------------------------------------
# Pattern 2: Anagram with * marker
# ---------------------------------------------------------------------------

def try_parse_anagram_star(explanation, answer):
    """Parse anagram notation: (FODDER)* or FODDER*

    Examples:
        PROUST* -> anagram of PROUST = STUPOR
        (NOW SO)* -> anagram of NOWSO = SWOON
        D[ivisive] + RACIST* -> D + anagram of RACIST (composite)
    """
    answer_norm = norm_letters(answer)

    if '*' not in explanation:
        return None

    # Find the starred group: (LETTERS)* or LETTERS*
    # Try parenthesized form first
    m = re.search(r'\(([A-Z\s]+)\)\*', explanation)
    if m:
        fodder = m.group(1)
    else:
        # Try bare WORD*
        m = re.search(r'([A-Z][A-Z,.\s]*)\*', explanation)
        if not m:
            return None
        fodder = m.group(1)

    fodder_letters = norm_letters(fodder)

    # Simple case: entire explanation is just the anagram
    if sorted(fodder_letters) == sorted(answer_norm) and fodder_letters != answer_norm:
        return {
            "wordplay_type": "anagram",
            "pieces": [{"letters": fodder.strip(), "source": fodder.strip(), "annotation": "anagram"}],
        }

    # Composite case: anagram is part of a charade (e.g. D[ivisive] + RACIST*)
    # Try parsing as charade with the * fodder as one piece
    if '+' in explanation:
        result = try_parse_charade_plus(explanation.replace('*', ''), answer)
        if result:
            # Mark the anagrammed piece
            for p in result["pieces"]:
                if norm_letters(p["letters"]) == fodder_letters:
                    p["annotation"] = "anagram"
            result["wordplay_type"] = "charade"  # composite
            return result

    return None


# ---------------------------------------------------------------------------
# Pattern 3: Prose double definition
# ---------------------------------------------------------------------------

def try_parse_dd(explanation, answer):
    """Parse double definition from prose keywords.

    Examples:
        DD, Double definition, Two meanings
    """
    expl_lower = explanation.lower().strip()
    if expl_lower.startswith(('dd', 'double def', 'two meanings', 'double definition')):
        return {
            "wordplay_type": "double_definition",
            "pieces": [],
        }
    return None


# ---------------------------------------------------------------------------
# Pattern 4: Prose hidden word
# ---------------------------------------------------------------------------

def try_parse_hidden(explanation, answer):
    """Parse hidden word from prose or notation."""
    answer_norm = norm_letters(answer)
    expl_lower = explanation.lower()

    if 'hidden' not in expl_lower:
        return None

    # Verify answer is actually hidden in the explanation's quoted text or clue
    # Look for the answer in any quoted string
    quoted = re.findall(r'"([^"]+)"', explanation)
    for q in quoted:
        q_letters = norm_letters(q)
        if answer_norm in q_letters or answer_norm[::-1] in q_letters:
            direction = "hidden_reversed" if answer_norm[::-1] in q_letters and answer_norm not in q_letters else "hidden"
            return {
                "wordplay_type": direction,
                "pieces": [{"letters": answer.upper(), "source": q, "annotation": "hidden"}],
            }

    return None


# ---------------------------------------------------------------------------
# Pattern 5: Prose homophone
# ---------------------------------------------------------------------------

def try_parse_homophone(explanation, answer):
    """Parse homophone from prose keywords."""
    expl_lower = explanation.lower()
    if 'homophone' in expl_lower or 'sounds like' in expl_lower:
        return {
            "wordplay_type": "homophone",
            "pieces": [],
        }
    return None


# ---------------------------------------------------------------------------
# JSONL output
# ---------------------------------------------------------------------------

def build_result(clue_id, parse_result, clue_text, answer, definition,
                 source, puzzle_number, clue_number, explanation):
    """Build JSONL record from parse result."""
    wtype = parse_result["wordplay_type"]
    pieces = parse_result["pieces"]

    # Build ai_pieces format
    ai_pieces = []
    for p in pieces:
        ai_pieces.append({
            "clue_word": p.get("source", ""),
            "letters": p["letters"],
            "mechanism": p.get("annotation", wtype),
        })

    # Build explanation text
    if wtype == "double_definition":
        expl_text = explanation  # Use original
    elif wtype in ("hidden", "hidden_reversed"):
        expl_text = explanation
    elif wtype == "charade":
        part_strs = [p["letters"] for p in pieces]
        expl_text = " + ".join(part_strs) + " = " + answer.upper()
    elif wtype == "anagram":
        fodder = " + ".join(p["letters"] for p in pieces)
        expl_text = "anagram of %s = %s" % (fodder, answer.upper())
    else:
        expl_text = explanation

    return {
        "clue_id": clue_id,
        "action": "high_solve",
        "payload": {
            "definition": definition,
            "wordplay_type": wtype,
            "ai_explanation": expl_text,
            "has_solution": 1,
            "reviewed": 1,
            "confidence": 1.0,
            "components": {
                "ai_pieces": ai_pieces,
                "assembly": {"op": wtype},
                "wordplay_type": wtype,
            },
            "wordplay_types": [wtype],
            "definition_start": None,
            "definition_end": None,
            "model_version": "explanation_parser_v2",
            "source": source,
            "puzzle_number": puzzle_number,
            "clue_number": clue_number,
        }
    }


# ---------------------------------------------------------------------------
# Main parser — tries each pattern in order
# ---------------------------------------------------------------------------

def parse_explanation(explanation, answer):
    """Try each parser in priority order. Return first match or None."""
    if not explanation or not answer:
        return None

    answer_norm = norm_letters(answer)
    if len(answer_norm) < 2:
        return None

    # Normalize smart quotes
    explanation = (explanation
        .replace('\u2018', "'").replace('\u2019', "'")
        .replace('\u201c', '"').replace('\u201d', '"'))

    # Try parsers in order
    for parser in [
        try_parse_dd,
        try_parse_hidden,
        try_parse_anagram_star,
        try_parse_charade_plus,
        try_parse_homophone,
    ]:
        result = parser(explanation, answer)
        if result:
            return result

    return None


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="Parse human explanations")
    parser.add_argument("--limit", type=int, default=0, help="Max clues (0=all)")
    parser.add_argument("--source", type=str, default=None, help="Filter by source")
    parser.add_argument("--output", type=str, default=RESULTS_PATH, help="Output JSONL path")
    parser.add_argument("--exclude", type=str, default=None,
                        help="JSONL file(s) of already-solved clue_ids to skip (comma-separated)")
    args = parser.parse_args()

    # Load already-solved IDs to skip
    skip_ids = set()
    if args.exclude:
        for path in args.exclude.split(","):
            path = path.strip()
            if os.path.exists(path):
                with open(path, encoding="utf-8") as f:
                    for line in f:
                        try:
                            skip_ids.add(json.loads(line)["clue_id"])
                        except (json.JSONDecodeError, KeyError):
                            pass
                print(f"  Loaded {len(skip_ids):,} IDs to skip from {path}")

    print("Loading unsolved clues with explanations...")
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    query = """
        SELECT id, source, puzzle_number, clue_number, clue_text, answer,
               enumeration, explanation, definition
        FROM clues
        WHERE answer IS NOT NULL AND answer != ''
          AND explanation IS NOT NULL AND explanation != ''
          AND (has_solution IS NULL OR has_solution = 0)
    """
    params = []
    if args.source:
        query += " AND source = ?"
        params.append(args.source)
    query += " ORDER BY publication_date DESC"
    if args.limit:
        query += " LIMIT ?"
        params.append(args.limit)

    rows = conn.execute(query, params).fetchall()
    conn.close()
    print(f"  {len(rows):,} clues loaded")

    # Process
    parsed = 0
    skipped = 0
    by_type = {}
    results = []
    t0 = time.time()

    for cid, source, pnum, cnum, clue_text, answer, enum, explanation, definition in rows:
        if cid in skip_ids:
            skipped += 1
            continue

        result = parse_explanation(explanation, answer)
        if result:
            parsed += 1
            wt = result["wordplay_type"]
            by_type[wt] = by_type.get(wt, 0) + 1

            results.append(build_result(
                cid, result, clue_text, answer, definition,
                source, pnum, cnum, explanation))

    elapsed = time.time() - t0

    # Write results
    with open(args.output, "w", encoding="utf-8") as f:
        for r in results:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    print(f"\nDone in {elapsed:.1f}s")
    print(f"  Processed: {len(rows):,}")
    print(f"  Skipped (already solved): {skipped:,}")
    print(f"  Parsed: {parsed:,}")
    for wt, n in sorted(by_type.items(), key=lambda x: -x[1]):
        print(f"    {wt}: {n:,}")
    print(f"  Output: {args.output}")


if __name__ == "__main__":
    main()
