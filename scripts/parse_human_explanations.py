"""Parse human explanations mechanically to extract definition, wordplay type, and pieces.

Works on clues that have a human explanation (from TFTT, fifteensquared, etc.)
but no structured solution yet. Extracts structured data by pattern matching
the explanation text, verified against the known answer.

Writes results to JSONL for upload via batch_upload.py.

Usage:
    python scripts/parse_human_explanations.py                    # all unsolved with explanations
    python scripts/parse_human_explanations.py --limit 1000       # test run
    python scripts/parse_human_explanations.py --source guardian   # one source
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

from signature_solver.db import RefDB

CLUES_DB = os.path.join(ROOT, "data", "clues_master.db")
RESULTS_PATH = os.path.join(ROOT, "data", "parsed_explanations_results.jsonl")


# ---------------------------------------------------------------------------
# Utilities
# ---------------------------------------------------------------------------

def norm_letters(s):
    return re.sub(r"[^A-Za-z]", "", s or "").lower()


def strip_enumeration(text):
    return re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', text).strip()


# ---------------------------------------------------------------------------
# Pattern parsers — each returns a dict or None
# ---------------------------------------------------------------------------

def try_parse_dd(expl, answer):
    """Double definition: 'DD', 'Double definition', 'Two meanings'."""
    if re.match(r'^(?:DD|[Dd]ouble def|[Tt]wo meanings|[Tt]wo definitions|dd\b)', expl.strip()):
        return {"wordplay_type": "double_definition", "pieces": []}
    # Also catch 'double def' anywhere at start after stripping
    if re.match(r'^[Dd]ouble\s+def', expl.strip()):
        return {"wordplay_type": "double_definition", "pieces": []}
    return None


def try_parse_cryptic_def(expl, answer):
    """Cryptic definition: 'cryptic def', '&lit'."""
    e = expl.strip().lower()
    if e.startswith('cryptic def') or e.startswith('&lit') or e.startswith('cd '):
        return {"wordplay_type": "cryptic_definition", "pieces": []}
    if 'cryptic definition' in e[:30]:
        return {"wordplay_type": "cryptic_definition", "pieces": []}
    return None


def try_parse_hidden(expl, answer, clue_text):
    """Hidden word: 'hidden', 'hidden in', check answer is contiguous in clue."""
    e = expl.strip().lower()
    if not (e.startswith('hidden') or 'hidden in' in e[:30] or 'hidden' in e[:15]):
        return None

    ans_clean = norm_letters(answer)
    clue_clean = norm_letters(clue_text)

    # Check forward
    if ans_clean in clue_clean:
        return {
            "wordplay_type": "hidden",
            "pieces": [{"clue_word": clue_text, "letters": answer.upper(), "mechanism": "hidden"}],
        }
    # Check reversed
    if ans_clean[::-1] in clue_clean:
        return {
            "wordplay_type": "hidden_reversed",
            "pieces": [{"clue_word": clue_text, "letters": answer.upper(), "mechanism": "hidden"}],
        }
    return None


def try_parse_homophone(expl, answer):
    """Homophone: 'Homophone of', 'sounds like', 'we hear'."""
    e = expl.strip().lower()
    if any(kw in e[:40] for kw in ('homophone', 'sounds like', 'sounds something like',
                                     'we hear', 'audibly', 'being discussed')):
        return {"wordplay_type": "homophone", "pieces": []}
    return None


def try_parse_anagram_star(expl, answer):
    """Anagram with star notation: 'PROUST*', '(ORGY SEE ROB)*'."""
    ans_clean = norm_letters(answer)

    # WORD* at start
    m = re.match(r'^([A-Z\s]+)\*', expl)
    if m:
        fodder = norm_letters(m.group(1))
        if sorted(fodder) == sorted(ans_clean):
            return {
                "wordplay_type": "anagram",
                "pieces": [{"clue_word": m.group(1).strip(), "letters": answer.upper(),
                            "mechanism": "anagram_fodder"}],
            }

    # (WORDS)* at start
    m = re.match(r'^\(([A-Z\s]+)\)\*', expl)
    if m:
        fodder = norm_letters(m.group(1))
        if sorted(fodder) == sorted(ans_clean):
            return {
                "wordplay_type": "anagram",
                "pieces": [{"clue_word": m.group(1).strip(), "letters": answer.upper(),
                            "mechanism": "anagram_fodder"}],
            }

    # *(word) or *(words) anywhere
    m = re.search(r'\*\(([^)]+)\)', expl)
    if m:
        fodder = norm_letters(m.group(1))
        if sorted(fodder) == sorted(ans_clean):
            return {
                "wordplay_type": "anagram",
                "pieces": [{"clue_word": m.group(1).strip(), "letters": answer.upper(),
                            "mechanism": "anagram_fodder"}],
            }

    return None


def try_parse_anagram_word(expl, answer):
    """Anagram with 'anagram of X' or 'anagram [indicator] of X'."""
    ans_clean = norm_letters(answer)

    # Match 'anagram of WORDS' or 'An anagram of WORDS'
    m = re.match(r'^[Aa]n?\s*[Aa]nagram\s+(?:\[[^\]]+\]\s*)?(?:of\s+)?(.+?)(?:\.\s|$)', expl)
    if m:
        fodder_text = m.group(1).strip()
        fodder = norm_letters(fodder_text)
        if sorted(fodder) == sorted(ans_clean):
            return {
                "wordplay_type": "anagram",
                "pieces": [{"clue_word": fodder_text, "letters": answer.upper(),
                            "mechanism": "anagram_fodder"}],
            }

    return None


def try_assemble_uppercase(expl, answer):
    """Extract all uppercase sequences and single uppercase letters, try to assemble to answer.

    This is the catch-all: if uppercase pieces in the explanation concatenate
    (in some order) or anagram to the answer, we have a parse.
    """
    ans_clean = norm_letters(answer).upper()
    if not ans_clean:
        return None

    # Extract all uppercase sequences (including single letters after specific patterns)
    # Get multi-char uppercase sequences
    all_upper = re.findall(r'[A-Z]{2,}', expl)
    # Get single uppercase letters that appear to be pieces (after + or before +, or in parens)
    singles = re.findall(r'(?:^|\+\s*|\(\s*)([A-Z])(?:\s*\+|\s*\(|\s*,|\s*$)', expl)
    all_pieces = [u for u in all_upper if u != ans_clean] + singles

    if not all_pieces:
        return None

    # Direct concatenation
    concat = ''.join(all_pieces)
    if concat == ans_clean:
        return _build_assembly_result(all_pieces, expl, answer, "charade")

    # Try subsets and permutations (cap at 8 pieces)
    if len(all_pieces) <= 8:
        for size in range(len(all_pieces), 1, -1):
            from itertools import combinations, permutations
            for combo in combinations(range(len(all_pieces)), size):
                pieces = [all_pieces[i] for i in combo]
                joined = ''.join(pieces)

                # Exact concatenation (in this order)
                if joined == ans_clean:
                    return _build_assembly_result(pieces, expl, answer, "charade")

                # Try permutations if same length
                if len(joined) == len(ans_clean) and size <= 6:
                    for perm in permutations(pieces):
                        if ''.join(perm) == ans_clean:
                            return _build_assembly_result(list(perm), expl, answer, "charade")

                # Anagram check
                if sorted(joined) == sorted(ans_clean):
                    is_anagram = ('*' in expl or 'anagram' in expl.lower() or
                                  'rearrang' in expl.lower() or 'mix' in expl.lower() or
                                  'shuffl' in expl.lower() or 'jumbl' in expl.lower())
                    if is_anagram:
                        return _build_assembly_result(pieces, expl, answer, "anagram")

    # Try with one piece reversed
    if 'revers' in expl.lower() or 'back' in expl.lower() or 'return' in expl.lower():
        for i in range(len(all_pieces)):
            modified = list(all_pieces)
            modified[i] = modified[i][::-1]
            concat = ''.join(modified)
            if concat == ans_clean:
                return _build_assembly_result(all_pieces, expl, answer, "reversal",
                                              reversed_idx=i)

    # Try container: X in Y or Y around X
    if len(all_pieces) >= 2:
        for i in range(len(all_pieces)):
            for j in range(len(all_pieces)):
                if i == j:
                    continue
                inner = all_pieces[i]
                outer = all_pieces[j]
                for k in range(1, len(outer)):
                    if outer[:k] + inner + outer[k:] == ans_clean:
                        return {
                            "wordplay_type": "container",
                            "pieces": [
                                _annotate_piece(outer, expl, "synonym"),
                                _annotate_piece(inner, expl, "synonym"),
                            ],
                        }

    return None


def _build_assembly_result(pieces, expl, answer, wordplay_type, reversed_idx=None):
    """Build result from assembled uppercase pieces."""
    annotated = []
    for i, p in enumerate(pieces):
        if reversed_idx is not None and i == reversed_idx:
            annotated.append(_annotate_piece(p, expl, "reversal"))
        else:
            annotated.append(_annotate_piece(p, expl, "synonym"))
    return {
        "wordplay_type": wordplay_type,
        "pieces": annotated,
    }


def _annotate_piece(letters, expl, default_mechanism):
    """Try to find the clue_word annotation for a piece from the explanation."""
    # Look for LETTERS (annotation) or LETTERS="annotation"
    patterns = [
        r'%s\s*\(([^)]+)\)' % re.escape(letters),
        r'%s\s*=\s*["\']?([^"\')+,]+)' % re.escape(letters),
        r'\(([^)]+)\)\s*%s' % re.escape(letters),
    ]
    for pat in patterns:
        m = re.search(pat, expl, re.IGNORECASE)
        if m:
            return {"clue_word": m.group(1).strip(), "letters": letters,
                    "mechanism": default_mechanism}

    return {"clue_word": letters, "letters": letters, "mechanism": default_mechanism}


def try_parse_deletion_bracket(expl, answer):
    """Deletion with brackets: CLERI[c], [L]awful, B[elief]."""
    ans_clean = norm_letters(answer).upper()

    # Find patterns like WORD[x] or [x]WORD or WORD[x]WORD
    bracket_parts = re.findall(r'([A-Z]*)\[([a-zA-Z]+)\]([A-Z]*)', expl)
    if not bracket_parts:
        return None

    # Build the result after removing bracketed letters
    assembled_parts = []
    for before, deleted, after in bracket_parts:
        assembled_parts.append(before + after)

    assembled = ''.join(assembled_parts)
    if not assembled:
        return None

    # Check if assembled pieces are in the answer
    if assembled == ans_clean:
        pieces = []
        for before, deleted, after in bracket_parts:
            full = before + deleted + after
            result = before + after
            pieces.append({
                "clue_word": full.lower(),
                "letters": result,
                "mechanism": "deletion",
            })
        return {"wordplay_type": "deletion", "pieces": pieces}

    return None


# ---------------------------------------------------------------------------
# Definition extraction from explanation text
# ---------------------------------------------------------------------------

def extract_definition_from_expl(expl, clue_text, answer, ref_db):
    """Try to extract the definition from the explanation text or from the clue.

    Strategies:
    1. Look for 'definition:' prefix in explanation
    2. Use RefDB to find which clue window is a synonym of the answer
    """
    # Strategy 1: explicit definition label
    m = re.search(r'[Dd]ef(?:inition)?[:\s]+["\']?(.+?)["\']?\s*(?:\.|$|,)', expl)
    if m:
        defn = m.group(1).strip().rstrip('.')
        if defn and len(defn) < 50:
            return defn

    # Strategy 2: RefDB lookup on clue windows
    text = strip_enumeration(clue_text)
    text = text.replace('\u2018', "'").replace('\u2019', "'")
    words = text.split()

    best_def = None
    best_len = 0
    max_window = min(4, len(words) - 1)

    for n in range(1, max_window + 1):
        window = " ".join(words[:n])
        if ref_db.is_definition_of(window, answer):
            if n > best_len:
                best_def = window
                best_len = n

        window = " ".join(words[-n:])
        if ref_db.is_definition_of(window, answer):
            if n > best_len:
                best_def = window
                best_len = n

    return best_def


# ---------------------------------------------------------------------------
# Main parse function
# ---------------------------------------------------------------------------

def parse_explanation(clue_text, answer, explanation, ref_db):
    """Parse a human explanation into structured data.

    Returns dict with wordplay_type, pieces, definition or None if unparseable.
    """
    ans_clean = norm_letters(answer).upper()
    if not ans_clean or not explanation:
        return None

    expl = explanation.strip()

    # Try each parser in priority order
    result = try_parse_dd(expl, answer)
    if result:
        return result

    result = try_parse_cryptic_def(expl, answer)
    if result:
        return result

    result = try_parse_hidden(expl, answer, clue_text)
    if result:
        return result

    result = try_parse_homophone(expl, answer)
    if result:
        return result

    result = try_parse_anagram_star(expl, answer)
    if result:
        return result

    result = try_parse_anagram_word(expl, answer)
    if result:
        return result

    result = try_parse_deletion_bracket(expl, answer)
    if result:
        return result

    # Catch-all: try assembling uppercase pieces
    result = try_assemble_uppercase(expl, answer)
    if result:
        return result

    return None


# ---------------------------------------------------------------------------
# Payload builder
# ---------------------------------------------------------------------------

def build_payload(definition, wordplay_type, pieces, explanation_text, row, confidence=1.0):
    def_start = None
    def_end = None
    if definition:
        idx = row["clue_text"].lower().find(definition.lower())
        if idx >= 0:
            def_start = idx
            def_end = idx + len(definition)

    return {
        "definition": definition,
        "wordplay_type": wordplay_type,
        "ai_explanation": explanation_text,
        "has_solution": 1,
        "reviewed": 1,
        "confidence": confidence,
        "components": {
            "ai_pieces": pieces,
            "assembly": {"op": wordplay_type},
            "wordplay_types": [wordplay_type],
        },
        "wordplay_types": [wordplay_type],
        "definition_start": def_start,
        "definition_end": def_end,
        "model_version": "mechanical_parse",
        "source": row["source"],
        "puzzle_number": row["puzzle_number"],
        "clue_number": row["clue_number"],
    }


def build_explanation_text(wordplay_type, pieces, definition, answer, original_expl):
    """Build explanation text, incorporating the original human explanation."""
    if wordplay_type == "double_definition":
        return "Double definition"
    if wordplay_type == "cryptic_definition":
        return "Cryptic definition"
    if wordplay_type == "homophone":
        return original_expl  # keep the human version
    if wordplay_type == "hidden" or wordplay_type == "hidden_reversed":
        return original_expl

    # For parsed types, build from pieces
    if pieces:
        parts = []
        for p in pieces:
            if p["mechanism"] == "anagram_fodder":
                parts.append(p["clue_word"].upper())
            elif p["mechanism"] == "deletion":
                parts.append("%s (deletion)" % p["letters"])
            else:
                parts.append('%s (%s="%s")' % (p["letters"], p["mechanism"], p["clue_word"]))

        if wordplay_type == "anagram":
            expl = "anagram of %s = %s" % (" + ".join(parts), answer.upper())
        elif wordplay_type == "container":
            expl = "%s in %s = %s" % (parts[1] if len(parts) > 1 else "?",
                                       parts[0] if parts else "?", answer.upper())
        else:
            expl = " + ".join(parts) + " = " + answer.upper()
    else:
        expl = original_expl

    if definition:
        expl += '; definition: "%s"' % definition

    return expl


# ---------------------------------------------------------------------------
# Batch runner
# ---------------------------------------------------------------------------

def load_clues(limit, source_filter):
    conn = sqlite3.connect(CLUES_DB, timeout=30)
    conn.row_factory = sqlite3.Row

    where = ("answer IS NOT NULL AND answer != '' "
             "AND clue_text IS NOT NULL AND clue_text != '' "
             "AND explanation IS NOT NULL AND explanation != '' "
             "AND (has_solution IS NULL OR has_solution = 0)")
    params = []
    if source_filter:
        where += " AND source = ?"
        params.append(source_filter)

    # Exclude telegraph — prose explanations, separate treatment
    where += " AND source != 'telegraph'"

    if limit:
        rows = conn.execute("""
            SELECT id, source, puzzle_number, clue_number, clue_text, answer,
                   enumeration, explanation
            FROM clues WHERE %s
            ORDER BY publication_date DESC
            LIMIT ?
        """ % where, params + [limit]).fetchall()
    else:
        rows = conn.execute("""
            SELECT id, source, puzzle_number, clue_number, clue_text, answer,
                   enumeration, explanation
            FROM clues WHERE %s
            ORDER BY publication_date DESC
        """ % where, params).fetchall()

    clues = [dict(r) for r in rows]
    conn.close()
    return clues


def run_batch(clues, ref_db, results_path):
    stats = {"dd": 0, "cryptic_def": 0, "hidden": 0, "homophone": 0,
             "anagram": 0, "charade": 0, "container": 0, "reversal": 0,
             "deletion": 0, "other_parsed": 0, "unparsed": 0}
    t0 = time.time()

    with open(results_path, "w", encoding="utf-8") as f:
        for i, row in enumerate(clues):
            answer = row["answer"]
            ans_clean = norm_letters(answer).upper()
            if not ans_clean:
                stats["unparsed"] += 1
                continue

            result = parse_explanation(row["clue_text"], ans_clean, row["explanation"], ref_db)

            if result:
                wtype = result["wordplay_type"]
                pieces = result.get("pieces", [])

                # Extract definition
                definition = extract_definition_from_expl(
                    row["explanation"], row["clue_text"], ans_clean, ref_db)

                expl_text = build_explanation_text(
                    wtype, pieces, definition, answer, row["explanation"])

                payload = build_payload(definition, wtype, pieces, expl_text, row)
                f.write(json.dumps({"clue_id": row["id"], "action": "high_solve",
                                    "payload": payload}) + "\n")

                # Track stats
                if wtype in stats:
                    stats[wtype] += 1
                else:
                    stats["other_parsed"] += 1
            else:
                stats["unparsed"] += 1

            if (i + 1) % 5000 == 0:
                elapsed = time.time() - t0
                solved = sum(v for k, v in stats.items() if k != "unparsed")
                print(f"  {i+1}/{len(clues)} ({elapsed:.0f}s) - {solved} parsed")

    elapsed = time.time() - t0
    solved = sum(v for k, v in stats.items() if k != "unparsed")

    print(f"\nDone in {elapsed:.0f}s")
    for k, v in sorted(stats.items(), key=lambda x: -x[1]):
        if v > 0:
            print(f"  {k:15} {v:>6,}")
    print(f"  Total parsed: {solved:,}")
    print(f"  Unparsed: {stats['unparsed']:,}")
    print(f"  Results written to: {results_path}")


def main():
    parser = argparse.ArgumentParser(description="Parse human explanations mechanically")
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--source", type=str, default=None)
    args = parser.parse_args()

    print("=" * 60)
    print("PARSE HUMAN EXPLANATIONS")
    if args.limit:
        print(f"  Limit: {args.limit}")
    if args.source:
        print(f"  Source: {args.source}")
    print("=" * 60)

    print("\nLoading clues from DB...")
    clues = load_clues(args.limit, args.source)
    print(f"Loaded {len(clues):,} clues with human explanations. DB connection closed.")

    print("Loading RefDB into memory...")
    ref_db = RefDB()
    print("RefDB loaded. All DB connections closed.\n")

    run_batch(clues, ref_db, RESULTS_PATH)


if __name__ == "__main__":
    main()
