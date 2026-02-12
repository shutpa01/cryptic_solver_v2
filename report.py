"""
Unified Puzzle Report Generator

Pulls clues from all pipeline stages, respects stage priority,
validates explanations independently, and only sends genuinely
unresolved clues to the API for review.

Usage:
    python puzzle_report.py
    python puzzle_report.py --run-id 0
    python puzzle_report.py --no-api          # skip API calls
    python puzzle_report.py --max-api 5       # limit API calls
"""

import argparse
import json
import os
import re
import sqlite3
import sys
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any, Tuple

# ============================================================
# USER CONFIGURATION
# ============================================================
PIPELINE_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\pipeline_stages.db"
CRYPTIC_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"
CLUES_DB = r"C:\Users\shuteclaude\PycharmProjects\cryptic_solver_V2\data\clues_master.db"
OUTPUT_FILE = r"C:\Users\shute\OneDrive\print\puzzle_report.txt"

# API configuration
USE_API = True
MAX_API_CALLS = 50  # safety limit
API_MODEL = "claude-sonnet-4-20250514"


# ============================================================
# DATA STRUCTURES
# ============================================================

@dataclass
class ClueResult:
    """Unified result for a single clue across all stages."""
    clue_id: int
    clue_number: str  # e.g. "14a"
    direction: str  # "across" or "down"
    clue_text: str
    answer: str  # canonical, always from cryptic_new.db
    enumeration: str

    # Resolution
    solved: bool = False
    solve_stage: str = ""  # dd, lurker, anagram, compound, general, api
    solve_quality: str = ""  # solved, partial, unsolved, cryptic_def, double_def

    # Explanation
    definition: str = ""
    formula: str = ""
    breakdown: List[str] = field(default_factory=list)
    word_roles: List[Dict] = field(default_factory=list)
    clue_type: str = ""

    # Gaps
    unresolved_words: List[str] = field(default_factory=list)
    letters_needed: str = ""
    gaps: List[Dict] = field(default_factory=list)

    # API review info
    api_reviewed: bool = False
    api_model: str = ""
    api_retried: bool = False
    api_confidence: float = 0.0


# ============================================================
# DATABASE HELPERS
# ============================================================

def get_canonical_clue_info(cryptic_db: str, clue_id: int) -> Optional[Dict]:
    """Get canonical clue info from cryptic_new.db."""
    conn = sqlite3.connect(cryptic_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT id, clue_number, direction, clue_text, answer, enumeration, "
        "source, puzzle_number FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    conn.close()
    if row:
        return dict(row)
    return None


def get_puzzle_info(pipeline_db: str, run_id: int) -> Dict:
    """Get puzzle metadata from pipeline_meta and stage_input."""
    conn = sqlite3.connect(pipeline_db)
    conn.row_factory = sqlite3.Row

    info = {"run_id": run_id}

    # Get from pipeline_meta
    for row in conn.execute("SELECT key, value FROM pipeline_meta"):
        info[row["key"]] = row["value"]

    # Get source/puzzle_number from first input row
    row = conn.execute(
        "SELECT source, puzzle_number FROM stage_input WHERE run_id = ? LIMIT 1",
        (run_id,)
    ).fetchone()
    if row:
        info["source"] = row["source"] or info.get("source", "unknown")
        info["puzzle_number"] = row["puzzle_number"] or info.get("puzzle_number", "")

    # Count total clues
    count = conn.execute(
        "SELECT COUNT(*) FROM stage_input WHERE run_id = ?", (run_id,)
    ).fetchone()[0]
    info["total_clues"] = count

    conn.close()
    return info


def get_all_clue_ids(pipeline_db: str, run_id: int) -> List[int]:
    """Get all clue_ids for a given run."""
    conn = sqlite3.connect(pipeline_db)
    rows = conn.execute(
        "SELECT DISTINCT clue_id FROM stage_input WHERE run_id = ? ORDER BY clue_id",
        (run_id,)
    ).fetchall()
    conn.close()
    return [r[0] for r in rows]


def get_latest_run_id(pipeline_db: str) -> int:
    """Get the most recent run_id."""
    conn = sqlite3.connect(pipeline_db)
    row = conn.execute("SELECT MAX(run_id) FROM stage_input").fetchone()
    conn.close()
    return row[0] if row[0] is not None else 0


# ============================================================
# STAGE CHECKERS - in cascade priority order
# ============================================================

def check_stage_dd(pipeline_db: str, run_id: int, clue_id: int, answer: str) -> Optional[
    ClueResult]:
    """Check if clue was solved by double definition stage."""
    conn = sqlite3.connect(pipeline_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM stage_dd WHERE run_id = ? AND clue_id = ? AND hit_found = 1",
        (run_id, clue_id)
    ).fetchone()
    conn.close()

    if not row:
        return None

    row = dict(row)

    result = ClueResult(
        clue_id=clue_id, clue_number="", direction="",
        clue_text=row["clue_text"], answer=answer, enumeration=""
    )
    result.solved = True
    result.solve_stage = "dd"
    result.solve_quality = "double_def"
    result.clue_type = "double_definition"

    # Parse windows for definitions
    windows = row["windows"]
    if windows:
        try:
            windows_list = json.loads(windows) if isinstance(windows, str) else windows
            if isinstance(windows_list, list) and len(windows_list) >= 2:
                result.definition = f"{windows_list[0]} / {windows_list[1]}"
                result.formula = f"Double definition: {windows_list[0]} + {windows_list[1]} = {answer}"
                result.breakdown = [
                    f'"{windows_list[0]}" = definition 1 for {answer}',
                    f'"{windows_list[1]}" = definition 2 for {answer}'
                ]
        except (json.JSONDecodeError, TypeError):
            result.definition = windows if isinstance(windows, str) else ""
            result.formula = f"Double definition = {answer}"

    return result


def check_stage_lurker(pipeline_db: str, run_id: int, clue_id: int, answer: str) -> \
Optional[ClueResult]:
    """Check if clue was solved by lurker (hidden word) stage."""
    conn = sqlite3.connect(pipeline_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM stage_lurker WHERE run_id = ? AND clue_id = ? AND hit_found = 1",
        (run_id, clue_id)
    ).fetchone()
    conn.close()

    if not row:
        return None

    row = dict(row)

    result = ClueResult(
        clue_id=clue_id, clue_number="", direction="",
        clue_text=row["clue_text"], answer=answer, enumeration=""
    )
    result.solved = True
    result.solve_stage = "lurker"
    result.solve_quality = "solved"
    result.clue_type = "hidden"

    container = row.get("container_text", "")
    result.formula = f"Hidden in: {container}"
    result.breakdown = [
        f'"{answer}" is hidden in the clue text',
        f'Container: {container}'
    ]

    return result


def check_stage_anagram(pipeline_db: str, run_id: int, clue_id: int, answer: str) -> \
Optional[ClueResult]:
    """Check if clue was solved by anagram stage."""
    conn = sqlite3.connect(pipeline_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM stage_anagram WHERE run_id = ? AND clue_id = ? AND hit_found = 1",
        (run_id, clue_id)
    ).fetchone()
    conn.close()

    if not row:
        return None

    row = dict(row)

    result = ClueResult(
        clue_id=clue_id, clue_number="", direction="",
        clue_text=row["clue_text"], answer=answer, enumeration=""
    )
    result.solved = True
    result.solve_stage = "anagram"
    result.solve_quality = "solved"
    result.clue_type = "anagram"

    fodder_words_raw = row.get("fodder_words", "[]")
    unused_words_raw = row.get("unused_words", "[]")
    solve_type = row.get("solve_type", "")

    try:
        fodder_words = json.loads(fodder_words_raw) if isinstance(fodder_words_raw,
                                                                  str) else fodder_words_raw
    except (json.JSONDecodeError, TypeError):
        fodder_words = []

    try:
        unused_words = json.loads(unused_words_raw) if isinstance(unused_words_raw,
                                                                  str) else unused_words_raw
    except (json.JSONDecodeError, TypeError):
        unused_words = []

    fodder_letters = row.get("fodder_letters", "")

    # Build formula
    fodder_desc = " + ".join(fodder_words) if fodder_words else fodder_letters
    result.formula = f"anagram({fodder_desc}) = {answer}"

    # Build breakdown — try to separate indicator from definition using position
    # In cryptic clues, definition is at the start or end.
    # The anagram indicator sits between definition and fodder.
    clue_words = re.findall(r"[a-zA-Z]+", result.clue_text)
    clue_words_lower = [w.lower() for w in clue_words]
    fodder_lower = set(w.lower() for w in fodder_words)
    unused_lower = [w.lower() for w in unused_words]

    # Find where fodder sits in the clue
    fodder_positions = [i for i, w in enumerate(clue_words_lower) if w in fodder_lower]
    unused_positions = {w: i for i, w in enumerate(clue_words_lower) if w in unused_lower}

    # Definition is at whichever end is furthest from the fodder
    definition_words = []
    indicator_words = []

    if fodder_positions and unused_positions:
        fodder_centre = sum(fodder_positions) / len(fodder_positions)
        clue_midpoint = len(clue_words) / 2

        for uw in unused_words:
            pos = unused_positions.get(uw.lower(), -1)
            if pos == -1:
                indicator_words.append(uw)
                continue

            # If fodder is near the end, definition is likely at the start (and vice versa)
            if fodder_centre > clue_midpoint:
                # Fodder at end, definition at start — early unused words are definition
                if pos < min(fodder_positions):
                    definition_words.append(uw)
                else:
                    indicator_words.append(uw)
            else:
                # Fodder at start, definition at end — late unused words are definition
                if pos > max(fodder_positions):
                    definition_words.append(uw)
                else:
                    indicator_words.append(uw)
    else:
        # Fallback: first unused word = definition, rest = indicators
        if unused_words:
            definition_words = [unused_words[0]]
            indicator_words = unused_words[1:]

    result.breakdown = []
    for fw in fodder_words:
        result.breakdown.append(f'"{fw}" = anagram fodder')
    for iw in indicator_words:
        result.breakdown.append(f'"{iw}" = anagram indicator')
    for dw in definition_words:
        result.breakdown.append(f'"{dw}" = definition for {answer}')

    result.definition = " ".join(definition_words) if definition_words else ""

    return result


def extract_letters(text: str) -> str:
    """Extract only uppercase letters from text."""
    return ''.join(c.upper() for c in text if c.isalpha())


def validate_formula_letters(formula: str, answer: str, clue_type: str = "") -> bool:
    """
    Check if the formula's letter math produces the answer.
    Returns True if valid, False if definitely wrong, None if can't check.
    """
    answer_clean = extract_letters(answer)
    if not answer_clean or not formula:
        return None

    # Skip validation for types we can't check
    if clue_type in ("cryptic_definition", "double_definition", "homophone", "hidden"):
        return None

    # Try to extract parts before = sign
    if "=" in formula:
        before_eq = formula.split("=")[0]
    else:
        before_eq = formula

    # For anagrams, check sorted letters match
    if "anagram" in formula.lower():
        # Extract fodder from anagram(X) notation
        match = re.search(r'anagram\(([^)]+)\)', formula, re.IGNORECASE)
        if match:
            fodder = extract_letters(match.group(1))
            return sorted(fodder) == sorted(answer_clean)

    # For charades (A + B + C = ANSWER), check concatenation
    if "+" in before_eq:
        parts = before_eq.split("+")
        combined = ''.join(extract_letters(p) for p in parts)
        if combined == answer_clean:
            return True
        # Also check if it's an anagram of the answer
        if sorted(combined) == sorted(answer_clean):
            return True

    return None


def check_stage_compound(pipeline_db: str, run_id: int, clue_id: int, answer: str) -> \
Optional[ClueResult]:
    """Check if clue was solved by compound wordplay stage."""
    conn = sqlite3.connect(pipeline_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM stage_compound WHERE run_id = ? AND clue_id = ?",
        (run_id, clue_id)
    ).fetchone()
    conn.close()

    if not row:
        return None

    row = dict(row)

    # Check if genuinely resolved
    fully_resolved = row.get("fully_resolved", 0)
    formula = row.get("formula", "") or ""
    db_answer = row.get("answer", "") or ""

    result = ClueResult(
        clue_id=clue_id, clue_number="", direction="",
        clue_text=row["clue_text"], answer=answer, enumeration=""
    )
    result.solve_stage = "compound"
    result.definition = row.get("definition_window", "") or ""
    result.formula = formula
    result.letters_needed = row.get("letters_still_needed", "") or ""

    # Parse breakdown
    breakdown_raw = row.get("breakdown", "[]")
    try:
        result.breakdown = json.loads(breakdown_raw) if isinstance(breakdown_raw,
                                                                   str) else (
                    breakdown_raw or [])
    except (json.JSONDecodeError, TypeError):
        result.breakdown = []

    # Parse word_roles
    word_roles_raw = row.get("word_roles", "[]")
    try:
        result.word_roles = json.loads(word_roles_raw) if isinstance(word_roles_raw,
                                                                     str) else (
                    word_roles_raw or [])
    except (json.JSONDecodeError, TypeError):
        result.word_roles = []

    # Parse unresolved
    unresolved_raw = row.get("unresolved_words") or row.get(
        "remaining_unresolved") or "[]"
    try:
        result.unresolved_words = json.loads(unresolved_raw) if isinstance(unresolved_raw,
                                                                           str) else (
                    unresolved_raw or [])
    except (json.JSONDecodeError, TypeError):
        result.unresolved_words = []

    # Validate: is this genuinely solved?
    if fully_resolved and formula and not result.letters_needed:
        # Check letter math
        valid = validate_formula_letters(formula, answer)
        if valid is not False:
            result.solved = True
            result.solve_quality = "solved"
        else:
            result.solved = False
            result.solve_quality = "partial"
    else:
        result.solved = False
        result.solve_quality = "partial"

    return result


def is_cryptic_definition(word_roles: List, breakdown: List, formula: str,
                          answer: str) -> bool:
    """
    Detect if this is a cryptic definition where the entire clue is the definition.
    CONSERVATIVE: Only returns True if the stage explicitly marked it as cryptic_definition
    with high confidence. We do NOT infer cryptic_definition just because all words
    are labelled 'definition' — that's more likely a sign the stage couldn't parse the
    wordplay, not that it's actually a cryptic definition.
    """
    if not word_roles and not breakdown:
        return False

    # Only trust word_roles if there's an explicit clue_type marker AND
    # the formula suggests cryptic definition (not just "? = ANSWER")
    if word_roles:
        roles = set()
        for wr in word_roles:
            if isinstance(wr, dict):
                roles.add(wr.get("role", ""))
        substantive_roles = roles - {"linker", "link", ""}
        # Must be ONLY definition roles, AND there must be no unresolved words
        # AND formula must be explicitly cryptic-def style
        if substantive_roles == {"definition"}:
            # Check if formula explicitly says cryptic definition
            if formula and "cryptic" in formula.lower():
                return True
            # Otherwise this is likely a failed parse — don't mark as solved
            return False

    return False


def check_stage_general(pipeline_db: str, run_id: int, clue_id: int, answer: str) -> \
Optional[ClueResult]:
    """Check if clue was solved by general wordplay stage."""
    conn = sqlite3.connect(pipeline_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM stage_general WHERE run_id = ? AND clue_id = ?",
        (run_id, clue_id)
    ).fetchone()
    conn.close()

    if not row:
        return None

    row = dict(row)

    fully_resolved = row.get("fully_resolved", 0)
    formula = row.get("formula", "") or ""
    definition = row.get("definition_window", "") or ""

    result = ClueResult(
        clue_id=clue_id, clue_number="", direction="",
        clue_text=row["clue_text"], answer=answer, enumeration=""
    )
    result.solve_stage = "general"
    result.definition = definition
    result.formula = formula
    result.letters_needed = row.get("letters_still_needed", "") or ""

    # Parse breakdown
    breakdown_raw = row.get("breakdown", "[]")
    try:
        result.breakdown = json.loads(breakdown_raw) if isinstance(breakdown_raw,
                                                                   str) else (
                    breakdown_raw or [])
    except (json.JSONDecodeError, TypeError):
        result.breakdown = []

    # Parse word_roles
    word_roles_raw = row.get("word_roles", "[]")
    try:
        result.word_roles = json.loads(word_roles_raw) if isinstance(word_roles_raw,
                                                                     str) else (
                    word_roles_raw or [])
    except (json.JSONDecodeError, TypeError):
        result.word_roles = []

    # Parse unresolved
    unresolved_raw = row.get("unresolved_words", "[]")
    try:
        result.unresolved_words = json.loads(unresolved_raw) if isinstance(unresolved_raw,
                                                                           str) else (
                    unresolved_raw or [])
    except (json.JSONDecodeError, TypeError):
        result.unresolved_words = []

    # Parse substitutions for partial context
    subs_raw = row.get("substitutions", "[]")
    try:
        found_subs = json.loads(subs_raw) if isinstance(subs_raw, str) else (
                    subs_raw or [])
    except (json.JSONDecodeError, TypeError):
        found_subs = []

    # Determine solve quality
    if is_cryptic_definition(result.word_roles, result.breakdown, formula, answer):
        result.solved = True
        result.solve_quality = "cryptic_def"
        result.clue_type = "cryptic_definition"
    elif fully_resolved and formula and not result.letters_needed:
        valid = validate_formula_letters(formula, answer)
        if valid is not False:
            result.solved = True
            result.solve_quality = "solved"
        else:
            result.solved = False
            result.solve_quality = "partial"
    else:
        result.solved = False
        result.solve_quality = "partial"

    return result


# ============================================================
# CASCADE RESOLVER
# ============================================================

def resolve_clue(pipeline_db: str, cryptic_db: str, run_id: int,
                 clue_id: int) -> ClueResult:
    """
    Resolve a clue through the cascade, respecting stage priority.
    Always uses canonical answer from cryptic_new.db.
    """
    # Get canonical info
    info = get_canonical_clue_info(cryptic_db, clue_id)
    if not info:
        return ClueResult(
            clue_id=clue_id, clue_number="?", direction="?",
            clue_text="(not found)", answer="(not found)", enumeration=""
        )

    answer = info["answer"]

    # Try each stage in priority order
    # 1. Double definition
    result = check_stage_dd(pipeline_db, run_id, clue_id, answer)
    if result and result.solved:
        result.clue_number = info["clue_number"]
        result.direction = info["direction"]
        result.enumeration = info.get("enumeration", "")
        return result

    # 2. Lurker (hidden word)
    result = check_stage_lurker(pipeline_db, run_id, clue_id, answer)
    if result and result.solved:
        result.clue_number = info["clue_number"]
        result.direction = info["direction"]
        result.enumeration = info.get("enumeration", "")
        return result

    # 3. Anagram
    result = check_stage_anagram(pipeline_db, run_id, clue_id, answer)
    if result and result.solved:
        result.clue_number = info["clue_number"]
        result.direction = info["direction"]
        result.enumeration = info.get("enumeration", "")
        return result

    # 4. Compound
    result = check_stage_compound(pipeline_db, run_id, clue_id, answer)
    if result and result.solved:
        result.clue_number = info["clue_number"]
        result.direction = info["direction"]
        result.enumeration = info.get("enumeration", "")
        return result

    # 5. General wordplay
    result = check_stage_general(pipeline_db, run_id, clue_id, answer)
    if result:
        result.clue_number = info["clue_number"]
        result.direction = info["direction"]
        result.enumeration = info.get("enumeration", "")
        return result

    # 6. Not found in any stage - use compound partial if available
    result = check_stage_compound(pipeline_db, run_id, clue_id, answer)
    if result:
        result.clue_number = info["clue_number"]
        result.direction = info["direction"]
        result.enumeration = info.get("enumeration", "")
        return result

    # Fallback: completely unresolved
    return ClueResult(
        clue_id=clue_id,
        clue_number=info["clue_number"],
        direction=info["direction"],
        clue_text=info["clue_text"],
        answer=answer,
        enumeration=info.get("enumeration", ""),
        solve_quality="unsolved"
    )


# ============================================================
# API REVIEWER (for unresolved clues only)
# ============================================================

def get_db_context_for_clue(cryptic_db: str, clue_text: str) -> Dict[str, Any]:
    """
    Query ALL tables in cryptic_new.db for every word in the clue.
    This is the fix for the original api_reviewer which only checked
    unresolved_words.
    """
    conn = sqlite3.connect(cryptic_db)
    cursor = conn.cursor()

    # Split clue into words, strip punctuation
    words = re.findall(r"[a-zA-Z]+", clue_text)

    context = {
        "synonyms": {},
        "wordplay": {},
        "indicators": {},
        "homophones": {},
        "definitions": {}
    }

    for word in words:
        w = word.lower()
        if len(w) < 2:
            # Still check single letters
            if len(w) == 1:
                context["synonyms"][w] = [w.upper()]
            continue

        # synonyms_pairs
        cursor.execute(
            "SELECT synonym FROM synonyms_pairs WHERE LOWER(word) = ? LIMIT 10", (w,)
        )
        syns = [r[0] for r in cursor.fetchall()]
        if syns:
            context["synonyms"][w] = syns

        # wordplay (abbreviations)
        cursor.execute(
            "SELECT substitution, category FROM wordplay WHERE LOWER(indicator) = ?", (w,)
        )
        abbrevs = cursor.fetchall()
        if abbrevs:
            context["wordplay"][w] = [{"sub": r[0], "cat": r[1]} for r in abbrevs]

        # indicators
        cursor.execute(
            "SELECT wordplay_type, subtype FROM indicators WHERE LOWER(word) = ?", (w,)
        )
        ind = cursor.fetchone()
        if ind:
            context["indicators"][w] = {"type": ind[0], "subtype": ind[1]}

        # homophones
        cursor.execute(
            "SELECT homophone FROM homophones WHERE LOWER(word) = ?", (w,)
        )
        homs = [r[0] for r in cursor.fetchall()]
        if homs:
            context["homophones"][w] = homs

        # definition_answers_augmented
        cursor.execute(
            "SELECT answer FROM definition_answers_augmented WHERE LOWER(definition) = ? LIMIT 10",
            (w,)
        )
        defs = [r[0] for r in cursor.fetchall()]
        if defs:
            context["definitions"][w] = defs

    conn.close()
    return context


def build_api_prompt(clue_result: ClueResult, db_context: Dict) -> str:
    """
    Build API prompt that includes existing partial work
    so the API fills gaps rather than starting from scratch.
    """
    # Build partial work summary
    partial_work = ""
    if clue_result.formula:
        partial_work += f"\nExisting formula: {clue_result.formula}"
    if clue_result.breakdown:
        partial_work += "\nExisting breakdown:"
        for b in clue_result.breakdown:
            partial_work += f"\n  {b}"
    if clue_result.unresolved_words:
        partial_work += f"\nWords still unresolved: {', '.join(clue_result.unresolved_words)}"
    if clue_result.letters_needed:
        partial_work += f"\nLetters still needed: {clue_result.letters_needed}"

    # Determine if this is an across or down clue for reversal guidance
    direction = clue_result.direction.lower() if clue_result.direction else "unknown"

    prompt = f"""You are an expert cryptic crossword solver. Your job is to explain how a cryptic clue produces its answer, with every word in the clue accounted for in a structured format.

CRYPTIC CROSSWORD PRINCIPLES:
- Every cryptic clue has a definition (usually at the start or end) and wordplay that builds the answer.
- Common wordplay types: charade (concatenation), anagram, container (one thing inside another), hidden word, reversal, deletion, homophone, double definition, cryptic definition.
- Abbreviations must be standard and widely recognised: cricket (C=caught, B=bowled, LBW), military (RE, RA, CO), music (P, PP, F, FF, MP), compass (N,S,E,W), NATO alphabet (Alpha=A, Charlie=C), units, Roman numerals, etc. Do NOT use "first letter of a random word" as a substitution unless there is an explicit first-letter indicator in the clue (e.g. "initially", "first", "head of").
- Reversal indicators in Down clues can indicate upward direction: "up", "arise", "raised", "climbing", "ascending" etc.
- Reversal indicators in Across clues indicate reading backwards: "back", "returning", "reflected" etc.
- Container indicators: "in", "around", "about", "holding", "clutching", "swallowing" etc.
- Anagram indicators suggest disorder: "wild", "broken", "crazy", "mixed", "drunk", "rewritten" etc.
- Deletion indicators: "losing", "without", "lacking", "not", "dropping" etc.

THIS IS A {direction.upper()} CLUE.

DATABASE LOOKUPS FOR CLUE WORDS:
{json.dumps(db_context, indent=2)}

This database contains ~78,000 synonym pairs, ~2,700 indicators, and extensive abbreviation and homophone tables. It is comprehensive but not exhaustive.

HOW TO USE THE DATABASE:
1. SEARCH THOROUGHLY. If an exact word isn't found, look for close variants in the results provided. "deer" might not be there but "american deer" -> ELK will be. "mushroom" might appear as "mushrooms" or "type of mushroom" -> CEPS.
2. TRUST THE DATABASE when it offers a match - prefer DB-backed substitutions over your own guesses.
3. DON'T FREEZE when the DB has a genuine gap. If you're confident in a standard, well-known cryptic substitution that simply isn't in the DB, use it and flag it as a gap.
4. If you can't find a substitution in the DB even approximately, treat that as a signal to question whether your approach is correct before assuming it's a gap.

HONESTY RULES:
- If you cannot confidently determine a word's role, assign it role "unknown". An honest unknown is far more valuable than a fabricated explanation.
- Never invent nonsense abbreviations or obscure synonym chains to force an answer to work. If a word doesn't map to any recognised abbreviation or synonym that fits, don't force it.
- If your confidence is below 50%, mark uncertain word roles as "unknown" rather than guessing.
- It is better to return a partially explained clue than a fully explained wrong one.

CLUE: {clue_result.clue_text}
ANSWER: {clue_result.answer} ({len(extract_letters(clue_result.answer))} letters)

{f"PARTIAL WORK ALREADY DONE:{partial_work}" if partial_work else ""}

Respond with JSON only, no markdown:

{{
  "clue_type": "charade|anagram|container|deletion|reversal|homophone|hidden|double_definition|cryptic_definition",
  "definition": "the definition word(s) from the clue",
  "formula": "traces letter-by-letter to answer, e.g. CON + TACT = CONTACT or anagram(PARTIES) = TRAIPSE or P(ELIC)AN = PELICAN",
  "word_roles": [
    {{
      "word": "exact word from clue",
      "role": "definition|fodder|indicator|linker|unknown",
      "produces": "LETTERS (required if role=fodder)",
      "substitution_type": "synonym|abbreviation|literal|first_letter|last_letter|reversal|anagram|deletion|container|homophone (required if role=fodder)",
      "in_db": true or false,
      "explanation": "brief justification, e.g. 'caught = C in cricket' or 'not in DB but standard abbreviation'"
    }}
  ],
  "gaps": [
    {{
      "table": "synonyms_pairs|wordplay|indicators|homophones",
      "data": {{"word": "...", "synonym": "..."}}
    }}
  ],
  "confidence": 0.0 to 1.0
}}

FORMULA NOTATION:
- Charade: A + B + C = ANSWER
- Anagram: anagram(FODDER) = ANSWER
- Container: A(B)C = ANSWER where B is inside AC
- Reversal: reverse(WORD) = DROW
- Deletion: WORD - W = ORD
- Homophone: homophone(PHRASE) = ANSWER
- Combinations are fine: A + reverse(B) + C = ANSWER"""

    return prompt


def build_retry_prompt(clue_result: ClueResult, previous_response: Dict,
                       validation_message: str, db_context: Dict) -> str:
    """
    Build retry prompt WITH db context and specific guidance for rethinking.
    """
    prev_formula = previous_response.get("formula", "N/A")
    prev_roles = previous_response.get("word_roles", [])

    direction = clue_result.direction.lower() if clue_result.direction else "unknown"

    prompt = f"""You are an expert cryptic crossword solver. Your previous attempt to explain this clue failed validation.

CLUE: {clue_result.clue_text}
ANSWER: {clue_result.answer} ({len(extract_letters(clue_result.answer))} letters)
THIS IS A {direction.upper()} CLUE.

YOUR PREVIOUS ATTEMPT:
Formula: {prev_formula}
Word roles: {json.dumps(prev_roles, indent=2)}

WHY IT FAILED:
{validation_message}

DATABASE LOOKUPS (search thoroughly for close matches):
{json.dumps(db_context, indent=2)}

BEFORE TRYING AGAIN, consider:
- Is your clue type wrong? Could this be a container rather than a charade? A deletion rather than an anagram?
- Are you using the right definition window? Try the other end of the clue.
- In a Down clue, could a word like "up", "arise", "raised" be a reversal indicator?
- Is the answer hidden inside consecutive words of the clue?
- Could this be a homophone clue? Look for indicators like "heard", "say", "broadcast", "sounds like".
- Are there standard abbreviations you're missing? Check the DB carefully for close matches.

HONESTY RULES:
- If you cannot confidently determine a word's role, assign it role "unknown".
- Never invent nonsense abbreviations or obscure synonym chains to force an answer to work.
- It is better to return a partially explained clue than a fully explained wrong one.

Respond with JSON only, no markdown:

{{
  "clue_type": "charade|anagram|container|deletion|reversal|homophone|hidden|double_definition|cryptic_definition",
  "definition": "the definition word(s) from the clue",
  "formula": "traces letter-by-letter to answer, e.g. CON + TACT = CONTACT or P(ELIC)AN = PELICAN",
  "word_roles": [
    {{
      "word": "exact word from clue",
      "role": "definition|fodder|indicator|linker|unknown",
      "produces": "LETTERS (required if role=fodder)",
      "substitution_type": "type (required if role=fodder)",
      "in_db": true or false,
      "explanation": "brief justification"
    }}
  ],
  "gaps": [
    {{
      "table": "synonyms_pairs|wordplay|indicators|homophones",
      "data": {{"word": "...", "synonym": "..."}}
    }}
  ],
  "confidence": 0.0 to 1.0
}}"""

    return prompt



def validate_api_response(response: Dict, answer: str) -> Tuple[bool, str]:
    """
    Validate an API response's letter math.
    Returns (valid, message).
    """
    answer_clean = extract_letters(answer)
    clue_type = response.get("clue_type", "")
    word_roles = response.get("word_roles", [])

    # Skip validation for unvalidatable types
    if clue_type in ("cryptic_definition", "double_definition", "homophone", "hidden"):
        return True, f"? {clue_type} - cannot validate letter math"

    # Extract fodder letters
    fodder_letters = []
    for wr in word_roles:
        if wr.get("role") == "fodder" and wr.get("produces"):
            fodder_letters.append(extract_letters(wr["produces"]))

    if not fodder_letters:
        return False, "No fodder words produce any letters"

    combined = ''.join(fodder_letters)

    # For anagrams, check sorted
    if clue_type == "anagram":
        if sorted(combined) == sorted(answer_clean):
            return True, f"anagram({combined}) = {answer_clean}"
        else:
            return False, f"anagram({combined}) != {answer_clean}"

    # For charades, check direct concatenation
    if combined == answer_clean:
        return True, f"{' + '.join(fodder_letters)} = {answer_clean}"

    # Check if it's an anagram match (container, etc.)
    if sorted(combined) == sorted(answer_clean):
        return True, f"{' + '.join(fodder_letters)} (rearranged) = {answer_clean}"

    return False, f"{' + '.join(fodder_letters)} = {combined} != {answer_clean}"


def call_api(prompt: str, model: str) -> Optional[Dict]:
    """Make a single API call and parse the JSON response."""
    try:
        import anthropic
        from dotenv import load_dotenv
        load_dotenv()

        client = anthropic.Anthropic(api_key=os.getenv("ANTHROPIC_API_KEY"))

        response = client.messages.create(
            model=model,
            max_tokens=2000,
            messages=[{"role": "user", "content": prompt}]
        )

        text = response.content[0].text.strip()

        # Strip markdown wrapping if present
        if "```json" in text:
            text = text.split("```json")[1].split("```")[0]
        elif "```" in text:
            text = text.split("```")[1].split("```")[0]

        # Find JSON object
        text = text.strip()
        if not text.startswith("{"):
            start = text.find("{")
            if start != -1:
                depth = 0
                end = start
                for i, c in enumerate(text[start:], start):
                    if c == "{":
                        depth += 1
                    elif c == "}":
                        depth -= 1
                        if depth == 0:
                            end = i + 1
                            break
                text = text[start:end]

        return json.loads(text)

    except Exception as e:
        return {"error": str(e)}


def review_with_api(clue_result: ClueResult, cryptic_db: str, model: str) -> ClueResult:
    """
    Send a clue to the API for review. Includes validation + retry.
    """
    # Get DB context for ALL words in the clue
    db_context = get_db_context_for_clue(cryptic_db, clue_result.clue_text)

    # Pass 1
    prompt = build_api_prompt(clue_result, db_context)
    response = call_api(prompt, model)

    if not response or "error" in response:
        clue_result.api_reviewed = True
        clue_result.api_model = model
        return clue_result

    # Validate
    valid, message = validate_api_response(response, clue_result.answer)

    if valid:
        # Apply API result
        apply_api_result(clue_result, response)
        clue_result.api_reviewed = True
        clue_result.api_model = model
        return clue_result

    # Pass 2: Retry with feedback + DB context
    retry_prompt = build_retry_prompt(clue_result, response, message, db_context)
    retry_response = call_api(retry_prompt, model)

    if not retry_response or "error" in retry_response:
        # Use pass 1 result anyway
        apply_api_result(clue_result, response)
        clue_result.api_reviewed = True
        clue_result.api_model = model
        return clue_result

    retry_valid, retry_message = validate_api_response(retry_response, clue_result.answer)

    if retry_valid:
        apply_api_result(clue_result, retry_response)
        clue_result.api_retried = True
    else:
        # Use whichever had higher confidence
        p1_conf = response.get("confidence", 0)
        p2_conf = retry_response.get("confidence", 0)
        apply_api_result(clue_result, retry_response if p2_conf >= p1_conf else response)
        clue_result.api_retried = True

    clue_result.api_reviewed = True
    clue_result.api_model = model
    return clue_result


def apply_api_result(clue_result: ClueResult, response: Dict):
    """Apply API response data to the clue result."""
    clue_result.clue_type = response.get("clue_type", clue_result.clue_type)
    clue_result.formula = response.get("formula", clue_result.formula)
    clue_result.definition = response.get("definition", clue_result.definition)
    clue_result.api_confidence = response.get("confidence", 0)

    word_roles = response.get("word_roles", [])
    has_unknowns = False
    if word_roles:
        clue_result.word_roles = word_roles
        # Rebuild breakdown from word_roles
        clue_result.breakdown = []
        for wr in word_roles:
            word = wr.get("word", "?")
            role = wr.get("role", "?")
            in_db = wr.get("in_db", None)

            # DB status symbol
            if in_db is True:
                db_sym = "✓"
            elif in_db is False:
                db_sym = "⚠️"
            else:
                db_sym = ""

            if role == "definition":
                clue_result.breakdown.append(
                    f'"{word}" = definition for {clue_result.answer}')
            elif role == "fodder":
                produces = wr.get("produces", "?")
                sub_type = wr.get("substitution_type", "")
                explanation = wr.get("explanation", "")
                db_note = f" {db_sym}" if db_sym else ""
                clue_result.breakdown.append(
                    f'{db_note} "{word}" → {produces} ({sub_type}){" — " + explanation if explanation else ""}')
            elif role == "indicator":
                ind_type = wr.get("explanation", wr.get("indicator_type", "indicator"))
                db_note = f" {db_sym}" if db_sym else ""
                clue_result.breakdown.append(f'{db_note} "{word}" = {ind_type}')
            elif role == "linker":
                clue_result.breakdown.append(f'"{word}" = linker')
            elif role == "unknown":
                has_unknowns = True
                clue_result.breakdown.append(f'? "{word}" = unknown role')

    # Collect gaps
    gaps = response.get("gaps", [])
    if gaps:
        clue_result.gaps = gaps

    # Determine if now solved — requires BOTH letter validation AND confidence >= 0.5
    valid, _ = validate_api_response(response, clue_result.answer)
    confidence = response.get("confidence", 0)

    if valid and confidence >= 0.5 and not has_unknowns:
        clue_result.solved = True
        clue_result.solve_quality = "solved"
        clue_result.solve_stage = "api"
    elif valid and (confidence < 0.5 or has_unknowns):
        # Letters work but low confidence or unknowns — partial
        clue_result.solved = False
        clue_result.solve_quality = "partial"
        clue_result.solve_stage = "api"
    else:
        clue_result.solved = False
        clue_result.solve_quality = "partial"
        clue_result.solve_stage = "api"


# ============================================================
# REPORT FORMATTER
# ============================================================

def sort_key(clue: ClueResult) -> tuple:
    """Sort clues by direction (across first) then by number."""
    num = clue.clue_number.rstrip("adAD")
    try:
        num_int = int(num)
    except ValueError:
        num_int = 999
    direction_order = 0 if clue.direction.lower() == "across" else 1
    return (direction_order, num_int)


def format_report(results: List[ClueResult], puzzle_info: Dict) -> str:
    """Format the complete puzzle report."""
    lines = []

    source = puzzle_info.get("source", "unknown")
    puzzle_num = puzzle_info.get("puzzle_number", "")
    total = len(results)

    # ---- HEADER ----
    lines.append("=" * 80)
    lines.append(f"PUZZLE REPORT: {source} #{puzzle_num}")
    lines.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"Total clues: {total}")
    lines.append("=" * 80)

    # ---- SUMMARY ----
    solved_count = sum(1 for r in results if r.solved)

    # Count by stage
    stage_counts = {}
    for r in results:
        if r.solved:
            stage = r.solve_stage
            stage_counts[stage] = stage_counts.get(stage, 0) + 1

    partial_count = sum(
        1 for r in results if not r.solved and r.solve_quality == "partial")
    unsolved_count = sum(
        1 for r in results if not r.solved and r.solve_quality != "partial")
    api_reviewed = sum(1 for r in results if r.api_reviewed)
    api_retried = sum(1 for r in results if r.api_retried)

    lines.append("")
    lines.append("SOLVE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Double definitions:      {stage_counts.get('dd', 0)}")
    lines.append(f"  Hidden words (lurkers):  {stage_counts.get('lurker', 0)}")
    lines.append(f"  Anagrams:                {stage_counts.get('anagram', 0)}")
    lines.append(f"  Compound wordplay:       {stage_counts.get('compound', 0)}")
    lines.append(f"  General wordplay:        {stage_counts.get('general', 0)}")
    lines.append(f"  API reviewed:            {stage_counts.get('api', 0)}")
    lines.append(f"  ---")
    lines.append(
        f"  SOLVED:                  {solved_count}/{total} ({100 * solved_count // total}%)")
    lines.append(f"  Partial (needs work):    {partial_count}")
    lines.append(f"  Unsolved:                {unsolved_count}")
    if api_reviewed:
        lines.append(
            f"  API calls made:          {api_reviewed} (retried: {api_retried})")
    lines.append("")

    # ---- PER-CLUE DETAIL ----
    # Sort: across by number, then down by number
    sorted_results = sorted(results, key=sort_key)

    lines.append("=" * 80)
    lines.append("ACROSS")
    lines.append("=" * 80)

    current_direction = "across"
    for r in sorted_results:
        if r.direction.lower() != current_direction:
            current_direction = r.direction.lower()
            lines.append("")
            lines.append("=" * 80)
            lines.append("DOWN")
            lines.append("=" * 80)

        lines.append("")
        lines.append(format_single_clue(r))

    # ---- PENDING DB ADDITIONS ----
    all_gaps = []
    for r in results:
        for gap in r.gaps:
            all_gaps.append((r.clue_text, r.answer, gap))

    if all_gaps:
        lines.append("")
        lines.append("=" * 80)
        lines.append("PENDING DATABASE ADDITIONS (review before applying)")
        lines.append("=" * 80)
        lines.append("")

        for clue_text, answer, gap in all_gaps:
            table = gap.get("table", "?")
            data = gap.get("data", {})
            lines.append(f"  -- From: {clue_text[:50]} -> {answer}")

            if table == "synonyms_pairs":
                word = data.get("word", "?")
                synonym = data.get("synonym", "?")
                lines.append(
                    f"  INSERT INTO synonyms_pairs (word, synonym) VALUES ('{word}', '{synonym}');")
            elif table == "wordplay":
                indicator = data.get("word", data.get("indicator", "?"))
                sub = data.get("substitution", "?")
                cat = data.get("category", "abbreviation")
                lines.append(
                    f"  INSERT INTO wordplay (indicator, substitution, category) VALUES ('{indicator}', '{sub}', '{cat}');")
            elif table == "indicators":
                word = data.get("word", "?")
                wp_type = data.get("wordplay_type", "?")
                lines.append(
                    f"  INSERT INTO indicators (word, wordplay_type) VALUES ('{word}', '{wp_type}');")
            elif table == "homophones":
                word = data.get("word", "?")
                hom = data.get("homophone", "?")
                lines.append(
                    f"  INSERT INTO homophones (word, homophone) VALUES ('{word}', '{hom}');")
            lines.append("")

    lines.append("")
    lines.append("=" * 80)
    lines.append("END OF REPORT")
    lines.append("=" * 80)

    return "\n".join(lines)


def format_single_clue(r: ClueResult) -> str:
    """Format a single clue for the report."""
    lines = []

    # Status tag
    if r.solved:
        tag = "[SOLVED]"
        if r.solve_stage == "api":
            tag = "[SOLVED*]"  # solved by API
    else:
        tag = "[PARTIAL]" if r.solve_quality == "partial" else "[UNSOLVED]"

    stage_label = {
        "dd": "double definition",
        "lurker": "hidden word",
        "anagram": "anagram",
        "compound": "compound wordplay",
        "general": "general wordplay",
        "api": "API review"
    }.get(r.solve_stage, r.solve_stage)

    lines.append(f"  {tag} {r.clue_number} CLUE: {r.clue_text}")
    lines.append(f"  ANSWER: {r.answer}")
    lines.append(f"  STAGE: {stage_label}")

    if r.clue_type:
        lines.append(f"  TYPE: {r.clue_type}")

    if r.definition:
        lines.append(f"  DEFINITION: {r.definition}")

    if r.formula:
        lines.append(f"  FORMULA: {r.formula}")

    if r.breakdown:
        lines.append(f"  BREAKDOWN:")
        for b in r.breakdown:
            if isinstance(b, str):
                lines.append(f"    {b}")
            elif isinstance(b, dict):
                lines.append(f"    {json.dumps(b)}")

    if r.letters_needed and not r.solved:
        lines.append(f"  LETTERS NEEDED: {r.letters_needed}")

    if r.unresolved_words and not r.solved:
        lines.append(f"  UNRESOLVED: {', '.join(r.unresolved_words)}")

    if r.api_reviewed:
        model_label = "Sonnet"
        conf = f"{r.api_confidence:.0%}" if r.api_confidence else "N/A"
        retry_note = " (retried)" if r.api_retried else ""
        conf_warning = " ⚠️ LOW CONFIDENCE" if r.api_confidence and r.api_confidence < 0.5 else ""
        lines.append(f"  API: {model_label}{retry_note}, confidence: {conf}{conf_warning}")

    lines.append(f"  {'─' * 70}")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate unified puzzle report")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Pipeline run ID (default: latest)")
    parser.add_argument("--no-api", action="store_true", help="Skip API calls")
    parser.add_argument("--max-api", type=int, default=MAX_API_CALLS,
                        help="Max API calls")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output file")
    args = parser.parse_args()

    # Get run ID
    run_id = args.run_id if args.run_id is not None else get_latest_run_id(PIPELINE_DB)
    print(f"Using run_id: {run_id}")

    # Get puzzle info
    puzzle_info = get_puzzle_info(PIPELINE_DB, run_id)
    print(
        f"Puzzle: {puzzle_info.get('source', '?')} #{puzzle_info.get('puzzle_number', '?')}")

    # Get all clue IDs
    clue_ids = get_all_clue_ids(PIPELINE_DB, run_id)
    print(f"Total clues: {len(clue_ids)}")

    # Resolve each clue through cascade
    results = []
    for cid in clue_ids:
        result = resolve_clue(PIPELINE_DB, CLUES_DB, run_id, cid)
        results.append(result)

    # Report pre-API status
    solved_pre = sum(1 for r in results if r.solved)
    unsolved = [r for r in results if not r.solved]
    print(f"\nPre-API: {solved_pre}/{len(results)} solved")
    print(f"Clues needing API review: {len(unsolved)}")

    # API review for unresolved clues
    if not args.no_api and unsolved:
        api_count = min(len(unsolved), args.max_api)
        print(f"\nSending {api_count} clues to API ({API_MODEL})...")

        for i, r in enumerate(unsolved[:api_count]):
            print(f"  [{i + 1}/{api_count}] {r.clue_number} {r.answer}...", end=" ")
            review_with_api(r, CRYPTIC_DB, API_MODEL)
            status = "SOLVED" if r.solved else "partial"
            retry = " (retried)" if r.api_retried else ""
            print(f"{status}{retry}")

    # Generate report
    report = format_report(results, puzzle_info)

    # Write to file
    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {args.output}")

    # Also print summary
    solved_total = sum(1 for r in results if r.solved)
    print(
        f"\nFinal: {solved_total}/{len(results)} solved ({100 * solved_total // len(results)}%)")


if __name__ == "__main__":
    main()