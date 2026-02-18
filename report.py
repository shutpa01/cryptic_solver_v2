"""
Unified Puzzle Report Generator — single entry point for the solver.

Runs the full pipeline (DD, lurker, anagram, compound, general),
then generates a report with per-clue explanations.

Usage:
    python report.py
    python report.py --source guardian --puzzle-number 29927
    python report.py --max-clues 10
    python report.py --report-only --run-id 0
"""

import argparse
import json
import re
import sqlite3
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional, List, Dict, Any

# ============================================================
# USER CONFIGURATION
# ============================================================
PIPELINE_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\pipeline_stages.db"
CRYPTIC_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\cryptic_new.db"
CLUES_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db"
OUTPUT_FILE = r"C:\Users\shute\PycharmProjects\cryptic_solver_V2\documents\puzzle_report.txt"

# ============================================================
# RUN CRITERIA (edit these or override via CLI args)
# ============================================================
SOURCE = "telegraph"          # telegraph, guardian, times, independent
PUZZLE_NUMBER = "31164"       # puzzle number to solve
MAX_CLUES = 100             # max clues to select


EXCLUDE_SOLVED = True        # skip clues that already have a solution
WORDPLAY_TYPE = "all"         # all, anagram, lurker, dd
SINGLE_CLUE_MATCH =""        #
# filter to single clue matching this text
USE_KNOWN_ANSWER = True       # use known answer as candidate
ONLY_MISSING_DEFINITION = False  # only clues where answer NOT in def candidates
MAX_DISPLAY = 50              # max clues to print
ANALYZE_FORWARDED_ANAGRAMS = False
MAX_FORWARDED_SAMPLES = 50
ANALYZE_SUCCESSFUL_ANAGRAMS = False
MAX_SUCCESSFUL_SAMPLES = 25
ENABLE_PERSISTENCE = True     # save stage data to SQLite


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
    solve_stage: str = ""  # dd, lurker, anagram, compound, general
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


def get_all_puzzle_clue_ids(clues_db: str, source: str, puzzle_number: str) -> List[int]:
    """Get ALL clue_ids for a puzzle from the clues table."""
    conn = sqlite3.connect(clues_db)
    rows = conn.execute(
        "SELECT id FROM clues WHERE source = ? AND puzzle_number = ? ORDER BY id",
        (source, puzzle_number)
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
    if fully_resolved and formula and not result.letters_needed and not has_unresolved_indicators(result.unresolved_words):
        # CRITICAL: Any unresolved indicator = NOT SOLVED
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


def has_unresolved_indicators(unresolved_words: List[str]) -> bool:
    """
    Check if any unresolved words are known indicators.
    CRITICAL RULE: Any unresolved indicator = NOT SOLVED.
    """
    if not unresolved_words:
        return False

    conn = sqlite3.connect(CRYPTIC_DB)
    cursor = conn.cursor()

    for word in unresolved_words:
        cursor.execute(
            "SELECT 1 FROM indicators WHERE word = ?", (word,))
        if cursor.fetchone():
            conn.close()
            return True

    conn.close()
    return False


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
    elif fully_resolved and formula and not result.letters_needed and not has_unresolved_indicators(result.unresolved_words):
        # CRITICAL: Any unresolved indicator = NOT SOLVED
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


def check_stage_secondary(pipeline_db: str, run_id: int, clue_id: int,
                          answer: str) -> Optional[ClueResult]:
    """Check if clue was solved by secondary stage."""
    conn = sqlite3.connect(pipeline_db)
    conn.row_factory = sqlite3.Row
    row = conn.execute(
        "SELECT * FROM stage_secondary WHERE run_id = ? AND clue_id = ? AND fully_resolved = 1",
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
    result.solve_stage = "secondary"
    result.solve_quality = "solved"

    result.formula = row.get("improved_formula", "") or ""
    result.definition = ""

    # Parse breakdown
    breakdown_raw = row.get("breakdown", "[]")
    try:
        result.breakdown = json.loads(breakdown_raw) if isinstance(breakdown_raw, str) else (
                    breakdown_raw or [])
    except (json.JSONDecodeError, TypeError):
        result.breakdown = []

    # Parse word_roles
    word_roles_raw = row.get("word_roles", "[]")
    try:
        result.word_roles = json.loads(word_roles_raw) if isinstance(word_roles_raw, str) else (
                    word_roles_raw or [])
    except (json.JSONDecodeError, TypeError):
        result.word_roles = []

    result.clue_type = row.get("helper_used", "")

    return result


# ============================================================
# CASCADE RESOLVER
# ============================================================

def resolve_clue(pipeline_db: str, cryptic_db: str, run_id: int,
                 clue_id: int) -> ClueResult:
    """
    Resolve a clue through the cascade, respecting stage priority.
    Always uses canonical answer from cryptic_new.db.
    Checks solved_clues table for previously solved clues not in this run.
    """
    # Get canonical info
    info = get_canonical_clue_info(cryptic_db, clue_id)
    if not info:
        return ClueResult(
            clue_id=clue_id, clue_number="?", direction="?",
            clue_text="(not found)", answer="(not found)", enumeration=""
        )

    answer = info["answer"]

    # Check if clue was already solved in a previous run
    # (before checking current run stages)
    conn = sqlite3.connect(cryptic_db)
    conn.row_factory = sqlite3.Row
    solved_row = conn.execute(
        """SELECT solve_stage, solve_quality, clue_type, definition,
                  formula, breakdown, word_roles, letters_needed, unresolved_words
           FROM solved_clues WHERE clue_id = ?""",
        (clue_id,)
    ).fetchone()
    conn.close()

    if solved_row:
        # Check if this clue was processed in current run
        # (current run takes precedence over historical solves)
        conn2 = sqlite3.connect(pipeline_db)
        in_current_run = conn2.execute(
            "SELECT 1 FROM stage_input WHERE run_id = ? AND clue_id = ?",
            (run_id, clue_id)
        ).fetchone()
        conn2.close()

        # If NOT in current run, use the historical solve
        if not in_current_run:
            return ClueResult(
                clue_id=clue_id,
                clue_number=info["clue_number"],
                direction=info["direction"],
                clue_text=info["clue_text"],
                answer=answer,
                enumeration=info.get("enumeration", ""),
                solved=True,  # Mark as solved
                solve_stage=solved_row["solve_stage"],
                solve_quality=solved_row["solve_quality"],  # Use original quality
                clue_type=solved_row["clue_type"],
                definition=solved_row["definition"],
                formula=solved_row["formula"],
                breakdown=json.loads(solved_row["breakdown"]) if solved_row["breakdown"] else [],
                word_roles=json.loads(solved_row["word_roles"]) if solved_row["word_roles"] else [],
                letters_needed=solved_row["letters_needed"],
                unresolved_words=json.loads(solved_row["unresolved_words"]) if solved_row["unresolved_words"] else []
            )

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
    if result and result.solved:
        result.clue_number = info["clue_number"]
        result.direction = info["direction"]
        result.enumeration = info.get("enumeration", "")
        return result

    # 6. Secondary (helpers on general failures)
    result = check_stage_secondary(pipeline_db, run_id, clue_id, answer)
    if result and result.solved:
        result.clue_number = info["clue_number"]
        result.direction = info["direction"]
        result.enumeration = info.get("enumeration", "")
        return result

    # 7. General wordplay (unsolved partial)
    result = check_stage_general(pipeline_db, run_id, clue_id, answer)
    if result:
        result.clue_number = info["clue_number"]
        result.direction = info["direction"]
        result.enumeration = info.get("enumeration", "")
        return result

    # 8. Not found in any stage - use compound partial if available
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
# REPORT FORMATTER
# ============================================================

def sort_key(clue: ClueResult) -> tuple:
    """Sort clues by direction (across first) then by number."""
    num = (clue.clue_number or "").rstrip("adAD")
    try:
        num_int = int(num)
    except ValueError:
        num_int = 999
    direction_order = 0 if (clue.direction or "").lower() == "across" else 1
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
    lines.append("")
    lines.append("SOLVE SUMMARY")
    lines.append("-" * 40)
    lines.append(f"  Double definitions:      {stage_counts.get('dd', 0)}")
    lines.append(f"  Hidden words (lurkers):  {stage_counts.get('lurker', 0)}")
    lines.append(f"  Anagrams:                {stage_counts.get('anagram', 0)}")
    lines.append(f"  Compound wordplay:       {stage_counts.get('compound', 0)}")
    lines.append(f"  General wordplay:        {stage_counts.get('general', 0)}")
    lines.append(f"  Secondary helpers:       {stage_counts.get('secondary', 0)}")
    lines.append(f"  ---")
    pct = (100 * solved_count // total) if total > 0 else 0
    lines.append(
        f"  SOLVED:                  {solved_count}/{total} ({pct}%)")
    lines.append(f"  Partial (needs work):    {partial_count}")
    lines.append(f"  Unsolved:                {unsolved_count}")
    lines.append("")

    # ---- PER-CLUE DETAIL ----
    # Sort: across by number, then down by number
    sorted_results = sorted(results, key=sort_key)

    lines.append("=" * 80)
    lines.append("ACROSS")
    lines.append("=" * 80)

    current_direction = "across"
    for r in sorted_results:
        if (r.direction or "").lower() != current_direction:
            current_direction = (r.direction or "").lower()
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
    else:
        tag = "[PARTIAL]" if r.solve_quality == "partial" else "[UNSOLVED]"

    stage_label = {
        "dd": "double definition",
        "lurker": "hidden word",
        "anagram": "anagram",
        "compound": "compound wordplay",
        "general": "general wordplay",
        "secondary": "secondary helper",
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

    lines.append(f"  {'─' * 70}")

    return "\n".join(lines)


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description="Generate unified puzzle report")
    parser.add_argument("--run-id", type=int, default=None,
                        help="Pipeline run ID (default: latest)")
    parser.add_argument("--report-only", action="store_true",
                        help="Skip pipeline, just regenerate report from existing data")
    parser.add_argument("--output", type=str, default=OUTPUT_FILE, help="Output file")
    parser.add_argument("--source", type=str, default=SOURCE,
                        help="Clue source (telegraph, guardian, times, independent)")
    parser.add_argument("--puzzle-number", type=str, default=PUZZLE_NUMBER,
                        help="Puzzle number to solve")
    parser.add_argument("--max-clues", type=int, default=MAX_CLUES,
                        help="Maximum number of clues to process")
    parser.add_argument("--exclude-solved", action="store_true", default=EXCLUDE_SOLVED,
                        help="Exclude clues that already have a solution in clues_master.db")
    parser.add_argument("--wordplay-type", type=str, default=WORDPLAY_TYPE,
                        help="Wordplay type filter (all, anagram, lurker, dd)")
    parser.add_argument("--use-known-answer", action=argparse.BooleanOptionalAction,
                        default=USE_KNOWN_ANSWER,
                        help="Use known answer as candidate (default: True)")
    parser.add_argument("--single-clue", type=str, default=SINGLE_CLUE_MATCH,
                        help="Filter to single clue matching this text")
    parser.add_argument("--only-missing-definition", action=argparse.BooleanOptionalAction,
                        default=ONLY_MISSING_DEFINITION,
                        help="Show only clues where answer NOT in def candidates")
    parser.add_argument("--max-display", type=int, default=MAX_DISPLAY,
                        help="Max number of clues to print")
    parser.add_argument("--analyze-forwarded-anagrams", action=argparse.BooleanOptionalAction,
                        default=ANALYZE_FORWARDED_ANAGRAMS,
                        help="Enable forwarded anagram cohort analysis")
    parser.add_argument("--max-forwarded-samples", type=int, default=MAX_FORWARDED_SAMPLES,
                        help="Max forwarded samples to show")
    parser.add_argument("--analyze-successful-anagrams", action=argparse.BooleanOptionalAction,
                        default=ANALYZE_SUCCESSFUL_ANAGRAMS,
                        help="Enable successful anagram analysis")
    parser.add_argument("--max-successful-samples", type=int, default=MAX_SUCCESSFUL_SAMPLES,
                        help="Max successful samples to show")
    parser.add_argument("--enable-persistence", action=argparse.BooleanOptionalAction,
                        default=ENABLE_PERSISTENCE,
                        help="Enable stage persistence to SQLite")
    args = parser.parse_args()

    # Wire CLI args to pipeline_simulator globals
    import pipeline_simulator
    pipeline_simulator.SOURCE = args.source
    pipeline_simulator.PUZZLE_NUMBER = args.puzzle_number
    pipeline_simulator.MAX_CLUES = args.max_clues
    pipeline_simulator.WORDPLAY_TYPE = args.wordplay_type
    pipeline_simulator.USE_KNOWN_ANSWER = args.use_known_answer
    pipeline_simulator.SINGLE_CLUE_MATCH = args.single_clue
    pipeline_simulator.ONLY_MISSING_DEFINITION = args.only_missing_definition
    pipeline_simulator.MAX_DISPLAY = args.max_display
    pipeline_simulator.ANALYZE_FORWARDED_ANAGRAMS = args.analyze_forwarded_anagrams
    pipeline_simulator.MAX_FORWARDED_SAMPLES = args.max_forwarded_samples
    pipeline_simulator.ANALYZE_SUCCESSFUL_ANAGRAMS = args.analyze_successful_anagrams
    pipeline_simulator.MAX_SUCCESSFUL_SAMPLES = args.max_successful_samples
    pipeline_simulator.ENABLE_PERSISTENCE = args.enable_persistence
    pipeline_simulator.EXCLUDE_SOLVED = args.exclude_solved

    # Step 1: Run the pipeline (unless --report-only)
    if not args.report_only:
        print("=" * 60)
        print("COMPLETE PUZZLE SOLVER")
        print("=" * 60)

        print("\nSTEP 1: Running pipeline (DD, lurker, anagram, compound)...")
        print("-" * 60)
        import anagram_analysis
        anagram_analysis.main()

        print("\n" + "-" * 60)
        print("STEP 2: Running general wordplay analysis...")
        print("-" * 60)
        from stages.general import run_general_analysis
        run_general_analysis(run_id=0)

        print("\n" + "-" * 60)
        print("STEP 3: Running secondary analysis on failures...")
        print("-" * 60)
        from stages.secondary import run_secondary_analysis
        run_secondary_analysis(run_id=0)

        print("\n" + "-" * 60)
        print("STEP 3.5: Caching API synonym meshes...")
        print("-" * 60)
        import subprocess
        subprocess.run([
            "C:/Users/shute/PycharmProjects/cryptic_solver_V2/.venv/Scripts/python.exe",
            "cache_api_synonyms.py"
        ], cwd="C:/Users/shute/PycharmProjects/cryptic_solver_V2")

        print("\n" + "-" * 60)
        print("STEP 4: Writing unified puzzle report...")
        print("-" * 60)

    # Step 2: Generate report
    run_id = args.run_id if args.run_id is not None else get_latest_run_id(PIPELINE_DB)
    print(f"Using run_id: {run_id}")

    puzzle_info = get_puzzle_info(PIPELINE_DB, run_id)
    source = puzzle_info.get('source', '?')
    puzzle_number = puzzle_info.get('puzzle_number', '?')
    print(f"Puzzle: {source} #{puzzle_number}")

    # Get ALL clues for the puzzle (not just those processed in this run)
    clue_ids = get_all_puzzle_clue_ids(CLUES_DB, source, puzzle_number)
    print(f"Total clues: {len(clue_ids)}")

    results = []
    for cid in clue_ids:
        result = resolve_clue(PIPELINE_DB, CLUES_DB, run_id, cid)
        results.append(result)

    report = format_report(results, puzzle_info)

    with open(args.output, "w", encoding="utf-8") as f:
        f.write(report)
    print(f"\nReport saved to {args.output}")
    try:
        print("\n" + report)
    except UnicodeEncodeError:
        print("(Report contains Unicode characters - view in file)")

    solved_total = sum(1 for r in results if r.solved)
    pct = (100 * solved_total // len(results)) if results else 0
    print(f"\nFinal: {solved_total}/{len(results)} solved ({pct}%)")

    # Persist solved clues to clues_master.db
    if not args.report_only:
        print("\n" + "-" * 60)
        print("STEP 5: Persisting solved clues to clues_master.db...")
        print("-" * 60)
        from persistence import persist_solved_clues
        persist_solved_clues(results)


if __name__ == "__main__":
    main()