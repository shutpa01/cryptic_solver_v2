#!/usr/bin/env python3
"""
Secondary Solving Stage

Receives failures from stage_general and applies targeted helpers:
 0. Cryptic definition detection (narrow — no letters needed, all roles=definition)
 1. Strip linker words from unresolved lists
 2. Acrostic — first/last letter extraction when acrostic indicator found
 3. Reversal resolution
 4. Partial resolution — parts/extraction (heart, outskirts, head, tail of X)
 5. Deletion resolution
 6. Container/insertion resolution
 7. Homophone resolution
 8. Near-miss synonym lookup for remaining 1-2 letter gaps
 9. Anagram with deletion — fresh parse when existing evidence is poor
10. Wider cryptic definition — heuristic scoring for missed cryptic defs
11. Cross-reference — lookup other clues with same answer in clues_master.db
Final: DB suggestion generator — produces SQL INSERT suggestions for unsolved clues

Only processes clues where stage_general.fully_resolved = 0.
Cannot reduce solve count — only adds solves.
"""

import sqlite3
import json
import re
from pathlib import Path
from typing import List, Dict, Any, Optional, Set
from itertools import combinations

from stages.compound import DatabaseLookup, WordRole
from resources import norm_letters

# Database paths
PIPELINE_DB_PATH = Path(
    r'C:\Users\shute\PycharmProjects\cryptic_solver_V2\pipeline_stages.db')
CRYPTIC_DB_PATH = Path(
    r'C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\cryptic_new.db')

# Words that commonly appear in clues as grammar/linkers but don't contribute letters
LINKERS = {
    'of', 'in', 'the', 'a', 'an', 'is', 'for', 'with', 'by', 'on',
    'to', 'and', 'at', 'as', 'it', 'its', 'not', 'be', 'or', 'from',
    'has', 'up', 'into', 'one', 'that', 'this', 'some'
}


def compute_word_coverage(clue_text: str, unresolved: List[str]) -> float:
    """Fraction of content words in the clue that have been assigned roles (not unresolved)."""
    clue_words = re.findall(r'[a-zA-Z]+', clue_text.lower())
    if not clue_words:
        return 0.0
    unresolved_lower = {w.lower() for w in unresolved}
    resolved_count = sum(1 for w in clue_words if w not in unresolved_lower)
    return resolved_count / len(clue_words)


def get_pipeline_connection():
    """Get pipeline database connection."""
    return sqlite3.connect(PIPELINE_DB_PATH)


def get_cryptic_connection():
    """Get cryptic database connection."""
    return sqlite3.connect(CRYPTIC_DB_PATH)


# ======================================================================
# TABLE MANAGEMENT
# ======================================================================

def init_secondary_table():
    """Create stage_secondary table if it doesn't exist."""
    conn = get_pipeline_connection()
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS stage_secondary (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            run_id INTEGER NOT NULL,
            clue_id INTEGER,
            clue_text TEXT,
            answer TEXT,
            original_formula TEXT,
            improved_formula TEXT,
            helper_used TEXT,
            fully_resolved INTEGER,
            letters_still_needed TEXT,
            unresolved_words TEXT,
            breakdown TEXT,
            word_roles TEXT,
            db_suggestions TEXT
        )
    """)
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_secondary_run ON stage_secondary(run_id)")
    cursor.execute(
        "CREATE INDEX IF NOT EXISTS idx_secondary_clue ON stage_secondary(clue_id)")

    # Add db_suggestions column if missing (existing tables won't have it)
    try:
        cursor.execute("ALTER TABLE stage_secondary ADD COLUMN db_suggestions TEXT")
    except sqlite3.OperationalError:
        pass  # Column already exists

    conn.commit()
    conn.close()


def clear_stage_secondary(run_id: int):
    """Clear previous secondary stage data."""
    conn = get_pipeline_connection()
    cursor = conn.cursor()
    cursor.execute("DELETE FROM stage_secondary WHERE run_id = ?", (run_id,))
    deleted = cursor.rowcount
    conn.commit()
    conn.close()
    if deleted > 0:
        print(f"  Cleared {deleted} old records from stage_secondary")


# ======================================================================
# LOAD FAILURES FROM GENERAL STAGE
# ======================================================================

def load_general_failures(run_id: int = 0) -> List[Dict[str, Any]]:
    """Load clues that need secondary processing:
    - fully_resolved=0 (letters still missing), OR
    - fully_resolved=1 but unresolved_words is non-empty (indicator words
      still unattributed — a clue cannot be fully resolved with unattributed
      non-linker words).
    """
    conn = get_pipeline_connection()
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    cursor.execute("""
        SELECT clue_id, clue_text, answer, formula, definition_window,
               substitutions, operation_indicators, fully_resolved,
               letters_still_needed, unresolved_words, breakdown, word_roles
        FROM stage_general
        WHERE run_id = ?
          AND (fully_resolved = 0
               OR (unresolved_words IS NOT NULL AND unresolved_words != '[]'))
        ORDER BY clue_id
    """, (run_id,))

    results = []
    for row in cursor.fetchall():
        record = dict(row)
        # Parse JSON fields
        for field in ('substitutions', 'operation_indicators', 'unresolved_words',
                      'breakdown', 'word_roles'):
            try:
                record[field] = json.loads(record[field] or '[]')
            except (json.JSONDecodeError, TypeError):
                record[field] = []
        results.append(record)

    conn.close()
    return results


# ======================================================================
# HELPER: LINKER STRIPPING
# ======================================================================

def strip_linkers(unresolved_words: List[str]) -> tuple:
    """
    Remove linker words from unresolved list.
    Returns (cleaned_list, removed_linkers).
    """
    cleaned = []
    removed = []
    for w in unresolved_words:
        if norm_letters(w) in LINKERS:
            removed.append(w)
        else:
            cleaned.append(w)
    return cleaned, removed


# ======================================================================
# HELPER: ACROSTIC
# ======================================================================

# Module-level cache for acrostic indicator words
_acrostic_indicators: Optional[Set[str]] = None


def _load_acrostic_indicators(db: DatabaseLookup) -> Set[str]:
    """Load acrostic indicator words from the indicators DB. Cached after first call."""
    global _acrostic_indicators
    if _acrostic_indicators is not None:
        return _acrostic_indicators
    conn = get_cryptic_connection()
    try:
        cursor = conn.execute(
            "SELECT DISTINCT word FROM indicators WHERE wordplay_type = 'acrostic'")
        _acrostic_indicators = {row[0].lower() for row in cursor.fetchall()}
    finally:
        conn.close()
    return _acrostic_indicators


def attempt_acrostic(db: DatabaseLookup, answer: str, letters_needed: str,
                     clue_text: str, definition_window: str,
                     unresolved: List[str]) -> Optional[Dict]:
    """
    Check if first (or last) letters of clue words spell the answer.
    Scans clue text directly for acrostic indicators since stage_general
    does not record them in operation_indicators.

    Returns dict with 'word', 'letters', 'method' if successful, else None.
    """
    if not answer or not clue_text:
        return None

    acrostic_words = _load_acrostic_indicators(db)
    clue_tokens = re.findall(r"[a-zA-Z']+", clue_text)
    clue_lower = [t.lower() for t in clue_tokens]

    # Find acrostic indicator position in the clue
    indicator_idx = None
    indicator_word = None
    for i, token in enumerate(clue_lower):
        if token in acrostic_words:
            indicator_idx = i
            indicator_word = clue_tokens[i]
            break

    if indicator_idx is None:
        return None

    # Determine definition window words (to exclude from fodder)
    def_words = set()
    if definition_window:
        def_words = {w.lower() for w in re.findall(r"[a-zA-Z']+", definition_window)}

    # Determine fodder: words that are NOT the indicator, NOT in definition
    # NOTE: Do NOT strip linkers here — in acrostic clues every word's first
    # letter matters, including "of", "in", "up", "as", etc.
    fodder = []
    for i, token in enumerate(clue_tokens):
        if i == indicator_idx:
            continue
        if token.lower() in def_words:
            continue
        fodder.append(token)

    if not fodder:
        return None

    # Try first letters
    first_letters = ''.join(w[0] for w in fodder).upper()
    if first_letters == answer:
        method_parts = ' + '.join(f'{w[0].upper()} ({w})' for w in fodder)
        return {
            'word': indicator_word,
            'letters': first_letters,
            'method': f'acrostic: {method_parts} = {answer}'
        }

    # Try last letters
    last_letters = ''.join(w[-1] for w in fodder).upper()
    if last_letters == answer:
        method_parts = ' + '.join(f'{w[-1].upper()} ({w})' for w in fodder)
        return {
            'word': indicator_word,
            'letters': last_letters,
            'method': f'last-letter acrostic: {method_parts} = {answer}'
        }

    # Try first letters filling a gap (partial acrostic)
    if letters_needed and first_letters == letters_needed:
        method_parts = ' + '.join(f'{w[0].upper()} ({w})' for w in fodder)
        return {
            'word': indicator_word,
            'letters': first_letters,
            'method': f'acrostic (gap fill): {method_parts}'
        }

    return None


# ======================================================================
# HELPER: REVERSAL
# ======================================================================

def attempt_reversal(db: DatabaseLookup, answer: str, letters_needed: str,
                     unresolved: List[str], operation_indicators: List) -> Optional[Dict]:
    """
    Try reversing unresolved words or their synonyms to match letters_needed.

    Returns dict with 'word', 'letters', 'method' if successful, else None.
    """
    if not letters_needed or not unresolved:
        return None

    # Check if any operation indicator is a reversal type
    has_reversal = False
    for ind in operation_indicators:
        if isinstance(ind, (list, tuple)) and len(ind) >= 2:
            if 'reversal' in str(ind[1]).lower():
                has_reversal = True
                break
        elif isinstance(ind, str) and 'reversal' in ind.lower():
            has_reversal = True
            break

    if not has_reversal:
        return None

    needed_upper = letters_needed.upper()

    for word in unresolved:
        word_clean = norm_letters(word).upper()

        # Try reversing the word itself
        if word_clean[::-1] == needed_upper:
            return {
                'word': word,
                'letters': word_clean[::-1],
                'method': f'reversal({word}) = {word_clean[::-1]}'
            }

        # Try reversing synonyms/substitutions
        subs = db.lookup_substitution(word, max_synonym_length=len(needed_upper) + 2)
        for sub in subs:
            sub_letters = norm_letters(sub.letters).upper()
            if sub_letters[::-1] == needed_upper:
                return {
                    'word': word,
                    'letters': sub_letters[::-1],
                    'method': f'{word} = {sub.letters}, reversal({sub.letters}) = {sub_letters[::-1]}'
                }

    return None


# ======================================================================
# HELPER: DELETION
# ======================================================================

def attempt_deletion(db: DatabaseLookup, answer: str, letters_needed: str,
                     unresolved: List[str], operation_indicators: List) -> Optional[Dict]:
    """
    Try applying deletion operations to unresolved words/synonyms.
    Also tries letter extraction (the dual of deletion): "heart of X" = middle letter,
    "outskirts of X" = first+last letters, etc.

    Returns dict with 'word', 'letters', 'method' if successful, else None.
    """
    if not letters_needed or not unresolved:
        return None

    # Check for deletion indicators and find subtype
    # Data uses: 'head', 'tail', 'middle', 'ends', None
    # Also collect indicator words so we don't use them as deletion targets
    deletion_subtype = None
    indicator_words = set()
    for ind in operation_indicators:
        if isinstance(ind, (list, tuple)) and len(ind) >= 2:
            ind_word = str(ind[0]).lower()
            ind_type = str(ind[1]).lower()
            indicator_words.add(ind_word)
            if 'deletion' in ind_type:
                sub = ind[2] if len(ind) >= 3 and ind[2] else None
                deletion_subtype = sub.lower() if sub else 'any'

    if deletion_subtype is None:
        return None

    needed_upper = letters_needed.upper()

    for word in unresolved:
        # Skip words that are themselves indicators — they describe the operation,
        # not the target. e.g., "ignoring" is the deletion indicator, not a word
        # to look up synonyms for and apply deletion to.
        if norm_letters(word).lower() in indicator_words:
            continue
        word_letters = norm_letters(word).upper()

        # === EXTRACTION MODE ===
        # "heart of X" = middle letter(s), "outskirts of X" = first+last, etc.
        # This handles short needed (1-2 letters) extracted from the word itself
        if len(needed_upper) <= 3:
            if deletion_subtype in ('middle', 'any') and len(word_letters) >= 3:
                mid = len(word_letters) // 2
                # Single middle letter
                if word_letters[mid] == needed_upper[0] and len(needed_upper) == 1:
                    return {
                        'word': word,
                        'letters': needed_upper,
                        'method': f'heart({word}) = {word_letters[mid]}'
                    }
                # Two middle letters (even-length word)
                if len(needed_upper) == 2 and len(word_letters) >= 4:
                    mid2 = mid - 1
                    if word_letters[mid2:mid2 + 2] == needed_upper:
                        return {
                            'word': word,
                            'letters': needed_upper,
                            'method': f'heart({word}) = {needed_upper}'
                        }

            if deletion_subtype in ('ends', 'any') and len(word_letters) >= 2:
                outer = word_letters[0] + word_letters[-1]
                if outer == needed_upper:
                    return {
                        'word': word,
                        'letters': needed_upper,
                        'method': f'ends({word}) = {outer}'
                    }

            if deletion_subtype in ('head', 'any') and word_letters:
                if word_letters[0] == needed_upper[0] and len(needed_upper) == 1:
                    return {
                        'word': word,
                        'letters': needed_upper,
                        'method': f'head({word}) = {word_letters[0]}'
                    }

            if deletion_subtype in ('tail', 'any') and word_letters:
                if word_letters[-1] == needed_upper[0] and len(needed_upper) == 1:
                    return {
                        'word': word,
                        'letters': needed_upper,
                        'method': f'tail({word}) = {word_letters[-1]}'
                    }

        # === DELETION MODE ===
        # Apply deletion to word or its synonyms, check if result = needed
        subs = db.lookup_substitution(word, max_synonym_length=len(needed_upper) + 3)
        candidates = [(word, word_letters)]
        for sub in subs:
            candidates.append((f"{word}={sub.letters}", norm_letters(sub.letters).upper()))

        for label, letters in candidates:
            if len(letters) < 2:
                continue

            result = None
            # head = remove first letter (beheadment)
            if deletion_subtype in ('head', 'any'):
                trimmed = letters[1:]
                if trimmed == needed_upper:
                    result = f'behead({label}) = {trimmed}'

            # tail = remove last letter (curtailment)
            if not result and deletion_subtype in ('tail', 'any'):
                trimmed = letters[:-1]
                if trimmed == needed_upper:
                    result = f'curtail({label}) = {trimmed}'

            # ends = remove first and last letters
            if not result and deletion_subtype in ('ends', 'any'):
                if len(letters) >= 3:
                    trimmed = letters[1:-1]
                    if trimmed == needed_upper:
                        result = f'shell({label}) = {trimmed}'

            # middle = remove middle letter(s)
            if not result and deletion_subtype in ('middle', 'any'):
                for i in range(1, len(letters) - 1):
                    trimmed = letters[:i] + letters[i + 1:]
                    if trimmed == needed_upper:
                        result = f'gut({label}, pos {i}) = {trimmed}'
                        break

            if result:
                return {
                    'word': word,
                    'letters': needed_upper,
                    'method': result
                }

    return None


# ======================================================================
# HELPER: CONTAINER / INSERTION
# ======================================================================

def attempt_container(db: DatabaseLookup, answer: str, letters_needed: str,
                      unresolved: List[str], operation_indicators: List,
                      found_letters: str) -> Optional[Dict]:
    """
    Try container/insertion: wrapping or inserting letters.

    Returns dict with 'word', 'letters', 'method' if successful, else None.
    """
    if not letters_needed or not unresolved:
        return None

    has_container = False
    for ind in operation_indicators:
        if isinstance(ind, (list, tuple)) and len(ind) >= 2:
            ind_type = str(ind[1]).lower()
            if ind_type in ('container', 'insertion'):
                has_container = True
                break

    if not has_container:
        return None

    answer_upper = answer.upper()
    needed_upper = letters_needed.upper()

    for word in unresolved:
        subs = db.lookup_substitution(word, max_synonym_length=len(answer_upper))
        candidates = [(word, norm_letters(word).upper())]
        for sub in subs:
            candidates.append((f"{word}={sub.letters}", norm_letters(sub.letters).upper()))

        for label, word_letters in candidates:
            if not word_letters:
                continue

            # Strategy 1: word_letters contains found_letters
            # e.g., found_letters="T", word_letters="RA" → check if T inside RA = RAT? No...
            # Actually: check if inserting found_letters into word_letters = answer
            if found_letters:
                found_upper = found_letters.upper()
                # Try inserting found_letters at each position in word_letters
                for i in range(len(word_letters) + 1):
                    combined = word_letters[:i] + found_upper + word_letters[i:]
                    if combined == answer_upper:
                        return {
                            'word': word,
                            'letters': word_letters,
                            'method': f'{label} around {found_upper} = {combined}'
                        }

                # Try inserting word_letters at each position in found_letters
                for i in range(len(found_upper) + 1):
                    combined = found_upper[:i] + word_letters + found_upper[i:]
                    if combined == answer_upper:
                        return {
                            'word': word,
                            'letters': word_letters,
                            'method': f'{word_letters} inside {found_upper} = {combined}'
                        }

            # Strategy 2: word_letters wraps around needed_letters
            for i in range(1, len(word_letters)):
                outer_left = word_letters[:i]
                outer_right = word_letters[i:]
                combined = outer_left + needed_upper + outer_right
                if combined == answer_upper:
                    return {
                        'word': word,
                        'letters': word_letters + needed_upper,
                        'method': f'{label} around {needed_upper} = {combined}'
                    }

    return None


# ======================================================================
# HELPER: MULTI-WORD CONTAINER WITH PARTS EXTRACTION
# ======================================================================

def attempt_multiword_container(db: DatabaseLookup, answer: str, letters_needed: str,
                                unresolved: List[str], operation_indicators: List,
                                found_letters: str, word_roles: List[Dict]) -> Optional[Dict]:
    """
    Handle complex container scenarios:
    - Parts indicator + fodder → outer wrapper (e.g., "seeming discontented" → SG)
    - Container indicator
    - Multiple words → inner content (e.g., "dog" + "home" → CUR+IN)
    - Combine: outer wraps inner

    Returns dict with 'letters', 'method' if successful, else None.
    """
    if not letters_needed or len(unresolved) < 3:
        return None

    answer_upper = answer.upper()
    needed_upper = letters_needed.upper()

    # Find parts indicators and container indicators
    parts_indicators = []
    container_indicators = []

    for ind in operation_indicators:
        if isinstance(ind, (list, tuple)) and len(ind) >= 2:
            word = str(ind[0]).lower()
            ind_type = str(ind[1]).lower()
            subtype = str(ind[2]).lower() if len(ind) >= 3 else None

            if ind_type == 'parts':
                parts_indicators.append((word, subtype))
            elif ind_type in ('container', 'insertion'):
                container_indicators.append((word, subtype))

    if not parts_indicators or not container_indicators:
        return None

    # Try to find parts extraction pairs (indicator + adjacent fodder)
    for i, word in enumerate(unresolved):
        word_clean = norm_letters(word).lower()

        # Check if this word is a parts indicator
        parts_match = None
        for ind_word, subtype in parts_indicators:
            if ind_word in word_clean or word_clean in ind_word:
                parts_match = subtype
                break

        if not parts_match:
            continue

        # Look for adjacent fodder word (before or after)
        fodder_words = []
        if i > 0:
            fodder_words.append((i-1, unresolved[i-1]))
        if i < len(unresolved) - 1:
            fodder_words.append((i+1, unresolved[i+1]))

        for fodder_idx, fodder in fodder_words:
            fodder_clean = norm_letters(fodder).upper()

            # Skip if fodder is also an indicator
            is_indicator = any(
                ind_word in norm_letters(fodder).lower()
                for ind_word, _ in parts_indicators + container_indicators
            )
            if is_indicator:
                continue

            # Apply parts extraction
            if parts_match == 'outer_use' and len(fodder_clean) >= 2:
                outer_letters = fodder_clean[0] + fodder_clean[-1]
            elif parts_match == 'first_use' and len(fodder_clean) >= 1:
                outer_letters = fodder_clean[0]
            elif parts_match == 'last_use' and len(fodder_clean) >= 1:
                outer_letters = fodder_clean[-1]
            elif parts_match == 'central_use' and len(fodder_clean) >= 3:
                outer_letters = fodder_clean[1:-1]
            else:
                continue

            if len(outer_letters) < 2:
                continue

            # Now look for other unresolved words to form inner content
            # Exclude the parts indicator and its fodder
            other_words = [
                unresolved[j] for j in range(len(unresolved))
                if j not in (i, fodder_idx)
            ]

            # Try combinations of other words as substitutions
            for num_words in range(1, min(4, len(other_words) + 1)):
                for word_combo in combinations(other_words, num_words):
                    # Look up substitutions for each word
                    inner_parts = []

                    for w in word_combo:
                        # Skip if this is a container indicator
                        if any(ind_word in norm_letters(w).lower()
                               for ind_word, _ in container_indicators):
                            continue

                        subs = db.lookup_substitution(w, max_synonym_length=12)
                        if subs:
                            # Use the first/shortest substitution
                            inner_parts.append((w, subs[0].letters.upper()))
                        else:
                            # Try the word itself
                            inner_parts.append((w, norm_letters(w).upper()))

                    if not inner_parts:
                        continue

                    # Combine inner parts
                    inner_content = ''.join(letters for _, letters in inner_parts)

                    # Try wrapping: split outer_letters and insert inner
                    if len(outer_letters) == 2:
                        # Simple case: first and last letters wrap
                        wrapped = outer_letters[0] + inner_content + outer_letters[1]
                    else:
                        # Try all split positions
                        wrapped = None
                        for split in range(1, len(outer_letters)):
                            candidate = outer_letters[:split] + inner_content + outer_letters[split:]
                            # Check if this produces the needed letters
                            if candidate == needed_upper:
                                wrapped = candidate
                                break
                            # Or check if prepending found_letters gives answer
                            if found_letters:
                                full = found_letters.upper() + candidate
                                if full == answer_upper:
                                    wrapped = candidate
                                    break

                        if not wrapped:
                            continue

                        # Use the found wrapped value
                        # (reassigning for consistency)
                        wrapped = candidate

                    # Check if result matches needed letters
                    success = False
                    if wrapped == needed_upper:
                        success = True
                    elif found_letters:
                        full = found_letters.upper() + wrapped
                        if full == answer_upper:
                            success = True

                    if success:
                        inner_desc = '+'.join(
                            f"{w}={letters}" if letters != norm_letters(w).upper() else w
                            for w, letters in inner_parts
                        )

                        return {
                            'letters': wrapped,
                            'method': f'{fodder}({parts_match}) = {outer_letters} around ({inner_desc}) = {wrapped}',
                            'fodder': fodder,
                            'parts_indicator': word,
                            'inner_words': [w for w, _ in inner_parts]
                        }

    return None


# ======================================================================
# HELPER: HOMOPHONE
# ======================================================================

def attempt_homophone(db: DatabaseLookup, answer: str, letters_needed: str,
                      unresolved: List[str],
                      operation_indicators: List) -> Optional[Dict]:
    """
    Try homophone lookup for unresolved words.

    Returns dict with 'word', 'letters', 'method' if successful, else None.
    """
    if not letters_needed or not unresolved:
        return None

    has_homophone = False
    homophone_indicator_words = set()
    for ind in operation_indicators:
        if isinstance(ind, (list, tuple)) and len(ind) >= 2:
            if 'homophone' in str(ind[1]).lower():
                has_homophone = True
                break

    # Also check if any unresolved word is a known homophone indicator
    if not has_homophone:
        for word in unresolved:
            indicator_match = db.lookup_indicator(word)
            if indicator_match and indicator_match.wordplay_type == 'homophone':
                has_homophone = True
                homophone_indicator_words.add(word.lower())
                break

    if not has_homophone:
        return None

    needed_upper = letters_needed.upper()

    for word in unresolved:
        word_clean = norm_letters(word)

        # Direct homophone lookup
        homophones = db.lookup_homophone(word_clean)
        for hom in homophones:
            hom_upper = norm_letters(hom).upper()
            if hom_upper == needed_upper:
                return {
                    'word': word,
                    'letters': hom_upper,
                    'method': f'homophone({word}) = {hom_upper}'
                }

        # Synonym → homophone chain
        subs = db.lookup_substitution(word, max_synonym_length=len(needed_upper) + 4)
        for sub in subs:
            syn_homophones = db.lookup_homophone(sub.letters.lower())
            for hom in syn_homophones:
                hom_upper = norm_letters(hom).upper()
                if hom_upper == needed_upper:
                    return {
                        'word': word,
                        'letters': hom_upper,
                        'method': f'{word} = {sub.letters}, homophone({sub.letters}) = {hom_upper}'
                    }

    return None


# ======================================================================
# HELPER: PARTIAL RESOLUTION (parts/extraction)
# ======================================================================

# Module-level cache for parts indicator words
_parts_indicators: Optional[Dict[str, str]] = None


def _load_parts_indicators(db: DatabaseLookup) -> Dict[str, str]:
    """Load parts indicator words from the DB. Returns {word: subtype}. Cached."""
    global _parts_indicators
    if _parts_indicators is not None:
        return _parts_indicators
    conn = get_cryptic_connection()
    try:
        cursor = conn.execute(
            "SELECT word, subtype FROM indicators WHERE wordplay_type = 'parts'")
        _parts_indicators = {}
        for row in cursor.fetchall():
            word = row[0].lower()
            subtype = (row[1] or 'first_use').lower()
            _parts_indicators[word] = subtype
    finally:
        conn.close()
    return _parts_indicators


def _find_extraction_target(indicator_idx: int, clue_tokens: List[str],
                            def_words: Set[str]) -> Optional[str]:
    """
    Find the target word for a parts extraction indicator.
    Looks for 'X of Y' pattern first, then adjacent words.
    Returns the target word or None.
    """
    lower_tokens = [t.lower() for t in clue_tokens]

    # Pattern: "indicator of TARGET"  (e.g., "heart of bathroom")
    if (indicator_idx + 2 < len(clue_tokens)
            and lower_tokens[indicator_idx + 1] == 'of'
            and lower_tokens[indicator_idx + 2] not in def_words):
        return clue_tokens[indicator_idx + 2]

    # Pattern: "TARGET's indicator" or adjacent before (e.g., "bathroom's heart")
    if (indicator_idx - 1 >= 0
            and lower_tokens[indicator_idx - 1] not in def_words
            and norm_letters(lower_tokens[indicator_idx - 1]) not in LINKERS):
        return clue_tokens[indicator_idx - 1]

    # Adjacent after (skip "of" if not present)
    if (indicator_idx + 1 < len(clue_tokens)
            and lower_tokens[indicator_idx + 1] not in def_words
            and norm_letters(lower_tokens[indicator_idx + 1]) not in LINKERS):
        return clue_tokens[indicator_idx + 1]

    return None


def _extract_letters(word: str, subtype: str) -> Optional[str]:
    """Extract letters from a word based on parts subtype."""
    letters = norm_letters(word).upper()
    if not letters:
        return None

    if subtype in ('first_use', 'initial'):
        return letters[0]
    elif subtype in ('last_use', 'final'):
        return letters[-1]
    elif subtype in ('central_use', 'center', 'central'):
        if len(letters) < 3:
            return None
        mid = len(letters) // 2
        return letters[mid]
    elif subtype in ('outer_use', 'outside', 'outer'):
        if len(letters) < 2:
            return None
        return letters[0] + letters[-1]
    elif subtype == 'alternate':
        # Extract alternate letters (every other letter)
        # Try both patterns: odd positions (0,2,4,...) and even positions (1,3,5,...)
        if len(letters) < 2:
            return None
        odd_positions = ''.join(letters[i] for i in range(0, len(letters), 2))  # 0,2,4,...
        even_positions = ''.join(letters[i] for i in range(1, len(letters), 2))  # 1,3,5,...
        # Return both patterns concatenated so caller can use whichever matches
        # But typically we want just one pattern - use odd positions as primary
        return odd_positions if odd_positions else even_positions
    return None


def attempt_partial_resolve(db: DatabaseLookup, answer: str, letters_needed: str,
                            clue_text: str, definition_window: str,
                            unresolved: List[str],
                            operation_indicators: List) -> Optional[Dict]:
    """
    Scan for parts-type indicators in the clue and execute letter extractions.
    Handles: first/last/middle/outer letter extraction from target words.

    Returns dict with 'word', 'letters', 'method' if extraction fills the gap.
    """
    if not letters_needed or not clue_text:
        return None

    parts_words = _load_parts_indicators(db)
    clue_tokens = re.findall(r"[a-zA-Z']+", clue_text)
    clue_lower = [t.lower() for t in clue_tokens]

    # Determine definition window words
    def_words = set()
    if definition_window:
        def_words = {w.lower() for w in re.findall(r"[a-zA-Z']+", definition_window)}

    # Also check operation_indicators for parts indicators
    parts_from_ops = {}
    for ind in operation_indicators:
        if isinstance(ind, (list, tuple)) and len(ind) >= 2:
            ind_type = str(ind[1]).lower()
            if ind_type == 'parts':
                ind_word = str(ind[0]).lower()
                ind_sub = ind[2].lower() if len(ind) >= 3 and ind[2] else 'first_use'
                parts_from_ops[ind_word] = ind_sub

    # Find all parts indicators in the clue text
    extractions = []
    for i, token in enumerate(clue_lower):
        subtype = parts_words.get(token) or parts_from_ops.get(token)
        if not subtype:
            continue
        if token in def_words:
            continue

        target = _find_extraction_target(i, clue_tokens, def_words)
        if not target:
            continue

        extracted = _extract_letters(target, subtype)
        if extracted:
            extractions.append({
                'indicator': clue_tokens[i],
                'target': target,
                'subtype': subtype,
                'letters': extracted
            })

    if not extractions:
        return None

    # Combine all extracted letters
    all_extracted = ''.join(e['letters'] for e in extractions)
    needed_upper = letters_needed.upper()

    # Check if extractions fill the gap exactly
    remaining = _subtract_letters(needed_upper, all_extracted)
    if not remaining:
        method_parts = ' + '.join(
            f"{e['letters']} ({e['subtype']}({e['target']}))" for e in extractions)
        return {
            'word': extractions[0]['indicator'],
            'letters': all_extracted,
            'method': f'parts extraction: {method_parts}'
        }

    # Check if extractions fill part of the gap and a single unresolved word
    # could provide the rest via DB lookup
    if len(remaining) <= 3:
        for word in unresolved:
            if word.lower() in def_words:
                continue
            if word.lower() in {e['indicator'].lower() for e in extractions}:
                continue
            if word.lower() in {e['target'].lower() for e in extractions}:
                continue
            subs = db.lookup_substitution(word, max_synonym_length=len(remaining))
            for sub in subs:
                sub_letters = norm_letters(sub.letters).upper()
                if _subtract_letters(remaining, sub_letters) == '':
                    combined = all_extracted + sub_letters
                    method_parts = ' + '.join(
                        f"{e['letters']} ({e['subtype']}({e['target']}))"
                        for e in extractions)
                    method_parts += f" + {sub.letters} ({word}={sub.category})"
                    return {
                        'word': extractions[0]['indicator'],
                        'letters': combined,
                        'method': f'parts extraction: {method_parts}'
                    }

    return None


# ======================================================================
# HELPER: SUBSTITUTION SWAP
# ======================================================================

def attempt_substitution_swap(db: DatabaseLookup, answer: str,
                              letters_needed: str,
                              word_roles: List[Dict]) -> Optional[Dict]:
    """
    When letters are still needed, try swapping already-resolved substitutions
    for alternative ones that better fit the remaining gap.

    For each resolved word with a known letter contribution (OLD), compute
    the target: target = letters_still_needed + OLD (as a multiset). If any
    alternative substitution for that word exactly matches target, swap it in
    — the gap is now filled.

    Example: "point" resolved as N, but letters_needed = IET.
    target = IET + N = IENT. Substitution TINE sorts to EINT = IENT. Match!
    → "point" = TINE fills the gap, clue fully resolved.
    """
    if not letters_needed or not word_roles:
        return None

    needed_sorted = sorted(letters_needed.upper())

    # Roles that carry letter contributions we can try swapping
    # Skip: definition, linker, indicator, fodder (anagram fodder is scrambled)
    SKIP_ROLES = {'definition', 'linker', 'link', 'fodder', 'anagram'}

    for wr in word_roles:
        if not isinstance(wr, dict):
            continue
        role = (wr.get('role') or '').lower()
        if role in SKIP_ROLES or 'indicator' in role or 'definition' in role:
            continue

        word = wr.get('word', '')
        old_contrib = norm_letters(wr.get('contributes') or '').upper()

        if not word or not old_contrib:
            continue

        # Target letters the new substitution must provide exactly
        target = sorted(letters_needed.upper() + old_contrib)
        target_len = len(target)

        # Look up all substitutions for this word
        subs = db.lookup_substitution(word, max_synonym_length=target_len + 1)

        for sub in subs:
            sub_letters = norm_letters(sub.letters).upper()
            # Must be exact multiset match and different from current
            if sub_letters != old_contrib and sorted(sub_letters) == target:
                return {
                    'word': word,
                    'letters': letters_needed,
                    'new_contrib': sub_letters,
                    'old_contrib': old_contrib,
                    'method': (f'{word} = {sub_letters} (replaces {old_contrib}), '
                               f'fills gap: {letters_needed}')
                }

    return None


# ======================================================================
# HELPER: NEAR-MISS (1-2 letter gaps)
# ======================================================================

def attempt_near_miss(db: DatabaseLookup, answer: str, letters_needed: str,
                      unresolved: List[str]) -> Optional[Dict]:
    """
    For clues missing only 1-2 letters, try short synonyms and abbreviations.

    Returns dict with 'word', 'letters', 'method' if successful, else None.
    """
    if not letters_needed or len(letters_needed) > 2 or not unresolved:
        return None

    needed_upper = letters_needed.upper()

    for word in unresolved:
        # Check wordplay table for abbreviations (e.g., "note" → "DO", "RE")
        subs = db.lookup_substitution(word, max_synonym_length=2)
        for sub in subs:
            sub_letters = norm_letters(sub.letters).upper()
            if sub_letters == needed_upper:
                return {
                    'word': word,
                    'letters': sub_letters,
                    'method': f'{word} = {sub.letters} ({sub.category})'
                }

        # Check single-letter abbreviations specifically
        subs_1 = db.lookup_substitution(word, max_synonym_length=1)
        for sub in subs_1:
            sub_letters = norm_letters(sub.letters).upper()
            if sub_letters and sub_letters[0] == needed_upper[0]:
                if len(needed_upper) == 1:
                    return {
                        'word': word,
                        'letters': sub_letters,
                        'method': f'{word} = {sub.letters} ({sub.category})'
                    }

    return None


# ======================================================================
# HELPER: DEFINITION SWAP
# ======================================================================

def attempt_definition_swap(db: DatabaseLookup, answer: str, letters_needed: str,
                            clue_text: str, word_roles: List[Dict],
                            unresolved: List[str]) -> Optional[Dict]:
    """
    Check if the assigned definition word has a substitution that appears as a
    contiguous chunk in the answer. If so, the definition assignment is likely
    wrong — the word is actually a substitution.

    Example: "Company keeps packaging minute luxuries" → COMFORTS
             "Company" tagged as definition, but company → CO and CO starts COMFORTS.
             Swap: Company becomes substitution (CO), freeing those letters.

    Returns dict with 'letters' (newly freed), 'method', 'swapped_word' on success.
    Does NOT attempt to find the alternative definition — downstream handlers
    will work with the updated letters_needed.
    """
    if not letters_needed or not answer:
        return None

    answer_upper = answer.upper().replace(' ', '')

    # Find definition-tagged words
    def_words = []
    for wr in word_roles:
        if isinstance(wr, dict) and wr.get('role') == 'definition':
            def_words.append(wr.get('word', ''))

    if not def_words:
        return None

    for def_word in def_words:
        subs = lookup_with_plural(db, def_word, max_synonym_length=len(answer_upper))
        for sub in subs:
            sub_letters = sub['letters']
            if not sub_letters or len(sub_letters) > len(letters_needed):
                continue

            # Check if substitution appears in the answer as a contiguous chunk
            pos = answer_upper.find(sub_letters)
            if pos < 0:
                continue

            # Check the substitution matches letters we still need
            # (not letters already accounted for)
            temp_needed = list(letters_needed)
            all_in_needed = True
            for c in sub_letters:
                if c in temp_needed:
                    temp_needed.remove(c)
                else:
                    all_in_needed = False
                    break

            if not all_in_needed:
                continue

            # Strong evidence: definition word's substitution is in the answer
            # and covers needed letters
            freed = sub_letters
            method = f"{def_word} = {sub_letters} ({sub['category']})"
            return {
                'letters': freed,
                'method': method,
                'swapped_word': def_word
            }

    return None


# ======================================================================
# HELPER: PLURAL-AWARE SUBSTITUTION LOOKUP
# ======================================================================

def lookup_with_plural(db: DatabaseLookup, word: str,
                       max_synonym_length: int = 8) -> list:
    """
    Look up substitutions for a word. If the word ends in 's', also try the
    singular form and return each result with 'S' appended.

    Returns list of SubstitutionMatch-like dicts: {letters, category, notes, plural}.
    """
    results = []

    # Direct lookup
    for sub in db.lookup_substitution(word, max_synonym_length=max_synonym_length):
        results.append({
            'letters': norm_letters(sub.letters).upper(),
            'category': sub.category,
            'notes': sub.notes or '',
            'plural': False
        })

    # Plural fallback: if word ends in 's', try singular and pluralize results
    word_clean = norm_letters(word).lower()
    if word_clean.endswith('s') and len(word_clean) >= 4:
        singular = word_clean[:-1]
        for sub in db.lookup_substitution(singular, max_synonym_length=max_synonym_length - 1):
            pluralized = norm_letters(sub.letters).upper() + 'S'
            # Skip if already covered by direct lookup
            if not any(r['letters'] == pluralized for r in results):
                results.append({
                    'letters': pluralized,
                    'category': sub.category,
                    'notes': f"plural of {singular} -> {sub.letters}",
                    'plural': True
                })

    return results


# ======================================================================
# HELPER: SYNONYM OVERRIDE
# ======================================================================

def attempt_synonym_override(db: DatabaseLookup, answer: str, letters_needed: str,
                             word_roles: List[Dict], unresolved: List[str]
                             ) -> Optional[Dict]:
    """
    When stage_general tagged a word as an indicator, but that word has a
    synonym/substitution matching a contiguous chunk at the current start of
    letters_needed, override it as a synonym and consume those letters.

    Also checks unresolved words the same way.

    letters_needed preserves answer-order, so matching from the left is valid.

    Returns dict with 'letters', 'method', 'overrides' if any letters consumed,
    else None.
    """
    if not letters_needed:
        return None

    # Gather candidate words: indicator-tagged words + unresolved words
    indicator_roles = {'container_indicator', 'insertion_indicator',
                       'hidden_indicator', 'anagram_indicator',
                       'reversal_indicator', 'deletion_indicator',
                       'homophone_indicator'}

    candidates = []  # (word, source_label)

    for wr in word_roles:
        if isinstance(wr, dict) and wr.get('role', '') in indicator_roles:
            candidates.append((wr.get('word', ''), 'indicator'))

    for w in unresolved:
        wl = w.lower()
        # Skip linkers — they're unlikely to be synonyms
        if wl in LINKERS:
            continue
        # Avoid duplicates if already in candidates
        if not any(c[0].lower() == wl for c in candidates):
            candidates.append((w, 'unresolved'))

    if not candidates:
        return None

    remaining = letters_needed.upper()
    overrides = []
    max_syn_len = len(remaining)  # Synonyms can be as long as what's still needed

    # Greedy left-to-right: try each candidate against the start of remaining
    changed = True
    while changed and remaining:
        changed = False
        for word, source in candidates:
            if not remaining:
                break
            # Already used this word
            if any(o['word'].lower() == word.lower() for o in overrides):
                continue

            subs = lookup_with_plural(db, word, max_synonym_length=max_syn_len)
            for sub in subs:
                sub_letters = sub['letters']
                if not sub_letters:
                    continue
                if remaining.startswith(sub_letters):
                    overrides.append({
                        'word': word,
                        'letters': sub_letters,
                        'category': sub['category'],
                        'source': source
                    })
                    remaining = remaining[len(sub_letters):]
                    changed = True
                    break

    if not overrides:
        return None

    consumed = ''.join(o['letters'] for o in overrides)
    parts = [f"{o['word']} = {o['letters']} ({o['category']})" for o in overrides]
    method = ' + '.join(parts)

    return {
        'letters': consumed,
        'method': method,
        'overrides': overrides
    }


# ======================================================================
# HELPER: SYNONYM SUBSTRING
# ======================================================================

def attempt_synonym_substring(db: DatabaseLookup, answer: str, letters_needed: str,
                              found_letters: str, unresolved: List[str],
                              word_roles: List[Dict]) -> Optional[Dict]:
    """
    Check if an unresolved word (or indicator-tagged word) has a synonym that
    appears as a contiguous substring of the answer, with the outside letters
    covered by what the pipeline already found.

    Example: DELIBERATION, found DE, unresolved "release" → LIBERATION
             LIBERATION is substring at pos 2, outside = "DE", already found. Solved.
    """
    if not letters_needed or not answer:
        return None

    answer_upper = answer.upper().replace(' ', '')
    found_upper = norm_letters(found_letters).upper()

    # Gather candidate words: unresolved + indicator-tagged
    indicator_roles = {'container_indicator', 'insertion_indicator',
                       'hidden_indicator', 'anagram_indicator',
                       'reversal_indicator', 'deletion_indicator',
                       'homophone_indicator'}

    candidates = []
    for w in unresolved:
        if norm_letters(w).lower() not in LINKERS:
            candidates.append(w)
    for wr in word_roles:
        if isinstance(wr, dict) and wr.get('role', '') in indicator_roles:
            word = wr.get('word', '')
            if word and not any(c.lower() == word.lower() for c in candidates):
                candidates.append(word)

    if not candidates:
        return None

    # Minimum synonym length: must cover at least half the answer
    min_syn_len = max(3, len(answer_upper) // 2)

    for word in candidates:
        subs = lookup_with_plural(db, word, max_synonym_length=len(answer_upper))
        for sub in subs:
            syn = sub['letters']
            if len(syn) < min_syn_len:
                continue

            # Check if synonym is a contiguous substring of the answer
            pos = answer_upper.find(syn)
            if pos < 0:
                continue

            # Letters outside the synonym substring
            outside = answer_upper[:pos] + answer_upper[pos + len(syn):]

            # Check: are the outside letters a subset of found_letters?
            outside_remaining = list(outside)
            found_remaining = list(found_upper)
            all_covered = True
            for c in outside_remaining:
                if c in found_remaining:
                    found_remaining.remove(c)
                else:
                    all_covered = False
                    break

            if all_covered:
                method = f"{word} = {syn} ({sub['category']})"
                return {
                    'letters': letters_needed,  # Accounts for all remaining
                    'method': method,
                    'word': word,
                    'synonym': syn
                }

    return None


# ======================================================================
# HELPER: WIDER CRYPTIC DEFINITION
# ======================================================================

# Words that suggest the clue is a cryptic definition
_CRYPTIC_DEF_SIGNALS = {'perhaps', 'maybe', 'possibly', 'say', 'conceivably',
                         'potentially', 'supposedly', 'apparently'}


def attempt_wider_cryptic_def(answer: str, clue_text: str, formula: str,
                               quality: str, letters_needed: str,
                               operation_indicators: List,
                               word_roles: List) -> Optional[Dict]:
    """
    Detect cryptic definitions that the narrow Step 0 handler missed.
    Uses heuristic scoring: question mark, short clue, signal words, no indicators.

    Returns dict with 'word', 'letters', 'method' if confident, else None.
    """
    if not answer or not clue_text:
        return None

    # Only fire when pipeline found no useful evidence
    has_real_formula = formula and '?' not in formula and formula.strip()
    if has_real_formula:
        return None

    clue_words = re.findall(r'[a-zA-Z]+', clue_text.lower())
    score = 0

    # Question mark at end of clue
    if clue_text.rstrip().endswith('?'):
        score += 2

    # Short clue (few words)
    if len(clue_words) <= 4:
        score += 2
    elif len(clue_words) <= 6:
        score += 1

    # Signal words present
    for w in clue_words:
        if w in _CRYPTIC_DEF_SIGNALS:
            score += 1
            break

    # No operation indicators
    if not operation_indicators:
        score += 1

    # All word roles are definition/linker (pipeline labelled everything as definition)
    if word_roles:
        roles = {wr.get('role', '') for wr in word_roles if isinstance(wr, dict)}
        if roles <= {'definition', 'linker', 'link', ''}:
            score += 1

    # Quality is 'none' — pipeline had no confidence
    if quality == 'none':
        score += 1

    if score >= 4:
        return {
            'word': '(cryptic definition)',
            'letters': answer,
            'method': f'cryptic definition (score={score}): entire clue = {answer}'
        }

    return None


# ======================================================================
# HELPER: ANAGRAM WITH DELETION
# ======================================================================

# Module-level cache for anagram indicator words
_anagram_indicators: Optional[Set[str]] = None
_deletion_indicators: Optional[Set[str]] = None


def _load_anagram_indicators(db: DatabaseLookup) -> Set[str]:
    """Load anagram indicator words from the DB. Cached."""
    global _anagram_indicators
    if _anagram_indicators is not None:
        return _anagram_indicators
    conn = get_cryptic_connection()
    try:
        cursor = conn.execute(
            "SELECT DISTINCT word FROM indicators WHERE wordplay_type = 'anagram'")
        _anagram_indicators = {row[0].lower() for row in cursor.fetchall()}
    finally:
        conn.close()
    return _anagram_indicators


def _load_deletion_indicators(db: DatabaseLookup) -> Set[str]:
    """Load deletion indicator words from the DB. Cached."""
    global _deletion_indicators
    if _deletion_indicators is not None:
        return _deletion_indicators
    conn = get_cryptic_connection()
    try:
        cursor = conn.execute(
            "SELECT DISTINCT word FROM indicators WHERE wordplay_type = 'deletion'")
        _deletion_indicators = {row[0].lower() for row in cursor.fetchall()}
    finally:
        conn.close()
    return _deletion_indicators


def _is_anagram(letters1: str, letters2: str) -> bool:
    """Check if two strings are anagrams of each other (case-insensitive)."""
    return sorted(letters1.upper()) == sorted(letters2.upper())


def _build_anagram_deletion_roles(clue_tokens: List[str], def_words: set,
                                   anagram_idxs: List[int],
                                   deletion_info: List[tuple],
                                   fodder_tokens: List[str],
                                   target_word: str = None,
                                   target_sub: str = None) -> List[Dict]:
    """Build fresh word_roles for an anagram_deletion solve."""
    roles = []
    fodder_lower = {t.lower() for t in fodder_tokens}
    anagram_idx_set = set(anagram_idxs)
    del_idx_set = {d[0] for d in deletion_info}
    target_lower = target_word.lower() if target_word else None

    for i, token in enumerate(clue_tokens):
        token_lower = token.lower()
        if token_lower in def_words:
            roles.append({'word': token, 'role': 'definition',
                          'contributes': '',
                          'source': 'secondary_anagram_deletion'})
        elif i in anagram_idx_set:
            roles.append({'word': token, 'role': 'anagram_indicator',
                          'contributes': '',
                          'source': 'secondary_anagram_deletion'})
        elif i in del_idx_set:
            roles.append({'word': token, 'role': 'deletion_indicator',
                          'contributes': '',
                          'source': 'secondary_anagram_deletion'})
        elif target_lower and token_lower == target_lower:
            roles.append({'word': token, 'role': 'deletion_target',
                          'contributes': target_sub or '',
                          'source': 'secondary_anagram_deletion'})
        elif token_lower in fodder_lower:
            roles.append({'word': token, 'role': 'anagram_fodder',
                          'contributes': norm_letters(token).upper(),
                          'source': 'secondary_anagram_deletion'})
        else:
            roles.append({'word': token, 'role': 'linker',
                          'contributes': '',
                          'source': 'secondary_anagram_deletion'})
    return roles


def attempt_anagram_deletion(db: DatabaseLookup, answer: str, clue_text: str,
                              definition_window: str,
                              word_coverage: float) -> Optional[Dict]:
    """
    Fresh parse for anagram+deletion combos.
    When anagram fodder has more letters than the answer, look for a deletion
    indicator that tells us what to remove.

    Only fires when word_coverage < 0.5 (existing parse is unreliable).

    Returns dict with 'word', 'letters', 'method' if successful, else None.
    """
    if not answer or not clue_text:
        return None

    # Gate: only try fresh parse when existing evidence is poor
    # Threshold 0.8: stage_general assigns roles to most words even when wrong
    if word_coverage >= 0.8:
        return None

    anagram_words = _load_anagram_indicators(db)
    deletion_words = _load_deletion_indicators(db)

    clue_tokens = re.findall(r"[a-zA-Z']+", clue_text)
    clue_lower = [t.lower() for t in clue_tokens]

    # Find anagram indicator(s)
    anagram_idxs = []
    for i, token in enumerate(clue_lower):
        if token in anagram_words:
            anagram_idxs.append(i)

    if not anagram_idxs:
        return None

    # Find deletion indicator(s) and what follows them
    deletion_info = []
    for i, token in enumerate(clue_lower):
        if token in deletion_words:
            deletion_info.append((i, token))

    # Determine definition window words (to exclude from fodder)
    def_words = set()
    if definition_window:
        def_words = {w.lower() for w in re.findall(r"[a-zA-Z']+", definition_window)}

    # Identify fodder: everything not definition, not anagram indicator, not deletion indicator
    indicator_idxs = set(anagram_idxs) | {d[0] for d in deletion_info}
    fodder_tokens = []
    fodder_idxs = []
    for i, token in enumerate(clue_tokens):
        if i in indicator_idxs:
            continue
        if token.lower() in def_words:
            continue
        fodder_tokens.append(token)
        fodder_idxs.append(i)

    if not fodder_tokens:
        return None

    fodder_letters = ''.join(norm_letters(t) for t in fodder_tokens).upper()
    answer_upper = answer.upper()
    answer_len = len(answer_upper)

    # Case 1: Fodder letters exactly match answer (pure anagram missed by earlier stage)
    if len(fodder_letters) == answer_len and _is_anagram(fodder_letters, answer_upper):
        fodder_str = ' + '.join(fodder_tokens)
        return {
            'word': clue_tokens[anagram_idxs[0]],
            'letters': answer_upper,
            'method': f'anagram({fodder_str}) = {answer_upper}',
            'fresh_word_roles': _build_anagram_deletion_roles(
                clue_tokens, def_words, anagram_idxs, deletion_info, fodder_tokens)
        }

    # Case 2: Fodder has excess letters — look for deletion to reconcile
    excess = len(fodder_letters) - answer_len

    if excess < 1 or not deletion_info:
        return None

    # For each deletion indicator, try strategies to reconcile the excess
    fodder_lower_set = {t.lower() for t in fodder_tokens}

    for del_idx, del_word in deletion_info:

        # Strategy A: deletion target word is sitting in fodder, inflating count
        # e.g., "ignoring one" → "one" is in fodder but should be excluded,
        # then its substitution (I) removed from remaining fodder letters
        if del_idx + 1 < len(clue_tokens):
            next_word = clue_tokens[del_idx + 1]
            next_lower = next_word.lower()

            if next_lower in fodder_lower_set:
                # Remove target word from fodder and recalculate
                reduced_tokens = [t for t in fodder_tokens if t.lower() != next_lower]
                reduced_letters = ''.join(norm_letters(t) for t in reduced_tokens).upper()
                reduced_excess = len(reduced_letters) - answer_len

                # Pure anagram after removing target word
                if reduced_excess == 0 and _is_anagram(reduced_letters, answer_upper):
                    fodder_str = ' + '.join(reduced_tokens)
                    return {
                        'word': clue_tokens[anagram_idxs[0]],
                        'letters': answer_upper,
                        'method': f'anagram({fodder_str}) = {answer_upper}',
                        'fresh_word_roles': _build_anagram_deletion_roles(
                            clue_tokens, def_words, anagram_idxs, deletion_info,
                            reduced_tokens, next_word)
                    }

                # Target's substitution accounts for remaining excess
                if 1 <= reduced_excess <= 3:
                    subs = db.lookup_substitution(next_lower,
                                                   max_synonym_length=reduced_excess)
                    for sub in subs:
                        sub_letters = norm_letters(sub.letters).upper()
                        if len(sub_letters) == reduced_excess:
                            final = list(reduced_letters)
                            valid = True
                            for c in sub_letters:
                                if c in final:
                                    final.remove(c)
                                else:
                                    valid = False
                                    break
                            if valid and _is_anagram(''.join(final), answer_upper):
                                fodder_str = ' + '.join(reduced_tokens)
                                return {
                                    'word': clue_tokens[anagram_idxs[0]],
                                    'letters': answer_upper,
                                    'method': (f'anagram({fodder_str}) minus '
                                               f'{sub.letters}({next_word}) = {answer_upper}'),
                                    'fresh_word_roles': _build_anagram_deletion_roles(
                                        clue_tokens, def_words, anagram_idxs,
                                        deletion_info, reduced_tokens,
                                        next_word, sub.letters)
                                }

            # Original approach: target not in fodder, or in-fodder approach failed
            # Only viable when original excess is small
            if 1 <= excess <= 3:
                subs = db.lookup_substitution(next_lower, max_synonym_length=excess)
                for sub in subs:
                    sub_letters = norm_letters(sub.letters).upper()
                    if len(sub_letters) == excess:
                        reduced = list(fodder_letters)
                        valid = True
                        for c in sub_letters:
                            if c in reduced:
                                reduced.remove(c)
                            else:
                                valid = False
                                break
                        if valid and _is_anagram(''.join(reduced), answer_upper):
                            fodder_str = ' + '.join(fodder_tokens)
                            return {
                                'word': clue_tokens[anagram_idxs[0]],
                                'letters': answer_upper,
                                'method': (f'anagram({fodder_str}) minus '
                                           f'{sub.letters}({next_word}) = {answer_upper}'),
                                'fresh_word_roles': _build_anagram_deletion_roles(
                                    clue_tokens, def_words, anagram_idxs,
                                    deletion_info, fodder_tokens,
                                    next_word, sub.letters)
                            }

                # Try the word itself as letters to remove
                next_letters = norm_letters(next_lower).upper()
                if len(next_letters) == excess and next_lower not in def_words:
                    reduced = list(fodder_letters)
                    valid = True
                    for c in next_letters:
                        if c in reduced:
                            reduced.remove(c)
                        else:
                            valid = False
                            break
                    if valid and _is_anagram(''.join(reduced), answer_upper):
                        fodder_str = ' + '.join(fodder_tokens)
                        return {
                            'word': clue_tokens[anagram_idxs[0]],
                            'letters': answer_upper,
                            'method': (f'anagram({fodder_str}) minus '
                                       f'{next_letters}({next_word}) = {answer_upper}'),
                            'fresh_word_roles': _build_anagram_deletion_roles(
                                clue_tokens, def_words, anagram_idxs,
                                deletion_info, fodder_tokens,
                                next_word, next_letters)
                        }

        # Strategy B: self-contained deletion indicator modifies a fodder word
        # e.g., "endlessly" → remove last letter of an adjacent fodder word
        if 1 <= excess <= 2:
            del_subtype = None
            for ind_word, ind_sub in [('endlessly', 'tail'), ('headless', 'head'),
                                       ('heartless', 'middle'), ('topless', 'head'),
                                       ('endless', 'tail'), ('curtailed', 'tail')]:
                if del_word == ind_word:
                    del_subtype = ind_sub
                    break

            if del_subtype:
                for fi, ft in enumerate(fodder_tokens):
                    ft_letters = norm_letters(ft).upper()
                    if len(ft_letters) < 2:
                        continue

                    if del_subtype == 'tail':
                        modified = ft_letters[:-1]
                    elif del_subtype == 'head':
                        modified = ft_letters[1:]
                    elif del_subtype == 'middle' and len(ft_letters) >= 3:
                        mid = len(ft_letters) // 2
                        modified = ft_letters[:mid] + ft_letters[mid + 1:]
                    else:
                        continue

                    new_fodder = ''.join(
                        norm_letters(fodder_tokens[j]).upper() if j != fi else modified
                        for j in range(len(fodder_tokens))
                    )
                    if len(new_fodder) == answer_len and _is_anagram(new_fodder, answer_upper):
                        fodder_str = ' + '.join(fodder_tokens)
                        return {
                            'word': clue_tokens[anagram_idxs[0]],
                            'letters': answer_upper,
                            'method': (f'anagram({fodder_str}) with '
                                       f'{del_subtype}({ft}) = {answer_upper}'),
                            'fresh_word_roles': _build_anagram_deletion_roles(
                                clue_tokens, def_words, anagram_idxs,
                                deletion_info, fodder_tokens)
                        }

    return None


# ======================================================================
# HELPER: API-BASED SYNONYM LOOKUP (Merriam-Webster)
# ======================================================================

def attempt_api_synonym(db: DatabaseLookup, answer: str, letters_needed: str,
                        unresolved: List[str]) -> Optional[Dict]:
    """
    Use Merriam-Webster API to check if any unresolved word is a synonym
    that matches the needed letters.

    Only calls API if local DB lookup failed. Caches results.

    Returns dict with 'word', 'letters', 'method' if match found.
    """
    if not letters_needed or not unresolved:
        return None

    # Import here to avoid startup cost if not needed
    try:
        from external_apis import check_synonym_via_api
    except ImportError:
        return None

    needed_upper = letters_needed.upper()

    for word in unresolved:
        word_clean = word.strip().lower()
        if len(word_clean) < 3:
            continue

        # Check if this word's synonym could be the needed letters
        # Try API lookup: is needed_letters a synonym of this word?
        if check_synonym_via_api(word_clean, needed_upper.lower()):
            return {
                'word': word,
                'letters': needed_upper,
                'method': f'{word} = {needed_upper} (via API)',
                'role': 'wordplay'
            }

    return None


def attempt_api_double_definition(answer: str, clue_text: str) -> Optional[Dict]:
    """
    For very short clues (2-3 words), check if both halves are valid definitions
    for the answer using MW API + local DB.

    This handles cases like "Pale-looking pie" = PASTY where both parts are
    definitions but neither is in the local DB.

    Returns dict with 'method' if both parts validate as definitions.
    """
    if not answer or not clue_text:
        return None

    # Import here
    try:
        from external_apis import validate_definition_via_api
    except ImportError:
        return None

    words = clue_text.split()
    if len(words) < 2 or len(words) > 4:
        return None

    # Helper to check both API and local DB
    def is_valid_definition(defn: str, ans: str) -> bool:
        # Check API first
        if validate_definition_via_api(defn, ans):
            return True
        # Check local DB
        conn = sqlite3.connect(CRYPTIC_DB_PATH)
        cursor = conn.cursor()
        cursor.execute("""
            SELECT 1 FROM definition_answers_augmented
            WHERE LOWER(definition) = ? AND UPPER(REPLACE(answer, ' ', '')) = ?
            LIMIT 1
        """, (defn.lower(), ans.upper().replace(' ', '')))
        exists = cursor.fetchone() is not None
        conn.close()
        if exists:
            return True

        # Try stripping hyphenated suffix (e.g., "Pale-looking" -> "Pale")
        if '-' in defn:
            base = defn.split('-')[0]
            if len(base) >= 3:
                conn = sqlite3.connect(CRYPTIC_DB_PATH)
                cursor = conn.cursor()
                cursor.execute("""
                    SELECT 1 FROM definition_answers_augmented
                    WHERE LOWER(definition) = ? AND UPPER(REPLACE(answer, ' ', '')) = ?
                    LIMIT 1
                """, (base.lower(), ans.upper().replace(' ', '')))
                exists = cursor.fetchone() is not None
                conn.close()
                if exists:
                    return True

        return False

    # Try splitting at each position
    for split_point in range(1, len(words)):
        part1 = ' '.join(words[:split_point]).strip()
        part2 = ' '.join(words[split_point:]).strip()

        # Remove trailing punctuation and hyphens
        part1 = part1.rstrip('?!.,;:').rstrip('-')
        part2 = part2.rstrip('?!.,;:').rstrip('-')

        if is_valid_definition(part1, answer) and is_valid_definition(part2, answer):
            return {
                'method': f'double_definition_api: {part1} / {part2} = {answer}',
                'def1': part1,
                'def2': part2
            }

    return None


# ======================================================================
# HELPER: CROSS-REFERENCE (clues_master.db lookup)
# ======================================================================

# Path to clues master database
CLUES_DB_PATH = Path(
    r'C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\clues_master.db')


def attempt_cross_reference(answer: str, clue_text: str,
                             letters_needed: str, unresolved: List[str],
                             db: DatabaseLookup) -> Optional[Dict]:
    """
    Look up other clues with the same answer in clues_master.db.
    Extract common definition words and wordplay patterns.

    Primarily a suggestion mechanism — auto-solves only with high confidence.

    Returns dict with 'word', 'letters', 'method' if high-confidence match,
    else None.
    """
    if not answer or not CLUES_DB_PATH.exists():
        return None

    try:
        conn = sqlite3.connect(CLUES_DB_PATH)
        cursor = conn.execute(
            "SELECT clue_text FROM clues WHERE UPPER(answer) = ? AND clue_text != ? LIMIT 50",
            (answer.upper(), clue_text))
        other_clues = [row[0] for row in cursor.fetchall()]
        conn.close()
    except Exception:
        return None

    if len(other_clues) < 2:
        return None

    # Extract common words across other clues for this answer
    # Words appearing in >30% of other clues are likely definition words
    word_counts = {}
    for oc in other_clues:
        oc_words = set(re.findall(r'[a-zA-Z]+', oc.lower()))
        for w in oc_words:
            if len(w) > 2 and norm_letters(w) not in LINKERS:
                word_counts[w] = word_counts.get(w, 0) + 1

    threshold = max(2, len(other_clues) * 0.3)
    common_def_words = {w for w, c in word_counts.items() if c >= threshold}

    if not common_def_words:
        return None

    # Check if any of our unresolved words appear commonly as definition words
    # for this answer — if so, they're likely part of the definition, not wordplay
    # This doesn't directly solve, but provides a hint
    our_words = {w.lower() for w in unresolved}
    definition_hints = common_def_words & our_words

    # Try to find common wordplay breakdowns:
    # If many other clues share a word with ours that maps to a known substitution,
    # that substitution is likely correct for our clue too
    for word in unresolved:
        word_lower = word.lower()
        if word_lower in common_def_words:
            continue  # This word is probably definition, skip

        # Check if this word appears in other clues AND has a known substitution
        word_in_others = sum(1 for oc in other_clues
                             if word_lower in oc.lower())
        if word_in_others >= 2:
            subs = db.lookup_substitution(word, max_synonym_length=len(letters_needed))
            for sub in subs:
                sub_letters = norm_letters(sub.letters).upper()
                if sub_letters == letters_needed.upper():
                    return {
                        'word': word,
                        'letters': sub_letters,
                        'method': (f'cross-ref: {word}={sub.letters} '
                                   f'(confirmed in {word_in_others} other clues)')
                    }

    return None


# ======================================================================
# DB SUGGESTION GENERATOR
# ======================================================================

def generate_db_suggestions(answer: str, clue_text: str, letters_needed: str,
                            unresolved: List[str],
                            definition_window: str) -> List[Dict]:
    """
    Generate SQL INSERT suggestions for missing DB entries.
    Runs on unsolved clues to produce actionable output.

    Returns list of suggestion dicts with 'table', 'sql', 'confidence', 'source'.
    """
    suggestions = []
    if not letters_needed or not unresolved:
        return suggestions

    needed_upper = letters_needed.upper()

    # Determine definition words to exclude
    def_words = set()
    if definition_window:
        def_words = {w.lower() for w in re.findall(r"[a-zA-Z']+", definition_window)}

    # Filter unresolved to non-definition, non-linker words
    candidate_words = [w for w in unresolved
                       if w.lower() not in def_words
                       and norm_letters(w) not in LINKERS]

    if not candidate_words:
        return suggestions

    # If there's exactly one candidate word and it could produce the missing letters
    if len(candidate_words) == 1:
        word = candidate_words[0]
        word_clean = norm_letters(word).lower()
        if len(needed_upper) <= 4:
            sql = (f"INSERT INTO wordplay (indicator, substitution, category, notes) "
                   f"VALUES ('{word_clean}', '{needed_upper}', 'abbreviation', "
                   f"'suggested from clue: {answer}');")
            suggestions.append({
                'table': 'wordplay',
                'sql': sql,
                'confidence': 'medium' if len(needed_upper) <= 2 else 'low',
                'source': 'single_unresolved_word'
            })

    # If all candidate words together form a phrase that should map to remaining letters
    if len(candidate_words) >= 2:
        phrase = ' '.join(w.lower() for w in candidate_words)
        phrase_clean = norm_letters(phrase).lower()
        # Only suggest if the phrase is short enough to be a reasonable synonym
        if len(candidate_words) <= 4 and len(needed_upper) >= 3:
            sql = (f"INSERT INTO synonyms_pairs (word, synonym) "
                   f"VALUES ('{needed_upper.lower()}', '{phrase}');")
            suggestions.append({
                'table': 'synonyms_pairs',
                'sql': sql,
                'confidence': 'low',
                'source': 'unresolved_phrase'
            })

    return suggestions


# ======================================================================
# INDICATOR ATTRIBUTION
# ======================================================================

def _detect_mechanisms_from_roles(answer: str, word_roles: List[Dict]) -> Set[str]:
    """
    Identify which cryptic mechanisms are demonstrably present
    from already-resolved word roles.
    """
    mechanisms = set()

    for wr in word_roles:
        if not isinstance(wr, dict):
            continue
        role = (wr.get('role') or '').lower()
        source = (wr.get('source') or '').lower()

        if 'anagram' in role or 'anagram' in source or role == 'fodder':
            mechanisms.add('anagram')
        if 'reversal' in role or 'reversal' in source:
            mechanisms.add('reversal')
        if role == 'homophone' or 'homophone' in source:
            mechanisms.add('homophone')
        if role in ('first_use', 'last_use', 'central_use', 'outer_use',
                    'head_use', 'tail_use'):
            mechanisms.add('parts')
        if 'deletion' in role or 'deletion' in source:
            mechanisms.add('deletion')

    # Container detection: a non-anagram, non-definition contribution
    # appears as a non-edge substring of the answer
    for wr in word_roles:
        if not isinstance(wr, dict):
            continue
        role = (wr.get('role') or '').lower()
        if 'anagram' in role or role in ('definition', 'linker', 'link'):
            continue
        contrib = (wr.get('contributes') or '').upper().replace(' ', '')
        if contrib:
            pos = answer.upper().find(contrib)
            if 0 < pos < len(answer) - len(contrib):
                mechanisms.add('container')

    return mechanisms


def attempt_indicator_attribution(db: DatabaseLookup, answer: str,
                                  word_roles: List[Dict],
                                  unresolved: List[str]) -> Optional[Dict]:
    """
    Last-resort attribution: when all answer letters are accounted for but
    words remain unresolved, attribute them in two passes:

    Pass 1 — Indicator attribution: words found in the indicators table
    whose type matches a mechanism demonstrably present in the word roles
    are labelled as indicators (e.g. "moving = anagram indicator").

    Pass 2 — Surface text: any remaining words (not indicators, or indicators
    whose mechanism wasn't detected) are labelled as surface text. Since all
    letters are already accounted for, these words exist only to make the
    clue read naturally — they play no wordplay role.

    Returns dict with 'attributed', 'remaining'=[], 'method', 'letters'.
    Always fully resolves the clue when letters are complete.
    """
    if not unresolved:
        return None

    mechanisms = _detect_mechanisms_from_roles(answer, word_roles)

    # Map indicator wordplay_type to the mechanism name used above
    TYPE_TO_MECHANISM = {
        'anagram': 'anagram',
        'container': 'container',
        'insertion': 'container',
        'reversal': 'reversal',
        'deletion': 'deletion',
        'head': 'parts',
        'tail': 'parts',
        'first_use': 'parts',
        'last_use': 'parts',
        'central_use': 'parts',
        'outer_use': 'parts',
        'homophone': 'homophone',
    }

    attributed = []
    method_parts = []

    for word in unresolved:
        # Pass 1: try to attribute as an indicator
        ind = db.lookup_indicator(word)
        if ind and mechanisms:
            mech = TYPE_TO_MECHANISM.get(ind.wordplay_type)
            if mech and mech in mechanisms:
                attributed.append({
                    'word': word,
                    'indicator_type': ind.wordplay_type,
                    'method': f'{word} = {ind.wordplay_type} indicator'
                })
                method_parts.append(f'{word} = {ind.wordplay_type} indicator')
                continue

        # Pass 2: label as surface text — letters are complete so this word
        # exists purely for surface reading
        attributed.append({
            'word': word,
            'indicator_type': 'surface',
            'method': f'{word} = surface text'
        })
        method_parts.append(f'{word} = surface text')

    if not attributed:
        return None

    return {
        'attributed': attributed,
        'remaining': [],
        'method': '; '.join(method_parts),
        'letters': '',
    }


# ======================================================================
# ORCHESTRATOR
# ======================================================================

def analyze_failure(record: Dict[str, Any], db: DatabaseLookup) -> Dict[str, Any]:
    """
    Analyze a single failure from stage_general.
    Applies helpers in sequence:
      0. cryptic_definition (narrow)
      1. linker_strip
      2. acrostic
      3. reversal
      4. partial_resolve (parts/extraction)
      5. deletion
      6. container
      7. homophone
      8. near_miss
      9. anagram_deletion (fresh parse, gated by word coverage)
     10. wider_cryptic_def (heuristic)
     11. cross_reference (clues_master.db lookup)
     Finally: db_suggestion_generator (always runs on unsolved)

    Returns enriched record with helper results.
    """
    answer = (record.get('answer') or '').upper().replace(' ', '')
    formula = record.get('formula', '') or ''
    letters_needed = (record.get('letters_still_needed') or '').upper()
    unresolved = list(record.get('unresolved_words') or [])
    operation_indicators = record.get('operation_indicators') or []
    word_roles = record.get('word_roles') or []
    clue_text = record.get('clue_text', '') or ''

    # Word coverage: fraction of clue words with assigned roles
    word_coverage = compute_word_coverage(clue_text, unresolved)

    # Step 0: Cryptic definition detection
    # If no letters needed, no unresolved words, and all roles are 'definition',
    # then the entire clue is the definition — it's a cryptic definition.
    if not letters_needed and not unresolved:
        roles = set()
        for wr in word_roles:
            if isinstance(wr, dict):
                roles.add(wr.get('role', ''))
        if roles <= {'definition', 'linker', 'link', ''}:
            return _build_result(
                record, answer,
                f'cryptic_definition = {answer}',
                '', 'cryptic_definition', True, [], [])

    # Compute found_letters: answer minus letters_needed
    found_letters = answer
    temp_needed = list(letters_needed)
    for c in temp_needed:
        idx = found_letters.find(c)
        if idx >= 0:
            found_letters = found_letters[:idx] + found_letters[idx + 1:]

    # Step 1: Strip linkers from unresolved
    cleaned_unresolved, removed_linkers = strip_linkers(unresolved)

    # If linker stripping resolved everything
    if not cleaned_unresolved and not letters_needed:
        return _build_result(record, answer, formula, '', 'linker_strip',
                             True, [], removed_linkers)

    # Recalculate after linker stripping
    working_unresolved = cleaned_unresolved
    working_needed = letters_needed

    # Step 1.2: Definition swap — if the definition word has a substitution
    # matching answer letters, it's probably wordplay, not the definition
    result = attempt_definition_swap(db, answer, working_needed, clue_text,
                                      word_roles, working_unresolved)
    if result:
        remaining = _subtract_letters(working_needed, result['letters'])
        if not remaining:
            improved = _improve_formula(formula, result)
            return _build_result(record, answer, improved, '', 'definition_swap',
                                 True, [], removed_linkers, result)
        else:
            # Partial progress — update working_needed and found_letters
            working_needed = remaining
            found_letters = answer
            temp = list(working_needed)
            for c in temp:
                idx = found_letters.find(c)
                if idx >= 0:
                    found_letters = found_letters[:idx] + found_letters[idx + 1:]

    # Step 1.5: Synonym override — reclaim indicator-tagged words as synonyms
    # when their synonym matches the start of the needed letters
    result = attempt_synonym_override(db, answer, working_needed,
                                       word_roles, working_unresolved)
    if result:
        remaining = _subtract_letters(working_needed, result['letters'])
        if not remaining:
            improved = _improve_formula(formula, result)
            return _build_result(record, answer, improved, '', 'synonym_override',
                                 True, [], removed_linkers, result)
        else:
            # Partial progress — update working_needed for later handlers
            working_needed = remaining
            # Recompute found_letters for container handler
            found_letters = answer
            temp = list(working_needed)
            for c in temp:
                idx = found_letters.find(c)
                if idx >= 0:
                    found_letters = found_letters[:idx] + found_letters[idx + 1:]

    # Step 1.6: Synonym substring — check if unresolved/indicator word has a synonym
    # that is a contiguous substring of the answer, covering remaining letters
    result = attempt_synonym_substring(db, answer, working_needed, found_letters,
                                        working_unresolved, word_roles)
    if result:
        improved = _improve_formula(formula, result)
        return _build_result(record, answer, improved, '', 'synonym_substring',
                             True, [], removed_linkers, result)

    # Step 2: Try acrostic
    definition_window = record.get('definition_window', '') or ''
    result = attempt_acrostic(db, answer, working_needed, clue_text,
                              definition_window, working_unresolved)
    if result:
        remaining = _subtract_letters(working_needed, result['letters'])
        if not remaining:
            improved = _improve_formula(formula, result)
            return _build_result(record, answer, improved, '', 'acrostic',
                                 True, [], removed_linkers, result)

    # Step 3: Try reversal
    result = attempt_reversal(db, answer, working_needed, working_unresolved,
                              operation_indicators)
    if result and not working_needed.replace(result['letters'], '', 1):
        improved = _improve_formula(formula, result)
        return _build_result(record, answer, improved, '', 'reversal',
                             True, [], removed_linkers, result)

    # Step 4: Try partial resolution (parts/extraction)
    result = attempt_partial_resolve(db, answer, working_needed, clue_text,
                                     definition_window, working_unresolved,
                                     operation_indicators)
    if result:
        remaining = _subtract_letters(working_needed, result['letters'])
        if not remaining:
            improved = _improve_formula(formula, result)
            return _build_result(record, answer, improved, '', 'partial_resolve',
                                 True, [], removed_linkers, result)

    # Step 5: Try deletion
    result = attempt_deletion(db, answer, working_needed, working_unresolved,
                              operation_indicators)
    if result:
        remaining = _subtract_letters(working_needed, result['letters'])
        if not remaining:
            improved = _improve_formula(formula, result)
            return _build_result(record, answer, improved, '', 'deletion',
                                 True, [], removed_linkers, result)

    # Step 5.5: Try multi-word container with parts extraction
    # (e.g., "seeming discontented about dog at home" → S(CUR+IN)G)
    result = attempt_multiword_container(db, answer, working_needed, working_unresolved,
                                        operation_indicators, found_letters, word_roles)
    if result:
        improved = _improve_formula(formula, result)
        return _build_result(record, answer, improved, '', 'multiword_container',
                             True, [], removed_linkers, result)

    # Step 6: Try container/insertion
    result = attempt_container(db, answer, working_needed, working_unresolved,
                               operation_indicators, found_letters)
    if result:
        improved = _improve_formula(formula, result)
        return _build_result(record, answer, improved, '', 'container',
                             True, [], removed_linkers, result)

    # Step 7: Try homophone
    result = attempt_homophone(db, answer, working_needed, working_unresolved,
                               operation_indicators)
    if result:
        remaining = _subtract_letters(working_needed, result['letters'])
        if not remaining:
            improved = _improve_formula(formula, result)
            return _build_result(record, answer, improved, '', 'homophone',
                                 True, [], removed_linkers, result)

    # Step 7.5: Try substitution swap — replace an already-resolved substitution
    # with an alternative that exactly fills the remaining letter gap
    result = attempt_substitution_swap(db, answer, working_needed, word_roles)
    if result:
        remaining = _subtract_letters(working_needed, result['letters'])
        if not remaining:
            improved = _improve_formula(formula, result)
            return _build_result(record, answer, improved, '', 'substitution_swap',
                                 True, [], removed_linkers, result)

    # Step 8: Try near-miss (1-2 letter gaps)
    result = attempt_near_miss(db, answer, working_needed, working_unresolved)
    if result:
        remaining = _subtract_letters(working_needed, result['letters'])
        if not remaining:
            improved = _improve_formula(formula, result)
            return _build_result(record, answer, improved, '', 'near_miss',
                                 True, [], removed_linkers, result)

    # Step 9: Try anagram with deletion (fresh parse, gated by word coverage)
    result = attempt_anagram_deletion(db, answer, clue_text,
                                       definition_window, word_coverage)
    if result:
        improved = _improve_formula(formula, result)
        return _build_result(record, answer, improved, '', 'anagram_deletion',
                             True, [], removed_linkers, result)

    # Step 10: Try API-based double definition (for short clues)
    # Run if: short clue AND (no letters needed OR all letters needed = no partial progress)
    # IMPORTANT: Must run BEFORE wider_cryptic_def to get proper double-definition detection
    answer_norm = answer.upper().replace(' ', '')
    if len(clue_text.split()) <= 4 and (not working_needed or working_needed == answer_norm):
        result = attempt_api_double_definition(answer, clue_text)
        if result:
            return _build_result(record, answer, result['method'], '',
                                'api_double_definition',
                                True, [], removed_linkers, result)

    # Step 10.5: Try API-based synonym lookup for remaining letters
    if working_needed and len(working_needed) >= 3:
        result = attempt_api_synonym(db, answer, working_needed, working_unresolved)
        if result:
            remaining = _subtract_letters(working_needed, result['letters'])
            if not remaining:
                improved = _improve_formula(formula, result)
                return _build_result(record, answer, improved, '', 'api_synonym',
                                    True, [], removed_linkers, result)

    # Step 11: Try wider cryptic definition
    quality = record.get('quality', '') or ''
    result = attempt_wider_cryptic_def(answer, clue_text, formula, quality,
                                        working_needed, operation_indicators,
                                        word_roles)
    if result:
        improved = result['method']
        return _build_result(record, answer, improved, '', 'wider_cryptic_def',
                             True, [], removed_linkers, result)

    # Step 12: Try cross-reference
    result = attempt_cross_reference(answer, clue_text, working_needed,
                                      working_unresolved, db)
    if result:
        remaining = _subtract_letters(working_needed, result['letters'])
        if not remaining:
            improved = _improve_formula(formula, result)
            return _build_result(record, answer, improved, '', 'cross_reference',
                                 True, [], removed_linkers, result)

    # Step 13: Indicator attribution — all letters found but indicator words
    # remain unresolved; attribute them from the resolved word roles
    if not working_needed and working_unresolved:
        result = attempt_indicator_attribution(db, answer, word_roles,
                                               working_unresolved)
        if result:
            fully_now = not result['remaining']
            return _build_result(record, answer, formula, '', 'indicator_attribution',
                                 fully_now, result['remaining'], removed_linkers, result)

    # Generate DB suggestions for unsolved clues (always runs)
    db_suggestions = generate_db_suggestions(answer, clue_text, working_needed,
                                              working_unresolved, definition_window)

    # No helper resolved it — return with linker cleanup only
    new_needed = working_needed
    result_dict = _build_result(record, answer, formula, new_needed, 'none',
                                False, working_unresolved, removed_linkers)
    result_dict['db_suggestions'] = db_suggestions
    return result_dict


def _subtract_letters(needed: str, found: str) -> str:
    """Remove found letters from needed (one occurrence each)."""
    remaining = list(needed.upper())
    for c in found.upper():
        if c in remaining:
            remaining.remove(c)
    return ''.join(remaining)


def _improve_formula(original: str, helper_result: Dict) -> str:
    """Build improved formula incorporating helper result."""
    method = helper_result.get('method', '')
    if not original:
        return method
    # Replace the "? = ANSWER" tail or append
    if '?' in original:
        return original.replace('?', method, 1)
    return f"{original} + {method}"


def _build_result(record: Dict, answer: str, formula: str,
                  letters_needed: str, helper_used: str,
                  fully_resolved: bool, remaining_unresolved: List[str],
                  removed_linkers: List[str],
                  helper_result: Optional[Dict] = None) -> Dict[str, Any]:
    """Build a result dict for saving."""

    # Use fresh_word_roles from handler if provided (e.g., anagram_deletion
    # does a fresh parse and knows the correct roles for every word)
    if helper_result and 'fresh_word_roles' in helper_result:
        word_roles = helper_result['fresh_word_roles']
        # Rebuild breakdown from fresh roles instead of inheriting stage_general's
        breakdown = []
        for wr in word_roles:
            w = wr.get('word', '')
            role = wr.get('role', '')
            contrib = wr.get('contributes', '')
            if contrib:
                breakdown.append(f'"{w}" = {role} ({contrib})')
            else:
                breakdown.append(f'"{w}" = {role}')
        breakdown.append(f"Secondary: {helper_result.get('method', '')}")
    else:
        breakdown = list(record.get('breakdown') or [])

        if removed_linkers:
            breakdown.append(f"Linkers stripped: {', '.join(removed_linkers)}")

        if helper_result:
            breakdown.append(f"Secondary: {helper_result.get('method', '')}")

        word_roles = list(record.get('word_roles') or [])
        if helper_result:
            word_roles.append({
                'word': helper_result.get('word', ''),
                'role': 'substitution',
                'contributes': helper_result.get('letters', ''),
                'source': f'secondary_{helper_used}'
            })

        for lw in removed_linkers:
            word_roles.append({
                'word': lw,
                'role': 'linker',
                'contributes': '',
                'source': 'secondary_linker_strip'
            })

    # VALIDATION: Check for orphaned indicators
    # Only meaningful when there are still unresolved words — if unresolved is
    # empty, all words are attributed by definition and there are no orphans.
    if fully_resolved and remaining_unresolved:
        indicator_types_found = set()
        for wr in word_roles:
            role = wr.get('role', '').lower()
            if 'indicator' in role:
                # Extract the indicator type (e.g., "homophone indicator" -> "homophone")
                indicator_type = role.replace(' indicator', '').strip()
                indicator_types_found.add(indicator_type)

        # Check if any indicator was found but not used in the solve
        if indicator_types_found:
            # Map helper_used to indicator types
            helper_to_indicator = {
                'homophone': 'homophone',
                'anagram': 'anagram',
                'anagram_deletion': 'anagram',
                'container': 'container',
                'insertion': 'insertion',
                'reversal': 'reversal',
                'acrostic': 'acrostic',
                'deletion': 'deletion',
                'partial': 'parts',
            }

            used_indicator = helper_to_indicator.get(helper_used)

            if used_indicator not in indicator_types_found:
                fully_resolved = False

    return {
        'clue_id': record.get('clue_id'),
        'clue_text': record.get('clue_text', ''),
        'answer': answer,
        'original_formula': record.get('formula', ''),
        'improved_formula': formula,
        'helper_used': helper_used,
        'fully_resolved': 1 if fully_resolved else 0,
        'letters_still_needed': letters_needed,
        'unresolved_words': remaining_unresolved,
        'breakdown': breakdown,
        'word_roles': word_roles,
    }


# ======================================================================
# SAVE RESULTS
# ======================================================================

def save_stage_secondary(run_id: int, records: List[Dict[str, Any]]):
    """Save secondary analysis results to pipeline_stages.db."""
    conn = get_pipeline_connection()
    cursor = conn.cursor()

    for rec in records:
        cursor.execute("""
            INSERT INTO stage_secondary
            (run_id, clue_id, clue_text, answer, original_formula, improved_formula,
             helper_used, fully_resolved, letters_still_needed, unresolved_words,
             breakdown, word_roles, db_suggestions)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            run_id,
            rec.get('clue_id'),
            rec.get('clue_text', ''),
            rec.get('answer', ''),
            rec.get('original_formula', ''),
            rec.get('improved_formula', ''),
            rec.get('helper_used', ''),
            rec.get('fully_resolved', 0),
            rec.get('letters_still_needed', ''),
            json.dumps(rec.get('unresolved_words', [])),
            json.dumps(rec.get('breakdown', [])),
            json.dumps(rec.get('word_roles', [])),
            json.dumps(rec.get('db_suggestions', [])),
        ))

    conn.commit()
    conn.close()
    print(f"  Saved {len(records)} records to stage_secondary (run_id={run_id})")


# ======================================================================
# ENTRY POINT
# ======================================================================

def run_secondary_analysis(run_id: int = 0):
    """
    Run secondary analysis on stage_general failures.
    Entry point called from report.py after general stage.
    """
    init_secondary_table()
    clear_stage_secondary(run_id)

    print(f"\nLoading general-stage failures (run_id={run_id})...")
    failures = load_general_failures(run_id)
    print(f"  Total: {len(failures)} failures to process")

    if not failures:
        print("  No failures to process.")
        return []

    db = DatabaseLookup()
    results = []

    try:
        for i, record in enumerate(failures):
            try:
                result = analyze_failure(record, db)
                results.append(result)
            except Exception as e:
                print(f"  Warning: Error on clue {record.get('clue_id')}: {e}")
                continue

            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1}/{len(failures)} failures...")
    finally:
        db.close()

    # Summary
    solved_count = sum(1 for r in results if r.get('fully_resolved'))
    helper_counts = {}
    for r in results:
        h = r.get('helper_used', 'none')
        if r.get('fully_resolved'):
            helper_counts[h] = helper_counts.get(h, 0) + 1

    # Count DB suggestions
    suggestion_count = sum(len(r.get('db_suggestions', []))
                           for r in results if not r.get('fully_resolved'))

    print(f"\n  Secondary stage results:")
    print(f"    New solves: {solved_count}/{len(results)}")
    for helper, count in sorted(helper_counts.items()):
        print(f"      {helper}: {count}")
    if suggestion_count:
        print(f"    DB suggestions generated: {suggestion_count}")

    print(f"\nSaving secondary results...")
    save_stage_secondary(run_id, results)

    return results
