"""
05_self_learning_enrichment.py — Self-learning enrichment from pipeline failures.

For each failure where the answer is known, infers missing DB entries and
upserts them into cryptic_new.db, then re-runs the pipeline to measure improvement.

What it does:
  A. Definition pairs — extracted from definition-tagged word_roles + heuristic windows
  B. Single-word inference — 1 unresolved word, letters_still_needed is exact match
  C. Two-word anchor inference — try all splits of the full answer; if one word->split
     is already in DB (anchor), insert the other word->split
  D. Re-runs report.py and reports solve count improvement

Performance: loads all DB lookup tables into memory at startup to avoid full-table scans.

Usage:
  python -m enrichment.05_self_learning_enrichment
  python -m enrichment.05_self_learning_enrichment --dry-run
  python -m enrichment.05_self_learning_enrichment --source guardian --puzzle-number 29927
"""

import argparse
import json
import re
import subprocess
import sqlite3
import sys
from pathlib import Path
from typing import List, Optional, Set, Tuple

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from enrichment.common import (
    get_cryptic_conn, get_pipeline_conn,
    insert_wordplay, insert_synonym_pair, insert_definition_answer, insert_indicator,
    InsertCounter, add_common_args, apply_common_args, DRY_RUN, norm_letters
)

SOURCE_TAG = 'self_learning'

STOP_WORDS = {
    'a', 'an', 'the', 'of', 'in', 'on', 'at', 'to', 'for', 'and', 'or',
    'is', 'it', 'by', 'be', 'as', 'if', 'so', 'no', 'do', 'up', 'he',
    'she', 'we', 'me', 'my', 'his', 'her', 'its', 'not', 'but', 'all',
    'are', 'was', 'has', 'had', 'who', 'how', 'this', 'that', 'with',
    'from', 'have', 'where', 'when', 'what', 'which',
}

# Confirmed link words (must match the LINKERS set in report.py).
LINKERS = {
    'of', 'in', 'the', 'a', 'an', 'to', 'for', 'with', 'and', 'or',
    'by', 'from', 'as', 'on', 'at', 'but', 'so', 'yet', 'if', 'not',
    'nor', 'up', 'it', 'its', 'into', 'onto', 'within', 'without',
    'that', 'which', 'when', 'where', 'while', 'how', 'why', 'who',
    'this', 'these', 'those', 'such', 'one', 'ones', 'some', 'any',
    'all', 'here', 'there',
    'is', 'are', 'be', 'been', 'being', 'was', 'were',
    'has', 'have', 'had', 'having',
    'will', 'would', 'could', 'should', 'must', 'may', 'might',
    'get', 'gets', 'got', 'getting',
    'give', 'gives', 'gave', 'given', 'giving',
    'make', 'makes', 'made', 'making',
    'need', 'needs',
    'thus', 'hence', 'therefore', 'maybe',
    'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'cant', 'isnt', 'arent',
    # Temporal/adverbial surface words
    'once',
}


def _norm(word: str) -> str:
    """Normalize a word: lowercase, strip all non-alphabetic characters."""
    return re.sub(r'[^a-z]', '', word.lower())


# ============================================================
# PATTERN A HELPERS
# ============================================================

# Maps formula operation type → the role name used in word_roles for a confirmed indicator
_FORMULA_TYPE_TO_INDICATOR_ROLE = {
    'anagram':   'anagram_indicator',
    'insertion': 'insertion_indicator',
    'container': 'container_indicator',
    'reversal':  'reversal_indicator',
    'deletion':  'deletion_indicator',
}


def _infer_formula_type(formula: str, word_roles: list) -> Optional[str]:
    """Infer the cryptic operation type from word_roles (fodder signal) and the formula string.

    Checks word_roles for a [fodder] role first (anagram signal), then parses
    the formula string for known operation markers.
    Returns a type string ('anagram', 'insertion', etc.) or None if indeterminate.
    """
    for wr in word_roles:
        if wr.get('role') == 'fodder':
            return 'anagram'
    fl = (formula or '').lower()
    if 'with insertion' in fl:
        return 'insertion'
    if 'contained' in fl:
        return 'container'
    if 'reversed' in fl:
        return 'reversal'
    if 'deletion' in fl:
        return 'deletion'
    return None


def _has_indicator_for_type(word_roles: list, formula_type: str) -> bool:
    """Return True if an indicator of the given type already exists in word_roles.

    If the formula already has a confirmed indicator of this type, Pattern A
    should not propose new ones.
    """
    target_role = _FORMULA_TYPE_TO_INDICATOR_ROLE.get(formula_type)
    if not target_role:
        return False
    return any(wr.get('role') == target_role for wr in word_roles)


def _build_word_groups(clue_text: str, unresolved_words: list,
                       word_roles: list) -> List[List[Tuple[str, int]]]:
    """Build contiguous groups of unresolved words from the clue text.

    Rules:
    - LINKER words do not break groups (absorbed as interior connectors).
    - Non-LINKER USED words (those with a non-linker role in word_roles) break groups.
    - UNRESOLVED words start or extend groups.

    Returns a list of groups; each group is a list of (token, position) tuples.
    """
    unresolved_norm = {_norm(w) for w in unresolved_words}

    # Words with any non-linker role are USED and break groups
    used_norm: Set[str] = set()
    for wr in word_roles:
        role = wr.get('role', '')
        if role and role != 'linker':
            wn = _norm(wr.get('word', ''))
            if wn:
                used_norm.add(wn)

    clue_clean = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text)
    tokens = clue_clean.split()

    groups: List[List[Tuple[str, int]]] = []
    current_group: List[Tuple[str, int]] = []
    pending_linkers: List[Tuple[str, int]] = []

    for i, token in enumerate(tokens):
        tn = _norm(token)
        if not tn:
            continue

        if tn in unresolved_norm and tn not in used_norm:
            # Unresolved: flush pending linkers into group, then add this word
            current_group.extend(pending_linkers)
            current_group.append((token, i))
            pending_linkers = []
        elif tn in LINKERS and tn not in used_norm:
            # Linker: hold as pending — only joins a group if another unresolved follows
            if current_group:
                pending_linkers.append((token, i))
            # If no active group, linkers do not start one
        else:
            # Used non-linker word: close any active group
            if current_group:
                groups.append(current_group)
                current_group = []
            pending_linkers = []

    if current_group:
        groups.append(current_group)

    return groups


def _infer_definition_groups(
        groups: List[List[Tuple[str, int]]],
        clue_text: str,
        word_roles: list,
) -> Tuple[List, List]:
    """Separate unresolved groups into definition-extension groups and indicator groups.

    Rules:
    - If a definition word is identified in word_roles:
        Groups adjacent to the definition word(s) with only LINKERs between
        them → definition extension.
    - If no definition found:
        Groups at either edge of the clue (start or end) → definition.
    - All other groups → indicator candidates.

    Returns (definition_groups, indicator_groups).
    """
    clue_clean = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text)
    tokens = clue_clean.split()
    clue_len = len(tokens)

    def_words_norm = {_norm(wr.get('word', ''))
                      for wr in word_roles if wr.get('role') == 'definition'}
    def_positions = {i for i, t in enumerate(tokens) if _norm(t) in def_words_norm}

    def group_span(g):
        positions = [pos for _, pos in g]
        return min(positions), max(positions)

    def only_linkers_between(start_idx: int, end_idx: int) -> bool:
        """True if every token in range(start_idx, end_idx) is a LINKER."""
        for j in range(start_idx, end_idx):
            if j < len(tokens) and _norm(tokens[j]) not in LINKERS:
                return False
        return True

    definition_groups: List = []
    indicator_groups: List = []

    if def_positions:
        def_min = min(def_positions)
        def_max = max(def_positions)
        for group in groups:
            gmin, gmax = group_span(group)
            adjacent = False
            if gmax < def_min:
                # Group before definition: gap must be all LINKERs
                if only_linkers_between(gmax + 1, def_min):
                    adjacent = True
            elif gmin > def_max:
                # Group after definition: gap must be all LINKERs
                if only_linkers_between(def_max + 1, gmin):
                    adjacent = True
            if adjacent:
                definition_groups.append(group)
            else:
                indicator_groups.append(group)
    else:
        # No definition identified: groups at the clue edges are assumed to be definitions
        for group in groups:
            gmin, gmax = group_span(group)
            if gmin == 0 or gmax >= clue_len - 1:
                definition_groups.append(group)
            else:
                indicator_groups.append(group)

    return definition_groups, indicator_groups


# ============================================================
# IN-MEMORY CACHE
# ============================================================

class DBCache:
    """Loads key lookup tables into memory for fast repeated queries."""

    def __init__(self, conn: sqlite3.Connection):
        print("  Loading DB cache...")

        # synonyms_pairs: set of (word_lower, synonym_lower)
        self.synonym_pairs: Set[Tuple[str, str]] = set()
        for w, s in conn.execute("SELECT word, synonym FROM synonyms_pairs WHERE word != ''"):
            if w and s:
                self.synonym_pairs.add((w.lower(), s.lower()))
        print(f"    synonyms_pairs: {len(self.synonym_pairs):,}")

        # wordplay: set of (indicator_lower, substitution_lower)
        self.wordplay_pairs: Set[Tuple[str, str]] = set()
        for ind, sub in conn.execute("SELECT indicator, substitution FROM wordplay"):
            if ind and sub:
                self.wordplay_pairs.add((ind.lower(), sub.lower()))
        print(f"    wordplay: {len(self.wordplay_pairs):,}")

        # definition_answers_augmented: set of (definition_lower, answer_lower)
        self.def_answer_pairs: Set[Tuple[str, str]] = set()
        for d, a in conn.execute("SELECT definition, answer FROM definition_answers_augmented"):
            if d and a:
                self.def_answer_pairs.add((d.lower(), a.lower()))
        for d, a in conn.execute("SELECT definition, answer FROM definition_answers"):
            if d and a:
                self.def_answer_pairs.add((d.lower(), a.lower()))
        print(f"    definition_answers: {len(self.def_answer_pairs):,}")

        # indicators: set of word_lower + typed dict for wordplay_type lookup
        self.indicators: Set[str] = set()
        self.indicator_typed: dict = {}  # word_lower -> set of wordplay_type strings
        for w, wtype in conn.execute("SELECT word, wordplay_type FROM indicators"):
            if w:
                wl = w.lower()
                self.indicators.add(wl)
                if wl not in self.indicator_typed:
                    self.indicator_typed[wl] = set()
                if wtype:
                    self.indicator_typed[wl].add(wtype)
        print(f"    indicators: {len(self.indicators):,}")

        print("  Cache loaded.\n")

    def is_indicator(self, word: str) -> bool:
        return word.lower() in self.indicators

    def indicator_is_parts(self, word: str) -> bool:
        """Return True if word is a known 'parts' type indicator (first_use, last_use, etc.)."""
        return 'parts' in self.indicator_typed.get(word.lower(), set())

    def word_maps_to(self, word: str, letters: str) -> bool:
        """Check if word->letters already in DB (forward direction only)."""
        wl = word.lower()
        ll = letters.lower()
        lu = letters.upper()
        return (
            (wl, ll) in self.wordplay_pairs or
            (wl, lu.lower()) in self.wordplay_pairs or
            (wl, ll) in self.synonym_pairs
        )

    def def_known(self, definition: str, answer: str) -> bool:
        return (definition.lower(), answer.lower()) in self.def_answer_pairs

    def add_synonym(self, word: str, letters: str):
        self.synonym_pairs.add((word.lower(), letters.lower()))

    def add_wordplay(self, indicator: str, substitution: str):
        self.wordplay_pairs.add((indicator.lower(), substitution.lower()))

    def add_def(self, definition: str, answer: str):
        self.def_answer_pairs.add((definition.lower(), answer.lower()))


# ============================================================
# DATA LOADING
# ============================================================

def build_cross_reference_map(source: str, puzzle_number: str) -> dict:
    """Return {('22', 'across'): 'ANIMAL', ...} for the current puzzle."""
    db_path = PROJECT_ROOT / 'data' / 'clues_master.db'
    conn = sqlite3.connect(str(db_path))
    rows = conn.execute(
        "SELECT clue_number, direction, answer FROM clues WHERE source=? AND puzzle_number=?",
        (source, str(puzzle_number))
    ).fetchall()
    conn.close()
    return {(str(num), dir_.lower()): ans.upper().replace(' ', '')
            for num, dir_, ans in rows if num and dir_ and ans}


def resolve_cross_references(clue_text: str, cross_ref_map: dict) -> str:
    """Replace '22 Across', '14 Down' etc. with the actual answer from the puzzle."""
    def _replace(match):
        num = match.group(1)
        direction = match.group(2).lower()
        answer = cross_ref_map.get((num, direction))
        return answer if answer else match.group(0)
    return re.sub(r'\b(\d+)\s+(Across|Down)\b', _replace, clue_text)


def load_failures(pipeline_db: str, run_id: int = 0,
                  cross_ref_map: Optional[dict] = None) -> List[dict]:
    """Load unique stage_secondary failures (skip HTML entity duplicates)."""
    conn = sqlite3.connect(pipeline_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT clue_text, answer, letters_still_needed, unresolved_words,
                  word_roles, original_formula
           FROM stage_secondary
           WHERE run_id = ? AND fully_resolved = 0""",
        (run_id,)
    ).fetchall()
    conn.close()

    seen = set()
    failures = []
    for row in rows:
        clue_text = row['clue_text'] or ''
        if cross_ref_map:
            clue_text = resolve_cross_references(clue_text, cross_ref_map)
        answer = (row['answer'] or '').upper().replace(' ', '')

        # Skip HTML entity duplicates (keep plain-text version)
        if '&#' in clue_text:
            continue
        if answer in seen:
            continue
        seen.add(answer)

        failures.append({
            'clue_text': clue_text,
            'answer': answer,
            'letters_still_needed': (row['letters_still_needed'] or '').upper(),
            'unresolved_words': json.loads(row['unresolved_words'] or '[]'),
            'word_roles': json.loads(row['word_roles'] or '[]'),
            'original_formula': row['original_formula'] or '',
        })
    return failures


def load_anagram_hits(pipeline_db: str, run_id: int = 0,
                      cross_ref_map: Optional[dict] = None) -> List[dict]:
    """Load anagram stage rows where an exact answer was found but unused words remain.

    These are Pattern C candidates: all fodder letters account for the answer,
    but some clue words were not used in the anagram. Those unused words are
    the anagram indicator plus the definition phrase.
    """
    conn = sqlite3.connect(pipeline_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT clue_text, answer, unused_words, indicator_words, definition_words
           FROM stage_anagram
           WHERE run_id = ? AND hit_found = 1 AND unused_words != '[]'""",
        (run_id,)
    ).fetchall()
    conn.close()

    results = []
    for row in rows:
        answer = (row['answer'] or '').upper().replace(' ', '')
        if not answer:
            continue
        clue_text = row['clue_text'] or ''
        if cross_ref_map:
            clue_text = resolve_cross_references(clue_text, cross_ref_map)
        results.append({
            'clue_text': clue_text,
            'answer': answer,
            'unused_words': json.loads(row['unused_words'] or '[]'),
            'indicator_words': json.loads(row['indicator_words'] or '[]'),
            'definition_words': json.loads(row['definition_words'] or '[]'),
        })
    return results


def load_general_failures(pipeline_db: str, run_id: int = 0) -> List[dict]:
    """Load stage_general rows that were not solved and still have unresolved letters.

    These are the right rows for Patterns E and F: the pipeline has a partial
    formula but could not account for all answer letters from the clue words.
    """
    conn = sqlite3.connect(pipeline_db)
    conn.row_factory = sqlite3.Row
    rows = conn.execute(
        """SELECT clue_text, answer, letters_still_needed, unresolved_words,
                  word_roles, formula
           FROM stage_general
           WHERE run_id = ?
             AND letters_still_needed IS NOT NULL
             AND letters_still_needed != ''""",
        (run_id,)
    ).fetchall()
    conn.close()

    seen = set()
    results = []
    for row in rows:
        clue_text = row['clue_text'] or ''
        if '&#' in clue_text:
            continue
        answer = (row['answer'] or '').upper().replace(' ', '')
        if not answer or answer in seen:
            continue
        seen.add(answer)
        results.append({
            'clue_text': clue_text,
            'answer': answer,
            'letters_still_needed': (row['letters_still_needed'] or '').upper(),
            'unresolved_words': json.loads(row['unresolved_words'] or '[]'),
            'word_roles': json.loads(row['word_roles'] or '[]'),
            'formula': row['formula'] or '',
        })
    return results


# ============================================================
# ENRICHMENT STEPS
# ============================================================

def step_a_definition_pairs(failure: dict, conn: sqlite3.Connection,
                             cache: DBCache, counter: InsertCounter):
    """Insert definition-answer pairs from word_roles + heuristic windows."""
    answer = failure['answer']
    clue_text = failure['clue_text']
    word_roles = failure['word_roles']

    def maybe_insert(window: str, answer: str):
        window = window.lower().strip()
        if not window or window in STOP_WORDS:
            return
        if cache.def_known(window, answer):
            return
        inserted = insert_definition_answer(conn, window, answer, SOURCE_TAG)
        if inserted:
            cache.add_def(window, answer)
        counter.record('definition_answers_augmented', inserted, f"'{window}' -> {answer}")

    # Method 1: definition-tagged word roles
    for wr in word_roles:
        if not isinstance(wr, dict):
            continue
        if (wr.get('role') or '').lower() == 'definition':
            word = (wr.get('word') or '').strip()
            if word:
                maybe_insert(word, answer)

    # Method 2 (heuristic first/last N words) intentionally removed.
    # It inserted wordplay words as definitions without any audit, poisoning
    # the DB with entries like "alarm," → ALBERT and "at the front" → ALBERT.
    # Only Method 1 (definition-tagged word_roles from the pipeline) is used.


def _is_clean_word(word: str) -> bool:
    """Return True if word is a usable plain English word."""
    if not word:
        return False
    if '&#' in word:
        return False
    if not re.match(r'^[A-Za-z][A-Za-z\s\-]*$', word):
        return False
    if word.lower() in STOP_WORDS:
        return False
    return True


def step_b_single_word_inference(failure: dict, conn: sqlite3.Connection,
                                 cache: DBCache, counter: InsertCounter):
    """If exactly 1 unresolved word and letters_still_needed is its exact mapping."""
    unresolved = failure['unresolved_words']
    letters_needed = failure['letters_still_needed']
    answer = failure['answer']

    if len(unresolved) != 1 or not letters_needed:
        return

    word = unresolved[0].rstrip('.,;:!?\'"')
    if not _is_clean_word(word):
        return
    if cache.is_indicator(word):
        return  # It's an indicator, not a substitution target
    if cache.word_maps_to(word, letters_needed):
        return  # Already known

    n = len(letters_needed)

    # Guard: single-letter inferences must be initial-letter abbreviations.
    # Prevents false positives like european->Y (Y!=E).
    # Allows league->L, bells->B, etc.
    if n == 1 and letters_needed.upper() != word[0].upper():
        return

    if 1 <= n <= 4:
        inserted = insert_wordplay(conn, word.lower(), letters_needed.upper(),
                                   'abbreviation', confidence='low',
                                   notes=f'inferred from {answer}',
                                   source_tag=SOURCE_TAG)
        if inserted:
            cache.add_wordplay(word, letters_needed)
        counter.record('wordplay', inserted, f"{word} -> {letters_needed} (for {answer})")
    elif 5 <= n <= 12:
        inserted = insert_synonym_pair(conn, word.lower(), letters_needed.lower(),
                                       f'{SOURCE_TAG}:{answer}')
        if inserted:
            cache.add_synonym(word, letters_needed)
        counter.record('synonyms_pairs', inserted, f"{word} -> {letters_needed} (for {answer})")


def step_c_two_word_anchor_inference(failure: dict, conn: sqlite3.Connection,
                                      cache: DBCache, counter: InsertCounter):
    """
    For exactly 2 unresolved words: try splits of the full answer AND letters_still_needed.
    If one (word, fragment) pair is already in DB (anchor), insert the other.
    """
    unresolved = failure['unresolved_words']
    answer = failure['answer']
    letters_needed = failure['letters_still_needed']

    if len(unresolved) != 2:
        return

    word1, word2 = unresolved[0].rstrip('.,;:!?\'"'), unresolved[1].rstrip('.,;:!?\'"')
    if not _is_clean_word(word1) or not _is_clean_word(word2):
        return

    def try_insert(word: str, letters: str):
        """Insert word->letters if it's a valid new mapping."""
        if not _is_clean_word(word) or not letters:
            return
        if not re.match(r'^[A-Za-z]+$', letters):
            return
        n = len(letters)
        if n < 1 or n > 15:
            return
        # Sanity: inferred letters shouldn't be massively longer than the word
        # (synonyms are roughly same length; abbreviations are shorter)
        if n > len(word) * 2 + 2:
            return
        if cache.is_indicator(word):
            return
        if cache.word_maps_to(word, letters):
            return

        if 1 <= n <= 4:
            inserted = insert_wordplay(conn, word.lower(), letters.upper(),
                                       'abbreviation', confidence='low',
                                       notes=f'anchor-inferred from {answer}',
                                       source_tag=SOURCE_TAG)
            if inserted:
                cache.add_wordplay(word, letters)
            counter.record('wordplay', inserted, f"{word} -> {letters} (anchor:{answer})")
        elif 5 <= n <= 15:
            inserted = insert_synonym_pair(conn, word.lower(), letters.lower(),
                                           f'{SOURCE_TAG}:{answer}')
            if inserted:
                cache.add_synonym(word, letters)
            counter.record('synonyms_pairs', inserted, f"{word} -> {letters} (anchor:{answer})")

    # --- Approach B: try splits of letters_still_needed ---
    # Note: Approach A (full-answer splits) was removed because it generates false
    # positives when existing word_roles have incorrect substitutions (wrong anchors).
    # This handles cases where existing word_roles are correct.
    if letters_needed and len(letters_needed) >= 2:
        for split in range(1, len(letters_needed)):
            frag1 = letters_needed[:split]
            frag2 = letters_needed[split:]

            if cache.word_maps_to(word1, frag1):
                try_insert(word2, frag2)
            if cache.word_maps_to(word2, frag2):
                try_insert(word1, frag1)
            if cache.word_maps_to(word1, frag2):
                try_insert(word2, frag1)
            if cache.word_maps_to(word2, frag1):
                try_insert(word1, frag2)


def step_pattern_c_anagram_boundary(anagram_hit: dict, conn: sqlite3.Connection,
                                     cache: DBCache, counter: InsertCounter,
                                     candidates: list):
    """Pattern C: anagram exact match with unused surface words.

    After removing the confirmed indicator word(s) and link words from
    unused_words, whatever remains is the definition phrase.  Add it to
    definition_answers_augmented.

    High-confidence because the anagram stage has already identified the
    indicator separately (indicator_words column), so we can isolate the
    definition by subtraction.
    """
    answer = anagram_hit['answer']
    clue_text = anagram_hit['clue_text']
    unused_words = anagram_hit['unused_words']
    indicator_words = anagram_hit['indicator_words']

    if not unused_words:
        return

    # Normalise indicator set (strip punctuation for matching)
    indicator_norm = {re.sub(r'[^a-z]', '', w.lower()) for w in indicator_words if w}

    # Step 1: remove indicator words (any position in the list).
    after_indicator = []
    for w in unused_words:
        w_norm = re.sub(r'[^a-z]', '', w.lower())
        if not w_norm:
            continue
        if w_norm in indicator_norm:
            continue
        after_indicator.append(w)

    if not after_indicator:
        return

    # Step 2: trim leading LINKERS only (from the front of the list).
    while after_indicator:
        w_norm = re.sub(r'[^a-z]', '', after_indicator[0].lower())
        if w_norm in LINKERS:
            after_indicator.pop(0)
        else:
            break

    # Step 3: trim trailing LINKERS only (from the back of the list).
    while after_indicator:
        w_norm = re.sub(r'[^a-z]', '', after_indicator[-1].lower())
        if w_norm in LINKERS:
            after_indicator.pop()
        else:
            break

    if not after_indicator:
        return

    # Step 4: build phrase — lowercase, strip punctuation from each word,
    # preserving interior LINKERS so the phrase matches the clue text.
    def_candidates = [re.sub(r'[^a-z ]', '', w.lower()).strip() for w in after_indicator]
    def_candidates = [w for w in def_candidates if w]

    if not def_candidates:
        return

    phrase = ' '.join(def_candidates)
    if len(phrase) < 3:
        return

    if cache.def_known(phrase, answer) or cache.word_maps_to(phrase, answer):
        return

    candidates.append({
        'pattern': 'C',
        'type': 'definition_pair',
        'phrase': phrase,
        'answer': answer,
        'clue': clue_text,
        'confidence': 'high',
    })

    inserted = insert_definition_answer(conn, phrase, answer, SOURCE_TAG)
    if inserted:
        cache.add_def(phrase, answer)
    counter.record('definition_answers_augmented', inserted,
                   f"pattern-C: '{phrase}' -> {answer}")


def step_pattern_d_outer_letters(failure: dict, conn: sqlite3.Connection,
                                  cache: DBCache, counter: InsertCounter,
                                  candidates: list):
    """Pattern D: letters_still_needed is exactly 2 chars = first+last of an unresolved word.

    The remaining unresolved words (after removing the container word and link words)
    are proposed as an outer_use indicator.

    Example: STYE, needed='ST', unresolved=['Seaport','cleared','out','once,']
      -> 'Seaport' first=S last=T -> container
      -> 'cleared out' (non-linker remainder) -> outer_use indicator candidate
    """
    answer = failure['answer']
    clue_text = failure['clue_text']
    letters_needed = failure['letters_still_needed']
    unresolved = failure['unresolved_words']

    if len(letters_needed) != 2:
        return
    if len(unresolved) < 2:
        return

    first_needed = letters_needed[0].upper()
    last_needed = letters_needed[1].upper()

    # Find which unresolved word has outer letters = letters_still_needed
    container_idx = None
    container_word = None
    for i, w in enumerate(unresolved):
        w_clean = re.sub(r'[^a-zA-Z]', '', w)
        if len(w_clean) >= 2:
            if w_clean[0].upper() == first_needed and w_clean[-1].upper() == last_needed:
                container_idx = i
                container_word = w
                break

    if container_word is None:
        return

    # Remaining unresolved words (excluding container and link words) = indicator candidates
    indicator_words = []
    for i, w in enumerate(unresolved):
        if i == container_idx:
            continue
        w_norm = re.sub(r'[^a-z]', '', w.lower())
        if w_norm and w_norm not in LINKERS:
            indicator_words.append(w_norm)

    if not indicator_words:
        return

    indicator_phrase = ' '.join(indicator_words)

    if cache.is_indicator(indicator_phrase):
        return

    candidates.append({
        'pattern': 'D',
        'type': 'indicator',
        'phrase': indicator_phrase,
        'wordplay_type': 'parts',
        'subtype': 'outer_use',
        'container': container_word,
        'letters': letters_needed,
        'answer': answer,
        'clue': clue_text,
        'confidence': 'high',
    })

    inserted = insert_indicator(conn, indicator_phrase, 'parts', subtype='outer_use',
                                confidence='medium')
    if inserted:
        cache.indicators.add(indicator_phrase.lower())
    counter.record('indicators', inserted,
                   f"pattern-D: '{indicator_phrase}' (outer_use) for {answer}")


def step_pattern_e_first_letter(failure: dict, conn: sqlite3.Connection,
                                 cache: DBCache, counter: InsertCounter,
                                 candidates: list):
    """Pattern E: an unresolved word's first letter fills a missing answer letter,
    confirmed by a known 'parts' indicator adjacent to it in the clue.

    Example: ALBERT, needed='B', unresolved includes 'bells'
      - 'bells' is not an indicator; bells[0]='B' is in needed ✓
      - 'front' (parts/first_use) is within ±4 tokens of 'bells' in the clue ✓
      - Insert wordplay('bells', 'B')

    Self-certifying: first-letter extraction is mechanical, no API needed.
    """
    answer = failure['answer']
    clue_text = failure['clue_text']
    letters_needed = failure['letters_still_needed']
    unresolved = failure['unresolved_words']

    # Only fire when exactly one letter is still missing. With multiple letters
    # needed, we cannot determine which unresolved word provides which letter
    # — that ambiguity belongs to Pattern F (acrostic) instead.
    if len(letters_needed) != 1 or not unresolved:
        return

    clue_clean = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text)
    tokens = clue_clean.split()
    token_norms = [_norm(t) for t in tokens]

    for raw_word in unresolved:
        w_clean = re.sub(r'[^a-zA-Z]', '', raw_word)
        if len(w_clean) < 2:
            continue
        w_norm = w_clean.lower()

        if w_norm in LINKERS:
            continue
        if cache.is_indicator(w_norm):
            continue  # It's an indicator itself, not a source word

        first_letter = w_clean[0].upper()
        if first_letter != letters_needed:
            continue

        if cache.word_maps_to(w_norm, first_letter):
            continue  # Already in DB

        # Find position of this word in the clue token sequence
        pos = next((i for i, tn in enumerate(token_norms) if tn == w_norm), None)
        if pos is None:
            continue

        # Check for a known parts indicator within ±4 tokens
        window_start = max(0, pos - 4)
        window_end = min(len(token_norms), pos + 5)
        has_adjacent_parts_indicator = any(
            cache.indicator_is_parts(token_norms[i])
            for i in range(window_start, window_end)
            if i != pos and token_norms[i]
        )
        if not has_adjacent_parts_indicator:
            continue

        candidates.append({
            'pattern': 'E',
            'type': 'wordplay_substitution',
            'phrase': w_norm,
            'letters': first_letter,
            'answer': answer,
            'clue': clue_text,
            'confidence': 'high',
        })
        inserted = insert_wordplay(conn, w_norm, first_letter,
                                   'abbreviation', confidence='medium',
                                   notes=f'first-letter inferred for {answer}',
                                   source_tag=SOURCE_TAG)
        if inserted:
            cache.add_wordplay(w_norm, first_letter)
        counter.record('wordplay', inserted,
                       f"pattern-E: '{w_norm}' -> '{first_letter}' (for {answer})")


def step_pattern_f_acrostic_indicator(failure: dict, conn: sqlite3.Connection,
                                       cache: DBCache, counter: InsertCounter,
                                       candidates: list):
    """Pattern F: consecutive unresolved words whose first letters spell exactly
    letters_still_needed. The non-LINKER token(s) immediately preceding that run
    are the candidate acrostic indicator.

    Example: SCAM, needed='SCAM', clue contains 'starts to sound clearer after maintenance'
      - run [sound, clearer, after, maintenance] → first letters S,C,A,M = 'SCAM' ✓
      - Preceding non-LINKER before 'sound' (skipping LINKER 'to') = 'starts'
      - 'starts' + LINKER 'to' → candidate phrase 'starts to'
      - Not in indicators → propose as parts/first_use

    Self-certifying: exact letter-sequence match with no alternative explanation.
    Only fires for 2–6 letter targets (acrostics for longer answers are too noisy).
    """
    answer = failure['answer']
    clue_text = failure['clue_text']
    letters_needed = failure['letters_still_needed']
    unresolved = failure['unresolved_words']

    if not letters_needed or not (2 <= len(letters_needed) <= 6):
        return
    if not unresolved:
        return

    clue_clean = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text)
    tokens = clue_clean.split()
    token_norms = [_norm(t) for t in tokens]

    unresolved_norms = {_norm(w) for w in unresolved if _norm(w)}
    target = letters_needed.upper()
    n = len(target)

    for start_pos in range(len(tokens)):
        # Build a run of unresolved words starting at start_pos,
        # allowing LINKER tokens inside but not attributed non-LINKER words.
        letter_contributors = []  # first letters from unresolved words in order
        pos = start_pos
        run_end = start_pos

        while pos < len(tokens) and len(letter_contributors) < n:
            tn = token_norms[pos]
            if tn in unresolved_norms:
                first = tn[0].upper() if tn else ''
                if first:
                    letter_contributors.append(first)
                run_end = pos
            elif tn in LINKERS:
                pass  # LINKER tokens are allowed within the run
            else:
                break  # Non-LINKER attributed word breaks the run
            pos += 1

        if len(letter_contributors) != n:
            continue
        if ''.join(letter_contributors) != target:
            continue

        # Found a matching run. Now identify the indicator phrase.
        # Walk backwards from start_pos collecting LINKER tokens, then
        # take the next non-LINKER attributed token as the indicator root.
        trailing_linkers = []
        look_pos = start_pos - 1
        while look_pos >= 0 and token_norms[look_pos] in LINKERS:
            trailing_linkers.insert(0, token_norms[look_pos])
            look_pos -= 1

        # The token at look_pos should be a non-LINKER attributed (non-unresolved) word
        if look_pos < 0:
            continue
        root_norm = token_norms[look_pos]
        if not root_norm or root_norm in unresolved_norms:
            continue  # Not attributed — can't confirm it's an indicator

        # Build candidate phrase: root + any trailing LINKERS between it and the run
        indicator_tokens = [root_norm] + trailing_linkers
        indicator_phrase = ' '.join(indicator_tokens)

        if not indicator_phrase or len(indicator_phrase) < 3:
            continue
        if cache.is_indicator(indicator_phrase):
            continue  # Already in DB

        candidates.append({
            'pattern': 'F',
            'type': 'indicator',
            'phrase': indicator_phrase,
            'wordplay_type': 'parts',
            'subtype': 'first_use',
            'answer': answer,
            'clue': clue_text,
            'letters': letters_needed,
            'confidence': 'high',
        })
        inserted = insert_indicator(conn, indicator_phrase, 'parts',
                                    subtype='first_use', confidence='medium')
        if inserted:
            cache.indicators.add(indicator_phrase.lower())
        counter.record('indicators', inserted,
                       f"pattern-F: '{indicator_phrase}' (parts/first_use) for {answer}")
        break  # Take the first matching run only


def step_pattern_a_indicator_inference(failure: dict, conn: sqlite3.Connection,
                                       cache: DBCache, counter: InsertCounter,
                                       candidates: list):
    """Pattern A: infer indicator words when all letters are found but clue words are unresolved.

    When the pipeline accounts for all answer letters (letters_still_needed='') but
    some clue words have no assigned role, this step:
    1. Determines the formula operation type (anagram, insertion, container, etc.)
    2. Skips if an indicator of that type is already confirmed in word_roles.
    3. Groups contiguous unresolved words (LINKERs do not break groups).
    4. Classifies groups adjacent to the definition → definition extensions.
    5. Checks extended definition phrases against definition_answers AND synonyms_pairs.
    6. Proposes remaining groups as indicator candidates of the inferred type.
    """
    answer = failure['answer']
    clue_text = failure['clue_text']
    letters_needed = failure['letters_still_needed']
    unresolved = failure['unresolved_words']
    word_roles = failure['word_roles']
    formula = failure['original_formula']

    # Only fire when all letters are found but unresolved words remain
    if letters_needed or not unresolved:
        return

    # Step 1: Determine formula type
    formula_type = _infer_formula_type(formula, word_roles)
    if formula_type is None:
        return

    # Step 2: Track whether indicator already exists.
    # Only skip indicator proposals (step 6) — NOT definition extension (steps 3-5).
    # Unresolved words adjacent to the definition must still be proposed as candidates
    # even when the indicator is already confirmed.
    skip_indicator_proposals = _has_indicator_for_type(word_roles, formula_type)

    # Step 3: Build contiguous groups of unresolved words
    groups = _build_word_groups(clue_text, unresolved, word_roles)
    if not groups:
        return

    # Step 4: Separate definition-extension groups from indicator groups
    def_groups, indicator_groups = _infer_definition_groups(groups, clue_text, word_roles)

    # Step 5: Check definition extensions — propose extended definition pairs if uncovered
    if def_groups:
        clue_clean = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text)
        tokens = clue_clean.split()
        def_words_norm = {_norm(wr.get('word', ''))
                          for wr in word_roles if wr.get('role') == 'definition'}
        def_positions = sorted(
            i for i, t in enumerate(tokens) if _norm(t) in def_words_norm
        )
        for group in def_groups:
            group_positions = [pos for _, pos in group]
            all_positions = sorted(group_positions + def_positions)
            if not all_positions:
                continue
            span_start, span_end = min(all_positions), max(all_positions)
            group_pos_set = set(group_positions)
            def_pos_set = set(def_positions)
            phrase_tokens = []
            for i in range(span_start, span_end + 1):
                if i >= len(tokens):
                    break
                tn = _norm(tokens[i])
                if i in group_pos_set or i in def_pos_set or tn in LINKERS:
                    cleaned = re.sub(r'[^a-z ]', '', tokens[i].lower()).strip()
                    if cleaned:
                        phrase_tokens.append(cleaned)
            phrase = ' '.join(phrase_tokens)
            if not phrase or len(phrase) < 3:
                continue
            # Check both tables before proposing
            if cache.def_known(phrase, answer) or cache.word_maps_to(phrase, answer):
                continue
            candidates.append({
                'pattern': 'A',
                'type': 'definition_pair',
                'phrase': phrase,
                'answer': answer,
                'clue': clue_text,
                'confidence': 'medium',
            })
            inserted = insert_definition_answer(conn, phrase, answer, SOURCE_TAG)
            if inserted:
                cache.add_def(phrase, answer)
            counter.record('definition_answers_augmented', inserted,
                           f"pattern-A: '{phrase}' -> {answer}")

    # Step 6: Propose indicator candidates from non-definition groups
    # (skipped if a confirmed indicator of this type already exists in word_roles)
    if skip_indicator_proposals:
        return
    wordplay_type = formula_type
    for group in indicator_groups:
        phrase = ' '.join(_norm(w) for w, _ in group if _norm(w))
        if not phrase or len(phrase) < 2:
            continue
        if cache.is_indicator(phrase):
            continue  # Already in DB
        candidates.append({
            'pattern': 'A',
            'type': 'indicator',
            'phrase': phrase,
            'wordplay_type': wordplay_type,
            'subtype': None,
            'answer': answer,
            'clue': clue_text,
            'formula': formula,
            'confidence': 'medium',
        })
        inserted = insert_indicator(conn, phrase, wordplay_type,
                                    subtype=None, confidence='medium')
        if inserted:
            cache.indicators.add(phrase.lower())
        counter.record('indicators', inserted,
                       f"pattern-A: '{phrase}' ({wordplay_type}) for {answer}")


def write_candidates_file(candidates: list, output_path: Path):
    """Write structured candidate list to file for Phase 3 API audit.

    One JSON object per line.  Each record has enough context for the
    audit prompt to be constructed without re-querying the DB.
    """
    if not candidates:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open('w', encoding='utf-8') as f:
        for c in candidates:
            f.write(json.dumps(c, ensure_ascii=False) + '\n')
    print(f"\nCandidates written to: {output_path}  ({len(candidates)} records)")


# ============================================================
# PIPELINE RE-RUN
# ============================================================

def rerun_pipeline(source: str, puzzle_number: str) -> Optional[str]:
    """Re-run report.py, return the 'Final: X/Y solved' line."""
    cmd = [sys.executable, str(PROJECT_ROOT / 'report.py'),
           '--source', source, '--puzzle-number', str(puzzle_number)]
    print(f"\nRe-running pipeline: report.py --source {source} --puzzle-number {puzzle_number}")
    result = subprocess.run(cmd, capture_output=True, text=True,
                            cwd=str(PROJECT_ROOT), timeout=300)
    for line in (result.stdout + result.stderr).splitlines():
        if 'Final:' in line and 'solved' in line:
            return line.strip()
    return None


# ============================================================
# MAIN
# ============================================================

def main():
    parser = argparse.ArgumentParser(description='Self-learning enrichment from pipeline failures')
    parser.add_argument('--source', default='telegraph')
    parser.add_argument('--puzzle-number', default='31164')
    parser.add_argument('--run-id', type=int, default=0)
    parser.add_argument('--no-rerun', action='store_true',
                        help='Skip re-running the pipeline after enrichment')
    add_common_args(parser)
    args = parser.parse_args()
    apply_common_args(args)

    pipeline_db = str(PROJECT_ROOT / 'pipeline_stages.db')
    cross_ref_map = build_cross_reference_map(args.source, args.puzzle_number)
    failures = load_failures(pipeline_db, args.run_id, cross_ref_map=cross_ref_map)
    anagram_hits = load_anagram_hits(pipeline_db, args.run_id, cross_ref_map=cross_ref_map)
    general_failures = load_general_failures(pipeline_db, args.run_id)

    print(f"\n{'=' * 60}")
    print(f"SELF-LEARNING ENRICHMENT")
    print(f"{'=' * 60}")
    print(f"Source: {args.source} #{args.puzzle_number}  run_id={args.run_id}")
    print(f"Unique failures to process: {len(failures)}")
    print(f"Anagram hits with unused words: {len(anagram_hits)}")
    print(f"General failures with letters_still_needed: {len(general_failures)}")

    if not failures and not anagram_hits and not general_failures:
        print("No failures or anagram hits found. Run the pipeline first (report.py).")
        return

    # Baseline from report
    report_file = PROJECT_ROOT / 'documents' / 'puzzle_report.txt'
    before_summary = None
    if report_file.exists():
        for line in report_file.read_text(encoding='utf-8', errors='replace').splitlines():
            if 'SOLVED:' in line:
                before_summary = line.strip()
                break
    print(f"Baseline: {before_summary or '(unknown)'}\n")

    conn = get_cryptic_conn()
    cache = DBCache(conn)
    counter = InsertCounter("05_self_learning_enrichment")
    candidates = []  # Accumulates structured records for the API audit file

    # --- Pattern C: anagram boundary-group definitions ---
    print(f"--- Pattern C: anagram boundary-group definitions ---")
    for hit in anagram_hits:
        answer = hit['answer']
        unused = hit['unused_words']
        indicators = hit['indicator_words']
        print(f"  {answer}: unused={unused}, indicators={indicators}")
        step_pattern_c_anagram_boundary(hit, conn, cache, counter, candidates)

    # --- Existing steps on stage_secondary failures ---
    print(f"\n--- Stage secondary failures ---")
    for failure in failures:
        answer = failure['answer']
        unresolved = failure['unresolved_words']
        needed = failure['letters_still_needed']
        print(f"  {answer}: unresolved={unresolved}, needed={needed!r}")

        step_a_definition_pairs(failure, conn, cache, counter)
        step_b_single_word_inference(failure, conn, cache, counter)
        step_c_two_word_anchor_inference(failure, conn, cache, counter)
        step_pattern_d_outer_letters(failure, conn, cache, counter, candidates)
        step_pattern_a_indicator_inference(failure, conn, cache, counter, candidates)

    # --- Patterns E and F: first-letter substitution and acrostic indicator ---
    # These run on stage_general rows where letters_still_needed != '',
    # which is the right signal before secondary processing obscures the gap.
    print(f"\n--- Stage general failures (Patterns E and F) ---")
    for failure in general_failures:
        answer = failure['answer']
        unresolved = failure['unresolved_words']
        needed = failure['letters_still_needed']
        print(f"  {answer}: unresolved={unresolved}, needed={needed!r}")

        step_pattern_e_first_letter(failure, conn, cache, counter, candidates)
        step_pattern_f_acrostic_indicator(failure, conn, cache, counter, candidates)

    conn.commit()
    conn.close()

    counter.report()

    # Write candidates file (always, so dry-run is inspectable)
    candidates_path = PROJECT_ROOT / 'documents' / 'self_learning_candidates.txt'
    write_candidates_file(candidates, candidates_path)

    # Audit and apply candidates automatically when not in dry-run mode
    import enrichment.common as _common
    if not _common.DRY_RUN and candidates:
        audit_path = PROJECT_ROOT / 'documents' / 'self_learning_audit.txt'
        apply_log_path = PROJECT_ROOT / 'documents' / 'self_learning_inserts.txt'

        print(f"\n--- Auto-auditing {len(candidates)} candidates with Sonnet ---")
        from enrichment.audit_candidates import audit_candidates as run_audit
        run_audit(candidates_path=candidates_path, output_path=audit_path, verbose=False)

        print(f"\n--- Auto-applying approved candidates ---")
        from enrichment.apply_candidates import apply_entry
        from datetime import datetime

        audit_lines = [l.strip() for l in audit_path.read_text(encoding='utf-8').splitlines() if l.strip()]
        audited = []
        for line in audit_lines:
            try:
                audited.append(json.loads(line))
            except json.JSONDecodeError:
                pass

        conn2 = get_cryptic_conn()
        counter2 = InsertCounter("apply_approved")
        log_lines = [
            f"# self_learning_inserts.txt — {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"# Rollback: run the DELETE statements below against cryptic_new.db",
            "",
        ]
        for entry in audited:
            if entry.get('verdict') == 'YES':
                phrase = entry.get('phrase', '')
                answer = entry.get('answer', '')
                ctype = entry.get('type', '')
                print(f"  Applying: {ctype} '{phrase}' -> {answer}")
                apply_entry(entry, conn2, counter2, log_lines)
        conn2.commit()
        conn2.close()
        counter2.report()
        apply_log_path.write_text('\n'.join(log_lines) + '\n', encoding='utf-8')
        print(f"Insert log: {apply_log_path}")

    # Re-run pipeline to measure improvement (not in dry-run mode)
    if not args.no_rerun and not _common.DRY_RUN:
        try:
            after_line = rerun_pipeline(args.source, args.puzzle_number)
            print(f"Before: {before_summary or '(unknown)'}")
            print(f"After:  {after_line or '(unknown)'}")
        except subprocess.TimeoutExpired:
            print("Pipeline re-run timed out.")
        except Exception as e:
            print(f"Pipeline re-run failed: {e}")


if __name__ == '__main__':
    main()
