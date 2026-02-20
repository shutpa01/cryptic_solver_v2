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

        # indicators: set of word_lower
        self.indicators: Set[str] = set()
        for (w,) in conn.execute("SELECT word FROM indicators"):
            if w:
                self.indicators.add(w.lower())
        print(f"    indicators: {len(self.indicators):,}")

        print("  Cache loaded.\n")

    def is_indicator(self, word: str) -> bool:
        return word.lower() in self.indicators

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

def load_failures(pipeline_db: str, run_id: int = 0) -> List[dict]:
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


def load_anagram_hits(pipeline_db: str, run_id: int = 0) -> List[dict]:
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
        results.append({
            'clue_text': row['clue_text'] or '',
            'answer': answer,
            'unused_words': json.loads(row['unused_words'] or '[]'),
            'indicator_words': json.loads(row['indicator_words'] or '[]'),
            'definition_words': json.loads(row['definition_words'] or '[]'),
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

    # Method 2: heuristic first/last N words of clue
    words = re.sub(r'\s*\([0-9,\-\s]+\)\s*$', '', clue_text).split()
    for n in (1, 2, 3):
        if n > len(words):
            continue
        maybe_insert(' '.join(words[:n]), answer)
        maybe_insert(' '.join(words[-n:]), answer)


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

    word = unresolved[0]
    if not _is_clean_word(word):
        return
    if cache.is_indicator(word):
        return  # It's an indicator, not a substitution target
    if cache.word_maps_to(word, letters_needed):
        return  # Already known

    n = len(letters_needed)
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

    word1, word2 = unresolved
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

    # Step 2: Skip if a confirmed indicator of this type already exists
    if _has_indicator_for_type(word_roles, formula_type):
        return

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
    failures = load_failures(pipeline_db, args.run_id)
    anagram_hits = load_anagram_hits(pipeline_db, args.run_id)

    print(f"\n{'=' * 60}")
    print(f"SELF-LEARNING ENRICHMENT")
    print(f"{'=' * 60}")
    print(f"Source: {args.source} #{args.puzzle_number}  run_id={args.run_id}")
    print(f"Unique failures to process: {len(failures)}")
    print(f"Anagram hits with unused words: {len(anagram_hits)}")

    if not failures and not anagram_hits:
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

    conn.commit()
    conn.close()

    counter.report()

    # Write candidates file for Phase 3 API audit (always, even in dry-run)
    candidates_path = PROJECT_ROOT / 'documents' / 'self_learning_candidates.txt'
    write_candidates_file(candidates, candidates_path)

    # Re-run pipeline to measure improvement (not in dry-run mode)
    import enrichment.common as _common
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
