#!/usr/bin/env python3
"""
Compound Wordplay Analyzer Engine - DATABASE-INTEGRATED VERSION

This engine:
1. Receives evidence-enhanced cases from evidence analysis
2. Queries the indicators table to identify wordplay types for remaining words
3. Queries the wordplay table for substitutions
4. Applies operation solvers (insertion, container, deletion, reversal, etc.)
5. Builds complete explanations with formula notation

Database tables used:
- indicators: word -> wordplay_type, subtype, confidence
- wordplay: indicator -> substitution, category
- synonyms_pairs: word -> synonym (fallback)
"""

import sys
import sqlite3
import re
from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass
from enum import Enum

# Add project root to path

from resources import norm_letters

DB_PATH = r'C:\Users\shute\PycharmProjects\cryptic_solver_V2\data\cryptic_new.db'


def format_answer_with_enumeration(answer: str, enumeration: str) -> str:
    """
    Format answer with spaces according to enumeration.

    Example: format_answer_with_enumeration("PATROLCAR", "(6,3)") → "PATROL CAR"
    Example: format_answer_with_enumeration("INSPITEOFTHAT", "(2,5,2,4)") → "IN SPITE OF THAT"
    """
    if not answer or not enumeration:
        return answer

    # Extract numbers from enumeration like (6,3) or (2,4,3)
    parts = re.findall(r'\d+', enumeration)

    if not parts or len(parts) == 1:
        return answer  # Single word, no formatting needed

    # Remove any existing spaces from answer for clean formatting
    clean_answer = answer.replace(' ', '')

    # Verify total length matches
    expected_len = sum(int(p) for p in parts)
    if len(clean_answer) != expected_len:
        return answer  # Length mismatch, return as-is

    # Insert spaces at the right positions
    result = []
    pos = 0
    for length in parts:
        length = int(length)
        result.append(clean_answer[pos:pos + length])
        pos += length

    return ' '.join(result)


class WordplayType(Enum):
    """Wordplay types from indicators table."""
    ANAGRAM = 'anagram'
    CONTAINER = 'container'
    INSERTION = 'insertion'
    DELETION = 'deletion'
    REVERSAL = 'reversal'
    HIDDEN = 'hidden'
    HOMOPHONE = 'homophone'
    PARTS = 'parts'
    ACROSTIC = 'acrostic'
    SELECTION = 'selection'
    UNKNOWN = 'unknown'


@dataclass
class IndicatorMatch:
    """Result of looking up a word in the indicators table."""
    word: str
    wordplay_type: str
    subtype: Optional[str]
    confidence: str
    frequency: int = 0  # Empirical frequency from corpus analysis


@dataclass
class SubstitutionMatch:
    """Result of looking up a word in the wordplay table."""
    word: str
    letters: str
    category: str
    notes: Optional[str] = None


@dataclass
class WordRole:
    """Role of a word in the clue."""
    word: str
    role: str  # definition, fodder, indicator, substitution, positional, link, etc.
    contributes: str  # What letters/meaning it contributes
    source: str  # Where we determined this (evidence, database, heuristic)


class DatabaseLookup:
    """Handles all database queries for indicators and substitutions."""

    def __init__(self, db_path: str = DB_PATH):
        self.db_path = db_path
        self._conn = None
        self._indicator_cache = {}
        self._substitution_cache = {}

    def _get_connection(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def lookup_indicator(self, word: str) -> Optional[IndicatorMatch]:
        """Look up a word in the indicators table.

        Returns the match with highest frequency if multiple matches exist.
        """
        # Strip punctuation but preserve spaces for two-word indicators
        word_clean = ''.join(c for c in word.lower() if c.isalpha() or c == ' ')
        word_clean = ' '.join(word_clean.split())  # Normalize whitespace

        if not word_clean:
            # Numeric tokens (e.g., "1") — look up directly
            word_clean = ''.join(c for c in word if c.isdigit())
            if not word_clean:
                return None

        if word_clean in self._indicator_cache:
            return self._indicator_cache[word_clean]

        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT word, wordplay_type, subtype, confidence, COALESCE(frequency, 0) as freq
            FROM indicators
            WHERE LOWER(word) = ?
            ORDER BY freq DESC
        """, (word_clean,))

        result = cursor.fetchone()

        if result:
            match = IndicatorMatch(
                word=result[0],
                wordplay_type=result[1],
                subtype=result[2],
                confidence=result[3],
                frequency=result[4] or 0
            )
            self._indicator_cache[word_clean] = match
            return match

        self._indicator_cache[word_clean] = None
        return None

    def lookup_substitution(self, word: str, max_synonym_length: int = 8) -> List[
        SubstitutionMatch]:
        """Look up a word in the wordplay table for substitutions.

        Also checks synonyms_pairs for synonyms that could be letter contributions.
        """
        # Strip punctuation for lookup
        word_clean = ''.join(c for c in word.lower() if c.isalpha())

        if not word_clean:
            # Numeric tokens (e.g., "1", "50", "100") — look up directly
            word_clean = ''.join(c for c in word if c.isdigit())
            if not word_clean:
                return []

        if word_clean in self._substitution_cache:
            all_matches = self._substitution_cache[word_clean]
            return [m for m in all_matches if len(m.letters) <= max_synonym_length
                    or m.category != 'synonym']

        conn = self._get_connection()
        cursor = conn.cursor()

        # First check wordplay table (no length filter — these are abbreviations/mappings)
        cursor.execute("""
            SELECT indicator, substitution, category, notes
            FROM wordplay
            WHERE LOWER(indicator) = ?
        """, (word_clean,))

        results = cursor.fetchall()

        matches = [
            SubstitutionMatch(
                word=r[0],
                letters=r[1],
                category=r[2],
                notes=r[3]
            )
            for r in results
        ]

        # Also check synonyms_pairs — fetch ALL synonyms, filter by length on return
        cursor.execute("""
            SELECT synonym FROM synonyms_pairs
            WHERE LOWER(word) = ?
        """, (word_clean,))

        synonym_results = cursor.fetchall()

        for r in synonym_results:
            synonym = r[0]
            # Only add if not already covered by wordplay table
            if not any(m.letters.upper() == synonym.upper() for m in matches):
                matches.append(SubstitutionMatch(
                    word=word_clean,
                    letters=synonym.upper(),
                    category='synonym',
                    notes='from synonyms_pairs'
                ))

        # Also try plural/singular variations for synonyms
        # Rule: plural query returns plural, singular query returns singular
        if word_clean.endswith('s') and len(word_clean) >= 4:
            # Word is plural - try singular form in DB and pluralize results
            singular = word_clean[:-1]
            cursor.execute("""
                SELECT synonym FROM synonyms_pairs
                WHERE LOWER(word) = ?
            """, (singular,))

            for r in cursor.fetchall():
                synonym = r[0]
                # Pluralize the result
                pluralized = synonym.upper() + 'S'
                if not any(m.letters == pluralized for m in matches):
                    matches.append(SubstitutionMatch(
                        word=word_clean,
                        letters=pluralized,
                        category='synonym',
                        notes=f'plural from {singular} → {synonym}'
                    ))
        else:
            # Word is singular - try plural form in DB and singularize results
            plural = word_clean + 's'
            cursor.execute("""
                SELECT synonym FROM synonyms_pairs
                WHERE LOWER(word) = ?
            """, (plural,))

            for r in cursor.fetchall():
                synonym = r[0]
                # Singularize the result if it ends in 's'
                if synonym.endswith('s') and len(synonym) >= 2:
                    singular_result = synonym[:-1].upper()
                    if not any(m.letters == singular_result for m in matches):
                        matches.append(SubstitutionMatch(
                            word=word_clean,
                            letters=singular_result,
                            category='synonym',
                            notes=f'singular from {plural} → {synonym}'
                        ))

        self._substitution_cache[word_clean] = matches
        # Return filtered by max_synonym_length (wordplay entries always pass)
        return [m for m in matches if len(m.letters) <= max_synonym_length
                or m.category != 'synonym']

    def lookup_phrase_substitution(self, phrase: str, max_synonym_length: int = 8) -> \
            List[SubstitutionMatch]:
        """Look up a multi-word phrase in the wordplay table and synonyms_pairs for substitutions.

        Handles phrases like "the french" -> LA, "group of animals" -> HERD, etc.

        Args:
            phrase: A multi-word phrase (e.g., "the french")

        Returns:
            List of SubstitutionMatch objects for any matches found
        """
        phrase_clean = ' '.join(
            ''.join(c for c in word if c.isalpha()) for word in phrase.lower().split())

        if not phrase_clean:
            return []

        # Check cache first
        if phrase_clean in self._substitution_cache:
            return self._substitution_cache[phrase_clean]

        conn = self._get_connection()
        cursor = conn.cursor()

        # First check wordplay table
        cursor.execute("""
            SELECT indicator, substitution, category, notes
            FROM wordplay
            WHERE LOWER(indicator) = ?
        """, (phrase_clean,))

        results = cursor.fetchall()

        matches = [
            SubstitutionMatch(
                word=r[0],
                letters=r[1],
                category=r[2],
                notes=r[3]
            )
            for r in results
        ]

        # Also check synonyms_pairs for short synonyms
        cursor.execute("""
            SELECT synonym FROM synonyms_pairs
            WHERE LOWER(word) = ? AND LENGTH(synonym) <= ?
        """, (phrase_clean, max_synonym_length))

        synonym_results = cursor.fetchall()

        for r in synonym_results:
            synonym = r[0]
            if not any(m.letters.upper() == synonym.upper() for m in matches):
                matches.append(SubstitutionMatch(
                    word=phrase_clean,
                    letters=synonym.upper(),
                    category='synonym',
                    notes='from synonyms_pairs'
                ))

        self._substitution_cache[phrase_clean] = matches
        return matches

    def lookup_synonym_as_substitution(self, word: str, max_length: int = 3) -> List[
        Tuple[str, str]]:
        """
        Look up synonyms that could be short substitutions.
        Returns list of (synonym, source) tuples.
        """
        conn = self._get_connection()
        cursor = conn.cursor()

        cursor.execute("""
            SELECT synonym FROM synonyms_pairs
            WHERE LOWER(word) = ? AND LENGTH(synonym) <= ?
        """, (word.lower(), max_length))

        return [(r[0], 'synonym') for r in cursor.fetchall()]

    def lookup_homophone(self, word: str) -> List[str]:
        """
        Look up homophones for a word.
        Returns list of words that sound like the input word.

        Example: lookup_homophone('acts') -> ['ax']
                 lookup_homophone('able') -> ['abel']
        """
        word_clean = ''.join(c for c in word.lower() if c.isalpha())

        if not word_clean:
            return []

        conn = self._get_connection()
        cursor = conn.cursor()

        # Look up the word in the homophones table
        cursor.execute("""
            SELECT homophone FROM homophones
            WHERE LOWER(word) = ?
        """, (word_clean,))

        results = cursor.fetchall()
        return [r[0] for r in results]


class CompoundSolver:
    """
    Solves compound wordplay by identifying and applying operations.
    """

    def __init__(self, db_lookup: DatabaseLookup):
        self.db = db_lookup

    def find_substitution_for_letters(self, word: str, needed_letters: str,
                                      used_letters: Set[str]) -> Optional[
        SubstitutionMatch]:
        """
        Find a substitution that provides exactly the needed letters.
        Returns the first valid match.
        """
        matches = self.db.lookup_substitution(word,
                                              max_synonym_length=len(needed_letters))

        for match in matches:
            if match.letters.upper() == needed_letters.upper():
                # Check letters aren't already used
                if not any(c in used_letters for c in match.letters.upper()):
                    return match

        return None

    def solve_insertion(self, anagram_letters: str, extra_letters: str,
                        answer: str) -> Optional[Dict[str, Any]]:
        """
        Solve insertion: extra_letters go INTO anagram_letters to form answer.

        Example: ABUNDANT = anagram(TUNABAN) with D inserted
        """
        answer_upper = answer.upper().replace(' ', '')
        anagram_upper = anagram_letters.upper()
        extra_upper = extra_letters.upper()

        # Try inserting extra_letters at each position in anagram
        for i in range(len(anagram_upper) + 1):
            combined = anagram_upper[:i] + extra_upper + anagram_upper[i:]
            if sorted(combined) == sorted(answer_upper):
                return {
                    'operation': 'insertion',
                    'base': anagram_letters,
                    'inserted': extra_letters,
                    'position': i,
                    'result': answer
                }

        return None

    def solve_container(self, outer_letters: str, inner_letters: str,
                        answer: str) -> Optional[Dict[str, Any]]:
        """
        Solve container: outer_letters wrap AROUND inner_letters.

        Example: C + ART + ON = CARTON (ART inside CON)
        """
        answer_upper = answer.upper().replace(' ', '')

        # Try each split of outer as prefix + suffix around inner
        outer_upper = outer_letters.upper()
        inner_upper = inner_letters.upper()

        for i in range(len(outer_upper) + 1):
            prefix = outer_upper[:i]
            suffix = outer_upper[i:]
            combined = prefix + inner_upper + suffix
            if combined == answer_upper:
                return {
                    'operation': 'container',
                    'outer': outer_letters,
                    'inner': inner_letters,
                    'prefix': prefix,
                    'suffix': suffix,
                    'result': answer
                }

        return None

    def solve_deletion(self, base_letters: str, delete_letters: str,
                       answer: str) -> Optional[Dict[str, Any]]:
        """
        Solve deletion: remove delete_letters from base_letters.
        """
        base_upper = base_letters.upper()
        delete_upper = delete_letters.upper()
        answer_upper = answer.upper().replace(' ', '')

        # Try removing the delete letters
        remaining = base_upper
        for c in delete_upper:
            idx = remaining.find(c)
            if idx >= 0:
                remaining = remaining[:idx] + remaining[idx + 1:]

        if sorted(remaining) == sorted(answer_upper):
            return {
                'operation': 'deletion',
                'base': base_letters,
                'deleted': delete_letters,
                'result': answer
            }

        return None

    def solve_reversal(self, letters: str, answer: str) -> Optional[Dict[str, Any]]:
        """
        Solve reversal: reversed letters form the answer.
        """
        if letters.upper()[::-1] == answer.upper().replace(' ', ''):
            return {
                'operation': 'reversal',
                'original': letters,
                'result': answer
            }
        return None


class CompoundWordplayAnalyzer:
    """
    Main analyzer that integrates evidence analysis with database lookups
    to build complete explanations.
    """

    def __init__(self, db_path: str = DB_PATH):
        self.db = DatabaseLookup(db_path)
        self.solver = CompoundSolver(self.db)

        # Link words that don't contribute letters
        # Expanded based on analysis of unresolved words
        self.link_words = {
            # Articles and prepositions
            'of', 'in', 'the', 'a', 'an', 'to', 'for', 'with',
            'and', 'or', 'by', 'from', 'as', 'on', 'at',
            # Common verbs used as links
            'is', 'are', 'be', 'being', 'been',
            'has', 'have', 'having', 'had',
            'was', 'were', 'will', 'would',
            'could', 'should', 'must', 'may', 'might',
            'gets', 'get', 'getting', 'got',
            'needs', 'need', 'needs',
            'makes', 'make', 'making', 'made',
            'gives', 'give', 'given', 'giving',
            'sees', 'see', 'seen', 'seeing',
            'brings', 'bring', 'bringing', 'brought',
            # Contractions (with apostrophe)
            "it's", "that's", "there's", "here's", "what's",
            "i'm", "i've", "i'd", "you're", "you've", "you'd",
            "he's", "she's", "we're", "we've", "they're", "they've",
            "don't", "doesn't", "didn't", "won't", "wouldn't",
            "can't", "couldn't", "shouldn't", "isn't", "aren't",
            # Contractions (apostrophe-stripped - for norm_letters matching)
            'its', 'thats', 'theres', 'heres', 'whats',
            'im', 'ive', 'id', 'youre', 'youve', 'youd',
            'hes', 'shes', 'were', 'weve', 'theyre', 'theyve',
            'dont', 'doesnt', 'didnt', 'wont', 'wouldnt',
            'cant', 'couldnt', 'shouldnt', 'isnt', 'arent',
            # Conjunctions and connectors
            'but', 'that', 'which', 'when', 'where', 'while',
            'so', 'yet', 'thus', 'hence', 'therefore',
            # Other common links
            'this', 'these', 'those', 'such',
            'one', 'ones', 'some', 'any', 'all',
            'here', 'there', 'maybe',
            # Common link phrases often appearing
            'into', 'onto', 'within', 'without',
            'find', 'found', 'finding', 'show', 'showing',
            'put', 'set', 'provide', 'providing',
            'if', 'how', 'why', 'who', 'whom',
            # Words that connect fodder to definition
            'giving', 'producing', 'causing', 'creating',
            'offering', 'providing', 'yielding', 'making',
        }


    def close(self):
        self.db.close()

    def _is_enumeration(self, word: str) -> bool:
        """Check if a word is an enumeration pattern like '8', '2,5', '3-4', '2,3,4'."""
        # Remove common punctuation
        cleaned = word.strip('()[]')

        # Pure digits
        if cleaned.isdigit():
            return True

        # Comma or hyphen separated digits: "2,5" or "3-4" or "2,3,4"
        if all(c.isdigit() or c in ',.-' for c in cleaned) and any(
                c.isdigit() for c in cleaned):
            return True

        return False

    def analyze_case(self, case: Dict[str, Any]) -> Dict[str, Any]:
        """
        Analyze a single case with evidence data and build complete attribution.

        IMPORTANT: We use the MATCHED CANDIDATE from brute force as our "likely answer",
        NOT the database answer. The database answer is only for comparison/verification.
        """

        clue_text = case.get('clue', '')
        clue_words = clue_text.replace('(', ' ').replace(')', ' ').split()

        # Filter out enumeration patterns (digits, comma-separated numbers)
        clue_words = [w for w in clue_words if not self._is_enumeration(w)]

        # Database answer - ONLY for display/comparison, never for solving
        db_answer = case.get('answer', '').upper().replace(' ', '')

        # PRIORITY: Check if anagram stage found an EXACT match (brute force)
        # This is more reliable than evidence_analysis which can produce garbage
        anagrams = case.get('anagrams', [])
        exact_match = None
        for hit in anagrams:
            if hit.get('solve_type') == 'anagram_exact':
                # Verify this hit matches the db_answer
                if norm_letters(hit.get('answer', '')) == norm_letters(db_answer):
                    exact_match = hit
                    break

        if exact_match:
            # Brute force found exact match - use this data directly
            likely_answer = exact_match.get('answer', '').upper().replace(' ', '')
            fodder_words_original = exact_match.get('fodder_words', [])
            fodder_letters_original = exact_match.get('fodder_letters', '')

            # Get ALL valid definition windows and try each
            all_windows = self._get_all_valid_definition_windows(case, clue_words)
            if not all_windows:
                all_windows = [None]  # Try with no definition window as fallback

            best_result = None
            best_score = -1  # Higher is better

            for definition_window in all_windows:
                # Reset fodder for each attempt
                fodder_words = list(fodder_words_original)
                fodder_letters = fodder_letters_original

                # Find anagram indicator using our method (not evidence garbage)
                anagram_indicator = self._find_anagram_indicator(
                    clue_words, fodder_words, definition_window
                )

                # Check if any fodder words should be reassigned as insertion material
                # (e.g., "Me" in "Me sporting tailored sateen" should be insertion, not fodder)
                insertion_material = []
                if fodder_words and anagram_indicator:
                    fodder_words, insertion_material = self._reassign_fodder_near_insertion_indicators(
                        fodder_words, clue_words, anagram_indicator
                    )
                    # Recalculate fodder_letters if we removed any words
                    if insertion_material:
                        fodder_letters = ''.join(
                            c.upper() for w in fodder_words for c in w if c.isalpha()
                        )

                # Build word roles for exact match
                word_roles = []
                accounted_words = set()

                # Account for indicator first - add each word separately for breakdown lookup
                if anagram_indicator:
                    for ind_word in anagram_indicator.split():
                        word_roles.append(
                            WordRole(ind_word, 'anagram_indicator', '', 'database'))
                        accounted_words.add(ind_word.lower())

                # Fodder words
                for w in fodder_words:
                    word_roles.append(
                        WordRole(w, 'fodder', fodder_letters.upper(), 'brute_force'))
                    accounted_words.add(w.lower())

                # Definition words
                if definition_window:
                    for w in definition_window.split():
                        if w.lower() not in accounted_words:
                            word_roles.append(
                                WordRole(w, 'definition', likely_answer, 'pipeline'))
                            accounted_words.add(w.lower())

                # Find remaining words for compound analysis
                # Use norm_letters to handle punctuation (e.g., "Yes," vs "yes")
                accounted_normalized = {norm_letters(a) for a in accounted_words}
                remaining_words = [w for w in clue_words if
                                   norm_letters(w) not in accounted_normalized]

                # Do compound analysis if there are remaining words (especially insertion material)
                compound_solution = None
                if remaining_words:
                    compound_solution = self._analyze_remaining_words(
                        remaining_words, fodder_letters, likely_answer, word_roles,
                        accounted_words, clue_words, definition_window
                    )

                # Check if indicator reassignment is needed (orphaned substitution case)
                if compound_solution and anagram_indicator:
                    unresolved = compound_solution.get('unresolved_words', [])
                    reassignment = self._check_indicator_reassignment_needed(
                        anagram_indicator, compound_solution, unresolved,
                        fodder_words, definition_window
                    )

                    if reassignment:
                        new_anagram_indicator, operation_type = reassignment

                        # Update word_roles: change old anagram indicator to operation indicator
                        for wr in word_roles:
                            if wr.word.lower() == anagram_indicator.lower() and wr.role == 'anagram_indicator':
                                wr.role = 'operation_indicator'
                                wr.source = f'database ({operation_type})'
                                break

                        # Add new anagram indicator to word_roles
                        word_roles.append(WordRole(
                            new_anagram_indicator, 'anagram_indicator', '',
                            'database (reassigned)'
                        ))
                        accounted_words.add(new_anagram_indicator.lower())

                        # Update anagram_indicator for explanation
                        old_indicator = anagram_indicator
                        anagram_indicator = new_anagram_indicator

                        # Update compound_solution with operation indicator
                        compound_solution['operation_indicators'] = [
                            (old_indicator, operation_type, '')
                        ]
                        compound_solution['unresolved_words'] = [
                            w for w in unresolved
                            if norm_letters(w) != norm_letters(new_anagram_indicator)
                        ]

                # Build explanation
                from stages.explanation import ExplanationBuilder
                explainer = ExplanationBuilder()
                explanation = explainer.build_explanation(
                    case, word_roles, fodder_words, fodder_letters,
                    anagram_indicator, definition_window, compound_solution, clue_words,
                    likely_answer
                )

                # Check if all words are accounted for (pure anagram, no compound needed)
                # Use norm_letters to handle punctuation (e.g., "Yes," vs "yes")
                accounted_normalized = {norm_letters(a) for a in accounted_words}
                remaining_unresolved = [w for w in clue_words if
                                        norm_letters(w) not in accounted_normalized]

                # If no compound_solution but all words accounted for, mark as fully resolved
                if compound_solution is None and not remaining_unresolved:
                    compound_solution = {
                        'fully_resolved': True,
                        'operation': 'pure_anagram'
                    }
                elif compound_solution is None:
                    compound_solution = {
                        'fully_resolved': False,
                        'unresolved_words': remaining_unresolved
                    }

                # Mark partial anagrams as 2 (stays in anagram cohort, not forwarded to general)
                # Values: 0 = no anagram, 1 = fully resolved, 2 = partial anagram
                # Check if anagram stage already validated this as a partial anagram
                anagrams = case.get('anagrams', [])
                is_validated_partial = any(
                    hit.get('solve_type') == 'anagram_evidence_partial'
                    for hit in anagrams
                )
                if compound_solution and is_validated_partial:
                    if compound_solution.get('letters_still_needed'):
                        compound_solution['fully_resolved'] = 2

                    # Format answers with spaces for multi-word enumerations
                    enumeration = case.get('enumeration', '')
                    likely_answer_formatted = format_answer_with_enumeration(
                        likely_answer,
                        enumeration)
                    db_answer_formatted = format_answer_with_enumeration(db_answer,
                                                                         enumeration)

                result = {
                    'clue': clue_text,
                    'likely_answer': likely_answer_formatted,
                    'db_answer': db_answer_formatted,
                    'answer_matches': norm_letters(likely_answer) == norm_letters(
                        db_answer),
                    'word_roles': word_roles,
                    'definition_window': definition_window,
                    'anagram_component': {
                        'fodder_words': fodder_words,
                        'fodder_letters': fodder_letters,
                        'indicator': anagram_indicator
                    },
                    'compound_solution': compound_solution,
                    'explanation': explanation,
                    'remaining_unresolved': remaining_unresolved
                }

                # Score this result: fully_resolved=1 is best, then fewer letters_still_needed
                fully_resolved = compound_solution.get('fully_resolved',
                                                       False) if compound_solution else False
                if fully_resolved == 1 or fully_resolved is True:
                    # Perfect - return immediately
                    return result

                # Calculate score for partial results (lower letters_still_needed is better)
                letters_needed = len(compound_solution.get('letters_still_needed',
                                                           '')) if compound_solution else 999
                score = 100 - letters_needed  # Higher score = fewer letters needed

                if score > best_score:
                    best_score = score
                    best_result = result

            # End of window loop - return best result found
            if best_result:
                return best_result

            # SURGICAL FIX: Reconstruct fodder_words from word_roles if empty
            if not fodder_words and word_roles:
                fodder_from_roles = [wr.word for wr in word_roles if wr.role == 'fodder']
                if fodder_from_roles:
                    fodder_words = fodder_from_roles
                    if not fodder_letters:
                        fodder_letters = ''.join(
                            c.upper() for w in fodder_words for c in w if c.isalpha()
                        )

        # Fall back to evidence_analysis path for non-exact matches
        # Get evidence analysis results
        evidence_analysis = case.get('evidence_analysis', {})
        scored_candidates = evidence_analysis.get('scored_candidates', [])

        if not scored_candidates:
            # Lazy import to avoid circular dependency
            from stages.explanation import ExplanationBuilder
            explainer = ExplanationBuilder()
            return explainer.build_fallback(case, clue_words, db_answer)

        # Find the first candidate WITH evidence (not just highest scored)
        # A candidate might rank high from definition_support but have no wordplay evidence
        top_candidate = None
        evidence = None
        for sc in scored_candidates:
            if sc.get('evidence'):
                top_candidate = sc
                evidence = sc.get('evidence')
                break

        if not top_candidate or not evidence:
            # No candidates have evidence - fall back
            # Lazy import to avoid circular dependency
            from stages.explanation import ExplanationBuilder
            explainer = ExplanationBuilder()
            return explainer.build_fallback(case, clue_words, db_answer)

        # LIKELY ANSWER comes from the matched candidate with evidence
        likely_answer = top_candidate.get('candidate', '').upper().replace(' ', '')

        # Get definition from pipeline data (needed for indicator inference)
        definition_window = self._get_definition_window(case, clue_words)

        # Extract anagram component
        fodder_words = evidence.fodder_words or []
        fodder_letters = evidence.fodder_letters or ''

        # First try to get indicator from evidence (already found by evidence system)
        anagram_indicator = None
        evidence_indicator = None

        if hasattr(evidence, 'indicator_words') and evidence.indicator_words:
            # Evidence system found an indicator
            evidence_indicator = ' '.join(evidence.indicator_words)

        # ALWAYS search for indicator using frequency-based selection
        # This ensures we pick the highest-frequency indicator
        frequency_indicator = self._find_anagram_indicator(clue_words, fodder_words,
                                                           definition_window)

        # Prefer frequency-based indicator if found, otherwise use evidence indicator
        if frequency_indicator:
            anagram_indicator = frequency_indicator
        elif evidence_indicator:
            anagram_indicator = evidence_indicator

        # Check if any fodder words should be reassigned as insertion material
        # (e.g., "Me" in "Me sporting tailored sateen" should be insertion, not fodder)
        insertion_material = []
        if fodder_words and anagram_indicator:
            fodder_words, insertion_material = self._reassign_fodder_near_insertion_indicators(
                fodder_words, clue_words, anagram_indicator
            )
            # Recalculate fodder_letters if we removed any words
            if insertion_material:
                fodder_letters = ''.join(
                    c.upper() for w in fodder_words for c in w if c.isalpha()
                )

        # Build word roles tracking
        word_roles = []
        accounted_words = set()

        # Account for anagram indicator FIRST (so we can exclude from definition)
        # Add each word separately for breakdown lookup
        if anagram_indicator:
            for ind_word in anagram_indicator.split():
                word_roles.append(
                    WordRole(ind_word, 'anagram_indicator', '', 'evidence'))
                accounted_words.add(ind_word.lower())

        # Account for anagram fodder (after potential reassignment)
        for fw in fodder_words:
            word_roles.append(WordRole(fw, 'fodder', fodder_letters, 'evidence'))
            accounted_words.add(fw.lower())

        # Account for definition (excluding indicator and fodder)
        if definition_window:
            def_words = definition_window.split()
            for w in def_words:
                w_norm = norm_letters(w)
                # Skip if already accounted (indicator or fodder)
                if w_norm in {norm_letters(a) for a in accounted_words}:
                    continue
                word_roles.append(WordRole(w, 'definition', likely_answer, 'pipeline'))
                accounted_words.add(w.lower())

        # NOTE: Link words are NOT marked here anymore.
        # They will be included in remaining_words so compound analysis can use them
        # (for fodder, parts indicators like "close to", etc.)
        # Unused link words will be marked at the end.

        # Helper to normalize word for comparison (strip punctuation)
        def normalize_word(w):
            return ''.join(c.lower() for c in w if c.isalpha())

        # Find remaining words (compare normalized forms)
        # Include link words - they may be fodder or part of indicators
        remaining_words = []
        for w in clue_words:
            w_norm = normalize_word(w)
            # Check if normalized form matches any accounted word
            if w_norm not in {normalize_word(aw) for aw in accounted_words}:
                remaining_words.append(w)

        # Analyze remaining words using database
        # Use LIKELY_ANSWER (from candidate), not db_answer
        compound_solution = None
        if remaining_words:
            compound_solution = self._analyze_remaining_words(
                remaining_words, fodder_letters, likely_answer, word_roles,
                accounted_words,
                clue_words, definition_window
            )

        # Check if indicator reassignment is needed (orphaned substitution case)
        # This handles "terribly honest about character" where "about" should be
        # container indicator, not anagram indicator
        if compound_solution and anagram_indicator:
            unresolved = compound_solution.get('unresolved_words', [])
            reassignment = self._check_indicator_reassignment_needed(
                anagram_indicator, compound_solution, unresolved,
                fodder_words, definition_window
            )

            if reassignment:
                new_anagram_indicator, operation_type = reassignment

                # Update word_roles: change old anagram indicator to operation indicator
                for wr in word_roles:
                    if wr.word.lower() == anagram_indicator.lower() and wr.role == 'anagram_indicator':
                        # Modify in place
                        wr.role = 'operation_indicator'
                        wr.source = f'database ({operation_type})'
                        break

                # Add new anagram indicator to word_roles
                word_roles.append(WordRole(
                    new_anagram_indicator, 'anagram_indicator', '',
                    'database (reassigned)'
                ))
                accounted_words.add(new_anagram_indicator.lower())

                # Update anagram_indicator for explanation
                old_indicator = anagram_indicator
                anagram_indicator = new_anagram_indicator

                # Re-run compound analysis with the corrected remaining words
                # (now excluding the new indicator)
                remaining_words = [w for w in remaining_words
                                   if
                                   norm_letters(w) != norm_letters(new_anagram_indicator)]

                if remaining_words:
                    # Add the operation indicator to compound_solution
                    compound_solution['operation_indicators'] = [
                        (old_indicator, operation_type, '')
                    ]
                    compound_solution['unresolved_words'] = [
                        w for w in unresolved
                        if norm_letters(w) != norm_letters(new_anagram_indicator)
                    ]

        # NOW mark remaining link words (those not used by compound analysis)
        for w in clue_words:
            w_norm = norm_letters(w)
            if w_norm in self.link_words and w_norm not in {norm_letters(a) for a in
                                                            accounted_words}:
                word_roles.append(WordRole(w, 'link', '', 'heuristic'))
                accounted_words.add(w.lower())

        # Build explanation using LIKELY_ANSWER
        # Lazy import to avoid circular dependency
        from stages.explanation import ExplanationBuilder
        explainer = ExplanationBuilder()
        explanation = explainer.build_explanation(
            case, word_roles, fodder_words, fodder_letters,
            anagram_indicator, definition_window, compound_solution, clue_words,
            likely_answer
        )

        # Check if all words are accounted for (pure anagram, no compound needed)
        remaining_unresolved = [w for w in clue_words
                                if norm_letters(w) not in {norm_letters(a) for a in
                                                           accounted_words}]

        # If no compound_solution but all words accounted for, mark as fully resolved
        if compound_solution is None and not remaining_unresolved:
            compound_solution = {
                'fully_resolved': True,
                'operation': 'pure_anagram'
            }
        elif compound_solution is None:
            compound_solution = {
                'fully_resolved': False,
                'unresolved_words': remaining_unresolved
            }

        # Mark partial anagrams as 2 (stays in anagram cohort, not forwarded to general)
        # Values: 0 = no anagram, 1 = fully resolved, 2 = partial anagram
        # Check if anagram stage already validated this as a partial anagram
        anagrams = case.get('anagrams', [])
        is_validated_partial = any(
            hit.get('solve_type') == 'anagram_evidence_partial'
            for hit in anagrams
        )
        if compound_solution and is_validated_partial:
            if compound_solution.get('letters_still_needed'):
                compound_solution['fully_resolved'] = 2

        # SURGICAL FIX: Reconstruct fodder_words from word_roles if empty
        if not fodder_words and word_roles:
            fodder_from_roles = [wr.word for wr in word_roles if wr.role == 'fodder']
            if fodder_from_roles:
                fodder_words = fodder_from_roles
                if not fodder_letters:
                    fodder_letters = ''.join(
                        c.upper() for w in fodder_words for c in w if c.isalpha()
                    )

        # Format answers with spaces for multi-word enumerations
        enumeration = case.get('enumeration', '')
        likely_answer_formatted = format_answer_with_enumeration(likely_answer,
                                                                 enumeration)
        db_answer_formatted = format_answer_with_enumeration(db_answer, enumeration)

        return {
            'clue': clue_text,
            'likely_answer': likely_answer_formatted,  # What we solved
            'db_answer': db_answer_formatted,  # For comparison only
            'answer_matches': norm_letters(likely_answer) == norm_letters(db_answer),
            # Verification
            'word_roles': word_roles,
            'definition_window': definition_window,
            'anagram_component': {
                'fodder_words': fodder_words,
                'fodder_letters': fodder_letters,
                'indicator': anagram_indicator
            },
            'compound_solution': compound_solution,
            'explanation': explanation,
            'remaining_unresolved': remaining_unresolved
        }

    def _find_anagram_indicator(self, clue_words: List[str],
                                fodder_words: List[str],
                                definition_window: Optional[str] = None) -> Optional[str]:
        """
        Find the anagram indicator in the clue.

        CORRECTED: Now searches only remaining words (clue - definition - fodder).
        The indicator must be in the remaining words, not in the definition or fodder.

        UPDATED: Now prefers higher frequency indicators when multiple candidates exist.

        1. First try two-word indicators in database (excluding definition and fodder)
        2. Then try single-word indicators in database (excluding definition and fodder)
        3. If multiple found, prefer highest frequency
        4. If not found, infer based on proximity to fodder and distance from definition
        """
        fodder_lower = {norm_letters(w) for w in fodder_words}
        def_words_lower = set()
        if definition_window:
            def_words_lower = {norm_letters(w) for w in definition_window.split()}

        # Collect all candidate indicators with their frequency
        indicator_candidates = []  # List of (indicator_str, frequency, match)

        # First pass: look for TWO-WORD indicators in database
        for i in range(len(clue_words) - 1):
            word1 = clue_words[i]
            word2 = clue_words[i + 1]
            # Skip if either word is fodder or definition
            if norm_letters(word1) in fodder_lower or norm_letters(word2) in fodder_lower:
                continue
            if norm_letters(word1) in def_words_lower or norm_letters(
                    word2) in def_words_lower:
                continue
            # Build two-word phrase (strip punctuation for lookup)
            two_word = f"{norm_letters(word1)} {norm_letters(word2)}"
            indicator_match = self.db.lookup_indicator(two_word)
            if indicator_match and indicator_match.wordplay_type == 'anagram':
                # Use raw frequency - no boost for two-word indicators
                freq = indicator_match.frequency
                indicator_candidates.append((f"{word1} {word2}", freq, indicator_match))

        # Second pass: look for SINGLE-WORD indicators in database
        for word in clue_words:
            # Skip if word is fodder or definition
            if norm_letters(word) in fodder_lower:
                continue
            if norm_letters(word) in def_words_lower:
                continue
            indicator_match = self.db.lookup_indicator(word)
            if indicator_match and indicator_match.wordplay_type == 'anagram':
                indicator_candidates.append(
                    (word, indicator_match.frequency, indicator_match))

        # If we found indicators, return the one with highest frequency
        if indicator_candidates:
            # Sort by frequency descending, return the best one
            indicator_candidates.sort(key=lambda x: x[1], reverse=True)
            return indicator_candidates[0][0]

        # Third pass: infer indicator based on proximity
        # Find indices of fodder words
        fodder_indices = []
        for i, word in enumerate(clue_words):
            if norm_letters(word) in fodder_lower:
                fodder_indices.append(i)

        if not fodder_indices:
            return None

        # Words adjacent to fodder (immediately before first or after last fodder)
        first_fodder_idx = min(fodder_indices)
        last_fodder_idx = max(fodder_indices)

        adjacent_candidates = []

        # Check word before first fodder
        if first_fodder_idx > 0:
            candidate = clue_words[first_fodder_idx - 1]
            if (norm_letters(candidate) not in fodder_lower and
                    norm_letters(candidate) not in def_words_lower and
                    norm_letters(candidate) not in self.link_words):
                adjacent_candidates.append((candidate, 'before', first_fodder_idx - 1))

        # Check word after last fodder
        if last_fodder_idx < len(clue_words) - 1:
            candidate = clue_words[last_fodder_idx + 1]
            if (norm_letters(candidate) not in fodder_lower and
                    norm_letters(candidate) not in def_words_lower and
                    norm_letters(candidate) not in self.link_words):
                adjacent_candidates.append((candidate, 'after', last_fodder_idx + 1))

        # Also check TWO-WORD combinations adjacent to fodder for inference
        # Check two words after last fodder
        if last_fodder_idx < len(clue_words) - 2:
            word1 = clue_words[last_fodder_idx + 1]
            word2 = clue_words[last_fodder_idx + 2]
            if (norm_letters(word1) not in fodder_lower and
                    norm_letters(word2) not in fodder_lower and
                    norm_letters(word1) not in def_words_lower):
                adjacent_candidates.append(
                    (f"{word1} {word2}", 'after_two', last_fodder_idx + 1))

        if not adjacent_candidates:
            return None

        # Prefer candidate that is furthest from definition
        # If definition is at start, prefer candidate after fodder
        # If definition is at end, prefer candidate before fodder
        best_candidate = None

        if def_words_lower:
            # Find where definition is (start or end of clue)
            first_word_norm = norm_letters(clue_words[0])
            last_word_norm = norm_letters(clue_words[-1])

            def_at_start = first_word_norm in def_words_lower
            def_at_end = last_word_norm in def_words_lower

            for candidate, position, idx in adjacent_candidates:
                if def_at_start and position in ('after', 'after_two'):
                    best_candidate = candidate
                    break
                elif def_at_end and position == 'before':
                    best_candidate = candidate
                    break

            # If no preference matched, just take first adjacent
            if not best_candidate and adjacent_candidates:
                best_candidate = adjacent_candidates[0][0]
        else:
            # No definition window, just take first adjacent candidate
            best_candidate = adjacent_candidates[0][0] if adjacent_candidates else None

        if best_candidate:
            # REMOVED: Do not insert inferred indicators into database - this pollutes data
            # self._insert_inferred_indicator(best_candidate, 'anagram')
            return best_candidate

        return None

    def _check_indicator_reassignment_needed(self, anagram_indicator: str,
                                             compound_solution: Dict,
                                             unresolved_words: List[str],
                                             fodder_words: List[str],
                                             definition_window: Optional[str]) -> \
            Optional[Tuple[str, str]]:
        """
        Check if the anagram indicator should be reassigned as a container/insertion indicator.

        This handles cases like "terribly honest about character" where:
        - "about" was selected as anagram indicator (high frequency as anagram)
        - "character" -> CARD was found as substitution (orphaned - no operation indicator)
        - But "about" is ALSO a container indicator
        - "terribly" is a valid anagram indicator in unresolved words

        Returns: (new_anagram_indicator, operation_indicator_type) or None if no reassignment needed
        """
        if not compound_solution or not anagram_indicator:
            return None

        # Check if there are substitutions but NO operation indicators
        substitutions = compound_solution.get('substitutions', [])
        operation_indicators = compound_solution.get('operation_indicators', [])

        if not substitutions or operation_indicators:
            # Either no substitutions, or already have operation indicator - no reassignment needed
            return None

        # We have orphaned substitution(s) - check if current anagram indicator is also container/insertion
        indicator_word = norm_letters(anagram_indicator.split()[0])  # Handle multi-word
        indicator_match = self.db.lookup_indicator(indicator_word)

        if not indicator_match:
            return None

        # Check if this indicator is ALSO a container or insertion indicator
        # Query DB directly to check all types for this word
        conn = self.db._get_connection()
        cursor = conn.cursor()
        cursor.execute("""
            SELECT wordplay_type FROM indicators 
            WHERE LOWER(word) = ? AND wordplay_type IN ('container', 'insertion')
        """, (indicator_word,))
        alt_types = cursor.fetchall()

        if not alt_types:
            # Current indicator is not also a container/insertion indicator
            return None

        operation_type = alt_types[0][0]  # 'container' or 'insertion'

        # Now check if there's another valid anagram indicator in unresolved_words
        fodder_lower = {norm_letters(w) for w in fodder_words}
        def_words_lower = set()
        if definition_window:
            def_words_lower = {norm_letters(w) for w in definition_window.split()}

        # Look for alternative anagram indicator in unresolved words
        for word in unresolved_words:
            word_norm = norm_letters(word)

            # Skip if it's fodder or definition
            if word_norm in fodder_lower or word_norm in def_words_lower:
                continue

            # Check if this word is a valid anagram indicator
            alt_indicator = self.db.lookup_indicator(word)
            if alt_indicator and alt_indicator.wordplay_type == 'anagram':
                # Found an alternative anagram indicator!
                # Return the reassignment: (new anagram indicator, operation type for old indicator)
                return (word, operation_type)

        return None

    def _reassign_fodder_near_insertion_indicators(self, fodder_words: List[str],
                                                   clue_words: List[str],
                                                   anagram_indicator: Optional[str]) -> \
            Tuple[List[str], List[str]]:
        """
        Check if any fodder words should be reassigned as insertion/container material.

        If a fodder word is adjacent to an insertion/container indicator in the clue
        (and NOT adjacent to the anagram indicator), it should be treated as
        insertion material, not anagram fodder.

        Example: "Me sporting tailored sateen"
        - "sporting" is insertion indicator
        - "Me" is adjacent to "sporting", should be insertion material
        - "tailored" is anagram indicator
        - "sateen" is adjacent to "tailored", remains fodder

        Returns: (updated_fodder_words, insertion_material_words)
        """
        if not fodder_words or len(fodder_words) < 2:
            return fodder_words, []

        # Normalize helper
        def norm(w):
            return ''.join(c.lower() for c in w if c.isalpha())

        # Find positions of all words in clue
        word_positions = {}
        for i, cw in enumerate(clue_words):
            word_positions[norm(cw)] = i

        # Get anagram indicator position(s)
        anagram_ind_positions = set()
        if anagram_indicator:
            for ind_word in anagram_indicator.split():
                ind_norm = norm(ind_word)
                if ind_norm in word_positions:
                    anagram_ind_positions.add(word_positions[ind_norm])

        # Find insertion/container indicators in the clue
        insertion_ind_positions = {}
        for i, cw in enumerate(clue_words):
            indicator = self.db.lookup_indicator(cw)
            if indicator and indicator.wordplay_type in ('insertion', 'container'):
                insertion_ind_positions[i] = cw

        if not insertion_ind_positions:
            return fodder_words, []

        # Check each fodder word
        insertion_material = []
        updated_fodder = []

        for fw in fodder_words:
            fw_norm = norm(fw)
            if fw_norm not in word_positions:
                updated_fodder.append(fw)
                continue

            fw_pos = word_positions[fw_norm]

            # Check if this fodder word is adjacent to an insertion indicator
            adjacent_to_insertion = False
            for ins_pos in insertion_ind_positions:
                if abs(fw_pos - ins_pos) == 1:
                    adjacent_to_insertion = True
                    break

            # Check if this fodder word is adjacent to the anagram indicator
            adjacent_to_anagram = False
            for ana_pos in anagram_ind_positions:
                if abs(fw_pos - ana_pos) == 1:  # Strictly adjacent (distance = 1)
                    adjacent_to_anagram = True
                    break

            # Reassign if adjacent to insertion but NOT adjacent to anagram indicator
            if adjacent_to_insertion and not adjacent_to_anagram:
                insertion_material.append(fw)
            else:
                updated_fodder.append(fw)

        return updated_fodder, insertion_material

    def _insert_inferred_indicator(self, word: str, wordplay_type: str):
        """Insert an inferred indicator into the database with low confidence."""
        # For two-word indicators, clean each word but keep the space
        if ' ' in word:
            parts = word.split()
            word_clean = ' '.join(norm_letters(p) for p in parts)
        else:
            word_clean = norm_letters(word)

        if not word_clean:
            return

        # Check if already exists
        existing = self.db.lookup_indicator(word_clean)
        if existing:
            return  # Already in database

        try:
            conn = self.db._get_connection()
            cursor = conn.cursor()
            cursor.execute("""
                INSERT OR IGNORE INTO indicators (word, wordplay_type, subtype, confidence)
                VALUES (?, ?, ?, ?)
            """, (word_clean, wordplay_type, 'inferred', 'low'))
            conn.commit()

        except Exception as e:
            pass  # Silently fail - don't break analysis

    def _get_definition_window(self, case: Dict[str, Any],
                               clue_words: List[str]) -> Optional[str]:
        """Extract definition window from pipeline data (returns first match for backward compat)."""
        windows = self._get_all_valid_definition_windows(case, clue_words)
        return windows[0] if windows else None

    def _get_all_valid_definition_windows(self, case: Dict[str, Any],
                                          clue_words: List[str]) -> List[str]:
        """Extract ALL valid definition windows from pipeline data."""
        window_support = case.get('window_support', {})
        answer = case.get('answer', '').upper().replace(' ', '')
        valid_windows = []

        if window_support:
            for window_text, candidates in window_support.items():
                if isinstance(candidates, list):
                    normalized = [c.upper().replace(' ', '') for c in candidates]
                    if answer in normalized:
                        valid_windows.append(window_text)

        # Fallback: use definition from case if available
        if not valid_windows and case.get('definition'):
            valid_windows.append(case.get('definition'))

        # NEW FALLBACK: Query definition_answers_augmented table
        if not valid_windows:
            definition = self._find_definition_from_db(answer, clue_words)
            if definition:
                valid_windows.append(definition)

        return valid_windows

    def _find_definition_from_db(self, answer: str, clue_words: List[str]) -> Optional[
        str]:
        """
        Find definition by looking up known definitions for this answer
        in definition_answers_augmented table, then checking which appears in clue.

        Returns the definition phrase if found, None otherwise.
        """
        if not answer:
            return None

        answer_upper = answer.upper().replace(' ', '')

        conn = self.db._get_connection()
        cursor = conn.cursor()

        # Get all known definitions for this answer
        cursor.execute("""
            SELECT DISTINCT definition 
            FROM definition_answers_augmented 
            WHERE UPPER(REPLACE(answer, ' ', '')) = ?
        """, (answer_upper,))

        known_definitions = [row[0] for row in cursor.fetchall()]

        if not known_definitions:
            return None

        # Build clue text for matching (lowercase, without enumeration)
        clue_text_lower = ' '.join(clue_words).lower()

        # Sort by length (prefer longer/more specific definitions)
        known_definitions.sort(key=len, reverse=True)

        # Check which definition appears in the clue
        for defn in known_definitions:
            if defn.lower() in clue_text_lower:
                return defn

        return None

    def _analyze_remaining_words(self, remaining_words: List[str],
                                 anagram_letters: str, answer: str,
                                 word_roles: List[WordRole],
                                 accounted_words: Set[str],
                                 clue_words: List[str],
                                 definition_window: Optional[str],
                                 skip_indicator_words: Optional[Set[str]] = None,
                                 skip_substitutions: Optional[
                                     Set[Tuple[str, str]]] = None,
                                 _backtrack_depth: int = 0) -> Optional[
        Dict[str, Any]]:
        """
        Analyze remaining words by querying the database.
        Identifies substitutions and construction operations.
        Handles both ADDITIONS (answer > anagram) and DELETIONS (anagram > answer).

        skip_substitutions: Set of (word, letters) tuples to skip during substitution lookup.
        _backtrack_depth: Internal counter to prevent infinite recursion in backtracking.
        """
        answer_upper = answer.upper().replace(' ', '')
        anagram_upper = anagram_letters.upper()

        # Save original state for backtracking (before any modifications)
        original_word_roles = list(word_roles)
        original_accounted_words = set(accounted_words)

        # Calculate what letters we still need (for additions)
        needed_letters = ''
        temp_anagram = list(anagram_upper)
        for c in answer_upper:
            if c in temp_anagram:
                temp_anagram.remove(c)
            else:
                needed_letters += c

        # Calculate excess letters (for deletions)
        # What's left in temp_anagram after removing all answer letters
        excess_letters = ''.join(sorted(temp_anagram))

        # Handle DELETION case: anagram has MORE letters than answer
        if not needed_letters and excess_letters:
            deletion_result = self._handle_deletion_compound(
                remaining_words, anagram_letters, answer, excess_letters,
                word_roles, accounted_words, clue_words, definition_window
            )
            # If deletion succeeded (found indicator), return it
            if deletion_result is not None:
                return deletion_result

            # No deletion indicator - try reducing fodder and looking for additions instead
            # This handles cases like GREATDANE where DOG+NEAR+GATE was grabbed but
            # the real solution is NEAR+GATE + D (daughter)
            alternative = self._try_reduced_fodder(
                remaining_words, anagram_letters, answer, excess_letters,
                word_roles, accounted_words, clue_words, definition_window
            )
            if alternative:
                return alternative

            # Still nothing - return unresolved
            return {
                'operation': 'unresolved_excess',
                'excess_letters': excess_letters,
                'fully_resolved': False,
                'note': 'Excess letters but no deletion indicator or alternative found'
            }

        if not needed_letters and not excess_letters:
            # Pure anagram - no additions or deletions needed
            return self._classify_remaining_as_indicators(
                remaining_words, word_roles, accounted_words,
                clue_words, definition_window
            )

        # Handle ADDITION case: answer has MORE letters than anagram
        # We need additional letters - look for substitutions OR additional fodder
        found_substitutions = []
        additional_fodder = []  # Words that provide needed letters directly
        operation_indicators = []
        positional_indicators = []

        # Get definition words to exclude from positional check
        def_words_lower = set()
        if definition_window:
            def_words_lower = {w.lower() for w in definition_window.split()}

        # Helper to get letters from word (strip punctuation)
        def get_letters(w):
            return ''.join(c.upper() for c in w if c.isalpha())

        # Helper to check if two strings are anagrams
        def is_anagram(s1, s2):
            return sorted(s1.upper()) == sorted(s2.upper())

        # FIRST PASS: Find all insertion/container indicators in remaining_words
        # This is needed so we can check adjacency before labeling fodder
        # IMPORTANT: Check THREE-WORD then TWO-WORD indicators FIRST to prevent splitting
        insertion_indicators_in_remaining = {}  # position -> (word, indicator_match)
        words_used_by_two_word_indicators = set()  # Track words used in multi-word indicators

        # Check THREE-word indicators first (like "on the radio", "by the sound of it")
        for i in range(len(remaining_words) - 2):
            word1 = remaining_words[i]
            word2 = remaining_words[i + 1]
            word3 = remaining_words[i + 2]
            three_word = f"{norm_letters(word1)} {norm_letters(word2)} {norm_letters(word3)}"
            indicator_match = self.db.lookup_indicator(three_word)

            if indicator_match and indicator_match.wordplay_type in ('homophone', 'parts',
                                                                     'acrostic'):
                # Protect three-word indicators from being split
                words_used_by_two_word_indicators.add(word1.lower())
                words_used_by_two_word_indicators.add(word2.lower())
                words_used_by_two_word_indicators.add(word3.lower())

        # Check two-word container/insertion indicators first
        for i in range(len(remaining_words) - 1):
            word1 = remaining_words[i]
            word2 = remaining_words[i + 1]

            # Skip if already part of a three-word indicator
            if word1.lower() in words_used_by_two_word_indicators or word2.lower() in words_used_by_two_word_indicators:
                continue

            two_word = f"{norm_letters(word1)} {norm_letters(word2)}"
            indicator_match = self.db.lookup_indicator(two_word)

            if indicator_match and indicator_match.wordplay_type in ('insertion',
                                                                     'container'):
                # Found a two-word indicator like "found in"
                insertion_indicators_in_remaining[i] = (f"{word1} {word2}",
                                                        indicator_match)
                words_used_by_two_word_indicators.add(word1.lower())
                words_used_by_two_word_indicators.add(word2.lower())
            elif indicator_match and indicator_match.wordplay_type in ('acrostic',
                                                                       'parts',
                                                                       'homophone'):
                # Protect two-word acrostic/parts/homophone indicators like "leaders in", "sounds like"
                # They'll be processed later, but we need to prevent single-word matching
                words_used_by_two_word_indicators.add(word1.lower())
                words_used_by_two_word_indicators.add(word2.lower())

        # Now check single-word indicators, but skip words already used in two-word indicators
        for i, word in enumerate(remaining_words):
            if word.lower() in words_used_by_two_word_indicators:
                continue  # Skip - already part of a two-word indicator
            indicator_match = self.db.lookup_indicator(word)
            if indicator_match and indicator_match.wordplay_type in ('insertion',
                                                                     'container'):
                insertion_indicators_in_remaining[i] = (word, indicator_match)

        # SECOND: Check for multi-word parts indicators
        # These extract first/last letters from adjacent words
        parts_found = []
        words_used_by_parts = set()

        # Check THREE-word indicators first (like "out of bounds")
        for i, word in enumerate(remaining_words[:-2]):
            three_word = f"{norm_letters(word)} {norm_letters(remaining_words[i + 1])} {norm_letters(remaining_words[i + 2])}"
            indicator_match = self.db.lookup_indicator(three_word)

            # THREE-WORD HOMOPHONE: e.g., "on the radio", "for the audience"
            if indicator_match and indicator_match.wordplay_type == 'homophone':
                homophone_found = False
                source_word = None

                # Try source word BEFORE indicator first (e.g., "passage on the radio")
                if i > 0 and not homophone_found:
                    source_word = remaining_words[i - 1]
                    source_letters = get_letters(source_word)

                    if source_word.lower() not in words_used_by_parts and source_letters:
                        # Strategy 1: Direct homophone lookup
                        homophones = self.db.lookup_homophone(source_word)

                        for homophone in homophones:
                            homophone_letters = get_letters(homophone)

                            temp_needed = needed_letters
                            can_use = True
                            for c in homophone_letters:
                                if c in temp_needed:
                                    temp_needed = temp_needed.replace(c, '', 1)
                                else:
                                    can_use = False
                                    break

                            if can_use:
                                ind_display = f"{word} {remaining_words[i + 1]} {remaining_words[i + 2]}"
                                parts_found.append({
                                    'indicator': ind_display,
                                    'indicator_words': [word, remaining_words[i + 1],
                                                        remaining_words[i + 2]],
                                    'source_word': source_word,
                                    'extracted_letter': homophone_letters,
                                    'subtype': f'homophone ({homophone})',
                                    'is_delete': False,
                                    'is_homophone': True
                                })
                                words_used_by_parts.update(
                                    [word.lower(), remaining_words[i + 1].lower(),
                                     remaining_words[i + 2].lower(), source_word.lower()])
                                for c in homophone_letters:
                                    needed_letters = needed_letters.replace(c, '', 1)
                                homophone_found = True
                                break

                        # Strategy 2: Synonym → Homophone
                        if not homophone_found:
                            synonyms = self.db.lookup_substitution(source_word,
                                                                   max_synonym_length=12)

                            for syn_match in synonyms:
                                syn_word = syn_match.letters.lower()
                                syn_homophones = self.db.lookup_homophone(syn_word)

                                for homophone in syn_homophones:
                                    homophone_letters = get_letters(homophone)

                                    temp_needed = needed_letters
                                    can_use = True
                                    for c in homophone_letters:
                                        if c in temp_needed:
                                            temp_needed = temp_needed.replace(c, '', 1)
                                        else:
                                            can_use = False
                                            break

                                    if can_use:
                                        ind_display = f"{word} {remaining_words[i + 1]} {remaining_words[i + 2]}"
                                        parts_found.append({
                                            'indicator': ind_display,
                                            'indicator_words': [word,
                                                                remaining_words[i + 1],
                                                                remaining_words[i + 2]],
                                            'source_word': source_word,
                                            'extracted_letter': homophone_letters,
                                            'subtype': f'homophone ({syn_word} → {homophone})',
                                            'is_delete': False,
                                            'is_homophone': True
                                        })
                                        words_used_by_parts.update(
                                            [word.lower(), remaining_words[i + 1].lower(),
                                             remaining_words[i + 2].lower(),
                                             source_word.lower()])
                                        for c in homophone_letters:
                                            needed_letters = needed_letters.replace(c, '',
                                                                                    1)
                                        homophone_found = True
                                        break

                                if homophone_found:
                                    break

                # Try source word AFTER indicator (e.g., "on the radio passage")
                if i + 3 < len(remaining_words) and not homophone_found:
                    source_word = remaining_words[i + 3]
                    source_letters = get_letters(source_word)

                    if source_word.lower() not in words_used_by_parts and source_letters:
                        # Strategy 1: Direct homophone lookup
                        homophones = self.db.lookup_homophone(source_word)

                        for homophone in homophones:
                            homophone_letters = get_letters(homophone)

                            temp_needed = needed_letters
                            can_use = True
                            for c in homophone_letters:
                                if c in temp_needed:
                                    temp_needed = temp_needed.replace(c, '', 1)
                                else:
                                    can_use = False
                                    break

                            if can_use:
                                ind_display = f"{word} {remaining_words[i + 1]} {remaining_words[i + 2]}"
                                parts_found.append({
                                    'indicator': ind_display,
                                    'indicator_words': [word, remaining_words[i + 1],
                                                        remaining_words[i + 2]],
                                    'source_word': source_word,
                                    'extracted_letter': homophone_letters,
                                    'subtype': f'homophone ({homophone})',
                                    'is_delete': False,
                                    'is_homophone': True
                                })
                                words_used_by_parts.update(
                                    [word.lower(), remaining_words[i + 1].lower(),
                                     remaining_words[i + 2].lower(), source_word.lower()])
                                for c in homophone_letters:
                                    needed_letters = needed_letters.replace(c, '', 1)
                                homophone_found = True
                                break

                        # Strategy 2: Synonym → Homophone
                        if not homophone_found:
                            synonyms = self.db.lookup_substitution(source_word,
                                                                   max_synonym_length=12)

                            for syn_match in synonyms:
                                syn_word = syn_match.letters.lower()
                                syn_homophones = self.db.lookup_homophone(syn_word)

                                for homophone in syn_homophones:
                                    homophone_letters = get_letters(homophone)

                                    temp_needed = needed_letters
                                    can_use = True
                                    for c in homophone_letters:
                                        if c in temp_needed:
                                            temp_needed = temp_needed.replace(c, '', 1)
                                        else:
                                            can_use = False
                                            break

                                    if can_use:
                                        ind_display = f"{word} {remaining_words[i + 1]} {remaining_words[i + 2]}"
                                        parts_found.append({
                                            'indicator': ind_display,
                                            'indicator_words': [word,
                                                                remaining_words[i + 1],
                                                                remaining_words[i + 2]],
                                            'source_word': source_word,
                                            'extracted_letter': homophone_letters,
                                            'subtype': f'homophone ({syn_word} → {homophone})',
                                            'is_delete': False,
                                            'is_homophone': True
                                        })
                                        words_used_by_parts.update(
                                            [word.lower(), remaining_words[i + 1].lower(),
                                             remaining_words[i + 2].lower(),
                                             source_word.lower()])
                                        for c in homophone_letters:
                                            needed_letters = needed_letters.replace(c, '',
                                                                                    1)
                                        homophone_found = True
                                        break

                                if homophone_found:
                                    break

                # Mark three-word indicator as used even if no homophone found
                if homophone_found:
                    continue

            if indicator_match and indicator_match.wordplay_type == 'parts':
                subtype = indicator_match.subtype or ''

                # Find source word - try PREVIOUS word first
                source_word = None
                source_letters = None
                source_idx = None

                if i > 0:
                    source_word = remaining_words[i - 1]
                    # Strip possessive 's before getting letters (Tallinn's → TALLINN)
                    source_clean = re.sub(r"[''\u2019]s$", "", source_word,
                                          flags=re.IGNORECASE)
                    source_letters = get_letters(source_clean)
                    source_idx = i - 1
                elif i + 3 < len(remaining_words):
                    source_word = remaining_words[i + 3]
                    source_clean = re.sub(r"[''\u2019]s$", "", source_word,
                                          flags=re.IGNORECASE)
                    source_letters = get_letters(source_clean)
                    source_idx = i + 3

                if source_word and source_letters and len(source_letters) >= 3:
                    # OUTER DELETE: Remove first and last letters
                    if 'outer' in subtype.lower() and 'delete' in subtype.lower():
                        remaining_fodder = source_letters[1:-1]  # Remove first and last

                        # Check if remaining matches needed_letters
                        temp_needed = list(needed_letters)
                        can_use = True
                        for c in remaining_fodder:
                            if c in temp_needed:
                                temp_needed.remove(c)
                            else:
                                can_use = False
                                break

                        if can_use and remaining_fodder:
                            ind_display = f"{word} {remaining_words[i + 1]} {remaining_words[i + 2]}"
                            parts_found.append({
                                'indicator': ind_display,
                                'indicator_words': [word, remaining_words[i + 1],
                                                    remaining_words[i + 2]],
                                'source_word': source_word,
                                'remaining_fodder': remaining_fodder,
                                'subtype': subtype,
                                'is_delete': True
                            })
                            words_used_by_parts.update(
                                [word.lower(), remaining_words[i + 1].lower(),
                                 remaining_words[i + 2].lower(), source_word.lower()])
                            # Update needed letters
                            for c in remaining_fodder:
                                needed_letters = needed_letters.replace(c, '', 1)

        # Check TWO-word indicators (like "close to", "at first")
        for i, word in enumerate(remaining_words[:-1]):
            # Skip if already used by three-word indicator
            if word.lower() in words_used_by_parts:
                continue

            # Check two-word combination
            two_word = f"{norm_letters(word)} {norm_letters(remaining_words[i + 1])}"
            indicator_match = self.db.lookup_indicator(two_word)

            # TWO-WORD HOMOPHONE: e.g., "sounds like", "we hear"
            if indicator_match and indicator_match.wordplay_type == 'homophone':
                homophone_found = False
                source_word = None

                # Try source word BEFORE indicator first (e.g., "creature, we hear")
                if i > 0 and not homophone_found:
                    source_word = remaining_words[i - 1]
                    source_letters = get_letters(source_word)

                    if source_word.lower() not in words_used_by_parts and source_letters:
                        # Strategy 1: Direct homophone lookup
                        homophones = self.db.lookup_homophone(source_word)

                        for homophone in homophones:
                            homophone_letters = get_letters(homophone)

                            temp_needed = needed_letters
                            can_use = True
                            for c in homophone_letters:
                                if c in temp_needed:
                                    temp_needed = temp_needed.replace(c, '', 1)
                                else:
                                    can_use = False
                                    break

                            if can_use:
                                ind_display = f"{word} {remaining_words[i + 1]}"
                                parts_found.append({
                                    'indicator': ind_display,
                                    'indicator_words': [word, remaining_words[i + 1]],
                                    'source_word': source_word,
                                    'extracted_letter': homophone_letters,
                                    'subtype': f'homophone ({homophone})',
                                    'is_delete': False,
                                    'is_homophone': True
                                })
                                words_used_by_parts.update(
                                    [word.lower(), remaining_words[i + 1].lower(),
                                     source_word.lower()])
                                for c in homophone_letters:
                                    needed_letters = needed_letters.replace(c, '', 1)
                                homophone_found = True
                                break

                        # Strategy 2: Synonym → Homophone
                        if not homophone_found:
                            synonyms = self.db.lookup_substitution(source_word,
                                                                   max_synonym_length=12)

                            for syn_match in synonyms:
                                syn_word = syn_match.letters.lower()
                                syn_homophones = self.db.lookup_homophone(syn_word)

                                for homophone in syn_homophones:
                                    homophone_letters = get_letters(homophone)

                                    temp_needed = needed_letters
                                    can_use = True
                                    for c in homophone_letters:
                                        if c in temp_needed:
                                            temp_needed = temp_needed.replace(c, '', 1)
                                        else:
                                            can_use = False
                                            break

                                    if can_use:
                                        ind_display = f"{word} {remaining_words[i + 1]}"
                                        parts_found.append({
                                            'indicator': ind_display,
                                            'indicator_words': [word,
                                                                remaining_words[i + 1]],
                                            'source_word': source_word,
                                            'extracted_letter': homophone_letters,
                                            'subtype': f'homophone ({syn_word} → {homophone})',
                                            'is_delete': False,
                                            'is_homophone': True
                                        })
                                        words_used_by_parts.update(
                                            [word.lower(), remaining_words[i + 1].lower(),
                                             source_word.lower()])
                                        for c in homophone_letters:
                                            needed_letters = needed_letters.replace(c, '',
                                                                                    1)
                                        homophone_found = True
                                        break

                                if homophone_found:
                                    break

                # Try source word AFTER indicator (e.g., "sounds like creature")
                if i + 2 < len(remaining_words) and not homophone_found:
                    source_word = remaining_words[i + 2]
                    source_letters = get_letters(source_word)

                    if source_word.lower() not in words_used_by_parts and source_letters:
                        # Strategy 1: Direct homophone lookup
                        homophones = self.db.lookup_homophone(source_word)

                        for homophone in homophones:
                            homophone_letters = get_letters(homophone)

                            temp_needed = needed_letters
                            can_use = True
                            for c in homophone_letters:
                                if c in temp_needed:
                                    temp_needed = temp_needed.replace(c, '', 1)
                                else:
                                    can_use = False
                                    break

                            if can_use:
                                ind_display = f"{word} {remaining_words[i + 1]}"
                                parts_found.append({
                                    'indicator': ind_display,
                                    'indicator_words': [word, remaining_words[i + 1]],
                                    'source_word': source_word,
                                    'extracted_letter': homophone_letters,
                                    'subtype': f'homophone ({homophone})',
                                    'is_delete': False,
                                    'is_homophone': True
                                })
                                words_used_by_parts.update(
                                    [word.lower(), remaining_words[i + 1].lower(),
                                     source_word.lower()])
                                for c in homophone_letters:
                                    needed_letters = needed_letters.replace(c, '', 1)
                                homophone_found = True
                                break

                        # Strategy 2: Synonym → Homophone
                        if not homophone_found:
                            synonyms = self.db.lookup_substitution(source_word,
                                                                   max_synonym_length=12)

                            for syn_match in synonyms:
                                syn_word = syn_match.letters.lower()
                                syn_homophones = self.db.lookup_homophone(syn_word)

                                for homophone in syn_homophones:
                                    homophone_letters = get_letters(homophone)

                                    temp_needed = needed_letters
                                    can_use = True
                                    for c in homophone_letters:
                                        if c in temp_needed:
                                            temp_needed = temp_needed.replace(c, '', 1)
                                        else:
                                            can_use = False
                                            break

                                    if can_use:
                                        ind_display = f"{word} {remaining_words[i + 1]}"
                                        parts_found.append({
                                            'indicator': ind_display,
                                            'indicator_words': [word,
                                                                remaining_words[i + 1]],
                                            'source_word': source_word,
                                            'extracted_letter': homophone_letters,
                                            'subtype': f'homophone ({syn_word} → {homophone})',
                                            'is_delete': False,
                                            'is_homophone': True
                                        })
                                        words_used_by_parts.update(
                                            [word.lower(), remaining_words[i + 1].lower(),
                                             source_word.lower()])
                                        for c in homophone_letters:
                                            needed_letters = needed_letters.replace(c, '',
                                                                                    1)
                                        homophone_found = True
                                        break

                                if homophone_found:
                                    break

            if indicator_match and indicator_match.wordplay_type in ('parts', 'acrostic'):
                subtype = indicator_match.subtype or ''

                # ALTERNATING: Check for alternating subtypes (e.g., "regularly ignored")
                # These extract every other letter from PREVIOUS word
                is_alternating = any(
                    x in subtype.lower() for x in ['alternate', 'odd', 'even', 'regular'])
                if is_alternating and i > 0:
                    source_word = remaining_words[i - 1]
                    source_letters = get_letters(source_word)

                    if source_word.lower() not in words_used_by_parts and source_letters and needed_letters:
                        # Try odd letters (positions 0, 2, 4... = 1st, 3rd, 5th...)
                        odd_letters = ''.join(
                            source_letters[j] for j in range(0, len(source_letters), 2))
                        # Try even letters (positions 1, 3, 5... = 2nd, 4th, 6th...)
                        even_letters = ''.join(
                            source_letters[j] for j in range(1, len(source_letters), 2))

                        extracted = None
                        extract_desc = None

                        # Check which works with needed_letters
                        temp_needed = list(needed_letters)
                        odd_works = True
                        for c in odd_letters:
                            if c in temp_needed:
                                temp_needed.remove(c)
                            else:
                                odd_works = False
                                break

                        temp_needed = list(needed_letters)
                        even_works = True
                        for c in even_letters:
                            if c in temp_needed:
                                temp_needed.remove(c)
                            else:
                                even_works = False
                                break

                        if odd_works:
                            extracted = odd_letters
                            extract_desc = 'odd letters of'
                        elif even_works:
                            extracted = even_letters
                            extract_desc = 'even letters of'

                        if extracted and extract_desc:
                            ind_display = f"{word} {remaining_words[i + 1]}"
                            parts_found.append({
                                'indicator': ind_display,
                                'indicator_words': [word, remaining_words[i + 1]],
                                'source_word': source_word,
                                'extracted_letter': extracted,
                                'subtype': f"{subtype} (alternating)",
                                'is_delete': False
                            })
                            words_used_by_parts.update(
                                [word.lower(), remaining_words[i + 1].lower(),
                                 source_word.lower()])
                            # Update needed letters
                            for c in extracted:
                                needed_letters = needed_letters.replace(c, '', 1)
                            continue  # Skip other handling

                # ACROSTIC: Extract first letter from EACH subsequent word
                # "leaders in Oslo will expect results" -> O+W+E+R
                if indicator_match.wordplay_type == 'acrostic' or (
                        'initial' in subtype.lower() or 'first' in subtype.lower()):
                    acrostic_letters = ''
                    acrostic_sources = []
                    for j in range(i + 2, len(remaining_words)):
                        src = remaining_words[j]
                        src_letters = get_letters(src)
                        if not src_letters:
                            continue
                        first_char = src_letters[0].upper()
                        if first_char in needed_letters:
                            # Verify this specific letter is still needed
                            temp = needed_letters
                            for c in acrostic_letters + first_char:
                                temp = temp.replace(c, '', 1)
                            if len(temp) < len(needed_letters) - len(acrostic_letters):
                                acrostic_letters += first_char
                                acrostic_sources.append(src)
                        else:
                            break  # Stop at first word that doesn't contribute

                    if len(acrostic_letters) >= 2:
                        # Mark indicator
                        ind_display = f"{word} {remaining_words[i + 1]}"
                        parts_found.append({
                            'indicator': ind_display,
                            'indicator_words': [word, remaining_words[i + 1]],
                            'source_word': ' '.join(acrostic_sources),
                            'extracted_letter': acrostic_letters,
                            'subtype': subtype,
                            'is_delete': False
                        })
                        words_used_by_parts.update(
                            [word.lower(), remaining_words[i + 1].lower()] +
                            [s.lower() for s in acrostic_sources])
                        # Update needed letters
                        for c in acrostic_letters:
                            needed_letters = needed_letters.replace(c, '', 1)
                        continue  # Skip the single-word source handling below

                # NON-ACROSTIC: Find single source word (original behavior)
                source_word = None
                source_idx = None
                if i + 2 < len(remaining_words):
                    source_word = remaining_words[i + 2]
                    source_letters = get_letters(source_word)

                    # Check if this is a DELETE operation
                    if 'delete' in subtype.lower() and source_letters:
                        remaining_fodder = None

                        if 'first' in subtype.lower():
                            remaining_fodder = source_letters[1:]
                        elif 'last' in subtype.lower():
                            remaining_fodder = source_letters[:-1]

                        if remaining_fodder:
                            parts_found.append({
                                'indicator': f"{word} {remaining_words[i + 1]}",
                                'indicator_words': [word, remaining_words[i + 1]],
                                'source_word': source_word,
                                'remaining_fodder': remaining_fodder,
                                'subtype': subtype,
                                'is_delete': True
                            })
                            words_used_by_parts.update(
                                [word.lower(), remaining_words[i + 1].lower(),
                                 source_word.lower()])

                    # EXTRACT operation
                    else:
                        extracted_letter = None
                        extracted_letters = ""  # For outer/edges (multiple letters)

                        if 'first' in subtype.lower() and source_letters:
                            extracted_letter = source_letters[0]
                        elif 'last' in subtype.lower() and source_letters:
                            extracted_letter = source_letters[-1]
                        elif ('outer' in subtype.lower() or 'edge' in subtype.lower() or
                              'case' in subtype.lower() or 'border' in subtype.lower()) and len(
                            source_letters) >= 2:
                            # Outer/edges/case = first AND last letters
                            extracted_letters = source_letters[0] + source_letters[-1]

                        # Handle single letter extraction
                        if extracted_letter and extracted_letter.upper() in needed_letters:
                            parts_found.append({
                                'indicator': f"{word} {remaining_words[i + 1]}",
                                'indicator_words': [word, remaining_words[i + 1]],
                                'source_word': source_word,
                                'extracted_letter': extracted_letter,
                                'subtype': subtype,
                                'is_delete': False
                            })
                            words_used_by_parts.update(
                                [word.lower(), remaining_words[i + 1].lower(),
                                 source_word.lower()])

                            # Update needed letters
                            needed_letters = needed_letters.replace(
                                extracted_letter.upper(), '',
                                1)

                        # Handle outer/edges (two letters)
                        elif extracted_letters:
                            # Check if BOTH letters are needed
                            temp_needed = needed_letters
                            can_use = True
                            for c in extracted_letters.upper():
                                if c in temp_needed:
                                    temp_needed = temp_needed.replace(c, '', 1)
                                else:
                                    can_use = False
                                    break

                            if can_use:
                                parts_found.append({
                                    'indicator': f"{word} {remaining_words[i + 1]}",
                                    'indicator_words': [word, remaining_words[i + 1]],
                                    'source_word': source_word,
                                    'extracted_letter': extracted_letters,  # Both letters
                                    'subtype': subtype,
                                    'is_delete': False
                                })
                                words_used_by_parts.update(
                                    [word.lower(), remaining_words[i + 1].lower(),
                                     source_word.lower()])

                                # Update needed letters
                                needed_letters = temp_needed

        # Process parts indicators found
        for part in parts_found:
            # Check if this is a homophone entry
            is_homophone = part.get('is_homophone', False)

            # Add indicator words to word_roles
            for ind_word in part['indicator_words']:
                if is_homophone:
                    word_roles.append(WordRole(
                        ind_word, 'homophone_indicator', '',
                        f"database ({part['subtype']})"
                    ))
                else:
                    word_roles.append(WordRole(
                        ind_word, 'parts_indicator', '', f"database ({part['subtype']})"
                    ))
                accounted_words.add(ind_word.lower())

            if part.get('is_delete'):
                # DELETE operation: add source word as truncated fodder
                subtype_lower = part['subtype'].lower()
                if 'outer' in subtype_lower:
                    delete_desc = 'minus outer letters'
                elif 'last' in subtype_lower:
                    delete_desc = 'minus last letter'
                else:
                    delete_desc = 'minus first letter'
                word_roles.append(WordRole(
                    part['source_word'], 'fodder', part['remaining_fodder'],
                    delete_desc
                ))
                accounted_words.add(part['source_word'].lower())

                # Add to additional_fodder for formula building
                additional_fodder.append((part['source_word'], part['remaining_fodder']))
            elif is_homophone:
                # HOMOPHONE operation: add source word with homophone description
                # subtype contains something like "homophone (aisle → isle)" or "homophone (isle)"
                homophone_desc = part['subtype']  # e.g., "homophone (aisle → isle)"
                extracted = part['extracted_letter']

                word_roles.append(WordRole(
                    part['source_word'], 'substitution', extracted,
                    homophone_desc
                ))
                accounted_words.add(part['source_word'].lower())

                # Add to substitutions list for formula building
                found_substitutions.append((
                    part['source_word'],
                    SubstitutionMatch(
                        word=part['source_word'],
                        letters=extracted,
                        category='homophone',
                        notes=homophone_desc
                    )
                ))
            else:
                # EXTRACT operation: add source word as providing the extracted letter
                source_desc = 'last letter of' if 'last' in part[
                    'subtype'].lower() else 'first letter of'
                source_words = part['source_word'].split()
                extracted = part['extracted_letter']

                if len(source_words) > 1 and len(extracted) == len(source_words):
                    # Acrostic: each source word contributes one letter
                    for sw, letter in zip(source_words, extracted):
                        word_roles.append(WordRole(
                            sw, 'substitution', letter,
                            source_desc
                        ))
                        accounted_words.add(sw.lower())
                else:
                    # Single source word
                    word_roles.append(WordRole(
                        part['source_word'], 'substitution', extracted,
                        source_desc
                    ))
                    accounted_words.add(part['source_word'].lower())

                # Add to substitutions list for formula building
                found_substitutions.append((
                    part['source_word'],
                    SubstitutionMatch(
                        word=part['source_word'],
                        letters=part['extracted_letter'],
                        category=source_desc
                    )
                ))

        # Add two-word container/insertion indicators to operation_indicators
        # These were found in the FIRST PASS and should be prioritized
        for pos, (indicator_phrase,
                  indicator_match) in insertion_indicators_in_remaining.items():
            if ' ' in indicator_phrase:  # This is a two-word indicator
                operation_indicators.append((indicator_phrase, indicator_match))
                word_roles.append(WordRole(
                    indicator_phrase, f'{indicator_match.wordplay_type}_indicator', '',
                    'database'
                ))
                # Mark both words as accounted
                for w in indicator_phrase.split():
                    accounted_words.add(w.lower())

        # PRE-PASS: When retrying with skipped indicators, process those words
        # as synonyms FIRST before the main loop. This prevents other words from
        # greedily consuming needed letters and shrinking max_synonym_length below
        # what the skipped indicator's synonym actually needs.
        if skip_indicator_words and needed_letters:
            for word in remaining_words:
                word_lower = word.lower()
                if word_lower not in skip_indicator_words:
                    continue
                if word_lower in accounted_words or word_lower in def_words_lower:
                    continue
                subs = self.db.lookup_substitution(word,
                                                   max_synonym_length=len(needed_letters))
                for sub in subs:
                    sub_letters = sub.letters.upper()
                    temp_needed = list(needed_letters)
                    can_use = True
                    for c in sub_letters:
                        if c in temp_needed:
                            temp_needed.remove(c)
                        else:
                            can_use = False
                            break
                    if can_use:
                        found_substitutions.append((word, sub))
                        word_roles.append(WordRole(
                            word, 'substitution', sub_letters,
                            f'database ({sub.category})'
                        ))
                        accounted_words.add(word_lower)
                        for c in sub_letters:
                            needed_letters = needed_letters.replace(c, '', 1)
                        break  # Stop at first valid substitution for this word

        # Pre-check: does this clue contain container/insertion indicators?
        # Used to decide whether substitutions must be contiguous in answer.
        # In charade clues (no container/insertion), each piece must be a
        # contiguous substring. Container/insertion clues legitimately split words.
        has_container_insertion = bool(insertion_indicators_in_remaining)
        if not has_container_insertion:
            for w in remaining_words:
                # Check if this word has container/insertion as ANY indicator type,
                # not just the top-ranked one. Words like "about" rank as anagram
                # (freq 104) but also have container (freq 0). Since we're past the
                # anagram stage, container is a valid interpretation.
                word_clean = ''.join(c for c in w.lower() if c.isalpha())
                if word_clean:
                    conn = self.db._get_connection()
                    cursor = conn.cursor()
                    cursor.execute("""
                        SELECT 1 FROM indicators
                        WHERE LOWER(word) = ? AND wordplay_type IN ('insertion', 'container')
                        LIMIT 1
                    """, (word_clean,))
                    if cursor.fetchone():
                        has_container_insertion = True
                    break

        # Check for MULTI-WORD indicators (2-5 words) of ANY type
        # Try longest phrases first so "not the first" matches before "not" alone
        # This generalizes the two-word insertion/container check above
        for phrase_len in range(min(5, len(remaining_words)), 1, -1):
            for i in range(len(remaining_words) - phrase_len + 1):
                phrase_words = remaining_words[i:i + phrase_len]
                # Filter out pure-punctuation tokens (e.g., en-dash '–')
                alpha_words = [w for w in phrase_words if norm_letters(w)]
                if len(alpha_words) < 2:
                    continue
                # Skip if any meaningful word already claimed by another multi-word indicator
                if any(w.lower() in words_used_by_two_word_indicators for w in
                       alpha_words):
                    continue
                phrase = ' '.join(norm_letters(w) for w in alpha_words)
                indicator_match = self.db.lookup_indicator(phrase)
                if indicator_match:
                    op_type = indicator_match.wordplay_type
                    # Add to operation_indicators for all relevant types
                    if op_type in ('insertion', 'container', 'deletion', 'reversal',
                                   'hidden', 'parts', 'acrostic'):
                        phrase_display = ' '.join(alpha_words)
                        operation_indicators.append((phrase_display, indicator_match))
                        word_roles.append(WordRole(
                            phrase_display, f'{op_type}_indicator', '', 'database'
                        ))
                        for w in phrase_words:
                            accounted_words.add(w.lower())
                            words_used_by_two_word_indicators.add(w.lower())
                        # Update container/insertion flag if needed
                        if op_type in ('insertion', 'container'):
                            has_container_insertion = True

        for word in remaining_words:
            word_lower = word.lower()
            word_letters = get_letters(word)

            # Skip words already processed by parts indicators
            if word_lower in words_used_by_parts:
                continue

            # Skip words already used in two-word indicators (e.g., "found" and "in" from "found in")
            if word_lower in words_used_by_two_word_indicators:
                continue

            # Check indicators table FIRST for operation type
            # This takes priority over positional_words heuristic
            # BUT skip if this word was flagged for retry as a synonym
            indicator_match = self.db.lookup_indicator(word)
            if indicator_match and not (
                    skip_indicator_words and word_lower in skip_indicator_words):
                op_type = indicator_match.wordplay_type
                if op_type in ('insertion', 'container', 'deletion', 'reversal',
                               'hidden', 'homophone'):
                    operation_indicators.append((word, indicator_match))
                    # Don't account deletion indicators yet - only account them
                    # if they're actually used with an adjacent source word
                    # This allows phrases like "put clothes on" to still match
                    # even if "clothes" is also a deletion indicator
                    if op_type != 'deletion':
                        word_roles.append(WordRole(
                            word, f'{op_type}_indicator', '', 'database'
                        ))
                        accounted_words.add(word_lower)
                    continue

                # Handle single-word parts/acrostic indicators (like "initially", "finally", "almost")
                # Acrostics are initial-letter by definition, no subtype required
                elif op_type in ('parts', 'acrostic'):
                    subtype = indicator_match.subtype or ''

                    # Check for alternating subtypes (e.g., "regularly", "oddly", "evenly")
                    # These extract every other letter, not first/last
                    is_alternating = any(x in subtype.lower() for x in
                                         ['alternate', 'odd', 'even', 'regular'])
                    if is_alternating:
                        try:
                            current_idx = remaining_words.index(word)
                            source_word = None
                            source_letters = None

                            # Try previous word first (e.g., "aunt regularly")
                            if current_idx > 0:
                                cand = remaining_words[current_idx - 1]
                                if cand.lower() not in accounted_words:
                                    source_word = cand
                                    source_letters = get_letters(cand)
                            # Fallback to next word
                            if source_word is None and current_idx + 1 < len(
                                    remaining_words):
                                cand = remaining_words[current_idx + 1]
                                if cand.lower() not in accounted_words:
                                    source_word = cand
                                    source_letters = get_letters(cand)

                            if source_word and source_letters and needed_letters:
                                # Try odd letters (positions 0, 2, 4... = 1st, 3rd, 5th...)
                                odd_letters = ''.join(source_letters[i] for i in
                                                      range(0, len(source_letters), 2))
                                # Try even letters (positions 1, 3, 5... = 2nd, 4th, 6th...)
                                even_letters = ''.join(source_letters[i] for i in
                                                       range(1, len(source_letters), 2))

                                extracted = None
                                extract_desc = None

                                # Check which works with needed_letters
                                temp_needed = list(needed_letters)
                                odd_works = True
                                for c in odd_letters:
                                    if c in temp_needed:
                                        temp_needed.remove(c)
                                    else:
                                        odd_works = False
                                        break

                                temp_needed = list(needed_letters)
                                even_works = True
                                for c in even_letters:
                                    if c in temp_needed:
                                        temp_needed.remove(c)
                                    else:
                                        even_works = False
                                        break

                                if odd_works:
                                    extracted = odd_letters
                                    extract_desc = 'odd letters of'
                                elif even_works:
                                    extracted = even_letters
                                    extract_desc = 'even letters of'

                                if extracted and extract_desc:
                                    # Add indicator
                                    word_roles.append(WordRole(
                                        word, 'alternating_indicator', '',
                                        f"database ({subtype or 'alternating'})"
                                    ))
                                    accounted_words.add(word_lower)

                                    # Add source word
                                    word_roles.append(WordRole(
                                        source_word, 'substitution', extracted,
                                        f"{extract_desc} {source_word}"
                                    ))
                                    accounted_words.add(source_word.lower())

                                    # Add to substitutions
                                    found_substitutions.append((
                                        source_word,
                                        SubstitutionMatch(
                                            word=source_word,
                                            letters=extracted,
                                            category=extract_desc
                                        )
                                    ))

                                    # Update needed letters
                                    for c in extracted:
                                        needed_letters = needed_letters.replace(c, '', 1)
                                    continue
                        except ValueError:
                            pass

                    # Find adjacent word in remaining_words to operate on
                    # For 'last'/'final' indicators, prefer PREVIOUS word (e.g., "Detainee finally")
                    # For 'first'/'initial'/acrostic indicators, prefer NEXT word (e.g., "initially Detainee")
                    try:
                        current_idx = remaining_words.index(word)
                        source_word = None
                        source_letters = None

                        is_last_type = 'last' in subtype.lower() or subtype.lower() == 'final'
                        is_first_type = 'first' in subtype.lower() or subtype.lower() == 'initial' or op_type == 'acrostic'

                        # MULTI-WORD EXTRACTION: Try extracting last/first letters from multiple words
                        # For last letters: "Finally you went" → U + T (subsequent words)
                        # For first letters: "only matter initially" → O + M (previous words)
                        if is_last_type and current_idx + 1 < len(remaining_words):
                            # Try extracting last letters from subsequent unresolved words
                            multi_letters = ''
                            multi_sources = []
                            for j in range(current_idx + 1, len(remaining_words)):
                                src = remaining_words[j]
                                src_letters = get_letters(src)
                                if not src_letters:
                                    continue
                                if src.lower() in accounted_words:
                                    continue
                                last_char = src_letters[-1].upper()
                                if last_char in needed_letters:
                                    # Verify this specific letter is still needed
                                    temp = needed_letters
                                    for c in multi_letters + last_char:
                                        temp = temp.replace(c, '', 1)
                                    if len(temp) < len(needed_letters) - len(
                                            multi_letters):
                                        multi_letters += last_char
                                        multi_sources.append(src)
                                else:
                                    break  # Stop at first word that doesn't contribute

                            if len(multi_letters) >= 2:
                                # Successfully extracted multiple last letters
                                word_roles.append(WordRole(
                                    word, 'parts_indicator', '',
                                    f"database ({subtype})"
                                ))
                                accounted_words.add(word_lower)

                                # Add source words
                                word_roles.append(WordRole(
                                    ' '.join(multi_sources), 'substitution',
                                    multi_letters,
                                    'last letters of'
                                ))
                                for src in multi_sources:
                                    accounted_words.add(src.lower())
                                    words_used_by_parts.add(src.lower())

                                # Add to substitutions
                                found_substitutions.append((
                                    ' '.join(multi_sources),
                                    SubstitutionMatch(
                                        word=' '.join(multi_sources),
                                        letters=multi_letters,
                                        category='last letters of'
                                    )
                                ))

                                # Update needed letters
                                for c in multi_letters:
                                    needed_letters = needed_letters.replace(c, '', 1)
                                continue  # Skip single-word handling

                        elif is_first_type and current_idx > 0:
                            # Try extracting first letters from previous unresolved words
                            # "only matter initially" → O + M
                            multi_letters = ''
                            multi_sources = []
                            # Find consecutive unaccounted words before the indicator
                            start_idx = current_idx - 1
                            while start_idx >= 0:
                                src = remaining_words[start_idx]
                                if src.lower() in accounted_words:
                                    break
                                start_idx -= 1
                            start_idx += 1  # Move back to first unaccounted word

                            # Now iterate forward from start_idx to current_idx
                            for j in range(start_idx, current_idx):
                                src = remaining_words[j]
                                src_letters = get_letters(src)
                                if not src_letters:
                                    continue
                                if src.lower() in accounted_words:
                                    continue
                                first_char = src_letters[0].upper()
                                if first_char in needed_letters:
                                    # Verify this specific letter is still needed
                                    temp = needed_letters
                                    for c in multi_letters + first_char:
                                        temp = temp.replace(c, '', 1)
                                    if len(temp) < len(needed_letters) - len(
                                            multi_letters):
                                        multi_letters += first_char
                                        multi_sources.append(src)
                                else:
                                    break  # Stop at first word that doesn't contribute

                            if len(multi_letters) >= 2:
                                # Successfully extracted multiple first letters
                                word_roles.append(WordRole(
                                    word, 'parts_indicator', '',
                                    f"database ({subtype})"
                                ))
                                accounted_words.add(word_lower)

                                # Add source words
                                word_roles.append(WordRole(
                                    ' '.join(multi_sources), 'substitution',
                                    multi_letters,
                                    'first letters of'
                                ))
                                for src in multi_sources:
                                    accounted_words.add(src.lower())
                                    words_used_by_parts.add(src.lower())

                                # Add to substitutions
                                found_substitutions.append((
                                    ' '.join(multi_sources),
                                    SubstitutionMatch(
                                        word=' '.join(multi_sources),
                                        letters=multi_letters,
                                        category='first letters of'
                                    )
                                ))

                                # Update needed letters
                                for c in multi_letters:
                                    needed_letters = needed_letters.replace(c, '', 1)
                                continue  # Skip single-word handling

                        # SINGLE-WORD EXTRACTION: Build list of candidate source words in preference order
                        candidate_indices = []
                        # Prepositions to skip as source words - they link indicators to actual sources
                        skip_prepositions = {'from', 'of', 'in', 'with', 'by', 'for',
                                             'to', 'at'}
                        if is_last_type:
                            # For 'finally'/'last', try previous word first, then next, then further out
                            if current_idx > 0:
                                candidate_indices.append(current_idx - 1)
                            if current_idx + 1 < len(remaining_words):
                                candidate_indices.append(current_idx + 1)
                            if current_idx + 2 < len(remaining_words):
                                candidate_indices.append(current_idx + 2)
                        else:
                            # For 'initially'/'first'/acrostic, try next word first, then previous
                            if current_idx + 1 < len(remaining_words):
                                candidate_indices.append(current_idx + 1)
                            if current_idx + 2 < len(remaining_words):
                                candidate_indices.append(current_idx + 2)
                            if current_idx > 0:
                                candidate_indices.append(current_idx - 1)

                        # Try each candidate, use first one whose extracted letter is in needed_letters
                        # Skip short words (3 letters or less) as they're likely full substitutes
                        for cand_idx in candidate_indices:
                            cand_word = remaining_words[cand_idx]
                            cand_letters = get_letters(cand_word)
                            if not cand_letters:
                                continue
                            # Skip if already accounted for
                            if cand_word.lower() in accounted_words:
                                continue
                            # Skip prepositions - they link indicators to sources, not sources themselves
                            if cand_word.lower().rstrip(',.;:') in skip_prepositions:
                                continue
                            # Skip very short words - they're likely used as full substitutes, not for extraction
                            if len(cand_letters) <= 3:
                                continue
                            # Check if extracted letter would be useful
                            if is_last_type:
                                test_letter = cand_letters[-1] if cand_letters else ''
                            else:
                                test_letter = cand_letters[0] if cand_letters else ''
                            if test_letter and needed_letters and test_letter in needed_letters:
                                source_word = cand_word
                                source_letters = cand_letters
                                break

                        # Fallback: if no long candidate matched, try short words too
                        if source_word is None and candidate_indices:
                            for cand_idx in candidate_indices:
                                cand_word = remaining_words[cand_idx]
                                cand_letters = get_letters(cand_word)
                                if not cand_letters:
                                    continue
                                if cand_word.lower() in accounted_words:
                                    continue
                                # Skip prepositions
                                if cand_word.lower().rstrip(',.;:') in skip_prepositions:
                                    continue
                                if is_last_type:
                                    test_letter = cand_letters[-1] if cand_letters else ''
                                else:
                                    test_letter = cand_letters[0] if cand_letters else ''
                                if test_letter and needed_letters and test_letter in needed_letters:
                                    source_word = cand_word
                                    source_letters = cand_letters
                                    break

                        # Final fallback: if nothing matched needed_letters, use first available
                        if source_word is None and candidate_indices:
                            for cand_idx in candidate_indices:
                                cand_word = remaining_words[cand_idx]
                                cand_letters = get_letters(cand_word)
                                # Skip prepositions even in final fallback
                                if cand_word.lower().rstrip(',.;:') in skip_prepositions:
                                    continue
                                if cand_letters and cand_word.lower() not in accounted_words:
                                    source_word = cand_word
                                    source_letters = cand_letters
                                    break

                        if source_word and source_letters:

                            # Check if this is a DELETE operation (remove letter, use rest as fodder)
                            if 'delete' in subtype.lower() and source_letters:
                                remaining_fodder = None

                                if 'first' in subtype.lower() or subtype.lower() == 'initial':
                                    remaining_fodder = source_letters[
                                                       1:]  # Remove first letter
                                elif 'last' in subtype.lower() or subtype.lower() == 'final':
                                    remaining_fodder = source_letters[
                                                       :-1]  # Remove last letter
                                elif 'middle' in subtype.lower() or 'heart' in subtype.lower() or 'core' in subtype.lower():
                                    # Remove middle letter(s)
                                    if len(source_letters) >= 3:
                                        mid = len(source_letters) // 2
                                        if len(source_letters) % 2 == 1:
                                            # Odd length: remove single middle letter
                                            remaining_fodder = source_letters[
                                                               :mid] + source_letters[
                                                                       mid + 1:]
                                        else:
                                            # Even length: remove two middle letters
                                            remaining_fodder = source_letters[
                                                               :mid - 1] + source_letters[
                                                                           mid + 1:]
                                elif 'outer' in subtype.lower():
                                    # Remove first AND last letters
                                    if len(source_letters) >= 3:
                                        remaining_fodder = source_letters[1:-1]

                                if remaining_fodder:
                                    # Check if remaining letters contribute to needed_letters
                                    can_use = True
                                    temp_needed = list(
                                        needed_letters) if needed_letters else []
                                    for c in remaining_fodder:
                                        if c in temp_needed:
                                            temp_needed.remove(c)
                                        else:
                                            can_use = False
                                            break

                                    if can_use or not needed_letters:
                                        # Add indicator
                                        subtype_lower = subtype.lower()
                                        if 'last' in subtype_lower or subtype_lower == 'final':
                                            delete_desc = 'minus last letter'
                                        elif 'middle' in subtype_lower or 'heart' in subtype_lower or 'core' in subtype_lower:
                                            delete_desc = 'minus middle letter'
                                        elif 'outer' in subtype_lower:
                                            delete_desc = 'minus outer letters'
                                        else:
                                            delete_desc = 'minus first letter'
                                        word_roles.append(WordRole(
                                            word, 'parts_indicator', '',
                                            f"database ({subtype})"
                                        ))
                                        accounted_words.add(word_lower)

                                        # Add source word as truncated fodder
                                        word_roles.append(WordRole(
                                            source_word, 'fodder', remaining_fodder,
                                            delete_desc
                                        ))
                                        accounted_words.add(source_word.lower())
                                        words_used_by_parts.add(source_word.lower())

                                        # Add to additional_fodder for formula building
                                        additional_fodder.append(
                                            (source_word, remaining_fodder))

                                        # Update needed letters
                                        if needed_letters:
                                            for c in remaining_fodder:
                                                needed_letters = needed_letters.replace(c,
                                                                                        '',
                                                                                        1)
                                        continue

                            # EXTRACT operation (extract letter as substitution)
                            elif needed_letters:
                                extracted_letter = None
                                extracted_letters = None  # For outer/extremes (multiple letters)

                                # Check 'outer'/'extremes' - extract BOTH first and last letters
                                if (any(x in subtype.lower() for x in
                                        ['outer', 'extreme', 'border', 'edge'])
                                        and source_letters and len(source_letters) >= 2):
                                    extracted_letters = source_letters[0] + \
                                                        source_letters[-1]
                                # Check 'last'/'final' FIRST so subtype takes precedence over acrostic default
                                elif (
                                        'last' in subtype.lower() or subtype.lower() == 'final') and source_letters:
                                    extracted_letter = source_letters[-1]
                                elif (
                                        'first' in subtype.lower() or subtype.lower() == 'initial'
                                        or op_type == 'acrostic') and source_letters:
                                    extracted_letter = source_letters[0]

                                # Handle outer/extremes case (two letters)
                                if extracted_letters:
                                    # Check both letters are in needed_letters
                                    temp_needed = list(needed_letters)
                                    can_use = True
                                    for c in extracted_letters:
                                        if c in temp_needed:
                                            temp_needed.remove(c)
                                        else:
                                            can_use = False
                                            break

                                    if can_use:
                                        # Add indicator
                                        word_roles.append(WordRole(
                                            word, 'parts_indicator', '',
                                            f"database ({subtype})"
                                        ))
                                        accounted_words.add(word_lower)

                                        # Add source word
                                        word_roles.append(WordRole(
                                            source_word, 'substitution',
                                            extracted_letters,
                                            'first and last letters of'
                                        ))
                                        accounted_words.add(source_word.lower())
                                        words_used_by_parts.add(source_word.lower())

                                        # Add to substitutions
                                        found_substitutions.append((
                                            source_word,
                                            SubstitutionMatch(
                                                word=source_word,
                                                letters=extracted_letters,
                                                category='first and last letters of'
                                            )
                                        ))

                                        # Update needed letters
                                        for c in extracted_letters:
                                            needed_letters = needed_letters.replace(c, '',
                                                                                    1)
                                        continue

                                if extracted_letter and extracted_letter in needed_letters:
                                    # Add indicator
                                    word_roles.append(WordRole(
                                        word, 'parts_indicator', '',
                                        f"database ({subtype})"
                                    ))
                                    accounted_words.add(word_lower)

                                    # Add source word with readable description
                                    source_desc = 'last letter of' if (
                                            'last' in subtype.lower() or subtype.lower() == 'final') else 'first letter of'
                                    word_roles.append(WordRole(
                                        source_word, 'substitution', extracted_letter,
                                        source_desc
                                    ))
                                    accounted_words.add(source_word.lower())
                                    words_used_by_parts.add(source_word.lower())

                                    # Add to substitutions
                                    found_substitutions.append((
                                        source_word,
                                        SubstitutionMatch(
                                            word=source_word,
                                            letters=extracted_letter,
                                            category=source_desc
                                        )
                                    ))

                                    # Update needed letters
                                    needed_letters = needed_letters.replace(
                                        extracted_letter, '', 1)
                                    continue
                    except ValueError:
                        pass  # word not in remaining_words

                # Handle alternating letter indicators (like "regularly", "oddly", "evenly")
                # Extracts every other letter from adjacent word
                elif op_type in ('alternating', 'alternate'):
                    subtype = indicator_match.subtype or ''
                    try:
                        current_idx = remaining_words.index(word)
                        source_word = None
                        source_letters = None

                        # Try previous word first (e.g., "Diet regularly")
                        if current_idx > 0:
                            source_word = remaining_words[current_idx - 1]
                            source_letters = get_letters(source_word)
                        # Fallback to next word
                        if source_word is None and current_idx + 1 < len(remaining_words):
                            source_word = remaining_words[current_idx + 1]
                            source_letters = get_letters(source_word)

                        if source_word and source_letters and needed_letters:
                            # Try odd letters (positions 0, 2, 4... in 0-indexed = 1st, 3rd, 5th...)
                            odd_letters = ''.join(source_letters[i] for i in
                                                  range(0, len(source_letters), 2))
                            # Try even letters (positions 1, 3, 5... in 0-indexed = 2nd, 4th, 6th...)
                            even_letters = ''.join(source_letters[i] for i in
                                                   range(1, len(source_letters), 2))

                            extracted = None
                            extract_desc = None

                            # If subtype specifies, use that; otherwise try both
                            if 'odd' in subtype.lower():
                                if all(c in needed_letters for c in odd_letters):
                                    extracted = odd_letters
                                    extract_desc = 'odd letters of'
                            elif 'even' in subtype.lower():
                                if all(c in needed_letters for c in even_letters):
                                    extracted = even_letters
                                    extract_desc = 'even letters of'
                            else:
                                # Try both, prefer whichever matches needed_letters
                                temp_needed = list(needed_letters)
                                odd_works = True
                                for c in odd_letters:
                                    if c in temp_needed:
                                        temp_needed.remove(c)
                                    else:
                                        odd_works = False
                                        break

                                temp_needed = list(needed_letters)
                                even_works = True
                                for c in even_letters:
                                    if c in temp_needed:
                                        temp_needed.remove(c)
                                    else:
                                        even_works = False
                                        break

                                if odd_works:
                                    extracted = odd_letters
                                    extract_desc = 'odd letters of'
                                elif even_works:
                                    extracted = even_letters
                                    extract_desc = 'even letters of'

                            if extracted and extract_desc:
                                # Add indicator
                                word_roles.append(WordRole(
                                    word, 'alternating_indicator', '',
                                    f"database ({subtype or 'alternating'})"
                                ))
                                accounted_words.add(word_lower)

                                # Add source word
                                word_roles.append(WordRole(
                                    source_word, 'substitution', extracted,
                                    f"{extract_desc} {source_word}"
                                ))
                                accounted_words.add(source_word.lower())

                                # Add to substitutions
                                found_substitutions.append((
                                    source_word,
                                    SubstitutionMatch(
                                        word=source_word,
                                        letters=extracted,
                                        category=extract_desc
                                    )
                                ))

                                # Update needed letters
                                for c in extracted:
                                    needed_letters = needed_letters.replace(c, '', 1)
                                continue
                    except ValueError:
                        pass  # word not in remaining_words


            # Check if this word's letters are contained in needed letters (partial fodder)
            # This handles cases like "ill" providing ILL when we need WILL
            if word_letters and needed_letters:
                # Check if all letters in this word are present in needed_letters
                temp_needed = list(needed_letters)
                can_use = True
                for c in word_letters:
                    if c in temp_needed:
                        temp_needed.remove(c)
                    else:
                        can_use = False
                        break

                if can_use:
                    # Check if this word is adjacent to an insertion indicator
                    # If so, label as insertion_material, not fodder
                    is_insertion_material = False
                    if insertion_indicators_in_remaining:
                        # Find this word's position in remaining_words
                        try:
                            word_idx_in_remaining = remaining_words.index(word)
                            # Check if adjacent to any insertion indicator
                            for ins_idx in insertion_indicators_in_remaining:
                                if abs(word_idx_in_remaining - ins_idx) == 1:
                                    is_insertion_material = True
                                    break
                        except ValueError:
                            pass  # Word not in remaining_words

                    if is_insertion_material:
                        word_roles.append(WordRole(
                            word, 'insertion_material', word_letters, 'compound_analysis'
                        ))
                    else:
                        # In charade context (no anagram), single letters are
                        # direct contributions, not anagram fodder
                        if not anagram_letters and len(word_letters) == 1:
                            found_substitutions.append((word, SubstitutionMatch(
                                word=word, letters=word_letters, category='single_letter'
                            )))
                            word_roles.append(WordRole(
                                word, 'substitution', word_letters,
                                'database (single_letter)'
                            ))
                        else:
                            additional_fodder.append((word, word_letters))
                            word_roles.append(WordRole(
                                word, 'fodder', word_letters, 'compound_analysis'
                            ))
                    accounted_words.add(word_lower)
                    # Update needed_letters by removing the used letters
                    for c in word_letters:
                        needed_letters = needed_letters.replace(c, '', 1)
                    continue

            # Check if this word might be a parts indicator followed by "of"
            # Skip substitution lookup for words like "case", "edges", "borders", etc.
            parts_indicator_words = {'case', 'cases', 'edges', 'edge', 'borders',
                                     'border',
                                     'extremes', 'extreme', 'limits', 'outsides',
                                     'outside',
                                     'ends', 'head', 'tail', 'front', 'back', 'top',
                                     'bottom'}

            # Find position of current word in remaining_words
            word_idx = None
            for idx, w in enumerate(remaining_words):
                if w.lower() == word_lower:
                    word_idx = idx
                    break

            # If this word is a parts indicator and next word is "of", skip synonym lookup
            if word_lower in parts_indicator_words:
                if word_idx is not None and word_idx + 1 < len(remaining_words):
                    next_word = remaining_words[word_idx + 1].lower()
                    if next_word == 'of':
                        # This is likely "case of X" - skip synonym lookup, let parts handling catch it
                        # But check if we need the outer letters
                        if word_idx + 2 < len(remaining_words):
                            source_word = remaining_words[word_idx + 2]
                            source_letters = get_letters(source_word)
                            if len(source_letters) >= 2:
                                outer_letters = (source_letters[0] + source_letters[
                                    -1]).upper()
                                # Check if both outer letters are needed
                                temp_needed = needed_letters
                                can_use = True
                                for c in outer_letters:
                                    if c in temp_needed:
                                        temp_needed = temp_needed.replace(c, '', 1)
                                    else:
                                        can_use = False
                                        break

                                if can_use:
                                    word_roles.append(WordRole(
                                        word, 'parts_indicator', '',
                                        f"outer letters indicator"
                                    ))
                                    word_roles.append(WordRole(
                                        source_word, 'fodder', outer_letters,
                                        f"outer letters of {source_word}"
                                    ))
                                    accounted_words.add(word_lower)
                                    accounted_words.add('of')
                                    accounted_words.add(source_word.lower())
                                    needed_letters = temp_needed
                                    continue

            # Check for substitution
            # Pass needed_letters length so we can find longer synonyms when needed
            # When deletion indicators are present, allow longer synonyms (they'll be shortened by deletion)
            # Strip possessives for lookup (editor's -> editor)
            has_deletion_indicator = any(
                i.wordplay_type == 'deletion' for w, i in operation_indicators)
            # Allow 2 extra letters if deletion is involved (e.g., YEARNS -> YARNS)
            synonym_length_limit = len(
                needed_letters) + 2 if has_deletion_indicator else len(needed_letters)
            word_for_lookup = re.sub(r"[''\u2019]s$", "", word, flags=re.IGNORECASE)
            subs = self.db.lookup_substitution(word_for_lookup,
                                               max_synonym_length=synonym_length_limit)
            # Also try the full word (with apostrophe removed but 's kept) for cases like "dad's" -> "dads" -> PAS
            word_full = ''.join(c for c in word if c.isalpha())
            if word_full.lower() != word_for_lookup.lower():
                subs_full = self.db.lookup_substitution(word_full,
                                                        max_synonym_length=synonym_length_limit)
                # Add any new substitutions not already found
                existing_letters = {s.letters.upper() for s in subs}
                for s in subs_full:
                    if s.letters.upper() not in existing_letters:
                        subs.append(s)
            for sub in subs:
                # Check if this substitution provides letters we need (with correct counts)
                sub_letters = sub.letters.upper()
                temp_needed = list(needed_letters)
                can_use = True
                excess_letters = []
                deletion_source_word = None
                for c in sub_letters:
                    if c in temp_needed:
                        temp_needed.remove(c)
                    else:
                        # Letter not needed - could be deleted if deletion indicator present
                        if has_deletion_indicator:
                            excess_letters.append(c)
                        else:
                            can_use = False
                            break

                # If we have excess letters and deletion indicators, check if excess can be deleted
                if can_use and excess_letters and has_deletion_indicator:
                    # Look for unresolved words that could provide the excess letters for deletion
                    excess_str = ''.join(sorted(excess_letters))
                    can_delete_excess = False

                    # Check if any unresolved word provides exactly the excess letters
                    for other_word in remaining_words:
                        if other_word.lower() in accounted_words:
                            continue
                        if other_word.lower() == word_lower:
                            continue
                        # Check single-letter substitutions
                        other_subs = self.db.lookup_substitution(
                            other_word.lower().rstrip(',.;:'), max_synonym_length=3)
                        for other_sub in other_subs:
                            if other_sub.letters.upper() == excess_str:
                                can_delete_excess = True
                                deletion_source_word = other_word
                                break
                        if can_delete_excess:
                            break

                    if not can_delete_excess:
                        can_use = False

                if can_use:
                    # In charade clues (no container/insertion), reject substitutions
                    # whose letters are scattered in the answer (e.g., GET in DETERGENT).
                    # Each charade piece must be a contiguous substring of the answer.
                    # BUT: skip this check if we have excess letters that will be deleted
                    if not has_container_insertion and len(
                            sub_letters) > 1 and not excess_letters:
                        if sub_letters not in answer_upper:
                            continue
                    # Skip if this substitution is in the exclusion set (backtracking)
                    if skip_substitutions and (word_lower,
                                               sub_letters) in skip_substitutions:
                        continue
                    found_substitutions.append((word, sub))
                    word_roles.append(WordRole(
                        word, 'substitution', sub_letters,
                        f'database ({sub.category})'
                    ))
                    accounted_words.add(word_lower)
                    # Update needed letters
                    for c in sub_letters:
                        needed_letters = needed_letters.replace(c, '', 1)

                    # If we used deletion to handle excess letters, account for the deletion source
                    if excess_letters and deletion_source_word:
                        excess_str = ''.join(excess_letters)
                        word_roles.append(WordRole(
                            deletion_source_word, 'deletion_target', excess_str,
                            f'deleted from {word} → {sub_letters}'
                        ))
                        accounted_words.add(deletion_source_word.lower())

                    # Check for adjacent positional/insertion indicator
                    # This claims words like "in" as positional indicators for the substitution
                    # preventing them from being incorrectly used as anagram fodder
                    try:
                        sub_word_idx = remaining_words.index(word)
                        adjacent_indices = []
                        if sub_word_idx > 0:
                            adjacent_indices.append(sub_word_idx - 1)
                        if sub_word_idx < len(remaining_words) - 1:
                            adjacent_indices.append(sub_word_idx + 1)

                        for adj_idx in adjacent_indices:
                            adj_word = remaining_words[adj_idx]
                            adj_lower = adj_word.lower()

                            if adj_lower in accounted_words:
                                continue

                            adj_indicator = self.db.lookup_indicator(adj_word)
                            if adj_indicator and adj_indicator.wordplay_type in (
                                    'insertion', 'container'):
                                positional_indicators.append(adj_word)
                                word_roles.append(WordRole(
                                    adj_word, 'positional_indicator', '',
                                    f'insertion indicator for {word}→{sub_letters}'
                                ))
                                accounted_words.add(adj_lower)
                                break
                    except ValueError:
                        pass

                    break  # Stop at first valid substitution for this word

        # Check for MULTI-WORD phrase substitutions (e.g., "the french" -> LA,
        # "group of animals" -> HERD, "fend off" -> DETER).
        # Tries all contiguous phrases up to 10 words.
        # If a phrase word was provisionally claimed by a single-word substitution,
        # override it if the phrase provides more letters.
        if needed_letters:
            max_phrase_len = min(len(remaining_words), 10)
            for phrase_len in range(2, max_phrase_len + 1):
                for i in range(len(remaining_words) - phrase_len + 1):
                    phrase_words = remaining_words[i:i + phrase_len]
                    # Strip possessives from phrase words (guy's -> guy)
                    phrase_words_lower = [
                        re.sub(r"[''\u2019]s$", "", w.lower(), flags=re.IGNORECASE) for w
                        in phrase_words]

                    # Check which phrase words are already claimed
                    blocked_by_non_sub = False
                    overridable_subs = []  # (word, sub, WordRole) that could be undone
                    for pw in phrase_words_lower:
                        if pw in accounted_words:
                            # Is it claimed by a single-word substitution? If so, it's overridable
                            matching_sub = None
                            for fs_word, fs_sub in found_substitutions:
                                if fs_word.lower() == pw:
                                    matching_sub = (fs_word, fs_sub)
                                    break
                            if matching_sub:
                                overridable_subs.append(matching_sub)
                            else:
                                # Claimed by indicator, definition, etc - can't override
                                blocked_by_non_sub = True
                                break

                    if blocked_by_non_sub:
                        continue

                    # All unclaimed, or all blocked words are overridable subs
                    # Build the phrase and look it up
                    phrase = ' '.join(phrase_words_lower)
                    phrase_subs = self.db.lookup_phrase_substitution(phrase)

                    for sub in phrase_subs:
                        sub_letters = sub.letters.upper()

                        # Skip if this phrase substitution is in the exclusion set (backtracking)
                        if skip_substitutions and (phrase,
                                                   sub_letters) in skip_substitutions:
                            continue

                        # Calculate what needed_letters would be if we undo overridable subs
                        test_needed = needed_letters
                        overridable_letter_count = 0
                        for ov_word, ov_sub in overridable_subs:
                            ov_letters = ov_sub.letters.upper()
                            overridable_letter_count += len(ov_letters)
                            test_needed += ov_letters  # Give back the letters

                        # Only override if phrase provides strictly more letters
                        if overridable_subs and len(
                                sub_letters) <= overridable_letter_count:
                            continue

                        # Check if phrase provides letters we need (after giving back overridden ones)
                        temp_needed = list(test_needed)
                        can_use = True
                        for c in sub_letters:
                            if c in temp_needed:
                                temp_needed.remove(c)
                            else:
                                can_use = False
                                break
                        if can_use:
                            # Undo any overridable single-word substitutions
                            for ov_word, ov_sub in overridable_subs:
                                found_substitutions.remove((ov_word, ov_sub))
                                # Remove from word_roles
                                word_roles[:] = [wr for wr in word_roles
                                                 if
                                                 not (wr.word.lower() == ov_word.lower()
                                                      and wr.role == 'substitution')]
                                accounted_words.discard(ov_word.lower())

                            # Apply phrase substitution
                            needed_letters = ''.join(temp_needed)
                            found_substitutions.append((phrase, sub))
                            word_roles.append(WordRole(
                                phrase, 'substitution', sub_letters,
                                f'database phrase ({sub.category})'
                            ))
                            # Also add word_roles for each word in the phrase
                            # so they show correctly in the breakdown
                            for i, w in enumerate(phrase_words):
                                if i == 0:
                                    # First word gets the substitution attribution
                                    word_roles.append(WordRole(
                                        w, 'substitution', sub_letters,
                                        f'phrase "{phrase}" → {sub_letters}'
                                    ))
                                else:
                                    # Other words are marked as part of the phrase
                                    word_roles.append(WordRole(
                                        w, 'substitution', '',
                                        f'(part of "{phrase}")'
                                    ))
                                accounted_words.add(w.lower())
                            break  # Stop at first valid substitution for this phrase

        # Build compound solution
        # Collect unresolved words for self-learning
        unresolved_words = [w for w in remaining_words
                            if
                            w.lower() not in accounted_words and w.lower() not in def_words_lower]

        # Check for orphaned deletion indicators (indicators without source words)
        # This handles "step half missing" where "half" and "missing" are deletion indicators
        # but "step" hasn't been connected as the deletion source
        deletion_indicators_found = [(w, i) for w, i in operation_indicators
                                     if i.wordplay_type == 'deletion']

        # Also consider link words as potential deletion fodder (e.g., "the endless" -> TH)
        potential_deletion_fodder = list(unresolved_words)
        if deletion_indicators_found and needed_letters:
            for w in remaining_words:
                w_lower = w.lower()
                w_norm = norm_letters(w)
                if w_norm and w_norm in self.link_words and w_lower not in [x.lower() for
                                                                            x in
                                                                            potential_deletion_fodder]:
                    potential_deletion_fodder.append(w)

        if deletion_indicators_found and needed_letters and potential_deletion_fodder:
            # We have deletion indicators but still need letters
            # Check unresolved words to see if applying deletion provides needed letters

            # Find positions of deletion indicators in clue
            deletion_ind_positions = {}
            for del_word, del_ind in deletion_indicators_found:
                # For multi-word indicators, find position of first word in clue
                first_word = del_word.split()[0]
                first_word_norm = norm_letters(first_word)
                if first_word_norm:
                    for idx, cw in enumerate(clue_words):
                        if norm_letters(cw) == first_word_norm:
                            deletion_ind_positions[idx] = (del_word, del_ind)
                            break

            for unres_word in list(
                    potential_deletion_fodder):  # Copy list since we modify it
                unres_lower = unres_word.lower()
                unres_letters = get_letters(unres_word)

                if len(unres_letters) < 2:
                    continue

                # Find position of unresolved word
                unres_idx = None
                for idx, cw in enumerate(clue_words):
                    if norm_letters(cw) == norm_letters(unres_word):
                        unres_idx = idx
                        break

                if unres_idx is None:
                    continue

                # Check if adjacent to any deletion indicator (within 2 positions)
                adjacent_deletion = None
                for del_idx, (del_word, del_ind) in deletion_ind_positions.items():
                    if abs(unres_idx - del_idx) <= 2:
                        adjacent_deletion = (del_word, del_ind)
                        break

                if not adjacent_deletion:
                    continue

                del_word, del_ind = adjacent_deletion
                del_subtype = (del_ind.subtype or '').lower()
                del_word_norm = norm_letters(del_word)

                # Determine what part to take based on deletion indicator
                candidate_letters = None
                deletion_desc = ''

                # "half" indicators - take first or second half
                # Only applies to words with even number of letters
                if ('half' in del_word_norm or 'half' in del_subtype) and len(
                        unres_letters) % 2 == 0:
                    half_len = len(unres_letters) // 2
                    first_half = unres_letters[:half_len].upper()
                    second_half = unres_letters[half_len:].upper()

                    # Check which half provides needed letters
                    for half, desc in [(first_half, 'first half'),
                                       (second_half, 'second half')]:
                        temp_needed = needed_letters
                        can_use = True
                        for c in half:
                            if c in temp_needed:
                                temp_needed = temp_needed.replace(c, '', 1)
                            else:
                                can_use = False
                                break
                        if can_use:
                            candidate_letters = half
                            deletion_desc = desc
                            break

                # "first" / "head" / "opener" indicators - missing first means take rest
                elif any(x in del_subtype for x in
                         ['first', 'head', 'open', 'start', 'initial']):
                    remaining = unres_letters[1:].upper()
                    temp_needed = needed_letters
                    can_use = True
                    for c in remaining:
                        if c in temp_needed:
                            temp_needed = temp_needed.replace(c, '', 1)
                        else:
                            can_use = False
                            break
                    if can_use:
                        candidate_letters = remaining
                        deletion_desc = 'minus first letter'

                # "last" / "tail" / "end" indicators - missing last means take rest
                elif any(x in del_subtype for x in ['last', 'tail', 'final']) or (
                        del_subtype == 'end'):
                    remaining = unres_letters[:-1].upper()
                    temp_needed = needed_letters
                    can_use = True
                    for c in remaining:
                        if c in temp_needed:
                            temp_needed = temp_needed.replace(c, '', 1)
                        else:
                            can_use = False
                            break
                    if can_use:
                        candidate_letters = remaining
                        deletion_desc = 'minus last letter'

                # "ends" / "bare" / "exposed" indicators - remove BOTH first AND last letters
                elif 'ends' in del_subtype or 'bare' in del_word_norm:
                    if len(unres_letters) > 2:
                        remaining = unres_letters[1:-1].upper()  # Remove first and last
                        temp_needed = needed_letters
                        can_use = True
                        for c in remaining:
                            if c in temp_needed:
                                temp_needed = temp_needed.replace(c, '', 1)
                            else:
                                can_use = False
                                break
                        if can_use:
                            candidate_letters = remaining
                            deletion_desc = 'minus ends'

                # "middle" / "heart" / "core" / "heartless" / "gutless" indicators - remove middle letter(s)
                elif any(x in del_subtype for x in
                         ['middle', 'heart', 'core', 'gut', 'centre', 'center']):
                    if len(unres_letters) >= 3:
                        mid = len(unres_letters) // 2
                        if len(unres_letters) % 2 == 1:
                            # Odd length: remove single middle letter
                            remaining = (unres_letters[:mid] + unres_letters[
                                                               mid + 1:]).upper()
                        else:
                            # Even length: remove two middle letters
                            remaining = (unres_letters[:mid - 1] + unres_letters[
                                                                   mid + 1:]).upper()
                        temp_needed = needed_letters
                        can_use = True
                        for c in remaining:
                            if c in temp_needed:
                                temp_needed = temp_needed.replace(c, '', 1)
                            else:
                                can_use = False
                                break
                        if can_use:
                            candidate_letters = remaining
                            deletion_desc = 'minus middle letter'

                if candidate_letters:
                    # Found a match! Add this word as deletion source
                    word_roles.append(WordRole(
                        unres_word, 'deletion_source', candidate_letters,
                        f'{deletion_desc} of {unres_word}'
                    ))
                    accounted_words.add(unres_lower)

                    # Also add the deletion indicator to word_roles and account it now
                    word_roles.append(WordRole(
                        del_word, 'deletion_indicator', '', 'database'
                    ))
                    accounted_words.add(del_word.lower())

                    # Update needed_letters
                    for c in candidate_letters:
                        needed_letters = needed_letters.replace(c, '', 1)

                    # Remove from unresolved and potential_deletion_fodder
                    unresolved_words = [w for w in unresolved_words
                                        if norm_letters(w) != norm_letters(unres_word)]
                    potential_deletion_fodder = [w for w in potential_deletion_fodder
                                                 if norm_letters(w) != norm_letters(
                            unres_word)]
                    break  # Only process one deletion source

        # ================================================================
        # HOMOPHONE HANDLING
        # ================================================================
        # Check for homophone indicators and their adjacent source words
        # Example: "sounds like acts" -> AX (homophone of ACTS)
        homophone_indicators_found = [(w, i) for w, i in operation_indicators
                                      if i.wordplay_type == 'homophone']

        if homophone_indicators_found and needed_letters and unresolved_words:
            # Find positions of homophone indicators in clue
            homophone_ind_positions = {}
            for hom_word, hom_ind in homophone_indicators_found:
                first_word = hom_word.split()[0]
                first_word_norm = norm_letters(first_word)
                if first_word_norm:
                    for idx, cw in enumerate(clue_words):
                        if norm_letters(cw) == first_word_norm:
                            homophone_ind_positions[idx] = (hom_word, hom_ind)
                            break

            for unres_word in list(unresolved_words):
                unres_lower = unres_word.lower()
                unres_letters = get_letters(unres_word)

                if not unres_letters:
                    continue

                # Find position of unresolved word
                unres_idx = None
                for idx, cw in enumerate(clue_words):
                    if norm_letters(cw) == norm_letters(unres_word):
                        unres_idx = idx
                        break

                if unres_idx is None:
                    continue

                # Check if adjacent to any homophone indicator (within 2 positions)
                adjacent_homophone = None
                for hom_idx, (hom_word, hom_ind) in homophone_ind_positions.items():
                    if abs(unres_idx - hom_idx) <= 2:
                        adjacent_homophone = (hom_word, hom_ind)
                        break

                if not adjacent_homophone:
                    continue

                hom_word, hom_ind = adjacent_homophone

                # Strategy 1: Direct homophone lookup
                # e.g., "right" → RITE (if "right" is the source word)
                homophones = self.db.lookup_homophone(unres_word)

                homophone_found = False
                for homophone in homophones:
                    homophone_letters = get_letters(homophone)

                    # Check if this homophone provides needed letters
                    temp_needed = needed_letters
                    can_use = True
                    for c in homophone_letters:
                        if c in temp_needed:
                            temp_needed = temp_needed.replace(c, '', 1)
                        else:
                            can_use = False
                            break

                    if can_use:
                        # Found a match! Add this as homophone substitution
                        word_roles.append(WordRole(
                            unres_word, 'substitution', homophone_letters,
                            f'sounds like {unres_word} = {homophone}'
                        ))
                        accounted_words.add(unres_lower)

                        # Account for the homophone indicator
                        if hom_word.lower() not in accounted_words:
                            word_roles.append(WordRole(
                                hom_word, 'homophone_indicator', '', 'database'
                            ))
                            accounted_words.add(hom_word.lower())

                        # Add to substitutions for formula building
                        found_substitutions.append((
                            unres_word,
                            SubstitutionMatch(
                                word=unres_word,
                                letters=homophone_letters,
                                category='homophone',
                                notes=f'sounds like {homophone}'
                            )
                        ))

                        # Update needed_letters
                        for c in homophone_letters:
                            needed_letters = needed_letters.replace(c, '', 1)

                        # Remove from unresolved
                        unresolved_words = [w for w in unresolved_words
                                            if
                                            norm_letters(w) != norm_letters(unres_word)]
                        homophone_found = True
                        break  # Found a homophone match, stop trying others

                if homophone_found:
                    continue  # Move to next unresolved word

                # Strategy 2: Synonym → Homophone
                # e.g., "creature" → MOOSE (synonym) → MOUSSE (homophone)
                # e.g., "beautiful" → FAIR (synonym) → FARE (homophone)
                synonyms = self.db.lookup_substitution(unres_word, max_synonym_length=12)

                for syn_match in synonyms:
                    syn_word = syn_match.letters.lower()
                    syn_homophones = self.db.lookup_homophone(syn_word)

                    for homophone in syn_homophones:
                        homophone_letters = get_letters(homophone)

                        # Check if this homophone provides needed letters
                        temp_needed = needed_letters
                        can_use = True
                        for c in homophone_letters:
                            if c in temp_needed:
                                temp_needed = temp_needed.replace(c, '', 1)
                            else:
                                can_use = False
                                break

                        if can_use:
                            # Found a match via synonym!
                            word_roles.append(WordRole(
                                unres_word, 'substitution', homophone_letters,
                                f'{unres_word} = {syn_word} sounds like {homophone}'
                            ))
                            accounted_words.add(unres_lower)

                            # Account for the homophone indicator
                            if hom_word.lower() not in accounted_words:
                                word_roles.append(WordRole(
                                    hom_word, 'homophone_indicator', '', 'database'
                                ))
                                accounted_words.add(hom_word.lower())

                            # Add to substitutions for formula building
                            found_substitutions.append((
                                unres_word,
                                SubstitutionMatch(
                                    word=unres_word,
                                    letters=homophone_letters,
                                    category='homophone',
                                    notes=f'{syn_word} sounds like {homophone}'
                                )
                            ))

                            # Update needed_letters
                            for c in homophone_letters:
                                needed_letters = needed_letters.replace(c, '', 1)

                            # Remove from unresolved
                            unresolved_words = [w for w in unresolved_words
                                                if norm_letters(w) != norm_letters(
                                    unres_word)]
                            homophone_found = True
                            break  # Found a homophone match

                    if homophone_found:
                        break  # Stop trying more synonyms

                # If we found a homophone match for this word, continue to next unresolved
                if unres_lower in accounted_words:
                    continue

        # ================================================================
        # REVERSE HOMOPHONE LOOKUP (Strategy 3)
        # ================================================================
        # If we have a homophone indicator but still need letters:
        # 1. Look up homophones OF THE ANSWER
        # 2. Check if any unresolved word has a synonym matching that homophone
        #
        # Example: "Ceremony that's appropriate to be heard" = RITE
        # - Homophones of RITE = [RIGHT, WRITE]
        # - Does "appropriate" have synonym RIGHT? Yes!
        # - So: appropriate = RIGHT sounds like RITE

        if homophone_indicators_found and needed_letters and unresolved_words:
            # Get homophones of the answer
            answer_homophones = self.db.lookup_homophone(answer_upper.lower())

            if answer_homophones:
                # Get the first homophone indicator for attribution
                hom_word, hom_ind = homophone_indicators_found[0]

                for answer_homophone in answer_homophones:
                    answer_homophone_upper = answer_homophone.upper()
                    answer_homophone_letters = get_letters(answer_homophone)

                    # Check if this homophone matches all needed letters
                    if sorted(answer_homophone_letters) != sorted(needed_letters):
                        continue

                    # Now check if any unresolved word has this as a synonym
                    for unres_word in list(unresolved_words):
                        unres_lower = unres_word.lower()

                        # Look up synonyms of the unresolved word
                        synonyms = self.db.lookup_substitution(unres_word,
                                                               max_synonym_length=len(
                                                                   answer_homophone) + 2)

                        for syn_match in synonyms:
                            syn_letters = syn_match.letters.upper()

                            # Check if this synonym matches the answer's homophone
                            if syn_letters == answer_homophone_upper:
                                # Found it! The unresolved word = synonym, which sounds like answer
                                word_roles.append(WordRole(
                                    unres_word, 'substitution', answer_upper,
                                    f'{unres_word} = {syn_letters} sounds like {answer_upper}'
                                ))
                                accounted_words.add(unres_lower)

                                # Account for the homophone indicator
                                if hom_word.lower() not in accounted_words:
                                    word_roles.append(WordRole(
                                        hom_word, 'homophone_indicator', '', 'database'
                                    ))
                                    accounted_words.add(hom_word.lower())

                                # Add to substitutions for formula building
                                found_substitutions.append((
                                    unres_word,
                                    SubstitutionMatch(
                                        word=unres_word,
                                        letters=answer_upper,
                                        category='homophone',
                                        notes=f'{syn_letters} sounds like {answer_upper}'
                                    )
                                ))

                                # Clear needed_letters since we explained the whole answer
                                needed_letters = ''

                                # Remove from unresolved
                                unresolved_words = [w for w in unresolved_words
                                                    if norm_letters(w) != norm_letters(
                                        unres_word)]
                                break

                        if not needed_letters:
                            break  # Found the explanation

                    if not needed_letters:
                        break  # Found the explanation

        # Reorder needed_letters to match their position in the answer.
        # replace(c,'',1) removes letters left-to-right which can jumble
        # the remainder (e.g. DRAMA minus A gives DRMA instead of DRAM).
        # Restore answer ordering so partial results are human-readable.
        if needed_letters and len(needed_letters) <= len(answer_upper):
            for start in range(len(answer_upper) - len(needed_letters) + 1):
                substr = answer_upper[start:start + len(needed_letters)]
                if sorted(substr) == sorted(needed_letters):
                    needed_letters = substr
                    break

        # FALLBACK: Exact-match synonym lookup for unresolved words.
        # The main loop (line ~2091) rejects synonyms whose letters aren't
        # contiguous in the answer (charade rule). But if a synonym's letters
        # are an EXACT match for ALL remaining needed_letters, there is zero
        # ambiguity — it must be the explanation. Example: CHALLENGE clue where
        # "argue" -> ALLEGE exactly matches needed letters ALLEGE after CH has
        # been accounted for. Contiguity is irrelevant here.
        # Also tries multi-word phrases (e.g., "fend off" -> DETER).
        if needed_letters and unresolved_words:
            needed_sorted = sorted(needed_letters.upper())
            unresolved_lower = {w.lower() for w in unresolved_words}
            found_fallback = False

            # Try phrases from 1 word up to all remaining words, using
            # remaining_words (clue-ordered) so multi-word phrases are
            # genuinely contiguous in the clue.
            max_phrase = min(len(remaining_words), 10)
            for phrase_len in range(1, max_phrase + 1):
                if found_fallback:
                    break
                for i in range(len(remaining_words) - phrase_len + 1):
                    window = remaining_words[i:i + phrase_len]
                    # Every word in the window must be unresolved
                    if not all(w.lower() in unresolved_lower for w in window):
                        continue

                    # Skip if any word in window is part of a multi-word indicator
                    # (e.g., don't match "the radio" when "on the radio" is a 3-word indicator)
                    if any(w.lower() in words_used_by_two_word_indicators for w in
                           window):
                        continue

                    phrase = ' '.join(w.lower() for w in window)
                    word_clean = ''.join(c for c in phrase if c.isalpha() or c == ' ')
                    if not word_clean.strip():
                        continue

                    if phrase_len == 1:
                        subs = self.db.lookup_substitution(word_clean.strip(),
                                                           max_synonym_length=len(
                                                               needed_letters))
                    else:
                        subs = self.db.lookup_phrase_substitution(word_clean)

                    for sub in subs:
                        sub_letters = sub.letters.upper()
                        if sorted(sub_letters) == needed_sorted:
                            # Exact match — accept without contiguity check
                            display_word = window[0] if phrase_len == 1 else phrase
                            cat_prefix = 'database' if phrase_len == 1 else 'database phrase'
                            found_substitutions.append((display_word, sub))
                            word_roles.append(WordRole(
                                display_word, 'substitution', sub_letters,
                                f'{cat_prefix} ({sub.category})'
                            ))
                            for w in window:
                                accounted_words.add(w.lower())
                            needed_letters = ''
                            unresolved_words = [w for w in unresolved_words
                                                if w.lower() not in
                                                {x.lower() for x in window}]
                            found_fallback = True
                            break
                    if found_fallback:
                        break

        solution = {
            'needed_letters_original': answer_upper,
            'anagram_provides': anagram_upper,
            'additional_fodder': [(w, letters) for w, letters in additional_fodder],
            'substitutions': [(w, s.letters, s.category) for w, s in found_substitutions],
            'operation_indicators': [(w, i.wordplay_type, i.subtype)
                                     for w, i in operation_indicators],
            'positional_indicators': positional_indicators,
            'letters_still_needed': needed_letters,
            'unresolved_words': unresolved_words,
            'fully_resolved': len(needed_letters) == 0,
            'word_roles': list(word_roles)  # Include for backtracking sync
        }

        # Try to solve the compound construction
        if found_substitutions:
            sub_letters = ''.join(s.letters for _, s in found_substitutions)
            op_type = operation_indicators[0][
                1].wordplay_type if operation_indicators else None

            if op_type == 'insertion':
                result = self.solver.solve_insertion(anagram_letters, sub_letters, answer)
                if result:
                    solution['construction'] = result
            elif op_type == 'container':
                result = self.solver.solve_container(anagram_letters, sub_letters, answer)
                if result:
                    solution['construction'] = result
            else:
                # Default: try concatenation (check if anagram + sub = answer)
                combined = anagram_upper + sub_letters.upper()
                if sorted(combined) == sorted(answer_upper):
                    solution['construction'] = {
                        'operation': 'concatenation',
                        'parts': [anagram_letters, sub_letters],
                        'result': answer
                    }

        # PAIRWISE ASSEMBLY: When existing assembly didn't find a construction
        # (or only found concatenation) and we have multiple substitutions,
        # try container/insertion/reversal between individual components.
        # This handles pure charade clues like ACE inside RR = RACER.
        if found_substitutions and len(found_substitutions) >= 2:
            existing_op = solution.get('construction', {}).get('operation')
            if existing_op is None or existing_op == 'concatenation':
                pairwise = self._try_pairwise_assembly(
                    found_substitutions, anagram_upper, answer_upper,
                    operation_indicators
                )
                if pairwise:
                    solution['construction'] = pairwise

        # BACKTRACKING: If not fully resolved and we have substitutions, try alternatives
        if not solution[
            'fully_resolved'] and found_substitutions and _backtrack_depth == 0:
            best_solution = solution
            best_letters_needed = len(solution['letters_still_needed'])

            for word, sub in found_substitutions:
                # Try excluding this substitution
                exclude_set = skip_substitutions.copy() if skip_substitutions else set()
                exclude_set.add((word.lower(), sub.letters.upper()))

                # Re-run with this substitution excluded, using ORIGINAL state
                retry_solution = self._analyze_remaining_words(
                    remaining_words, anagram_letters, answer,
                    list(original_word_roles),  # Original state
                    set(original_accounted_words),  # Original state
                    clue_words, definition_window,
                    skip_indicator_words, exclude_set,
                    _backtrack_depth=1  # Prevent further recursion
                )

                if retry_solution:
                    retry_letters = len(retry_solution.get('letters_still_needed', ''))
                    if retry_solution.get('fully_resolved'):
                        # Found a complete solution - sync caller's word_roles and return
                        word_roles.clear()
                        word_roles.extend(retry_solution.get('word_roles', []))
                        return retry_solution
                    elif retry_letters < best_letters_needed:
                        # Better partial solution
                        best_solution = retry_solution
                        best_letters_needed = retry_letters

            solution = best_solution
            # Sync caller's word_roles with the chosen solution
            word_roles.clear()
            word_roles.extend(solution.get('word_roles', []))

        return solution

    def _try_pairwise_assembly(self, found_substitutions, anagram_upper,
                               answer_upper, operation_indicators):
        """
        Try assembling individual substitution components with container/insertion/reversal.

        When the general wordplay path has no anagram letters, the original assembly logic
        tries solve_container('', combined_subs, answer) which is meaningless. This method
        instead tries each substitution as outer/inner with the remaining components,
        handling clues like:
            RACER: ACE inside RR   (one=ACE, car=RR, "in"=container)
            SEEDS: D inside SEES   (Day=D, Understands=SEES, "Boxing"=container)
            RESCUED: SCUE inside RED (second=S, hint=CUE inside Embarrassed=RED)
        """
        from itertools import permutations

        # Get operation types from indicators
        op_types = set()
        for _, ind in operation_indicators:
            op_types.add(ind.wordplay_type)
        has_container = 'container' in op_types or 'insertion' in op_types
        has_reversal = 'reversal' in op_types

        # Build list of individual components: (display_word, letters)
        components = []
        if anagram_upper:
            components.append(('anagram_fodder', anagram_upper))
        for word, sub in found_substitutions:
            components.append((word, sub.letters.upper()))

        if len(components) < 2:
            return None

        n = len(components)

        # --- Strategy 1: One component as outer, rest combined as inner ---
        for outer_idx in range(n):
            outer_word, outer_letters = components[outer_idx]

            # Gather remaining components (preserving identity for permutation)
            inner_components = [components[i] for i in range(n) if i != outer_idx]

            # Try all orderings of inner components
            for perm in permutations(range(len(inner_components))):
                inner_letters = ''.join(inner_components[p][1] for p in perm)

                # Container: outer wraps around inner
                if has_container:
                    result = self.solver.solve_container(outer_letters, inner_letters,
                                                         answer_upper)
                    if result:
                        result['assembly_source'] = 'pairwise_container'
                        result['outer_word'] = outer_word
                        result['inner_words'] = [inner_components[p][0] for p in perm]
                        return result

                    # Insertion: inner goes into outer (same solver, swapped args)
                    result = self.solver.solve_insertion(outer_letters, inner_letters,
                                                         answer_upper)
                    if result:
                        result['assembly_source'] = 'pairwise_insertion'
                        result['base_word'] = outer_word
                        result['inserted_words'] = [inner_components[p][0] for p in perm]
                        return result

                # Reversal + container: reverse inner, then try container
                if has_reversal:
                    reversed_inner = inner_letters[::-1]
                    result = self.solver.solve_container(outer_letters, reversed_inner,
                                                         answer_upper)
                    if result:
                        result['assembly_source'] = 'reversal_container'
                        result['reversed_component'] = inner_letters
                        return result

                    result = self.solver.solve_insertion(outer_letters, reversed_inner,
                                                         answer_upper)
                    if result:
                        result['assembly_source'] = 'reversal_insertion'
                        result['reversed_component'] = inner_letters
                        return result

                    # Reverse outer, then try container with original inner
                    reversed_outer = outer_letters[::-1]
                    result = self.solver.solve_container(reversed_outer, inner_letters,
                                                         answer_upper)
                    if result:
                        result['assembly_source'] = 'reversal_container'
                        result['reversed_component'] = outer_letters
                        return result

        # --- Strategy 2: Reversal-based concatenation ---
        # Try reversing individual components and concatenating in various orders
        if has_reversal:
            # For each component, try reversed version; try all orderings
            for perm in permutations(range(n)):
                # Try each bitmask of which components to reverse
                for mask in range(1, 2 ** n):
                    parts = []
                    for bit_pos, comp_idx in enumerate(perm):
                        letters = components[comp_idx][1]
                        if mask & (1 << bit_pos):
                            letters = letters[::-1]
                        parts.append(letters)
                    combined = ''.join(parts)
                    if combined == answer_upper:
                        return {
                            'operation': 'reversal_concatenation',
                            'parts': [(components[perm[i]][0],
                                       components[perm[i]][1],
                                       bool(mask & (1 << i)))
                                      for i in range(n)],
                            'result': answer_upper
                        }

        return None

    def _try_pairwise_assembly(self, found_substitutions, anagram_upper,
                               answer_upper, operation_indicators):
        """
        Try assembling individual substitution components with container/insertion/reversal.

        When the general wordplay path has no anagram letters, the original assembly logic
        tries solve_container('', combined_subs, answer) which is meaningless. This method
        instead tries each substitution as outer/inner with the remaining components.

        IMPORTANT: We do NOT gate on indicator type. Words like "in" often get classified
        as "hidden" rather than "container", and reversal indicators may be missing from
        the DB. If the components physically fit via container/reversal, that IS the evidence.

        Handles clues like:
            RACER:   ACE inside RR        ("in" classified as hidden, not container)
            SEEDS:   D inside SEES        ("Boxing" = container)
            RESCUED: S+CUE inside RED     (multiple inner components)
            APACE:   reversed CAP in A_E  (reversal + container)
            BLOSSOM: LOSS in reversed MOB (reversal + container)
            SPECKLE: reversed CEPS + reversed ELK (reversal + concatenation)
        """
        from itertools import permutations

        # Build list of individual components: (display_word, letters)
        components = []
        if anagram_upper:
            components.append(('anagram_fodder', anagram_upper))
        for word, sub in found_substitutions:
            components.append((word, sub.letters.upper()))

        if len(components) < 2:
            return None

        # Safety cap: with >5 components, permutations explode (5!=120, 6!=720)
        # In practice cryptic clues rarely have more than 4 wordplay components
        n = len(components)
        if n > 5:
            return None

        # --- Strategy 1: Container/insertion (no indicator gate) ---
        # For each component as outer, try all orderings of remaining as inner
        for outer_idx in range(n):
            outer_word, outer_letters = components[outer_idx]
            inner_components = [components[i] for i in range(n) if i != outer_idx]

            for perm in permutations(range(len(inner_components))):
                inner_letters = ''.join(inner_components[p][1] for p in perm)

                # Container: outer wraps around inner
                result = self.solver.solve_container(outer_letters, inner_letters,
                                                     answer_upper)
                if result:
                    result['assembly_source'] = 'pairwise_container'
                    result['outer_word'] = outer_word
                    result['inner_words'] = [inner_components[p][0] for p in perm]
                    return result

                # Insertion: inner goes into outer
                result = self.solver.solve_insertion(outer_letters, inner_letters,
                                                     answer_upper)
                if result:
                    result['assembly_source'] = 'pairwise_insertion'
                    result['base_word'] = outer_word
                    result['inserted_words'] = [inner_components[p][0] for p in perm]
                    return result

        # --- Strategy 2: Reversal + container/insertion (no indicator gate) ---
        # Try reversing one component at a time, then container/insertion
        for rev_idx in range(n):
            rev_word, rev_letters = components[rev_idx]
            reversed_letters = rev_letters[::-1]

            # Skip if reversing is a no-op (palindrome or single letter)
            if reversed_letters == rev_letters:
                continue

            # Try reversed component as outer
            other_components = [components[i] for i in range(n) if i != rev_idx]
            for perm in permutations(range(len(other_components))):
                inner_letters = ''.join(other_components[p][1] for p in perm)

                result = self.solver.solve_container(reversed_letters, inner_letters,
                                                     answer_upper)
                if result:
                    result['assembly_source'] = 'reversal_container'
                    result['reversed_word'] = rev_word
                    result['reversed_from'] = rev_letters
                    result['inner_words'] = [other_components[p][0] for p in perm]
                    return result

                result = self.solver.solve_insertion(reversed_letters, inner_letters,
                                                     answer_upper)
                if result:
                    result['assembly_source'] = 'reversal_insertion'
                    result['reversed_word'] = rev_word
                    result['reversed_from'] = rev_letters
                    result['inserted_words'] = [other_components[p][0] for p in perm]
                    return result

            # Try reversed component as inner
            for outer_idx in range(len(other_components)):
                outer_word, outer_letters = other_components[outer_idx]
                remaining = [other_components[i] for i in range(len(other_components))
                             if i != outer_idx]
                # Reversed component + any remaining, as inner
                for perm in permutations(range(len(remaining))):
                    inner_letters = reversed_letters + ''.join(
                        remaining[p][1] for p in perm)

                    result = self.solver.solve_container(outer_letters, inner_letters,
                                                         answer_upper)
                    if result:
                        result['assembly_source'] = 'reversal_container'
                        result['reversed_word'] = rev_word
                        result['reversed_from'] = rev_letters
                        result['outer_word'] = outer_word
                        return result

                    # Also try reversed component after the remaining
                    if remaining:
                        inner_letters_alt = ''.join(
                            remaining[p][1] for p in perm) + reversed_letters

                        result = self.solver.solve_container(
                            outer_letters, inner_letters_alt, answer_upper)
                        if result:
                            result['assembly_source'] = 'reversal_container'
                            result['reversed_word'] = rev_word
                            result['reversed_from'] = rev_letters
                            result['outer_word'] = outer_word
                            return result

        # --- Strategy 3: Reversal + concatenation ---
        # Try reversing individual components and concatenating in all orderings
        for perm in permutations(range(n)):
            # Try each bitmask of which components to reverse (skip 0 = no reversals)
            for mask in range(1, 2 ** n):
                parts = []
                for bit_pos, comp_idx in enumerate(perm):
                    letters = components[comp_idx][1]
                    if mask & (1 << bit_pos):
                        letters = letters[::-1]
                    parts.append(letters)
                combined = ''.join(parts)
                if combined == answer_upper:
                    return {
                        'operation': 'reversal_concatenation',
                        'assembly_source': 'reversal_concatenation',
                        'parts': [(components[perm[i]][0],
                                   components[perm[i]][1],
                                   bool(mask & (1 << i)))
                                  for i in range(n)],
                        'result': answer_upper
                    }

        return None

    def _handle_deletion_compound(self, remaining_words: List[str],
                                  anagram_letters: str, answer: str,
                                  excess_letters: str,
                                  word_roles: List[WordRole],
                                  accounted_words: Set[str],
                                  clue_words: List[str],
                                  definition_window: Optional[str]) -> Dict[str, Any]:
        """
        Handle deletion compounds where anagram has MORE letters than answer.
        Example: LOVE + IRISH (9) - O (duck) = LIVERISH (8)

        Looks for:
        1. A word that substitutes to the excess letters (e.g., "duck" -> O)
        2. A deletion indicator (e.g., "out", "without", "missing")
        """
        def_words_lower = set()
        if definition_window:
            def_words_lower = {w.lower() for w in definition_window.split()}

        deletion_substitution = None
        deletion_indicator = None
        other_indicators = []

        for word in remaining_words:
            word_lower = word.lower()

            # Skip definition words
            if word_lower in def_words_lower:
                continue

            # Strip possessives for lookup (university's -> university)
            # Handle both straight (') and curly (') apostrophes
            word_for_lookup = re.sub(r"[''\u2019]s$", "", word, flags=re.IGNORECASE)
            word_for_lookup_lower = word_for_lookup.lower()

            # Check if this word substitutes to the excess letters
            subs = self.db.lookup_substitution(word_for_lookup,
                                               max_synonym_length=len(excess_letters))
            # Also try the full word (with apostrophe removed but 's kept)
            word_full = ''.join(c for c in word if c.isalpha())
            if word_full.lower() != word_for_lookup_lower:
                subs_full = self.db.lookup_substitution(word_full,
                                                        max_synonym_length=len(
                                                            excess_letters))
                existing_letters = {s.letters.upper() for s in subs}
                for s in subs_full:
                    if s.letters.upper() not in existing_letters:
                        subs.append(s)
            for sub in subs:
                sub_letters = sub.letters.upper()
                if sorted(sub_letters) == sorted(excess_letters):
                    deletion_substitution = (word, sub)
                    word_roles.append(WordRole(
                        word, 'deletion_target', sub_letters,
                        f'database ({sub.category}) - letters to remove'
                    ))
                    accounted_words.add(word_lower)
                    break

            if deletion_substitution and word_lower == deletion_substitution[0].lower():
                continue

            # Check for deletion indicator (also try stripped version)
            indicator_match = self.db.lookup_indicator(word_for_lookup)
            if not indicator_match:
                indicator_match = self.db.lookup_indicator(word)  # Try original too
            if indicator_match:
                if indicator_match.wordplay_type == 'deletion':
                    deletion_indicator = (word, indicator_match)
                    word_roles.append(WordRole(
                        word, 'deletion_indicator', '', 'database'
                    ))
                    accounted_words.add(word_lower)
                elif indicator_match.wordplay_type == 'anagram':
                    # Already used for anagram, just note it
                    other_indicators.append((word, indicator_match))
                    word_roles.append(WordRole(
                        word, 'anagram_indicator', '', 'database'
                    ))
                    accounted_words.add(word_lower)

        # Build solution
        # CRITICAL: Only return a deletion solution if we have a deletion indicator
        # Without an indicator, we shouldn't assume deletion - the fodder selection may be wrong
        if deletion_indicator is None:
            # No deletion indicator - don't commit to deletion interpretation
            # Return None so the system can try other approaches
            return None

        fully_resolved = deletion_substitution is not None and deletion_indicator is not None

        solution = {
            'operation': 'deletion',
            'anagram_provides': anagram_letters.upper(),
            'excess_letters': excess_letters,
            'deletion_target': (deletion_substitution[0],
                                deletion_substitution[1].letters,
                                deletion_substitution[
                                    1].category) if deletion_substitution else None,
            'deletion_indicator': (deletion_indicator[0],
                                   deletion_indicator[1].wordplay_type,
                                   deletion_indicator[
                                       1].subtype) if deletion_indicator else None,
            'fully_resolved': fully_resolved,
            'construction': {
                'operation': 'deletion',
                'base': anagram_letters,
                'remove': excess_letters,
                'result': answer
            } if fully_resolved else None
        }

        return solution

    def _try_reduced_fodder(self, remaining_words: List[str],
                            anagram_letters: str,
                            answer: str,
                            excess_letters: str,
                            word_roles: List[WordRole],
                            accounted_words: Set[str],
                            clue_words: List[str],
                            definition_window: Optional[str]) -> Optional[Dict[str, Any]]:
        """
        When we have excess letters but no deletion indicator, try reducing fodder
        and looking for substitutions that could provide the missing letters.

        Example: DOG+NEAR+GATE = 11 letters for 9-letter answer GREATDANE
        - Look for substitutions in remaining words: "daughter" → D
        - Try removing fodder: if we remove DOG, we have NEARGATE = 8 letters
        - 8 + D = 9 = GREATDANE ✓
        """
        answer_upper = answer.upper().replace(' ', '')
        answer_len = len(answer_upper)

        # Helper to get letters from word
        def get_letters(w):
            return ''.join(c.upper() for c in w if c.isalpha())

        # First, find what substitutions are available in remaining words
        available_subs = []
        for word in remaining_words:
            subs = self.db.lookup_substitution(word)
            for sub in subs:
                available_subs.append((word, sub.letters.upper(), sub.category))

        if not available_subs:
            return None

        # Get current fodder words from word_roles
        fodder_roles = [wr for wr in word_roles if wr.role == 'fodder']
        if len(fodder_roles) < 2:
            return None  # Need at least 2 fodder words to try removing one

        # Try removing each fodder word and see if remaining + substitution works
        for remove_role in fodder_roles:
            # Calculate letters without this fodder word
            remaining_fodder_letters = ''
            remaining_fodder_words = []
            for fr in fodder_roles:
                if fr.word != remove_role.word:
                    remaining_fodder_letters += get_letters(fr.word)
                    remaining_fodder_words.append(fr.word)

            remaining_len = len(remaining_fodder_letters)
            needed_len = answer_len - remaining_len

            if needed_len <= 0:
                continue  # Still too many letters

            # Check if any substitution provides exactly the needed letters
            for sub_word, sub_letters, sub_category in available_subs:
                if len(sub_letters) == needed_len:
                    # Check if remaining fodder + substitution = answer (as anagram)
                    combined = remaining_fodder_letters + sub_letters
                    if sorted(combined.upper()) == sorted(answer_upper):
                        # Found a working combination!
                        # Update word_roles: remove the excluded fodder, add substitution
                        new_word_roles = [wr for wr in word_roles if
                                          wr.word != remove_role.word]
                        new_word_roles.append(WordRole(
                            sub_word, 'substitution', sub_letters,
                            f'database ({sub_category})'
                        ))

                        # Update accounted words
                        new_accounted = set(accounted_words)
                        new_accounted.discard(remove_role.word.lower())
                        new_accounted.add(sub_word.lower())

                        # Check if removed word is in definition window - mark as definition
                        if definition_window:
                            def_words = {w.lower() for w in definition_window.split()}
                            if norm_letters(remove_role.word) in {norm_letters(w) for w in
                                                                  def_words}:
                                new_word_roles.append(WordRole(
                                    remove_role.word, 'definition', answer_upper,
                                    'reduced_fodder'
                                ))
                                new_accounted.add(remove_role.word.lower())

                        # Look for indicators (anagram, container, insertion)
                        operation_indicators = []
                        anagram_indicator = None
                        for word in remaining_words:
                            word_norm = norm_letters(word)
                            if word_norm in {norm_letters(a) for a in new_accounted}:
                                continue
                            indicator_match = self.db.lookup_indicator(word)
                            if indicator_match:
                                op_type = indicator_match.wordplay_type
                                if op_type == 'anagram' and anagram_indicator is None:
                                    anagram_indicator = (word, indicator_match)
                                    new_word_roles.append(WordRole(
                                        word, 'anagram_indicator', '', 'database'
                                    ))
                                    new_accounted.add(word.lower())
                                elif op_type in ('insertion', 'container'):
                                    operation_indicators.append((word, indicator_match))
                                    new_word_roles.append(WordRole(
                                        word, f'{op_type}_indicator', '', 'database'
                                    ))
                                    new_accounted.add(word.lower())

                        # Update the main word_roles and accounted_words
                        word_roles.clear()
                        word_roles.extend(new_word_roles)
                        accounted_words.clear()
                        accounted_words.update(new_accounted)

                        return {
                            'operation': 'reduced_fodder',
                            'original_fodder': anagram_letters,
                            'reduced_fodder': remaining_fodder_letters,
                            'removed_word': remove_role.word,
                            'substitutions': [(sub_word, sub_letters, sub_category)],
                            'additional_fodder': [],
                            'operation_indicators': [(w, i.wordplay_type, i.subtype)
                                                     for w, i in operation_indicators],
                            'anagram_indicator': (anagram_indicator[0], anagram_indicator[
                                1].wordplay_type) if anagram_indicator else None,
                            'fully_resolved': True,
                            'construction': {
                                'operation': 'insertion' if operation_indicators else 'concatenation',
                                'base': remaining_fodder_letters,
                                'add': sub_letters,
                                'result': answer_upper
                            }
                        }

        return None

    def _classify_remaining_as_indicators(self, remaining_words: List[str],
                                          word_roles: List[WordRole],
                                          accounted_words: Set[str],
                                          clue_words: List[str],
                                          definition_window: Optional[str]) -> Dict[
        str, Any]:
        """When anagram is complete, classify remaining words as indicators."""
        def_words_lower = set()
        if definition_window:
            def_words_lower = {w.lower() for w in definition_window.split()}

        classified = []

        for word in remaining_words:
            word_lower = word.lower()

            # Skip if in definition
            if word_lower in def_words_lower:
                continue

            # Check indicators table
            # For pure anagrams, only mark additional anagram indicators
            # Other indicator types (deletion, acrostic, etc.) are not performing any operation
            indicator_match = self.db.lookup_indicator(word)
            if indicator_match and indicator_match.wordplay_type == 'anagram':
                classified.append((word, indicator_match.wordplay_type))
                word_roles.append(WordRole(
                    word, f'{indicator_match.wordplay_type}_indicator', '', 'database'
                ))
                accounted_words.add(word_lower)

        return {
            'fully_resolved': True,
            'classified_indicators': classified
        }


def test_database_lookup():
    """Test the database lookup functionality."""
    db = DatabaseLookup()

    print("=" * 60)
    print("TESTING DATABASE LOOKUPS")
    print("=" * 60)

    print("\n1. INDICATOR LOOKUPS:")
    print("-" * 40)
    test_indicators = ['drunk', 'involving', 'around', 'without', 'back', 'after',
                       'containing']
    for word in test_indicators:
        result = db.lookup_indicator(word)
        if result:
            print(
                f"  ✓ {word:15} -> {result.wordplay_type:12} / {result.subtype or 'none'}")
        else:
            print(f"  ✗ {word:15} -> NOT FOUND")

    print("\n2. SUBSTITUTION LOOKUPS:")
    print("-" * 40)
    test_subs = ['party', 'Germany', 'love', 'time', 'church', 'company', 'English']
    for word in test_subs:
        results = db.lookup_substitution(word)
        if results:
            for r in results[:2]:  # Show max 2 per word
                print(f"  ✓ {word:15} -> {r.letters:5} ({r.category})")
        else:
            print(f"  ✗ {word:15} -> NOT FOUND")

    db.close()
    print("\n" + "=" * 60)


def test_with_sample_case():
    """Test with a sample case like DOORMAN."""
    print("\n" + "=" * 60)
    print("TESTING WITH SAMPLE CASE: DOORMAN")
    print("=" * 60)

    # Simulate a case as it would come from evidence analysis
    from dataclasses import dataclass

    @dataclass
    class MockEvidence:
        candidate: str = 'DOORMAN'
        fodder_words: list = None
        fodder_letters: str = 'NORMA'
        evidence_type: str = 'partial_anagram'
        confidence: float = 0.85

        def __post_init__(self):
            if self.fodder_words is None:
                self.fodder_words = ['Norma']

    sample_case = {
        'clue': 'One who may help guest Norma drunk after party (7)',
        'answer': 'DOORMAN',
        'definition': 'One who may help guest',
        'window_support': {
            'One who may help guest': ['DOORMAN']
        },
        'evidence_analysis': {
            'scored_candidates': [
                {
                    'candidate': 'DOORMAN',
                    'evidence': MockEvidence()
                }
            ]
        }
    }

    analyzer = CompoundWordplayAnalyzer()
    result = analyzer.analyze_case(sample_case)

    print(f"\nClue: {result['clue']}")
    print(f"Answer: {result['answer']}")
    print(f"\nExplanation:")
    print(f"  Formula: {result['explanation']['formula']}")
    print(f"  Quality: {result['explanation']['quality']}")
    print(f"\nBreakdown:")
    for line in result['explanation']['breakdown']:
        print(f"  {line}")

    if result.get('compound_solution'):
        print(f"\nCompound Solution:")
        cs = result['compound_solution']
        if cs.get('substitutions'):
            print(f"  Substitutions: {cs['substitutions']}")
        if cs.get('operation_indicators'):
            print(f"  Operation indicators: {cs['operation_indicators']}")
        if cs.get('positional_indicators'):
            print(f"  Positional indicators: {cs['positional_indicators']}")
        if cs.get('letters_still_needed'):
            missing = cs['letters_still_needed']
            unresolved = cs.get('unresolved_words', [])
            if unresolved:
                print(
                    f"  Indicator and fodder unaccounted for: {missing.upper()} from {unresolved}")
            else:
                print(f"  Letters unaccounted for: {missing.upper()}")
        print(f"  Fully resolved: {cs.get('fully_resolved', False)}")

    if result.get('remaining_unresolved'):
        print(f"\nRemaining unresolved: {result['remaining_unresolved']}")

    analyzer.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    test_database_lookup()
    test_with_sample_case()