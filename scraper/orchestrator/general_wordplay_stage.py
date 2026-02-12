#!/usr/bin/env python3
"""
General Wordplay Stage

Handles clues forwarded from the anagram/compound stage:
- Charade: Chain substitutions/synonyms (D + ACE = DACE)
- Reversal: Synonym + reverse (BONK → KNOB)
- Deletion: Synonym + delete letters (PAINTER - IN = PATER)
- Acrostic: First letters of consecutive words
- Container: X around Y

Uses same tables as anagram_evidence_system.py:
- synonyms_pairs
- wordplay
- indicators
"""

import re
import sqlite3
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
from itertools import combinations, permutations


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class WordplayEvidence:
    """Evidence for a wordplay parse."""
    evidence_type: str  # charade, reversal, deletion, acrostic, container
    answer: str
    components: List[Dict]  # List of {source, operation, letters, indicator}
    definition_words: List[str]
    indicator_words: List[str]
    confidence: float
    derivation: str = ""

    @property
    def is_complete(self) -> bool:
        """Check if components fully explain the answer."""
        built = ''.join(c['letters'] for c in self.components).upper()
        target = self.answer.upper().replace(' ', '').replace('-', '')
        return built == target


# =============================================================================
# GENERAL WORDPLAY DETECTOR
# =============================================================================

class GeneralWordplayDetector:
    """
    Detects non-anagram wordplay: charade, reversal, deletion, acrostic, container.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None

        # Indicator sets
        self.reversal_indicators: Set[str] = set()
        self.deletion_indicators: Set[str] = set()
        self.container_indicators: Set[str] = set()
        self.acrostic_indicators: Set[str] = set()
        self.first_letter_indicators: Set[str] = set()
        self.last_letter_indicators: Set[str] = set()

        # Caches
        self._synonym_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._substitution_cache: Dict[str, List[Tuple[str, str]]] = {}

        self._load_indicators()

    def _get_conn(self):
        if self._conn is None:
            self._conn = sqlite3.connect(self.db_path)
        return self._conn

    def close(self):
        if self._conn:
            self._conn.close()
            self._conn = None

    def _load_indicators(self):
        """Load indicators from database."""
        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("""
            SELECT word, wordplay_type, subtype 
            FROM indicators
        """)

        for word, wtype, subtype in cur.fetchall():
            word_lower = word.lower().strip()

            if wtype == 'reversal':
                self.reversal_indicators.add(word_lower)
            elif wtype == 'deletion':
                self.deletion_indicators.add(word_lower)
            elif wtype == 'insertion':
                self.container_indicators.add(word_lower)
            elif wtype == 'parts':
                if subtype in ('first_letter', 'initial', 'head', 'start', 'opener'):
                    self.first_letter_indicators.add(word_lower)
                elif subtype in ('last_letter', 'final', 'tail', 'end', 'closer'):
                    self.last_letter_indicators.add(word_lower)
                elif subtype in ('acrostic', 'initials', 'heads', 'starts'):
                    self.acrostic_indicators.add(word_lower)

        # Add common indicators that might be missing
        self.reversal_indicators.update({
            'back', 'returned', 'up', 'over', 'around', 'about',
            'reversed', 'backwards', 'retiring', 'recalled', 'reflected',
            'going back', 'turning', 'turned', 'flipped', 'upset'
        })

        self.deletion_indicators.update({
            'drops', 'dropping', 'dropped', 'loses', 'losing', 'lost',
            'without', 'lacking', 'missing', 'removed', 'removing',
            'less', 'no', 'not', 'excluding', 'except', 'cut', 'cuts',
            'sheds', 'shedding', 'discards', 'discarding'
        })

        self.acrostic_indicators.update({
            'initially', 'heads', 'leaders', 'starters', 'openers',
            'first letters', 'capitals', 'at first', 'to start',
            'primarily', 'fronts', 'tips'
        })

        self.first_letter_indicators.update({
            'first', 'head', 'start', 'opening', 'initially', 'at first',
            'front', 'lead', 'top', 'head of', 'start of', 'beginning',
            'opener', 'origin', 'capital'
        })

        self.last_letter_indicators.update({
            'last', 'finally', 'end', 'tail', 'closing', 'at last',
            'back', 'rear', 'finish', 'end of', 'lastly', 'ultimate',
            'ending', 'terminal', 'final'
        })

        print(f"General Wordplay - Loaded indicators:")
        print(f"  Reversal: {len(self.reversal_indicators)}")
        print(f"  Deletion: {len(self.deletion_indicators)}")
        print(f"  Container: {len(self.container_indicators)}")
        print(f"  First letter: {len(self.first_letter_indicators)}")
        print(f"  Last letter: {len(self.last_letter_indicators)}")
        print(f"  Acrostic: {len(self.acrostic_indicators)}")

    def _norm_letters(self, s: str) -> str:
        """Extract just letters, lowercase."""
        return re.sub(r'[^a-zA-Z]', '', s).lower()

    def _find_definition(self, tokens: List[str], answer: str, used_indices: Set[int]) -> \
    List[str]:
        """
        Find which unused token(s) form the definition.
        Definition should be a synonym of the answer.
        Returns empty list if no valid definition found.
        """
        answer_norm = self._norm_letters(answer).upper()
        answer_len = len(answer_norm)

        unused = [(i, tokens[i]) for i in range(len(tokens)) if i not in used_indices]

        if not unused:
            return []

        # Check single words first (prefer start or end of clue)
        # Sort by position - start and end words are more likely to be definitions
        start_unused = [u for u in unused if u[0] < 3]  # First 3 positions
        end_unused = [u for u in unused if u[0] >= len(tokens) - 3]  # Last 3 positions
        priority_unused = start_unused + end_unused + unused  # Check start/end first

        seen = set()
        for i, token in priority_unused:
            if token in seen:
                continue
            seen.add(token)

            syns = self.lookup_synonyms(token, answer_len)
            for syn in syns:
                if self._norm_letters(syn).upper() == answer_norm:
                    return [token]

        # Check two-word phrases at start or end
        if len(unused) >= 2:
            # First two unused
            if unused[0][0] < 3:  # Near start
                phrase = ' '.join([unused[0][1], unused[1][1]])
                syns = self.lookup_synonyms(phrase, answer_len)
                for syn in syns:
                    if self._norm_letters(syn).upper() == answer_norm:
                        return [unused[0][1], unused[1][1]]

            # Last two unused
            if unused[-1][0] >= len(tokens) - 3:  # Near end
                phrase = ' '.join([unused[-2][1], unused[-1][1]])
                syns = self.lookup_synonyms(phrase, answer_len)
                for syn in syns:
                    if self._norm_letters(syn).upper() == answer_norm:
                        return [unused[-2][1], unused[-1][1]]

        # No valid definition found
        return []

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens."""
        # Remove enumeration
        text = re.sub(r'\(\d+(?:,\d+)*\)$', '', text).strip()
        text = re.sub(r'\(\d+(?:-\d+)*\)$', '', text).strip()
        tokens = text.split()
        return [t.strip('.,;:!?"()[]{}') for t in tokens if t.strip('.,;:!?"()[]{}')]

    # =========================================================================
    # TABLE LOOKUPS
    # =========================================================================

    def lookup_synonyms(self, word: str, target_length: int = None) -> List[str]:
        """Look up synonyms for a word from synonyms_pairs table."""
        word_clean = word.lower().strip()

        cache_key = f"{word_clean}_{target_length}"
        if cache_key in self._synonym_cache:
            return self._synonym_cache[cache_key]

        conn = self._get_conn()
        cur = conn.cursor()

        # Check both directions in synonyms_pairs
        cur.execute("""
            SELECT synonym FROM synonyms_pairs
            WHERE LOWER(word) = ?
        """, (word_clean,))
        results = [row[0] for row in cur.fetchall()]

        cur.execute("""
            SELECT word FROM synonyms_pairs
            WHERE LOWER(synonym) = ?
        """, (word_clean,))
        results.extend([row[0] for row in cur.fetchall()])

        # Also check definition_answers_augmented
        cur.execute("""
            SELECT answer FROM definition_answers_augmented
            WHERE LOWER(definition) = ?
        """, (word_clean,))
        results.extend([row[0] for row in cur.fetchall()])

        # Filter by length if specified
        if target_length:
            results = [r for r in results if len(self._norm_letters(r)) == target_length]

        # Remove duplicates
        results = list(set(results))

        self._synonym_cache[cache_key] = results
        return results

    def lookup_substitution(self, word: str) -> List[Tuple[str, str]]:
        """Look up substitutions from wordplay table. Returns [(letters, category), ...]"""
        word_clean = word.lower().strip()

        if word_clean in self._substitution_cache:
            return self._substitution_cache[word_clean]

        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("""
            SELECT substitution, category, COALESCE(frequency, 0) as freq
            FROM wordplay
            WHERE LOWER(indicator) = ?
            ORDER BY freq DESC
        """, (word_clean,))

        results = [(row[0], row[1]) for row in cur.fetchall()]
        self._substitution_cache[word_clean] = results
        return results

    # =========================================================================
    # REVERSAL DETECTION
    # =========================================================================

    def detect_reversal(self, clue_text: str, answer: str, debug: bool = False) -> \
    Optional[WordplayEvidence]:
        """
        Detect reversal: word → synonym → reverse = answer
        Example: "hit back" → BONK → KNOB
        """
        tokens = self._tokenize(clue_text)
        answer_norm = self._norm_letters(answer).upper()
        answer_len = len(answer_norm)

        if debug:
            print(f"\n  REVERSAL: Looking for reversed synonym of length {answer_len}")

        # Find reversal indicators
        reversal_positions = []
        for i, token in enumerate(tokens):
            if token.lower() in self.reversal_indicators:
                reversal_positions.append(i)

        if not reversal_positions:
            return None

        if debug:
            print(f"    Reversal indicators at: {reversal_positions}")

        # Check each non-indicator word for synonyms that reverse to answer
        for i, token in enumerate(tokens):
            if token.lower() in self.reversal_indicators:
                continue

            # Get synonyms of this word with correct length
            synonyms = self.lookup_synonyms(token, answer_len)

            for syn in synonyms:
                syn_norm = self._norm_letters(syn).upper()
                reversed_syn = syn_norm[::-1]

                if reversed_syn == answer_norm:
                    # Find closest reversal indicator
                    closest_ind = min(reversal_positions, key=lambda x: abs(x - i))
                    indicator = tokens[closest_ind]

                    # Determine definition (words not used in wordplay)
                    used_positions = {i, closest_ind}
                    definition_words = [tokens[j] for j in range(len(tokens)) if
                                        j not in used_positions]

                    derivation = f"{token}→{syn_norm} reversed→{answer_norm}"

                    if debug:
                        print(f"    FOUND: {derivation}")

                    return WordplayEvidence(
                        evidence_type='reversal',
                        answer=answer,
                        components=[{
                            'source': token,
                            'operation': 'synonym_reversal',
                            'letters': answer_norm,
                            'indicator': indicator,
                            'synonym': syn_norm
                        }],
                        definition_words=definition_words,
                        indicator_words=[indicator],
                        confidence=0.9,
                        derivation=derivation
                    )

        # Also check: direct reversal of substitution
        for i, token in enumerate(tokens):
            if token.lower() in self.reversal_indicators:
                continue

            subs = self.lookup_substitution(token)
            for letters, category in subs:
                letters_norm = self._norm_letters(letters).upper()
                reversed_letters = letters_norm[::-1]

                if reversed_letters == answer_norm:
                    closest_ind = min(reversal_positions, key=lambda x: abs(x - i))
                    indicator = tokens[closest_ind]

                    used_positions = {i, closest_ind}
                    definition_words = [tokens[j] for j in range(len(tokens)) if
                                        j not in used_positions]

                    derivation = f"{token}→{letters_norm} reversed→{answer_norm}"

                    return WordplayEvidence(
                        evidence_type='reversal',
                        answer=answer,
                        components=[{
                            'source': token,
                            'operation': 'substitution_reversal',
                            'letters': answer_norm,
                            'indicator': indicator,
                            'substitution': letters_norm
                        }],
                        definition_words=definition_words,
                        indicator_words=[indicator],
                        confidence=0.9,
                        derivation=derivation
                    )

        return None

    # =========================================================================
    # DELETION DETECTION
    # =========================================================================

    def detect_deletion(self, clue_text: str, answer: str, debug: bool = False) -> \
    Optional[WordplayEvidence]:
        """
        Detect deletion: word → synonym → delete letters = answer
        Example: "Artist drops in" → PAINTER - IN → PATER
        """
        tokens = self._tokenize(clue_text)
        answer_norm = self._norm_letters(answer).upper()
        answer_len = len(answer_norm)

        if debug:
            print(f"\n  DELETION: Looking for synonym minus letters = {answer_norm}")

        # Find deletion indicators
        deletion_positions = []
        for i, token in enumerate(tokens):
            if token.lower() in self.deletion_indicators:
                deletion_positions.append(i)

        if not deletion_positions:
            return None

        if debug:
            print(f"    Deletion indicators at: {deletion_positions}")

        # Strategy: Find word X with synonym S, and word Y where S - Y = answer
        for i, token_x in enumerate(tokens):
            if token_x.lower() in self.deletion_indicators:
                continue

            # Get synonyms (longer than answer, so deletion is possible)
            synonyms = self.lookup_synonyms(token_x)

            for syn in synonyms:
                syn_norm = self._norm_letters(syn).upper()

                # Synonym must be longer than answer
                if len(syn_norm) <= answer_len:
                    continue

                # Check each other word as potential deletion target
                for j, token_y in enumerate(tokens):
                    if i == j or token_y.lower() in self.deletion_indicators:
                        continue

                    del_letters = self._norm_letters(token_y).upper()

                    # Check if removing del_letters from syn gives answer
                    result = self._remove_substring(syn_norm, del_letters)

                    if result and result == answer_norm:
                        # Find closest deletion indicator
                        closest_ind = min(deletion_positions, key=lambda x: abs(x - i))
                        indicator = tokens[closest_ind]

                        used_positions = {i, j, closest_ind}
                        definition_words = [tokens[k] for k in range(len(tokens)) if
                                            k not in used_positions]

                        derivation = f"{token_x}→{syn_norm} - {del_letters} = {answer_norm}"

                        if debug:
                            print(f"    FOUND: {derivation}")

                        return WordplayEvidence(
                            evidence_type='deletion',
                            answer=answer,
                            components=[{
                                'source': token_x,
                                'operation': 'synonym_deletion',
                                'letters': answer_norm,
                                'indicator': indicator,
                                'synonym': syn_norm,
                                'deleted': del_letters
                            }],
                            definition_words=definition_words,
                            indicator_words=[indicator],
                            confidence=0.9,
                            derivation=derivation
                        )

        return None

    def _remove_substring(self, s: str, sub: str) -> Optional[str]:
        """Remove contiguous substring from string if present."""
        idx = s.find(sub)
        if idx >= 0:
            return s[:idx] + s[idx + len(sub):]
        return None

    # =========================================================================
    # ACROSTIC DETECTION
    # =========================================================================

    def detect_acrostic(self, clue_text: str, answer: str, debug: bool = False) -> \
    Optional[WordplayEvidence]:
        """
        Detect acrostic: first letters of consecutive words = answer
        Example: "Leaders of Secret Police Yield" → S P Y = SPY
        """
        tokens = self._tokenize(clue_text)
        answer_norm = self._norm_letters(answer).upper()
        answer_len = len(answer_norm)

        if debug:
            print(
                f"\n  ACROSTIC: Looking for {answer_len} consecutive first letters = {answer_norm}")

        # Find acrostic/first letter indicators
        acrostic_positions = []
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if token_lower in self.acrostic_indicators or token_lower in self.first_letter_indicators:
                acrostic_positions.append(i)

        if debug:
            print(f"    Acrostic indicators at: {acrostic_positions}")

        # Try consecutive sequences of tokens
        for start in range(len(tokens) - answer_len + 1):
            # Skip if starting with an indicator
            if start in acrostic_positions:
                continue

            # Get first letters of consecutive words
            first_letters = ''
            words_used = []
            for k in range(answer_len):
                idx = start + k
                if idx >= len(tokens):
                    break
                word = tokens[idx]
                if word.lower() in self.acrostic_indicators or word.lower() in self.first_letter_indicators:
                    break  # Don't include indicators in the sequence
                letters = self._norm_letters(word)
                if letters:
                    first_letters += letters[0].upper()
                    words_used.append(word)

            if first_letters == answer_norm:
                # Find indicator (usually before the sequence)
                indicator = None
                if acrostic_positions:
                    # Prefer indicator just before the sequence
                    for pos in acrostic_positions:
                        if pos == start - 1:
                            indicator = tokens[pos]
                            break
                    if not indicator:
                        indicator = tokens[acrostic_positions[0]]

                used_positions = set(range(start, start + len(words_used)))
                if indicator:
                    used_positions.add(tokens.index(indicator))

                definition_words = [tokens[k] for k in range(len(tokens)) if
                                    k not in used_positions]

                derivation = f"first letters of {'+'.join(words_used)} = {answer_norm}"

                if debug:
                    print(f"    FOUND: {derivation}")

                return WordplayEvidence(
                    evidence_type='acrostic',
                    answer=answer,
                    components=[{
                        'source': words_used,
                        'operation': 'acrostic',
                        'letters': answer_norm,
                        'indicator': indicator
                    }],
                    definition_words=definition_words,
                    indicator_words=[indicator] if indicator else [],
                    confidence=0.85 if indicator else 0.6,
                    derivation=derivation
                )

        return None

    # =========================================================================
    # CHARADE DETECTION
    # =========================================================================

    def detect_charade(self, clue_text: str, answer: str, debug: bool = False) -> \
    Optional[WordplayEvidence]:
        """
        Detect charade: chain of substitutions/synonyms = answer
        Example: "Doctor's fish" → D + ACE = DACE

        MUST have 2+ components - single synonym is not a charade.
        """
        tokens = self._tokenize(clue_text)
        answer_norm = self._norm_letters(answer).upper()
        answer_len = len(answer_norm)

        if debug:
            print(f"\n  CHARADE: Building {answer_norm} from components")

        # Get all possible contributions from each token
        contributions = {}  # token_idx -> [(letters, source_type, details), ...]

        # Find first/last letter indicators first
        first_letter_indicator_positions = set()
        last_letter_indicator_positions = set()
        for i, token in enumerate(tokens):
            token_lower = token.lower()
            if token_lower in self.first_letter_indicators:
                first_letter_indicator_positions.add(i)
            if token_lower in self.last_letter_indicators:
                last_letter_indicator_positions.add(i)

        for i, token in enumerate(tokens):
            token_contribs = []
            token_lower = token.lower()

            # Skip common link words and indicators
            if token_lower in {'a', 'an', 'the', 'of', 'to', 'in', 'for', 'with', 'by',
                               'and', 'is', 'has'}:
                continue
            if token_lower in self.first_letter_indicators or token_lower in self.last_letter_indicators:
                continue

            # 1. Substitutions from wordplay table
            subs = self.lookup_substitution(token)
            for letters, category in subs:
                letters_norm = self._norm_letters(letters).upper()
                if letters_norm and letters_norm in answer_norm:
                    token_contribs.append(
                        (letters_norm, 'substitution', {'category': category}))

            # 2. Synonyms that appear in answer (but NOT full answer - that's just definition)
            for length in range(1, min(answer_len - 1,
                                       6) + 1):  # Max 6 letters, must be shorter than answer
                syns = self.lookup_synonyms(token, length)
                for syn in syns:
                    syn_norm = self._norm_letters(syn).upper()
                    if syn_norm in answer_norm and len(syn_norm) < answer_len:
                        token_contribs.append((syn_norm, 'synonym', {'synonym': syn}))

            # 3. Literal single letters (only for very short words like 'a', 'I', 'O')
            token_letters = self._norm_letters(token)
            if len(token_letters) == 1:
                letter = token_letters.upper()
                if letter in answer_norm:
                    token_contribs.append((letter, 'literal', {}))

            # 4. First letter - ONLY if there's a first_letter indicator nearby
            if first_letter_indicator_positions:
                nearby_first = any(
                    abs(i - ind_pos) <= 2 for ind_pos in first_letter_indicator_positions)
                if nearby_first and token_letters:
                    first = token_letters[0].upper()
                    if first in answer_norm:
                        token_contribs.append((first, 'first_letter', {}))

            # 5. Last letter - ONLY if there's a last_letter indicator nearby
            if last_letter_indicator_positions:
                nearby_last = any(
                    abs(i - ind_pos) <= 2 for ind_pos in last_letter_indicator_positions)
                if nearby_last and token_letters:
                    last = token_letters[-1].upper()
                    if last in answer_norm:
                        token_contribs.append((last, 'last_letter', {}))

            if token_contribs:
                contributions[i] = token_contribs

        if debug:
            print(f"    Contributions: {len(contributions)} tokens have potential")

        # Need at least 2 contributing tokens for a charade
        if len(contributions) < 2:
            return None

        # Try to chain contributions to build answer
        # Use recursive search with memoization
        result = self._build_charade(answer_norm, contributions, tokens, debug)

        if result:
            components, used_indices = result

            # MUST have 2+ components - single synonym is NOT a charade
            if len(components) < 2:
                return None

            # Find proper definition using synonym lookup
            definition_words = self._find_definition(tokens, answer, used_indices)

            # Require that we actually found a valid definition
            if not definition_words:
                if debug:
                    print(f"    REJECTED: No valid definition found")
                return None

            derivation_parts = []
            for comp in components:
                src = comp['source']
                letters = comp['letters']
                op = comp['operation']
                if op == 'substitution':
                    derivation_parts.append(f"{src}→{letters}")
                elif op == 'synonym':
                    derivation_parts.append(f"{src}={letters}")
                else:
                    derivation_parts.append(f"{src}({op})→{letters}")

            derivation = ' + '.join(derivation_parts) + f" = {answer_norm}"

            if debug:
                print(f"    FOUND: {derivation}")
                print(f"    Definition: {definition_words}")

            return WordplayEvidence(
                evidence_type='charade',
                answer=answer,
                components=components,
                definition_words=definition_words,
                indicator_words=[],
                confidence=0.85,
                derivation=derivation
            )

        return None

    def _build_charade(self, target: str, contributions: Dict, tokens: List[str],
                       debug: bool = False, depth: int = 0, max_depth: int = 6) -> \
    Optional[Tuple[List[Dict], Set[int]]]:
        """
        Recursively build target from contributions.
        Returns (components, used_indices) or None.

        Limited to max_depth to prevent exponential blowup.
        """
        if not target:
            return ([], set())

        # Prevent infinite recursion
        if depth > max_depth:
            return None

        # Try each contribution that starts with the right letter
        for idx, contribs in contributions.items():
            for letters, source_type, details in contribs:
                if target.startswith(letters):
                    remaining = target[len(letters):]

                    # Remove this contribution and recurse
                    remaining_contribs = {k: v for k, v in contributions.items() if
                                          k != idx}

                    sub_result = self._build_charade(remaining, remaining_contribs,
                                                     tokens, debug, depth + 1, max_depth)

                    if sub_result is not None:
                        sub_components, sub_indices = sub_result

                        component = {
                            'source': tokens[idx],
                            'operation': source_type,
                            'letters': letters,
                            'indicator': None,
                            **details
                        }

                        return ([component] + sub_components, {idx} | sub_indices)

        return None

    # =========================================================================
    # MAIN DETECTION
    # =========================================================================

    def detect(self, clue_text: str, answer: str, debug: bool = False) -> Optional[
        WordplayEvidence]:
        """
        Try all detection methods and return first match.
        Order: Reversal, Deletion, Acrostic, Charade
        """
        if debug:
            print(f"\n{'=' * 60}")
            print(f"GENERAL WORDPLAY: {clue_text}")
            print(f"ANSWER: {answer}")

        # Try reversal first (simplest)
        evidence = self.detect_reversal(clue_text, answer, debug)
        if evidence and evidence.is_complete:
            return evidence

        # Try deletion
        evidence = self.detect_deletion(clue_text, answer, debug)
        if evidence and evidence.is_complete:
            return evidence

        # Try acrostic
        evidence = self.detect_acrostic(clue_text, answer, debug)
        if evidence and evidence.is_complete:
            return evidence

        # Try charade (most complex)
        evidence = self.detect_charade(clue_text, answer, debug)
        if evidence and evidence.is_complete:
            return evidence

        return None


# =============================================================================
# TEST
# =============================================================================

if __name__ == "__main__":
    import sys

    DB_PATH = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"

    detector = GeneralWordplayDetector(DB_PATH)

    # Test cases from the unsolved cohort
    test_cases = [
        ("Boss has hit back (4)", "KNOB"),  # reversal: hit→BONK→KNOB
        ("Artist drops in to see old man (5)", "PATER"),  # deletion: PAINTER - IN
        ("Doctor's fish (4)", "DACE"),  # charade: D + ACE
        ("Fuss caused by commercial, then nothing (3)", "ADO"),  # charade: AD + O
    ]

    print("\n" + "=" * 60)
    print("GENERAL WORDPLAY DETECTOR TEST")
    print("=" * 60)

    for clue, answer in test_cases:
        print(f"\nClue: {clue}")
        print(f"Answer: {answer}")

        evidence = detector.detect(clue, answer, debug=True)

        if evidence:
            print(f"\n✓ SOLVED: {evidence.evidence_type}")
            print(f"  Derivation: {evidence.derivation}")
            print(f"  Definition: {evidence.definition_words}")
        else:
            print(f"\n✗ FAILED")

    detector.close()