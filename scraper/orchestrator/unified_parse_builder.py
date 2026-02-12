#!/usr/bin/env python3
"""
Unified Parse Builder for Cryptic Crossword Explanations

Core principle: We KNOW the answer. We need to find the combination of
operations that builds it from the wordplay tokens.

Every token in the wordplay portion must be one of:
- Indicator (marks an operation on adjacent fodder)
- Fodder (contributes letters via an operation)
- Link word (ignored, contributes nothing)

We try combinations until: contributed_letters == answer
"""

import re
import sqlite3
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field
from collections import Counter
from itertools import permutations, combinations


# =============================================================================
# DATA STRUCTURES
# =============================================================================

@dataclass
class LetterContribution:
    """A contribution of letters to the answer."""
    source_words: List[str]  # Words from clue that produced this
    operation: str  # How letters were derived
    letters: str  # The actual letters contributed
    indicator: Optional[str]  # Indicator word that signaled this operation
    details: Dict = field(default_factory=dict)  # Extra info (e.g., category)

    def __repr__(self):
        return f"{self.source_words} --[{self.operation}]--> {self.letters}"


@dataclass
class ParseResult:
    """A complete parse of a clue's wordplay."""
    answer: str
    definition_words: List[str]
    contributions: List[LetterContribution]
    link_words: List[str]
    unparsed_words: List[str]
    confidence: float

    @property
    def is_complete(self) -> bool:
        """Check if contributions fully explain the answer."""
        built = ''.join(c.letters for c in self.contributions).upper()
        return built == self.answer.upper().replace(' ', '').replace('-', '')

    @property
    def derivation(self) -> str:
        """Human-readable derivation string."""
        parts = []
        container_outer = None
        container_inner = None

        for c in self.contributions:
            if c.operation == 'anagram':
                parts.append(f"anagram({'+'.join(c.source_words)})")
            elif c.operation == 'reversal':
                parts.append(f"rev({c.source_words[0]}→{c.letters})")
            elif c.operation == 'substitution':
                parts.append(f"{c.source_words[0]}→{c.letters}")
            elif c.operation == 'first_letter':
                parts.append(f"first({c.source_words[0]})→{c.letters}")
            elif c.operation == 'last_letter':
                parts.append(f"last({c.source_words[0]})→{c.letters}")
            elif c.operation == 'acrostic':
                parts.append(f"acrostic({'+'.join(c.source_words)})→{c.letters}")
            elif c.operation == 'literal':
                parts.append(c.letters)
            elif c.operation == 'synonym':
                parts.append(f"{c.source_words[0]}={c.letters}")
            elif c.operation == 'container_outer':
                container_outer = c
            elif c.operation == 'container_inner':
                container_inner = c
            else:
                parts.append(f"{c.operation}({c.letters})")

        # Format container specially
        if container_outer and container_inner:
            prefix = container_inner.details.get('prefix', '')
            suffix = container_inner.details.get('suffix', '')
            parts.append(
                f"{container_outer.source_words[0]}→{prefix}({container_inner.letters}){suffix} [container]")

        return ' + '.join(parts) + f" = {self.answer}"


# =============================================================================
# LINK WORDS
# =============================================================================

LINK_WORDS = {
    'to', 'of', 'in', 'for', 'with', 'by', 'from', 'a', 'an', 'the',
    'and', 'is', 'are', 'needs', 'about', 'on', 'after', 'at', 'as', 'or',
    'be', 'being', 'been', 'has', 'have', 'had', 'having',
    'was', 'were', 'will', 'would', 'could', 'should', 'must', 'may', 'might',
    'gets', 'get', 'getting', 'got', 'makes', 'make', 'making', 'made',
    'gives', 'give', 'given', 'giving', 'sees', 'see', 'seen', 'seeing',
    'but', 'that', 'which', 'when', 'where', 'while', 'so', 'yet',
    'this', 'these', 'those', 'such', 'one', 'ones', 'some', 'any', 'all',
    'here', 'there', 'into', 'onto', 'within', 'without',
    'if', 'how', 'why', 'who', 'whom', 'you', 'it', 'its',
    'producing', 'making', 'creating', 'forming', 'giving', 'showing',
}


# =============================================================================
# UNIFIED PARSE BUILDER
# =============================================================================

class UnifiedParseBuilder:
    """
    Builds explanations by finding operation combinations that produce the answer.
    """

    def __init__(self, db_path: str):
        self.db_path = db_path
        self._conn = None

        # Indicator sets by type
        self.anagram_indicators: Set[str] = set()
        self.reversal_indicators: Set[str] = set()
        self.container_indicators: Set[str] = set()
        self.deletion_indicators: Set[str] = set()
        self.first_letter_indicators: Set[str] = set()
        self.last_letter_indicators: Set[str] = set()
        self.acrostic_indicators: Set[str] = set()
        self.hidden_indicators: Set[str] = set()

        # Substitution cache
        self._substitution_cache: Dict[str, List[Tuple[str, str]]] = {}
        self._synonym_cache: Dict[str, List[str]] = {}

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
        """Load all indicators from database."""
        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("""
            SELECT word, wordplay_type, subtype 
            FROM indicators
        """)

        for word, wtype, subtype in cur.fetchall():
            word_lower = word.lower().strip()

            if wtype == 'anagram':
                self.anagram_indicators.add(word_lower)
            elif wtype == 'reversal':
                self.reversal_indicators.add(word_lower)
            elif wtype == 'insertion':
                self.container_indicators.add(word_lower)
            elif wtype == 'deletion':
                self.deletion_indicators.add(word_lower)
            elif wtype == 'hidden':
                self.hidden_indicators.add(word_lower)
            elif wtype == 'parts':
                # Parts indicators have subtypes
                if subtype in ('first_letter', 'initial', 'head', 'start', 'opener'):
                    self.first_letter_indicators.add(word_lower)
                elif subtype in ('last_letter', 'final', 'tail', 'end', 'closer'):
                    self.last_letter_indicators.add(word_lower)
                elif subtype in ('acrostic', 'initials', 'heads', 'starts'):
                    self.acrostic_indicators.add(word_lower)
                else:
                    # Default parts to first letter
                    self.first_letter_indicators.add(word_lower)

        # Add common acrostic indicators that might be missing
        self.acrostic_indicators.update({
            'initially', 'heads', 'leaders', 'starters', 'openers',
            'first letters', 'capitals', 'at first', 'to start'
        })

        # Add common first/last letter indicators
        self.first_letter_indicators.update({
            'first', 'head', 'start', 'opening', 'initially', 'at first',
            'front', 'lead', 'top', 'head of', 'start of', 'beginning'
        })
        self.last_letter_indicators.update({
            'last', 'finally', 'end', 'tail', 'closing', 'at last',
            'back', 'rear', 'finish', 'end of', 'lastly', 'ultimate'
        })

        print(f"Loaded indicators:")
        print(f"  Anagram: {len(self.anagram_indicators)}")
        print(f"  Reversal: {len(self.reversal_indicators)}")
        print(f"  Container: {len(self.container_indicators)}")
        print(f"  Deletion: {len(self.deletion_indicators)}")
        print(f"  First letter: {len(self.first_letter_indicators)}")
        print(f"  Last letter: {len(self.last_letter_indicators)}")
        print(f"  Acrostic: {len(self.acrostic_indicators)}")

    def _lookup_indicator(self, phrase: str) -> Optional[Tuple[str, str]]:
        """
        Look up ANY phrase in the indicators table.
        Returns (wordplay_type, subtype) or None.
        No restrictions on length - just query the database.
        """
        phrase_clean = phrase.lower().strip()

        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("""
            SELECT wordplay_type, subtype
            FROM indicators
            WHERE LOWER(word) = ?
        """, (phrase_clean,))

        result = cur.fetchone()
        if result:
            return (result[0], result[1])
        return None

    def _lookup_substitution(self, word: str) -> List[Tuple[str, str]]:
        """Look up substitutions for a word. Returns [(letters, category), ...]"""
        word_clean = word.lower().strip('.,;:!?"\'-')

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

    def _lookup_synonym(self, word: str, target_len: int = None) -> List[str]:
        """Look up synonyms for a word. Optionally filter by length."""
        word_clean = word.lower().strip('.,;:!?"\'-')

        cache_key = f"{word_clean}_{target_len}"
        if cache_key in self._synonym_cache:
            return self._synonym_cache[cache_key]

        conn = self._get_conn()
        cur = conn.cursor()

        # Check synonyms_pairs
        cur.execute("""
            SELECT synonym FROM synonyms_pairs
            WHERE LOWER(word) = ?
        """, (word_clean,))

        results = [row[0] for row in cur.fetchall()]

        # Also check definition_answers_augmented
        cur.execute("""
            SELECT answer FROM definition_answers_augmented
            WHERE LOWER(definition) = ?
        """, (word_clean,))

        results.extend([row[0] for row in cur.fetchall()])

        # Filter by length if specified
        if target_len:
            results = [r for r in results if
                       len(r.replace(' ', '').replace('-', '')) == target_len]

        self._synonym_cache[cache_key] = results
        return results

    def _tokenize(self, text: str) -> List[str]:
        """Split text into tokens, preserving apostrophes."""
        tokens = text.split()
        cleaned = []
        for token in tokens:
            clean = token.strip('.,;:!?"()[]{}')
            if clean:
                cleaned.append(clean)
        return cleaned

    def _norm_letters(self, s: str) -> str:
        """Extract just letters, lowercase."""
        return re.sub(r'[^a-zA-Z]', '', s).lower()

    def _is_anagram(self, s1: str, s2: str) -> bool:
        """Check if two strings are anagrams."""
        return Counter(self._norm_letters(s1)) == Counter(self._norm_letters(s2))

    # =========================================================================
    # OPERATION EXTRACTORS
    # =========================================================================

    def _try_substitution(self, word: str, needed: str) -> Optional[LetterContribution]:
        """Try to get letters from substitution table."""
        subs = self._lookup_substitution(word)
        needed_lower = needed.lower()

        for letters, category in subs:
            letters_norm = self._norm_letters(letters)
            # Check if this substitution's letters are all in what we need
            if letters_norm and letters_norm in needed_lower:
                return LetterContribution(
                    source_words=[word],
                    operation='substitution',
                    letters=letters_norm.upper(),
                    indicator=None,
                    details={'category': category}
                )
        return None

    def _try_first_letter(self, word: str, indicator: str = None) -> Optional[
        LetterContribution]:
        """Extract first letter of word."""
        letters = self._norm_letters(word)
        if letters:
            return LetterContribution(
                source_words=[word],
                operation='first_letter',
                letters=letters[0].upper(),
                indicator=indicator
            )
        return None

    def _try_last_letter(self, word: str, indicator: str = None) -> Optional[
        LetterContribution]:
        """Extract last letter of word."""
        letters = self._norm_letters(word)
        if letters:
            return LetterContribution(
                source_words=[word],
                operation='last_letter',
                letters=letters[-1].upper(),
                indicator=indicator
            )
        return None

    def _try_acrostic(self, words: List[str], indicator: str = None) -> Optional[
        LetterContribution]:
        """Extract first letters of multiple words."""
        letters = ''.join(
            self._norm_letters(w)[0] if self._norm_letters(w) else '' for w in words)
        if letters:
            return LetterContribution(
                source_words=words,
                operation='acrostic',
                letters=letters.upper(),
                indicator=indicator
            )
        return None

    def _try_reversal(self, word: str, indicator: str = None) -> Optional[
        LetterContribution]:
        """Reverse the letters of a word."""
        letters = self._norm_letters(word)
        if letters:
            return LetterContribution(
                source_words=[word],
                operation='reversal',
                letters=letters[::-1].upper(),
                indicator=indicator
            )
        return None

    def _try_reversal_of_substitution(self, word: str, needed: str,
                                      indicator: str = None) -> Optional[
        LetterContribution]:
        """Try: word → substitution → reverse."""
        subs = self._lookup_substitution(word)
        needed_lower = needed.lower()

        for letters, category in subs:
            letters_norm = self._norm_letters(letters)
            reversed_letters = letters_norm[::-1]
            if reversed_letters and reversed_letters in needed_lower:
                return LetterContribution(
                    source_words=[word],
                    operation='reversal',
                    letters=reversed_letters.upper(),
                    indicator=indicator,
                    details={'original': letters_norm, 'category': category}
                )
        return None

    def _try_anagram(self, words: List[str], target: str, indicator: str = None) -> \
    Optional[LetterContribution]:
        """Check if words anagram to target."""
        fodder = ''.join(self._norm_letters(w) for w in words)
        if self._is_anagram(fodder, target):
            return LetterContribution(
                source_words=words,
                operation='anagram',
                letters=target.upper(),
                indicator=indicator
            )
        return None

    def _try_literal(self, word: str) -> Optional[LetterContribution]:
        """Use the word's letters literally (for short words like 'a', 'I')."""
        letters = self._norm_letters(word)
        if len(letters) <= 2:  # Only for very short words
            return LetterContribution(
                source_words=[word],
                operation='literal',
                letters=letters.upper(),
                indicator=None
            )
        return None

    def _try_synonym(self, word: str, target: str) -> Optional[LetterContribution]:
        """Check if word has target as a synonym."""
        synonyms = self._lookup_synonym(word, len(target))
        target_upper = target.upper().replace(' ', '').replace('-', '')

        for syn in synonyms:
            if syn.upper().replace(' ', '').replace('-', '') == target_upper:
                return LetterContribution(
                    source_words=[word],
                    operation='synonym',
                    letters=target_upper,
                    indicator=None
                )
        return None

    def _lookup_phrase_substitution(self, words: List[str]) -> List[Tuple[str, str]]:
        """Look up multi-word phrase substitutions like 'the German' → DER."""
        phrase = ' '.join(w.lower().strip('.,;:!?"\'-') for w in words)

        if phrase in self._substitution_cache:
            return self._substitution_cache[phrase]

        conn = self._get_conn()
        cur = conn.cursor()

        cur.execute("""
            SELECT substitution, category
            FROM wordplay
            WHERE LOWER(indicator) = ?
        """, (phrase,))

        results = [(row[0], row[1]) for row in cur.fetchall()]
        self._substitution_cache[phrase] = results

        # DEBUG
        if results:
            print(f"      DB FOUND: '{phrase}' → {results}")

        return results

    def _try_container(self, outer_letters: str, inner_letters: str, answer: str) -> \
    Optional[Dict]:
        """
        Try container: outer_letters wrap AROUND inner_letters.
        Example: DER contains EAR → D(EAR)ER = DEARER
        """
        answer_upper = answer.upper().replace(' ', '').replace('-', '')
        outer_upper = outer_letters.upper()
        inner_upper = inner_letters.upper()

        # Try each split of outer as prefix + suffix around inner
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

    def _try_insertion(self, base_letters: str, insert_letters: str, answer: str) -> \
    Optional[Dict]:
        """
        Try insertion: insert_letters go INTO base_letters.
        Example: EAR inserted into DER → D(EAR)ER = DEARER
        """
        # Container and insertion are the same operation, just different perspective
        return self._try_container(base_letters, insert_letters, answer)
        return None

    # =========================================================================
    # MAIN PARSE BUILDER
    # =========================================================================

    def parse(self, clue_text: str, answer: str, definition_words: List[str] = None,
              debug: bool = False) -> Optional[ParseResult]:
        """
        Parse a clue to explain how the answer is derived.

        Algorithm:
        1. Find definition (from edges of clue)
        2. For remaining words, find indicator + adjacent fodder
        3. Check if fodder letters are in answer, eliminate if so
        4. Repeat until done
        5. Report what's found and what's left
        """
        answer_norm = self._norm_letters(answer).upper()
        tokens = self._tokenize(clue_text)

        if debug:
            print(f"\n{'=' * 60}")
            print(f"PARSING: {clue_text}")
            print(f"ANSWER: {answer} ({answer_norm})")
            print(f"TOKENS: {tokens}")

        # Step 1: Find definition (if not provided)
        if definition_words is None:
            definition_words = self._find_definition(tokens, answer_norm, debug)

        if debug:
            print(f"DEFINITION: {definition_words}")

        # Remove definition words from working set
        def_set = set(w.lower() for w in definition_words)
        working_tokens = [(i, t) for i, t in enumerate(tokens) if
                          t.lower() not in def_set]
        remaining_letters = answer_norm  # Letters still to explain

        if debug:
            print(f"WORKING TOKENS: {[t for i, t in working_tokens]}")
            print(f"LETTERS TO EXPLAIN: {remaining_letters}")

        # Step 2: Process remaining words
        contributions = []
        used_positions = set()

        # Keep trying until no more progress
        made_progress = True
        while made_progress and remaining_letters:
            made_progress = False

            # Look for indicators in unused tokens
            for pos, token in working_tokens:
                if pos in used_positions:
                    continue

                # Check if this token (or phrase starting here) is an indicator
                indicator_info = self._find_indicator_at(working_tokens, pos,
                                                         used_positions, debug)
                if not indicator_info:
                    continue

                ind_type, ind_phrase, ind_positions = indicator_info

                if debug:
                    print(
                        f"  FOUND INDICATOR: '{ind_phrase}' ({ind_type}) at positions {ind_positions}")

                # Find adjacent fodder
                fodder_result = self._find_adjacent_fodder(
                    working_tokens, ind_positions, used_positions,
                    remaining_letters, ind_type, debug
                )

                if fodder_result:
                    fodder_word, fodder_letters, fodder_positions, operation = fodder_result

                    if debug:
                        print(f"  FODDER: '{fodder_word}' → {fodder_letters}")

                    # Record contribution
                    contributions.append(LetterContribution(
                        source_words=[fodder_word],
                        operation=operation,
                        letters=fodder_letters,
                        indicator=ind_phrase
                    ))

                    # Mark as used
                    used_positions.update(ind_positions)
                    used_positions.update(fodder_positions)

                    # Remove letters from remaining
                    remaining_letters = self._remove_letters(remaining_letters,
                                                             fodder_letters)

                    if debug:
                        print(f"  REMAINING LETTERS: {remaining_letters}")

                    made_progress = True
                    break  # Restart the loop

        # Step 3: Try simple substitutions for remaining tokens
        for pos, token in working_tokens:
            if pos in used_positions:
                continue

            # Try substitution
            subs = self._lookup_substitution(token)
            for letters, category in subs:
                letters_norm = self._norm_letters(letters).upper()
                if letters_norm and self._letters_exist_in(letters_norm,
                                                           remaining_letters):
                    contributions.append(LetterContribution(
                        source_words=[token],
                        operation='substitution',
                        letters=letters_norm,
                        indicator=None,
                        details={'category': category}
                    ))
                    used_positions.add(pos)
                    remaining_letters = self._remove_letters(remaining_letters,
                                                             letters_norm)
                    if debug:
                        print(
                            f"  SIMPLE SUB: {token} → {letters_norm}, remaining: {remaining_letters}")
                    break

        # Step 4: Report results
        unparsed = [t for pos, t in working_tokens if pos not in used_positions]

        if debug:
            print(f"  UNPARSED WORDS: {unparsed}")
            print(f"  UNEXPLAINED LETTERS: {remaining_letters}")

        # Build result
        if contributions:
            return ParseResult(
                answer=answer,
                definition_words=definition_words,
                contributions=contributions,
                link_words=[],
                unparsed_words=unparsed,
                confidence=1.0 if not remaining_letters else (len(answer_norm) - len(
                    remaining_letters)) / len(answer_norm)
            )

        return None

    def _find_definition(self, tokens: List[str], answer_norm: str,
                         debug: bool = False) -> List[str]:
        """
        Find definition by checking edges of clue against definition_answers_augmented.
        Start with single words at edges, expand until found.
        """
        strip_chars = '.,;:!?"\'-'
        conn = self._get_conn()
        cur = conn.cursor()

        # Get all definitions that map to this answer
        cur.execute("""
            SELECT LOWER(definition) FROM definition_answers_augmented
            WHERE UPPER(REPLACE(REPLACE(answer, ' ', ''), '-', '')) = ?
        """, (answer_norm,))
        known_definitions = set(row[0] for row in cur.fetchall())

        # Also check synonyms_pairs
        cur.execute("""
            SELECT LOWER(word) FROM synonyms_pairs
            WHERE UPPER(REPLACE(REPLACE(synonym, ' ', ''), '-', '')) = ?
        """, (answer_norm,))
        known_definitions.update(row[0] for row in cur.fetchall())

        if debug:
            print(
                f"  KNOWN DEFINITIONS FOR {answer_norm}: {len(known_definitions)} found")

        # Check from edges, expanding outward
        # Try first N words, then last N words, increasing N
        for length in range(1, len(tokens)):
            # First N words
            phrase = ' '.join(w.lower().strip(strip_chars) for w in tokens[:length])
            if phrase in known_definitions:
                return tokens[:length]

            # Last N words
            phrase = ' '.join(w.lower().strip(strip_chars) for w in tokens[-length:])
            if phrase in known_definitions:
                return tokens[-length:]

        return []

    def _find_indicator_at(self, working_tokens: List[Tuple[int, str]], start_pos: int,
                           used_positions: set, debug: bool = False) -> Optional[
        Tuple[str, str, set]]:
        """
        Check if there's an indicator starting at given position.
        Returns (indicator_type, phrase, positions_used) or None.
        """
        strip_chars = '.,;:!?"\'-'

        # Build list of available tokens from start_pos onward
        available = [(p, t) for p, t in working_tokens if
                     p >= start_pos and p not in used_positions]

        # Try phrases of decreasing length
        for length in range(len(available), 0, -1):
            phrase_tokens = available[:length]
            positions = set(p for p, t in phrase_tokens)
            phrase = ' '.join(t.lower().strip(strip_chars) for p, t in phrase_tokens)

            result = self._lookup_indicator(phrase)
            if result:
                wtype, subtype = result
                if wtype == 'insertion':
                    wtype = 'container'
                elif wtype == 'parts':
                    if subtype in ('first_letter', 'initial', 'head', 'start', 'opener'):
                        wtype = 'first_letter'
                    elif subtype in ('last_letter', 'final', 'tail', 'end', 'closer'):
                        wtype = 'last_letter'
                    else:
                        wtype = 'first_letter'

                return (wtype, phrase, positions)

        return None

    def _find_adjacent_fodder(self, working_tokens: List[Tuple[int, str]],
                              ind_positions: set,
                              used_positions: set, remaining_letters: str, ind_type: str,
                              debug: bool = False) -> Optional[Tuple[str, str, set, str]]:
        """
        Find fodder adjacent to indicator that produces letters in the answer.
        Returns (fodder_word, letters, positions, operation) or None.
        """
        strip_chars = '.,;:!?"\'-'

        # Find positions adjacent to indicator
        min_ind = min(ind_positions)
        max_ind = max(ind_positions)

        # Get adjacent unused tokens
        adjacent = [(p, t) for p, t in working_tokens
                    if p not in used_positions
                    and p not in ind_positions
                    and (p == min_ind - 1 or p == max_ind + 1 or abs(
                p - min_ind) <= 2 or abs(p - max_ind) <= 2)]

        if debug:
            print(f"    ADJACENT TOKENS: {[t for p, t in adjacent]}")

        # For CONTAINER indicators, we need TWO pieces of fodder
        if ind_type == 'container':
            result = self._find_container_fodder(adjacent, remaining_letters, debug)
            if result:
                return result

        # Try single words first
        for pos, token in adjacent:
            subs = self._lookup_substitution(token)
            for letters, category in subs:
                letters_norm = self._norm_letters(letters).upper()
                if letters_norm and self._letters_exist_in(letters_norm,
                                                           remaining_letters):
                    if debug:
                        print(f"    SINGLE WORD MATCH: {token} → {letters_norm}")
                    return (token, letters_norm, {pos}, 'substitution')

        # Try two-word phrases
        for i, (pos1, token1) in enumerate(adjacent):
            for pos2, token2 in adjacent[i + 1:]:
                if abs(pos1 - pos2) == 1:  # Adjacent to each other
                    phrase = f"{token1} {token2}"
                    if debug:
                        print(
                            f"    CHECKING PHRASE: '{phrase}' (positions {pos1}, {pos2})")
                    subs = self._lookup_phrase_substitution([token1, token2])
                    if debug and subs:
                        print(f"    PHRASE LOOKUP: '{phrase}' → {subs}")
                    for letters, category in subs:
                        letters_norm = self._norm_letters(letters).upper()
                        if letters_norm and self._letters_exist_in(letters_norm,
                                                                   remaining_letters):
                            if debug:
                                print(f"    PHRASE MATCH: {phrase} → {letters_norm}")
                            return (phrase, letters_norm, {pos1, pos2}, 'substitution')

        return None

    def _find_container_fodder(self, adjacent: List[Tuple[int, str]],
                               remaining_letters: str,
                               debug: bool = False) -> Optional[
        Tuple[str, str, set, str]]:
        """
        Find TWO pieces of fodder for container: outer contains inner = answer.
        Returns combined result showing container operation.

        Also tries INFERENCE: if we know one piece from DB, compute what the other must be.
        """
        # Collect all possible substitutions from adjacent tokens
        all_subs = []  # [(positions, phrase, letters, category)]

        # Single word substitutions
        for pos, token in adjacent:
            subs = self._lookup_substitution(token)
            for letters, category in subs:
                letters_norm = self._norm_letters(letters).upper()
                if letters_norm:
                    all_subs.append(({pos}, token, letters_norm, category))

        # Two-word phrase substitutions
        for i, (pos1, token1) in enumerate(adjacent):
            for pos2, token2 in adjacent[i + 1:]:
                if abs(pos1 - pos2) == 1:
                    subs = self._lookup_phrase_substitution([token1, token2])
                    for letters, category in subs:
                        letters_norm = self._norm_letters(letters).upper()
                        if letters_norm:
                            all_subs.append(
                                ({pos1, pos2}, f"{token1} {token2}", letters_norm,
                                 category))

        if debug:
            print(
                f"    CONTAINER - all substitutions found: {[(p, l) for _, p, l, _ in all_subs]}")

        # Try all pairs: one as outer, one as inner
        for positions1, phrase1, letters1, cat1 in all_subs:
            for positions2, phrase2, letters2, cat2 in all_subs:
                if positions1 & positions2:  # Can't use same positions
                    continue

                # Try letters1 as outer, letters2 as inner
                if self._check_container(letters1, letters2, remaining_letters):
                    combined_positions = positions1 | positions2
                    combined_phrase = f"{phrase1}({letters1}) contains {phrase2}({letters2})"
                    if debug:
                        print(
                            f"    CONTAINER MATCH: {combined_phrase} = {remaining_letters}")
                    return (combined_phrase, remaining_letters, combined_positions,
                            'container')

        # INFERENCE: Try each known substitution as outer, infer inner from remaining word
        for positions1, phrase1, letters1, cat1 in all_subs:
            # Find remaining adjacent tokens not used by this substitution
            remaining_adjacent = [(p, t) for p, t in adjacent if p not in positions1]

            if not remaining_adjacent:
                continue

            # Try this as outer - what inner(s) would we need?
            inner_possibilities = self._infer_inner_for_container(letters1,
                                                                  remaining_letters)

            if inner_possibilities and debug:
                print(
                    f"    INFERENCE: If {phrase1}({letters1}) is outer, inner could be: {inner_possibilities}")

            if inner_possibilities:
                # The remaining words must provide these letters
                # Take the first remaining word as the source
                for pos2, token2 in remaining_adjacent:
                    combined_positions = positions1 | {pos2}
                    # Report all valid possibilities
                    if len(inner_possibilities) == 1:
                        inner = inner_possibilities[0]
                        combined_phrase = f"{phrase1}({letters1}) contains {token2}({inner})"
                    else:
                        options = " or ".join(inner_possibilities)
                        combined_phrase = f"{phrase1}({letters1}) contains {token2}({options})"
                    if debug:
                        print(
                            f"    INFERRED CONTAINER: {combined_phrase} = {remaining_letters}")
                    return (combined_phrase, remaining_letters, combined_positions,
                            'container')

        return None

    def _infer_inner_for_container(self, outer_letters: str, answer: str) -> List[str]:
        """
        Given outer letters and answer, compute ALL possible inner letters.
        Example: outer=DER, answer=DEARER → inner could be EAR or ARE
        Returns list of all valid possibilities (that are actual words).
        """
        answer_upper = answer.upper()
        outer_upper = outer_letters.upper()

        possibilities = []

        # Try each split of outer as prefix + suffix
        for i in range(len(outer_upper) + 1):
            prefix = outer_upper[:i]
            suffix = outer_upper[i:]

            # Answer must start with prefix and end with suffix
            if answer_upper.startswith(prefix) and answer_upper.endswith(suffix):
                # Inner is what's in between
                if len(prefix) + len(suffix) <= len(answer_upper):
                    if suffix:
                        inner = answer_upper[len(prefix):len(answer_upper) - len(suffix)]
                    else:
                        inner = answer_upper[len(prefix):]

                    if inner and len(inner) >= 2:  # Must have at least 2 letters
                        # Check if it's a valid word
                        if self._is_valid_word(inner):
                            possibilities.append(inner)

        return possibilities

    def _is_valid_word(self, word: str) -> bool:
        """Check if word exists in our wordlist/synonyms."""
        word_lower = word.lower()

        conn = self._get_conn()
        cur = conn.cursor()

        # Check if it's in synonyms_pairs (as word or synonym)
        cur.execute("""
            SELECT 1 FROM synonyms_pairs 
            WHERE LOWER(word) = ? OR LOWER(synonym) = ?
            LIMIT 1
        """, (word_lower, word_lower))

        if cur.fetchone():
            return True

        # Check if it's in definition_answers_augmented
        cur.execute("""
            SELECT 1 FROM definition_answers_augmented
            WHERE LOWER(definition) = ? OR LOWER(answer) = ?
            LIMIT 1
        """, (word_lower, word_lower))

        if cur.fetchone():
            return True

        # Check if it's an indicator
        cur.execute("""
            SELECT 1 FROM indicators
            WHERE LOWER(word) = ?
            LIMIT 1
        """, (word_lower,))

        if cur.fetchone():
            return True

        return False

        return None

    def _check_container(self, outer_letters: str, inner_letters: str,
                         answer: str) -> bool:
        """
        Check if outer contains inner = answer.
        Example: DER contains EAR → D(EAR)ER = DEARER
        """
        answer_upper = answer.upper()
        outer_upper = outer_letters.upper()
        inner_upper = inner_letters.upper()

        # Try each split of outer as prefix + suffix around inner
        for i in range(len(outer_upper) + 1):
            prefix = outer_upper[:i]
            suffix = outer_upper[i:]
            combined = prefix + inner_upper + suffix
            if combined == answer_upper:
                return True

        return False

    def _letters_exist_in(self, letters: str, source: str) -> bool:
        """Check if all letters exist in source (each used once)."""
        source_list = list(source)
        for c in letters:
            if c in source_list:
                source_list.remove(c)
            else:
                return False
        return True

    def _remove_letters(self, source: str, to_remove: str) -> str:
        """Remove letters from source (each letter removed once)."""
        result = list(source)
        for c in to_remove:
            if c in result:
                result.remove(c)
        return ''.join(result)


# =============================================================================
# TEST
# =============================================================================

def load_random_clues(db_path: str, count: int = 50) -> List[Dict]:
    """Load random clues from clues_master.db."""
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    cur = conn.cursor()

    cur.execute("""
        SELECT id, clue_text, answer, enumeration, wordplay_type
        FROM clues
        WHERE answer IS NOT NULL
        AND clue_text IS NOT NULL
        AND LENGTH(answer) >= 3
        ORDER BY RANDOM()
        LIMIT ?
    """, (count,))

    clues = [dict(row) for row in cur.fetchall()]
    conn.close()
    return clues


if __name__ == "__main__":
    import sys
    from datetime import datetime

    # Paths
    CRYPTIC_DB = r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"
    CLUES_DB = r"C:\Users\shute\PycharmProjects\AI_Solver\data\clues_master.db"
    OUTPUT_DIR = r"C:\Users\shute\PycharmProjects\AI_Solver\Solver\orchestrator\logs"

    # Parse command line args
    count = 50
    for arg in sys.argv[1:]:
        if arg.startswith('--count='):
            count = int(arg.split('=')[1])
        elif arg.isdigit():
            count = int(arg)

    # Output file
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_file = f"{OUTPUT_DIR}\\parse_results_{timestamp}.txt"

    # Ensure output dir exists
    import os

    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("=" * 60)
    print("UNIFIED PARSE BUILDER TEST")
    print("=" * 60)
    print(f"Loading {count} random clues...")

    clues = load_random_clues(CLUES_DB, count)
    print(f"Loaded {len(clues)} clues")

    builder = UnifiedParseBuilder(CRYPTIC_DB)

    # Stats
    solved = 0
    partial = 0
    failed = 0

    # Results for file
    results_text = []
    results_text.append("=" * 70)
    results_text.append(f"UNIFIED PARSE BUILDER RESULTS")
    results_text.append(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    results_text.append(f"Cohort size: {count}")
    results_text.append("=" * 70)
    results_text.append("")

    for i, clue in enumerate(clues, 1):
        clue_text = clue['clue_text']
        answer = clue['answer']
        wordplay_type = clue.get('wordplay_type', 'unknown')

        # Parse without debug (too verbose for batch)
        result = builder.parse(clue_text, answer, definition_words=None, debug=False)

        if result and result.is_complete:
            status = "SOLVED"
            solved += 1
            derivation = result.derivation
        elif result:
            status = f"PARTIAL ({result.confidence:.0%})"
            partial += 1
            derivation = result.derivation
        else:
            status = "FAILED"
            failed += 1
            derivation = "No parse found"

        # Console output (brief)
        print(f"[{i:3d}/{count}] [{status:12s}] {answer}")

        # File output (detailed)
        results_text.append(f"[{status}] {clue_text}")
        results_text.append(f"  Answer: {answer}")
        results_text.append(f"  Tagged: {wordplay_type}")
        results_text.append(f"  Parse: {derivation}")
        if result:
            results_text.append(f"  Definition: {result.definition_words}")
            results_text.append(f"  Link words: {result.link_words}")
            results_text.append(f"  Unparsed: {result.unparsed_words}")
        results_text.append("")

    builder.close()

    # Summary
    total = solved + partial + failed
    summary = [
        "",
        "=" * 70,
        "SUMMARY",
        "=" * 70,
        f"Total clues:    {total}",
        f"Fully solved:   {solved} ({solved / total * 100:.1f}%)",
        f"Partial parse:  {partial} ({partial / total * 100:.1f}%)",
        f"Failed:         {failed} ({failed / total * 100:.1f}%)",
        "",
        f"Solve rate:     {solved}/{total} ({solved / total * 100:.1f}%)",
        f"Coverage rate:  {solved + partial}/{total} ({(solved + partial) / total * 100:.1f}%)",
    ]

    for line in summary:
        print(line)
        results_text.append(line)

    # Write to file
    with open(output_file, 'w', encoding='utf-8') as f:
        f.write('\n'.join(results_text))

    print(f"\nResults written to: {output_file}")