#!/usr/bin/env python3
"""
Comprehensive Anagram Evidence Detection System

This module provides sophisticated anagram detection that goes beyond simple letter counting:
- Tests all possible word combinations from clue text
- Handles deletion cases (DOLLOP → PLOD with deletion indicator)
- Handles insertion cases (fodder + extra letters → candidate)
- Integrates with existing candidate scoring system
- Provides detailed evidence reporting
- Uses progressive expansion from indicators to find anagram fodder

CORRECTED VERSION: Enforces four rules for fodder:
1. Indicator detection - single word first, then two-word if needed, with positions
2. Proximity - fodder must be adjacent to indicator (one link word allowed)
3. Contiguity - fodder words must be next to each other in the clue
4. Whole words - fodder is complete words, not cherry-picked letters
"""

import re
import itertools
from collections import Counter
from typing import List, Dict, Tuple, Optional, Set
from dataclasses import dataclass, field

# Import the working anagram function
import sys
import os

# sys.path.append(r'C:\Users\shute\PycharmProjects\cryptic_solver')


# generate_anagram_hypotheses imported lazily in analyze_and_rank_anagram_candidates to avoid circular import


@dataclass
class AnagramEvidence:
    """Structured anagram evidence with detailed information."""
    candidate: str
    fodder_words: List[str]
    fodder_letters: str
    evidence_type: str  # "exact", "deletion", "insertion", "partial"
    confidence: float
    excess_letters: str = ""  # Letters not used (deletion case)
    needed_letters: str = ""  # Letters needed (insertion case)

    # NEW: Complete word attribution for compound analysis
    indicator_words: List[str] = None  # The anagram indicator(s)
    indicator_position: int = -1  # Position of indicator in token list
    definition_words: List[str] = None  # Definition window words (if known)
    link_words: List[str] = None  # Link words identified
    remaining_words: List[str] = None  # Words available for compound wordplay

    # Deprecated: keeping for backward compatibility
    unused_clue_words: List[str] = None

    def __post_init__(self):
        if self.unused_clue_words is None:
            self.unused_clue_words = []
        if self.indicator_words is None:
            self.indicator_words = []
        if self.definition_words is None:
            self.definition_words = []
        if self.link_words is None:
            self.link_words = []
        if self.remaining_words is None:
            self.remaining_words = []


@dataclass
class IndicatorMatch:
    """Stores indicator match with position info."""
    words: List[str]
    start_pos: int
    end_pos: int  # inclusive
    is_multi_word: bool


@dataclass
class ContiguousFodder:
    """A valid contiguous fodder sequence adjacent to an indicator."""
    words: List[str]
    positions: List[int]
    letters: str
    indicator: IndicatorMatch
    side: str  # 'left' or 'right' of indicator


# Link words that can appear between indicator and fodder
LINK_WORDS = {
    # Articles and prepositions
    'to', 'of', 'in', 'for', 'with', 'by', 'from', 'a', 'an', 'the',
    'and', 'is', 'are', 'needs', 'about', 'on', 'after', 'at', 'as', 'or',
    # Common verbs
    'be', 'being', 'been', 'has', 'have', 'had', 'having',
    'was', 'were', 'will', 'would', 'could', 'should', 'must', 'may', 'might',
    'gets', 'get', 'getting', 'got', 'makes', 'make', 'making', 'made',
    'gives', 'give', 'given', 'giving', 'sees', 'see', 'seen', 'seeing',
    # Contractions (with apostrophe)
    "it's", "that's", "there's", "here's", "what's",
    # Contractions (apostrophe-stripped)
    'its', 'thats', 'theres', 'heres', 'whats', 'im', 'ive', 'id',
    'youre', 'youve', 'youd', 'hes', 'shes', 'theyre', 'theyve',
    'dont', 'doesnt', 'didnt', 'wont', 'wouldnt', 'cant', 'couldnt',
    # Conjunctions and connectors
    'but', 'that', 'which', 'when', 'where', 'while', 'so', 'yet',
    # Other common links
    'this', 'these', 'those', 'such', 'one', 'ones', 'some', 'any', 'all',
    'here', 'there', 'into', 'onto', 'within', 'without',
    # Note: 'find', 'found', 'finding' removed - they can be part of container indicators like "found in"
    'show', 'showing', 'put', 'set',
    'if', 'how', 'why', 'who', 'whom', 'you',
}

# Link words that can appear BETWEEN fodder words without breaking contiguity
# These are words that setters commonly use to join fodder: "birds WITH ale", "cats AND dogs"
TRANSPARENT_LINK_WORDS = {'with', 'and', 'or'}


class ComprehensiveWordplayDetector:
    """Comprehensive wordplay evidence detection system using all database indicators."""

    def __init__(self, db_path: str = None):
        """Initialize with database path to load all indicators."""
        self.db_path = db_path or r"C:\Users\shute\PycharmProjects\cryptic_solver\data\cryptic_new.db"

        # Load all indicators from database by type
        self.anagram_indicators = []  # Keep for backward compatibility
        self.anagram_indicators_single: Set[str] = set()  # NEW: single-word only
        self.anagram_indicators_two_word: Set[str] = set()  # NEW: two-word only
        self.insertion_indicators = []
        self.deletion_indicators = []
        self.reversal_indicators = []
        self.hidden_indicators = []
        self.parts_indicators = []

        # Store confidence levels and frequency
        self.indicator_confidence = {}
        self.indicator_frequency = {}
        self.indicators_loaded = False

        self._load_all_indicators_from_database()

    def _load_all_indicators_from_database(self):
        """Load all indicators from the database organized by wordplay type."""
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Load all indicators with their types, confidence and frequency
            cursor.execute("""
                SELECT word, wordplay_type, confidence, COALESCE(frequency, 0) as freq 
                FROM indicators 
                ORDER BY freq DESC, word
            """)

            all_indicators = cursor.fetchall()

            # Organize by wordplay type
            for word, wordplay_type, confidence, frequency in all_indicators:
                word_lower = word.lower().strip()
                self.indicator_confidence[word_lower] = confidence
                self.indicator_frequency[word_lower] = frequency or 0

                if wordplay_type == 'anagram':
                    self.anagram_indicators.append(word_lower)
                    # NEW: Split into single and two-word
                    if ' ' in word_lower:
                        self.anagram_indicators_two_word.add(word_lower)
                    else:
                        self.anagram_indicators_single.add(word_lower)
                elif wordplay_type == 'insertion':
                    self.insertion_indicators.append(word_lower)
                elif wordplay_type == 'deletion':
                    self.deletion_indicators.append(word_lower)
                elif wordplay_type == 'reversal':
                    self.reversal_indicators.append(word_lower)
                elif wordplay_type == 'hidden':
                    self.hidden_indicators.append(word_lower)
                elif wordplay_type == 'parts':
                    self.parts_indicators.append(word_lower)

            conn.close()

            print(f"Loaded comprehensive indicators from database:")
            print(
                f"  Anagram indicators: {len(self.anagram_indicators)} ({len(self.anagram_indicators_single)} single-word, {len(self.anagram_indicators_two_word)} two-word)")
            print(f"  Insertion indicators: {len(self.insertion_indicators)}")
            print(f"  Deletion indicators: {len(self.deletion_indicators)}")
            print(f"  Reversal indicators: {len(self.reversal_indicators)}")
            print(f"  Hidden indicators: {len(self.hidden_indicators)}")
            print(f"  Parts indicators: {len(self.parts_indicators)}")

            self.indicators_loaded = True

        except Exception as e:
            print(f"Warning: Could not load indicators from database: {e}")
            print("Falling back to minimal hardcoded indicators...")
            self._load_fallback_indicators()
            self.indicators_loaded = True

    def _load_fallback_indicators(self):
        """Load minimal fallback indicators if database unavailable."""
        self.anagram_indicators = ['broken', 'wild', 'crazy', 'mixed', 'drunk', 'mad',
                                   'out', 'off', 'confused', 'damaged', 'ruined',
                                   'smashed',
                                   'awful', 'bad', 'upset', 'destroyed', 'wrecked',
                                   'mangled']
        self.anagram_indicators_single = set(self.anagram_indicators)
        self.anagram_indicators_two_word = set()
        self.insertion_indicators = ['in', 'into', 'inside', 'within', 'holding',
                                     'containing', 'around']
        self.deletion_indicators = ['without', 'losing', 'lacking', 'missing', 'dropped',
                                    'removed']
        self.reversal_indicators = ['back', 'up', 'returned', 'reversed', 'retiring',
                                    'recalled']
        self.hidden_indicators = ['in', 'within', 'part of', 'some', 'hidden in',
                                  'concealed']
        self.parts_indicators = ['initially', 'first', 'finally', 'last', 'head', 'tail']

    def lookup_wordplay_substitutions(self, words: List[str], needed_letters: str) -> List[Tuple[str, str, str]]:
        """
        Look up wordplay substitutions for given words that could provide needed letters.
        
        Checks both single words and two-word phrases (e.g., "the french" -> LA).
        
        Args:
            words: List of remaining clue words to check
            needed_letters: Letters still needed to complete the anagram
            
        Returns:
            List of (phrase, substitution, category) tuples where substitution 
            provides some or all of needed_letters
        """
        if not words or not needed_letters:
            return []
        
        results = []
        needed_counter = Counter(needed_letters.lower())
        
        try:
            import sqlite3
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Check single words
            for word in words:
                word_clean = word.lower().strip()
                cursor.execute("""
                    SELECT indicator, substitution, category FROM wordplay
                    WHERE LOWER(indicator) = ?
                """, (word_clean,))
                
                for indicator, substitution, category in cursor.fetchall():
                    sub_letters = ''.join(c.lower() for c in substitution if c.isalpha())
                    sub_counter = Counter(sub_letters)
                    
                    # Check if substitution provides letters we need
                    provides_needed = True
                    for letter, count in sub_counter.items():
                        if needed_counter[letter] < count:
                            provides_needed = False
                            break
                    
                    if provides_needed and sub_letters:
                        results.append((indicator, substitution.upper(), category))
            
            # Check two-word phrases (e.g., "the french" -> LA)
            for i in range(len(words) - 1):
                phrase = f"{words[i].lower().strip()} {words[i+1].lower().strip()}"
                cursor.execute("""
                    SELECT indicator, substitution, category FROM wordplay
                    WHERE LOWER(indicator) = ?
                """, (phrase,))
                
                for indicator, substitution, category in cursor.fetchall():
                    sub_letters = ''.join(c.lower() for c in substitution if c.isalpha())
                    sub_counter = Counter(sub_letters)
                    
                    # Check if substitution provides letters we need
                    provides_needed = True
                    for letter, count in sub_counter.items():
                        if needed_counter[letter] < count:
                            provides_needed = False
                            break
                    
                    if provides_needed and sub_letters:
                        results.append((indicator, substitution.upper(), category))
            
            conn.close()
            
        except Exception as e:
            # Silently fail - substitution lookup is an enhancement, not critical
            pass
        
        # Check "half X" patterns (e.g., "half like" -> LI or "like half" -> LI)
        # This handles parts/deletion indicators that take half of an adjacent word
        # Must be OUTSIDE the try block since it doesn't use the database
        half_indicators = {'half', 'halved', 'halves', 'partly', 'partial', 'semi'}
        
        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,;:!?')
            
            if word_lower in half_indicators:
                # Check word AFTER the indicator (e.g., "half like")
                if i + 1 < len(words):
                    target = words[i + 1]
                    target_clean = target.lower().strip('.,;:!?')
                    target_letters = ''.join(c.lower() for c in target if c.isalpha())
                    
                    if len(target_letters) >= 2:
                        half_len = len(target_letters) // 2
                        first_half = target_letters[:half_len]
                        second_half = target_letters[half_len:]
                        
                        # Check if either half provides needed letters
                        for half, desc in [(first_half, 'first'), (second_half, 'second')]:
                            if not half:
                                continue
                            half_counter = Counter(half)
                            provides_needed = all(needed_counter[c] >= half_counter[c] for c in half_counter)
                            
                            if provides_needed:
                                phrase = f"{word_lower} {target_clean}"
                                results.append((phrase, half.upper(), f'{desc}_half'))
                
                # Check word BEFORE the indicator (e.g., "like half")
                if i > 0:
                    target = words[i - 1]
                    target_clean = target.lower().strip('.,;:!?')
                    target_letters = ''.join(c.lower() for c in target if c.isalpha())
                    
                    if len(target_letters) >= 2:
                        half_len = len(target_letters) // 2
                        first_half = target_letters[:half_len]
                        second_half = target_letters[half_len:]
                        
                        # Check if either half provides needed letters
                        for half, desc in [(first_half, 'first'), (second_half, 'second')]:
                            if not half:
                                continue
                            half_counter = Counter(half)
                            provides_needed = all(needed_counter[c] >= half_counter[c] for c in half_counter)
                            
                            if provides_needed:
                                phrase = f"{target_clean} {word_lower}"
                                results.append((phrase, half.upper(), f'{desc}_half'))
        
        return results

    def normalize_letters(self, text: str) -> str:
        """Extract and lowercase only alphabetic characters."""
        return ''.join(c.lower() for c in text if c.isalpha())

    def is_anagram(self, word1: str, word2: str) -> bool:
        """Check if two strings are anagrams (ignoring case and non-letters)."""
        return Counter(self.normalize_letters(word1)) == Counter(
            self.normalize_letters(word2))

    def can_contribute_letters(self, target: str, source: str) -> Tuple[
        bool, float, str]:
        """
        Check if source letters can contribute to target.
        Returns (can_contribute, contribution_ratio, remaining_letters_needed).
        """
        target_counter = Counter(self.normalize_letters(target))
        source_counter = Counter(self.normalize_letters(source))

        # Check each letter in source exists in target
        for letter, count in source_counter.items():
            if target_counter[letter] < count:
                return False, 0.0, ""

        # Calculate what letters are still needed
        remaining = target_counter - source_counter
        remaining_letters = ''.join(remaining.elements())

        # Contribution ratio is how much of target is explained
        contribution = sum(source_counter.values())
        total = sum(target_counter.values())
        ratio = contribution / total if total > 0 else 0.0

        return True, ratio, remaining_letters

    def can_form_by_deletion_strict(self, candidate: str, fodder: str,
                                    max_excess: int = 2) -> Tuple[bool, str]:
        """
        Check if candidate can be formed by deleting ≤max_excess letters from fodder.
        Returns (can_form, excess_letters).
        """
        candidate_counter = Counter(self.normalize_letters(candidate))
        fodder_counter = Counter(self.normalize_letters(fodder))

        # Fodder must have at least all letters in candidate
        for letter, count in candidate_counter.items():
            if fodder_counter[letter] < count:
                return False, ""

        # Calculate excess letters
        excess = fodder_counter - candidate_counter
        excess_letters = ''.join(sorted(excess.elements()))

        if len(excess_letters) <= max_excess:
            return True, excess_letters

        return False, ""

    def _apply_deletion_indicators(self, fodder_words: List[str],
                                   fodder_indicator: 'IndicatorMatch') -> List[dict]:
        """
        Check if fodder contains deletion indicators (like "almost") and apply them.

        For example: ["almost", "gets", "meaner"] with "almost" being last_delete
        becomes: ["GET", "meaner"] (removing S from "gets")

        Returns list of modified fodder options with metadata:
        [{'words': [...], 'letters': '...', 'deletion_applied': {...}}, ...]
        """
        results = []

        for i, word in enumerate(fodder_words):
            word_lower = word.lower().strip('.,;:!?')

            # Check if this word is a parts indicator
            if word_lower in self.parts_indicators:
                # Look up the indicator to get subtype
                # We need to check the database for the subtype
                # For now, check common deletion indicators
                deletion_indicators = {
                    'almost': 'last_delete',
                    'nearly': 'last_delete',
                    'mostly': 'last_delete',
                    'largely': 'last_delete',
                    'about': 'last_delete',  # can mean truncation
                    'short': 'last_delete',
                    'shortly': 'last_delete',
                    'headless': 'first_delete',
                    'beheaded': 'first_delete',
                    'topless': 'first_delete',
                }

                if word_lower in deletion_indicators:
                    subtype = deletion_indicators[word_lower]

                    # The next word is the target for deletion
                    if i + 1 < len(fodder_words):
                        target_word = fodder_words[i + 1]
                        target_letters = ''.join(c for c in target_word if c.isalpha())

                        if target_letters:
                            # Apply deletion
                            if 'last' in subtype:
                                modified_letters = target_letters[
                                                   :-1]  # Remove last letter
                            elif 'first' in subtype:
                                modified_letters = target_letters[
                                                   1:]  # Remove first letter
                            else:
                                continue

                            if modified_letters:
                                # Build new fodder list: skip the indicator, use modified target
                                new_words = []
                                for j, w in enumerate(fodder_words):
                                    if j == i:
                                        continue  # Skip the indicator
                                    elif j == i + 1:
                                        new_words.append(
                                            target_word)  # Keep original word for display
                                    else:
                                        new_words.append(w)

                                # Calculate total letters with modification
                                total_letters = ''
                                for j, w in enumerate(fodder_words):
                                    if j == i:
                                        continue  # Skip indicator letters
                                    elif j == i + 1:
                                        total_letters += modified_letters.lower()
                                    else:
                                        total_letters += ''.join(
                                            c.lower() for c in w if c.isalpha())

                                results.append({
                                    'words': new_words,
                                    'letters': total_letters,
                                    'deletion_applied': {
                                        'indicator': word,
                                        'target': target_word,
                                        'original': target_letters,
                                        'modified': modified_letters,
                                        'subtype': subtype
                                    },
                                    'indicator': fodder_indicator
                                })

        return results

    def _tokenize_clue(self, clue_text: str) -> List[str]:
        """Split clue into tokens, preserving words with apostrophes."""
        # Split on whitespace, keeping punctuation attached
        tokens = clue_text.split()
        # Clean each token but preserve apostrophes within words
        cleaned = []
        for token in tokens:
            # Remove leading/trailing punctuation except apostrophes
            clean = token.strip('.,;:!?"()[]{}')
            if clean:
                cleaned.append(clean)
        return cleaned

    def detect_wordplay_indicators(self, clue_text: str) -> Dict[str, List[str]]:
        """
        Detect all wordplay indicators in clue, returning their positions.

        CORRECTED: Now returns indicator positions for proper fodder adjacency checking.
        Uses single-word matching first, then two-word matching for those indicator types.

        Returns dict with keys:
        - 'anagram': List of anagram indicator words found
        - 'anagram_matches': List of IndicatorMatch objects with positions
        - 'insertion': List of insertion indicator words found
        - etc.
        """
        tokens = self._tokenize_clue(clue_text)
        clue_lower = clue_text.lower()

        found = {
            'anagram': [],
            'anagram_matches': [],  # NEW: with positions
            'insertion': [],
            'deletion': [],
            'reversal': [],
            'hidden': [],
            'parts': []
        }

        # First pass: single-word indicators with positions
        for i, token in enumerate(tokens):
            token_lower = token.lower().strip('.,;:!?"\'')

            if token_lower in self.anagram_indicators_single:
                found['anagram'].append(token)
                found['anagram_matches'].append(IndicatorMatch(
                    words=[token],
                    start_pos=i,
                    end_pos=i,
                    is_multi_word=False
                ))

            if token_lower in self.insertion_indicators:
                found['insertion'].append(token)

            if token_lower in self.deletion_indicators:
                found['deletion'].append(token)

            if token_lower in self.reversal_indicators:
                found['reversal'].append(token)

            if token_lower in self.hidden_indicators:
                found['hidden'].append(token)

            if token_lower in self.parts_indicators:
                found['parts'].append(token)

        # Second pass: two-word anagram indicators (ALWAYS check, not just fallback)
        # Both single-word and two-word indicators should be found so contiguity
        # logic can choose the best one based on proximity to fodder
        for i in range(len(tokens) - 1):
            two_word = f"{tokens[i].lower().strip('.,;:!?\"')} {tokens[i + 1].lower().strip('.,;:!?\"')}"
            if two_word in self.anagram_indicators_two_word:
                found['anagram'].append(f"{tokens[i]} {tokens[i + 1]}")
                found['anagram_matches'].append(IndicatorMatch(
                    words=[tokens[i], tokens[i + 1]],
                    start_pos=i,
                    end_pos=i + 1,
                    is_multi_word=True
                ))

        return found

    def get_progressive_fodder_words(self, clue_text: str, indicators: dict,
                                     candidates: List[str]) -> list:
        """
        Progressive expansion from indicator positions to find anagram fodder.

        CORRECTED: Actually implements contiguous expansion from indicators.
        This method now returns words that are:
        1. Adjacent to an indicator (with at most one link word between)
        2. Contiguous (next to each other)
        3. Whole words

        Returns list of fodder word lists (each is a contiguous sequence).
        """
        anagram_matches = indicators.get('anagram_matches', [])
        if not anagram_matches:
            return []

        tokens = self._tokenize_clue(clue_text)
        all_fodder_sequences = []

        for indicator in anagram_matches:
            # Expand left
            left_sequences = self._expand_from_indicator(tokens, indicator, 'left')
            all_fodder_sequences.extend(left_sequences)

            # Expand right
            right_sequences = self._expand_from_indicator(tokens, indicator, 'right')
            all_fodder_sequences.extend(right_sequences)

        # Return flat list of words from all sequences for backward compatibility
        # The caller can use get_contiguous_fodder_sequences for structured data
        all_words = []
        for seq in all_fodder_sequences:
            all_words.extend(seq.words)

        return list(set(all_words))  # Deduplicate

    def _expand_from_indicator(self, tokens: List[str], indicator: IndicatorMatch,
                               direction: str) -> List[ContiguousFodder]:
        """
        Expand in one direction from indicator to find valid contiguous fodder.

        UPDATED: Now tries BOTH with and without link words as fodder.
        Link words at boundaries are first tried as fodder, then also tried as skippable.
        This allows "in doves" to be fodder (7 letters) rather than just "doves" (5 letters).

        Returns list of ContiguousFodder objects (all valid contiguous sequences).
        """
        results = []
        n = len(tokens)

        if direction == 'left':
            boundary = indicator.start_pos - 1
            step = -1
        else:
            boundary = indicator.end_pos + 1
            step = 1

        if boundary < 0 or boundary >= n:
            return results

        # Try TWO expansion strategies:
        # 1. Include link words as potential fodder (try first)
        # 2. Skip boundary link words (original behavior)

        start_positions = [boundary]  # Always try from the boundary first

        # If boundary is a link word, ALSO try skipping it (but try including it first)
        if tokens[boundary].lower() in LINK_WORDS:
            skip_pos = boundary + step
            if 0 <= skip_pos < n:
                start_positions.append(skip_pos)

        for start_pos in start_positions:
            # Expand contiguously from start_pos
            current_words = []
            current_positions = []
            pos = start_pos

            while 0 <= pos < n:
                token = tokens[pos]
                token_lower = token.lower()

                # NOTE: We no longer stop at other indicator types (insertion, deletion, etc.)
                # Words like "in" could be fodder ("in doves" = INDOVES) or indicators.
                # Let letter matching decide which interpretation is correct.
                # Only stop at definition words (handled by caller) or end of clue.

                # Link words CAN be fodder - they contribute letters (e.g., "in" contributes I,N)
                # Include them and let letter matching decide if they're needed
                # Note: We don't skip or stop for link words anymore - they're valid fodder

                # Add this word to fodder (INCLUDING link words - they contribute letters)
                current_words.append(token)
                current_positions.append(pos)

                # Get ordered words (left expansion needs reversal)
                if direction == 'left':
                    ordered_words = list(reversed(current_words))
                    ordered_positions = list(reversed(current_positions))
                else:
                    ordered_words = current_words[:]
                    ordered_positions = current_positions[:]

                letters = self.normalize_letters(' '.join(ordered_words))

                # Record this as a valid contiguous sequence
                results.append(ContiguousFodder(
                    words=ordered_words,
                    positions=ordered_positions,
                    letters=letters,
                    indicator=indicator,
                    side=direction
                ))

                pos += step

        return results

    def get_contiguous_fodder_sequences(self, clue_text: str, indicators: dict,
                                        target_length: int = None) -> List[
        ContiguousFodder]:
        """
        Get all valid contiguous fodder sequences for a clue.

        This is the main method for finding fodder that enforces all four rules:
        1. Indicator position tracking
        2. Proximity to indicator
        3. Contiguity of fodder words
        4. Whole words only

        Args:
            clue_text: The clue text
            indicators: Dict from detect_wordplay_indicators
            target_length: Optional filter for exact letter count match

        Returns:
            List of ContiguousFodder objects
        """
        anagram_matches = indicators.get('anagram_matches', [])
        if not anagram_matches:
            return []

        tokens = self._tokenize_clue(clue_text)
        all_sequences = []

        for indicator in anagram_matches:
            # Expand left
            left = self._expand_from_indicator(tokens, indicator, 'left')
            all_sequences.extend(left)

            # Expand right
            right = self._expand_from_indicator(tokens, indicator, 'right')
            all_sequences.extend(right)

        # Generate deletion variants for sequences containing deletion indicators
        deletion_variants = []
        for seq in all_sequences:
            variants = self._generate_deletion_variants(seq)
            deletion_variants.extend(variants)

        all_sequences.extend(deletion_variants)

        # Filter by target length if specified
        if target_length is not None:
            all_sequences = [s for s in all_sequences if len(s.letters) == target_length]

        return all_sequences

    def _generate_deletion_variants(self, fodder: ContiguousFodder) -> List[
        ContiguousFodder]:
        """
        Generate fodder variants by applying modifier indicators (deletion AND doubling).

        Deletion: ["almost", "gets", "meaner"] -> ["gets", "meaner"] with "getmeaner"
        Doubling: ["ace", "doubly", "hot"] -> ["ace", "hot"] with "acehothot"
        """
        deletion_indicators = {
            'almost': 'last_delete',
            'nearly': 'last_delete',
            'mostly': 'last_delete',
            'largely': 'last_delete',
            'about': 'last_delete',
            'short': 'last_delete',
            'shortly': 'last_delete',
            'endless': 'last_delete',
            'endlessly': 'last_delete',
            'limitless': 'last_delete',
            'headless': 'first_delete',
            'beheaded': 'first_delete',
            'topless': 'first_delete',
            'leading': 'first_delete',  # can mean "remove leading letter"
        }

        doubling_indicators = {
            'doubly', 'twice', 'double', 'doubled', 'two', 'dual',
            'repeated', 'repeating', 'again', 'twofold'
        }

        variants = []
        words = fodder.words

        for i, word in enumerate(words):
            word_lower = word.lower().strip('.,;:!?')

            # Handle DELETION indicators
            if word_lower in deletion_indicators:
                subtype = deletion_indicators[word_lower]

                # Next word is the target for deletion
                if i + 1 < len(words):
                    target_word = words[i + 1]
                    target_letters = ''.join(
                        c.lower() for c in target_word if c.isalpha())

                    if len(target_letters) > 1:  # Must have letters to remove
                        if 'last' in subtype:
                            modified_letters = target_letters[:-1]
                        elif 'first' in subtype:
                            modified_letters = target_letters[1:]
                        else:
                            continue

                        # Build new word list and letters
                        new_words = []
                        new_letters = ''

                        for j, w in enumerate(words):
                            if j == i:
                                continue  # Skip the deletion indicator
                            elif j == i + 1:
                                new_words.append(w)  # Keep word for display
                                new_letters += modified_letters
                            else:
                                new_words.append(w)
                                new_letters += ''.join(
                                    c.lower() for c in w if c.isalpha())

                        if new_words and new_letters:
                            variant = ContiguousFodder(
                                words=new_words,
                                positions=fodder.positions,
                                letters=new_letters,
                                indicator=fodder.indicator,
                                side=fodder.side
                            )
                            variant.deletion_info = {
                                'indicator': word,
                                'target': target_word,
                                'original': target_letters,
                                'modified': modified_letters,
                                'subtype': subtype
                            }
                            variants.append(variant)

            # Handle DOUBLING indicators
            elif word_lower in doubling_indicators:
                # Next word gets doubled
                if i + 1 < len(words):
                    target_word = words[i + 1]
                    target_letters = ''.join(
                        c.lower() for c in target_word if c.isalpha())

                    if target_letters:
                        doubled_letters = target_letters + target_letters

                        # Build new word list and letters
                        new_words = []
                        new_letters = ''

                        for j, w in enumerate(words):
                            if j == i:
                                continue  # Skip the doubling indicator
                            elif j == i + 1:
                                new_words.append(w)  # Keep word for display
                                new_letters += doubled_letters
                            else:
                                new_words.append(w)
                                new_letters += ''.join(
                                    c.lower() for c in w if c.isalpha())

                        if new_words and new_letters:
                            variant = ContiguousFodder(
                                words=new_words,
                                positions=fodder.positions,
                                letters=new_letters,
                                indicator=fodder.indicator,
                                side=fodder.side
                            )
                            variant.doubling_info = {
                                'indicator': word,
                                'target': target_word,
                                'original': target_letters,
                                'doubled': doubled_letters
                            }
                            variants.append(variant)

        return variants

    def _can_word_contribute_to_candidates(self, word: str,
                                           candidates: List[str]) -> bool:
        """
        FIXED: Check if a word can contribute letters to any of the candidates.
        CRITICAL FIX: Parameter order corrected.
        """
        word_normalized = self.normalize_letters(word)
        if not word_normalized:
            return False

        for candidate in candidates:
            candidate_normalized = self.normalize_letters(candidate)
            # FIXED: Correct parameter order - can source (word) contribute to target (candidate)
            can_contribute, _, _ = self.can_contribute_letters(candidate_normalized,
                                                               word_normalized)
            if can_contribute:
                return True

        return False

    def test_anagram_evidence(self, candidate: str, clue_text: str,
                              indicators: Dict[str, List[str]], enumeration: str = None,
                              debug: bool = False) -> Optional[AnagramEvidence]:
        """
        Test for anagram evidence using contiguous fodder from indicators.

        CORRECTED: Now only tests contiguous fodder sequences adjacent to indicators,
        not all possible word combinations.

        Returns AnagramEvidence if match found, None otherwise.
        """
        candidate_letters = self.normalize_letters(candidate)
        if not candidate_letters:
            return None

        # STAGE B HYGIENE: reject self-anagrams (matching your proven anagram engine)
        if candidate.lower() in clue_text.lower():
            if debug:
                print(
                    f"    REJECTED: Self-anagram - '{candidate}' appears verbatim in clue")
            return None

        # Get contiguous fodder sequences (enforces all four rules)
        target_length = len(candidate_letters)
        fodder_sequences = self.get_contiguous_fodder_sequences(
            clue_text, indicators, target_length=None  # Get all, filter later
        )

        if not fodder_sequences:
            if debug:
                print(f"    NO CONTIGUOUS FODDER SEQUENCES found near indicators")
            return None

        if debug:
            print(
                f"    DEBUG: Testing candidate '{candidate}' ({len(candidate_letters)} letters)")
            print(f"    Found {len(fodder_sequences)} contiguous fodder sequences")

        # Get all tokens for unused word calculation
        tokens = self._tokenize_clue(clue_text)

        best_evidence = None
        best_score = 0.0

        for i, fodder in enumerate(fodder_sequences):
            fodder_letters = fodder.letters

            if debug and i < 10:
                print(
                    f"      [{i + 1:2d}] Testing: {fodder.words} → '{fodder_letters}' ({len(fodder_letters)} letters)")

            # Test exact anagram match first (highest priority)
            if self.is_anagram(candidate_letters, fodder_letters):
                if debug:
                    print(f"      ★ EXACT ANAGRAM MATCH!")

                # Calculate complete word attribution
                indicator_words = list(fodder.indicator.words)
                fodder_word_set = set(w.lower() for w in fodder.words)
                indicator_word_set = set(w.lower() for w in indicator_words)

                # Check if this is a variant (deletion or doubling)
                deletion_info = getattr(fodder, 'deletion_info', None)
                doubling_info = getattr(fodder, 'doubling_info', None)

                # Determine evidence type and modifier indicator word
                if deletion_info:
                    evidence_type = "exact_with_deletion"
                    confidence = 0.88
                    modifier_indicator_word = deletion_info['indicator'].lower()
                elif doubling_info:
                    evidence_type = "exact_with_doubling"
                    confidence = 0.88
                    modifier_indicator_word = doubling_info['indicator'].lower()
                else:
                    evidence_type = "exact"
                    confidence = 0.9
                    modifier_indicator_word = None

                # Identify link words and remaining words
                link_words_found = []
                remaining_words = []
                for t in tokens:
                    t_lower = t.lower()
                    if t_lower in fodder_word_set:
                        continue  # Already accounted as fodder
                    if t_lower in indicator_word_set:
                        continue  # Already accounted as indicator
                    if modifier_indicator_word and t_lower == modifier_indicator_word:
                        continue  # The modifier indicator is accounted for
                    if t_lower in LINK_WORDS:
                        link_words_found.append(t)
                    else:
                        remaining_words.append(t)

                evidence = AnagramEvidence(
                    candidate=candidate,
                    fodder_words=list(fodder.words),
                    fodder_letters=fodder_letters,
                    evidence_type=evidence_type,
                    confidence=confidence,
                    indicator_words=indicator_words,
                    indicator_position=fodder.indicator.start_pos,
                    link_words=link_words_found,
                    remaining_words=remaining_words,
                    unused_clue_words=remaining_words  # Backward compatibility
                )

                # Attach variant info for downstream use
                if deletion_info:
                    evidence.deletion_info = deletion_info
                if doubling_info:
                    evidence.doubling_info = doubling_info

                return evidence

            # Test with deletion indicators applied (e.g., "almost gets" -> "GET")
            deletion_variants = self._apply_deletion_indicators(list(fodder.words),
                                                                fodder.indicator)
            for variant in deletion_variants:
                variant_letters = variant['letters']

                if debug and i < 10:
                    print(
                        f"           Deletion variant: {variant['words']} → '{variant_letters}' ({len(variant_letters)} letters)")

                if self.is_anagram(candidate_letters, variant_letters):
                    if debug:
                        print(f"      ★ EXACT ANAGRAM MATCH (with deletion)!")

                    deletion_info = variant['deletion_applied']
                    indicator_words = list(fodder.indicator.words)

                    # Words that contribute to the anagram
                    fodder_word_set = set(w.lower() for w in variant['words'])
                    indicator_word_set = set(w.lower() for w in indicator_words)
                    deletion_indicator_word = deletion_info['indicator'].lower()

                    link_words_found = []
                    remaining_words = []
                    for t in tokens:
                        t_lower = t.lower()
                        if t_lower in fodder_word_set:
                            continue
                        if t_lower in indicator_word_set:
                            continue
                        if t_lower == deletion_indicator_word:
                            continue  # The deletion indicator is accounted for
                        if t_lower in LINK_WORDS:
                            link_words_found.append(t)
                        else:
                            deletion_enum_bonus = -50
                    
                    deletion_score = deletion_enum_bonus + deletion_explained - excess_penalty + (1.0 / len(fodder.words))
                    
                    if debug and i < 10:
                        print(f"           Deletion score: {deletion_score:.2f} (enum={deletion_enum_bonus}, explained={deletion_explained}, excess_penalty={excess_penalty})")
                    
                    # Only use deletion evidence if better than current best
                    if deletion_score > best_score:
                        # Calculate complete word attribution
                        indicator_words = list(fodder.indicator.words)
                        fodder_word_set = set(w.lower() for w in fodder.words)
                        indicator_word_set = set(w.lower() for w in indicator_words)

                    # Return evidence with deletion info
                    evidence = AnagramEvidence(
                        candidate=candidate,
                        fodder_words=variant['words'],
                        fodder_letters=variant_letters,
                        evidence_type="exact_with_deletion",
                        confidence=0.88,
                        indicator_words=indicator_words,
                        indicator_position=fodder.indicator.start_pos,
                        link_words=link_words_found,
                        remaining_words=remaining_words,
                        unused_clue_words=remaining_words
                    )
                    # Attach deletion info for downstream use
                    evidence.deletion_info = deletion_info
                    return evidence

            # Test partial contribution (only if anagram indicators present)
            if indicators.get('anagram'):
                can_contribute, contribution_ratio, remaining_letters = self.can_contribute_letters(
                    candidate_letters, fodder_letters)

                if debug and i < 10:
                    print(
                        f"           Partial: can_contribute={can_contribute}, remaining='{remaining_letters}'")

                if can_contribute:
                    # Calculate explained letters (secondary scoring factor)
                    explained_letters = len(candidate_letters) - len(remaining_letters)
                    total_letters = len(candidate_letters)

                    # Safety check to avoid division by zero
                    if total_letters == 0:
                        continue

                    # PRIMARY: Check enumeration pattern match (highest priority)
                    enumeration_bonus = 0
                    if enumeration:
                        if self._matches_enumeration_pattern(candidate, enumeration):
                            enumeration_bonus = 100  # Massive bonus for correct pattern
                        else:
                            enumeration_bonus = -50  # Penalty for wrong pattern

                    # SECONDARY: Explained letters (0-8 for 8-letter words)
                    primary_score = explained_letters

                    # TERTIARY: Word count penalty (fewer words = higher score)
                    word_count_factor = 1.0 / len(fodder.words)

                    # Combined score: enumeration dominates, then letters, then coherence
                    evidence_score = enumeration_bonus + primary_score + word_count_factor
                    
                    # Calculate confidence for display (0.0 to 1.0)
                    confidence = explained_letters / total_letters

                    # Calculate complete word attribution BEFORE comparison
                    # This is needed to compute compound bonus for fair comparison
                    indicator_words = list(fodder.indicator.words)
                    fodder_word_set = set(w.lower() for w in fodder.words)
                    indicator_word_set = set(w.lower() for w in indicator_words)

                    link_words_found = []
                    remaining_words = []
                    for t in tokens:
                        t_lower = t.lower()
                        if t_lower in fodder_word_set:
                            continue
                        if t_lower in indicator_word_set:
                            continue
                        if t_lower in LINK_WORDS:
                            link_words_found.append(t)
                        else:
                            remaining_words.append(t)

                    # Check if remaining words can provide needed letters via wordplay
                    # This detects compound evidence like "THE FRENCH" -> LA
                    # MOVED BEFORE comparison so all fodders get fair compound scoring
                    compound_subs = []
                    compound_evidence_type = "partial"
                    compound_confidence = confidence
                    letters_still_needed = remaining_letters
                    
                    if remaining_letters and remaining_words:
                        # Check wordplay table for substitutions
                        subs = self.lookup_wordplay_substitutions(remaining_words, remaining_letters)
                        
                        if subs:
                            # Try to find substitution(s) that complete the needed letters
                            temp_needed = remaining_letters.lower()
                            
                            for phrase, sub_letters, category in subs:
                                sub_lower = sub_letters.lower()
                                
                                # Check if this substitution's letters are all still needed
                                temp_check = temp_needed
                                can_use = True
                                for c in sub_lower:
                                    if c in temp_check:
                                        temp_check = temp_check.replace(c, '', 1)
                                    else:
                                        can_use = False
                                        break
                                
                                if can_use:
                                    compound_subs.append((phrase, sub_letters, category))
                                    temp_needed = temp_check
                                    
                                    if debug:
                                        print(f"           ★ COMPOUND: '{phrase}' -> {sub_letters} ({category})")
                            
                            letters_still_needed = temp_needed
                            
                            # If we found substitutions that explain ALL remaining letters
                            if not letters_still_needed:
                                compound_evidence_type = "compound"
                                compound_confidence = 0.95  # High confidence for complete compound
                                # Massive bonus for complete compound evidence
                                evidence_score += 75
                                
                                if debug:
                                    print(f"           ★★ COMPLETE COMPOUND EVIDENCE! New score: {evidence_score:.2f}")
                            elif len(letters_still_needed) < len(remaining_letters):
                                # Partial compound - some letters explained
                                evidence_score += 25
                                compound_confidence = (len(candidate_letters) - len(letters_still_needed)) / len(candidate_letters)

                    if debug and i < 10:
                        enum_status = "✅" if enumeration_bonus > 0 else "❌" if enumeration_bonus < 0 else "?"
                        print(
                            f"           → Score: {evidence_score:.2f} ({enum_status} enum={enumeration_bonus}, letters={explained_letters}/{total_letters}, words={len(fodder.words)}, confidence={confidence:.2f})")

                    # NOW compare with full compound-enhanced score
                    if evidence_score > best_score:
                        best_evidence = AnagramEvidence(
                            candidate=candidate,
                            fodder_words=list(fodder.words),
                            fodder_letters=fodder_letters,
                            evidence_type=compound_evidence_type,
                            confidence=compound_confidence,
                            needed_letters=letters_still_needed,
                            indicator_words=indicator_words,
                            indicator_position=fodder.indicator.start_pos,
                            link_words=link_words_found,
                            remaining_words=remaining_words,
                            unused_clue_words=remaining_words  # Backward compatibility
                        )
                        # Store compound substitutions for downstream use
                        if compound_subs:
                            best_evidence.compound_substitutions = compound_subs
                        best_score = evidence_score

                        if debug:
                            print(
                                f"           ★ NEW BEST {'COMPOUND' if compound_evidence_type == 'compound' else 'PARTIAL'} EVIDENCE! Score: {best_score:.2f}")

            # Also test for deletion anagrams (≤2 excess letters)
            if indicators.get('anagram'):
                can_delete, excess = self.can_form_by_deletion_strict(candidate_letters,
                                                                      fodder_letters)
                if can_delete:
                    if debug and i < 10:
                        print(
                            f"           Deletion: can_delete={can_delete}, excess='{excess}'")

                    # Calculate score for deletion evidence
                    # All letters are explained but we have excess, so penalize
                    deletion_explained = len(candidate_letters)
                    excess_penalty = len(excess) * 10  # Penalty per excess letter
                    
                    # Check enumeration match
                    deletion_enum_bonus = 0
                    if enumeration:
                        if self._matches_enumeration_pattern(candidate, enumeration):
                            deletion_enum_bonus = 100
                        else:
                            deletion_enum_bonus = -50
                    
                    deletion_score = deletion_enum_bonus + deletion_explained - excess_penalty + (1.0 / len(fodder.words))
                    
                    if debug and i < 10:
                        print(f"           Deletion score: {deletion_score:.2f} (enum={deletion_enum_bonus}, explained={deletion_explained}, excess_penalty={excess_penalty})")
                    
                    # Only use deletion evidence if better than current best
                    if deletion_score > best_score:
                        # Calculate complete word attribution
                        indicator_words = list(fodder.indicator.words)
                        fodder_word_set = set(w.lower() for w in fodder.words)
                        indicator_word_set = set(w.lower() for w in indicator_words)

                        link_words_found = []
                        remaining_words = []
                        for t in tokens:
                            t_lower = t.lower()
                            if t_lower in fodder_word_set:
                                continue
                            if t_lower in indicator_word_set:
                                continue
                            if t_lower in LINK_WORDS:
                                link_words_found.append(t)
                            else:
                                remaining_words.append(t)

                        deletion_confidence = 0.8

                        best_evidence = AnagramEvidence(
                            candidate=candidate,
                            fodder_words=list(fodder.words),
                            fodder_letters=fodder_letters,
                            evidence_type="deletion",
                            confidence=deletion_confidence,
                            excess_letters=excess,
                            indicator_words=indicator_words,
                            indicator_position=fodder.indicator.start_pos,
                            link_words=link_words_found,
                            remaining_words=remaining_words,
                            unused_clue_words=remaining_words  # Backward compatibility
                        )
                        best_score = deletion_score
                        
                        if debug:
                            print(f"           ★ NEW BEST DELETION EVIDENCE! Score: {best_score:.2f}")

        return best_evidence

    def _matches_enumeration_pattern(self, candidate: str, enumeration: str) -> bool:
        """
        Check if candidate matches the enumeration pattern.
        E.g., "DOORMAN" matches "(7)", "FIRE ENGINE" matches "(4,6)"
        """
        if not enumeration:
            return True

        # Parse enumeration pattern like "(4,6)" or "(7)"
        pattern = enumeration.strip('()')
        parts = [int(p.strip()) for p in pattern.split(',') if p.strip().isdigit()]

        if not parts:
            return True

        # For single words, just check total length
        candidate_letters = self.normalize_letters(candidate)
        total_expected = sum(parts)

        return len(candidate_letters) == total_expected

    def analyze_clue_for_anagram_evidence(self, clue_text: str, candidates: List[str],
                                          enumeration: str = None,
                                          debug: bool = False) -> List[AnagramEvidence]:
        """
        Analyze a clue for anagram evidence across all candidates.

        CORRECTED: Now uses proper contiguous fodder detection from indicators.

        Returns list of AnagramEvidence for candidates with evidence found.
        """
        if debug:
            print(f"\n{'=' * 60}")
            print(f"ANALYZING: {clue_text}")
            print(f"Candidates: {len(candidates)}")
            print(f"{'=' * 60}")

        # Detect all indicators
        indicators = self.detect_wordplay_indicators(clue_text)

        if debug:
            print(f"\nIndicators found:")
            for ind_type, ind_list in indicators.items():
                if ind_list and ind_type != 'anagram_matches':
                    print(f"  {ind_type}: {ind_list}")
            if indicators.get('anagram_matches'):
                print(
                    f"  anagram_matches: {len(indicators['anagram_matches'])} positions")
                for match in indicators['anagram_matches']:
                    print(f"    - {match.words} at pos {match.start_pos}-{match.end_pos}")

        # No anagram indicators = no anagram evidence possible
        if not indicators.get('anagram'):
            if debug:
                print("  No anagram indicators found - skipping anagram analysis")
            return []

        evidence_list = []

        for candidate in candidates:
            if debug:
                print(f"\n  Testing candidate: {candidate}")

            evidence = self.test_anagram_evidence(
                candidate, clue_text, indicators, enumeration, debug=debug
            )

            if evidence:
                evidence_list.append(evidence)
                if debug:
                    print(f"    ✓ Evidence found: {evidence.evidence_type}")

        if debug:
            print(f"\n{'=' * 60}")
            print(f"RESULT: {len(evidence_list)} candidates with anagram evidence")
            print(f"{'=' * 60}")

        return evidence_list

    def is_fodder_contiguous(self, candidate: str, fodder_letters: str) -> bool:
        """
        Check if fodder letters appear contiguously in candidate as an anagram.

        Args:
            candidate: The candidate word (e.g., "UNALTERED")
            fodder_letters: The fodder letters (e.g., "TREE")

        Returns:
            True if any contiguous substring of candidate is an anagram of fodder
        """
        candidate_normalized = self.normalize_letters(candidate).upper()
        fodder_normalized = fodder_letters.upper()
        fodder_length = len(fodder_normalized)

        if fodder_length == 0:
            return True

        # Check all contiguous substrings of length equal to fodder
        for i in range(len(candidate_normalized) - fodder_length + 1):
            substring = candidate_normalized[i:i + fodder_length]
            if self.is_anagram(substring, fodder_normalized):
                return True

        return False

    def calculate_anagram_score_boost(self, evidence: AnagramEvidence) -> float:
        """
        Calculate score boost for candidate based on anagram evidence quality.
        Now supports partial evidence for multi-stage solving.
        Returns additive score boost.
        """
        # CRITICAL: If no legitimate fodder letters, score is ZERO
        # This prevents candidates like PLUMMET scoring when fodder is just "7" (enumeration)
        if not evidence.fodder_letters or evidence.confidence == 0.0:
            return 0.0
        
        # Also reject if fodder_words only contains enumeration-like entries
        if evidence.fodder_words:
            real_fodder = [w for w in evidence.fodder_words 
                          if not w.strip('()[]').replace(',', '').replace('-', '').isdigit()]
            if not real_fodder:
                return 0.0
        
        base_boost = {
            'exact': 20.0,  # Complete anagram match
            'partial': 8.0,  # Partial contribution (new)
            'compound': 18.0,  # Complete compound (fodder + substitution)
            'deletion': 15.0,  # Deletion anagram
            'insertion': 12.0  # Insertion anagram
        }

        boost = base_boost.get(evidence.evidence_type, 0.0)

        # Adjust based on evidence confidence (how much of candidate is explained)
        boost *= evidence.confidence

        # Check if fodder appears contiguously in candidate
        is_contiguous = self.is_fodder_contiguous(evidence.candidate,
                                                  evidence.fodder_letters)

        # Apply 50% reduction for scattered (non-contiguous) fodder
        if not is_contiguous:
            boost *= 0.5

        # Bonus for using more clue words
        word_count_bonus = len(evidence.fodder_words) * 1.5
        
        # Bonus for compound evidence with substitutions
        if hasattr(evidence, 'compound_substitutions') and evidence.compound_substitutions:
            # Each substitution found adds credibility
            boost += len(evidence.compound_substitutions) * 5.0
        
        # NEW: Boost based on indicator confidence/frequency
        indicator_boost = 0.0
        if evidence.indicator_words:
            for ind_word in evidence.indicator_words:
                ind_lower = ind_word.lower()
                # Confidence-based boost
                conf = self.indicator_confidence.get(ind_lower, 'low')
                conf_boost = {
                    'very_high': 15.0,
                    'high': 10.0,
                    'medium': 5.0,
                    'low': 0.0
                }.get(conf, 0.0)
                indicator_boost += conf_boost
                
                # Frequency-based bonus (scaled)
                freq = self.indicator_frequency.get(ind_lower, 0)
                if freq > 50:
                    indicator_boost += 5.0  # Common indicator
                elif freq > 20:
                    indicator_boost += 2.0  # Moderately common

        # Special handling for partial evidence
        if evidence.evidence_type == "partial":
            # Additional bonus based on how much of candidate is explained
            if evidence.needed_letters:
                explained_ratio = 1.0 - (
                        len(evidence.needed_letters) / len(evidence.candidate))
                boost += explained_ratio * 5.0  # Up to 5 extra points

        return boost + word_count_bonus + indicator_boost

    def analyze_and_rank_anagram_candidates(self, clue_text: str, candidates: List[str],
                                            answer: str, debug: bool = False,
                                            definition_support: Dict[
                                                str, List[str]] = None) -> Dict[
        str, any]:
        """
        ACTUAL WORKING LOGIC MOVED FROM evidence_analysis.py

        Performs comprehensive anagram analysis and ranking for all candidates.
        This is the proven working method that evidence_analysis.py was using directly.

        UPDATED: Now uses test_anagram_evidence for each candidate to support
        deletion variants (e.g., "almost gets" -> "GET").

        UPDATED: Now accepts definition_support to weight candidates by definition
        match quality (specific phrases like "biting pain" rank higher than generic "pain").

        Args:
            clue_text: The cryptic clue text
            candidates: List of all definition candidates to analyze
            answer: The target answer for validation
            debug: Enable debug output
            definition_support: Dict mapping window phrases to candidate lists

        Returns:
            Dict containing complete ranked candidate information
        """
        if not candidates:
            return {
                "evidence_list": [],
                "scored_candidates": [],
                "answer_rank_original": None,
                "answer_rank_evidence": None,
                "ranking_improved": False,
                "evidence_found": 0
            }

        # Detect indicators once for all candidates
        indicators = self.detect_wordplay_indicators(clue_text)
        enumeration_num = len(answer) if answer else 0
        enumeration_str = str(enumeration_num) if enumeration_num else None

        if debug:
            print(f"DEBUG: Testing {len(candidates)} candidates with evidence system")
            print(f"DEBUG: Indicators found: {indicators}")

        # Test each candidate using test_anagram_evidence (supports deletion variants)
        evidence_list = []
        evidence_by_candidate = {}

        for candidate in candidates:
            candidate_upper = candidate.upper().replace(' ', '')

            # Skip if wrong length
            if enumeration_num and len(candidate_upper) != enumeration_num:
                continue

            # Use test_anagram_evidence which supports deletion variants
            evidence = self.test_anagram_evidence(
                candidate_upper, clue_text, indicators,
                enumeration=enumeration_str, debug=False
            )

            if evidence:
                evidence_list.append(evidence)
                evidence_by_candidate[candidate_upper] = evidence

                if debug:
                    print(
                        f"DEBUG: {candidate} -> evidence_type={evidence.evidence_type}, confidence={evidence.confidence}")

        # Also run brute force for candidates not found by evidence system
        # This ensures we don't miss any valid anagrams
        from solver.wordplay.anagram.anagram_stage import generate_anagram_hypotheses
        hypotheses = generate_anagram_hypotheses(clue_text, enumeration_num, candidates)

        for hyp in hypotheses:
            hyp_candidate = hyp.get("answer", "").upper().replace(' ', '')
            if hyp_candidate not in evidence_by_candidate:
                # Convert to AnagramEvidence
                confidence_raw = hyp.get("confidence", 1.0)
                if isinstance(confidence_raw, str):
                    confidence_map = {'provisional': 0.5, 'high': 0.9, 'medium': 0.7,
                                      'low': 0.3}
                    confidence = confidence_map.get(confidence_raw.lower(), 0.5)
                else:
                    confidence = confidence_raw

                evidence = AnagramEvidence(
                    candidate=hyp.get("answer", ""),
                    fodder_words=hyp.get("fodder_words", []),
                    fodder_letters=hyp.get("fodder_letters", ""),
                    evidence_type=hyp.get("evidence_type",
                                          hyp.get("solve_type", "exact")),
                    confidence=confidence,
                    excess_letters=hyp.get("excess_letters", ""),
                    needed_letters=hyp.get("needed_letters", ""),
                    unused_clue_words=hyp.get("unused_words", [])
                )
                evidence_list.append(evidence)
                evidence_by_candidate[hyp_candidate] = evidence

        # Create scored candidates list - PRESERVES ALL RANKED CANDIDATE INFORMATION
        scored_candidates = []

        for candidate in candidates:
            candidate_upper = candidate.upper().replace(' ', '')
            evidence = evidence_by_candidate.get(candidate_upper)

            # Calculate evidence score boost using existing proven method
            evidence_score = 0.0
            if evidence:
                evidence_score = self.calculate_anagram_score_boost(evidence)

                # Boost exact/compound matches (including deletion/doubling variants) over partial
                if evidence.evidence_type in ('exact', 'exact_with_deletion',
                                              'exact_with_doubling', 'compound'):
                    evidence_score += 50  # Significant boost for complete evidence

            # Add definition match quality bonus (if definition_support provided)
            # ONLY add if there's already legitimate anagram evidence (score > 0)
            # This prevents definition-only matches from ranking in the anagram track
            if definition_support and evidence_score > 0:
                # Find windows that produced this candidate
                candidate_windows = set()
                for window, cands in definition_support.items():
                    if candidate in cands or candidate.upper() in [c.upper() for c in
                                                                   cands]:
                        candidate_windows.add(window)

                if candidate_windows:
                    # Bonus for number of matching windows
                    evidence_score += len(candidate_windows) * 5

                    # Bonus for longer/more specific windows
                    # "biting pain" (2 words) > "pain" (1 word)
                    max_window_words = max(len(w.split()) for w in candidate_windows)
                    evidence_score += max_window_words * 10  # Significant boost for specific phrases

            scored_candidates.append({
                "candidate": candidate,
                "evidence_score": evidence_score,
                "evidence": evidence,
                "has_evidence": evidence is not None
            })

        # Sort by evidence score (highest first) - PRESERVES COMPLETE RANKING
        scored_candidates.sort(key=lambda x: x["evidence_score"], reverse=True)

        # Find answer ranking in scored list
        answer_rank_evidence = None
        for i, scored in enumerate(scored_candidates, 1):
            if scored["candidate"].upper() == answer.upper():
                answer_rank_evidence = i
                break

        # Find original answer ranking (unscored)
        answer_rank_original = None
        for i, candidate in enumerate(candidates, 1):
            if candidate.upper() == answer.upper():
                answer_rank_original = i
                break

        # Return complete ranked candidate information
        return {
            "evidence_list": evidence_list,
            "scored_candidates": scored_candidates,  # COMPLETE RANKED LIST
            "answer_rank_original": answer_rank_original,
            "answer_rank_evidence": answer_rank_evidence,
            "ranking_improved": (answer_rank_evidence and answer_rank_original and
                                 answer_rank_evidence < answer_rank_original),
            "evidence_found": len(evidence_list)
        }