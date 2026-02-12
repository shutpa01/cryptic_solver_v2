#!/usr/bin/env python3
"""
Explanation Builder - Presentation layer for cryptic crossword solver.

This module handles:
1. Building formula notation for solved clues
2. Creating word-by-word breakdowns
3. Assessing explanation quality
4. Formatting output for display

Separated from compound_wordplay_analyzer.py to maintain separation of concerns:
- Analysis (compound_wordplay_analyzer.py) - finding indicators, solving wordplay
- Presentation (this file) - building explanations, formatting output
"""

import re
import sys
from typing import List, Dict, Any, Optional


# Local implementation of norm_letters
def norm_letters(s: str) -> str:
    """Normalize a string to only lowercase letters."""
    return ''.join(c.lower() for c in s if c.isalpha())


# Import from local compound_wordplay_analyzer
from compound_wordplay_analyzer import (
    WordRole, CompoundWordplayAnalyzer
)


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


class ExplanationBuilder:
    """Builds explanations from analyzed compound wordplay cases."""

    def __init__(self):
        # Link words to exclude from word counts
        self.link_words = {
            'a', 'an', 'the', 'to', 'for', 'of', 'in', 'on', 'at', 'by',
            'is', 'it', 'as', 'be', 'or', 'and', 'with', 'from', 'into',
            'that', 'this', 'are', 'was', 'were', 'been', 'being',
            'has', 'have', 'had', 'do', 'does', 'did', 'will', 'would',
            'could', 'should', 'may', 'might', 'must', 'shall',
            'its', 'his', 'her', 'your', 'their', 'our', 'my',
            'but', 'yet', 'so', 'if', 'then', 'than', 'when', 'where',
            'who', 'what', 'which', 'how', 'why',
            'one', 'ones', 'some', 'any', 'all', 'each', 'every',
            'no', 'not', 'nor', 'neither', 'either',
            'up', 'down', 'out', 'off', 'over', 'under',
            'again', 'further', 'once', 'here', 'there',
            'about', 'after', 'before', 'between', 'through',
            'during', 'above', 'below', 'against', 'among',
            'such', 'only', 'just', 'also', 'very', 'too',
            'well', 'back', 'even', 'still', 'already',
            'always', 'never', 'ever', 'often', 'sometimes',
            'perhaps', 'maybe', 'rather', 'quite', 'almost',
            'get', 'gets', 'getting', 'got', 'make', 'makes', 'making', 'made',
            'give', 'gives', 'giving', 'gave', 'given',
            'take', 'takes', 'taking', 'took', 'taken',
            'come', 'comes', 'coming', 'came',
            'go', 'goes', 'going', 'went', 'gone',
            'see', 'sees', 'seeing', 'saw', 'seen',
            'say', 'says', 'saying', 'said',
            'know', 'knows', 'knowing', 'knew', 'known',
            'think', 'thinks', 'thinking', 'thought',
            'find', 'finds', 'finding', 'found',
            'want', 'wants', 'wanting', 'wanted',
            'use', 'uses', 'using', 'used',
            'try', 'tries', 'trying', 'tried',
            'need', 'needs', 'needing', 'needed',
            'seem', 'seems', 'seeming', 'seemed',
            'feel', 'feels', 'feeling', 'felt',
            'become', 'becomes', 'becoming', 'became',
            'leave', 'leaves', 'leaving', 'left',
            'put', 'puts', 'putting',
            'mean', 'means', 'meaning', 'meant',
            'keep', 'keeps', 'keeping', 'kept',
            'let', 'lets', 'letting',
            'begin', 'begins', 'beginning', 'began', 'begun',
            'show', 'shows', 'showing', 'showed', 'shown',
            'hear', 'hears', 'hearing', 'heard',
            'play', 'plays', 'playing', 'played',
            'run', 'runs', 'running', 'ran',
            'move', 'moves', 'moving', 'moved',
            'like', 'likes', 'liking', 'liked',
            'live', 'lives', 'living', 'lived',
            'believe', 'believes', 'believing', 'believed',
            'hold', 'holds', 'holding', 'held',
            'bring', 'brings', 'bringing', 'brought',
            'happen', 'happens', 'happening', 'happened',
            'write', 'writes', 'writing', 'wrote', 'written',
            'provide', 'provides', 'providing', 'provided',
            'sit', 'sits', 'sitting', 'sat',
            'stand', 'stands', 'standing', 'stood',
            'lose', 'loses', 'losing', 'lost',
            'pay', 'pays', 'paying', 'paid',
            'meet', 'meets', 'meeting', 'met',
            'include', 'includes', 'including', 'included',
            'continue', 'continues', 'continuing', 'continued',
            'set', 'sets', 'setting',
            'learn', 'learns', 'learning', 'learned', 'learnt',
            'change', 'changes', 'changing', 'changed',
            'lead', 'leads', 'leading', 'led',
            'understand', 'understands', 'understanding', 'understood',
            'watch', 'watches', 'watching', 'watched',
            'follow', 'follows', 'following', 'followed',
            'stop', 'stops', 'stopping', 'stopped',
            'create', 'creates', 'creating', 'created',
            'speak', 'speaks', 'speaking', 'spoke', 'spoken',
            'read', 'reads', 'reading',
            'allow', 'allows', 'allowing', 'allowed',
            'add', 'adds', 'adding', 'added',
            'spend', 'spends', 'spending', 'spent',
            'grow', 'grows', 'growing', 'grew', 'grown',
            'open', 'opens', 'opening', 'opened',
            'walk', 'walks', 'walking', 'walked',
            'win', 'wins', 'winning', 'won',
            'offer', 'offers', 'offering', 'offered',
            'remember', 'remembers', 'remembering', 'remembered',
            'love', 'loves', 'loving', 'loved',
            'consider', 'considers', 'considering', 'considered',
            'appear', 'appears', 'appearing', 'appeared',
            'buy', 'buys', 'buying', 'bought',
            'wait', 'waits', 'waiting', 'waited',
            'serve', 'serves', 'serving', 'served',
            'die', 'dies', 'dying', 'died',
            'send', 'sends', 'sending', 'sent',
            'expect', 'expects', 'expecting', 'expected',
            'build', 'builds', 'building', 'built',
            'stay', 'stays', 'staying', 'stayed',
            'fall', 'falls', 'falling', 'fell', 'fallen',
            'cut', 'cuts', 'cutting',
            'reach', 'reaches', 'reaching', 'reached',
            'kill', 'kills', 'killing', 'killed',
            'remain', 'remains', 'remaining', 'remained',
            'suggest', 'suggests', 'suggesting', 'suggested',
            'raise', 'raises', 'raising', 'raised',
            'pass', 'passes', 'passing', 'passed',
            'sell', 'sells', 'selling', 'sold',
            'require', 'requires', 'requiring', 'required',
            'report', 'reports', 'reporting', 'reported',
            'decide', 'decides', 'deciding', 'decided',
            'pull', 'pulls', 'pulling', 'pulled'
        }

    def build_explanation(self, case: Dict[str, Any],
                          word_roles: List[WordRole],
                          fodder_words: List[str],
                          fodder_letters: str,
                          anagram_indicator: Optional[str],
                          definition_window: Optional[str],
                          compound_solution: Optional[Dict[str, Any]],
                          clue_words: List[str],
                          likely_answer: str) -> Dict[str, Any]:
        """
        Build a complete explanation following the format spec.

        Uses likely_answer (from matched candidate), NOT database answer.
        """
        # Format likely_answer with spaces for multi-word answers
        enumeration = case.get('enumeration', '')
        likely_answer = format_answer_with_enumeration(likely_answer, enumeration)

        # Get substitutions from compound solution
        subs = []
        if compound_solution and compound_solution.get('substitutions'):
            subs = compound_solution['substitutions']

        # Get additional fodder from compound solution
        additional_fodder = []
        if compound_solution and compound_solution.get('additional_fodder'):
            additional_fodder = compound_solution['additional_fodder']

        # Check for deletion_source contributions in word_roles
        deletion_sources = [(wr.word, wr.contributes, wr.source)
                            for wr in word_roles if wr.role == 'deletion_source']

        # Build fodder part - include both original and additional fodder
        all_fodder_words = list(fodder_words) if fodder_words else []
        for word, letters in additional_fodder:
            all_fodder_words.append(word.upper().replace("'", ""))  # Strip apostrophes

        fodder_part = ' + '.join(
            w.upper() for w in all_fodder_words) if all_fodder_words else ''

        # Check for deletion operation (only if validated with indicator)
        if compound_solution and compound_solution.get('operation') == 'deletion':
            deletion_target = compound_solution.get('deletion_target')
            excess = compound_solution.get('excess_letters', '')

            if deletion_target:
                word, letters, category = deletion_target
                formula = f"anagram({fodder_part}) - {letters} ({word}) = {likely_answer}"
            else:
                formula = f"anagram({fodder_part}) - {excess} = {likely_answer}"
        elif deletion_sources:
            # Orphaned deletion case - deletion_source found via word_roles
            del_part = ' + '.join(
                f"{letters} ({word})" for word, letters, _ in deletion_sources)
            formula = f"{del_part} + anagram({fodder_part}) = {likely_answer}"
        elif compound_solution and compound_solution.get('operation') == 'reduced_fodder':
            # Reduced fodder with substitution - show the corrected fodder
            reduced = compound_solution.get('reduced_fodder', '')
            subs_from_compound = compound_solution.get('substitutions', [])
            if subs_from_compound:
                sub_part = ' + '.join(
                    f"{letters} ({word})" for word, letters, _ in subs_from_compound)
                # Get the actual fodder words (not the removed one)
                actual_fodder = [wr.word for wr in word_roles if wr.role == 'fodder']
                fodder_part = ' + '.join(w.upper() for w in actual_fodder)
                formula = f"anagram({fodder_part}) + {sub_part} = {likely_answer}"
            else:
                formula = f"anagram({reduced}) = {likely_answer}"
        elif compound_solution and compound_solution.get(
                'operation') == 'unresolved_excess':
            # Excess letters but no deletion indicator - show honest formula
            formula = f"anagram({fodder_part}) = {likely_answer} [excess letters unresolved]"
        elif subs:
            # Compound with substitutions (additions)
            sub_part = ' + '.join(f"{letters} ({word})" for word, letters, _ in subs)

            construction = compound_solution.get('construction',
                                                 {}) if compound_solution else {}
            op = construction.get('operation', 'concatenation')

            if op == 'insertion':
                formula = f"anagram({fodder_part}) with {sub_part} inserted = {likely_answer}"
            elif op == 'container':
                formula = f"{sub_part} inside anagram({fodder_part}) = {likely_answer}"
            else:
                formula = f"anagram({fodder_part}) + {sub_part} = {likely_answer}"
        else:
            # Pure anagram
            formula = f"anagram({fodder_part}) = {likely_answer}"

        # Build word-by-word explanation following clue order
        explanations = []
        # Use norm_letters for lookup to handle punctuation (e.g., "gate," matches "gate")
        role_lookup = {norm_letters(wr.word): wr for wr in word_roles}

        # Also build a lookup for phrase roles (roles where word contains space)
        phrase_role_lookup = {}
        for wr in word_roles:
            if ' ' in wr.word:
                # Store by normalized form of the phrase
                phrase_key = norm_letters(wr.word)
                phrase_role_lookup[phrase_key] = wr

        explained_words = set()
        skip_next = 0  # Number of words to skip (for phrase matches)

        for i, word in enumerate(clue_words):
            # Skip if this word was part of a previous phrase match
            if skip_next > 0:
                skip_next -= 1
                continue

            word_norm = norm_letters(word)

            if word_norm in explained_words:
                continue

            # First check for phrase matches (current word + next word(s))
            phrase_matched = False
            if i < len(clue_words) - 1:
                # Try two-word phrase
                two_word_phrase = f"{word} {clue_words[i + 1]}"
                two_word_norm = norm_letters(two_word_phrase)
                if two_word_norm in phrase_role_lookup:
                    wr = phrase_role_lookup[two_word_norm]
                    explained_words.add(two_word_norm)

                    if wr.role == 'substitution':
                        explanations.append(
                            f'• "{two_word_phrase.lower()}" = {wr.contributes} ({wr.source})')
                    else:
                        explanations.append(f'• "{two_word_phrase.lower()}" = {wr.role}')

                    skip_next = 1  # Skip the next word
                    phrase_matched = True

            if phrase_matched:
                continue

            if word_norm in role_lookup:
                wr = role_lookup[word_norm]
                explained_words.add(word_norm)

                if wr.role == 'definition':
                    explanations.append(f'• "{word}" = definition for {likely_answer}')
                elif wr.role == 'fodder':
                    # Check if this is truncated fodder (has source like "minus last letter")
                    if wr.source and 'minus' in wr.source.lower():
                        explanations.append(
                            f'• "{word}" = {wr.contributes} ({wr.source})')
                    else:
                        explanations.append(f'• "{word}" = anagram fodder')
                elif wr.role == 'anagram_indicator':
                    explanations.append(f'• "{word}" = anagram indicator')
                elif wr.role == 'substitution':
                    explanations.append(f'• "{word}" = {wr.contributes} ({wr.source})')
                elif wr.role == 'deletion_target':
                    explanations.append(f'• "{word}" = {wr.contributes} (to be removed)')
                elif wr.role == 'deletion_indicator':
                    explanations.append(f'• "{word}" = deletion indicator')
                elif wr.role == 'deletion_source':
                    explanations.append(f'• "{word}" = {wr.contributes} ({wr.source})')
                elif wr.role == 'positional_indicator':
                    explanations.append(
                        f'• "{word}" = positional indicator (construction order)')
                elif wr.role == 'insertion_indicator':
                    explanations.append(f'• "{word}" = insertion indicator')
                elif wr.role == 'insertion_material':
                    explanations.append(f'• "{word}" = {wr.contributes} (inserted)')
                elif wr.role == 'container_indicator':
                    explanations.append(f'• "{word}" = container indicator')
                elif wr.role == 'operation_indicator':
                    # Extract operation type from source (e.g., "database (container)")
                    if 'container' in wr.source.lower():
                        explanations.append(f'• "{word}" = container indicator')
                    elif 'insertion' in wr.source.lower():
                        explanations.append(f'• "{word}" = insertion indicator')
                    else:
                        explanations.append(f'• "{word}" = operation indicator')
                elif wr.role == 'parts_indicator':
                    # Extract subtype from source if available
                    if 'delete' in wr.source.lower():
                        if 'last' in wr.source.lower():
                            explanations.append(
                                f'• "{word}" = truncation indicator (remove last letter)')
                        elif 'first' in wr.source.lower():
                            explanations.append(
                                f'• "{word}" = truncation indicator (remove first letter)')
                        else:
                            explanations.append(f'• "{word}" = truncation indicator')
                    elif 'last' in wr.source.lower():
                        explanations.append(f'• "{word}" = last letter indicator')
                    elif 'first' in wr.source.lower():
                        explanations.append(f'• "{word}" = first letter indicator')
                    else:
                        explanations.append(f'• "{word}" = letter selector')
                elif wr.role == 'link':
                    pass  # Skip link words in explanation
                else:
                    # Generic indicator
                    if '_indicator' in wr.role:
                        ind_type = wr.role.replace('_indicator', '')
                        explanations.append(f'• "{word}" = {ind_type} indicator')

        # Add any unaccounted words at the end
        for word in clue_words:
            word_norm = norm_letters(word)
            if word_norm and word_norm not in explained_words:
                # Check if it's part of a phrase we already explained
                already_in_phrase = False
                for explained in explained_words:
                    if word_norm in explained:
                        already_in_phrase = True
                        break

                if not already_in_phrase:
                    # Check if it's a common link word
                    if word_norm in self.link_words:
                        explanations.append(f'• "{word}" = link word')
                    else:
                        explanations.append(f'• "{word}" = wordplay connector')

        return {
            'formula': formula,
            'breakdown': explanations,
            'quality': self.assess_quality(compound_solution, word_roles, clue_words,
                                           likely_answer, case.get('answer', ''),
                                           fodder_letters)
        }

    def assess_quality(self, compound_solution: Optional[Dict[str, Any]],
                       word_roles: List[WordRole],
                       clue_words: List[str],
                       likely_answer: str = '',
                       db_answer: str = '',
                       fodder_letters: str = '') -> str:
        """
        Assess explanation quality based on word coverage AND answer correctness.

        CRITICAL: An explanation is only 'solved' if:
        1. The likely_answer matches the db_answer
        2. The letter math works (fodder + substitutions = answer length)
        3. Word coverage is high
        """
        # CRITICAL CHECK 1: Answer must match
        answer_matches = False
        if likely_answer and db_answer:
            # Normalize for comparison (strip spaces, uppercase)
            likely_norm = likely_answer.upper().replace(' ', '')
            db_norm = db_answer.upper().replace(' ', '')
            answer_matches = (likely_norm == db_norm)

        # CRITICAL CHECK 2: Letter math must work
        letter_math_valid = False
        if fodder_letters and likely_answer:
            answer_letters = len(likely_answer.replace(' ', ''))
            fodder_len = len(fodder_letters)

            # Get substitution letters if any
            sub_letters = 0
            if compound_solution and compound_solution.get('substitutions'):
                for _, letters, _ in compound_solution['substitutions']:
                    sub_letters += len(letters)

            # For exact anagram: fodder = answer
            # For compound: fodder + substitutions = answer
            # For deletion: fodder - excess = answer
            if compound_solution and compound_solution.get('operation') == 'deletion':
                excess = compound_solution.get('excess_letters', '')
                letter_math_valid = (fodder_len - len(excess) == answer_letters)
            else:
                # Exact or compound
                letter_math_valid = (fodder_len + sub_letters == answer_letters)

        # Calculate word coverage (existing logic)
        accounted = {norm_letters(wr.word) for wr in word_roles}
        total_words = len(
            [w for w in clue_words if norm_letters(w) not in self.link_words])
        accounted_content = len([w for w in clue_words
                                 if norm_letters(w) in accounted and norm_letters(
                w) not in self.link_words])

        # Calculate coverage ratio
        coverage = accounted_content / total_words if total_words > 0 else 0

        # Check for required elements
        has_definition = any(wr.role == 'definition' for wr in word_roles)
        has_fodder = any(wr.role == 'fodder' for wr in word_roles)
        has_indicator = any('indicator' in wr.role for wr in word_roles)

        # CRITICAL: Wrong answer = INCORRECT
        if not answer_matches:
            return 'INCORRECT'

        # Answer matches - now check letter math
        if not letter_math_valid and fodder_letters:
            # Letter counts don't add up - suspicious, cap at 'medium'
            if coverage >= 0.7:
                return 'medium'
            else:
                return 'low'

        # Both checks passed - use coverage-based quality
        # Compound with substitutions
        if compound_solution and compound_solution.get('fully_resolved'):
            if coverage >= 0.9:
                return 'solved'
            else:
                return 'high'

        # Pure anagram - check if fully explained
        if has_definition and has_fodder and has_indicator:
            if coverage >= 0.9:
                return 'solved'
            elif coverage >= 0.7:
                return 'high'
            else:
                return 'medium'

        # Partial explanation
        if compound_solution and compound_solution.get('substitutions'):
            return 'medium'
        elif has_fodder and has_indicator:
            return 'medium'
        elif has_fodder:
            return 'low'
        else:
            return 'none'

    def build_fallback(self, case: Dict[str, Any],
                       clue_words: List[str], db_answer: str) -> Dict[str, Any]:
        """Fallback when no evidence available."""
        # Format db_answer with spaces for multi-word enumerations
        enumeration = case.get('enumeration', '')
        db_answer_formatted = format_answer_with_enumeration(db_answer, enumeration)

        return {
            'clue': case.get('clue', ''),
            'likely_answer': '',  # No likely answer - couldn't solve
            'db_answer': db_answer_formatted,  # For comparison only
            'answer_matches': False,
            'word_roles': [],
            'definition_window': None,
            'anagram_component': None,
            'compound_solution': None,
            'explanation': {
                'formula': 'Unable to analyze',
                'breakdown': [],
                'quality': 'none'
            },
            'remaining_unresolved': clue_words
        }


class ExplanationSystemBuilder:
    """
    Wrapper class maintaining backward compatibility with pipeline.
    Processes cases through the CompoundWordplayAnalyzer and ExplanationBuilder.
    """

    def __init__(self):
        self.analyzer = CompoundWordplayAnalyzer()
        self.explainer = ExplanationBuilder()

    def close(self):
        self.analyzer.close()

    def enhance_pipeline_data(self, evidence_enhanced_cases: List[Dict[str, Any]]) -> \
            List[Dict[str, Any]]:
        """
        Process evidence-enhanced cases through compound analysis.
        """
        enhanced = []

        print("Analyzing compound wordplay with database lookups...")

        for i, case in enumerate(evidence_enhanced_cases):
            if (i + 1) % 100 == 0:
                print(f"  Processed {i + 1} cases...")

            try:
                result = self.analyzer.analyze_case(case)
                enhanced.append(result)
            except Exception as e:
                print(f"Warning: Could not analyze case {i + 1}: {e}")
                enhanced.append({
                    'clue': case.get('clue', ''),
                    'answer': case.get('answer', ''),
                    'error': str(e)
                })

        print(f"Analyzed {len(enhanced)} cases.")
        return enhanced

    def build_explanations(self, enhanced_cases: List[Dict[str, Any]]) -> List[
        Dict[str, Any]]:
        """
        Extract explanations from analyzed cases.
        """
        return [
            {
                'clue': case.get('clue', ''),
                'likely_answer': case.get('likely_answer', ''),
                'db_answer': case.get('db_answer', ''),
                'answer_matches': case.get('answer_matches', False),
                'explanation': case.get('explanation', {}),
                'quality': case.get('explanation', {}).get('quality', 'none')
            }
            for case in enhanced_cases
        ]