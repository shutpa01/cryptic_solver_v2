"""Tier 2 test: present grammar-derived candidates to AI as multiple choice.

For each clue that Tier 1 can't solve mechanically, we:
1. Identify possible word roles from grammar + DB
2. Build 2-3 candidate decompositions
3. Present to AI: "which of these explains this clue?"
4. Verify AI's choice against the answer

This tests the hypothesis that focused multiple-choice is cheaper
and more accurate than open-ended "solve this clue".
"""

import sqlite3
import sys
import re
import os
import json
import spacy
from anthropic import Anthropic

sys.path.insert(0, '.')

from cryptic_taxonomy.analysis.evidence_triage import QuickRefDB
from cryptic_taxonomy.analysis.mine_positional_signatures import IndicatorDB
from signature_solver.tokens import LINK_WORDS


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")


def build_candidates(wp_words, answer, db, ind_db):
    """Build candidate decompositions for a clue.

    For each word, determine what it COULD be:
    - If it has synonyms/abbreviations that appear in the answer: FODDER candidate
    - If it's in the indicator DB: INDICATOR candidate
    - If it's a link word: LINK candidate
    - Positional: first/last letter
    - Reversal: reversed synonym in answer

    Returns a structured description of possibilities.
    """
    n = len(wp_words)
    answer_len = len(answer)

    word_analysis = []
    for i, word in enumerate(wp_words):
        roles = []
        raw = ''.join(c for c in word.upper() if c.isalpha())

        # Synonym/abbreviation values that appear in answer
        for val, src in db.get_values(word, answer_len):
            if val in answer:
                roles.append(f'{src}={val}')

        # Phrase lookups (2-word)
        if i < n - 1:
            for val, src in db.get_phrase_values(wp_words[i:i+2], answer_len):
                if val in answer:
                    roles.append(f'phrase({wp_words[i]}+{wp_words[i+1]})={val}')

        # Raw letters in answer
        if raw and raw in answer:
            roles.append(f'raw={raw}')

        # First/last letter
        if raw:
            if raw[0] in answer:
                roles.append(f'first_letter={raw[0]}')
            if len(raw) >= 2 and raw[-1] in answer:
                roles.append(f'last_letter={raw[-1]}')

        # Reversal
        for val, src in db.get_values(word, answer_len):
            if len(val) >= 2 and val[::-1] in answer and val[::-1] != val:
                roles.append(f'reversal={val[::-1]}(from {val})')

        # Indicator types
        ind_types = ind_db.get_indicator_types(word)
        if ind_types:
            roles.append(f'indicator({",".join(sorted(ind_types))})')

        # Link word
        if _clean(word) in LINK_WORDS:
            roles.append('link')

        word_analysis.append({
            'word': word,
            'possible_roles': roles,
        })

    return word_analysis


def build_prompt(clue_text, answer, definition, word_analysis):
    """Build the Tier 2 AI prompt."""

    analysis_text = []
    for wa in word_analysis:
        if wa['possible_roles']:
            analysis_text.append(f"  {wa['word']}: {', '.join(wa['possible_roles'])}")
        else:
            analysis_text.append(f"  {wa['word']}: (no known role)")

    prompt = f"""You are solving a cryptic crossword clue. The answer is known.
Your job: explain HOW the wordplay produces the answer.

Clue: {clue_text}
Answer: {answer} ({len(answer.replace(' ','').replace('-',''))} letters)
Definition: {definition or 'unknown - you must identify it'}

For each word in the wordplay, here are its possible roles based on database lookups:
{chr(10).join(analysis_text)}

Using ONLY the roles listed above, explain which role each word plays and how the
pieces assemble to produce the answer. The pieces must concatenate, be inserted into
each other, be reversed, or be anagrammed to give the exact answer.

Format your response as:
OPERATION: [charade/container/anagram/reversal/deletion]
DEFINITION: [the definition word(s)]
DECOMPOSITION: [word1]=VALUE(role) + [word2]=VALUE(role) + ... = ANSWER
EXPLANATION: [brief explanation]

If you cannot find a valid decomposition from the listed roles, say CANNOT_SOLVE."""

    return prompt


def run():
    nlp = spacy.load('en_core_web_sm')
    ind_db = IndicatorDB()
    db = QuickRefDB()

    from dotenv import load_dotenv
    load_dotenv()
    client = Anthropic()

    conn = sqlite3.connect('data/clues_master.db')
    rows = conn.execute('''
        SELECT clue_number, direction, clue_text, answer, definition
        FROM clues
        WHERE source = 'telegraph' AND puzzle_number = 31187
        ORDER BY clue_number, direction
    ''').fetchall()
    conn.close()

    # The clues that Tier 1 failed on
    failed_labels = {'1a','1d','11d','17a','2d','20d','22a','24a',
                     '25a','26a','27a','3d','4d','5d','7d','8d'}

    total_cost = 0
    results = []

    for num, direction, clue_text, answer, definition in rows:
        label = f'{num}{direction[0] if direction else "?"}'
        if label not in failed_labels:
            continue

        answer_clean = answer.upper().replace(' ', '').replace('-', '')

        # Extract wordplay window
        if definition:
            clue_lower = clue_text.lower().strip()
            defn_lower = definition.lower().strip()
            if clue_lower.startswith(defn_lower):
                wp = clue_text[len(definition):].strip().strip('.,;:!? ')
            elif clue_lower.endswith(defn_lower):
                wp = clue_text[:len(clue_text) - len(definition)].strip().strip('.,;:!? ')
            else:
                wp = clue_text
        else:
            wp = clue_text

        wp_words = wp.split() if wp else clue_text.split()

        # Build candidate analysis
        word_analysis = build_candidates(wp_words, answer_clean, db, ind_db)

        # Build prompt
        prompt = build_prompt(clue_text, answer, definition, word_analysis)

        print(f'\n{"="*60}')
        print(f'{label}: {clue_text}')
        print(f'Answer: {answer}')
        print(f'Definition: {definition}')
        print(f'Wordplay: {" ".join(wp_words)}')
        print(f'Word analysis:')
        for wa in word_analysis:
            print(f'  {wa["word"]}: {wa["possible_roles"][:5]}')

        # Call API
        response = client.messages.create(
            model='claude-sonnet-4-20250514',
            max_tokens=300,
            temperature=0,
            messages=[{'role': 'user', 'content': prompt}],
        )

        reply = response.content[0].text
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens
        cost = (input_tokens * 3 / 1_000_000) + (output_tokens * 15 / 1_000_000)
        total_cost += cost

        print(f'\nAI response ({input_tokens}+{output_tokens} tokens, ${cost:.4f}):')
        safe_reply = reply.encode('ascii', errors='replace').decode('ascii')
        print(safe_reply)

        results.append({
            'label': label,
            'clue': clue_text,
            'answer': answer,
            'reply': reply,
            'cost': cost,
        })

    print(f'\n{"="*60}')
    print(f'SUMMARY')
    print(f'{"="*60}')
    print(f'Clues tested: {len(results)}')
    print(f'Total cost: ${total_cost:.4f}')
    print(f'Avg cost per clue: ${total_cost/len(results):.4f}')


if __name__ == '__main__':
    run()
