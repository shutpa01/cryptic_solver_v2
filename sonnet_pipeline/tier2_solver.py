"""Tier 2 solver: grammar-guided AI multiple choice.

Replaces the expensive open-ended Sonnet call with a cheap structured
prompt. Builds word-level analysis from grammar + DB, asks AI to pick
the correct decomposition.

Returns the same result dict format as sonnet_pipeline.solver.solve_clue()
so it plugs directly into the existing pipeline.
"""

import os
import re
import json

from dotenv import load_dotenv
load_dotenv()

from anthropic import Anthropic

from signature_solver.grammar_triage import (
    _clean, _get_word_values, _get_phrase_values,
)
from signature_solver.tokens import LINK_WORDS


_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        _CLIENT = Anthropic()
    return _CLIENT


TIER2_MODEL = "claude-sonnet-4-6"


def build_word_analysis(wp_words, answer, db):
    """Build structured analysis of possible roles for each word."""
    answer_len = len(answer)
    n = len(wp_words)
    analysis = []

    for i, word in enumerate(wp_words):
        roles = []
        raw = ''.join(c for c in word.upper() if c.isalpha())

        # Synonym/abbreviation values in answer
        for val, src in _get_word_values(word, db, answer_len):
            if val in answer:
                roles.append('%s=%s' % (src, val))

        # Phrase lookups
        if i < n - 1:
            for val, src in _get_phrase_values(wp_words[i:i+2], db, answer_len):
                if val in answer:
                    roles.append('phrase(%s+%s)=%s' % (wp_words[i], wp_words[i+1], val))

        # Raw letters
        if raw and raw in answer:
            roles.append('raw=%s' % raw)

        # First/last letter
        if raw:
            if raw[0] in answer:
                roles.append('first_letter=%s' % raw[0])
            if len(raw) >= 2 and raw[-1] in answer:
                roles.append('last_letter=%s' % raw[-1])

        # Reversal
        for val, src in _get_word_values(word, db, answer_len):
            if len(val) >= 2 and val[::-1] in answer and val[::-1] != val:
                roles.append('reversal=%s(from %s)' % (val[::-1], val))

        # Indicator types
        ind_types = db.get_indicator_types(_clean(word))
        if ind_types:
            types = sorted(set(t for t, _, _ in ind_types))
            roles.append('indicator(%s)' % ','.join(types))

        # Link word
        if _clean(word) in LINK_WORDS:
            roles.append('link')

        analysis.append({'word': word, 'roles': roles})

    return analysis


def build_prompt(clue_text, answer, definition, word_analysis):
    """Build the Tier 2 structured prompt."""
    analysis_lines = []
    for wa in word_analysis:
        if wa['roles']:
            analysis_lines.append('  %s: %s' % (wa['word'], ', '.join(wa['roles'])))
        else:
            analysis_lines.append('  %s: (no known role)' % wa['word'])

    prompt = (
        "You are solving a cryptic crossword clue. The answer is known.\n"
        "Your job: explain HOW the wordplay produces the answer.\n\n"
        "Clue: %s\n"
        "Answer: %s (%d letters)\n"
        "Definition: %s\n\n"
        "For each word in the wordplay, here are its possible roles:\n"
        "%s\n\n"
        "Using ONLY the roles listed above, explain which role each word plays\n"
        "and how the pieces assemble to produce the answer.\n\n"
        "Respond with ONLY valid JSON (no markdown, no explanation outside JSON):\n"
        '{"definition": "...", "wordplay_type": "...", '
        '"pieces": [{"clue_word": "...", "letters": "...", "mechanism": "..."}]}\n\n'
        "mechanism must be one of: synonym, abbreviation, first_letter, last_letter,\n"
        "anagram_fodder, reversal, raw, container_outer, container_inner, indicator, link\n\n"
        "If you cannot find a valid decomposition, respond: {\"error\": \"cannot_solve\"}"
    ) % (
        clue_text, answer, len(answer.replace(' ', '').replace('-', '')),
        definition or 'unknown - you must identify it',
        '\n'.join(analysis_lines),
    )

    return prompt


def tier2_solve(clue, answer, db, ref_db=None):
    """Run Tier 2 solver on a single clue.

    Returns a result dict compatible with sonnet_pipeline.solver.solve_clue()
    output, or None if Tier 2 can't solve it.
    """
    target = answer.upper().replace(' ', '').replace('-', '')

    # Use ref_db if provided, otherwise db
    lookup_db = ref_db if ref_db is not None else db

    # Extract definition to separate it from wordplay
    definition = None
    wp_words = clue.strip().split()
    try:
        from signature_solver.solver import extract_definition_candidates, _normalize_clue
        clue_words = _normalize_clue(clue).strip().split()
        candidates = extract_definition_candidates(clue_words, target, lookup_db)
        if candidates:
            definition = candidates[0][0]
            wp_words = candidates[0][1]
        else:
            # Try Haiku fallback
            try:
                from signature_solver.haiku_definition import find_definition
                haiku_result = find_definition(clue, answer)
                if haiku_result:
                    definition = haiku_result[0]
                    wp_words = haiku_result[1]
            except Exception:
                pass
    except Exception:
        pass

    # Build word analysis on WORDPLAY words only (not definition)
    word_analysis = build_word_analysis(wp_words, target, lookup_db)

    # Check if there are any useful roles at all
    has_roles = any(wa['roles'] for wa in word_analysis)
    if not has_roles:
        return None

    # Build and send prompt
    prompt = build_prompt(clue, answer, definition, word_analysis)

    try:
        client = _get_client()
        response = client.messages.create(
            model=TIER2_MODEL,
            max_tokens=400,
            temperature=0,
            messages=[{'role': 'user', 'content': prompt}],
        )

        reply = response.content[0].text.strip()
        input_tokens = response.usage.input_tokens
        output_tokens = response.usage.output_tokens

    except Exception as e:
        print("   Tier2 API error: %s" % e)
        return None

    # Parse JSON response
    try:
        # Strip markdown code fences if present
        if reply.startswith('```'):
            reply = reply.split('\n', 1)[1].rsplit('```', 1)[0].strip()
        ai_output = json.loads(reply)
    except (json.JSONDecodeError, IndexError):
        return None

    if 'error' in ai_output:
        return None

    # Extract pieces
    pieces = ai_output.get('pieces', [])
    if not pieces:
        return None

    definition = ai_output.get('definition', '')
    wordplay_type = ai_output.get('wordplay_type', 'charade')

    # Build piece letters for assembly check
    sonnet_pieces = []
    for p in pieces:
        letters = p.get('letters', '')
        if letters:
            sonnet_pieces.append(letters.upper())

    # Quick assembly check: do pieces concatenate to answer?
    assembled = ''.join(sonnet_pieces)
    if assembled != target:
        # Try without link/indicator pieces
        non_link = [p.get('letters', '').upper() for p in pieces
                    if p.get('mechanism') not in ('indicator', 'link', '')]
        if ''.join(non_link) != target:
            return None

    # Build result dict matching solve_clue() output format
    result = {
        "ai_output": ai_output,
        "sonnet_pieces": sonnet_pieces,
        "sonnet_wtype": wordplay_type,
        "sonnet_def": definition,
        "tokens_in": input_tokens,
        "tokens_out": output_tokens,
        "tier": "Tier2",
        "fallback_method": None,
        "assembly": {
            "op": wordplay_type,
            "pieces": sonnet_pieces,
            "order": list(range(len(sonnet_pieces))),
        },
        "validation": {
            "confidence": "medium",
            "score": 65,
            "checks": {
                "definition": "yes" if definition else "no",
                "yields": "yes",
                "type": wordplay_type,
            },
        },
    }

    return result
