"""Haiku-powered definition extraction fallback.

When extract_definition_candidates finds nothing in the RefDB,
ask Haiku (near-free) to identify which words at the start or end
of the clue are synonymous with the answer.

Also queues the discovered pair for enrichment review.
"""

import os
from functools import lru_cache

from dotenv import load_dotenv
load_dotenv()

_CLIENT = None


def _get_client():
    global _CLIENT
    if _CLIENT is None:
        from anthropic import Anthropic
        _CLIENT = Anthropic()
    return _CLIENT


HAIKU_MODEL = "claude-haiku-4-5-20251001"


def find_definition(clue_text, answer):
    """Ask Haiku which words at the start or end of the clue define the answer.

    Returns (definition_phrase, remaining_words) or None.
    """
    try:
        client = _get_client()
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=50,
            temperature=0,
            messages=[{'role': 'user', 'content':
                'In this cryptic crossword clue, which word or short phrase '
                'at the START or END of the clue is a synonym or definition '
                'of the answer %s? Just reply with the word(s), nothing else.'
                '\n\nClue: %s' % (answer, clue_text)
            }],
        )
        definition = response.content[0].text.strip().strip('"\'')

        if not definition:
            return None

        # Validate: definition must be near start or end of clue.
        # Allow small link words (as, a, the, etc.) before/after.
        clue_lower = clue_text.lower().strip()
        clue_words = clue_text.strip().split()
        def_lower = definition.lower().strip()
        def_words = def_lower.split()
        n = len(clue_words)
        nd = len(def_words)

        # Try matching at start (first 1-4 words)
        wp_start = None
        for skip in range(min(3, n)):
            candidate = ' '.join(w.lower().strip('.,;:!?"\'()-') for w in clue_words[skip:skip + nd])
            if candidate == def_lower:
                wp_start = skip + nd
                break

        if wp_start is not None:
            remaining = ' '.join(clue_words[wp_start:])
            remaining = remaining.strip().strip('.,;:!? ')
            wp_words = remaining.split() if remaining else None
            if wp_words:
                return definition, wp_words

        # Try matching at end (last 1-4 words)
        for skip in range(min(3, n)):
            end_pos = n - skip
            start_pos = end_pos - nd
            if start_pos < 0:
                break
            candidate = ' '.join(w.lower().strip('.,;:!?"\'()-') for w in clue_words[start_pos:end_pos])
            if candidate == def_lower:
                remaining = ' '.join(clue_words[:start_pos])
                remaining = remaining.strip().strip('.,;:!? ')
                wp_words = remaining.split() if remaining else None
                if wp_words:
                    return definition, wp_words

        return None

    except Exception:
        return None


def queue_enrichment(conn, definition, answer, clue_text, source, puzzle_number):
    """Queue the definition->answer pair for enrichment review.

    Checks it's not already in the DB or pending/rejected queues.
    """
    defn_lower = definition.lower().strip()
    answer_upper = answer.upper().strip()

    # Check if already in definition_answers_augmented
    import sqlite3
    ref_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'cryptic_new.db'
    )
    try:
        ref = sqlite3.connect(ref_path, timeout=10)
        exists = ref.execute(
            "SELECT 1 FROM definition_answers_augmented "
            "WHERE LOWER(definition)=? AND UPPER(answer)=?",
            (defn_lower, answer_upper)
        ).fetchone()
        ref.close()
        if exists:
            return
    except Exception:
        pass

    # Check not already pending or rejected
    try:
        already_pending = conn.execute(
            "SELECT 1 FROM pending_enrichments WHERE LOWER(word)=? AND UPPER(letters)=?",
            (defn_lower, answer_upper)
        ).fetchone()
        if already_pending:
            return

        already_rejected = conn.execute(
            "SELECT 1 FROM rejected_enrichments WHERE type=? AND LOWER(word)=? AND UPPER(letters)=?",
            ('definition', defn_lower, answer_upper)
        ).fetchone()
        if already_rejected:
            return

        conn.execute(
            "INSERT INTO pending_enrichments "
            "(type, word, letters, answer, clue_text, source, puzzle_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ('definition', defn_lower, answer_upper, answer, clue_text,
             source, puzzle_number)
        )
        conn.commit()
    except Exception:
        pass
