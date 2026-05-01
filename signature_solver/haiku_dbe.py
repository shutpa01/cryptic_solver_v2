"""Haiku-powered category-mate / famous-bearer lookup for DBE-marked words.

When a clue word is flagged by a definition-by-example marker (maybe,
perhaps, say, for example, ...), it stands for an EXAMPLE rather than
itself. The wordplay piece is something related to the word — its
category (Garibaldi -> BISCUIT), a famous bearer's other name
(Hill -> DAMON, Wood -> NATALIE), or a type/instance.

If the existing synonyms_pairs table has the right candidate, the
solver finds it on its own. This module is the fallback when it
doesn't: ask Haiku, surface candidates that fit the answer, and queue
them in pending_enrichments for human review through the existing
enrichment workflow.

Mirrors signature_solver/haiku_definition.py — same structural pattern,
same try/except discipline, same queue table.
"""

import os

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


def _alpha_upper(s):
    return ''.join(c for c in s.upper() if c.isalpha())


def find_dbe_candidates(word, answer, clue_text):
    """Ask Haiku what a DBE-marked word could refer to.

    Args:
        word: the clue word that's adjacent to the DBE marker
        answer: the known answer (uppercase, no spaces/hyphens)
        clue_text: the full clue, for context

    Returns:
        list[str] of candidate substitutions, uppercase, alphabetic only.
        Each candidate is a substring of the answer or its reverse.
        Empty list on any failure (network, parse, no fits).
    """
    answer_clean = _alpha_upper(answer)
    answer_rev = answer_clean[::-1]

    try:
        client = _get_client()
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=200,
            temperature=0,
            messages=[{'role': 'user', 'content':
                'In a cryptic crossword clue, the word "%s" is preceded or '
                'followed by a definition-by-example marker (e.g. "maybe", '
                '"perhaps", "say"). This means the wordplay piece refers to '
                'something related to "%s" rather than "%s" itself.\n\n'
                'Suggest words that "%s" could substitute for, considering:\n'
                '- The category it belongs to (Garibaldi -> BISCUIT)\n'
                '- A famous bearer\'s other name (Hill -> DAMON, Wood -> NATALIE)\n'
                '- A type or instance of it (tulip -> FLOWER)\n\n'
                'Clue: %s\n'
                'Answer: %s\n\n'
                'Reply with candidate words only, one per line, uppercase, '
                'no punctuation, no explanation.'
                % (word, word, word, word, clue_text, answer)
            }],
        )
        text = response.content[0].text.strip()
    except Exception:
        return []

    candidates = []
    seen = set()
    for line in text.splitlines():
        cand = _alpha_upper(line)
        if not cand or cand in seen:
            continue
        if cand == answer_clean:
            # The full answer itself isn't a useful wordplay piece
            continue
        if cand in answer_clean or cand in answer_rev:
            candidates.append(cand)
            seen.add(cand)

    return candidates


def queue_dbe_enrichment(conn, word, candidate, answer, clue_text,
                          source, puzzle_number):
    """Queue a DBE-derived (word -> candidate) synonym pair for human review.

    Writes to the existing pending_enrichments table with type='synonym',
    matching the convention used by other synonym suggestions. Skips if
    the pair is already present in synonyms_pairs, pending_enrichments,
    or rejected_enrichments.

    Args:
        conn: sqlite3 connection to clues_master.db
        word: the DBE-marked clue word (lowercase)
        candidate: the suggested substitution (uppercase letters)
        answer: the answer this contributed to
        clue_text: the original clue
        source: source name (e.g. 'telegraph')
        puzzle_number: puzzle id
    """
    word_lower = word.lower().strip()
    cand_upper = _alpha_upper(candidate)
    if not word_lower or not cand_upper:
        return

    # Check the synonym isn't already known in the reference DB.
    import sqlite3
    ref_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'cryptic_new.db'
    )
    try:
        ref = sqlite3.connect(ref_path, timeout=10)
        exists = ref.execute(
            "SELECT 1 FROM synonyms_pairs "
            "WHERE LOWER(word)=? AND UPPER(synonym)=?",
            (word_lower, cand_upper)
        ).fetchone()
        ref.close()
        if exists:
            return
    except Exception:
        pass

    # Skip if already pending or previously rejected.
    try:
        already_pending = conn.execute(
            "SELECT 1 FROM pending_enrichments "
            "WHERE type=? AND LOWER(word)=? AND UPPER(letters)=?",
            ('synonym', word_lower, cand_upper)
        ).fetchone()
        if already_pending:
            return

        already_rejected = conn.execute(
            "SELECT 1 FROM rejected_enrichments "
            "WHERE type=? AND LOWER(word)=? AND UPPER(letters)=?",
            ('synonym', word_lower, cand_upper)
        ).fetchone()
        if already_rejected:
            return

        conn.execute(
            "INSERT INTO pending_enrichments "
            "(type, word, letters, answer, clue_text, source, puzzle_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ('synonym', word_lower, cand_upper, answer, clue_text,
             source, puzzle_number)
        )
        conn.commit()
    except Exception:
        pass
