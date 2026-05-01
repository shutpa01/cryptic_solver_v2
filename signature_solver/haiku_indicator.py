"""Haiku-powered fallback for missing indicator detection.

When rule-based detection (indicator_detect.py) doesn't yield a working
indicator suggestion (or its top candidate doesn't enable a solve), ask
Haiku which clue word might be an unrecognised cryptic indicator.

Mirrors the haiku_dbe.py / haiku_definition.py pattern.
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

# The set of indicator types we ask Haiku to consider. Map from the
# Haiku-facing label to (wordplay_type, subtype). Subtype-aware so the
# returned suggestions can carry through the queue and dashboard.
_HAIKU_LABEL_TO_TYPE = {
    'first letter use':  ('parts', 'first_use'),
    'last letter use':   ('parts', 'last_use'),
    'outer letters use': ('parts', 'outer_use'),
    'middle letters use':('parts', 'center_use'),
    'first letter delete':('parts', 'first_delete'),
    'last letter delete':('parts', 'last_delete'),
    'outer letters delete':('parts', 'outer_delete'),
    'alternate letters': ('parts', 'alternate'),
    'acrostic':          ('acrostic', None),
    'reversal':          ('reversal', None),
    'anagram':           ('anagram', None),
    'container':         ('container', None),
    'deletion':          ('deletion', None),
    'homophone':         ('homophone', None),
    'hidden':            ('hidden', None),
}


def find_indicator_candidate(clue_text, answer, fodder_word, gap_letters):
    """Ask Haiku which clue word might be an unrecognised indicator and what type.

    Args:
        clue_text: full clue
        answer: the known answer
        fodder_word: the clue word that yields the missing letters
        gap_letters: the letters needed (e.g. 'IC')

    Returns:
        list of dicts with keys:
            indicator_word: str
            indicator_type: str ('parts', 'reversal', etc.)
            subtype: str or None
        Empty list on any failure or no useful suggestion.
    """
    try:
        client = _get_client()
        labels = ', '.join(sorted(_HAIKU_LABEL_TO_TYPE.keys()))
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=300,
            temperature=0,
            messages=[{'role': 'user', 'content':
                'In a cryptic crossword clue, the wordplay needs to extract '
                'the letters "%s" from the word "%s". An INDICATOR word '
                'somewhere ELSE in the clue licenses this extraction, but it '
                'is not yet known to our database.\n\n'
                'Identify the indicator. Important rules:\n'
                '- The indicator is NOT "%s" itself (that is the fodder).\n'
                '- The indicator is some OTHER word in the clue.\n'
                '- Choose its type from this list: %s.\n\n'
                'Clue: %s\n'
                'Answer: %s\n\n'
                'Reply with up to 3 lines, each formatted exactly as:\n'
                'WORD | type\n'
                'No explanation. Use lowercase for both. Example: '
                '"exhausted | outer letters use".'
                % (gap_letters, fodder_word, fodder_word, labels,
                   clue_text, answer)
            }],
        )
        text = response.content[0].text.strip()
    except Exception:
        return []

    suggestions = []
    seen = set()
    clue_words_lower = [w.lower().strip(",.;:!?\"'()") for w in clue_text.split()]
    fodder_lower = fodder_word.lower().strip(",.;:!?\"'()")

    for line in text.splitlines():
        if '|' not in line:
            continue
        parts = line.split('|', 1)
        word = parts[0].strip().strip('.,;:!?"\'()').lower()
        type_label = parts[1].strip().lower()
        if not word or not type_label:
            continue
        if (word, type_label) in seen:
            continue
        # Reject the fodder word itself — that's not a valid indicator.
        if word == fodder_lower:
            continue
        # Validate: word must appear in the clue
        if word not in clue_words_lower:
            continue
        # Validate: type must be in our known label set
        if type_label not in _HAIKU_LABEL_TO_TYPE:
            continue
        wp_type, subtype = _HAIKU_LABEL_TO_TYPE[type_label]
        suggestions.append({
            'indicator_word': word,
            'indicator_type': wp_type,
            'subtype': subtype,
        })
        seen.add((word, type_label))

    return suggestions


_SUBTYPE_LABELS = {
    'first_use':    'use the first letter(s) of the fodder',
    'last_use':     'use the last letter(s) of the fodder',
    'outer_use':    'use the first AND last letter (drop the middle)',
    'center_use':   'use the middle letter(s) of the fodder',
    'inner_use':    'use the middle letter(s) of the fodder',
    'first_delete': 'remove the first letter from the fodder',
    'last_delete':  'remove the last letter from the fodder',
    'tail_delete':  'remove the last letter from the fodder',
    'outer_delete': 'remove the first AND last letters (keep the middle)',
    'center_delete':'remove the middle letter(s) from the fodder',
    'alternate':    'take alternating letters from the fodder',
}


def disambiguate_subtype(indicator_word, candidate_subtypes, clue_text):
    """When rule-based detection gives multiple candidate subtypes that all
    produce the same letters from the same fodder (e.g. 'last_use' and
    'outer_use' both yielding 'IC' from 'IDEALISTIC'), ask Haiku which
    subtype is semantically right for the given indicator word.

    Args:
        indicator_word: the candidate indicator (e.g. "exhausted")
        candidate_subtypes: list of subtype strings to choose between
        clue_text: full clue, for context

    Returns:
        the chosen subtype (a member of candidate_subtypes) or None on
        failure / no clear pick.
    """
    if not candidate_subtypes:
        return None
    if len(candidate_subtypes) == 1:
        return candidate_subtypes[0]
    try:
        client = _get_client()
        options = '\n'.join(
            '  - %s : %s' % (st, _SUBTYPE_LABELS.get(st, st))
            for st in candidate_subtypes
        )
        response = client.messages.create(
            model=HAIKU_MODEL,
            max_tokens=20,
            temperature=0,
            messages=[{'role': 'user', 'content':
                'In cryptic crosswords, indicator words follow conventions:\n'
                '  - "exhausted", "drained", "spent", "hollow", "empty" '
                '-> the middle is gone, OUTER letters survive (outer_use)\n'
                '  - "heart", "centre", "core", "middle" '
                '-> the inner letters (center_use)\n'
                '  - "last", "end", "ending", "tail", "final", "back" '
                '-> the last letter(s) (last_use)\n'
                '  - "first", "initial", "head", "start", "leading" '
                '-> the first letter(s) (first_use)\n'
                '  - "headless", "topless" -> remove first (first_delete)\n'
                '  - "tailless", "endless" -> remove last (last_delete)\n\n'
                'For the clue below, the word "%s" is a positional indicator. '
                'Which option below best matches its CRYPTIC meaning?\n\n'
                '%s\n\n'
                'Clue: %s\n\n'
                'Reply with one option name only (no explanation, lowercase). '
                'Example: outer_use'
                % (indicator_word, options, clue_text)
            }],
        )
        pick = response.content[0].text.strip().lower()
    except Exception:
        return None
    if pick in candidate_subtypes:
        return pick
    # Haiku may have spelled it slightly differently — try to match.
    for st in candidate_subtypes:
        if st.lower() == pick or pick in st.lower():
            return st
    return None


def queue_indicator_enrichment(conn, indicator_word, indicator_type, subtype,
                                answer, clue_text, source, puzzle_number):
    """Queue an indicator suggestion in pending_enrichments.

    The 'letters' field uses the format established in step A:
      - bare TYPE (uppercase) for non-positional types: 'ANAGRAM', 'CONTAINER', etc.
      - TYPE:subtype for positional/parts types: 'PARTS:last_use', 'ACROSTIC:initial'

    Skips if a matching row already exists in pending or rejected.
    """
    word_lower = indicator_word.lower().strip()
    if not word_lower or not indicator_type:
        return

    type_field = indicator_type.upper()
    if subtype:
        letters = '%s:%s' % (type_field, subtype)
    else:
        letters = type_field

    # Check the indicator isn't already known in the reference DB.
    import sqlite3
    ref_path = os.path.join(
        os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
        'data', 'cryptic_new.db'
    )
    try:
        ref = sqlite3.connect(ref_path, timeout=10)
        if subtype:
            exists = ref.execute(
                "SELECT 1 FROM indicators "
                "WHERE LOWER(word)=? AND wordplay_type=? AND subtype=?",
                (word_lower, indicator_type, subtype)
            ).fetchone()
        else:
            exists = ref.execute(
                "SELECT 1 FROM indicators "
                "WHERE LOWER(word)=? AND wordplay_type=?",
                (word_lower, indicator_type)
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
            "WHERE type=? AND LOWER(word)=? AND letters=?",
            ('indicator', word_lower, letters)
        ).fetchone()
        if already_pending:
            return

        already_rejected = conn.execute(
            "SELECT 1 FROM rejected_enrichments "
            "WHERE type=? AND LOWER(word)=? AND letters=?",
            ('indicator', word_lower, letters)
        ).fetchone()
        if already_rejected:
            return

        conn.execute(
            "INSERT INTO pending_enrichments "
            "(type, word, letters, answer, clue_text, source, puzzle_number) "
            "VALUES (?, ?, ?, ?, ?, ?, ?)",
            ('indicator', word_lower, letters, answer, clue_text,
             source, puzzle_number)
        )
        conn.commit()
    except Exception:
        pass
