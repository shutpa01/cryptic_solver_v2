"""Step 1: Build labeled dataset of word roles from TFTT explanations.

For each clue where the improved mapper succeeds, outputs one row per
wordplay-window word with its assigned role.

Output: data/word_roles.db — SQLite table 'word_roles' with columns:
  clue_id, puzzle_number, clue_text, answer, definition, operation,
  word_position, word_text, window_length, answer_length, assigned_role,
  piece_letters (if fodder, what letters it contributed)
"""

import sqlite3
import sys
import re
import time
from collections import Counter

sys.path.insert(0, '.')

from cryptic_taxonomy.analysis.notation_parser import parse_explanation
from cryptic_taxonomy.analysis.improved_mapper import (
    improved_map_pieces_to_words, MappingDB,
)


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")
from cryptic_taxonomy.analysis.mine_span_signatures import (
    extract_wordplay_window, piece_to_token
)
from cryptic_taxonomy.analysis.mine_positional_signatures import IndicatorDB
from signature_solver.tokens import (
    LINK_WORDS, INDICATOR_TYPE_TO_TOKEN, INDICATOR_TOKENS
)


# Complete mapping of operation to expected indicator DB types.
# The version in mine_positional_signatures.py is missing several compound ops.
_OP_TO_EXPECTED_IND = {
    'anagram': {'anagram'},
    'anagram_charade': {'anagram'},
    'charade': set(),
    'container': {'container', 'insertion'},
    'container_charade': {'container', 'insertion'},
    'reversal': {'reversal'},
    'reversal_charade': {'reversal'},
    'reversal_container': {'reversal', 'container', 'insertion'},
    'del': {'deletion'},
    'deletion': {'deletion'},
    'hidden': {'hidden'},
    'homophone': {'homophone'},
    'acrostic': {'acrostic', 'selection', 'parts'},
    'alternating': {'alternating'},
    'positional_charade': {'deletion', 'parts', 'selection', 'acrostic'},
    'trim_charade': {'deletion'},
    'synonym': set(),
    'abr': set(),
}


def _classify_unmapped_words(wp_words, used_indices, operation, ind_db):
    """Classify words not covered by piece mappings.

    Returns dict: word_index -> token (indicator token or LNK).
    """
    expected = _OP_TO_EXPECTED_IND.get(operation, set())
    result = {}

    for i in range(len(wp_words)):
        if i in used_indices:
            continue

        word = wp_words[i]
        w_clean = _clean(word)

        # Check if it's an indicator for this operation
        if expected:
            ind_types = ind_db.get_indicator_types(word)
            matched = ind_types & expected
            if matched:
                for t in matched:
                    token = INDICATOR_TYPE_TO_TOKEN.get(t)
                    if token:
                        result[i] = token
                        break
                if i in result:
                    continue

        # Check if it's a link word
        if w_clean in LINK_WORDS:
            result[i] = 'LNK'
            continue

        # Unknown — default to LNK
        result[i] = 'LNK'

    return result


# Map operation to expected indicator types for span relabeling
_OP_TO_IND_TYPES = {
    'container': {'container', 'insertion'},
    'container_charade': {'container', 'insertion'},
    'reversal': {'reversal'},
    'reversal_charade': {'reversal'},
    'reversal_container': {'reversal', 'container', 'insertion'},
    'anagram': {'anagram'},
    'anagram_charade': {'anagram'},
    'del': {'deletion'},
}

# Map indicator DB types to our tokens
_IND_TYPE_TO_TOKEN = {
    'container': 'CON_I',
    'insertion': 'CON_I',
    'reversal': 'REV_I',
    'anagram': 'ANA_I',
    'deletion': 'DEL_I',
    'hidden': 'HID_I',
    'homophone': 'HOM_I',
}


def _indicator_relabel(mappings, wp_words, word_roles, operation, ind_db):
    """Find indicators hiding inside mapped fodder spans and relabel them.

    When the notation parser returns a container as a single CON piece,
    the mapper assigns all words in the span to SYN_F. But one of those
    words is actually the indicator (e.g. "breaking" in a container).

    Also handles: reversal indicators inside reversal spans, anagram
    indicators inside anagram spans, etc.
    """
    expected_types = _OP_TO_IND_TYPES.get(operation)
    if not expected_types:
        return

    for piece, start, span in mappings:
        if span < 2:
            continue

        # Check each word in this span for indicator role
        for i in range(start, start + span):
            word = wp_words[i]
            ind_types = ind_db.get_indicator_types(word)
            if not ind_types:
                continue

            # Does this word indicate the right type for this operation?
            matched = ind_types & expected_types
            if matched:
                # Pick the first matched type and convert to token
                for t in matched:
                    token = _IND_TYPE_TO_TOKEN.get(t)
                    if token:
                        word_roles[i] = (token, None)
                        break
                break  # Only relabel one indicator per span


def build_dataset():
    print("Loading mapping DB...")
    mapping_db = MappingDB()
    print("Loading indicator DB...")
    ind_db = IndicatorDB()

    print("Loading TFTT explanations...")
    conn = sqlite3.connect('data/times_explanations.db', timeout=30)
    rows = conn.execute('''
        SELECT rowid, puzzle_number, clue_text, answer, definition, explanation
        FROM clues
        WHERE explanation IS NOT NULL AND explanation != ''
        AND definition IS NOT NULL AND definition != ''
        AND clue_text IS NOT NULL AND clue_text != ''
        AND answer IS NOT NULL AND answer != ''
    ''').fetchall()
    conn.close()
    print(f"  {len(rows)} rows loaded")

    # Create output DB
    out_conn = sqlite3.connect('data/word_roles.db')
    out_conn.execute('DROP TABLE IF EXISTS word_roles')
    out_conn.execute('''
        CREATE TABLE word_roles (
            clue_id INTEGER,
            puzzle_number INTEGER,
            clue_text TEXT,
            answer TEXT,
            definition TEXT,
            operation TEXT,
            word_position INTEGER,
            word_text TEXT,
            window_length INTEGER,
            answer_length INTEGER,
            assigned_role TEXT,
            piece_letters TEXT
        )
    ''')
    out_conn.execute('''
        CREATE TABLE IF NOT EXISTS build_stats (
            key TEXT PRIMARY KEY,
            value INTEGER
        )
    ''')

    stats = Counter()
    t0 = time.time()
    rows_written = 0
    batch = []

    for idx, (rowid, puzzle_number, clue_text, answer, definition, explanation) in enumerate(rows):
        if idx % 5000 == 0 and idx > 0:
            elapsed = time.time() - t0
            rate = idx / elapsed
            print(f"  {idx}/{len(rows)} ({rate:.0f}/s) — {rows_written} word rows written")

        answer_clean = re.sub(r'[\s\-]', '', answer.upper())
        answer_length = len(answer_clean)

        # Parse the explanation
        result = parse_explanation(explanation, answer_clean)
        if not result or not result.verified:
            stats['parse_unverified'] += 1
            continue
        stats['parse_verified'] += 1

        operation = result.operation

        # Skip whole-clue types — no wordplay window decomposition
        if operation in ('double_definition', 'cryptic_definition'):
            stats[f'skip_{operation}'] += 1
            continue

        # Extract wordplay window
        wp_words = extract_wordplay_window(clue_text, definition)
        if not wp_words:
            stats['no_wp_window'] += 1
            continue

        window_length = len(wp_words)

        # Hidden: special case — all wp words are HID_F
        if operation == 'hidden':
            for pos, word in enumerate(wp_words):
                batch.append((
                    rowid, puzzle_number, clue_text, answer, definition,
                    operation, pos, word, window_length, answer_length,
                    'HID_F', None
                ))
                rows_written += 1
            stats['mapped_hidden'] += 1
            continue

        # Homophone: special case — need indicator + fodder but mapper
        # doesn't handle well, skip for now
        if operation == 'homophone':
            stats['skip_homophone'] += 1
            continue

        if not result.pieces:
            stats['no_pieces'] += 1
            continue

        # Map pieces to clue words
        mappings = improved_map_pieces_to_words(result.pieces, wp_words, mapping_db)
        if mappings is None:
            stats['mapping_failed'] += 1
            continue

        # Build word-level role assignments
        # Start with fodder pieces from mapper
        word_roles = {}  # position -> (role, piece_letters)
        used_indices = set()

        for piece, start, span in mappings:
            token = piece_to_token(piece, operation)
            for i in range(start, start + span):
                if i == start:
                    word_roles[i] = (token, piece.letters)
                else:
                    # Multi-word span: mark continuation words with same token
                    word_roles[i] = (token, piece.letters if i == start else None)
                used_indices.add(i)

        # Post-process: find indicators hiding inside mapped spans.
        # Container pieces (CON) often span 3+ words where one is the
        # indicator (e.g. "detective breaking agreements" — "breaking" is
        # CON_I, not SYN_F). Same for reversals inside mapped spans.
        #
        # Strategy: for each mapped span of 2+ words, check if any word
        # is a known indicator for this operation. If so, relabel it.
        _indicator_relabel(mappings, wp_words, word_roles, operation, ind_db)

        # Classify remaining words (indicators, LNK)
        unmapped = _classify_unmapped_words(wp_words, used_indices, operation, ind_db)
        for i, token in unmapped.items():
            word_roles[i] = (token, None)

        # Write one row per word
        for pos in range(window_length):
            if pos in word_roles:
                role, letters = word_roles[pos]
            else:
                role = 'UNKNOWN'
                letters = None

            batch.append((
                rowid, puzzle_number, clue_text, answer, definition,
                operation, pos, wp_words[pos], window_length, answer_length,
                role, letters
            ))
            rows_written += 1

        stats['mapped_ok'] += 1

        # Batch insert every 1000 clues
        if len(batch) >= 10000:
            out_conn.executemany(
                'INSERT INTO word_roles VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
                batch
            )
            out_conn.commit()
            batch = []

    # Final batch
    if batch:
        out_conn.executemany(
            'INSERT INTO word_roles VALUES (?,?,?,?,?,?,?,?,?,?,?,?)',
            batch
        )
        out_conn.commit()

    # Write stats
    for key, val in stats.items():
        out_conn.execute(
            'INSERT OR REPLACE INTO build_stats VALUES (?, ?)',
            (key, val)
        )
    out_conn.commit()

    # Create indexes for analysis
    print("Creating indexes...")
    out_conn.execute('CREATE INDEX idx_role ON word_roles(assigned_role)')
    out_conn.execute('CREATE INDEX idx_operation ON word_roles(operation)')
    out_conn.execute('CREATE INDEX idx_word ON word_roles(word_text)')
    out_conn.commit()
    out_conn.close()

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")
    print(f"Word rows written: {rows_written}")
    print(f"\nStats:")
    for key, val in sorted(stats.items(), key=lambda x: -x[1]):
        print(f"  {key:30s} {val:>7d}")


if __name__ == '__main__':
    build_dataset()
