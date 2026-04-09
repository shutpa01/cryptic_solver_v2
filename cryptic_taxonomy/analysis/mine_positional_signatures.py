"""Mine POSITIONAL signatures from parsed Times explanations.

Unlike mine_span_signatures.py which only records fodder tokens,
this miner builds complete left-to-right token sequences covering
every word in the wordplay window:

  - Fodder tokens (SYN_F, ABR_F, ANA_F, etc.) at their mapped positions
  - Indicator tokens (ANA_I, REV_I, CON_I, etc.) identified via DB lookup
  - Link words (LNK) for connective words
  - Definition position (start or end of full clue)

Output signature format:
  ANA_I(1w)+ANA_F(3w) anagram def:start
  SYN_F(1w)+CON_I(1w)+SYN_F(1w) container def:end
"""

import sqlite3
import sys
import re
from collections import Counter
from typing import List, Optional, Tuple, Dict, Set

sys.path.insert(0, '.')
from cryptic_taxonomy.analysis.notation_parser import parse_explanation, Piece
from cryptic_taxonomy.analysis.improved_mapper import (
    improved_map_pieces_to_words, MappingDB, _clean
)
from cryptic_taxonomy.analysis.mine_span_signatures import (
    extract_wordplay_window, piece_to_token
)
from signature_solver.tokens import (
    LINK_WORDS, INDICATOR_TYPE_TO_TOKEN, INDICATOR_TOKENS
)


class IndicatorDB:
    """Load indicator words from cryptic_new.db for classification."""

    def __init__(self, db_path='data/cryptic_new.db'):
        conn = sqlite3.connect(db_path, timeout=30)
        # word → set of wordplay_types it indicates
        self.indicators = {}
        for word, wp_type in conn.execute(
            "SELECT word, wordplay_type FROM indicators"
            " WHERE wordplay_type != ''"
        ):
            w = word.lower().strip()
            self.indicators.setdefault(w, set()).add(wp_type)
        conn.close()

    def get_indicator_types(self, word):
        """Return set of wordplay_types this word can indicate."""
        w = _clean(word)
        result = set()
        if w in self.indicators:
            result.update(self.indicators[w])
        # Try without trailing s (plural)
        if len(w) >= 4 and w.endswith('s') and not w.endswith('ss'):
            ws = w[:-1]
            if ws in self.indicators:
                result.update(self.indicators[ws])
        # Try gerund → base (e.g. "composing" → "compose")
        if w.endswith('ing') and len(w) >= 6:
            base = w[:-3]
            if base in self.indicators:
                result.update(self.indicators[base])
            if base + 'e' in self.indicators:
                result.update(self.indicators[base + 'e'])
        return result

    def get_indicator_token(self, word, operation):
        """Get the indicator token for this word, if it matches the operation.

        Returns the token (e.g. ANA_I) or None.
        """
        types = self.get_indicator_types(word)
        if not types:
            return None

        # Map operation to expected indicator type(s)
        op_to_types = {
            'anagram': {'anagram'},
            'charade': set(),  # charades don't need indicators
            'container': {'container', 'insertion'},
            'reversal': {'reversal'},
            'deletion': {'deletion'},
            'hidden': {'hidden'},
            'homophone': {'homophone'},
            'acrostic': {'acrostic', 'selection', 'parts'},
            'alternating': {'alternating'},
            'container_charade': {'container', 'insertion'},
            'positional_charade': {'deletion', 'parts', 'selection', 'acrostic'},
            'trim_charade': {'deletion'},
        }

        expected = op_to_types.get(operation, set())
        matched = types & expected
        if matched:
            # Use the first matched type to get the token
            for t in matched:
                if t in INDICATOR_TYPE_TO_TOKEN:
                    return INDICATOR_TYPE_TO_TOKEN[t]
        return None


def classify_unmapped_words(wp_words, used_indices, operation, ind_db):
    """Classify words not covered by piece mappings.

    Returns dict: word_index → token (indicator token or LNK).
    """
    result = {}
    for i in range(len(wp_words)):
        if i in used_indices:
            continue

        word = wp_words[i]
        w_clean = _clean(word)

        # Check if it's an indicator for this operation
        ind_token = ind_db.get_indicator_token(word, operation)
        if ind_token:
            result[i] = ind_token
            continue

        # Check if it's a link word
        if w_clean in LINK_WORDS:
            result[i] = 'LNK'
            continue

        # Unknown — could be an indicator we don't have, or part of
        # a multi-word indicator. Mark as LNK for now.
        result[i] = 'LNK'

    return result


def build_positional_signature(mappings, wp_words, operation, ind_db):
    """Build a complete positional signature covering every word.

    Returns signature string like: ANA_I(1w)+ANA_F(3w)
    Also returns the token list for analysis.
    """
    n = len(wp_words)

    # Build word→token map from piece mappings
    # Each mapping is (piece, start_idx, span)
    word_tokens = {}  # word_index → (token, span)
    used_indices = set()

    for piece, start, span in mappings:
        token = piece_to_token(piece, operation)
        word_tokens[start] = (token, span)
        used_indices.update(range(start, start + span))

    # Classify unmapped words
    unmapped = classify_unmapped_words(wp_words, used_indices, operation, ind_db)

    # Merge consecutive same-type unmapped tokens (e.g. two LNK words → LNK(2w))
    # Build complete left-to-right sequence
    sequence = []  # list of (token, span)
    i = 0
    while i < n:
        if i in word_tokens:
            token, span = word_tokens[i]
            sequence.append((token, span))
            i += span
        elif i in unmapped:
            token = unmapped[i]
            # Merge consecutive same-type tokens
            span = 1
            while i + span < n and unmapped.get(i + span) == token:
                span += 1
            sequence.append((token, span))
            i += span
        else:
            # Shouldn't happen — word is neither mapped nor unmapped
            sequence.append(('???', 1))
            i += 1

    # Build signature string
    parts = []
    for token, span in sequence:
        if span == 1:
            parts.append(token)
        else:
            parts.append(f"{token}({span}w)")

    return "+".join(parts), sequence


def get_def_position(clue_text, definition):
    """Determine if definition is at start or end of clue."""
    clue = clue_text.strip().lower()
    defn = definition.strip().lower()
    if clue.startswith(defn):
        return 'start'
    elif clue.endswith(defn):
        return 'end'
    return None


def run():
    mapping_db = MappingDB()
    ind_db = IndicatorDB()

    conn = sqlite3.connect('data/times_explanations.db', timeout=30)
    rows = conn.execute('''
        SELECT clue_text, answer, definition, explanation
        FROM clues
        WHERE explanation IS NOT NULL AND explanation != ''
        AND definition IS NOT NULL AND definition != ''
        AND clue_text IS NOT NULL AND clue_text != ''
    ''').fetchall()
    conn.close()

    print(f"Total rows: {len(rows)}")

    sig_counts = Counter()
    op_counts = Counter()
    total_verified = 0
    total_mapped = 0
    no_wp = 0

    # Track signatures with examples for debugging
    sig_examples = {}

    for clue_text, answer, definition, expl in rows:
        answer_clean = re.sub(r'[\s\-]', '', answer.upper())
        result = parse_explanation(expl, answer_clean)

        if not result or not result.verified:
            continue
        total_verified += 1

        # Skip whole-clue types
        if result.operation in ('double_definition', 'cryptic_definition'):
            sig = result.operation
            sig_counts[sig] += 1
            op_counts[result.operation] += 1
            total_mapped += 1
            continue

        # Hidden words: the answer is literally in the clue, no wordplay decomposition
        if result.operation == 'hidden':
            sig = 'HID_I+HID_F'
            sig_counts[sig] += 1
            op_counts[result.operation] += 1
            total_mapped += 1
            continue

        wp_words = extract_wordplay_window(clue_text, definition)
        if not wp_words:
            no_wp += 1
            continue

        if not result.pieces:
            continue

        # Map pieces to word positions
        mappings = improved_map_pieces_to_words(result.pieces, wp_words, mapping_db)
        if mappings is None:
            continue

        # Build positional signature
        def_pos = get_def_position(clue_text, definition)
        sig_str, sequence = build_positional_signature(
            mappings, wp_words, result.operation, ind_db
        )

        full_sig = f"{sig_str} {result.operation}"
        if def_pos:
            full_sig += f" def:{def_pos}"

        sig_counts[full_sig] += 1
        op_counts[result.operation] += 1
        total_mapped += 1

        # Store first example
        if full_sig not in sig_examples:
            sig_examples[full_sig] = (clue_text, answer, definition, wp_words, sequence)

    print(f"Verified: {total_verified}")
    print(f"Positional signatures built: {total_mapped}")
    print(f"No wordplay window: {no_wp}")
    print()

    # Operation breakdown
    print("Operation counts:")
    for op, cnt in op_counts.most_common():
        print(f"  {op:30s} {cnt:6d}")
    print()

    # Show positional signatures
    print(f"{'Positional Signature':70s} {'Count':>6s} {'%':>6s} {'Cum%':>6s}")
    print('-' * 90)
    cum = 0
    total = sum(sig_counts.values())
    for sig, cnt in sig_counts.most_common(80):
        pct = 100 * cnt / total
        cum += pct
        print(f"{sig:70s} {cnt:6d} {pct:5.1f}% {cum:5.1f}%")

    print()
    print(f"Total unique signatures: {len(sig_counts)}")
    print(f"  count >= 50: {sum(1 for c in sig_counts.values() if c >= 50)}")
    print(f"  count >= 20: {sum(1 for c in sig_counts.values() if c >= 20)}")
    print(f"  count >= 10: {sum(1 for c in sig_counts.values() if c >= 10)}")
    print(f"  count >=  5: {sum(1 for c in sig_counts.values() if c >= 5)}")

    # Show examples for top signatures
    print()
    print("=== Examples for top 20 signatures ===")
    for sig, cnt in sig_counts.most_common(20):
        if sig in sig_examples:
            clue, ans, defn, wp, seq = sig_examples[sig]
            print(f"\n{sig} (n={cnt})")
            print(f"  Clue: {clue}")
            print(f"  Answer: {ans}, Def: {defn}")
            print(f"  WP words: {wp}")
            print(f"  Sequence: {seq}")


if __name__ == '__main__':
    run()
