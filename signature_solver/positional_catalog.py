"""Positional signature catalog — generated from mined Times explanation data.

Each entry encodes a COMPLETE left-to-right token sequence including
indicators at their positions, the operation, definition position,
and word spans. LNK gaps between tokens are implicit (allowed anywhere).

Source: 72k Times explanations -> 29k verified -> 16k positional-mapped.
Entries with count >= 5 included (223 entries, 95.3% coverage).
"""

import json
import os
from .tokens import *


INDICATOR_TOKENS_SET = {
    ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I,
    POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
    POS_I_ALTERNATE, POS_I_TRIM_FIRST, POS_I_TRIM_LAST,
    POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER, POS_I_HALF,
}

FODDER_TOKENS_SET = {SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, POS_F, DEL_F}


class PositionalEntry:
    """One positional signature pattern.

    sequence: list of (token, span) tuples in left-to-right order.
              Includes indicators at their correct positions.
              LNK tokens are stripped — gaps allowed implicitly.
    operation: the wordplay operation type.
    def_pos: "start" or "end" (where definition sits in clue), or None.
    count: frequency in training data (higher = more common pattern).
    tier: 1 (count>=200), 2 (>=20), 3 (>=5) — controls search priority.
    """

    __slots__ = ('sequence', 'operation', 'def_pos', 'count', 'tier',
                 'tokens', 'spans', 'indicators', 'fodder_tokens',
                 'total_fodder_words', 'label')

    def __init__(self, sequence, operation, def_pos, count):
        self.sequence = tuple((tok, span) for tok, span in sequence)
        self.operation = operation
        self.def_pos = def_pos
        self.count = count
        self.tier = 1 if count >= 200 else 2 if count >= 20 else 3

        # Derived properties for fast filtering
        self.tokens = tuple(tok for tok, span in self.sequence)
        self.spans = tuple(span for tok, span in self.sequence)
        self.indicators = frozenset(
            tok for tok, span in self.sequence if tok in INDICATOR_TOKENS_SET
        )
        self.fodder_tokens = tuple(
            (tok, span) for tok, span in self.sequence
            if tok in FODDER_TOKENS_SET
        )
        self.total_fodder_words = sum(
            span for tok, span in self.sequence if tok in FODDER_TOKENS_SET
        )
        self.label = '+'.join(
            tok if span == 1 else f'{tok}({span}w)'
            for tok, span in self.sequence
        ) + f' {operation}'

    @property
    def min_words(self):
        """Minimum wordplay-window words needed (fodder + indicators)."""
        return sum(self.spans)

    def __repr__(self):
        dp = f' def:{self.def_pos}' if self.def_pos else ''
        return f'<Pos {self.label}{dp} T{self.tier} n={self.count}>'


def _load_catalog():
    """Load catalog entries from JSON data file."""
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'positional_catalog.json'
    )
    with open(json_path) as f:
        raw = json.load(f)

    entries = []
    for item in raw:
        seq = item.get('sequence', [])
        if not seq:
            # Whole-clue types (DD, CD) — no sequence
            continue
        entries.append(PositionalEntry(
            sequence=[(tok, span) for tok, span in seq],
            operation=item['operation'],
            def_pos=item.get('def_pos'),
            count=item['count'],
        ))
    return entries


POSITIONAL_CATALOG = _load_catalog()

# Pre-sorted by tier then count for search priority
POSITIONAL_CATALOG.sort(key=lambda e: (e.tier, -e.count))

# Index by operation for fast lookup
CATALOG_BY_OPERATION = {}
for entry in POSITIONAL_CATALOG:
    CATALOG_BY_OPERATION.setdefault(entry.operation, []).append(entry)
