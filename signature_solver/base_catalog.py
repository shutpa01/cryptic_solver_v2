"""Base pattern catalog — collapsed from positional catalog.

Instead of 694 specific span variants (e.g. ABR_F(1w)+SYN_F(2w) charade),
this has ~68 base patterns like F+F charade.  The matcher determines
ABR vs SYN at runtime via DB lookups, and tries span assignments
exhaustively (each F slot can be 1w, 2w, 3w, etc.).

Source: data/positional_catalog.json -> collapsed by replacing all
fodder tokens with F and all indicator tokens with I.
"""

import json
import os


class BaseEntry:
    """One base pattern entry.

    pattern: tuple of 'I' or 'F' tokens in left-to-right order.
    operation: the wordplay operation type.
    count: total frequency across all span/def_pos variants.
    tier: 1 (count>=200), 2 (>=20), 3 (>=5), 4 (<5).
    """

    __slots__ = ('pattern', 'operation', 'count', 'tier',
                 'n_fodder', 'n_indicator', 'label')

    def __init__(self, pattern, operation, count):
        self.pattern = tuple(pattern)
        self.operation = operation
        self.count = count
        self.tier = 1 if count >= 200 else 2 if count >= 20 else 3 if count >= 5 else 4

        self.n_fodder = sum(1 for t in self.pattern if t == 'F')
        self.n_indicator = sum(1 for t in self.pattern if t == 'I')
        self.label = '+'.join(self.pattern) + f' {operation}'

    @property
    def min_words(self):
        """Minimum wordplay-window words needed (1 per token slot)."""
        return len(self.pattern)

    def __repr__(self):
        return f'<Base {self.label} T{self.tier} n={self.count}>'


# --- Operation -> required indicator type mapping ---
# The operation tells us what kind of indicator to look for in I slots.
# Value is a single string or a list of strings (any one must be present).
OPERATION_INDICATOR_TYPE = {
    'anagram': 'ANA_I',
    'anagram_charade': 'ANA_I',
    'anagram_plus': 'ANA_I',
    'anagram_container': 'ANA_I',
    'container': 'CON_I',
    'container_charade': 'CON_I',
    'container_positional': 'CON_I',
    'reversal': 'REV_I',
    'reversal_charade': 'REV_I',
    'hidden': 'HID_I',
    'hidden_reversed': 'HID_I',
    'homophone': 'HOM_I',
    'deletion': 'DEL_I',
    'trim': 'DEL_I',
    'trim_charade': 'DEL_I',
    'alternate': 'POS_I_ALTERNATE',
    'acrostic': 'POS_I_FIRST',
    'positional_charade': [
        'POS_I_FIRST', 'POS_I_LAST', 'POS_I_OUTER', 'POS_I_MIDDLE',
        'POS_I_TRIM_FIRST', 'POS_I_TRIM_LAST', 'POS_I_TRIM_OUTER',
        'POS_I_TRIM_MIDDLE', 'POS_I_ALTERNATE', 'POS_I_HALF',
    ],
}

# --- Fodder token types to try per operation ---
# For F slots, the matcher tries these token types via _lookup_slot.
OPERATION_FODDER_TYPES = {
    'charade': ['SYN_F', 'ABR_F'],
    'reversal': ['SYN_F', 'ABR_F'],
    'reversal_charade': ['SYN_F', 'ABR_F'],
    'container': ['SYN_F', 'ABR_F'],
    'container_charade': ['SYN_F', 'ABR_F'],
    'anagram': ['ANA_F'],
    'anagram_charade': ['ANA_F', 'SYN_F', 'ABR_F'],
    'anagram_plus': ['ANA_F', 'ABR_F', 'SYN_F'],
    'anagram_container': ['ANA_F', 'SYN_F', 'ABR_F'],
    'hidden': ['HID_F'],
    'hidden_reversed': ['HID_F'],
    'homophone': ['HOM_F'],
    'deletion': ['SYN_F', 'ABR_F'],
    'trim': ['SYN_F'],
    'trim_charade': ['SYN_F', 'ABR_F'],
    'synonym': ['SYN_F'],
    'alternate': ['POS_F'],
    'acrostic': ['POS_F'],
    'positional_charade': ['POS_F', 'SYN_F', 'ABR_F'],
    'container_positional': ['POS_F', 'SYN_F', 'ABR_F'],
}


def _load_catalog():
    """Load base catalog entries from JSON data file."""
    json_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'base_catalog.json'
    )
    with open(json_path) as f:
        raw = json.load(f)

    entries = []
    for item in raw:
        entries.append(BaseEntry(
            pattern=item['pattern'],
            operation=item['operation'],
            count=item['count'],
        ))
    return entries


BASE_CATALOG = _load_catalog()

# Pre-sorted by tier then count for search priority
BASE_CATALOG.sort(key=lambda e: (e.tier, -e.count))

# Index by operation for fast lookup
CATALOG_BY_OPERATION = {}
for _entry in BASE_CATALOG:
    CATALOG_BY_OPERATION.setdefault(_entry.operation, []).append(_entry)
