"""Signature catalog — generated from mined Times explanation data.

Each entry encodes a signature: the fodder token sequence, required indicators,
the operation to execute, word spans, and tier (based on frequency in data).

Source: 72k Times explanations → 29k verified parses → 8k span-mapped signatures.
"""

from .tokens import *


class CatalogEntry:
    """One signature pattern.

    word_spans: tuple of ints, one per token, giving the number of contiguous
    clue words each token slot consumes.  E.g. word_spans=(1,2) means the
    first token maps to 1 word and the second to a 2-word phrase.  The sum
    of all word spans equals the total wordplay words consumed by fodder.
    Every entry MUST have word_spans — the matcher uses deterministic
    span-guided placement exclusively.
    """

    __slots__ = ('tokens', 'operation', 'tier', 'indicators',
                 'allow_extra_lnk', 'min_fodder_words', 'label',
                 'word_spans')

    def __init__(self, tokens, operation, tier, indicators=None,
                 allow_extra_lnk=True, min_fodder_words=None, label=None,
                 word_spans=None):
        self.tokens = tuple(tokens)          # fodder token sequence
        self.operation = operation            # executor to use
        self.tier = tier                      # 1-4, controls search order
        self.indicators = frozenset(indicators or set())
        self.allow_extra_lnk = allow_extra_lnk
        self.word_spans = tuple(word_spans) if word_spans else None

        # Minimum number of clue words needed for fodder slots.
        if min_fodder_words is not None:
            self.min_fodder_words = min_fodder_words
        elif self.word_spans:
            self.min_fodder_words = sum(self.word_spans)
        else:
            # ANA_F slots can absorb multiple words
            n_ana = sum(1 for t in self.tokens if t == ANA_F)
            n_other = len(self.tokens) - n_ana
            self.min_fodder_words = n_other + max(n_ana, 0)
        self.label = label or f"{' · '.join(self.tokens)}"

    @property
    def total_word_span(self):
        """Total clue words consumed by token slots (if word_spans set)."""
        return sum(self.word_spans) if self.word_spans else None

    @property
    def min_words(self):
        """Minimum wordplay-window words needed (fodder + indicators)."""
        # Each indicator type needs at least 1 word
        return self.min_fodder_words + len(self.indicators)

    @property
    def n_indicator_types(self):
        return len(self.indicators)

    def __repr__(self):
        return f"<Sig {self.label} T{self.tier}>"


# ============================================================
# The catalog — generated from mined Times explanation data
# ============================================================

CATALOG = []


def _add(tokens, spans, operation, count, indicators=None):
    """Add a catalog entry with tier derived from count."""
    tier = 1 if count >= 200 else 2 if count >= 20 else 3
    label_parts = [f"{t}({s}w)" for t, s in zip(tokens, spans)]
    CATALOG.append(CatalogEntry(
        tokens=tokens, operation=operation, tier=tier,
        indicators=indicators or set(), word_spans=spans,
        label=f"{'·'.join(label_parts)} {operation}",
    ))


# ============================================================
# CHARADE — no indicators needed
# Mined counts from Times explanations
# ============================================================

_CHARADE_DATA = [
    # tokens, spans, count
    ((SYN_F, SYN_F), (1, 1), 452),
    ((ABR_F, SYN_F), (1, 1), 265),
    ((SYN_F, SYN_F), (2, 1), 207),
    ((SYN_F, ABR_F), (1, 1), 136),
    ((SYN_F, SYN_F), (3, 1), 94),
    ((ABR_F, SYN_F, SYN_F), (1, 1, 1), 87),
    ((SYN_F, SYN_F, SYN_F), (1, 1, 1), 75),
    ((SYN_F, SYN_F, ABR_F), (1, 1, 1), 67),
    ((SYN_F, ABR_F), (2, 1), 66),
    ((SYN_F, ABR_F, SYN_F), (1, 1, 1), 63),
    ((ABR_F, ABR_F), (1, 1), 57),
    ((ABR_F, ABR_F, ABR_F), (1, 1, 1), 52),
    ((ABR_F, ABR_F, SYN_F), (1, 1, 1), 50),
    ((ABR_F, SYN_F, ABR_F), (1, 1, 1), 40),
    ((SYN_F, ABR_F, ABR_F), (1, 1, 1), 40),
    ((ABR_F, SYN_F), (2, 1), 37),
    ((SYN_F, SYN_F), (4, 1), 30),
    ((ABR_F, ABR_F, ABR_F, ABR_F), (1, 1, 1, 1), 30),
    ((SYN_F, ABR_F), (3, 1), 28),
    ((SYN_F, ABR_F, SYN_F), (2, 1, 1), 27),
    ((SYN_F, SYN_F, ABR_F), (2, 1, 1), 24),
    ((SYN_F, SYN_F, SYN_F), (2, 1, 1), 21),
    ((ABR_F, ABR_F), (2, 1), 20),
    ((ABR_F, SYN_F, SYN_F), (2, 1, 1), 20),
    ((SYN_F, ABR_F, ABR_F), (2, 1, 1), 19),
    ((ABR_F, ABR_F, ABR_F, ABR_F, ABR_F), (1, 1, 1, 1, 1), 18),
    ((SYN_F, SYN_F), (5, 1), 14),
    ((ABR_F, SYN_F), (3, 1), 13),
    ((ABR_F, SYN_F, ABR_F, SYN_F), (1, 1, 1, 1), 13),
    ((SYN_F, SYN_F, ABR_F, ABR_F), (1, 1, 1, 1), 12),
    ((SYN_F, SYN_F, SYN_F), (3, 1, 1), 11),
    ((SYN_F, ABR_F), (4, 1), 10),
    ((ABR_F, ABR_F), (3, 1), 10),
    ((ABR_F, SYN_F, SYN_F, SYN_F), (1, 1, 1, 1), 10),
    ((ABR_F, ABR_F, ABR_F, SYN_F), (1, 1, 1, 1), 9),
    ((ABR_F, SYN_F, SYN_F, ABR_F), (1, 1, 1, 1), 9),
    ((ABR_F, ABR_F, SYN_F, ABR_F), (1, 1, 1, 1), 8),
    ((SYN_F, SYN_F), (1, 3), 8),
    ((SYN_F, ABR_F, ABR_F, ABR_F), (1, 1, 1, 1), 8),
    ((SYN_F, ABR_F, ABR_F), (3, 1, 1), 8),
    ((SYN_F, SYN_F, ABR_F), (3, 1, 1), 7),
    ((SYN_F, SYN_F, ABR_F, SYN_F), (1, 1, 1, 1), 7),
    ((ABR_F, ABR_F, SYN_F), (2, 1, 1), 7),
    ((SYN_F, ABR_F, ABR_F, SYN_F), (1, 1, 1, 1), 7),
    ((SYN_F, SYN_F, SYN_F, SYN_F), (1, 1, 1, 1), 6),
    ((SYN_F, ABR_F, SYN_F), (3, 1, 1), 6),
    ((ABR_F, SYN_F, ABR_F, ABR_F), (1, 1, 1, 1), 6),
    ((ABR_F, SYN_F), (1, 2), 6),
    ((SYN_F, SYN_F), (6, 1), 4),
    ((ABR_F, ABR_F), (1, 3), 4),
    ((ABR_F, SYN_F), (4, 1), 4),
    ((SYN_F, ABR_F, ABR_F), (4, 1, 1), 4),
    ((SYN_F, ABR_F, SYN_F, ABR_F), (1, 1, 1, 1), 4),
    ((ABR_F, ABR_F, SYN_F, SYN_F), (1, 1, 1, 1), 5),
    ((SYN_F, ABR_F, SYN_F), (1, 1, 1), 5),  # note: also in reversal_charade
    ((SYN_F, SYN_F, SYN_F, ABR_F), (1, 1, 1, 1), 5),
    ((ABR_F, ABR_F), (1, 2), 5),
    ((ABR_F, ABR_F), (4, 1), 3),
    ((SYN_F, SYN_F), (2, 2), 3),
    ((SYN_F, SYN_F), (3, 3), 3),
    ((ABR_F, SYN_F, ABR_F, ABR_F), (2, 1, 1, 1), 3),
    ((SYN_F, SYN_F, ABR_F, ABR_F), (2, 1, 1, 1), 3),
    ((SYN_F, SYN_F, SYN_F, SYN_F), (3, 1, 1, 1), 3),
    ((SYN_F, ABR_F, SYN_F), (4, 1, 1), 3),
    ((SYN_F, ABR_F, SYN_F, SYN_F), (1, 1, 1, 1), 3),
]

for _t, _s, _c in _CHARADE_DATA:
    _add(_t, _s, "charade", _c)

# ============================================================
# REVERSAL — requires REV_I
# ============================================================

_REVERSAL_DATA = [
    ((SYN_F,), (1,), 16),
]

for _t, _s, _c in _REVERSAL_DATA:
    _add(_t, _s, "reversal", _c, {REV_I})

# ============================================================
# REVERSAL_CHARADE — requires REV_I
# ============================================================

_REVERSAL_CHARADE_DATA = [
    ((SYN_F, SYN_F), (1, 1), 78),
    ((SYN_F, SYN_F), (2, 1), 31),
    ((ABR_F, SYN_F), (1, 1), 22),
    ((SYN_F, ABR_F), (1, 1), 11),
    ((SYN_F, SYN_F), (3, 1), 12),
    ((ABR_F, SYN_F, SYN_F), (1, 1, 1), 12),
    ((SYN_F, SYN_F, ABR_F), (1, 1, 1), 9),
    ((SYN_F, SYN_F), (4, 1), 9),
    ((SYN_F, SYN_F, SYN_F), (1, 1, 1), 8),
    ((SYN_F, ABR_F, SYN_F), (2, 1, 1), 6),
    ((SYN_F, ABR_F), (2, 1), 6),
    ((SYN_F, ABR_F, SYN_F), (1, 1, 1), 5),
    ((SYN_F, ABR_F), (3, 1), 5),
    ((SYN_F, ABR_F, ABR_F), (1, 1, 1), 4),
    ((SYN_F, ABR_F), (4, 1), 4),
    ((SYN_F,), (1,), 5),
    ((ABR_F, SYN_F, ABR_F), (1, 1, 1), 3),
]

for _t, _s, _c in _REVERSAL_CHARADE_DATA:
    _add(_t, _s, "reversal_charade", _c, {REV_I})

# ============================================================
# CONTAINER — requires CON_I
# Mined as single SYN_F with span (the two container parts are
# treated as one unit in the notation parser)
# ============================================================

_CONTAINER_DATA = [
    ((SYN_F,), (1,), 39),
    ((SYN_F, ABR_F), (1, 1), 9),
    ((SYN_F,), (4,), 8),
    ((SYN_F,), (5,), 7),
    ((SYN_F,), (2,), 6),
    ((SYN_F,), (3,), 6),
    ((SYN_F, ABR_F), (3, 1), 5),
    ((SYN_F, ABR_F), (4, 1), 4),
    ((SYN_F, ABR_F), (2, 1), 4),
    ((SYN_F,), (6,), 3),
]

for _t, _s, _c in _CONTAINER_DATA:
    _add(_t, _s, "container", _c, {CON_I})

# Also add 2-piece container entries (inner + outer as separate tokens)
# These are essential for the matcher which tries inserting one piece into another
_CONTAINER_2PIECE_DATA = [
    ((SYN_F, SYN_F), (1, 1), 200),
    ((SYN_F, SYN_F), (2, 1), 50),
    ((SYN_F, SYN_F), (1, 2), 50),
    ((SYN_F, SYN_F), (3, 1), 20),
    ((SYN_F, SYN_F), (1, 3), 20),
    ((ABR_F, SYN_F), (1, 1), 150),
    ((ABR_F, SYN_F), (1, 2), 30),
    ((ABR_F, SYN_F), (1, 3), 15),
    ((SYN_F, ABR_F), (1, 1), 100),
    ((SYN_F, ABR_F), (2, 1), 20),
    ((ABR_F, ABR_F), (1, 1), 20),
]

for _t, _s, _c in _CONTAINER_2PIECE_DATA:
    _add(_t, _s, "container", _c, {CON_I})

# ============================================================
# CONTAINER_CHARADE — requires CON_I
# ============================================================

_CONTAINER_CHARADE_DATA = [
    ((SYN_F, SYN_F), (1, 1), 3),
    ((SYN_F, SYN_F), (2, 1), 3),
    # 3-token: piece + piece inside piece (e.g. ELI + IN inside MATE = ELIMINATE)
    ((ABR_F, ABR_F, SYN_F), (1, 1, 1), 10),
    ((ABR_F, SYN_F, SYN_F), (1, 1, 1), 10),
    ((SYN_F, ABR_F, SYN_F), (1, 1, 1), 10),
    ((SYN_F, SYN_F, ABR_F), (1, 1, 1), 10),
]

for _t, _s, _c in _CONTAINER_CHARADE_DATA:
    _add(_t, _s, "container_charade", _c, {CON_I})

# ============================================================
# ANAGRAM — requires ANA_I
# Single ANA_F piece, span = number of fodder words
# ============================================================

_ANAGRAM_DATA = [
    ((ANA_F,), (1,), 136),
    ((ANA_F,), (2,), 300),  # estimated from coarse anagram count
    ((ANA_F,), (3,), 200),
    ((ANA_F,), (4,), 100),
    ((ANA_F,), (5,), 50),
    ((ANA_F,), (6,), 30),
    ((ANA_F,), (7,), 15),
    ((ANA_F,), (8,), 8),
]

for _t, _s, _c in _ANAGRAM_DATA:
    _add(_t, _s, "anagram", _c, {ANA_I})

# ============================================================
# ANAGRAM_CHARADE — requires ANA_I
# ANA_F + SYN_F/ABR_F pieces (anagram of some + fixed piece)
# ============================================================

_ANAGRAM_CHARADE_DATA = [
    ((ANA_F,), (1,), 17),
    ((ANA_F, SYN_F), (1, 1), 10),
    ((ANA_F, ABR_F), (2, 1), 9),
    ((ANA_F, ABR_F), (1, 1), 8),
    ((ANA_F,), (3,), 7),
    ((ANA_F,), (2,), 6),
    ((ANA_F, SYN_F), (2, 1), 5),
    ((ANA_F,), (4,), 3),
    ((ANA_F, SYN_F), (1, 2), 3),  # estimated
    ((SYN_F, ANA_F), (1, 1), 5),  # estimated
    ((ABR_F, ANA_F), (1, 1), 5),  # estimated
    ((ABR_F, ANA_F), (1, 2), 3),  # estimated
]

for _t, _s, _c in _ANAGRAM_CHARADE_DATA:
    _add(_t, _s, "anagram_charade", _c, {ANA_I})

# Also add anagram_plus (all pieces anagrammed together)
_ANAGRAM_PLUS_DATA = [
    ((ANA_F, ABR_F), (1, 1), 50),
    ((ANA_F, ABR_F), (2, 1), 30),
    ((ANA_F, ABR_F), (3, 1), 20),
    ((ABR_F, ANA_F), (1, 1), 50),
    ((ABR_F, ANA_F), (1, 2), 30),
    ((ABR_F, ANA_F), (1, 3), 15),
    ((ANA_F, ABR_F, ANA_F), (1, 1, 1), 10),  # e.g. PARCHMENT
    ((ANA_F, SYN_F), (1, 1), 10),
    ((SYN_F, ANA_F), (1, 1), 10),
]

for _t, _s, _c in _ANAGRAM_PLUS_DATA:
    _add(_t, _s, "anagram_plus", _c, {ANA_I})

# ============================================================
# DELETION — requires DEL_I
# ============================================================

_DELETION_DATA = [
    ((SYN_F,), (1,), 35),
    ((SYN_F,), (2,), 31),
    ((SYN_F,), (3,), 27),
    ((SYN_F,), (4,), 26),
    ((SYN_F,), (5,), 14),
    ((SYN_F,), (6,), 4),
    ((SYN_F, ABR_F), (1, 1), 10),  # estimated
    ((ABR_F, SYN_F), (1, 1), 10),  # estimated
]

for _t, _s, _c in _DELETION_DATA:
    _add(_t, _s, "deletion", _c, {DEL_I})

# ============================================================
# TRIM — requires POS_I_TRIM_*
# (same signature as deletion but with trim indicators)
# ============================================================

_TRIM_DATA = [
    ((SYN_F,), (1,), 100),
    ((SYN_F,), (2,), 30),
    ((SYN_F,), (3,), 10),
]

for _t, _s, _c in _TRIM_DATA:
    for _ind in [POS_I_TRIM_LAST, POS_I_TRIM_FIRST, POS_I_TRIM_OUTER,
                 POS_I_TRIM_MIDDLE]:
        _ind_name = str(_ind).split('_')[-1].lower()
        _add(_t, _s, "trim", _c, {_ind})

# ============================================================
# TRIM_CHARADE — trim + charade
# ============================================================

_TRIM_CHARADE_DATA = [
    ((SYN_F, SYN_F), (1, 1), 30),
    ((SYN_F, ABR_F), (1, 1), 20),
    ((ABR_F, SYN_F), (1, 1), 20),
    ((SYN_F, SYN_F), (2, 1), 10),
]

for _t, _s, _c in _TRIM_CHARADE_DATA:
    for _ind in [POS_I_TRIM_LAST, POS_I_TRIM_FIRST, DEL_I]:
        _add(_t, _s, "trim_charade", _c, {_ind})

# ============================================================
# HIDDEN — optionally requires HID_I
# ============================================================

_HIDDEN_DATA = [
    ((HID_F,), (2,), 500),
    ((HID_F,), (3,), 400),
    ((HID_F,), (4,), 200),
    ((HID_F,), (5,), 50),
]

for _t, _s, _c in _HIDDEN_DATA:
    _add(_t, _s, "hidden", _c, {HID_I})
    _add(_t, _s, "hidden", _c // 2)  # no indicator variant

# ============================================================
# HIDDEN_REVERSED — requires HID_I + REV_I (or just REV_I)
# ============================================================

for _t, _s, _c in _HIDDEN_DATA:
    _add(_t, _s, "hidden_reversed", _c // 3, {HID_I, REV_I})
    _add(_t, _s, "hidden_reversed", _c // 4, {REV_I})

# ============================================================
# HOMOPHONE — requires HOM_I
# ============================================================

_HOMOPHONE_DATA = [
    ((HOM_F,), (1,), 500),
    ((HOM_F,), (2,), 100),
    ((HOM_F,), (3,), 20),
]

for _t, _s, _c in _HOMOPHONE_DATA:
    _add(_t, _s, "homophone", _c, {HOM_I})

# ============================================================
# ALTERNATE — requires POS_I_ALTERNATE
# ============================================================

_ALTERNATE_DATA = [
    ((POS_F,), (1,), 200),
    ((POS_F,), (2,), 300),
    ((POS_F,), (3,), 100),
    ((POS_F,), (4,), 30),
]

for _t, _s, _c in _ALTERNATE_DATA:
    _add(_t, _s, "alternate", _c, {POS_I_ALTERNATE})

# ============================================================
# ACROSTIC — requires POS_I_FIRST
# ============================================================

_ACROSTIC_DATA = [
    ((POS_F,), (3,), 100),
    ((POS_F,), (4,), 200),
    ((POS_F,), (5,), 200),
    ((POS_F,), (6,), 150),
    ((POS_F,), (7,), 80),
    ((POS_F,), (8,), 40),
    ((POS_F,), (9,), 20),
    ((POS_F,), (10,), 10),
]

for _t, _s, _c in _ACROSTIC_DATA:
    _add(_t, _s, "acrostic", _c, {POS_I_FIRST})

# ============================================================
# POSITIONAL_CHARADE — positional extraction + charade
# ============================================================

_POSITIONAL_CHARADE_DATA = [
    ((POS_F, SYN_F), (1, 1), 30),
    ((POS_F, ABR_F), (1, 1), 15),
    ((SYN_F, POS_F), (1, 1), 30),
    ((ABR_F, POS_F), (1, 1), 15),
]

for _t, _s, _c in _POSITIONAL_CHARADE_DATA:
    _add(_t, _s, "positional_charade", _c, {POS_I_OUTER})

# Also add POS_I_FIRST and POS_I_LAST variants (e.g. "Dad's leadership" = D)
for _t, _s, _c in _POSITIONAL_CHARADE_DATA:
    _add(_t, _s, "positional_charade", _c, {POS_I_FIRST})
    _add(_t, _s, "positional_charade", _c, {POS_I_LAST})

# ============================================================
# SYNONYM — direct synonym lookup (no indicators)
# ============================================================

_SYNONYM_DATA = [
    ((SYN_F,), (1,), 14),
    ((SYN_F,), (2,), 3),
    ((SYN_F,), (3,), 6),
    ((SYN_F,), (4,), 7),
]

for _t, _s, _c in _SYNONYM_DATA:
    _add(_t, _s, "synonym", _c)

# ============================================================
# Sort by tier for priority ordering
# ============================================================
CATALOG.sort(key=lambda e: e.tier)
