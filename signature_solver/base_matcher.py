"""Base pattern matcher — flexible span assignment over collapsed I/F patterns.

Instead of matching against 694 specific (token, span) entries, this matcher:
1. Iterates ~68 base patterns like F+F charade
2. For each pattern, generates all valid span assignments exhaustively
3. For each span assignment, determines fodder types (ABR/SYN) via DB lookups
4. Verifies using existing _verify_combo logic

Key advantage: a single base entry F+F charade replaces dozens of specific
entries like ABR_F(1w)+SYN_F(2w), SYN_F(1w)+SYN_F(1w), etc.
"""

from .tokens import *
from .base_catalog import (
    BASE_CATALOG, CATALOG_BY_OPERATION, BaseEntry,
    OPERATION_INDICATOR_TYPE, OPERATION_FODDER_TYPES,
)
from .positional_catalog import INDICATOR_TOKENS_SET, FODDER_TOKENS_SET
from .word_analyzer import clean_word
from .matcher import (
    _lookup_slot, _verify_combo, _remaining_are_valid, _build_assignment,
)


# Max span per slot type
MAX_F_SPAN = 4
MAX_I_SPAN = 2

# Cap total placements tried per base entry to avoid explosion
MAX_PLACEMENTS_PER_ENTRY = 20

# Cap total fodder-type combos per placement
MAX_FODDER_COMBOS = 50


def match_base(words, analyses, phrases, catalog, answer, db, def_pos=None):
    """Yield (entry, assignment) pairs in priority order.

    Args:
        words: list of wordplay window words
        analyses: list of WordAnalysis (one per word)
        phrases: dict of (i, j) -> WordAnalysis for multi-word phrases
        catalog: list of BaseEntry (pre-sorted by tier/count)
        answer: known answer (uppercase, no spaces/hyphens)
        db: RefDB instance
        def_pos: "start", "end", or None if unknown

    Yields:
        (entry, assignment) — same format as matcher.match_signatures
    """
    n = len(words)
    answer_len = len(answer)

    # Pre-compute per-word possible tokens (fast set lookups)
    word_possible = []
    for wa in analyses:
        word_possible.append(set(wa.roles.keys()))

    # Pre-compute which words can be indicators of each type
    word_indicator_map = {}  # indicator_type -> list of word indices
    phrase_indicator_map = {}  # indicator_type -> list of (i, j) tuples
    for i, wa in enumerate(analyses):
        for tok in wa.roles:
            if tok in INDICATOR_TOKENS_SET:
                word_indicator_map.setdefault(tok, []).append(i)
    for (pi, pj), pwa in phrases.items():
        for tok in pwa.roles:
            if tok in INDICATOR_TOKENS_SET:
                phrase_indicator_map.setdefault(tok, []).append((pi, pj))

    for entry in catalog:
        # === Quick filters ===

        # Word count: need at least 1 word per token, could have LNK gaps
        if n < entry.min_words:
            continue
        # Upper bound: each F up to MAX_F_SPAN, each I up to MAX_I_SPAN, plus LNK
        max_words = entry.n_fodder * MAX_F_SPAN + entry.n_indicator * MAX_I_SPAN + 5
        if n > max_words:
            continue

        # Letter budget (quick check)
        if not _letter_budget_ok(entry, answer_len):
            continue

        # Check required indicator type is present in the words
        ind_type_raw = OPERATION_INDICATOR_TYPE.get(entry.operation)
        if isinstance(ind_type_raw, list):
            # Multiple possible indicator types — find which ones are present
            ind_types_to_try = [
                t for t in ind_type_raw
                if word_indicator_map.get(t) or phrase_indicator_map.get(t)
            ]
            if entry.n_indicator > 0 and not ind_types_to_try:
                continue
        else:
            ind_types_to_try = [ind_type_raw] if ind_type_raw else [None]
            if entry.n_indicator > 0 and ind_type_raw:
                if not word_indicator_map.get(ind_type_raw) and not phrase_indicator_map.get(ind_type_raw):
                    continue

        # === Generate span assignments and try each ===
        placements_tried = 0
        found = False

        for ind_type in ind_types_to_try:
            if found:
                break

            for spans in _generate_span_assignments(entry.pattern, n):
                if placements_tried >= MAX_PLACEMENTS_PER_ENTRY:
                    break

                # Place tokens left-to-right with LNK gaps
                for placement in _place_spans(
                    entry.pattern, spans, n, word_possible, phrases, ind_type
                ):
                    placements_tried += 1
                    if placements_tried > MAX_PLACEMENTS_PER_ENTRY:
                        break

                    # Try to verify this placement
                    result = _verify_base_placement(
                        entry, placement, spans, words, analyses, phrases,
                        word_possible, answer, answer_len, db, ind_type
                    )
                    if result is not None:
                        yield result
                        found = True
                        break  # One verified result per entry is enough

            if found:
                break


def _letter_budget_ok(entry, answer_len):
    """Quick letter budget check for base pattern."""
    op = entry.operation
    if op.startswith('anagram') or op == 'hidden' or op == 'hidden_reversed':
        return True
    if op.startswith('del') or op.startswith('trim'):
        return True

    # For charade/container/reversal: need at least n_fodder letters,
    # at most n_fodder * answer_len letters
    min_total = entry.n_fodder  # each fodder contributes at least 1 letter
    max_total = entry.n_fodder * answer_len
    return min_total <= answer_len <= max_total


def _generate_span_assignments(pattern, n_words):
    """Generate all valid span tuples for the pattern.

    Each F slot gets span 1..MAX_F_SPAN, each I slot gets span 1..MAX_I_SPAN.
    Total span must be <= n_words (remaining words are LNK).
    Sorted by total span ascending (prefer tighter fits).

    Yields: tuple of spans, one per pattern token.
    """
    n_tokens = len(pattern)
    max_spans = []
    for tok in pattern:
        if tok == 'F':
            max_spans.append(MAX_F_SPAN)
        else:  # 'I'
            max_spans.append(MAX_I_SPAN)

    def recurse(idx, budget):
        if idx == n_tokens:
            if budget >= 0:
                yield ()
            return
        max_s = min(max_spans[idx], budget)
        for s in range(1, max_s + 1):
            for rest in recurse(idx + 1, budget - s):
                yield (s,) + rest

    # Budget = n_words (LNK words fill the rest)
    seen = set()
    for spans in recurse(0, n_words):
        if spans not in seen:
            seen.add(spans)
            yield spans


def _place_spans(pattern, spans, n, word_possible, phrases, ind_type,
                 max_results=5):
    """Place tokens left-to-right with LNK gaps.

    Similar to positional_matcher._place_recursive but with flexible spans
    from the base pattern.

    Yields: list of (start_pos, span) for each token.
    """
    results = []
    _place_recursive(pattern, spans, 0, 0, n, word_possible, phrases,
                     ind_type, [], results, max_results)
    return results


def _word_is_skippable(word_possible_set):
    """Check if a word can be left unassigned (as LNK or leftover indicator)."""
    if LNK in word_possible_set:
        return True
    # Indicators can also be leftover (they signal operations, not letters)
    return bool(word_possible_set & INDICATOR_TOKENS)


def _place_recursive(pattern, spans, seq_idx, min_pos, n,
                     word_possible, phrases, ind_type,
                     current, results, max_results):
    """Recursively place tokens left-to-right."""
    if len(results) >= max_results:
        return

    if seq_idx >= len(pattern):
        # All tokens placed — check remaining words are skippable
        used = set()
        for start, span in current:
            used.update(range(start, start + span))
        remaining = [i for i in range(n) if i not in used]
        if all(_word_is_skippable(word_possible[i]) for i in remaining):
            results.append(list(current))
        return

    tok_type = pattern[seq_idx]  # 'F' or 'I'
    span = spans[seq_idx]

    for start in range(min_pos, n - span + 1):
        # Check gap words (min_pos..start-1) are all skippable
        gap_ok = all(_word_is_skippable(word_possible[i])
                     for i in range(min_pos, start))
        if not gap_ok:
            break  # Can't skip past a non-skippable word

        # Check if words at this position can fill this slot
        if _can_fill_base(tok_type, span, start, word_possible, phrases, ind_type):
            current.append((start, span))
            _place_recursive(pattern, spans, seq_idx + 1, start + span, n,
                             word_possible, phrases, ind_type,
                             current, results, max_results)
            current.pop()

            if len(results) >= max_results:
                return


def _can_fill_base(tok_type, span, start, word_possible, phrases, ind_type):
    """Check if a base token (F or I) can be placed at start with given span."""
    if tok_type == 'I':
        # Indicator slot — check if word(s) have the required indicator type
        if ind_type is None:
            return False
        if span == 1:
            return ind_type in word_possible[start]
        else:
            key = (start, start + span)
            if key in phrases:
                return ind_type in set(phrases[key].roles.keys())
            return False

    elif tok_type == 'F':
        # Fodder slot — any word can potentially be fodder
        # (actual type determined at verification time via DB lookups)
        # But we can filter: at least one word should have SOME fodder role
        if span == 1:
            # Accept if word has any fodder role, or any role at all
            # (even LNK-only words might be RAW fodder)
            return True
        else:
            # Multi-word: always possible (phrases may form synonyms, anagrams, etc.)
            return True

    return False


def _verify_base_placement(entry, placement, spans, words, analyses, phrases,
                           word_possible, answer, answer_len, db, ind_type):
    """Verify a base placement by trying all fodder type combinations.

    For each F slot, try each possible fodder token type (SYN_F, ABR_F, etc.)
    and use _lookup_slot to find values. Then verify with _verify_combo.
    """
    op = entry.operation
    is_reversal = op in ("reversal", "reversal_charade")

    # Separate F and I slots
    f_slots = []  # (placement_idx, start, span)
    i_slots = []  # (placement_idx, start, span)
    used = set()

    for idx, ((start, span), tok_type) in enumerate(zip(placement, entry.pattern)):
        indices = list(range(start, start + span))
        used.update(indices)
        if tok_type == 'F':
            f_slots.append((idx, start, span))
        else:  # 'I'
            i_slots.append((idx, start, span))

    # Build indicator assignment
    ind_assignment = {}
    if ind_type and i_slots:
        for _, start, span in i_slots:
            if span == 1:
                ind_assignment[ind_type] = start
            else:
                ind_assignment[ind_type] = (start, start + span)

    # Remaining words
    leftover = [i for i in range(len(words)) if i not in used]

    # For container_positional / positional_charade, scan leftover words
    # for POS_I_* indicators and include them — _lookup_slot needs them
    # to know which positional extraction to apply to POS_F fodder,
    # and _remaining_are_valid needs them to accept those words.
    extra_indicators = set()
    if op in ('container_positional', 'positional_charade'):
        for li in leftover:
            for tok in word_possible[li]:
                if tok in (POS_I_FIRST, POS_I_LAST, POS_I_OUTER, POS_I_MIDDLE,
                           POS_I_ALTERNATE, POS_I_HALF, POS_I_TRIM_FIRST,
                           POS_I_TRIM_LAST, POS_I_TRIM_MIDDLE, POS_I_TRIM_OUTER):
                    extra_indicators.add(tok)

    # Check leftover words are valid
    assigned_ind_types = set(ind_assignment.keys()) | extra_indicators if ind_assignment or extra_indicators else None
    if leftover and not _remaining_are_valid(
            leftover, words, analyses, db, assigned_ind_types):
        return None

    # Get possible fodder types for this operation
    fodder_types = OPERATION_FODDER_TYPES.get(op, ['SYN_F', 'ABR_F'])

    # Build a mock entry-like object for _lookup_slot compatibility
    mock_entry = _MockEntry(op, set(ind_assignment.keys()) | extra_indicators)

    # For each F slot, try each fodder type and collect values
    # Then try all combinations
    slot_type_values = []  # list of list of (token_type, values)
    for _, start, span in f_slots:
        indices = list(range(start, start + span))
        type_vals = []
        for ftype in fodder_types:
            vals = _lookup_slot(
                indices, ftype, span, words, analyses,
                answer, answer_len, is_reversal, db, clean_word, mock_entry
            )
            if vals:
                type_vals.append((ftype, vals))
        if not type_vals:
            return None
        slot_type_values.append(type_vals)

    # Try combinations of fodder types across slots
    combo_count = 0
    for type_combo in _fodder_type_combos(slot_type_values):
        combo_count += 1
        if combo_count > MAX_FODDER_COMBOS:
            break

        slot_values = []
        slot_word_groups = []
        for i, (ftype, vals) in enumerate(type_combo):
            _, start, span = f_slots[i]
            indices = list(range(start, start + span))
            slot_values.append(vals)
            slot_word_groups.append((indices, ftype, span))

        result = _verify_combo(op, mock_entry, slot_values, slot_word_groups,
                               ind_assignment, leftover, answer, words)
        if result is not None:
            return result

    return None


def _fodder_type_combos(slot_type_values):
    """Generate combinations of (ftype, vals) across slots.

    slot_type_values: list of list of (ftype, vals) per slot.
    Yields: tuple of (ftype, vals), one per slot.
    """
    if not slot_type_values:
        yield ()
        return

    first, *rest = slot_type_values
    for item in first:
        for combo in _fodder_type_combos(rest):
            yield (item,) + combo


class _MockEntry:
    """Minimal entry-like object for compatibility with _lookup_slot and _verify_combo."""

    __slots__ = ('operation', 'indicators', 'tokens', 'word_spans', 'label')

    def __init__(self, operation, indicators):
        self.operation = operation
        self.indicators = frozenset(indicators)
        self.tokens = ()  # Not used by _lookup_slot in ways that matter
        self.word_spans = None
        self.label = f'base:{operation}'
