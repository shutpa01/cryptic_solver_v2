"""Positional matcher — word-first, left-to-right catalog matching.

Instead of iterating all catalog entries and trying to fit words,
this matcher:
1. Analyzes each word's possible roles
2. Matches catalog entry sequences left-to-right against word positions
3. Allows LNK gaps between sequence tokens
4. Uses def_pos to halve the search space

The key constraint: tokens in the entry sequence must appear in
left-to-right order in the clue, with only LNK words in between.
"""

from .tokens import *
from .positional_catalog import (
    POSITIONAL_CATALOG, CATALOG_BY_OPERATION,
    INDICATOR_TOKENS_SET, FODDER_TOKENS_SET,
)
from .word_analyzer import analyze_phrases, clean_word
from .matcher import (
    _lookup_slot, _verify_combo, _remaining_are_valid, _build_assignment,
)


def match_positional(words, analyses, phrases, catalog, answer, db, def_pos=None):
    """Yield (entry, assignment) pairs in priority order.

    Args:
        words: list of wordplay window words
        analyses: list of WordAnalysis (one per word)
        phrases: dict of (i, j) -> WordAnalysis for multi-word phrases
        catalog: list of PositionalEntry (pre-sorted by tier/count)
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

    # Pre-compute phrase data
    phrase_possible = {}
    for (pi, pj), pwa in phrases.items():
        phrase_possible[(pi, pj)] = set(pwa.roles.keys())

    for entry in catalog:
        # === Quick filters ===

        # Filter: def_pos must match (if both specified)
        if def_pos and entry.def_pos and def_pos != entry.def_pos:
            continue

        # Filter: word count
        if n < entry.min_words:
            continue
        max_words = sum(entry.spans) + 5
        if n > max_words:
            continue

        # Filter: letter budget
        if not _letter_budget_ok(entry, answer_len):
            continue

        # === Positional matching: find valid placements ===
        placements = []
        _place_recursive(entry.sequence, 0, 0, n,
                         word_possible, phrase_possible,
                         [], placements, max_results=5)

        # For each placement, do lookup + verification using existing matcher logic
        for placement in placements:
            result = _verify_placement(
                entry, placement, words, analyses, phrases,
                word_possible, answer, answer_len, db
            )
            if result is not None:
                yield result
                return  # One verified result per entry is enough


def _letter_budget_ok(entry, answer_len):
    """Quick letter budget check."""
    op = entry.operation
    if op.startswith('anagram') or op == 'hidden' or op.startswith('del'):
        return True

    min_total = 0
    max_total = 0
    for tok, span in entry.fodder_tokens:
        if tok == ABR_F:
            min_total += 1
            max_total += 3
        elif tok in (SYN_F, RAW, HOM_F, ANA_F):
            min_total += 1
            max_total += answer_len
        else:
            min_total += 1
            max_total += answer_len

    return min_total <= answer_len <= max_total


def _place_recursive(seq, seq_idx, min_pos, n,
                     word_possible, phrase_possible,
                     current, results, max_results=5):
    """Recursively place sequence tokens left-to-right.

    current: list of (start_pos, span) for tokens placed so far.
    Only checks if words CAN have the required role (fast filtering).
    Actual value lookup and verification happens after placement.
    """
    if len(results) >= max_results:
        return

    if seq_idx >= len(seq):
        # All tokens placed — check remaining words are LNK-able
        used = set()
        for start, span in current:
            used.update(range(start, start + span))
        remaining = [i for i in range(n) if i not in used]
        if all(LNK in word_possible[i] for i in remaining):
            results.append(list(current))
        return

    tok, span = seq[seq_idx]

    for start in range(min_pos, n - span + 1):
        # Check gap words (min_pos..start-1) are all LNK-able
        gap_ok = all(LNK in word_possible[i] for i in range(min_pos, start))
        if not gap_ok:
            break  # Can't skip past a non-LNK word

        # Check if word(s) at this position can have this role
        if _can_fill(tok, span, start, word_possible, phrase_possible):
            current.append((start, span))
            _place_recursive(seq, seq_idx + 1, start + span, n,
                             word_possible, phrase_possible,
                             current, results, max_results)
            current.pop()

            if len(results) >= max_results:
                return


def _can_fill(tok, span, start, word_possible, phrase_possible):
    """Check if token can be filled at position start with given span."""
    if tok in INDICATOR_TOKENS_SET:
        if span == 1:
            return tok in word_possible[start]
        else:
            key = (start, start + span)
            return key in phrase_possible and tok in phrase_possible[key]

    if tok in FODDER_TOKENS_SET:
        if span == 1:
            return tok in word_possible[start]
        else:
            if tok in (ANA_F, HID_F):
                return True  # Any words can be anagram/hidden fodder
            else:
                key = (start, start + span)
                return key in phrase_possible and tok in phrase_possible[key]

    if tok == LNK:
        return all(LNK in word_possible[start + i] for i in range(span))

    return False


def _verify_placement(entry, placement, words, analyses, phrases,
                      word_possible, answer, answer_len, db):
    """Verify a placement using the existing matcher's lookup + verify logic.

    Converts the positional placement into the format expected by
    _lookup_slot and _verify_combo, then runs verification.
    """
    seq = entry.sequence
    op = entry.operation
    is_reversal = op in ("reversal", "reversal_charade")

    # Separate fodder slots from indicator slots
    fodder_slots = []  # (indices, token, span)
    ind_assignment = {}  # token -> word_idx or (start, end) tuple
    used = set()

    for (start, span), (tok, _) in zip(placement, seq):
        indices = list(range(start, start + span))
        used.update(indices)

        if tok in INDICATOR_TOKENS_SET:
            if span == 1:
                ind_assignment[tok] = start
            else:
                ind_assignment[tok] = (start, start + span)
        else:
            fodder_slots.append((indices, tok, span))

    # Remaining words (not placed by any token)
    leftover = [i for i in range(len(words)) if i not in used]

    # Check leftover words are valid (LNK or redundant indicators)
    assigned_ind_types = set(ind_assignment.keys()) if ind_assignment else None
    if leftover and not _remaining_are_valid(
            leftover, words, analyses, db, assigned_ind_types):
        return None

    # Look up values for each fodder slot using existing _lookup_slot
    slot_values = []
    slot_word_groups = []
    for indices, token_type, span_size in fodder_slots:
        vals = _lookup_slot(
            indices, token_type, span_size, words, analyses,
            answer, answer_len, is_reversal, db, clean_word, entry
        )
        if not vals:
            return None
        slot_values.append(vals)
        slot_word_groups.append((indices, token_type, span_size))

    # Verify using existing _verify_combo
    result = _verify_combo(op, entry, slot_values, slot_word_groups,
                           ind_assignment, leftover, answer, words)
    return result
