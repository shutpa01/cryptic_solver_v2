"""Combined triage: three independent signals scoring each operation.

For each clue, compute a score per operation type from:
1. Letter budget — how well does the ratio match this operation's profile?
2. Indicator DB — are the right indicator types present?
3. POS bigrams — does the grammatical structure match this operation's fingerprint?

The operation with the highest combined score wins. If no operation
scores above a threshold, the clue is UNCLASSIFIED.

Every classification requires positive evidence from at least two
of the three signals.
"""

import sqlite3
import sys
import re
import spacy
import time
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

sys.path.insert(0, '.')

from cryptic_taxonomy.analysis.mine_positional_signatures import IndicatorDB
from signature_solver.tokens import LINK_WORDS


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")


# ============================================================
# Signal 1: Letter budget profiles (from empirical data)
# ============================================================

# Mean and typical range per operation (from Step 3 analysis)
BUDGET_PROFILES = {
    'anagram':           (2.03, 1.0, 2.5),   # (mean, low, high)
    'charade':           (3.55, 1.5, 7.0),
    'container':         (3.90, 2.0, 7.0),
    'reversal_charade':  (4.43, 2.0, 7.0),
    'reversal':          (4.72, 2.0, 8.0),
    'deletion':          (5.55, 2.5, 8.0),
    'hidden':            (4.64, 3.0, 8.0),
}


def score_budget(ratio, operation):
    """Score how well the letter budget matches this operation.
    Returns 0.0-1.0."""
    if operation not in BUDGET_PROFILES:
        return 0.3  # neutral for unknown ops
    mean, low, high = BUDGET_PROFILES[operation]
    if ratio < low or ratio > high:
        return 0.0
    # Gaussian-like: closer to mean = higher score
    diff = abs(ratio - mean) / max(mean, 1)
    return max(0.0, 1.0 - diff)


# ============================================================
# Signal 2: Indicator presence
# ============================================================

# Which indicator DB types support each operation
OPERATION_INDICATORS = {
    'anagram':          {'anagram'},
    'charade':          set(),  # NO indicators = charade signal
    'container':        {'container', 'insertion'},
    'reversal_charade': {'reversal'},
    'reversal':         {'reversal'},
    'deletion':         {'deletion'},
    'hidden':           {'hidden'},
}


def score_indicators(ind_types_present, operation):
    """Score indicator evidence for this operation.
    Returns 0.0-1.0."""
    required = OPERATION_INDICATORS.get(operation, set())

    if operation == 'charade':
        # Charade: ABSENCE of indicators is the signal
        if not ind_types_present:
            return 1.0  # strong signal
        # Indicators present but could be red herrings
        return 0.3

    if not required:
        return 0.3  # neutral

    # Check if required indicators are present
    if required & ind_types_present:
        return 0.8
    return 0.0  # required indicator missing


# ============================================================
# Signal 3: POS bigram scoring
# ============================================================

# Empirically derived bigram weights per operation.
# Each entry: (pos1, pos2) -> weight (from overrepresentation ratio)
# Only include bigrams that are >2x overrepresented in that operation.

CONTAINER_BIGRAMS = {
    ('NN', 'VBN'): 2.1, ('NNP', 'VBN'): 2.9, ('NNP', 'VBG'): 2.0,
    ('NN', 'VBG'): 1.5, ('NNS', 'VBG'): 1.5, ('NNS', 'VBN'): 1.5,
    ('VBN', 'TO'): 3.3, ('VB', 'VBG'): 3.1, ('VB', 'TO'): 2.5,
    ('CD', 'VBN'): 2.3,
}

REVERSAL_BIGRAMS = {
    ('VBD', 'RP'): 4.5, ('VBN', 'RP'): 4.0, ('RP', 'IN'): 3.6,
    ('RP', 'JJ'): 3.9, ('VBZ', 'RP'): 3.8, ('RP', 'TO'): 4.7,
    ('RP', 'NN'): 3.2, ('VBD', 'RB'): 3.0,
}

ANAGRAM_BIGRAMS = {
    ('PRP', 'VBD'): 3.9, ('MD', 'VB'): 3.7, ('VBP', 'JJ'): 3.5,
    ('NN', 'MD'): 3.4, ('PRP', 'VBP'): 2.9, ('NNS', 'VBP'): 2.8,
    ('PRP$', 'NN'): 2.7, ('VBD', 'TO'): 2.6,
}

DELETION_BIGRAMS = {
    ('JJ', 'TO'): 5.1, ('VBG', 'RP'): 4.6, ('RB', 'VBG'): 2.9,
    ('VBG', 'DT'): 2.8, ('NNP', 'VBG'): 2.6, ('JJ', 'VBG'): 2.5,
    ('VBG', 'NN'): 2.5,
}

HIDDEN_BIGRAMS = {
    # Hidden is already handled pre-S, but include for completeness
    ('VBN', 'IN'): 2.0, ('IN', 'JJ'): 1.5, ('IN', 'NNP'): 1.5,
}

OP_BIGRAM_WEIGHTS = {
    'container': CONTAINER_BIGRAMS,
    'reversal_charade': REVERSAL_BIGRAMS,
    'reversal': REVERSAL_BIGRAMS,
    'anagram': ANAGRAM_BIGRAMS,
    'deletion': DELETION_BIGRAMS,
    'hidden': HIDDEN_BIGRAMS,
    'charade': {},  # charade has no distinctive bigrams
}


def score_pos_bigrams(pos_tags, operation):
    """Score POS bigram evidence for this operation.
    Returns 0.0-1.0 based on how many distinctive bigrams are present."""
    weights = OP_BIGRAM_WEIGHTS.get(operation, {})
    if not weights:
        return 0.3  # neutral for operations without bigram data

    total_weight = 0.0
    max_possible = sum(weights.values())

    for i in range(len(pos_tags) - 1):
        bigram = (pos_tags[i], pos_tags[i + 1])
        if bigram in weights:
            total_weight += weights[bigram]

    if max_possible == 0:
        return 0.3

    # Normalize: one strong bigram hit should give ~0.5, two+ should approach 1.0
    score = min(1.0, total_weight / 5.0)
    return score


# ============================================================
# Combined scoring
# ============================================================

OPERATIONS = ['anagram', 'charade', 'container', 'reversal_charade',
              'reversal', 'deletion']

# Minimum combined score to classify (otherwise UNCLASSIFIED)
MIN_SCORE = 0.8
# Must have at least 2 signals contributing (>0.2 each)
MIN_SIGNALS = 2


def triage_clue(ratio, ind_types, pos_tags):
    """Score all operations and return best classification.

    Returns (operation, score, signal_breakdown) or ('UNCLASSIFIED', 0, {}).
    """
    best_op = None
    best_score = 0
    best_breakdown = {}

    for op in OPERATIONS:
        s_budget = score_budget(ratio, op)
        s_indicators = score_indicators(ind_types, op)
        s_pos = score_pos_bigrams(pos_tags, op)

        combined = s_budget + s_indicators + s_pos

        # Count signals contributing
        signals = sum(1 for s in [s_budget, s_indicators, s_pos] if s > 0.2)

        breakdown = {
            'budget': s_budget,
            'indicators': s_indicators,
            'pos_bigrams': s_pos,
            'signals': signals,
        }

        if combined > best_score and signals >= MIN_SIGNALS:
            best_score = combined
            best_op = op
            best_breakdown = breakdown

    if best_op and best_score >= MIN_SCORE:
        return best_op, best_score, best_breakdown

    return 'UNCLASSIFIED', 0, {}


# ============================================================
# Test harness
# ============================================================

def run():
    print("Loading spaCy model...")
    nlp = spacy.load('en_core_web_sm')
    ind_db = IndicatorDB()

    conn = sqlite3.connect('data/word_roles.db', timeout=60)

    clues = conn.execute('''
        SELECT DISTINCT clue_id, operation, letter_budget_ratio
        FROM word_roles WHERE word_position = 0
    ''').fetchall()

    print(f"Running combined triage on {len(clues)} clues...")
    t0 = time.time()

    results = []

    for i, (clue_id, actual_op, ratio) in enumerate(clues):
        if actual_op == 'hidden':
            continue

        if i % 3000 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(clues)} ({elapsed:.1f}s)")

        rows = conn.execute('''
            SELECT word_text FROM word_roles
            WHERE clue_id=? ORDER BY word_position
        ''', (clue_id,)).fetchall()
        words = [r[0] for r in rows]

        # Get indicator types
        ind_types = set()
        for word in words:
            for t in ind_db.get_indicator_types(word):
                ind_types.add(t)

        # POS tag
        wp_text = ' '.join(words)
        doc = nlp(wp_text)
        spacy_tokens = list(doc)
        pos_tags = []
        si = 0
        for word in words:
            if si < len(spacy_tokens):
                pos_tags.append(spacy_tokens[si].tag_)
                consumed = len(spacy_tokens[si].text)
                si += 1
                while consumed < len(word) and si < len(spacy_tokens):
                    consumed += len(spacy_tokens[si].text) + 1
                    si += 1
            else:
                pos_tags.append('XX')

        # Score
        predicted, score, breakdown = triage_clue(ratio, ind_types, pos_tags)
        results.append((clue_id, actual_op, predicted, score, breakdown))

    elapsed = time.time() - t0
    conn.close()
    print(f"\nDone in {elapsed:.1f}s")

    # === Analysis ===
    total = len(results)
    with_pred = sum(1 for _, _, p, _, _ in results if p != 'UNCLASSIFIED')
    correct = sum(1 for _, a, p, _, _ in results if a == p)

    # Also count reversal/reversal_charade as interchangeable
    correct_loose = sum(1 for _, a, p, _, _ in results
                        if a == p or (a in ('reversal', 'reversal_charade')
                                      and p in ('reversal', 'reversal_charade')))

    print(f"\n{'='*70}")
    print(f"COMBINED TRIAGE RESULTS")
    print(f"{'='*70}")
    print(f"Total clues: {total}")
    print(f"Classified: {with_pred} ({100*with_pred/total:.1f}%)")
    print(f"Correct (strict): {correct} ({100*correct/total:.1f}%)")
    print(f"Correct (rev/rev_ch grouped): {correct_loose} ({100*correct_loose/total:.1f}%)")
    print(f"UNCLASSIFIED: {total - with_pred} ({100*(total-with_pred)/total:.1f}%)")

    # By predicted operation
    print(f"\n--- By predicted operation ---")
    by_pred = defaultdict(lambda: Counter())
    for _, actual, pred, _, _ in results:
        by_pred[pred][actual] += 1

    print(f"{'Predicted':25s} {'Total':>6s} {'Correct':>7s} {'Prec':>6s}")
    print("-" * 50)
    for pred in sorted(by_pred.keys(), key=lambda x: -sum(by_pred[x].values())):
        total_pred = sum(by_pred[pred].values())
        correct_pred = by_pred[pred].get(pred, 0) if pred != 'UNCLASSIFIED' else 0
        prec = 100 * correct_pred / total_pred if total_pred else 0
        print(f"  {pred:23s} {total_pred:6d} {correct_pred:7d} {prec:5.1f}%")
        for actual, cnt in by_pred[pred].most_common(4):
            if actual != pred:
                pct = 100 * cnt / total_pred
                print(f"    actually {actual:20s} {cnt:5d} ({pct:.1f}%)")

    # Per-operation recall
    print(f"\n--- Per-operation recall ---")
    print(f"{'Operation':25s} {'Correct':>7s} {'Total':>7s} {'Recall':>7s}")
    print("-" * 50)
    by_actual = defaultdict(lambda: {'correct': 0, 'total': 0})
    for _, actual, pred, _, _ in results:
        by_actual[actual]['total'] += 1
        if actual == pred:
            by_actual[actual]['correct'] += 1
    for op in sorted(by_actual.keys(), key=lambda x: -by_actual[x]['total']):
        c = by_actual[op]['correct']
        t = by_actual[op]['total']
        print(f"  {op:23s} {c:7d} {t:7d} {100*c/t:6.1f}%")

    # Signal contribution analysis
    print(f"\n--- Signal contribution for correct predictions ---")
    correct_breakdowns = [b for _, a, p, _, b in results if a == p and b]
    if correct_breakdowns:
        avg_budget = sum(b['budget'] for b in correct_breakdowns) / len(correct_breakdowns)
        avg_ind = sum(b['indicators'] for b in correct_breakdowns) / len(correct_breakdowns)
        avg_pos = sum(b['pos_bigrams'] for b in correct_breakdowns) / len(correct_breakdowns)
        avg_signals = sum(b['signals'] for b in correct_breakdowns) / len(correct_breakdowns)
        print(f"  Avg budget score:     {avg_budget:.2f}")
        print(f"  Avg indicator score:  {avg_ind:.2f}")
        print(f"  Avg POS bigram score: {avg_pos:.2f}")
        print(f"  Avg signals active:   {avg_signals:.1f}")

    # Compare: what did we get right that the previous triage missed?
    print(f"\n--- Score distribution ---")
    scores = [s for _, _, p, s, _ in results if p != 'UNCLASSIFIED']
    if scores:
        for threshold in [0.8, 1.0, 1.2, 1.5, 2.0]:
            n_above = sum(1 for s in scores if s >= threshold)
            correct_above = sum(1 for _, a, p, s, _ in results
                               if p != 'UNCLASSIFIED' and s >= threshold and a == p)
            total_above = sum(1 for _, _, p, s, _ in results
                             if p != 'UNCLASSIFIED' and s >= threshold)
            prec = 100 * correct_above / total_above if total_above else 0
            print(f"  Score >= {threshold}: {total_above:5d} classified, {correct_above:5d} correct ({prec:.1f}%)")


if __name__ == '__main__':
    run()
