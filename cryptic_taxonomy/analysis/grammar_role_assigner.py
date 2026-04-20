"""Grammar-based role assignment: use POS tags to assign cryptic roles.

The grammar tells us what each word probably is:
- Nouns are fodder (SYN_F or ABR_F)
- Gerunds/past participles between nouns are indicators
- Prepositions are links or container indicators
- Particles signal reversal
- Base verbs are synonyms, not indicators

This module:
1. Learns role probabilities per POS tag from the labeled dataset
2. Learns role probabilities per POS BIGRAM (context matters)
3. Assigns roles to a new wordplay window
4. Measures: how accurate is pure grammar-based assignment?
"""

import sqlite3
import sys
import spacy
import time
from collections import Counter, defaultdict

sys.path.insert(0, '.')


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")


def learn_from_data():
    """Learn POS -> role distributions from the labeled dataset."""
    print("Loading spaCy model...")
    nlp = spacy.load('en_core_web_sm')

    conn = sqlite3.connect('data/word_roles.db', timeout=60)
    clue_ids = [r[0] for r in conn.execute(
        'SELECT DISTINCT clue_id FROM word_roles'
    ).fetchall()]

    print(f"Learning from {len(clue_ids)} clues...")

    # Unigram: P(role | POS_tag)
    pos_role_counts = defaultdict(lambda: Counter())

    # Bigram context: P(role | prev_POS, curr_POS)
    bigram_role_counts = defaultdict(lambda: Counter())

    # Trigram: P(role | prev_POS, curr_POS, next_POS)
    trigram_role_counts = defaultdict(lambda: Counter())

    # Position-aware: P(role | POS_tag, is_first, is_last)
    pos_position_role = defaultdict(lambda: Counter())

    total = 0

    for i, clue_id in enumerate(clue_ids):
        if i % 3000 == 0 and i > 0:
            print(f"  {i}/{len(clue_ids)}...")

        rows = conn.execute('''
            SELECT word_text, assigned_role
            FROM word_roles WHERE clue_id=? ORDER BY word_position
        ''', (clue_id,)).fetchall()

        words = [r[0] for r in rows]
        roles = [r[1] for r in rows]
        n = len(words)

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

        if len(pos_tags) != len(roles):
            continue

        for j in range(n):
            pos = pos_tags[j]
            role = roles[j]

            # Unigram
            pos_role_counts[pos][role] += 1

            # Bigram (previous context)
            prev_pos = pos_tags[j - 1] if j > 0 else '<START>'
            bigram_role_counts[(prev_pos, pos)][role] += 1

            # Trigram
            next_pos = pos_tags[j + 1] if j < n - 1 else '<END>'
            trigram_role_counts[(prev_pos, pos, next_pos)][role] += 1

            # Position
            is_first = j == 0
            is_last = j == n - 1
            pos_position_role[(pos, is_first, is_last)][role] += 1

            total += 1

    conn.close()
    print(f"Learned from {total} word instances")

    return {
        'unigram': dict(pos_role_counts),
        'bigram': dict(bigram_role_counts),
        'trigram': dict(trigram_role_counts),
        'position': dict(pos_position_role),
    }


def assign_roles(pos_tags, model):
    """Assign most probable role to each word based on POS context.

    Uses trigram if available, falls back to bigram, then unigram.
    Returns list of (role, confidence) tuples.
    """
    n = len(pos_tags)
    assignments = []

    for j in range(n):
        pos = pos_tags[j]
        prev_pos = pos_tags[j - 1] if j > 0 else '<START>'
        next_pos = pos_tags[j + 1] if j < n - 1 else '<END>'

        # Try trigram first
        tri_key = (prev_pos, pos, next_pos)
        if tri_key in model['trigram']:
            counts = model['trigram'][tri_key]
            total = sum(counts.values())
            if total >= 5:  # enough data
                best_role = counts.most_common(1)[0][0]
                conf = counts.most_common(1)[0][1] / total
                assignments.append((best_role, conf, 'trigram'))
                continue

        # Try bigram
        bi_key = (prev_pos, pos)
        if bi_key in model['bigram']:
            counts = model['bigram'][bi_key]
            total = sum(counts.values())
            if total >= 10:
                best_role = counts.most_common(1)[0][0]
                conf = counts.most_common(1)[0][1] / total
                assignments.append((best_role, conf, 'bigram'))
                continue

        # Fall back to unigram
        if pos in model['unigram']:
            counts = model['unigram'][pos]
            total = sum(counts.values())
            best_role = counts.most_common(1)[0][0]
            conf = counts.most_common(1)[0][1] / total
            assignments.append((best_role, conf, 'unigram'))
        else:
            assignments.append(('LNK', 0.0, 'none'))

    return assignments


def evaluate(model):
    """Evaluate grammar-based role assignment against known labels."""
    print("\nEvaluating grammar role assignment...")
    nlp = spacy.load('en_core_web_sm')

    conn = sqlite3.connect('data/word_roles.db', timeout=60)
    clue_ids = [r[0] for r in conn.execute(
        'SELECT DISTINCT clue_id FROM word_roles'
    ).fetchall()]

    total = 0
    correct = 0
    by_role_correct = Counter()
    by_role_total = Counter()
    by_source = Counter()  # which n-gram level was used
    confusion = defaultdict(lambda: Counter())  # actual -> predicted

    # Simplify roles for evaluation: group indicator subtypes
    def simplify(role):
        if role in ('ANA_I', 'CON_I', 'REV_I', 'DEL_I', 'HID_I', 'HOM_I'):
            return 'INDICATOR'
        if role in ('SYN_F', 'ABR_F', 'RAW', 'ANA_F', 'HID_F', 'POS_F'):
            return 'FODDER'
        if role == 'LNK':
            return 'LNK'
        return role

    for i, clue_id in enumerate(clue_ids):
        if i % 3000 == 0 and i > 0:
            print(f"  {i}/{len(clue_ids)}...")

        rows = conn.execute('''
            SELECT word_text, assigned_role
            FROM word_roles WHERE clue_id=? ORDER BY word_position
        ''', (clue_id,)).fetchall()

        words = [r[0] for r in rows]
        actual_roles = [r[1] for r in rows]

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

        if len(pos_tags) != len(actual_roles):
            continue

        predicted = assign_roles(pos_tags, model)

        for j in range(len(words)):
            actual = actual_roles[j]
            pred_role, conf, source = predicted[j]
            total += 1
            by_role_total[simplify(actual)] += 1
            by_source[source] += 1

            # Exact match
            if pred_role == actual:
                correct += 1
                by_role_correct[simplify(actual)] += 1
            # Simplified match (FODDER vs INDICATOR vs LNK)
            elif simplify(pred_role) == simplify(actual):
                correct += 1
                by_role_correct[simplify(actual)] += 1

            confusion[simplify(actual)][simplify(pred_role)] += 1

    conn.close()

    print(f"\n{'='*70}")
    print(f"GRAMMAR ROLE ASSIGNMENT RESULTS")
    print(f"{'='*70}")
    print(f"Total words: {total}")
    print(f"Correct (simplified FODDER/INDICATOR/LNK): {correct} ({100*correct/total:.1f}%)")

    print(f"\n--- By actual role ---")
    print(f"{'Role':15s} {'Correct':>7s} {'Total':>7s} {'Accuracy':>8s}")
    print("-" * 40)
    for role in ['FODDER', 'INDICATOR', 'LNK']:
        c = by_role_correct[role]
        t = by_role_total[role]
        print(f"  {role:13s} {c:7d} {t:7d} {100*c/t:7.1f}%")

    print(f"\n--- Confusion matrix ---")
    print(f"{'':15s} {'pred FODDER':>12s} {'pred INDICATOR':>14s} {'pred LNK':>10s}")
    for actual in ['FODDER', 'INDICATOR', 'LNK']:
        row = f"  actual {actual:7s}"
        for pred in ['FODDER', 'INDICATOR', 'LNK']:
            cnt = confusion[actual][pred]
            total_actual = by_role_total[actual]
            pct = 100 * cnt / total_actual if total_actual else 0
            row += f" {cnt:6d}({pct:4.1f}%)"
        print(row)

    print(f"\n--- By n-gram source ---")
    for source, cnt in by_source.most_common():
        print(f"  {source:10s} {cnt:7d} ({100*cnt/total:.1f}%)")

    return correct, total


if __name__ == '__main__':
    model = learn_from_data()
    evaluate(model)
