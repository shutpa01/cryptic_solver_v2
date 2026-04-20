"""Mechanism-based triage: detect which piece mechanisms are present.

Every clue (except standalone types) is a charade. The question is:
what mechanisms do its pieces use?

Step 1: Check standalone types (pure anagram, pure reversal, cryptic def)
Step 2: It's a charade. Detect which mechanisms are present:
        - CONTAINER mechanism (gerund between nouns, "in" between nouns)
        - REVERSAL mechanism (past verb + particle, particle patterns)
        - DELETION mechanism (gerund + particle patterns)
        - POSITIONAL mechanism (not yet implemented)
        - ANAGRAM sub-mechanism (letters rearrange to part of answer)
Step 3: Confirm with structural tests against the answer
Step 4: Assemble: try pieces with detected mechanisms first
"""

import sqlite3
import sys
import re
import spacy
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import Dict, List, Set, Tuple, Optional

sys.path.insert(0, '.')

from cryptic_taxonomy.analysis.evidence_triage import (
    QuickRefDB, WordRole, Evidence,
    container_evidence, reversal_evidence, deletion_evidence,
    anagram_evidence, charade_evidence,
)
from cryptic_taxonomy.analysis.mine_positional_signatures import IndicatorDB
from signature_solver.tokens import LINK_WORDS


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")


# ============================================================
# POS-based mechanism detectors
# ============================================================

# Each detector returns a confidence 0.0-1.0 based on POS bigrams

def detect_container_mechanism(pos_tags):
    """Gerund or preposition between nouns signals container."""
    score = 0.0
    noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
    container_verb_tags = {'VBG', 'VBN'}
    container_prep_tags = {'IN'}

    for i in range(len(pos_tags) - 2):
        left = pos_tags[i] in noun_tags
        right = pos_tags[i + 2] in noun_tags
        mid_verb = pos_tags[i + 1] in container_verb_tags
        mid_prep = pos_tags[i + 1] in container_prep_tags

        if left and right and mid_verb:
            score = max(score, 0.8)
        elif left and right and mid_prep:
            score = max(score, 0.5)  # "in" is common in non-containers too

    # Also check: NNP/NN + VBG anywhere (not just between nouns)
    for i in range(len(pos_tags) - 1):
        if pos_tags[i] in noun_tags and pos_tags[i + 1] in container_verb_tags:
            score = max(score, 0.4)

    return score


def detect_reversal_mechanism(pos_tags):
    """Past verb + particle, or particle patterns signal reversal."""
    score = 0.0
    reversal_bigrams = {
        ('VBD', 'RP'): 0.8,
        ('VBN', 'RP'): 0.8,
        ('VBZ', 'RP'): 0.7,
        ('RP', 'IN'): 0.5,
        ('RP', 'TO'): 0.6,
        ('VBD', 'RB'): 0.5,
        ('RP', 'NN'): 0.4,
        ('RP', 'JJ'): 0.4,
    }

    for i in range(len(pos_tags) - 1):
        bigram = (pos_tags[i], pos_tags[i + 1])
        if bigram in reversal_bigrams:
            score = max(score, reversal_bigrams[bigram])

    # Standalone RP is a weaker signal
    if 'RP' in pos_tags and score == 0:
        score = 0.3

    return score


def detect_deletion_mechanism(pos_tags):
    """VBG + RP, or specific deletion patterns."""
    score = 0.0
    deletion_bigrams = {
        ('VBG', 'RP'): 0.7,
        ('JJ', 'TO'): 0.5,
        ('RB', 'VBG'): 0.4,
        ('VBG', 'DT'): 0.3,
    }

    for i in range(len(pos_tags) - 1):
        bigram = (pos_tags[i], pos_tags[i + 1])
        if bigram in deletion_bigrams:
            score = max(score, deletion_bigrams[bigram])

    return score


def detect_anagram_mechanism(pos_tags):
    """Modal + base verb, pronoun + past verb signal anagram."""
    score = 0.0
    anagram_bigrams = {
        ('MD', 'VB'): 0.7,
        ('PRP', 'VBD'): 0.6,
        ('PRP', 'VBP'): 0.5,
        ('NNS', 'VBP'): 0.5,
        ('NN', 'MD'): 0.5,
        ('VBP', 'JJ'): 0.4,
    }

    for i in range(len(pos_tags) - 1):
        bigram = (pos_tags[i], pos_tags[i + 1])
        if bigram in anagram_bigrams:
            score = max(score, anagram_bigrams[bigram])

    return score


# ============================================================
# Standalone type detectors
# ============================================================

def check_pure_anagram(wp_words, answer):
    """All wordplay letters (minus indicator words) anagram to answer."""
    all_letters = ''.join(c for w in wp_words for c in w.upper() if c.isalpha())
    return sorted(all_letters) == sorted(answer)


def check_pure_reversal(wp_words, answer, db):
    """A single synonym of one word, reversed, equals the full answer."""
    for word in wp_words:
        for val, src in db.get_values(word, len(answer)):
            if len(val) == len(answer) and val[::-1] == answer:
                return True
    return False


# ============================================================
# Main mechanism triage
# ============================================================

def triage_clue(wp_words, answer, pos_tags, db, ind_db):
    """Detect mechanisms and confirm with structural tests.

    Returns Evidence or None.
    """
    answer_len = len(answer)
    n = len(wp_words)
    total_letters = sum(len(re.sub(r'[^a-zA-Z]', '', w)) for w in wp_words)
    ratio = total_letters / answer_len if answer_len > 0 else 0

    # === Standalone checks ===

    # Pure anagram: letter budget ~1x and letters match
    if 0.8 <= ratio <= 2.5:
        ev = anagram_evidence(wp_words, answer, db, ind_db)
        if ev:
            return ev

    # Pure reversal
    if check_pure_reversal(wp_words, answer, db):
        ev = reversal_evidence(wp_words, answer, db, ind_db)
        if ev and ev.operation == 'reversal':
            return ev

    # === Mechanism detection via POS ===
    con_score = detect_container_mechanism(pos_tags)
    rev_score = detect_reversal_mechanism(pos_tags)
    del_score = detect_deletion_mechanism(pos_tags)
    ana_score = detect_anagram_mechanism(pos_tags)

    # Collect detected mechanisms, sorted by confidence
    mechanisms = []
    if con_score > 0.3:
        mechanisms.append(('container', con_score))
    if rev_score > 0.3:
        mechanisms.append(('reversal', rev_score))
    if del_score > 0.3:
        mechanisms.append(('deletion', del_score))
    if ana_score > 0.3:
        mechanisms.append(('anagram_charade', ana_score))

    mechanisms.sort(key=lambda x: -x[1])

    # === Confirm each mechanism with structural tests ===
    for mechanism, score in mechanisms:
        if mechanism == 'container':
            ev = container_evidence(wp_words, answer, db, ind_db)
            if ev:
                return ev
        elif mechanism == 'reversal':
            ev = reversal_evidence(wp_words, answer, db, ind_db)
            if ev:
                return ev
        elif mechanism == 'deletion':
            ev = deletion_evidence(wp_words, answer, db, ind_db)
            if ev:
                return ev
        elif mechanism == 'anagram_charade':
            ev = anagram_evidence(wp_words, answer, db, ind_db)
            if ev:
                return ev

    # === No mechanism detected or confirmed — try pure charade ===
    ev = charade_evidence(wp_words, answer, db, ind_db)
    if ev:
        return ev

    # === Last resort: try all structural tests regardless of POS ===
    # (POS may have missed the signal)
    for test_fn in [container_evidence, reversal_evidence, deletion_evidence]:
        ev = test_fn(wp_words, answer, db, ind_db)
        if ev:
            return ev

    return None


# ============================================================
# Test harness
# ============================================================

def run():
    print("Loading resources...")
    nlp = spacy.load('en_core_web_sm')
    ind_db = IndicatorDB()
    ref_db = QuickRefDB()

    conn = sqlite3.connect('data/word_roles.db', timeout=60)

    clues = conn.execute('''
        SELECT DISTINCT clue_id, operation, letter_budget_ratio
        FROM word_roles WHERE word_position = 0
    ''').fetchall()

    clue_data = {}
    for clue_id, op, ratio in clues:
        rows = conn.execute('''
            SELECT word_text, answer FROM word_roles
            WHERE clue_id=? ORDER BY word_position
        ''', (clue_id,)).fetchall()
        words = [r[0] for r in rows]
        answer = rows[0][1].upper().replace(' ', '').replace('-', '') if rows else ''
        clue_data[clue_id] = (words, answer)

    conn.close()

    print(f"Running mechanism triage on {len(clues)} clues...")
    t0 = time.time()

    results = []
    mechanism_detected = Counter()  # which mechanisms were detected by POS
    mechanism_confirmed = Counter()  # which were confirmed by structural test

    for i, (clue_id, actual_op, ratio) in enumerate(clues):
        if actual_op == 'hidden':
            continue

        if i % 2000 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(clues)} ({elapsed:.1f}s)")

        words, answer = clue_data[clue_id]

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

        ev = triage_clue(words, answer, pos_tags, ref_db, ind_db)

        if ev:
            results.append((clue_id, actual_op, ev.operation, True))
        else:
            results.append((clue_id, actual_op, 'UNCLASSIFIED', False))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # === Analysis ===
    total = len(results)
    with_ev = sum(1 for _, _, _, has in results if has)
    # For correctness: map all operation labels to the charade-with-mechanism model
    # container/container_charade -> charade is correct if we found container mechanism
    # reversal/reversal_charade -> charade is correct if we found reversal mechanism

    def is_correct(actual, predicted):
        if actual == predicted:
            return True
        # Container mechanism found in what's labeled container or container_charade
        if predicted == 'container' and actual in ('container', 'container_charade'):
            return True
        # Reversal mechanism found
        if predicted in ('reversal', 'reversal_charade') and actual in ('reversal', 'reversal_charade', 'reversal_container'):
            return True
        # Charade predicted for any charade-family type
        if predicted == 'charade' and actual in ('charade', 'container_charade', 'reversal_charade', 'anagram_charade'):
            return True
        return False

    correct_strict = sum(1 for _, a, p, _ in results if a == p)
    correct_mechanism = sum(1 for _, a, p, _ in results if is_correct(a, p))

    print(f"\n{'='*70}")
    print(f"MECHANISM TRIAGE RESULTS")
    print(f"{'='*70}")
    print(f"Total clues: {total}")
    print(f"With evidence: {with_ev} ({100*with_ev/total:.1f}%)")
    print(f"Correct (strict label match): {correct_strict} ({100*correct_strict/total:.1f}%)")
    print(f"Correct (mechanism match): {correct_mechanism} ({100*correct_mechanism/total:.1f}%)")
    print(f"UNCLASSIFIED: {total - with_ev} ({100*(total-with_ev)/total:.1f}%)")

    # By predicted
    print(f"\n--- By predicted operation ---")
    by_pred = defaultdict(lambda: Counter())
    for _, actual, pred, _ in results:
        by_pred[pred][actual] += 1

    print(f"{'Predicted':25s} {'Total':>6s} {'Strict':>7s} {'Mech':>7s} {'Prec(M)':>7s}")
    print("-" * 55)
    for pred in sorted(by_pred.keys(), key=lambda x: -sum(by_pred[x].values())):
        total_pred = sum(by_pred[pred].values())
        strict = by_pred[pred].get(pred, 0) if pred != 'UNCLASSIFIED' else 0
        mech = sum(cnt for a, cnt in by_pred[pred].items() if is_correct(a, pred))
        prec = 100 * mech / total_pred if total_pred else 0
        print(f"  {pred:23s} {total_pred:6d} {strict:7d} {mech:7d} {prec:6.1f}%")
        for actual, cnt in by_pred[pred].most_common(4):
            if not is_correct(actual, pred):
                pct = 100 * cnt / total_pred
                if pct >= 2:
                    print(f"    wrong: {actual:20s} {cnt:5d} ({pct:.1f}%)")


if __name__ == '__main__':
    run()
