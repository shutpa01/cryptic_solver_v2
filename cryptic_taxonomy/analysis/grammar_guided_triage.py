"""Grammar-guided triage: POS signature -> role sequence -> verify against answer.

Flow:
1. POS-tag the wordplay window
2. Look up the POS signature in the catalog
3. For each candidate role sequence:
   a. For FODDER words: look up synonyms/abbreviations
   b. For INDICATOR words: skip (they don't contribute letters)
   c. For LINK words: skip
   d. Check if fodder values concatenate to the answer
4. If verified, return evidence. If not, try next candidate.
5. Fall back to mechanism triage if no signature matches.

The catalog is built from the labeled dataset at startup.
"""

import sqlite3
import sys
import re
import spacy
import time
from collections import Counter, defaultdict
from typing import List, Tuple, Optional

sys.path.insert(0, '.')

from cryptic_taxonomy.analysis.evidence_triage import (
    QuickRefDB, WordRole, Evidence,
    container_evidence, reversal_evidence, deletion_evidence,
    anagram_evidence, charade_evidence,
)
from cryptic_taxonomy.analysis.mechanism_triage import (
    triage_clue as mechanism_triage_clue,
    detect_container_mechanism, detect_reversal_mechanism,
    detect_deletion_mechanism, detect_anagram_mechanism,
    check_pure_anagram, check_pure_reversal,
)
from cryptic_taxonomy.analysis.mine_positional_signatures import IndicatorDB
from signature_solver.tokens import LINK_WORDS


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")


MID_MAP = {
    'NN': 'N', 'NNS': 'N', 'NNP': 'NP', 'NNPS': 'NP',
    'VB': 'Vb', 'VBD': 'Vi', 'VBG': 'Vi', 'VBN': 'Vi',
    'VBP': 'Vb', 'VBZ': 'Vi',
    'JJ': 'J', 'JJR': 'J', 'JJS': 'J',
    'RB': 'R', 'RBR': 'R', 'RBS': 'R',
    'IN': 'P', 'TO': 'P',
    'DT': 'D', 'WDT': 'D', 'PDT': 'D',
    'CC': 'C', 'RP': 'RP', 'CD': 'CD',
    'PRP': 'PR', 'PRP$': 'PR', 'WP': 'PR',
    'MD': 'MD', 'UH': 'X', 'FW': 'X', 'EX': 'X', 'POS': 'X',
}

FODDER_ROLES = {'SYN_F', 'ABR_F', 'RAW', 'ANA_F', 'HID_F', 'POS_F'}
INDICATOR_ROLES = {'ANA_I', 'CON_I', 'REV_I', 'DEL_I', 'HID_I', 'HOM_I'}


def build_signature_catalog():
    """Build POS signature -> role sequence catalog from labeled data."""
    print("Building signature catalog...")
    nlp = spacy.load('en_core_web_sm')
    conn = sqlite3.connect('data/word_roles.db', timeout=60)

    clue_ids = conn.execute(
        'SELECT DISTINCT clue_id FROM word_roles WHERE word_position = 0'
    ).fetchall()

    catalog = defaultdict(lambda: Counter())

    for i, (clue_id,) in enumerate(clue_ids):
        if i % 3000 == 0 and i > 0:
            print(f"  {i}/{len(clue_ids)}...")

        rows = conn.execute('''
            SELECT word_text, assigned_role
            FROM word_roles WHERE clue_id=? ORDER BY word_position
        ''', (clue_id,)).fetchall()

        words = [r[0] for r in rows]
        roles = tuple(r[1] for r in rows)

        doc = nlp(' '.join(words))
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

        mid_seq = tuple(MID_MAP.get(t, 'X') for t in pos_tags)
        catalog[mid_seq][roles] += 1

    conn.close()

    # Filter: only keep role sequences with n >= 3
    filtered = {}
    for pos_seq, role_counts in catalog.items():
        candidates = [(roles, cnt) for roles, cnt in role_counts.most_common(5)
                      if cnt >= 3]
        if candidates:
            filtered[pos_seq] = candidates

    print(f"  Catalog: {len(filtered)} POS signatures with viable role sequences")
    return filtered


def verify_role_assignment(wp_words, roles, answer, db):
    """Given a role assignment, look up values for fodder words and check
    if they concatenate to the answer.

    Returns Evidence or None.
    """
    n = len(wp_words)
    answer_len = len(answer)

    # Collect fodder words and their candidate values
    fodder_positions = []  # (word_index, role, candidates)
    for i in range(n):
        role = roles[i]
        if role in FODDER_ROLES:
            word = wp_words[i]
            candidates = []

            if role in ('SYN_F', 'ABR_F'):
                # Single word lookup
                for val, src in db.get_values(word, answer_len):
                    if val in answer:
                        candidates.append((val, src))
                # Raw letters
                raw = ''.join(c for c in word.upper() if c.isalpha())
                if raw in answer:
                    candidates.append((raw, 'raw'))
                # Positional: first/last letter
                if raw:
                    if raw[0] in answer:
                        candidates.append((raw[0], 'first_letter'))
                    if len(raw) >= 2 and raw[-1] in answer:
                        candidates.append((raw[-1], 'last_letter'))
                # Reversal
                for val, src in db.get_values(word, answer_len):
                    if len(val) >= 2 and val[::-1] in answer and val[::-1] != val:
                        candidates.append((val[::-1], 'reversal'))

                # Multi-word phrase: try with next word
                if i < n - 1 and roles[i + 1] in FODDER_ROLES:
                    for val, src in db.get_phrase_values(wp_words[i:i+2], answer_len):
                        if val in answer:
                            candidates.append((val, src + '_phrase'))

            elif role == 'ANA_F':
                raw = ''.join(c for c in word.upper() if c.isalpha())
                if raw:
                    candidates.append((raw, 'anagram_fodder'))

            if not candidates:
                # This fodder word has no viable values — assignment fails
                return None

            fodder_positions.append((i, role, candidates))

    if not fodder_positions:
        return None

    # Build position index and search (same approach as charade_evidence)
    pos_candidates = {}
    for word_idx, role, candidates in fodder_positions:
        for val, src in candidates:
            for pos in range(answer_len - len(val) + 1):
                if answer[pos:pos + len(val)] == val:
                    pos_candidates.setdefault(pos, []).append(
                        (word_idx, val, src))

    best = [None]
    fodder_indices = {fp[0] for fp in fodder_positions}

    def search(pos, assignments, used):
        if pos == answer_len:
            best[0] = list(assignments)
            return
        if best[0] is not None:
            return
        if pos not in pos_candidates:
            return
        for word_idx, val, src in pos_candidates[pos]:
            if word_idx in used:
                continue
            assignments.append((word_idx, val, src))
            used.add(word_idx)
            search(pos + len(val), assignments, used)
            if best[0] is not None:
                return
            assignments.pop()
            used.discard(word_idx)

    search(0, [], set())

    if best[0] is None:
        return None

    # Build evidence
    assignments = best[0]
    assigned = {a[0]: (a[1], a[2]) for a in assignments}

    word_roles = []
    for i in range(n):
        if i in assigned:
            val, src = assigned[i]
            if 'phrase' in src:
                role_name = 'SYN_F'
            elif src == 'first_letter' or src == 'last_letter':
                role_name = 'POS_F'
            elif src == 'reversal':
                role_name = 'REV_F'
            elif src == 'abbreviation':
                role_name = 'ABR_F'
            elif src == 'synonym':
                role_name = 'SYN_F'
            else:
                role_name = 'RAW'
            word_roles.append(WordRole(i, wp_words[i], role_name, val, src))
        elif roles[i] in INDICATOR_ROLES:
            word_roles.append(WordRole(i, wp_words[i], roles[i], None))
        else:
            word_roles.append(WordRole(i, wp_words[i], 'LNK', None))

    pieces = ' + '.join(a[1] for a in assignments)
    return Evidence(
        operation='charade',
        word_roles=word_roles,
        assembly=f'{pieces} = {answer}',
        answer=answer,
        confidence='high',  # grammar-guided + answer-verified
    )


def triage_clue(wp_words, answer, pos_tags, catalog, db, ind_db):
    """Grammar-guided triage with mechanism fallback."""
    answer_len = len(answer)
    n = len(wp_words)
    total_letters = sum(len(re.sub(r'[^a-zA-Z]', '', w)) for w in wp_words)
    ratio = total_letters / answer_len if answer_len > 0 else 0

    # === Standalone checks ===
    if 0.8 <= ratio <= 2.5:
        ev = anagram_evidence(wp_words, answer, db, ind_db)
        if ev:
            return ev, 'anagram'

    if check_pure_reversal(wp_words, answer, db):
        ev = reversal_evidence(wp_words, answer, db, ind_db)
        if ev and ev.operation == 'reversal':
            return ev, 'reversal'

    # === Grammar signature lookup ===
    mid_seq = tuple(MID_MAP.get(t, 'X') for t in pos_tags)

    if mid_seq in catalog:
        for role_seq, count in catalog[mid_seq]:
            ev = verify_role_assignment(wp_words, role_seq, answer, db)
            if ev:
                return ev, 'grammar'

    # === Fall back to mechanism triage ===
    ev = mechanism_triage_clue(wp_words, answer, pos_tags, db, ind_db)
    if ev:
        return ev, 'mechanism'

    return None, 'unclassified'


def run():
    print("Loading resources...")
    nlp = spacy.load('en_core_web_sm')
    ind_db = IndicatorDB()
    ref_db = QuickRefDB()

    catalog = build_signature_catalog()

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

    print(f"\nRunning grammar-guided triage on {len(clues)} clues...")
    t0 = time.time()

    results = []
    source_counts = Counter()

    for i, (clue_id, actual_op, ratio) in enumerate(clues):
        if actual_op == 'hidden':
            continue

        if i % 2000 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(clues)} ({elapsed:.1f}s)")

        words, answer = clue_data[clue_id]

        doc = nlp(' '.join(words))
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

        ev, source = triage_clue(words, answer, pos_tags, catalog, ref_db, ind_db)
        source_counts[source] += 1

        if ev:
            results.append((clue_id, actual_op, ev.operation, True, source))
        else:
            results.append((clue_id, actual_op, 'UNCLASSIFIED', False, source))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # === Analysis ===
    total = len(results)
    with_ev = sum(1 for r in results if r[3])

    def is_correct(actual, predicted):
        if actual == predicted:
            return True
        if predicted == 'container' and actual in ('container', 'container_charade'):
            return True
        if predicted in ('reversal', 'reversal_charade') and actual in ('reversal', 'reversal_charade', 'reversal_container'):
            return True
        if predicted == 'charade' and actual in ('charade', 'container_charade', 'reversal_charade', 'anagram_charade'):
            return True
        return False

    correct_strict = sum(1 for r in results if r[1] == r[2])
    correct_mech = sum(1 for r in results if is_correct(r[1], r[2]))

    print(f"\n{'='*70}")
    print(f"GRAMMAR-GUIDED TRIAGE RESULTS")
    print(f"{'='*70}")
    print(f"Total clues: {total}")
    print(f"With evidence: {with_ev} ({100*with_ev/total:.1f}%)")
    print(f"Correct (strict): {correct_strict} ({100*correct_strict/total:.1f}%)")
    print(f"Correct (mechanism match): {correct_mech} ({100*correct_mech/total:.1f}%)")
    print(f"UNCLASSIFIED: {total - with_ev} ({100*(total-with_ev)/total:.1f}%)")

    print(f"\n--- By source ---")
    for source, cnt in source_counts.most_common():
        correct_src = sum(1 for r in results if r[4] == source and is_correct(r[1], r[2]))
        total_src = sum(1 for r in results if r[4] == source)
        prec = 100 * correct_src / total_src if total_src else 0
        print(f"  {source:15s} {cnt:6d} ({100*cnt/total:.1f}%)  precision: {prec:.1f}%")

    # By predicted operation
    print(f"\n--- By predicted operation ---")
    by_pred = defaultdict(lambda: Counter())
    for r in results:
        by_pred[r[2]][r[1]] += 1

    print(f"{'Predicted':25s} {'Total':>6s} {'Mech':>7s} {'Prec':>7s}")
    print("-" * 50)
    for pred in sorted(by_pred.keys(), key=lambda x: -sum(by_pred[x].values())):
        total_pred = sum(by_pred[pred].values())
        mech = sum(cnt for a, cnt in by_pred[pred].items() if is_correct(a, pred))
        prec = 100 * mech / total_pred if total_pred else 0
        print(f"  {pred:23s} {total_pred:6d} {mech:7d} {prec:6.1f}%")


if __name__ == '__main__':
    run()
