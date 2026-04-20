"""Step 4: Level 2 structural tests — use the answer to identify operations.

For clues not resolved by Level 1 (anagram/pure charade/hidden), run
structural tests against the known answer:

1. Container test: can answer = prefix + inner + suffix, where
   prefix+suffix is a synonym of one clue word and inner is a synonym
   of another?
2. Reversal test: does any clue-word synonym, reversed, appear as a
   substring of the answer?
3. Deletion test: is any clue-word synonym 1-2 chars longer than the
   answer, and removing chars produces the answer?
4. Homophone test: does any clue-word homophone match the answer?

Runs the full hierarchical triage (Level 1 + Level 2) on all clues
and measures accuracy.
"""

import sqlite3
import sys
import re
import time
from collections import Counter, defaultdict

sys.path.insert(0, '.')

from cryptic_taxonomy.analysis.mine_positional_signatures import IndicatorDB
from signature_solver.tokens import LINK_WORDS


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")


class QuickRefDB:
    """Lightweight synonym/abbreviation/homophone lookups for structural tests."""

    def __init__(self):
        conn = sqlite3.connect('data/cryptic_new.db', timeout=30)

        # word -> list of synonym values (uppercase)
        self.synonyms = {}
        for word, syn in conn.execute("SELECT word, synonym FROM synonyms_pairs"):
            w = word.lower().strip()
            self.synonyms.setdefault(w, []).append(syn.strip().upper())
        for defn, ans in conn.execute(
            "SELECT definition, answer FROM definition_answers_augmented "
            "WHERE definition IS NOT NULL AND answer IS NOT NULL"
        ):
            w = defn.lower().strip()
            val = ans.strip().upper()
            if w and val:
                self.synonyms.setdefault(w, []).append(val)

        # word -> list of abbreviation values (uppercase)
        self.abbreviations = {}
        for ind, sub in conn.execute("SELECT indicator, substitution FROM wordplay"):
            w = ind.lower().strip()
            val = sub.strip().upper()
            if val:
                self.abbreviations.setdefault(w, []).append(val)

        # word -> list of homophones (uppercase)
        self.homophones = {}
        for word, hom in conn.execute("SELECT word, homophone FROM homophones"):
            w = word.lower().strip()
            self.homophones.setdefault(w, []).append(hom.strip().upper())

        conn.close()

    def get_values(self, word, answer_len=None):
        """Get all possible letter contributions for a clue word (syns + abbrs)."""
        w = _clean(word)
        variants = [w]
        if len(w) >= 4 and w.endswith('s') and not w.endswith('ss'):
            variants.append(w[:-1])

        vals = set()
        for v in variants:
            for s in self.synonyms.get(v, []):
                if answer_len is None or len(s) <= answer_len + 2:
                    vals.add(s)
            for a in self.abbreviations.get(v, []):
                vals.add(a)
        return vals

    def get_synonyms(self, word, max_len=None):
        w = _clean(word)
        variants = [w]
        if len(w) >= 4 and w.endswith('s') and not w.endswith('ss'):
            variants.append(w[:-1])
        result = []
        for v in variants:
            for s in self.synonyms.get(v, []):
                if max_len is None or len(s) <= max_len:
                    if s not in result:
                        result.append(s)
        return result

    def get_homophones(self, word):
        w = _clean(word)
        return self.homophones.get(w, [])


def container_test(wp_words, answer, db):
    """Can the answer be decomposed as outer containing inner,
    where outer and inner are values of different clue words?"""
    answer_len = len(answer)

    # Get all possible values per word
    word_vals = []
    for word in wp_words:
        word_vals.append(db.get_values(word, answer_len))

    # Try each pair of words as outer/inner source
    for i in range(len(wp_words)):
        for j in range(len(wp_words)):
            if i == j:
                continue
            for outer in word_vals[i]:
                if len(outer) >= answer_len or len(outer) < 2:
                    continue
                inner_len = answer_len - len(outer)
                if inner_len < 1:
                    continue
                # Try inserting at each position in outer
                for pos in range(1, len(outer)):
                    prefix = outer[:pos]
                    suffix = outer[pos:]
                    if answer.startswith(prefix) and answer.endswith(suffix):
                        inner_needed = answer[pos:pos + inner_len]
                        if inner_needed in word_vals[j]:
                            return True
    return False


def reversal_test(wp_words, answer, db):
    """Does any clue-word synonym, reversed, appear as a substring of the answer?
    Must be at least 3 chars to be meaningful."""
    for word in wp_words:
        for val in db.get_values(word, len(answer)):
            if len(val) >= 3 and val[::-1] in answer:
                return True
    return False


def deletion_test(wp_words, answer, db):
    """Is any clue-word synonym 1-2 chars longer than the answer,
    and removing 1-2 chars from it produces the answer?"""
    for word in wp_words:
        for syn in db.get_synonyms(word, max_len=len(answer) + 2):
            if len(syn) == len(answer) + 1:
                # Try removing each character
                for k in range(len(syn)):
                    if syn[:k] + syn[k+1:] == answer:
                        return True
            elif len(syn) == len(answer) + 2:
                # Try removing first or last char, or two specific chars
                if syn[1:] == answer or syn[:-1] == answer:
                    return True
                if syn[1:-1] == answer:
                    return True
    return False


def homophone_test(wp_words, answer, db):
    """Does any clue word have a homophone that equals the answer?
    Also checks synonym->homophone chain."""
    for word in wp_words:
        # Direct homophone
        for h in db.get_homophones(word):
            if h == answer:
                return True
        # Synonym -> homophone
        for syn in db.get_synonyms(word, max_len=len(answer) + 3):
            for h in db.get_homophones(syn.lower()):
                if h == answer:
                    return True
    return False


def anagram_charade_test(wp_words, answer, db):
    """Is there a subset of consecutive wordplay words whose letters anagram
    to a substring of the answer, with the remaining words providing the
    rest via synonym/abbreviation?

    Also covers pure anagram at higher budgets: all wordplay letters
    anagram to the full answer.
    """
    answer_len = len(answer)

    # Get raw letters per word
    word_letters = []
    for w in wp_words:
        word_letters.append(''.join(c for c in w.upper() if c.isalpha()))

    all_letters = ''.join(word_letters)

    # Check pure anagram of all words (for high budget anagrams missed by L1)
    if sorted(all_letters) == sorted(answer):
        return True

    n = len(wp_words)
    if n < 2:
        return False

    # Try each contiguous span of words as anagram fodder
    for start in range(n):
        for end in range(start + 1, n + 1):
            fodder = ''.join(word_letters[start:end])
            fodder_len = len(fodder)

            # Fodder must be shorter than answer (rest comes from synonyms)
            if fodder_len >= answer_len or fodder_len < 3:
                continue

            remaining_needed = answer_len - fodder_len

            # The remaining (non-fodder) words must provide exactly
            # remaining_needed letters via synonym/abbreviation
            remaining_words = wp_words[:start] + wp_words[end:]
            if not remaining_words:
                continue

            # Get possible values for remaining words
            remaining_vals = []
            for w in remaining_words:
                vals = db.get_values(w, answer_len)
                # Filter to plausible lengths
                plausible = [v for v in vals if 1 <= len(v) <= remaining_needed]
                if not plausible:
                    # This word might be a link/indicator — skip it
                    remaining_vals.append(None)
                else:
                    remaining_vals.append(plausible)

            # Quick check: can remaining values sum to remaining_needed?
            # Try single remaining word providing all remaining letters
            for rv in remaining_vals:
                if rv is None:
                    continue
                for val in rv:
                    if len(val) == remaining_needed:
                        # Check: does fodder anagram to the gap?
                        # answer = val_piece + anagram_piece (in some order)
                        # Find where val fits in the answer
                        for pos in range(answer_len - len(val) + 1):
                            if answer[pos:pos + len(val)] == val:
                                gap = answer[:pos] + answer[pos + len(val):]
                                if sorted(gap) == sorted(fodder):
                                    return True

            # Try two remaining words providing letters
            non_none = [(i, rv) for i, rv in enumerate(remaining_vals) if rv is not None]
            if len(non_none) >= 2 and len(non_none) <= 3:
                for ai, a_vals in non_none:
                    for bi, b_vals in non_none:
                        if ai >= bi:
                            continue
                        for av in a_vals[:5]:  # cap to avoid explosion
                            for bv in b_vals[:5]:
                                if len(av) + len(bv) == remaining_needed:
                                    combined = av + bv
                                    # Try both orderings in the answer
                                    for piece in [av + bv, bv + av]:
                                        for pos in range(answer_len - len(piece) + 1):
                                            if answer[pos:pos + len(piece)] == piece:
                                                gap = answer[:pos] + answer[pos + len(piece):]
                                                if sorted(gap) == sorted(fodder):
                                                    return True

    return False


def run_triage():
    print("Loading databases...")
    ind_db = IndicatorDB()
    ref_db = QuickRefDB()

    conn = sqlite3.connect('data/word_roles.db', timeout=60)

    # Get one row per clue with its features
    clues = conn.execute('''
        SELECT DISTINCT clue_id, operation, letter_budget_ratio
        FROM word_roles WHERE word_position = 0
    ''').fetchall()

    # Get words and indicator types per clue
    clue_words = {}
    clue_indicators = {}
    clue_answers = {}
    for clue_id, op, ratio in clues:
        rows = conn.execute('''
            SELECT word_text, has_indicator_types, answer
            FROM word_roles WHERE clue_id=?
            ORDER BY word_position
        ''', (clue_id,)).fetchall()

        words = [r[0] for r in rows]
        answer = rows[0][2].upper().replace(' ', '').replace('-', '') if rows else ''

        ind_types = set()
        for _, types_str, _ in rows:
            if types_str:
                for t in types_str.split(','):
                    ind_types.add(t.strip())

        clue_words[clue_id] = words
        clue_indicators[clue_id] = ind_types
        clue_answers[clue_id] = answer

    conn.close()

    print(f"Running triage on {len(clues)} clues...")
    t0 = time.time()

    results = []  # (clue_id, actual_op, predicted_op, level)

    level1_count = 0
    level2_count = 0

    for i, (clue_id, actual_op, ratio) in enumerate(clues):
        if i % 2000 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(clues)} ({elapsed:.1f}s)")

        words = clue_words[clue_id]
        answer = clue_answers[clue_id]
        ind_types = clue_indicators[clue_id]

        has_ana = 'anagram' in ind_types
        has_rev = 'reversal' in ind_types
        has_con = 'container' in ind_types or 'insertion' in ind_types
        has_del = 'deletion' in ind_types
        has_hid = 'hidden' in ind_types
        has_hom = 'homophone' in ind_types
        has_any = has_ana or has_rev or has_con or has_del or has_hid or has_hom

        # === LEVEL 1 ===

        # Anagram: budget 1.0-2.2 AND anagram indicator
        if has_ana and 1.0 <= ratio <= 2.2:
            results.append((clue_id, actual_op, 'anagram', 1))
            level1_count += 1
            continue

        # Pure charade: no indicators at all
        if not has_any:
            results.append((clue_id, actual_op, 'charade', 1))
            level1_count += 1
            continue

        # Hidden: already solved pre-S, skip these clues entirely
        if actual_op == 'hidden':
            continue

        # === LEVEL 2: Structural tests ===
        # Every classification requires POSITIVE evidence:
        # indicator present AND structural test passes.
        # If nothing matches, the clue is UNCLASSIFIED.
        level2_count += 1
        predicted = None

        # Run ALL structural tests — collect which ones pass
        passes_container = has_con and container_test(words, answer, ref_db)
        passes_reversal = has_rev and reversal_test(words, answer, ref_db)
        passes_deletion = has_del and deletion_test(words, answer, ref_db)
        passes_homophone = has_hom and homophone_test(words, answer, ref_db)

        # Anagram_charade: ANA_I present AND structural test confirms
        # a subset of wordplay letters actually anagram to part of the answer
        passes_anagram_charade = has_ana and anagram_charade_test(words, answer, ref_db)

        # Pure charade with indicators present: ALL structural tests fail
        # but synonym/abbreviation values concatenate to the answer.
        # This is the "indicator words are red herrings" case.
        # Positive rule: no structural test passed, AND the letter budget
        # is high (>2.5, meaning synonyms must be expanding the letters)
        passes_charade = (not passes_container and not passes_reversal
                          and not passes_deletion and not passes_homophone
                          and not passes_anagram_charade and ratio > 2.5)

        # Classify by most specific positive match
        # Container is most precise (93.5% in first run)
        if passes_container:
            predicted = 'container'
        elif passes_homophone:
            predicted = 'homophone'
        elif passes_deletion:
            predicted = 'del'
        elif passes_reversal:
            predicted = 'reversal_charade'
        elif passes_anagram_charade:
            predicted = 'anagram_charade'
        elif passes_charade:
            predicted = 'charade'
        else:
            predicted = 'UNCLASSIFIED'

        results.append((clue_id, actual_op, predicted, 2))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # === ANALYSIS ===
    print("\n" + "=" * 80)
    print("HIERARCHICAL TRIAGE RESULTS")
    print("=" * 80)

    total = len(results)
    correct = sum(1 for _, actual, pred, _ in results if actual == pred)
    l1_results = [(a, p) for _, a, p, l in results if l == 1]
    l2_results = [(a, p) for _, a, p, l in results if l == 2]

    l1_correct = sum(1 for a, p in l1_results if a == p)
    l2_correct = sum(1 for a, p in l2_results if a == p)

    print(f"\nOverall: {correct}/{total} = {100*correct/total:.1f}% correct")
    print(f"Level 1: {l1_correct}/{len(l1_results)} = {100*l1_correct/len(l1_results):.1f}% correct ({len(l1_results)} clues, {100*len(l1_results)/total:.0f}% of total)")
    print(f"Level 2: {l2_correct}/{len(l2_results)} = {100*l2_correct/len(l2_results):.1f}% correct ({len(l2_results)} clues, {100*len(l2_results)/total:.0f}% of total)")

    # Confusion: what did we predict vs actual?
    print(f"\n--- Level 1 breakdown ---")
    l1_by_pred = defaultdict(lambda: Counter())
    for actual, pred in l1_results:
        l1_by_pred[pred][actual] += 1
    for pred in sorted(l1_by_pred.keys()):
        total_pred = sum(l1_by_pred[pred].values())
        correct_pred = l1_by_pred[pred].get(pred, 0)
        acc = 100 * correct_pred / total_pred if total_pred else 0
        print(f"\n  Predicted '{pred}' ({total_pred} clues, {acc:.1f}% correct):")
        for actual, cnt in l1_by_pred[pred].most_common(5):
            print(f"    actually {actual:25s} {cnt:5d} ({100*cnt/total_pred:.1f}%)")

    print(f"\n--- Level 2 breakdown ---")
    l2_by_pred = defaultdict(lambda: Counter())
    for actual, pred in l2_results:
        l2_by_pred[pred][actual] += 1
    for pred in sorted(l2_by_pred.keys()):
        total_pred = sum(l2_by_pred[pred].values())
        correct_pred = l2_by_pred[pred].get(pred, 0)
        acc = 100 * correct_pred / total_pred if total_pred else 0
        print(f"\n  Predicted '{pred}' ({total_pred} clues, {acc:.1f}% correct):")
        for actual, cnt in l2_by_pred[pred].most_common(5):
            print(f"    actually {actual:25s} {cnt:5d} ({100*cnt/total_pred:.1f}%)")

    # Per-operation recall: how often do we correctly identify each type?
    print(f"\n--- Per-operation recall ---")
    print(f"{'Operation':25s} {'Correct':>7s} {'Total':>7s} {'Recall':>7s}")
    print("-" * 50)
    by_actual = defaultdict(lambda: {'correct': 0, 'total': 0})
    for _, actual, pred, _ in results:
        by_actual[actual]['total'] += 1
        if actual == pred:
            by_actual[actual]['correct'] += 1
    for op in sorted(by_actual.keys(), key=lambda x: -by_actual[x]['total']):
        c = by_actual[op]['correct']
        t = by_actual[op]['total']
        print(f"  {op:23s} {c:7d} {t:7d} {100*c/t:6.1f}%")

    # What operations does Level 2 miss most?
    print(f"\n--- Level 2 misses (actual op -> what we predicted instead) ---")
    l2_misses = defaultdict(lambda: Counter())
    for actual, pred in l2_results:
        if actual != pred:
            l2_misses[actual][pred] += 1
    for actual in sorted(l2_misses.keys(), key=lambda x: -sum(l2_misses[x].values())):
        total_missed = sum(l2_misses[actual].values())
        total_actual = sum(1 for a, _ in l2_results if a == actual)
        print(f"\n  {actual} ({total_missed} missed of {total_actual}):")
        for pred, cnt in l2_misses[actual].most_common(3):
            print(f"    -> predicted {pred:25s} {cnt:5d}")


if __name__ == '__main__':
    run_triage()
