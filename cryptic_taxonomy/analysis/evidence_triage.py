"""Step 5: Evidence-returning triage — structural tests return word-level
assignments, not just boolean pass/fail.

Each test returns an Evidence object containing:
  - operation: the identified operation type
  - word_roles: list of (word_index, word_text, role, value)
      role is one of: SYN_F, ABR_F, ANA_F, RAW, CON_I, REV_I, ANA_I, DEL_I, HOM_I, LNK
      value is the letters contributed (or None for indicators/links)
  - assembly: how the values produce the answer (for verification)

This evidence can be handed directly to the solver's verification and
explanation-building logic.
"""

import sqlite3
import sys
import re
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

sys.path.insert(0, '.')

from cryptic_taxonomy.analysis.mine_positional_signatures import IndicatorDB
from signature_solver.tokens import LINK_WORDS


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")


@dataclass
class WordRole:
    """Role assignment for a single wordplay word."""
    word_index: int
    word_text: str
    role: str           # SYN_F, ABR_F, ANA_F, RAW, CON_I, REV_I, ANA_I, DEL_I, HOM_I, LNK
    value: Optional[str]  # letters contributed (None for indicators/links)
    source_type: str = ''  # 'synonym', 'abbreviation', 'raw', 'homophone'


@dataclass
class Evidence:
    """Complete evidence for a triage classification."""
    operation: str
    word_roles: List[WordRole]
    assembly: str       # human-readable: "CAVE containing L = CALVE"
    answer: str
    confidence: str     # 'high', 'medium', 'low'

    def verify(self):
        """Check that the assigned values actually produce the answer."""
        values = [wr.value for wr in self.word_roles if wr.value]
        if self.operation == 'charade':
            return ''.join(values) == self.answer
        # Other operations verified by their specific test
        return True

    def summary(self):
        parts = []
        for wr in self.word_roles:
            if wr.value:
                parts.append(f'{wr.word_text}={wr.value}({wr.role})')
            else:
                parts.append(f'{wr.word_text}({wr.role})')
        return f"[{self.operation}] {' + '.join(parts)} -> {self.answer}"


class QuickRefDB:
    """Lightweight synonym/abbreviation/homophone lookups."""

    def __init__(self):
        conn = sqlite3.connect('data/cryptic_new.db', timeout=30)

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

        self.abbreviations = {}
        for ind, sub in conn.execute("SELECT indicator, substitution FROM wordplay"):
            w = ind.lower().strip()
            val = sub.strip().upper()
            if val:
                self.abbreviations.setdefault(w, []).append(val)

        self.homophones = {}
        for word, hom in conn.execute("SELECT word, homophone FROM homophones"):
            w = word.lower().strip()
            self.homophones.setdefault(w, []).append(hom.strip().upper())

        conn.close()

    def _variants(self, word):
        w = _clean(word)
        variants = [w]
        if len(w) >= 4 and w.endswith('s') and not w.endswith('ss'):
            variants.append(w[:-1])
        return variants

    def get_values(self, word, answer_len=None):
        """Get all (value, source_type) pairs for a clue word."""
        results = []
        for v in self._variants(word):
            for s in self.synonyms.get(v, []):
                if answer_len is None or len(s) <= answer_len + 2:
                    results.append((s, 'synonym'))
            for a in self.abbreviations.get(v, []):
                results.append((a, 'abbreviation'))
        return results

    def get_values_set(self, word, answer_len=None):
        """Get just the value strings as a set."""
        return {v for v, _ in self.get_values(word, answer_len)}

    def get_synonyms(self, word, max_len=None):
        result = []
        for v in self._variants(word):
            for s in self.synonyms.get(v, []):
                if max_len is None or len(s) <= max_len:
                    if s not in result:
                        result.append(s)
        return result

    def get_homophones(self, word):
        w = _clean(word)
        return self.homophones.get(w, [])

    def lookup_value(self, word, target_value):
        """Find what source_type produces target_value for this word.
        Returns 'synonym', 'abbreviation', or None."""
        for v in self._variants(word):
            if target_value in self.abbreviations.get(v, []):
                return 'abbreviation'
            if target_value in self.synonyms.get(v, []):
                return 'synonym'
        return None

    def get_phrase_values(self, words, answer_len=None):
        """Get (value, source_type) pairs for a multi-word phrase."""
        phrase = ' '.join(_clean(w) for w in words)
        results = []
        for s in self.synonyms.get(phrase, []):
            if answer_len is None or len(s) <= answer_len + 2:
                results.append((s, 'synonym'))
        for a in self.abbreviations.get(phrase, []):
            results.append((a, 'abbreviation'))
        return results


# ============================================================
# Evidence-returning structural tests
# ============================================================

def container_evidence(wp_words, answer, db, ind_db):
    """Find container evidence: outer(inner) = answer.
    Returns Evidence or None."""
    answer_len = len(answer)
    n = len(wp_words)

    # Get all possible values per word with source types
    word_vals = []
    for word in wp_words:
        word_vals.append(db.get_values(word, answer_len))

    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for outer, outer_src in word_vals[i]:
                if len(outer) >= answer_len or len(outer) < 2:
                    continue
                inner_len = answer_len - len(outer)
                if inner_len < 1:
                    continue

                for pos in range(1, len(outer)):
                    prefix = outer[:pos]
                    suffix = outer[pos:]
                    if answer.startswith(prefix) and answer.endswith(suffix):
                        inner_needed = answer[pos:pos + inner_len]
                        # Check if any other word produces inner_needed
                        for inner_val, inner_src in word_vals[j]:
                            if inner_val == inner_needed:
                                # Found it. Now classify remaining words.
                                roles = []
                                outer_role = 'SYN_F' if outer_src == 'synonym' else 'ABR_F'
                                inner_role = 'SYN_F' if inner_src == 'synonym' else 'ABR_F'

                                for k in range(n):
                                    if k == i:
                                        roles.append(WordRole(k, wp_words[k], outer_role, outer, outer_src))
                                    elif k == j:
                                        roles.append(WordRole(k, wp_words[k], inner_role, inner_needed, inner_src))
                                    else:
                                        # Check if this word is a container indicator
                                        ind_types = ind_db.get_indicator_types(wp_words[k])
                                        if ind_types & {'container', 'insertion'}:
                                            roles.append(WordRole(k, wp_words[k], 'CON_I', None))
                                        else:
                                            roles.append(WordRole(k, wp_words[k], 'LNK', None))

                                assembly = f'{outer}[{prefix}({inner_needed}){suffix}] = {answer}'
                                return Evidence(
                                    operation='container',
                                    word_roles=roles,
                                    assembly=assembly,
                                    answer=answer,
                                    confidence='high',
                                )
    return None


def reversal_evidence(wp_words, answer, db, ind_db):
    """Find reversal evidence: reversed synonym is substring of answer.
    Returns Evidence or None."""
    answer_len = len(answer)
    n = len(wp_words)

    for i in range(n):
        for val, src in db.get_values(wp_words[i], answer_len):
            if len(val) < 3:
                continue
            rev = val[::-1]
            if rev in answer:
                # This word contributes reversed letters
                # Find what provides the remaining letters
                remaining_needed = answer.replace(rev, '', 1)
                rev_start = answer.index(rev)
                before = answer[:rev_start]
                after = answer[rev_start + len(rev):]

                # Try to fill before and after from other words
                other_words = [(k, wp_words[k]) for k in range(n) if k != i]

                # Simple case: reversed piece IS the whole answer
                if rev == answer:
                    roles = []
                    val_role = 'SYN_F' if src == 'synonym' else 'ABR_F'
                    for k in range(n):
                        if k == i:
                            roles.append(WordRole(k, wp_words[k], val_role, val, src))
                        else:
                            ind_types = ind_db.get_indicator_types(wp_words[k])
                            if ind_types & {'reversal'}:
                                roles.append(WordRole(k, wp_words[k], 'REV_I', None))
                            else:
                                roles.append(WordRole(k, wp_words[k], 'LNK', None))
                    return Evidence(
                        operation='reversal',
                        word_roles=roles,
                        assembly=f'reverse({val}) = {answer}',
                        answer=answer,
                        confidence='high',
                    )

                # Reversal_charade: reversed piece + other pieces = answer
                if not before and not after:
                    continue

                # Try single other word providing the remaining piece(s)
                for k, other_word in other_words:
                    other_vals = db.get_values_set(other_word, answer_len)
                    if before and after:
                        # Need two pieces — skip for now (complex)
                        continue
                    elif before and before in other_vals:
                        roles = _build_reversal_charade_roles(
                            n, wp_words, i, val, src, k, before,
                            db.lookup_value(other_word, before),
                            ind_db, answer
                        )
                        if roles:
                            return Evidence(
                                operation='reversal_charade',
                                word_roles=roles,
                                assembly=f'{before} + reverse({val}) = {answer}',
                                answer=answer,
                                confidence='medium',
                            )
                    elif after and after in other_vals:
                        roles = _build_reversal_charade_roles(
                            n, wp_words, i, val, src, k, after,
                            db.lookup_value(other_word, after),
                            ind_db, answer
                        )
                        if roles:
                            return Evidence(
                                operation='reversal_charade',
                                word_roles=roles,
                                assembly=f'reverse({val}) + {after} = {answer}',
                                answer=answer,
                                confidence='medium',
                            )
    return None


def _build_reversal_charade_roles(n, wp_words, rev_idx, rev_val, rev_src,
                                   other_idx, other_val, other_src, ind_db, answer):
    """Build word roles for a reversal_charade."""
    roles = []
    rev_role = 'SYN_F' if rev_src == 'synonym' else 'ABR_F'
    other_role = 'SYN_F' if other_src == 'synonym' else 'ABR_F'

    for k in range(n):
        if k == rev_idx:
            roles.append(WordRole(k, wp_words[k], rev_role, rev_val, rev_src or ''))
        elif k == other_idx:
            roles.append(WordRole(k, wp_words[k], other_role, other_val, other_src or ''))
        else:
            ind_types = ind_db.get_indicator_types(wp_words[k])
            if ind_types & {'reversal'}:
                roles.append(WordRole(k, wp_words[k], 'REV_I', None))
            else:
                roles.append(WordRole(k, wp_words[k], 'LNK', None))
    return roles


def deletion_evidence(wp_words, answer, db, ind_db):
    """Find deletion evidence: synonym minus 1-2 chars = answer.
    Returns Evidence or None."""
    n = len(wp_words)

    for i in range(n):
        for syn in db.get_synonyms(wp_words[i], max_len=len(answer) + 2):
            if len(syn) == len(answer) + 1:
                for k in range(len(syn)):
                    if syn[:k] + syn[k+1:] == answer:
                        removed = syn[k]
                        roles = []
                        for m in range(n):
                            if m == i:
                                roles.append(WordRole(m, wp_words[m], 'SYN_F', syn, 'synonym'))
                            else:
                                ind_types = ind_db.get_indicator_types(wp_words[m])
                                if ind_types & {'deletion'}:
                                    roles.append(WordRole(m, wp_words[m], 'DEL_I', None))
                                else:
                                    roles.append(WordRole(m, wp_words[m], 'LNK', None))
                        return Evidence(
                            operation='deletion',
                            word_roles=roles,
                            assembly=f'{syn} minus {removed} = {answer}',
                            answer=answer,
                            confidence='medium',
                        )
            elif len(syn) == len(answer) + 2:
                for result, desc in [
                    (syn[1:], f'remove first({syn[0]})'),
                    (syn[:-1], f'remove last({syn[-1]})'),
                    (syn[1:-1], f'remove outer({syn[0]},{syn[-1]})'),
                ]:
                    if result == answer:
                        roles = []
                        for m in range(n):
                            if m == i:
                                roles.append(WordRole(m, wp_words[m], 'SYN_F', syn, 'synonym'))
                            else:
                                ind_types = ind_db.get_indicator_types(wp_words[m])
                                if ind_types & {'deletion'}:
                                    roles.append(WordRole(m, wp_words[m], 'DEL_I', None))
                                else:
                                    roles.append(WordRole(m, wp_words[m], 'LNK', None))
                        return Evidence(
                            operation='deletion',
                            word_roles=roles,
                            assembly=f'{syn} {desc} = {answer}',
                            answer=answer,
                            confidence='medium',
                        )
    return None


def charade_evidence(wp_words, answer, db, ind_db):
    """Find charade evidence: pieces concatenate to produce the answer.

    Pieces can be produced by any mechanism:
    - SYN/ABR: single-word or multi-word phrase lookup
    - RAW: literal letters of the clue word
    - POS: first/last letter extraction (with indicator in clue)
    - REV: reversed synonym/abbreviation (with indicator in clue)

    Word order in the clue does NOT have to match piece order in the
    answer. Leftover words are allowed (surface reading / indicators).
    """
    answer_len = len(answer)
    n = len(wp_words)

    # Build candidate pieces: each is (word_indices_tuple, value, source_type)
    # word_indices_tuple: which clue words this piece consumes
    # source_type: 'synonym', 'abbreviation', 'raw', 'first_letter',
    #              'last_letter', 'reversal'
    all_pieces = []  # list of (word_indices, value, source_type)

    for i in range(n):
        word = wp_words[i]
        raw = ''.join(c for c in word.upper() if c.isalpha())

        # Single-word synonym/abbreviation
        for val, src in db.get_values(word, answer_len):
            if val in answer:
                all_pieces.append(((i,), val, src))

        # Raw letters
        if raw and raw in answer:
            all_pieces.append(((i,), raw, 'raw'))

        # Positional: first letter
        if raw and raw[0] in answer:
            all_pieces.append(((i,), raw[0], 'first_letter'))

        # Positional: last letter
        if raw and len(raw) >= 2 and raw[-1] in answer:
            all_pieces.append(((i,), raw[-1], 'last_letter'))

        # Positional: first 2 letters
        if raw and len(raw) >= 3 and raw[:2] in answer:
            all_pieces.append(((i,), raw[:2], 'first_n'))

        # Positional: last 2 letters
        if raw and len(raw) >= 3 and raw[-2:] in answer:
            all_pieces.append(((i,), raw[-2:], 'last_n'))

        # Reversal: reversed synonym/abbreviation in answer
        for val, src in db.get_values(word, answer_len):
            rev = val[::-1]
            if len(val) >= 2 and rev in answer and rev != val:
                all_pieces.append(((i,), rev, 'reversal'))

        # Multi-word phrases: 2-word and 3-word spans starting at i
        for span in (2, 3):
            if i + span > n:
                break
            phrase_words = wp_words[i:i + span]
            indices = tuple(range(i, i + span))
            for val, src in db.get_phrase_values(phrase_words, answer_len):
                if val in answer:
                    all_pieces.append((indices, val, src))

    # Build position index: answer_pos -> list of (word_indices, value, src)
    pos_candidates = {}
    for indices, val, src in all_pieces:
        for pos in range(answer_len - len(val) + 1):
            if answer[pos:pos + len(val)] == val:
                pos_candidates.setdefault(pos, []).append((indices, val, src))

    # Recursive search: cover answer left-to-right
    best = [None]
    max_depth = 6

    def search(pos, assignments, used_words, depth):
        if pos == answer_len:
            best[0] = list(assignments)
            return
        if best[0] is not None:
            return
        if depth >= max_depth:
            return
        if pos not in pos_candidates:
            return

        for indices, val, src in pos_candidates[pos]:
            if any(idx in used_words for idx in indices):
                continue
            assignments.append((indices, val, src))
            used_words.update(indices)
            search(pos + len(val), assignments, used_words, depth + 1)
            if best[0] is not None:
                return
            assignments.pop()
            for idx in indices:
                used_words.discard(idx)

    search(0, [], set(), 0)

    if best[0] is None:
        return None

    assignments = best[0]

    # Build word roles
    roles = []
    assigned_words = {}  # word_idx -> (value, source_type)
    for indices, val, src in assignments:
        for idx in indices:
            assigned_words[idx] = (val, src)

    for k in range(n):
        if k in assigned_words:
            val, src = assigned_words[k]
            if src == 'synonym':
                role = 'SYN_F'
            elif src == 'abbreviation':
                role = 'ABR_F'
            elif src in ('first_letter', 'last_letter', 'first_n', 'last_n'):
                role = 'POS_F'
            elif src == 'reversal':
                role = 'REV_F'
            elif src == 'raw':
                role = 'RAW'
            else:
                role = 'SYN_F'
            roles.append(WordRole(k, wp_words[k], role, val, src))
        else:
            roles.append(WordRole(k, wp_words[k], 'LNK', None))

    pieces = ' + '.join(f'{a[1]}' for a in assignments)
    return Evidence(
        operation='charade',
        word_roles=roles,
        assembly=f'{pieces} = {answer}',
        answer=answer,
        confidence='medium',
    )


def anagram_evidence(wp_words, answer, db, ind_db):
    """Find anagram evidence: wordplay letters (minus indicator/link words)
    anagram to the answer.

    Tries progressively:
    1. All letters = answer (pure anagram, no excluded words)
    2. Exclude known ANA_I words from indicator DB
    3. Exclude each single word in turn (brute force)
    4. Exclude each pair of words (for clues with 2 non-fodder words)
    """
    n = len(wp_words)
    answer_sorted = sorted(answer)

    # Pre-compute per-word letters
    word_letters = []
    for word in wp_words:
        word_letters.append(''.join(c for c in word.upper() if c.isalpha()))

    all_letters = ''.join(word_letters)

    def _build_evidence(excluded):
        """Build Evidence with excluded words as ANA_I/LNK, rest as ANA_F."""
        fodder = ''.join(word_letters[k] for k in range(n) if k not in excluded)
        roles = []
        for k in range(n):
            if k in excluded:
                # Check if it's a known indicator
                ind_types = ind_db.get_indicator_types(wp_words[k])
                if ind_types & {'anagram'}:
                    roles.append(WordRole(k, wp_words[k], 'ANA_I', None))
                else:
                    roles.append(WordRole(k, wp_words[k], 'LNK', None))
            else:
                roles.append(WordRole(k, wp_words[k], 'ANA_F', word_letters[k], 'raw'))

        has_indicator = any(wr.role == 'ANA_I' for wr in roles)
        return Evidence(
            operation='anagram',
            word_roles=roles,
            assembly=f'anagram({fodder}) = {answer}',
            answer=answer,
            confidence='high' if has_indicator else 'medium',
        )

    # 1. Pure anagram: all letters
    if sorted(all_letters) == answer_sorted:
        return _build_evidence(set())

    # 2. Exclude known ANA_I words
    ind_words = set()
    for k in range(n):
        if ind_db.get_indicator_types(wp_words[k]) & {'anagram'}:
            ind_words.add(k)

    if ind_words:
        remaining = ''.join(word_letters[k] for k in range(n) if k not in ind_words)
        if sorted(remaining) == answer_sorted:
            return _build_evidence(ind_words)

    # 3. Exclude each single word in turn
    for exc in range(n):
        remaining = ''.join(word_letters[k] for k in range(n) if k != exc)
        if sorted(remaining) == answer_sorted:
            return _build_evidence({exc})

    # 4. Exclude each pair of words
    if n >= 4:  # only worth trying with 4+ words
        for i in range(n):
            for j in range(i + 1, n):
                remaining = ''.join(word_letters[k] for k in range(n)
                                    if k != i and k != j)
                if sorted(remaining) == answer_sorted:
                    return _build_evidence({i, j})

    return None


# ============================================================
# Main triage with evidence
# ============================================================

def triage_clue(wp_words, answer, db, ind_db):
    """Run hierarchical triage on a single clue, returning Evidence or None.

    Every classification requires positive evidence. Returns None if
    no positive test passes (UNCLASSIFIED).
    """
    answer_len = len(answer)
    n = len(wp_words)

    # Compute signals
    total_letters = sum(len(re.sub(r'[^a-zA-Z]', '', w)) for w in wp_words)
    ratio = total_letters / answer_len if answer_len > 0 else 0

    ind_types = set()
    for word in wp_words:
        for t in ind_db.get_indicator_types(word):
            ind_types.add(t)

    has_ana = 'anagram' in ind_types
    has_rev = 'reversal' in ind_types
    has_con = 'container' in ind_types or 'insertion' in ind_types
    has_del = 'deletion' in ind_types
    has_hom = 'homophone' in ind_types
    has_any = has_ana or has_rev or has_con or has_del or has_hom

    # === LEVEL 1: Anagram (budget + indicator + structural) ===
    if has_ana and 1.0 <= ratio <= 2.2:
        ev = anagram_evidence(wp_words, answer, db, ind_db)
        if ev:
            return ev

    # === LEVEL 1: Pure charade (no indicators) ===
    if not has_any:
        ev = charade_evidence(wp_words, answer, db, ind_db)
        if ev:
            return ev

    # === LEVEL 2: Structural tests with evidence ===

    # Container (highest precision: 93.5%)
    if has_con:
        ev = container_evidence(wp_words, answer, db, ind_db)
        if ev:
            return ev

    # Reversal
    if has_rev:
        ev = reversal_evidence(wp_words, answer, db, ind_db)
        if ev:
            return ev

    # Deletion
    if has_del:
        ev = deletion_evidence(wp_words, answer, db, ind_db)
        if ev:
            return ev

    # Anagram at higher budget (anagram_charade)
    if has_ana:
        ev = anagram_evidence(wp_words, answer, db, ind_db)
        if ev:
            return ev

    # Charade (structural tests all failed, but indicators present = red herrings)
    if ratio > 2.5:
        ev = charade_evidence(wp_words, answer, db, ind_db)
        if ev:
            return ev

    return None


# ============================================================
# Test harness
# ============================================================

def run():
    print("Loading databases...")
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
        clue_data[clue_id] = (words, answer, op)

    conn.close()

    print(f"Running evidence triage on {len(clues)} clues...")
    t0 = time.time()

    results = []  # (clue_id, actual_op, evidence_or_none)
    evidence_count = 0
    verified_count = 0

    for i, (clue_id, actual_op, ratio) in enumerate(clues):
        if actual_op == 'hidden':
            continue

        if i % 2000 == 0 and i > 0:
            elapsed = time.time() - t0
            print(f"  {i}/{len(clues)} ({elapsed:.1f}s) — {evidence_count} with evidence")

        words, answer, _ = clue_data[clue_id]

        ev = triage_clue(words, answer, ref_db, ind_db)

        if ev:
            evidence_count += 1
            results.append((clue_id, actual_op, ev))
        else:
            results.append((clue_id, actual_op, None))

    elapsed = time.time() - t0
    print(f"\nDone in {elapsed:.1f}s")

    # === Analysis ===
    total = len(results)
    with_evidence = sum(1 for _, _, ev in results if ev is not None)
    correct_op = sum(1 for _, actual, ev in results if ev and ev.operation == actual)

    print(f"\n{'='*70}")
    print(f"EVIDENCE TRIAGE RESULTS")
    print(f"{'='*70}")
    print(f"Total clues: {total}")
    print(f"With evidence: {with_evidence} ({100*with_evidence/total:.1f}%)")
    print(f"Correct operation: {correct_op} ({100*correct_op/total:.1f}%)")
    print(f"UNCLASSIFIED: {total - with_evidence} ({100*(total-with_evidence)/total:.1f}%)")

    # Breakdown by predicted operation
    print(f"\n--- By predicted operation ---")
    by_pred = defaultdict(lambda: Counter())
    for _, actual, ev in results:
        pred = ev.operation if ev else 'UNCLASSIFIED'
        by_pred[pred][actual] += 1

    print(f"{'Predicted':25s} {'Total':>6s} {'Correct':>7s} {'Prec':>6s}")
    print("-" * 50)
    for pred in sorted(by_pred.keys(), key=lambda x: -sum(by_pred[x].values())):
        total_pred = sum(by_pred[pred].values())
        correct_pred = by_pred[pred].get(pred, 0) if pred != 'UNCLASSIFIED' else 0
        prec = 100 * correct_pred / total_pred if total_pred else 0
        print(f"  {pred:23s} {total_pred:6d} {correct_pred:7d} {prec:5.1f}%")
        for actual, cnt in by_pred[pred].most_common(3):
            if actual != pred:
                print(f"    actually {actual:20s} {cnt:5d}")

    # Show sample evidence
    print(f"\n--- Sample evidence (first 10 correct) ---")
    shown = 0
    for _, actual, ev in results:
        if ev and ev.operation == actual and shown < 10:
            print(f"  {ev.summary()}")
            print(f"    assembly: {ev.assembly}")
            shown += 1

    # Per-operation recall
    print(f"\n--- Per-operation recall ---")
    print(f"{'Operation':25s} {'Correct':>7s} {'Total':>7s} {'Recall':>7s}")
    print("-" * 50)
    by_actual = defaultdict(lambda: {'correct': 0, 'total': 0})
    for _, actual, ev in results:
        by_actual[actual]['total'] += 1
        if ev and ev.operation == actual:
            by_actual[actual]['correct'] += 1
    for op in sorted(by_actual.keys(), key=lambda x: -by_actual[x]['total']):
        c = by_actual[op]['correct']
        t = by_actual[op]['total']
        print(f"  {op:23s} {c:7d} {t:7d} {100*c/t:6.1f}%")


if __name__ == '__main__':
    run()
