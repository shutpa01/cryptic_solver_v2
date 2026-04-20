"""Grammar-guided triage for the signature solver.

Uses POS signatures to predict word roles, then verifies against the
known answer. Returns a SolveResult compatible with the existing
signature solver pipeline.

Three paths:
1. Standalone detection (anagram, reversal, cryptic definition)
2. Grammar signature lookup + mechanical verification
3. Mechanism detection (POS bigrams) + structural confirmation

Falls through to None if nothing works — caller should then try
the existing catalog-based solver.
"""

import json
import os
import re
from itertools import permutations

try:
    import spacy
    _NLP = None

    def _get_nlp():
        global _NLP
        if _NLP is None:
            _NLP = spacy.load('en_core_web_sm')
        return _NLP
except ImportError:
    def _get_nlp():
        return None

from .solver import SolveResult, SignatureResult
from .tokens import *
from .db import RefDB


# POS tag abstraction map
_MID_MAP = {
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

_FODDER_ROLES = {SYN_F, ABR_F, RAW, ANA_F, HID_F, POS_F}
_INDICATOR_ROLES = {ANA_I, REV_I, CON_I, DEL_I, HID_I, HOM_I}


def _clean(w):
    return w.lower().strip(".,;:!?\"'()-\u2018\u2019\u201c\u201d")


# ============================================================
# Grammar catalog (loaded once)
# ============================================================

_CATALOG = None


def _load_catalog():
    global _CATALOG
    if _CATALOG is not None:
        return _CATALOG

    json_path = os.path.join(
        os.path.dirname(os.path.dirname(__file__)),
        'data', 'grammar_catalog.json'
    )
    if not os.path.exists(json_path):
        _CATALOG = {}
        return _CATALOG

    with open(json_path) as f:
        raw = json.load(f)

    _CATALOG = {}
    for entry in raw:
        pos_seq = tuple(entry['pos'])
        candidates = [(tuple(c['roles']), c['count']) for c in entry['candidates']]
        _CATALOG[pos_seq] = candidates

    return _CATALOG


# ============================================================
# POS tagging
# ============================================================

def _pos_tag(words):
    """POS-tag wordplay words. Returns list of fine-grained tags."""
    nlp = _get_nlp()
    if nlp is None:
        return None

    doc = nlp(' '.join(words))
    tokens = list(doc)
    pos_tags = []
    si = 0
    for word in words:
        if si < len(tokens):
            pos_tags.append(tokens[si].tag_)
            consumed = len(tokens[si].text)
            si += 1
            while consumed < len(word) and si < len(tokens):
                consumed += len(tokens[si].text) + 1
                si += 1
        else:
            pos_tags.append('XX')
    return pos_tags


def _mid_tags(pos_tags):
    """Convert fine-grained POS tags to mid-level abstraction."""
    return tuple(_MID_MAP.get(t, 'X') for t in pos_tags)


# ============================================================
# Value lookups (extends RefDB capabilities)
# ============================================================

def _get_word_values(word, db, answer_len):
    """Get all (value, source_type) pairs for a clue word."""
    results = []
    w = _clean(word)
    for s in db.get_synonyms(w, max_len=answer_len + 2):
        results.append((s, 'synonym'))
    for a in db.get_abbreviations(w):
        results.append((a, 'abbreviation'))
    return results


def _get_phrase_values(words, db, answer_len):
    """Get (value, source_type) for a multi-word phrase."""
    phrase = ' '.join(_clean(w) for w in words)
    results = []
    for s in db.get_synonyms(phrase, max_len=answer_len + 2):
        results.append((s, 'synonym'))
    for a in db.get_abbreviations(phrase):
        results.append((a, 'abbreviation'))
    return results


# ============================================================
# Structural tests — return evidence or None
# ============================================================

def _try_anagram(wp_words, answer, db):
    """Try anagram: all wordplay letters (minus excluded words) = answer."""
    n = len(wp_words)
    answer_sorted = sorted(answer)
    word_letters = [''.join(c for c in w.upper() if c.isalpha()) for w in wp_words]

    # All letters
    if sorted(''.join(word_letters)) == answer_sorted:
        return _build_anagram_result(wp_words, word_letters, set(), answer, db)

    # Exclude each single word
    for exc in range(n):
        remaining = ''.join(word_letters[k] for k in range(n) if k != exc)
        if sorted(remaining) == answer_sorted:
            return _build_anagram_result(wp_words, word_letters, {exc}, answer, db)

    # Exclude each pair
    if n >= 4:
        for i in range(n):
            for j in range(i + 1, n):
                remaining = ''.join(word_letters[k] for k in range(n) if k not in (i, j))
                if sorted(remaining) == answer_sorted:
                    return _build_anagram_result(wp_words, word_letters, {i, j}, answer, db)

    return None


def _build_anagram_result(wp_words, word_letters, excluded, answer, db):
    """Build SolveResult for an anagram."""
    word_roles = []
    fodder = ''.join(word_letters[k] for k in range(len(wp_words)) if k not in excluded)

    for k, word in enumerate(wp_words):
        if k in excluded:
            ind_types = db.get_indicator_types(_clean(word))
            is_ana_ind = any(t == 'anagram' for t, _, _ in ind_types)
            if is_ana_ind:
                word_roles.append((word, ANA_I, None))
            else:
                word_roles.append((word, LNK, None))
        else:
            word_roles.append((word, ANA_F, word_letters[k]))

    has_indicator = any(t == ANA_I for _, t, _ in word_roles)
    explanation = 'Anagram of "%s" = %s' % (fodder, answer)
    sig = SignatureResult([ANA_I, ANA_F] if has_indicator else [ANA_F],
                          word_roles, [explanation])
    confidence = 90 if has_indicator else 70
    return SolveResult(sig, confidence, [('anagram', 0)], [], {})


def _try_charade(wp_words, answer, db):
    """Try charade: pieces from word lookups concatenate to answer.

    Pieces can be: synonym, abbreviation, raw letters, first/last letter,
    reversed synonym, multi-word phrase.
    """
    answer_len = len(answer)
    n = len(wp_words)

    # Build all candidate pieces
    all_pieces = []  # (word_indices_tuple, value, source_type)

    for i in range(n):
        word = wp_words[i]
        raw = ''.join(c for c in word.upper() if c.isalpha())

        # Synonym/abbreviation
        for val, src in _get_word_values(word, db, answer_len):
            if val in answer:
                all_pieces.append(((i,), val, src))

        # Raw letters
        if raw and raw in answer:
            all_pieces.append(((i,), raw, 'raw'))

        # Positional: first/last letter, first/last 2
        if raw:
            if raw[0] in answer:
                all_pieces.append(((i,), raw[0], 'first_letter'))
            if len(raw) >= 2 and raw[-1] in answer:
                all_pieces.append(((i,), raw[-1], 'last_letter'))
            if len(raw) >= 3 and raw[:2] in answer:
                all_pieces.append(((i,), raw[:2], 'first_n'))
            if len(raw) >= 3 and raw[-2:] in answer:
                all_pieces.append(((i,), raw[-2:], 'last_n'))

        # Reversal
        for val, src in _get_word_values(word, db, answer_len):
            if len(val) >= 2 and val[::-1] in answer and val[::-1] != val:
                all_pieces.append(((i,), val[::-1], 'reversal'))

        # Multi-word phrases (2-word, 3-word)
        for span in (2, 3):
            if i + span > n:
                break
            indices = tuple(range(i, i + span))
            for val, src in _get_phrase_values(wp_words[i:i + span], db, answer_len):
                if val in answer:
                    all_pieces.append((indices, val, src))

    # Build position index
    pos_candidates = {}
    for indices, val, src in all_pieces:
        for pos in range(answer_len - len(val) + 1):
            if answer[pos:pos + len(val)] == val:
                pos_candidates.setdefault(pos, []).append((indices, val, src))

    # Recursive search
    best = [None]

    def search(pos, assignments, used, depth):
        if pos == answer_len:
            best[0] = list(assignments)
            return
        if best[0] is not None or depth >= 6:
            return
        if pos not in pos_candidates:
            return
        for indices, val, src in pos_candidates[pos]:
            if any(idx in used for idx in indices):
                continue
            assignments.append((indices, val, src))
            used.update(indices)
            search(pos + len(val), assignments, used, depth + 1)
            if best[0] is not None:
                return
            assignments.pop()
            for idx in indices:
                used.discard(idx)

    search(0, [], set(), 0)

    if best[0] is None:
        return None

    # Reject trivial matches: single piece = full answer (that's a definition, not wordplay)
    if len(best[0]) == 1 and best[0][0][1] == answer:
        return None

    return _build_charade_result(wp_words, best[0], answer, db)


def _build_charade_result(wp_words, assignments, answer, db):
    """Build SolveResult for a charade."""
    assigned = {}
    for indices, val, src in assignments:
        for idx in indices:
            assigned[idx] = (val, src)

    word_roles = []
    for k in range(len(wp_words)):
        if k in assigned:
            val, src = assigned[k]
            if src == 'synonym':
                tok = SYN_F
            elif src == 'abbreviation':
                tok = ABR_F
            elif src in ('first_letter', 'last_letter', 'first_n', 'last_n'):
                tok = SYN_F  # mapped to SYN_F for compatibility
            elif src == 'reversal':
                tok = SYN_F  # mapped for compatibility
            else:
                tok = RAW
            word_roles.append((wp_words[k], tok, val))
        else:
            word_roles.append((wp_words[k], LNK, None))

    pieces_str = ' + '.join(a[1] for a in assignments)
    explanation = '%s = %s' % (pieces_str, answer)
    sig_tokens = [tok for _, tok, _ in word_roles if tok != LNK]
    sig = SignatureResult(sig_tokens, word_roles, [explanation])

    # Score: higher if more pieces are DB-confirmed synonyms/abbreviations
    confirmed = sum(1 for _, _, src in assignments if src in ('synonym', 'abbreviation'))
    total = len(assignments)
    confidence = 80 + (10 * confirmed // max(total, 1))
    return SolveResult(sig, min(95, confidence), [('charade', 0)], [], {})


def _try_container(wp_words, answer, db):
    """Try container: outer(inner) = answer."""
    answer_len = len(answer)
    n = len(wp_words)

    word_vals = []
    for word in wp_words:
        word_vals.append(_get_word_values(word, db, answer_len))

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
                        for inner, inner_src in word_vals[j]:
                            if inner == inner_needed:
                                return _build_container_result(
                                    wp_words, i, outer, outer_src,
                                    j, inner, inner_src, answer, db
                                )
    return None


def _build_container_result(wp_words, outer_idx, outer, outer_src,
                             inner_idx, inner, inner_src, answer, db):
    """Build SolveResult for a container."""
    word_roles = []
    for k in range(len(wp_words)):
        if k == outer_idx:
            tok = SYN_F if outer_src == 'synonym' else ABR_F
            word_roles.append((wp_words[k], tok, outer))
        elif k == inner_idx:
            tok = SYN_F if inner_src == 'synonym' else ABR_F
            word_roles.append((wp_words[k], tok, inner))
        else:
            ind_types = db.get_indicator_types(_clean(wp_words[k]))
            is_con = any(t in ('container', 'insertion') for t, _, _ in ind_types)
            if is_con:
                word_roles.append((wp_words[k], CON_I, None))
            else:
                word_roles.append((wp_words[k], LNK, None))

    explanation = '%s containing %s = %s' % (outer, inner, answer)
    sig = SignatureResult([SYN_F, CON_I, SYN_F], word_roles, [explanation])
    return SolveResult(sig, 90, [('container', 0)], [], {})


def _try_reversal(wp_words, answer, db):
    """Try reversal: reversed synonym = answer (full or charade piece)."""
    answer_len = len(answer)
    n = len(wp_words)

    # Pure reversal: single word's synonym reversed = whole answer
    for i in range(n):
        for val, src in _get_word_values(wp_words[i], db, answer_len):
            if len(val) == answer_len and val[::-1] == answer:
                word_roles = []
                for k in range(n):
                    if k == i:
                        tok = SYN_F if src == 'synonym' else ABR_F
                        word_roles.append((wp_words[k], tok, val))
                    else:
                        ind_types = db.get_indicator_types(_clean(wp_words[k]))
                        is_rev = any(t == 'reversal' for t, _, _ in ind_types)
                        if is_rev:
                            word_roles.append((wp_words[k], REV_I, None))
                        else:
                            word_roles.append((wp_words[k], LNK, None))

                explanation = 'Reverse of %s = %s' % (val, answer)
                sig = SignatureResult([SYN_F, REV_I], word_roles, [explanation])
                return SolveResult(sig, 90, [('reversal', 0)], [], {})

    return None


# ============================================================
# POS mechanism detectors
# ============================================================

def _detect_container(pos_tags):
    noun_tags = {'NN', 'NNS', 'NNP', 'NNPS'}
    for i in range(len(pos_tags) - 2):
        if (pos_tags[i] in noun_tags and pos_tags[i + 2] in noun_tags
                and pos_tags[i + 1] in {'VBG', 'VBN'}):
            return True
    return False


def _detect_reversal(pos_tags):
    rev_bigrams = {('VBD', 'RP'), ('VBN', 'RP'), ('VBZ', 'RP'), ('RP', 'IN'),
                   ('RP', 'TO'), ('VBD', 'RB')}
    for i in range(len(pos_tags) - 1):
        if (pos_tags[i], pos_tags[i + 1]) in rev_bigrams:
            return True
    return False


# ============================================================
# Main triage entry point
# ============================================================

def grammar_triage(clue_text, answer, db, def_phrase=None, wp_words=None):
    """Run grammar-guided triage on a clue.

    Args:
        clue_text: full clue text
        answer: known answer (uppercase, no spaces/hyphens)
        db: RefDB instance
        def_phrase: definition phrase if known (from extract_definition_candidates)
        wp_words: wordplay words if already extracted

    Returns:
        SolveResult if solved, None otherwise.
        Caller should fall through to existing catalog solver if None.
    """
    if wp_words is None:
        return None

    answer_len = len(answer)
    n = len(wp_words)
    if n == 0:
        return None

    total_letters = sum(len(re.sub(r'[^a-zA-Z]', '', w)) for w in wp_words)
    ratio = total_letters / answer_len if answer_len > 0 else 0

    # === Standalone: anagram ===
    if 0.8 <= ratio <= 2.5:
        result = _try_anagram(wp_words, answer, db)
        if result and result.confidence >= 70:
            return result

    # === Standalone: pure reversal ===
    result = _try_reversal(wp_words, answer, db)
    if result:
        return result

    # === POS-guided path ===
    pos_tags = _pos_tag(wp_words)
    if pos_tags is None:
        # spaCy not available — skip grammar, try structural tests only
        pass
    else:
        mid = _mid_tags(pos_tags)
        catalog = _load_catalog()

        # Grammar signature lookup: try each candidate role sequence
        if mid in catalog:
            for role_seq, count in catalog[mid]:
                result = _verify_grammar_roles(wp_words, role_seq, answer, db)
                if result:
                    return result

        # Mechanism detection: POS bigrams guide which structural tests to try
        if _detect_container(pos_tags):
            result = _try_container(wp_words, answer, db)
            if result:
                return result

        if _detect_reversal(pos_tags):
            # Already tried pure reversal above; charade handles reversal pieces
            pass

    # === Structural tests without POS guidance ===
    result = _try_container(wp_words, answer, db)
    if result:
        return result

    result = _try_charade(wp_words, answer, db)
    if result:
        return result

    return None


def _verify_grammar_roles(wp_words, roles, answer, db):
    """Verify a grammar-predicted role sequence against the answer.

    For each FODDER word, look up values. Check if they concatenate to answer.
    """
    n = len(wp_words)
    answer_len = len(answer)

    if len(roles) != n:
        return None

    # Collect fodder candidates
    fodder_positions = []
    for i in range(n):
        role = roles[i]
        if role in _FODDER_ROLES:
            word = wp_words[i]
            candidates = []

            if role in (SYN_F, ABR_F):
                for val, src in _get_word_values(word, db, answer_len):
                    if val in answer:
                        candidates.append((val, src))
                raw = ''.join(c for c in word.upper() if c.isalpha())
                if raw and raw in answer:
                    candidates.append((raw, 'raw'))
                if raw:
                    if raw[0] in answer:
                        candidates.append((raw[0], 'first_letter'))
                    if len(raw) >= 2 and raw[-1] in answer:
                        candidates.append((raw[-1], 'last_letter'))
                for val, src in _get_word_values(word, db, answer_len):
                    if len(val) >= 2 and val[::-1] in answer and val[::-1] != val:
                        candidates.append((val[::-1], 'reversal'))
                # Phrase with next word
                if i < n - 1 and roles[i + 1] in _FODDER_ROLES:
                    for val, src in _get_phrase_values(wp_words[i:i+2], db, answer_len):
                        if val in answer:
                            candidates.append((val, src))

            elif role == ANA_F:
                raw = ''.join(c for c in word.upper() if c.isalpha())
                if raw:
                    candidates.append((raw, 'anagram_fodder'))

            if not candidates:
                return None
            fodder_positions.append((i, role, candidates))

    if not fodder_positions:
        return None

    # Position-indexed search
    pos_candidates = {}
    for word_idx, role, candidates in fodder_positions:
        for val, src in candidates:
            for pos in range(answer_len - len(val) + 1):
                if answer[pos:pos + len(val)] == val:
                    pos_candidates.setdefault(pos, []).append((word_idx, val, src))

    best = [None]

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

    return _build_charade_result(wp_words, [((a[0],), a[1], a[2]) for a in best[0]], answer, db)
