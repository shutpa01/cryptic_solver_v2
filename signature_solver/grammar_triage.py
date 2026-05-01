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
import time
from itertools import permutations

# Time limit for grammar triage per clue (seconds).
# Must be generous enough for 9-10 word clues with compound mechanisms.
TRIAGE_TIMEOUT = 1.0  # 1 second

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


# --- Positional-extraction licensing -------------------------------------
# Cryptic convention: chopping the first/last/outer letter(s) off a word
# requires an indicator that licenses the operation. Without one, the
# extraction is unlicensed and the resulting "piece" is a coincidence.
_POS_FIRST_SUBTYPES = {'first_use', 'first', 'initial', 'head'}
_POS_LAST_SUBTYPES = {'last_use', 'last', 'last_letter', 'last letter', 'tail'}
_POS_OUTER_SUBTYPES = {'outer_use', 'outer', 'outside_letters', 'ends'}


def _word_positional_kinds(word, db):
    """Return the SET of positional kinds a word indicates.

    A single word can indicate multiple kinds — e.g. 'at' is both an
    acrostic ('initial') and a parts/last indicator. The set form lets
    callers ask about any specific kind without losing alternatives.
    """
    kinds = set()
    for wtype, subtype, _conf in db.get_indicator_types(_clean(word)):
        if wtype == 'acrostic':
            kinds.add('first')
        elif wtype == 'parts':
            if subtype in _POS_FIRST_SUBTYPES:
                kinds.add('first')
            if subtype in _POS_LAST_SUBTYPES:
                kinds.add('last')
            if subtype in _POS_OUTER_SUBTYPES:
                kinds.add('outer')
    return kinds


def _has_positional_indicator(words, db, kind):
    """Is any word in this wordplay window a positional indicator of the given kind?"""
    for w in words:
        if kind in _word_positional_kinds(w, db):
            return True
    return False


def _dbe_marker_spans(wp_words):
    """Return list of (start, end_exclusive) spans for every DBE marker.

    Multi-word markers (e.g. 'for example') are detected first so that
    their constituent words aren't double-counted by the single-word check.
    """
    n = len(wp_words)
    cleans = [_clean(w) for w in wp_words]
    spans = []
    consumed = set()
    for i in range(n):
        for words_tuple in DBE_MARKERS_MULTI:
            j = len(words_tuple)
            if i + j <= n and tuple(cleans[i:i + j]) == words_tuple:
                spans.append((i, i + j))
                consumed.update(range(i, i + j))
    for i in range(n):
        if i in consumed:
            continue
        if cleans[i] in DBE_MARKERS_SINGLE:
            spans.append((i, i + 1))
            consumed.add(i)
    return spans


def _dbe_marker_indices(wp_words):
    """Return the set of word indices that ARE the DBE marker words.

    These are the 'maybe' / 'say' / 'perhaps' / 'for example' words
    themselves — NOT the example words they modify.
    """
    return {idx for start, end in _dbe_marker_spans(wp_words)
            for idx in range(start, end)}


def _dbe_marked_indices(wp_words):
    """Return the set of word indices that are DBE-marked.

    A word is DBE-marked when a definition-by-example marker (maybe,
    perhaps, say, for example, etc.) is adjacent to it. Convention is that
    the marker FOLLOWS the example word ('Hill maybe', 'Garibaldi say'),
    so we mark the word at index marker_idx-1. We also accept the marker
    BEFORE the word as a defensive fallback ('say, hill').

    DBE-marked words must NOT contribute their RAW letters as a wordplay
    piece — only their synonyms / abbreviations / category-mates may.
    """
    n = len(wp_words)
    spans = _dbe_marker_spans(wp_words)
    consumed = {idx for start, end in spans for idx in range(start, end)}
    marked = set()
    for start, end in spans:
        if start > 0 and (start - 1) not in consumed:
            marked.add(start - 1)
        elif end < n and end not in consumed:
            marked.add(end)
    return marked


def _annotate_dbe_markers(wp_words, word_roles):
    """Promote LNK->DBE_MARKER for clue words that ARE DBE markers.

    Mutates the word_roles list in place. Existing non-LNK roles are not
    changed (e.g. if the matcher decided 'maybe' was an indicator for some
    other reason, leave that alone).
    """
    marker_idx = _dbe_marker_indices(wp_words)
    if not marker_idx:
        return word_roles
    for i, role in enumerate(word_roles):
        if i in marker_idx and len(role) >= 2 and role[1] == LNK:
            # Preserve the rest of the tuple, replace the token.
            new = (role[0], DBE_MARKER) + tuple(role[2:])
            word_roles[i] = new
    return word_roles



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

    # Anagram with abbreviation substitution: some words contribute their
    # abbreviation/synonym value instead of raw letters (e.g. "western"=W, "area"=A)
    # Try substituting 1-2 words with their short DB values
    answer_len = len(answer)
    short_vals = {}  # word_idx -> list of short values (1-2 chars)
    for k in range(n):
        vals = []
        for val, src in _get_word_values(wp_words[k], db, answer_len):
            if len(val) <= 2:
                vals.append((val, src))
        if vals:
            short_vals[k] = vals

    if short_vals:
        from itertools import combinations
        sub_indices = list(short_vals.keys())

        # Try 1 substitution
        for si in sub_indices:
            for sub_val, sub_src in short_vals[si]:
                others = [k for k in range(n) if k != si]
                for n_exc in range(0, len(others) + 1):
                    for exc_combo in combinations(others, n_exc):
                        remaining = sub_val + ''.join(
                            word_letters[k] for k in range(n)
                            if k != si and k not in exc_combo
                        )
                        if sorted(remaining) == answer_sorted:
                            # Build result with substituted word
                            modified_letters = list(word_letters)
                            modified_letters[si] = sub_val
                            return _build_anagram_result(
                                wp_words, modified_letters, set(exc_combo), answer, db
                            )

        # Try 2 substitutions
        if len(sub_indices) >= 2:
            for s1, s2 in combinations(sub_indices, 2):
                for v1, src1 in short_vals[s1]:
                    for v2, src2 in short_vals[s2]:
                        others = [k for k in range(n) if k not in (s1, s2)]
                        for n_exc in range(0, len(others) + 1):
                            for exc_combo in combinations(others, n_exc):
                                remaining = v1 + v2 + ''.join(
                                    word_letters[k] for k in range(n)
                                    if k not in (s1, s2) and k not in exc_combo
                                )
                                if sorted(remaining) == answer_sorted:
                                    modified_letters = list(word_letters)
                                    modified_letters[s1] = v1
                                    modified_letters[s2] = v2
                                    return _build_anagram_result(
                                        wp_words, modified_letters,
                                        set(exc_combo), answer, db
                                    )

    return None


def _try_anagram_with_positional(wp_words, answer, db):
    """Try anagram where one word contributes only its first or last letter
    to the anagram fodder (e.g. 'At first, Harry' -> H joins the anagram).

    Strategy: enumerate which words are INCLUDED as fodder (2-5 words),
    plus one word contributing a positional letter. Check if the combined
    letters anagram to the answer. Much faster than enumerating exclusions.
    """
    from itertools import combinations

    n = len(wp_words)
    answer_sorted = sorted(answer)
    answer_len = len(answer)
    word_letters = [''.join(c for c in w.upper() if c.isalpha()) for w in wp_words]

    # Gate positional contribution on a licensing indicator being present.
    has_first = _has_positional_indicator(wp_words, db, 'first')
    has_last = _has_positional_indicator(wp_words, db, 'last')
    if not (has_first or has_last):
        return None

    # For each word that could contribute a positional letter
    for pos_idx in range(n):
        raw = word_letters[pos_idx]
        if len(raw) < 2:
            continue

        for letter, pos_type in [(raw[0], 'first_letter'), (raw[-1], 'last_letter')]:
            if pos_type == 'first_letter' and not has_first:
                continue
            if pos_type == 'last_letter' and not has_last:
                continue
            # How many more letters do we need from fodder words?
            needed = answer_len - len(letter)  # = answer_len - 1

            # Try including 1-5 other words as full-letter fodder
            other_indices = [k for k in range(n) if k != pos_idx]
            for n_fodder in range(1, min(6, len(other_indices) + 1)):
                for fodder_combo in combinations(other_indices, n_fodder):
                    fodder_letters = ''.join(word_letters[k] for k in fodder_combo)
                    if len(fodder_letters) != needed:
                        continue
                    combined = letter + fodder_letters
                    if sorted(combined) == answer_sorted:
                        excluded = set(k for k in range(n)
                                       if k != pos_idx and k not in fodder_combo)
                        # Build result
                        word_roles = []
                        for k in range(n):
                            if k == pos_idx:
                                word_roles.append((wp_words[k], ANA_F, letter))
                            elif k in fodder_combo:
                                word_roles.append((wp_words[k], ANA_F, word_letters[k]))
                            else:
                                ind_types = db.get_indicator_types(_clean(wp_words[k]))
                                is_ana = any(t == 'anagram' for t, _, _ in ind_types)
                                if is_ana:
                                    word_roles.append((wp_words[k], ANA_I, None))
                                else:
                                    word_roles.append((wp_words[k], LNK, None))

                        has_indicator = any(t == ANA_I for _, t, _ in word_roles)
                        fodder = combined
                        explanation = 'Anagram of "%s" = %s' % (fodder, answer)
                        sig = SignatureResult([ANA_I, ANA_F] if has_indicator else [ANA_F],
                                              word_roles, [explanation])
                        confidence = 85 if has_indicator else 65
                        return SolveResult(sig, confidence, [('anagram', 0)], [], {})

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

    dbe_marked = _dbe_marked_indices(wp_words)

    # Build all candidate pieces
    all_pieces = []  # (word_indices_tuple, value, source_type)

    for i in range(n):
        word = wp_words[i]
        raw = ''.join(c for c in word.upper() if c.isalpha())

        # Synonym/abbreviation
        for val, src in _get_word_values(word, db, answer_len):
            if val in answer:
                all_pieces.append(((i,), val, src))

        # Raw letters — blocked when a DBE marker (maybe / perhaps / say / …)
        # is adjacent to this word, because the cryptic convention says the
        # word stands for an example rather than itself.
        if raw and raw in answer and i not in dbe_marked:
            all_pieces.append(((i,), raw, 'raw'))

        # Positional: first/last/outer letters — gated on a licensing
        # indicator being present somewhere in the wordplay window.
        if raw:
            has_first = _has_positional_indicator(wp_words, db, 'first')
            has_last = _has_positional_indicator(wp_words, db, 'last')
            has_outer = _has_positional_indicator(wp_words, db, 'outer')
            if has_first and raw[0] in answer:
                all_pieces.append(((i,), raw[0], 'first_letter'))
            if has_last and len(raw) >= 2 and raw[-1] in answer:
                all_pieces.append(((i,), raw[-1], 'last_letter'))
            if has_first and len(raw) >= 3 and raw[:2] in answer:
                all_pieces.append(((i,), raw[:2], 'first_n'))
            if has_last and len(raw) >= 3 and raw[-2:] in answer:
                all_pieces.append(((i,), raw[-2:], 'last_n'))
            if has_outer and len(raw) >= 2 and (raw[0] + raw[-1]) in answer:
                all_pieces.append(((i,), raw[0] + raw[-1], 'outer_letters'))

        # Reversal — preserve the ORIGINAL synonym so the explanation can
        # honestly say "DAMON reversed = NOMAD" instead of "NOMAD (synonym
        # of Hill)". Carried via an optional 4th meta-element on the piece.
        for val, src in _get_word_values(word, db, answer_len):
            if len(val) >= 2 and val[::-1] in answer and val[::-1] != val:
                all_pieces.append(((i,), val[::-1], 'reversal',
                                   {'original_synonym': val}))

        # Multi-word phrases (2-word, 3-word)
        for span in (2, 3):
            if i + span > n:
                break
            indices = tuple(range(i, i + span))
            for val, src in _get_phrase_values(wp_words[i:i + span], db, answer_len):
                if val in answer:
                    all_pieces.append((indices, val, src))

    # Build position index — pieces are (indices, val, src) or
    # (indices, val, src, meta_dict). Carry the optional meta through.
    pos_candidates = {}
    for piece in all_pieces:
        indices, val, src = piece[0], piece[1], piece[2]
        meta = piece[3] if len(piece) > 3 else None
        for pos in range(answer_len - len(val) + 1):
            if answer[pos:pos + len(val)] == val:
                pos_candidates.setdefault(pos, []).append(
                    (indices, val, src, meta))

    # Sort each position's candidates by source priority so the recursive
    # search prefers more-direct sources. The strongest signal is "the
    # letters appear LITERALLY in the clue word" (raw); next come actual
    # synonyms / abbreviations from the DB; positional extraction is the
    # weakest because it requires an extra mechanism (and the literal
    # letter may already be in the clue, as in CORGI's 'I').
    _SRC_RANK = {
        'raw': 0,
        'synonym': 1,
        'abbreviation': 1,
        'reversal': 2,
        'first_letter': 3, 'last_letter': 3,
        'first_n': 3, 'last_n': 3, 'outer_letters': 3,
    }
    for pos, cands in pos_candidates.items():
        cands.sort(key=lambda c: _SRC_RANK.get(c[2], 9))

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
        for indices, val, src, meta in pos_candidates[pos]:
            if any(idx in used for idx in indices):
                continue
            assignments.append((indices, val, src, meta))
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


_SRC_TO_POS_TOKEN = {
    'first_letter':  POS_I_FIRST,
    'first_n':       POS_I_FIRST,
    'last_letter':   POS_I_LAST,
    'last_n':        POS_I_LAST,
    'outer_letters': POS_I_OUTER,
}
_POS_TOKEN_TO_KIND = {
    POS_I_FIRST: 'first',
    POS_I_LAST:  'last',
    POS_I_OUTER: 'outer',
}


def _build_charade_result(wp_words, assignments, answer, db):
    """Build SolveResult for a charade.

    Each assignment is a tuple (indices, val, src) or 4-tuple
    (indices, val, src, meta_dict). meta carries info like the original
    synonym for reversal pieces, so the explanation can be honest.
    """
    # Normalise to 4-tuple for consistent unpacking.
    norm_assignments = []
    for a in assignments:
        if len(a) >= 4:
            norm_assignments.append(a)
        else:
            norm_assignments.append((a[0], a[1], a[2], None))

    assigned = {}
    for indices, val, src, meta in norm_assignments:
        for idx in indices:
            assigned[idx] = (val, src, meta)

    # Determine which positional indicator kinds are needed for this parse,
    # and find a licensing word in the clue for each (don't reuse a fodder word).
    needed_pos_kinds = set()
    for _, _, src, _ in norm_assignments:
        pos_tok = _SRC_TO_POS_TOKEN.get(src)
        if pos_tok is not None:
            needed_pos_kinds.add(_POS_TOKEN_TO_KIND[pos_tok])

    indicator_assignments = {}  # word_idx -> POS_I_* token
    for kind in needed_pos_kinds:
        for k, w in enumerate(wp_words):
            if k in assigned or k in indicator_assignments:
                continue
            if kind in _word_positional_kinds(w, db):
                pos_tok = next(t for t, kk in _POS_TOKEN_TO_KIND.items() if kk == kind)
                indicator_assignments[k] = pos_tok
                break

    # If the parse uses a reversal piece, find a REV_I indicator word in the
    # clue and assign it that role so the explanation can show the reversal.
    has_reversal = any(src == 'reversal' for _, _, src, _ in norm_assignments)
    rev_indicator_word = None
    if has_reversal:
        for k, w in enumerate(wp_words):
            if k in assigned or k in indicator_assignments:
                continue
            cw = _clean(w)
            for wtype, _st, _conf in db.get_indicator_types(cw):
                if wtype == 'reversal':
                    indicator_assignments[k] = REV_I
                    rev_indicator_word = w
                    break
            if k in indicator_assignments:
                break

    word_roles = []
    for k in range(len(wp_words)):
        if k in assigned:
            val, src, meta = assigned[k]
            if src == 'synonym':
                tok = SYN_F
            elif src == 'abbreviation':
                tok = ABR_F
            elif src in _SRC_TO_POS_TOKEN:
                tok = POS_F
            elif src == 'reversal':
                tok = SYN_F  # mapped for compatibility
            else:
                tok = RAW

            # Compose role meta from existing meta + DBE-source detection.
            role_meta = dict(meta) if meta else {}
            display_val = val
            if src == 'reversal' and meta and meta.get('original_synonym'):
                # Show the ORIGINAL synonym in word_roles; mark transform.
                display_val = meta['original_synonym']
                role_meta['transform'] = 'reversed'
                role_meta['reversed_to'] = val
                if rev_indicator_word:
                    role_meta['reversal_indicator'] = rev_indicator_word
            # DBE-source check: did the (effective) synonym come from extras?
            if tok == SYN_F and db.is_extra_synonym(wp_words[k], display_val):
                role_meta['source'] = 'dbe'

            if role_meta:
                word_roles.append((wp_words[k], tok, display_val, role_meta))
            else:
                word_roles.append((wp_words[k], tok, display_val))
        elif k in indicator_assignments:
            word_roles.append((wp_words[k], indicator_assignments[k], None))
        else:
            word_roles.append((wp_words[k], LNK, None))

    pieces_str = ' + '.join(a[1] for a in norm_assignments)
    explanation = '%s = %s' % (pieces_str, answer)
    sig_tokens = [r[1] for r in word_roles if r[1] != LNK]
    sig = SignatureResult(sig_tokens, word_roles, [explanation])

    # Score: higher if more pieces are DB-confirmed synonyms/abbreviations
    confirmed = sum(1 for _, _, src, _ in norm_assignments
                    if src in ('synonym', 'abbreviation'))
    total = len(norm_assignments)
    confidence = 80 + (10 * confirmed // max(total, 1))
    return SolveResult(sig, min(95, confidence), [('charade', 0)], [], {})


def _try_container(wp_words, answer, db):
    """Try container: outer(inner) = answer."""
    answer_len = len(answer)
    n = len(wp_words)

    has_first = _has_positional_indicator(wp_words, db, 'first')
    has_last = _has_positional_indicator(wp_words, db, 'last')
    has_outer = _has_positional_indicator(wp_words, db, 'outer')

    word_vals = []
    for word in wp_words:
        vals = _get_word_values(word, db, answer_len)
        # Positional extractions — gated on a licensing indicator in the clue.
        raw = ''.join(c for c in word.upper() if c.isalpha())
        if raw and len(raw) >= 2:
            if has_first:
                vals.append((raw[0], 'first_letter'))
            if has_last:
                vals.append((raw[-1], 'last_letter'))
            if has_outer:
                vals.append((raw[0] + raw[-1], 'outer_letters'))
            if len(raw) >= 3:
                if has_first:
                    vals.append((raw[:2], 'first_n'))
                if has_last:
                    vals.append((raw[-2:], 'last_n'))
        word_vals.append(vals)

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


def _try_container_charade(wp_words, answer, db):
    """Try container + charade: container produces part of the answer,
    other pieces provide the rest via SYN/ABR/positional.

    E.g. EXCITE = EX(C)IT + E where EXIT contains C, plus E from 'here'.
    """
    answer_len = len(answer)
    n = len(wp_words)
    if n < 3:
        return None

    has_first = _has_positional_indicator(wp_words, db, 'first')
    has_last = _has_positional_indicator(wp_words, db, 'last')
    has_outer = _has_positional_indicator(wp_words, db, 'outer')

    word_vals = []
    for word in wp_words:
        vals = _get_word_values(word, db, answer_len)
        raw = ''.join(c for c in word.upper() if c.isalpha())
        if raw and len(raw) >= 2:
            if has_first:
                vals.append((raw[0], 'first_letter'))
            if has_last:
                vals.append((raw[-1], 'last_letter'))
            if has_outer:
                vals.append((raw[0] + raw[-1], 'outer_letters'))
        word_vals.append(vals)

    # For each pair (outer_word, inner_word), try container producing a substring
    for i in range(n):
        for j in range(n):
            if i == j:
                continue
            for outer, outer_src in word_vals[i]:
                if len(outer) < 2 or len(outer) >= answer_len:
                    continue
                for inner, inner_src in word_vals[j]:
                    if not inner or len(inner) + len(outer) > answer_len:
                        continue
                    # Try inserting inner at each position in outer
                    for pos in range(1, len(outer)):
                        container_result = outer[:pos] + inner + outer[pos:]
                        cr_len = len(container_result)
                        if cr_len >= answer_len or cr_len < 2:
                            continue

                        # Is this container result a substring of the answer?
                        idx = answer.find(container_result)
                        if idx < 0:
                            continue

                        # Remaining letters needed
                        before = answer[:idx]
                        after = answer[idx + cr_len:]
                        if not before and not after:
                            # Full answer — handled by _try_container
                            continue

                        # Try to fill before and after from other words
                        other_indices = [k for k in range(n) if k != i and k != j]

                        # Simple case: one extra piece provides before or after
                        for k in other_indices:
                            for val, src in word_vals[k]:
                                if before and not after and val == before:
                                    return _build_container_charade_result(
                                        wp_words, i, outer, outer_src,
                                        j, inner, inner_src, pos,
                                        [(k, val, src, 'before')],
                                        container_result, answer, db
                                    )
                                if after and not before and val == after:
                                    return _build_container_charade_result(
                                        wp_words, i, outer, outer_src,
                                        j, inner, inner_src, pos,
                                        [(k, val, src, 'after')],
                                        container_result, answer, db
                                    )

                        # Two extra pieces: one before, one after
                        if before and after:
                            for k1 in other_indices:
                                for v1, s1 in word_vals[k1]:
                                    if v1 == before:
                                        for k2 in other_indices:
                                            if k2 == k1:
                                                continue
                                            for v2, s2 in word_vals[k2]:
                                                if v2 == after:
                                                    return _build_container_charade_result(
                                                        wp_words, i, outer, outer_src,
                                                        j, inner, inner_src, pos,
                                                        [(k1, v1, s1, 'before'), (k2, v2, s2, 'after')],
                                                        container_result, answer, db
                                                    )

    return None


def _build_container_charade_result(wp_words, outer_idx, outer, outer_src,
                                     inner_idx, inner, inner_src, insert_pos,
                                     extra_pieces, container_result, answer, db):
    """Build SolveResult for a container+charade compound."""
    word_roles = []
    extra_map = {idx: (val, src) for idx, val, src, position in extra_pieces}

    for k in range(len(wp_words)):
        if k == outer_idx:
            tok = SYN_F if outer_src == 'synonym' else ABR_F
            word_roles.append((wp_words[k], tok, outer))
        elif k == inner_idx:
            tok = SYN_F if inner_src == 'synonym' else ABR_F
            word_roles.append((wp_words[k], tok, inner))
        elif k in extra_map:
            val, src = extra_map[k]
            tok = SYN_F if src == 'synonym' else ABR_F if src == 'abbreviation' else SYN_F
            word_roles.append((wp_words[k], tok, val))
        else:
            ind_types = db.get_indicator_types(_clean(wp_words[k]))
            is_con = any(t in ('container', 'insertion') for t, _, _ in ind_types)
            if is_con:
                word_roles.append((wp_words[k], CON_I, None))
            else:
                word_roles.append((wp_words[k], LNK, None))

    explanation = '%s containing %s + extras = %s' % (outer, inner, answer)
    sig = SignatureResult([SYN_F, CON_I, SYN_F], word_roles, [explanation])
    return SolveResult(sig, 85, [('container_charade', 0)], [], {})


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


def _try_anagram_charade(wp_words, answer, db):
    """Try anagram + charade compound: some words are anagram fodder,
    the rest provide pieces via SYN/ABR/positional, and together they
    produce the answer.

    Strategy:
    1. Identify which words COULD be anagram indicators
    2. For each possible indicator, the remaining words split into
       anagram fodder and charade pieces
    3. Try each split: charade pieces provide fixed letters at start/end
       of the answer, anagram fodder fills the gap
    """
    answer_len = len(answer)
    n = len(wp_words)
    if n < 3:
        return None

    answer_sorted = sorted(answer)
    word_letters = [''.join(c for c in w.upper() if c.isalpha()) for w in wp_words]

    # Find anagram indicator candidates
    ana_indicators = set()
    for k in range(n):
        ind_types = db.get_indicator_types(_clean(wp_words[k]))
        if any(t == 'anagram' for t, _, _ in ind_types):
            ana_indicators.add(k)

    if not ana_indicators:
        return None

    has_first = _has_positional_indicator(wp_words, db, 'first')
    has_last = _has_positional_indicator(wp_words, db, 'last')
    has_outer = _has_positional_indicator(wp_words, db, 'outer')

    # For each indicator, try splitting remaining words
    for ind_idx in ana_indicators:
        remaining = [k for k in range(n) if k != ind_idx]

        # Each remaining word is either anagram fodder or a charade piece.
        # Charade pieces contribute fixed values (SYN/ABR/first letter/last letter).
        # Anagram fodder contributes raw letters to be rearranged.
        #
        # Try: 1 charade piece + rest as fodder, 2 charade pieces + rest, etc.

        # Get possible charade values for each remaining word.
        # Positional pieces gated on a licensing indicator in the clue.
        charade_vals = {}  # word_idx -> list of (value, source_type)
        for k in remaining:
            vals = []
            for val, src in _get_word_values(wp_words[k], db, answer_len):
                vals.append((val, src))
            raw = word_letters[k]
            if raw:
                if has_first:
                    vals.append((raw[0], 'first_letter'))
                if len(raw) >= 2:
                    if has_last:
                        vals.append((raw[-1], 'last_letter'))
                    if has_outer:
                        vals.append((raw[0] + raw[-1], 'outer_letters'))
            charade_vals[k] = vals

        # Try 1 charade piece
        for c1 in remaining:
            fodder_indices = [k for k in remaining if k != c1]
            fodder_letters = ''.join(word_letters[k] for k in fodder_indices)

            for c1_val, c1_src in charade_vals[c1]:
                needed_from_anagram = answer_len - len(c1_val)
                if needed_from_anagram < 3 or needed_from_anagram != len(fodder_letters):
                    continue

                # Try c1 value at start of answer
                if answer.startswith(c1_val):
                    gap = answer[len(c1_val):]
                    if sorted(gap) == sorted(fodder_letters):
                        return _build_anagram_charade_result(
                            wp_words, word_letters, ind_idx,
                            [(c1, c1_val, c1_src)], fodder_indices, answer, db
                        )

                # Try c1 value at end of answer
                if answer.endswith(c1_val):
                    gap = answer[:answer_len - len(c1_val)]
                    if sorted(gap) == sorted(fodder_letters):
                        return _build_anagram_charade_result(
                            wp_words, word_letters, ind_idx,
                            [(c1, c1_val, c1_src)], fodder_indices, answer, db
                        )

        # Try 2 charade pieces
        for ci, c1 in enumerate(remaining):
            for c2 in remaining[ci + 1:]:
                fodder_indices = [k for k in remaining if k != c1 and k != c2]
                fodder_letters = ''.join(word_letters[k] for k in fodder_indices)

                for c1_val, c1_src in charade_vals[c1]:
                    for c2_val, c2_src in charade_vals[c2]:
                        needed = answer_len - len(c1_val) - len(c2_val)
                        if needed < 3 or needed != len(fodder_letters):
                            continue

                        # Try: c1 at start, c2 at end, anagram in middle
                        if answer.startswith(c1_val) and answer.endswith(c2_val):
                            gap = answer[len(c1_val):answer_len - len(c2_val)]
                            if len(gap) == needed and sorted(gap) == sorted(fodder_letters):
                                return _build_anagram_charade_result(
                                    wp_words, word_letters, ind_idx,
                                    [(c1, c1_val, c1_src), (c2, c2_val, c2_src)],
                                    fodder_indices, answer, db
                                )

                        # Try: c2 at start, c1 at end
                        if answer.startswith(c2_val) and answer.endswith(c1_val):
                            gap = answer[len(c2_val):answer_len - len(c1_val)]
                            if len(gap) == needed and sorted(gap) == sorted(fodder_letters):
                                return _build_anagram_charade_result(
                                    wp_words, word_letters, ind_idx,
                                    [(c2, c2_val, c2_src), (c1, c1_val, c1_src)],
                                    fodder_indices, answer, db
                                )

                        # Try: c1 at start, anagram, then c2
                        if answer.startswith(c1_val):
                            for split in range(len(c1_val), answer_len - len(c2_val) + 1):
                                if answer[split:split + len(c2_val)] == c2_val:
                                    before_gap = answer[len(c1_val):split]
                                    after_gap = answer[split + len(c2_val):]
                                    gap = before_gap + after_gap
                                    if len(gap) == needed and sorted(gap) == sorted(fodder_letters):
                                        return _build_anagram_charade_result(
                                            wp_words, word_letters, ind_idx,
                                            [(c1, c1_val, c1_src), (c2, c2_val, c2_src)],
                                            fodder_indices, answer, db
                                        )

    return None


def _build_anagram_charade_result(wp_words, word_letters, ind_idx,
                                   charade_pieces, fodder_indices, answer, db):
    """Build SolveResult for an anagram+charade compound."""
    word_roles = []
    fodder_str = ''.join(word_letters[k] for k in fodder_indices)
    charade_map = {idx: (val, src) for idx, val, src in charade_pieces}

    for k in range(len(wp_words)):
        if k == ind_idx:
            word_roles.append((wp_words[k], ANA_I, None))
        elif k in charade_map:
            val, src = charade_map[k]
            if src == 'synonym':
                tok = SYN_F
            elif src == 'abbreviation':
                tok = ABR_F
            elif src in ('first_letter', 'last_letter', 'outer_letters'):
                tok = SYN_F  # compatibility
            else:
                tok = RAW
            word_roles.append((wp_words[k], tok, val))
        elif k in fodder_indices:
            word_roles.append((wp_words[k], ANA_F, word_letters[k]))
        else:
            word_roles.append((wp_words[k], LNK, None))

    pieces_desc = []
    for idx, val, src in charade_pieces:
        pieces_desc.append('%s(%s)' % (val, wp_words[idx]))
    pieces_desc.append('anagram(%s)' % fodder_str)

    explanation = '%s = %s' % (' + '.join(pieces_desc), answer)
    sig = SignatureResult([ANA_I, ANA_F, SYN_F], word_roles, [explanation])
    return SolveResult(sig, 85, [('anagram_charade', 0)], [], {})


# ============================================================
# Container with deletion-derived inner
# ============================================================
# Specific case: a container compound where the INNER piece is the
# result of applying a deletion indicator to a synonym/abbreviation
# value, e.g. DOODLED = D(OODLE)D where OODLE = OODLES (lots) minus
# last letter ("mostly"), and DD (theologian) splits around it.
#
# This is its own structural test rather than an extension of the
# container test — see feedback_new_type_not_edit.md. New cases get
# their own slot in the cascade; existing tests are not broadened.

def _try_container_with_deletion(wp_words, answer, db):
    """Container whose inner is the deletion-result of a synonym/abbrev.

    Pre-conditions: clue must contain BOTH a container/insertion
    indicator AND a deletion indicator. Returns None otherwise (zero-cost
    early exit; cannot affect any other test).

    A clue word can carry an indicator classification in the DB AND
    serve as fodder — e.g. "theologian" is sometimes catalogued as a
    container indicator (low/medium confidence) but the right parse
    has it as fodder for DD. So we iterate role assignments rather than
    pre-excluding indicator-classified words from being fodder: pick
    one deletion-indicator word and one container-indicator word, then
    let the other words compete for outer/inner.
    """
    answer_len = len(answer)
    n = len(wp_words)
    if n < 4:
        return None

    del_candidates = []  # list of (idx, subtype)
    con_candidates = []  # list of idx
    for i, word in enumerate(wp_words):
        for wtype, subtype, _conf in db.get_indicator_types(_clean(word)):
            if wtype == 'deletion':
                del_candidates.append((i, subtype))
            elif wtype in ('container', 'insertion'):
                con_candidates.append(i)
    if not del_candidates or not con_candidates:
        return None

    def _apply_del(val, subtype):
        if not val or len(val) < 2:
            return None
        if subtype in ('tail', 'last_delete', 'general', None):
            return val[:-1]
        if subtype in ('head', 'first_delete'):
            return val[1:]
        return None

    word_vals = [_get_word_values(w, db, answer_len) for w in wp_words]

    # Try each (deletion-indicator, container-indicator) assignment.
    # The two roles must be distinct words; the OTHER words supply
    # outer and inner candidates.
    for del_idx, subtype in del_candidates:
        for con_idx in con_candidates:
            if del_idx == con_idx:
                continue
            used = {del_idx, con_idx}
            for outer_idx in range(n):
                if outer_idx in used:
                    continue
                for outer_val, outer_src in word_vals[outer_idx]:
                    if len(outer_val) < 2 or len(outer_val) >= answer_len:
                        continue
                    for inner_idx in range(n):
                        if inner_idx in used or inner_idx == outer_idx:
                            continue
                        for inner_val, inner_src in word_vals[inner_idx]:
                            if len(inner_val) < 2:
                                continue
                            derived = _apply_del(inner_val, subtype)
                            if derived is None:
                                continue
                            if (len(derived) + len(outer_val)
                                    != answer_len):
                                continue
                            for pos in range(1, len(outer_val)):
                                prefix = outer_val[:pos]
                                suffix = outer_val[pos:]
                                if not (answer.startswith(prefix)
                                        and answer.endswith(suffix)):
                                    continue
                                middle = answer[pos:answer_len - len(suffix)]
                                if middle == derived:
                                    return _build_container_with_deletion_result(
                                        wp_words,
                                        outer_idx, outer_val, outer_src,
                                        inner_idx, inner_val, inner_src,
                                        derived, subtype,
                                        del_idx, con_idx,
                                        answer, db)
    return None


def _build_container_with_deletion_result(wp_words,
                                          outer_idx, outer_val, outer_src,
                                          inner_idx, inner_val, inner_src,
                                          derived, subtype, del_idx, con_idx,
                                          answer, db):
    """Build SolveResult for a container-with-deletion compound."""
    word_roles = []
    for k in range(len(wp_words)):
        if k == outer_idx:
            tok = SYN_F if outer_src == 'synonym' else ABR_F
            word_roles.append((wp_words[k], tok, outer_val))
        elif k == inner_idx:
            tok = SYN_F if inner_src == 'synonym' else ABR_F
            meta = {'transform': 'deletion', 'subtype': subtype,
                    'derived': derived}
            word_roles.append((wp_words[k], tok, inner_val, meta))
        elif k == del_idx:
            word_roles.append((wp_words[k], DEL_I, None,
                               {'subtype': subtype}))
        elif k == con_idx:
            word_roles.append((wp_words[k], CON_I, None))
        else:
            ind_types = db.get_indicator_types(_clean(wp_words[k]))
            is_con = any(t in ('container', 'insertion')
                         for t, _, _ in ind_types)
            is_del = any(t == 'deletion' for t, _, _ in ind_types)
            if is_con:
                word_roles.append((wp_words[k], CON_I, None))
            elif is_del:
                word_roles.append((wp_words[k], DEL_I, None))
            else:
                word_roles.append((wp_words[k], LNK, None))

    explanation = '%s containing (%s minus letter) = %s' % (
        outer_val, inner_val, answer)
    sig = SignatureResult([SYN_F, DEL_I, SYN_F, CON_I],
                          word_roles, [explanation])
    return SolveResult(sig, 85,
                       [('container_with_deletion', 0)], [], {})


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

    t_start = time.time()

    def _timed_out():
        return (time.time() - t_start) > TRIAGE_TIMEOUT

    total_letters = sum(len(re.sub(r'[^a-zA-Z]', '', w)) for w in wp_words)
    ratio = total_letters / answer_len if answer_len > 0 else 0

    # === Standalone: anagram ===
    # Pure anagram: budget near 1x (most letters used)
    if 0.8 <= ratio <= 2.5:
        result = _try_anagram(wp_words, answer, db)
        if result and result.confidence >= 70:
            return result

    if _timed_out():
        return None

    # Anagram with abbreviation substitution: higher budget because some
    # words contribute short values (W, A) instead of raw letters, and
    # others are excluded (indicators, links)
    if ratio > 2.5:
        result = _try_anagram(wp_words, answer, db)
        if result and result.confidence >= 70:
            return result

    if _timed_out():
        return None

    # === Anagram with positional feed (e.g. first letter + anagram) ===
    if 1.5 <= ratio <= 4.0:
        result = _try_anagram_with_positional(wp_words, answer, db)
        if result:
            return result

    if _timed_out():
        return None

    # === Standalone: pure reversal ===
    result = _try_reversal(wp_words, answer, db)
    if result:
        return result

    if _timed_out():
        return None

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
                if _timed_out():
                    return None
                result = _verify_grammar_roles(wp_words, role_seq, answer, db)
                if result:
                    return result

        if _timed_out():
            return None

        # Mechanism detection: POS bigrams guide which structural tests to try
        if _detect_container(pos_tags):
            result = _try_container(wp_words, answer, db)
            if result:
                return result

    if _timed_out():
        return None

    # === Structural tests without POS guidance ===
    result = _try_container(wp_words, answer, db)
    if result:
        return result

    if _timed_out():
        return None

    # === Container + charade compound ===
    result = _try_container_charade(wp_words, answer, db)
    if result:
        return result

    if _timed_out():
        return None

    # === Container with deletion-derived inner ===
    # Specific case (own structural test, no shared logic with the
    # generic container test). Fires only when both a deletion and a
    # container indicator are present in the wordplay words.
    result = _try_container_with_deletion(wp_words, answer, db)
    if result:
        return result

    if _timed_out():
        return None

    # === Anagram + charade compound ===
    result = _try_anagram_charade(wp_words, answer, db)
    if result:
        return result

    if _timed_out():
        return None

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

    has_first = _has_positional_indicator(wp_words, db, 'first')
    has_last = _has_positional_indicator(wp_words, db, 'last')
    dbe_marked = _dbe_marked_indices(wp_words)

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
                # RAW blocked when word is DBE-marked.
                if raw and raw in answer and i not in dbe_marked:
                    candidates.append((raw, 'raw'))
                if raw:
                    if has_first and raw[0] in answer:
                        candidates.append((raw[0], 'first_letter'))
                    if has_last and len(raw) >= 2 and raw[-1] in answer:
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

            # Skip fodder slot when no candidates — role_seq is a hint, not
            # a contract; the word will be treated as LNK by the result
            # builder. Required so that removing unlicensed positional
            # candidates doesn't break role_seqs whose phantom positional
            # candidates were never actually used in the assignment.
            if candidates:
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
