"""Rule-based detection of missing positional/extraction indicators.

When a solve fails AND the answer has unfilled positions that could be
filled by extracting letters from a clue word, identify which other
unused clue word might be the licensing indicator. The solver can then
try injecting that (word, type) pair into the indicators DB temporarily,
re-attempt the solve, and queue the suggestion for human review if the
solve completes.

Returns suggestions ranked by simple plausibility heuristics:
  - adjacency to fodder word (indicators usually sit next to their fodder)
  - already-known-as-indicator (a word that's any kind of indicator is
    more likely to be a new kind of indicator than a noun)

A separate Haiku fallback (haiku_indicator.py) runs when the rule-based
path returns nothing or its top candidate doesn't enable a solve.
"""

from .word_analyzer import clean_word


# Map from extract_type to (wordplay_type, subtype) of the licensing
# indicator. The subtype names here MUST match the existing indicators
# table convention so the overlay/queue/dashboard chain stores compatibly.
_EXTRACT_TO_INDICATOR = {
    'first_letter': ('parts', 'first_use'),
    'first_2':      ('parts', 'first_use'),
    'last_letter':  ('parts', 'last_use'),
    'last_2':       ('parts', 'last_use'),
    'outer_2':      ('parts', 'outer_use'),
}

# Order matters: when multiple extracts of the same fodder yield the same
# value, we try the more common cryptic conventions first.
_EXTRACT_ORDER = ['last_letter', 'last_2', 'first_letter',
                  'first_2', 'outer_2']


def _alpha_upper(s):
    return ''.join(c for c in s.upper() if c.isalpha())


def _extracts(raw):
    """Return dict of extract_type -> value for a single word's letters."""
    if not raw:
        return {}
    out = {}
    out['first_letter'] = raw[0]
    if len(raw) >= 2:
        out['last_letter'] = raw[-1]
        out['outer_2']     = raw[0] + raw[-1]
    if len(raw) >= 2:
        out['first_2'] = raw[:2]
        out['last_2']  = raw[-2:]
    return out


def detect_missing_indicator(wp_words, answer, db, used_word_indices=None):
    """Find candidate (indicator_word, type, subtype) suggestions for a
    failed solve.

    Args:
        wp_words: list of words in the wordplay window
        answer: the known answer
        db: RefDB instance (used to skip already-known indicators)
        used_word_indices: set of word indices the solver already assigned
            to a fodder/indicator role. Other indices are 'free' and
            available as fodder OR indicator candidates. Pass None or
            empty for "treat all words as free".

    Returns:
        list of dicts (sorted, best first), each with:
            indicator_word: str — the suggested indicator clue word
            indicator_type: str — wordplay_type ('parts', 'acrostic', ...)
            subtype: str        — e.g. 'last_use', 'outer_use'
            fodder_word: str    — the word being acted on
            extract_type: str   — how the fodder yields the gap letters
            extract_value: str  — the actual letters
            score: int          — higher = more plausible
    """
    used = set(used_word_indices) if used_word_indices else set()
    n = len(wp_words)
    answer_clean = _alpha_upper(answer)
    free_indices = [i for i in range(n) if i not in used]
    if len(free_indices) < 2:
        return []  # need at least one fodder + one indicator candidate

    # Pass 1: find (fodder_idx, extract_type, extract_value) where the
    # extract is a substring of the answer (or of its reverse — common in
    # reversal_charade clues where the extracted piece is reversed).
    answer_rev = answer_clean[::-1]
    fodder_candidates = []
    for fi in free_indices:
        raw = _alpha_upper(wp_words[fi])
        if len(raw) < 2:
            continue
        extracts = _extracts(raw)
        for ext_type in _EXTRACT_ORDER:
            ext = extracts.get(ext_type)
            if not ext or ext == answer_clean:
                continue
            if ext in answer_clean or ext in answer_rev:
                fodder_candidates.append((fi, ext_type, ext))

    if not fodder_candidates:
        return []

    # Pass 2: for each fodder candidate, score every other free word as a
    # potential indicator. Skip words that are already a known indicator of
    # the SAME type — those don't need enrichment.
    suggestions = []
    for fi, ext_type, ext_val in fodder_candidates:
        wp_type, subtype = _EXTRACT_TO_INDICATOR[ext_type]
        for ii in free_indices:
            if ii == fi:
                continue
            indicator_word = wp_words[ii]
            cw = clean_word(indicator_word)
            if not cw:
                continue

            existing_types = db.get_indicator_types(cw)
            if any(t == wp_type and st == subtype
                   for t, st, _ in existing_types):
                continue  # already known with this exact role
            # If the word is already a HIGH-confidence indicator of any
            # OTHER type, its role is settled — don't suggest hijacking it.
            # 'flipping' (HIGH reversal) shouldn't get re-cast as a
            # positional indicator just because we need a positional one.
            if any(c == 'high' for _t, _s, c in existing_types):
                continue

            adjacency = 2 if abs(ii - fi) == 1 else 0
            # Words known as some non-high-conf indicator might plausibly
            # carry a similar role; give a small bonus.
            already_some_indicator = 1 if existing_types else 0
            # Longer extracts are far more specific and trustworthy than
            # single-letter extracts (which match almost any fodder).
            extract_len_bonus = len(ext_val)
            score = adjacency + already_some_indicator + extract_len_bonus

            suggestions.append({
                'indicator_word': indicator_word,
                'indicator_type': wp_type,
                'subtype': subtype,
                'fodder_word': wp_words[fi],
                'extract_type': ext_type,
                'extract_value': ext_val,
                'score': score,
            })

    suggestions.sort(key=lambda s: s['score'], reverse=True)
    # Dedup: same (indicator_word, indicator_type, subtype) appearing
    # multiple times keeps only the highest-scored entry.
    seen = set()
    deduped = []
    for s in suggestions:
        key = (s['indicator_word'].lower(),
               s['indicator_type'], s['subtype'])
        if key in seen:
            continue
        seen.add(key)
        deduped.append(s)
    return deduped
