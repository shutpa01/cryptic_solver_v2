"""Answer-constrained span/value recovery for container clues."""
from __future__ import annotations


def find_container_span_value_suggestion(clue_text, answer, db, candidates):
    """Find one validated phrase -> shell candidate for a failed container.

    The only generative step is deriving the shell by subtracting a known
    short inner atom from the answer. The semantic verifier is then asked
    about that single phrase/value pair.
    """
    if not candidates:
        return None

    from .haiku_span_value import verify_span_value

    answer = "".join(c for c in answer.upper() if c.isalpha())
    if len(answer) < 5:
        return None

    validation_calls = 0
    max_validation_calls = 1

    for _def_phrase, wp_words in candidates:
        if len(wp_words) < 3:
            continue

        indicator_sets = _container_indicator_index_sets(wp_words, db)
        if not indicator_sets:
            continue

        atom_candidates = _short_inner_atom_candidates(wp_words, answer, db)
        if not atom_candidates:
            continue

        for indicator_idxs in indicator_sets[:3]:
            for inner_idx, inner in atom_candidates[:8]:
                if inner_idx in indicator_idxs:
                    continue
                for pos in _occurrences(answer, inner):
                    shell = answer[:pos] + answer[pos + len(inner):]
                    if len(shell) < 3:
                        continue
                    if not db.is_real_word(shell):
                        continue
                    if shell[:pos] + inner + shell[pos:] != answer:
                        continue
                    for phrase in _remaining_phrase_candidates(
                            wp_words, indicator_idxs | {inner_idx}):
                        if _phrase_looks_operational(phrase, db):
                            continue
                        if _span_value_known_to_db(phrase, shell, db):
                            return phrase, shell
                        if validation_calls >= max_validation_calls:
                            return None
                        validation_calls += 1
                        if verify_span_value(
                                clue_text, phrase, shell, answer,
                                role_hint="container outer/shell"):
                            return phrase, shell
    return None


def span_suggestion_actually_used(sr_obj, phrase, value):
    if sr_obj is None or sr_obj.result is None:
        return False
    phrase_clean = phrase.lower().strip(",.;:!?\"'()-")
    value_clean = "".join(c for c in value.upper() if c.isalpha())
    for word, _tok, val, *_ in sr_obj.result.word_roles:
        word_clean = word.lower().strip(",.;:!?\"'()-")
        role_value = "".join(c for c in (val or "").upper() if c.isalpha())
        if word_clean == phrase_clean and role_value == value_clean:
            return True
    return False


def _container_indicator_index_sets(words, db):
    out = []
    n = len(words)
    for span in (1, 2, 3):
        for i in range(n - span + 1):
            phrase = " ".join(words[i:i + span])
            ind_types = db.get_indicator_types(phrase.lower())
            if any(t in ("container", "insertion") for t, _, _ in ind_types):
                out.append(set(range(i, i + span)))
    return out


def _short_inner_atom_candidates(words, answer, db):
    out = []
    seen = set()
    for i, word in enumerate(words):
        vals = []
        vals.extend(db.get_abbreviations(word.lower()))
        vals.extend(db.get_synonyms_substring_of(word.lower(), answer))
        for val in vals:
            clean = "".join(c for c in val.upper() if c.isalpha())
            key = (i, clean)
            if (not clean or clean == answer or clean not in answer
                    or len(clean) > 3 or key in seen):
                continue
            out.append((i, clean))
            seen.add(key)
    return out


def _occurrences(text, needle):
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos < 0:
            return
        yield pos
        start = pos + 1


def _remaining_phrase_candidates(words, used_idxs):
    """Prefer contiguous multi-word leftovers; they are the GT2 target."""
    n = len(words)
    phrases = []
    for span in range(min(4, n), 1, -1):
        for i in range(n - span + 1):
            idxs = set(range(i, i + span))
            if idxs & used_idxs:
                continue
            phrase = " ".join(words[i:i + span])
            if phrase not in phrases:
                phrases.append(phrase)
    return phrases


def _phrase_looks_operational(phrase, db):
    words = phrase.lower().split()
    if not words:
        return True
    for word in (words[0], words[-1]):
        if db.is_link_word(word):
            return True
        if db.get_indicator_types(word):
            return True
    return False


def _span_value_known_to_db(phrase, value, db):
    value = "".join(c for c in value.upper() if c.isalpha())
    vals = []
    vals.extend(db.get_abbreviations(phrase.lower()))
    vals.extend(db.get_synonyms(phrase.lower(), max_len=len(value)))
    return value in {"".join(c for c in v.upper() if c.isalpha()) for v in vals}
