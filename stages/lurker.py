from resources import norm_letters


def _letters_only_stream(clue_text: str) -> str:
    return "".join(ch.lower() for ch in clue_text if ch.isalpha())


def _word_spans_letters_only(clue_text: str) -> list[tuple[int, int]]:
    """
    Word spans in letters-only coordinates.
    Each span is [start, end) for one word.
    """
    spans = []
    idx = 0
    in_word = False
    start = 0

    for ch in clue_text:
        if ch.isalpha():
            if not in_word:
                in_word = True
                start = idx
            idx += 1
        else:
            if in_word:
                spans.append((start, idx))
                in_word = False

    if in_word:
        spans.append((start, idx))

    return spans


def _is_valid_lurker_span(span: tuple[int, int], word_spans: list[tuple[int, int]]) -> bool:
    """
    A valid lurker span must:
      - cross at least one word boundary (span 2+ words)
      - take a proper suffix of the first word
      - take a proper prefix of the last word
      - include complete middle words (if any)
    """
    s, e = span

    # find words intersected by the span
    touched = []
    for ws, we in word_spans:
        if s < we and e > ws:
            touched.append((ws, we))

    # must touch at least two words, up to three
    if len(touched) < 2 or len(touched) > 3:
        return False

    # Check touched words are adjacent
    for i in range(len(touched) - 1):
        if touched[i][1] != touched[i + 1][0]:
            return False

    # First word: must start inside (proper suffix)
    w1s, w1e = touched[0]
    if not (w1s < s < w1e):
        return False

    # Last word: must end inside (proper prefix)
    wLs, wLe = touched[-1]
    if not (wLs < e < wLe):
        return False

    return True


def _candidate_bounded_hypotheses(clue_text: str, enumeration: int, candidates) -> list:
    if not candidates:
        return []

    norm_candidates = {
        norm_letters(c): c
        for c in candidates
        if len(norm_letters(c)) == enumeration
    }
    if not norm_candidates:
        return []

    stream = _letters_only_stream(clue_text)
    n = len(stream)
    L = enumeration
    if n < L:
        return []

    word_spans = _word_spans_letters_only(clue_text)
    hypotheses = []

    for i in range(0, n - L + 1):
        span = (i, i + L)

        if not _is_valid_lurker_span(span, word_spans):
            continue

        window = stream[i:i + L]

        if window in norm_candidates:
            hypotheses.append({
                "answer": norm_candidates[window],
                "direction": "forward",
                "letters": window,
                "span": span,
                "solve_type": "lurker_provisional",
                "confidence": "provisional",
            })

        rev = window[::-1]
        if rev in norm_candidates:
            hypotheses.append({
                "answer": norm_candidates[rev],
                "direction": "reverse",
                "letters": rev,
                "span": span,
                "solve_type": "lurker_provisional",
                "confidence": "provisional",
            })

    return hypotheses


def _wordlist_fallback_hypotheses(clue_text: str, enumeration: int, wordlist) -> list:
    if not wordlist:
        return []

    wl_lookup = {
        norm_letters(w): w
        for w in wordlist
        if len(norm_letters(w)) == enumeration
    }
    if not wl_lookup:
        return []

    stream = _letters_only_stream(clue_text)
    n = len(stream)
    L = enumeration
    if n < L:
        return []

    word_spans = _word_spans_letters_only(clue_text)
    hypotheses = []

    for i in range(0, n - L + 1):
        span = (i, i + L)

        if not _is_valid_lurker_span(span, word_spans):
            continue

        window = stream[i:i + L]

        if window in wl_lookup:
            hypotheses.append({
                "answer": wl_lookup[window],
                "direction": "forward",
                "letters": window,
                "span": span,
                "solve_type": "lurker_provisional",
                "confidence": "low",
            })

        rev = window[::-1]
        if rev in wl_lookup:
            hypotheses.append({
                "answer": wl_lookup[rev],
                "direction": "reverse",
                "letters": rev,
                "span": span,
                "solve_type": "lurker_provisional",
                "confidence": "low",
            })

    return hypotheses


def generate_lurker_hypotheses(
    clue_text: str,
    enumeration: int,
    candidates=None,
    wordlist=None,
) -> list:
    """
    Generate provisional lurker hypotheses.

    A valid lurker must take a proper suffix of one word
    and a proper prefix of the next.
    """


    if not isinstance(enumeration, int) or enumeration <= 0:
        return []

    cand_hits = _candidate_bounded_hypotheses(
        clue_text=clue_text,
        enumeration=enumeration,
        candidates=candidates or [],
    )
    if cand_hits:
        return cand_hits

    return _wordlist_fallback_hypotheses(
        clue_text=clue_text,
        enumeration=enumeration,
        wordlist=wordlist or [],
    )
