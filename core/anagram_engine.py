"""Anagram engine — WORDPLAY-ONLY, fodder by letter-match + POS segmentation.

The definition is decided upstream; this engine receives only the wordplay words and
explains how they make the answer.

It does NOT rely on rigid catalog indicator slots (those over-constrained the
indicator and orphaned content words — see memory: anagram-grammar-segmentation).
Instead, like the hidden engine, it works flexibly:

  1. FODDER by evidence: find the contiguous run of wordplay words whose letters
     (with a contraction/possessive fallback, "Lionel's" -> LIONEL) equal the
     answer's letters exactly. That run is the fodder — robust, no DB needed.
  2. SEGMENT the rest with grammar: of the remaining words there must be exactly ONE
     contiguous run carrying an anagram indicator — that run IS the indicator phrase
     (outer function words peeled off as links, interior ones kept, e.g. "needing to
     be reorganized"). Any other remaining run is connective glue and must be
     function/connective words (links); a stray content noun there is left
     unaccounted and the parse fails honestly. Two indicator runs (fodder flanked by
     indicators) is rejected — not a real anagram shape.

INDICATOR-GATED: an anagram is never claimed without a confirmed anagram indicator.
HYGIENE: reject a self-anagram (a word IS the answer) and an exact reversal.

Provenance is per-fodder-word for COLOUR (each fodder word a colour, its letters
carry into the answer tiles), though an anagram's letters are formally span-level —
where two fodder words share a letter the assignment is one valid attribution.

Pure: a wfw_atoms context + the wordplay tokens + injected predicates. POS comes from
core.grammar (spaCy), degrading gracefully to is_link-only when unavailable.
"""

from collections import Counter
from itertools import combinations

from core import contractions, grammar
from core.wfw_model import Source, Link, Annotation, Parse

# Function-word POS — never the operative indicator; peeled from an indicator run
# and accepted as links in a connective run.
FUNCTION_POS = {"ADP", "PART", "AUX", "DET", "CCONJ", "SCONJ"}
# Allowed in a no-indicator connective/glue run ("found in" = VERB + ADP).
GLUE_POS = FUNCTION_POS | {"VERB", "ADV"}


def _raw(text):
    return "".join(c for c in (text or "").upper() if c.isalpha())


def _answer_letters(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _is_anag_indicator(text, indicator_types):
    try:
        return "anagram" in (indicator_types(text) or set())
    except Exception:
        return False


def _fodder_letters(tokens):
    """The fodder's letters as written, and the contraction-stripped fallback —
    so a possessive "Lionel's" can contribute LIONEL, not LIONELS."""
    as_written = [_raw(t.text) for t in tokens]
    stripped = [_raw(contractions.strip_suffixes(t.text)) for t in tokens]
    return as_written, stripped


def _segment(words, pos, i, j, is_link, indicator_types):
    """Classify the words OUTSIDE the fodder run words[i:j] into the indicator phrase
    and link words. Returns (indicator_tokens, link_tokens, unaccounted) or None when
    there is not exactly one indicator run (0 = no indicator; >=2 = fodder flanked)."""
    n = len(words)
    runs = [r for r in (list(range(0, i)), list(range(j, n))) if r]

    def is_ind(k):
        return _is_anag_indicator(words[k].text, indicator_types)

    def is_fn(k):
        return (is_link and is_link(words[k].text)) or (pos[k] in FUNCTION_POS)

    indicator_runs = [r for r in runs if any(is_ind(k) for k in r)]
    if len(indicator_runs) != 1:
        return None
    ind_run = indicator_runs[0]

    link_idx, unacc_idx = [], []
    # Peel OUTER function words off the indicator run (they are links); keep the
    # inner core as the indicator phrase (interior function words like "to be" stay).
    lo, hi = 0, len(ind_run) - 1
    while lo <= hi and is_fn(ind_run[lo]) and not is_ind(ind_run[lo]):
        link_idx.append(ind_run[lo])
        lo += 1
    while hi >= lo and is_fn(ind_run[hi]) and not is_ind(ind_run[hi]):
        link_idx.append(ind_run[hi])
        hi -= 1
    core = ind_run[lo:hi + 1]
    if not any(is_ind(k) for k in core):
        return None                              # peeled the indicator away

    # Other runs are connective glue: function/connective words -> links; a stray
    # content word (noun/adj/propn) is left unaccounted (honest fail).
    for r in runs:
        if r is ind_run:
            continue
        for k in r:
            if (is_link and is_link(words[k].text)) or pos[k] in GLUE_POS:
                link_idx.append(k)
            else:
                unacc_idx.append(k)

    indicator = [words[k] for k in core]
    links = [words[k] for k in sorted(link_idx)]
    unaccounted = [words[k] for k in sorted(unacc_idx)]
    return indicator, links, unaccounted


def solve_anagram(ctx, wordplay_tokens, is_link, indicator_types, templates=None):
    """Solve the WORDPLAY as an anagram. Returns a Parse (definition attached by the
    caller) or None. `templates` is accepted for call-site compatibility but unused —
    this engine is a direct search, not catalog-slot driven."""
    answer = _answer_letters(ctx)
    words = [t for t in wordplay_tokens if t.kind == "word"]
    n = len(words)
    if len(answer) < 3 or n < 2:
        return None
    if any(_raw(t.text) == answer for t in words):
        return None                              # self-anagram

    pos = grammar.pos_tags([t.text for t in words]) or [None] * n

    # Candidate fodder runs: contiguous, longest first (more words as fodder leaves
    # the cleanest indicator), then by position. Must leave room for an indicator.
    spans = sorted((( i, j) for i in range(n) for j in range(i + 1, n + 1)
                    if j - i < n),
                   key=lambda s: (-(s[1] - s[0]), s[0]))

    best = None
    for i, j in spans:
        span = list(range(i, j))
        # A link word INSIDE the fodder span may be a joiner whose letters are NOT
        # fodder ("old AND new") rather than fodder itself ("is done" -> EDISON). Try
        # excluding subsets of interior link words, FEWEST exclusions first (so a link
        # that IS fodder is kept by default); the exact letter-match decides.
        link_pos = [k for k in span if is_link and is_link(words[k].text)]
        matched = None
        for r in range(0, len(link_pos) + 1):
            for excl in combinations(link_pos, r):
                exclset = set(excl)
                kept = [words[k] for k in span if k not in exclset]
                if not kept:
                    continue
                as_written = [_raw(t.text) for t in kept]
                stripped = [_raw(contractions.strip_suffixes(t.text)) for t in kept]
                if sorted("".join(as_written)) == sorted(answer):
                    tl = as_written
                elif sorted("".join(stripped)) == sorted(answer):
                    tl = stripped
                else:
                    continue
                if "".join(tl)[::-1] == answer:
                    continue                     # exact reversal, not anagram
                matched = (kept, tl, [words[k] for k in excl])
                break
            if matched:
                break
        if not matched:
            continue
        fodder, token_letters, interior_links = matched

        seg = _segment(words, pos, i, j, is_link, indicator_types)
        if seg is None:
            continue
        indicator_tokens, link_tokens, unaccounted = seg
        parse = _build(ctx, answer, fodder, token_letters, indicator_tokens,
                       link_tokens + interior_links, unaccounted)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best


def _build(ctx, answer, fodder_tokens, token_letters, indicator_tokens, link_tokens,
           unaccounted):
    """Assemble the wordplay Parse (definition attached by the caller). One Source
    per fodder word (value = the letters it contributes, possibly stripped), each a
    colour; every answer letter is assigned to a fodder word that supplied it."""
    sources, remaining = [], []
    for t, raw in zip(fodder_tokens, token_letters):
        remaining.append([len(sources), Counter(raw)])
        sources.append(Source(clue_atom_ids=t.atom_ids, text=t.text,
                              value=raw, mechanism="anagram_fodder"))

    links = []
    for pos_i, ch in enumerate(answer, start=1):
        si = 0
        for entry in remaining:
            if entry[1].get(ch, 0) > 0:
                entry[1][ch] -= 1
                si = entry[0]
                break
        links.append(Link(answer_pos=pos_i, source_index=si, operation="anagram",
                          clue_atom_id=None, transform="anagram_of"))

    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in indicator_tokens for aid in t.atom_ids),
        text=" ".join(t.text for t in indicator_tokens),
        role="indicator", note="anagram indicator")]
    for t in link_tokens:
        annotations.append(Annotation(clue_atom_ids=t.atom_ids, text=t.text,
                                      role="link", note="link word"))

    parse = Parse(
        clue_text=ctx.clue_text, answer_text=ctx.answer_text,
        sources=sources, links=links, annotations=annotations,
        definition=None, operation="anagram", solved_by="catalog")
    parse.template_id = None
    parse.matched_signature = None
    _verify_wordplay(parse, indicator_tokens, unaccounted)
    return parse


def _verify_wordplay(parse, indicator_tokens, unaccounted):
    """Verdict on the WORDPLAY only (the caller folds in the definition):
      pass — full letter coverage, an indicator present, every word accounted.
      fail — a wordplay word is unaccounted (surfaced honestly).
    """
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully covered by the fodder")
    if not indicator_tokens:
        warnings.append("no anagram indicator found")
    if unaccounted:
        warnings.append("these wordplay words are unaccounted for: "
                        + ", ".join(repr(t.text) for t in unaccounted))
    parse.warnings = warnings
    parse.status = "pass" if not warnings else "fail"
