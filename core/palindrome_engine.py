"""Palindrome engine — the answer reads the same both ways (SAGAS, ROTOR).

A palindrome clue is a DEFINITION plus a palindrome indicator; the answer happens to
be its own reverse, so the wordplay produces NO letters of its own (unlike reversed
hidden, which draws a run from the clue). The engine therefore:

  - takes the edge definition the shared stage confirms (definition decided upstream),
  - requires the answer to equal its reverse (the answer-driven gate),
  - requires a palindrome indicator among the wordplay words (the indicator gate;
    without it a coincidental palindrome like EYE/PIP would be falsely claimed),
  - classifies the remaining wordplay words as links — every word accounted.

There is no piece lookup and no per-letter clue provenance (nothing in the clue
sources the letters); the breakdown is definition + indicator + the symmetry fact.
Isolated per-type engine on the wfw_model substrate; paired with core/palindrome_screen.
"""

from core import engine_common
from core import palindrome_indicators
from core.wfw_model import Source, Annotation, Parse


def solve_palindrome(ctx, defines, is_link, define_fallback=None, is_dbe=None):
    """Full palindrome solve. Abstains (None) unless the answer is a palindrome, a
    palindrome indicator is present, and the leftover words are all links. Returns a
    Parse (pass / pending) or None."""
    from core.definition_engine import find_definitions
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3 or answer != answer[::-1]:
        return None                                  # not a palindrome — abstain cheaply
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    # Among buildable splits prefer a PASS over a pending, then the SHORTEST definition
    # (definitions are edge-anchored and minimal — so "Legends", not "Legends from the",
    # with the link words left to the wordplay side).
    best, best_key = None, None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        parse = _build(ctx, answer, split, words, is_link)
        if parse is None:
            continue
        key = (0 if parse.status == "pass" else 1, len(split.phrase.split()))
        if best_key is None or key < best_key:
            best, best_key = parse, key
    return best


def _build(ctx, answer, split, words, is_link):
    """Assemble the Parse: edge definition + palindrome indicator, leftovers as links.
    Returns None if no palindrome indicator is present (gate) or a leftover content
    word is unaccounted (not a clean palindrome clue — abstain rather than claim)."""
    ind_pos = palindrome_indicators.find_indicator(words)
    if ind_pos is None:
        return None                                  # gated: no palindrome indicator
    ind_set = set(ind_pos)
    link_pos = []
    for p in range(len(words)):
        if p in ind_set:
            continue
        if is_link and is_link(words[p].text):
            link_pos.append(p)
        else:
            return None                              # unaccounted content word -> abstain

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    ind_toks = [words[i] for i in ind_pos]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="palindrome indicator")]
    for p in link_pos:
        annotations.append(Annotation(clue_atom_ids=words[p].atom_ids,
                                      text=words[p].text, role="link",
                                      note="link word"))
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=[], links=[], annotations=annotations,
                  definition=definition, operation="palindrome",
                  solved_by="palindrome")
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """Three-state verdict. pass: every clue word accounted + DB definition (indicator
    guaranteed by construction, symmetry by the solve gate). pending: provisional
    (AI-fallback) definition, queued. fail: a structural gap (should not occur given
    _build rejects an unaccounted word)."""
    warnings = []
    missing = parse.unexplained_words(ctx)
    w_unaccounted = engine_common.unaccounted_words_warning(ctx, parse)
    if w_unaccounted:
        warnings.append(w_unaccounted)
    w_def = engine_common.definition_warning(parse)
    if w_def:
        warnings.append(w_def)
    from core import role_validity
    bad = role_validity.unbacked_roles(parse)
    if bad:
        parse.warnings = warnings + bad
        parse.status = "fail"
        return
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"
