"""Acrostic engine — initial-letter (and final-letter) selection.

A pure acrostic spells the answer from the FIRST letter of each word in a
contiguous run, flagged by an acrostic indicator: PLUS = [P]uzzle [l]ike
[u]ltimate [s]olver, indicator "Starts off". The finals variant takes the LAST
letter of each word ("at last", "endings").

Answer-driven and INDICATOR-GATED. Per SOLVER_REDESIGN §4/§5.2 acrostic is a gated
operation (unlike the free hidden/DD quick checks): the engine finds the run whose
selected letters EXACTLY spell the answer AND requires a DB acrostic indicator among
the leftovers. The exact-letter match is the strong filter; the indicator gate stops
coincidental initial-runs (the indicator table is noisy — it lumps first-letter
DELETION words like "topless" under acrostic — so the exact match must carry the
precision, not the indicator alone).

Emits the §5.5 per-letter provenance: each answer letter is linked to the exact clue
letter it was taken from, mechanism first_letter / last_letter (§3.3). Isolated
per-type engine on the wfw_model substrate (the 2026-06-02 decision); definition
decided upstream; links from the link-list only, classified last. Paired with its own
screen, core/acrostic_screen.py.
"""

from core import selection
from core import engine_common
from core.wfw_model import Source, Link, Annotation, Parse

_MODE_MECHANISM = {"first": "first_letter", "last": "last_letter"}


def solve_acrostic(ctx, defines, is_link, indicator_types, define_fallback=None,
                   is_dbe=None):
    """Full acrostic solve. Walk each confirmed definition split; in its wordplay,
    look for a contiguous run of N words (N = answer length) whose first (or last)
    letters spell the answer exactly, with an acrostic indicator among the leftover
    wordplay words. Returns a Parse (pass / pending) or None (abstain — not an
    acrostic, or no acrostic indicator present)."""
    from core.definition_engine import find_definitions
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3 or indicator_types is None:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    best_pending = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        parse = _try_split(ctx, answer, split, words, is_link, indicator_types)
        if parse is None:
            continue
        if parse.status == "pass":
            return parse                         # answer-driven exact match — done
        if best_pending is None:
            best_pending = parse
    return best_pending


def _try_split(ctx, answer, split, words, is_link, indicator_types):
    """Find a run in this split's wordplay whose selected letters spell the answer and
    whose leftovers carry an acrostic indicator. Prefers a clean PASS."""
    N = len(answer)
    if len(words) < N + 1:                       # need the run AND >=1 indicator word
        return None
    best_pending = None
    for mode in ("first", "last"):
        for start in range(0, len(words) - N + 1):
            run = words[start:start + N]
            sel = selection.selected(ctx, run, mode)
            if sel is None or "".join(c for c, _ in sel) != answer:
                continue
            residue = list(range(0, start)) + list(range(start + N, len(words)))
            parse = _build(ctx, answer, split, words, run, sel, mode, set(residue),
                           is_link, indicator_types)
            if parse is None:
                continue
            if parse.status == "pass":
                return parse
            if best_pending is None:
                best_pending = parse
    return best_pending


def _find_indicator(words, residue_pos, indicator_types):
    """The acrostic indicator among the leftover words: the LONGEST contiguous run of
    leftover positions whose joined phrase the DB types 'acrostic' (so "at first" is
    taken whole). Returns the list of word positions, or None (no acrostic indicator
    -> the engine abstains; acrostic is gated). Delegates to the shared primitive."""
    return engine_common.find_typed_run(words, residue_pos, indicator_types,
                                        "acrostic", min_length=1)


def _build(ctx, answer, split, words, run, sel, mode, residue_pos, is_link,
           indicator_types):
    """Assemble the Parse for a matched run, classifying the leftovers as indicator +
    links. Returns None if no acrostic indicator is present (gate) or a leftover
    content word is unaccounted (not a clean acrostic — abstain rather than claim)."""
    ind_pos = _find_indicator(words, residue_pos, indicator_types)
    if ind_pos is None:
        return None                              # gated: no acrostic indicator
    ind_set = set(ind_pos)
    link_pos = []
    for p in sorted(residue_pos):
        if p in ind_set:
            continue
        if is_link and is_link(words[p].text):
            link_pos.append(p)
        else:
            return None                          # an unaccounted content word

    sources = [Source(clue_atom_ids=tok.atom_ids, text=tok.text, value=char,
                      mechanism=_MODE_MECHANISM[mode], source="db")
               for (char, _aid), tok in zip(sel, run)]
    links = [Link(answer_pos=i + 1, source_index=i, operation="acrostic",
                  clue_atom_id=aid, transform=None)
             for i, (_char, aid) in enumerate(sel)]
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    ind_toks = [words[i] for i in ind_pos]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="acrostic indicator")]
    for p in link_pos:
        annotations.append(Annotation(clue_atom_ids=words[p].atom_ids,
                                      text=words[p].text, role="link",
                                      note="link word"))
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="acrostic", solved_by="acrostic")
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """Three-state verdict, acrostic-specific. pass: every answer letter sourced, every
    clue word accounted, DB indicator (guaranteed by construction) and DB definition.
    pending: the definition is provisional (AI fallback, queued). fail: a structural
    gap (unaccounted word / no definition) — should not occur given _build rejects."""
    warnings = []
    n_letters = sum(1 for a in ctx.answer_atoms if a.kind == "letter")
    if len(parse.links) != n_letters:
        warnings.append("not every answer letter has a source")
    w_unaccounted = engine_common.unaccounted_words_warning(ctx, parse)
    if w_unaccounted:
        warnings.append(w_unaccounted)
    missing = parse.unexplained_words(ctx)
    w_def = engine_common.definition_warning(parse)
    if w_def:
        warnings.append(w_def)
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"
