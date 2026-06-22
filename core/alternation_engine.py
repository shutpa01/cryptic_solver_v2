"""Alternation engine — every-other-letter selection.

An alternation (alternate letters) spells the answer from every OTHER letter of a
contiguous run of clue words, read as ONE letter stream:

    "sordid play"  ->  S O R D I D P L A Y  ->  every other (2nd,4th,...) -> ODDLY
      def "Strangely", indicator "every other character's stripped off".

Both alignments are tried (start at the 1st letter, or the 2nd); the exact match
decides which. Answer-driven and INDICATOR-GATED, exactly like the acrostic engine:
the engine finds the run whose every-other letters EXACTLY spell the answer AND
requires a DB alternation indicator among the leftovers. The exact-letter match is the
strong filter; the indicator gate stops coincidental matches (a short answer can
alternate-match by chance), so the precision is carried by the match, not the indicator.

Emits per-letter provenance: each answer letter is linked to the exact clue letter it
was taken from, mechanism 'alternate'. Isolated per-type engine on the wfw_model
substrate; definition decided upstream; links taken from the link-list only.
"""

from core import engine_common
from core.selection import _letter_atoms
from core.wfw_model import Source, Link, Annotation, Parse

# The wordplay_types the DB uses for alternate-letter indicators. The vocabulary is
# scattered: mostly 'parts' (subtype alternate/even/odd), plus these close synonyms.
_ALT_TYPES = ("parts", "alternating", "alternation", "alternate")


def solve_alternation(ctx, defines, is_link, indicator_types, define_fallback=None,
                      is_dbe=None):
    """Full alternation solve. For each definition split, look in its wordplay for a
    contiguous run of words whose every-other letter spells the answer exactly, with an
    alternation indicator among the leftover words. Returns a Parse (pass/pending), or
    None (abstain — not an alternation, or no alternation indicator present)."""
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
    """Find a contiguous word run whose every-other letters spell the answer, with an
    alternation indicator among the leftovers. Prefers a clean PASS."""
    N = len(answer)
    if len(words) < 2:                           # need the fodder run AND >=1 indicator
        return None
    best_pending = None
    for a in range(len(words)):
        for b in range(a + 1, len(words) + 1):
            run = words[a:b]
            atoms = [at for t in run for at in _letter_atoms(ctx, t)]
            if len(atoms) < 2 * N - 1:           # too few letters to yield N by alternation
                continue
            for offset in (0, 1):
                sel = atoms[offset::2]
                if len(sel) != N or "".join(x.normalized for x in sel) != answer:
                    continue
                residue = set(range(0, a)) | set(range(b, len(words)))
                parse = _build(ctx, answer, split, words, list(range(a, b)), sel,
                               residue, is_link, indicator_types)
                if parse is None:
                    continue
                if parse.status == "pass":
                    return parse
                if best_pending is None:
                    best_pending = parse
    return best_pending


def _find_indicator(words, residue_pos, indicator_types):
    """The longest contiguous leftover run the DB types as an alternate-letter
    indicator (any of _ALT_TYPES). Returns the word positions, or None (no indicator)."""
    best = None
    for wptype in _ALT_TYPES:
        run = engine_common.find_typed_run(words, residue_pos, indicator_types,
                                           wptype, min_length=1)
        if run is not None and (best is None or len(run) > len(best)):
            best = run
    return best


def _build(ctx, answer, split, words, run_pos, sel, residue_pos, is_link,
           indicator_types):
    """Assemble the Parse for a matched run. GATE: an alternation indicator must be
    present among the leftovers. The leftover non-fodder, non-definition words ARE the
    alternation instruction — function/link words become links, the rest the indicator
    phrase (kept whole rather than orphaning a content word)."""
    if _find_indicator(words, residue_pos, indicator_types) is None:
        return None                              # gated: no alternation indicator
    link_pos, ind_pos = [], []
    for p in sorted(residue_pos):
        if is_link and is_link(words[p].text):
            link_pos.append(p)
        else:
            ind_pos.append(p)
    if not ind_pos:
        return None

    run = [words[i] for i in run_pos]
    # one Source per fodder word (the colour unit); value = the letters IT contributed
    sources, src_of_pos, pos_by_atom = [], {}, {}
    for wi, tok in zip(run_pos, run):
        tok_ids = set(tok.atom_ids)
        contributed = [x.normalized for x in sel if x.atom_id in tok_ids]
        src_of_pos[wi] = len(sources)
        sources.append(Source(clue_atom_ids=tok.atom_ids, text=tok.text,
                              value="".join(contributed), mechanism="alternate",
                              source="db"))
        for aid in tok.atom_ids:
            pos_by_atom[aid] = wi

    links = [Link(answer_pos=i + 1, source_index=src_of_pos[pos_by_atom[x.atom_id]],
                  operation="alternation", clue_atom_id=x.atom_id, transform=None)
             for i, x in enumerate(sel)]
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    ind_toks = [words[i] for i in ind_pos]
    annotations = [Annotation(
        clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in ind_toks), role="indicator",
        note="alternation indicator")]
    for p in link_pos:
        annotations.append(Annotation(clue_atom_ids=words[p].atom_ids,
                                      text=words[p].text, role="link",
                                      note="link word"))
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="alternation",
                  solved_by="alternation")
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """Three-state verdict (acrostic-style). pass: every answer letter sourced, every
    clue word accounted, indicator (by construction) and DB definition. pending: the
    definition is provisional (queued). fail: a structural gap (should not occur)."""
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
