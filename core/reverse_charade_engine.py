"""Reverse-of-charade — the WHOLE assembled charade is reversed to give the answer.

DISTINCT from reversal_charade (which reverses ONE piece IN PLACE inside an otherwise
forward charade): here the pieces concatenate in clue order and the ENTIRE result is
reversed, which FLIPS the piece order in the answer:

    TRAIN = reverse(new->N + international->I + art->ART) = reverse(NIART)
            "Revolutionary" = reversal indicator, "school" = definition

reversal_charade's monotonic clue-order tiling cannot reach this (the answer's first
letters come from the LAST clue piece). ANSWER-DRIVEN: tile reverse(answer) left-to-right
with the clue words IN CLUE ORDER (so the forward concatenation == reverse(answer), i.e.
the assembled fodder reversed == answer). Each piece is a DB value (synonym/abbreviation)
or the word's literal letters. Requires >= 2 pieces (a single reversed value is the plain
reversal engine), a reversal indicator among the leftovers, the rest links. Pure;
definition decided upstream; own verifier (role_validity-gated). A fresh stage; edits no
other engine.
"""

from core.wordplay import raw
from core.wfw_model import Source, Link, Annotation, Parse

MAX_RUN = 4
_VALUE_MECH = ("synonym", "abbreviation")


def solve_reverse_charade(ctx, defines, lookup_all, is_link, indicator_types,
                          define_fallback=None, is_dbe=None):
    """Full reverse-of-charade solve. First clean PASS, else best parse, else None."""
    from core.definition_engine import find_definitions
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    best = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 2:
            continue
        parse = _assemble(ctx, answer, split, words, lookup_all, is_link, indicator_types)
        if parse is not None:
            if parse.status == "pass":
                return parse
            if best is None:
                best = parse
    return best


def _piece_values(words, a, b, lookup_all):
    """(value, mechanism) for words[a:b]: DB synonym/abbreviation, then the literal letters
    (raw) as a last option — the reversal fodder may be the word shown (e.g. art -> ART)."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out, seen = [], set()
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in seen:
            seen.add(v)
            out.append((v, mech))
    lit = raw(phrase)
    if lit and lit not in seen:
        out.append((lit, "raw"))
    return out


def _assemble(ctx, answer, split, words, lookup_all, is_link, indicator_types):
    n, N = len(words), len(answer)
    rev = answer[::-1]                            # forward fodder must spell reverse(answer)

    from core.engine_common import has_typed_indicator
    if not has_typed_indicator(words, indicator_types, "reversal"):
        return None                              # GATE (phrase-aware): reversal indicator required

    vcache = {}

    def vals(a, b):
        if (a, b) not in vcache:
            vcache[(a, b)] = _piece_values(words, a, b, lookup_all)
        return vcache[(a, b)]

    def dfs(wi, pos, pieces, gaps):
        if pos == len(rev):
            return _finalize(ctx, answer, split, words, pieces,
                             gaps + list(range(wi, n)), indicator_types, is_link)
        if wi >= n:
            return None
        # skip wi as a gap (reversal indicator / link)
        r = dfs(wi + 1, pos, pieces, gaps + [wi])
        if r is not None:
            return r
        # place a piece starting at wi (its forward value is the next run of reverse(answer))
        for b in range(wi + 1, min(wi + MAX_RUN, n) + 1):
            for V, mech in vals(wi, b):
                if V and rev.startswith(V, pos):
                    r = dfs(b, pos + len(V),
                            pieces + [(wi, b, V, mech)], gaps)
                    if r is not None:
                        return r
        return None

    return dfs(0, 0, [], [])


def _finalize(ctx, answer, split, words, pieces, gaps, indicator_types, is_link):
    if len(pieces) < 2:
        return None                              # a charade is >= 2 pieces
    # PHRASE-AWARE leftovers (was per-word glue, which stranded half of a two-word
    # indicator): each contiguous leftover run must segment into DB link words and/or
    # reversal-indicator words OR phrases; >= 1 indicator segment must remain overall.
    from core.engine_common import contiguous_groups, classify_glue_run
    segments = []
    for runidx in contiguous_groups(sorted(gaps)):
        segs = classify_glue_run(words, runidx, indicator_types, "reversal", is_link)
        if segs is None:
            return None
        segments.extend(segs)
    if not any(kind == "indicator" for kind, _ in segments):
        return None                              # the licensing reversal indicator must remain

    N = len(answer)
    # The forward fodder is pieces concatenated in clue order; the answer is its reverse, so
    # answer letter j comes from fodder position N-1-j. Map each answer position to the piece
    # (source) that supplied it, for faithful per-letter render provenance.
    fodder_owner = []                            # fodder position -> source index
    sources = []
    for si, (a, b, V, mech) in enumerate(pieces):
        toks = words[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=V, mechanism=mech, source="db"))
        fodder_owner.extend([si] * len(V))
    links = []
    for j in range(N):
        si = fodder_owner[N - 1 - j]
        links.append(Link(answer_pos=j + 1, source_index=si,
                          operation="reversal_charade", clue_atom_id=None))

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition", source=split.source)
    annotations = []
    # ONE annotation per segment: an indicator segment carries the JOINED phrase, so
    # role_validity validates the DB row ("picked up"), never a bare component word.
    for kind, grp in segments:
        role, note = (("indicator", "reversal indicator") if kind == "indicator"
                      else ("link", "link word"))
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in grp for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in grp), role=role, note=note))
    from core.definition_engine import dbe_annotation
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="reversal_charade",
                  solved_by="catalog")
    parse.template_id = None
    parse.matched_signature = None
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully covered by the pieces")
    missing = parse.unexplained_words(ctx)
    if missing:
        warnings.append("these clue words are unaccounted for: "
                        + ", ".join(repr(m) for m in missing))
    if parse.definition is None:
        warnings.append("no definition found")
    elif getattr(parse.definition, "source", "db") == "pending":
        warnings.append("the definition is provisional (queued for enrichment)")
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
