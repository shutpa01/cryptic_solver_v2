"""Container-of-charade engine — an OUTER DB value wrapped around an INNER that is
itself a CHARADE of two or more DB values.

The plain container engine (core/container_engine) inserts ONE DB value into one outer.
This sibling allows the INNER to be a multi-piece charade — OUTER around (P1 + P2 + ...):

  LIMESTONE = LONE around (IM + EST)
    "unaccompanied" = LONE (outer)   "This compiler's" = IM   "established" = EST
    "in" = insertion indicator       "rock" = definition

A NEW stage (per the project rule: never edit a working engine to add a case). It is the
mirror of container_charade (a charade with one single-value container piece); here the
whole answer is a single container whose INNER is the charade. EVIDENCE-DRIVEN, links
classified last, gated on a container/insertion indicator. Pure and DB-decoupled — same
injected predicates as the other engines.
"""

from core import grammar
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation", "raw")
MAX_RUN = 5
MIN_INNER_PIECES = 2          # 1 piece is already the plain container's job


def _run_values(words, a, b, lookup_all):
    """DB synonym/abbreviation values for words[a:b] — UNFILTERED (outer is split around
    the inner, and each inner piece is matched by exact prefix, so no substring filter)."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out = []
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in out:
            out.append(v)
    return out


def _segment_inner(words, ia, ib, target, values):
    """Partition the CONTIGUOUS block words[ia:ib] into consecutive DB-value runs whose
    values concatenate (in clue order) to `target`. Returns the list of (a, b, value)
    pieces, or None. The block must be consumed entirely — a charade leaves no gap."""
    if ia == ib:
        return [] if target == "" else None
    for end in range(ia + 1, min(ia + MAX_RUN, ib) + 1):
        for val in values(ia, end):
            if target.startswith(val):
                rest = _segment_inner(words, end, ib, target[len(val):], values)
                if rest is not None:
                    return [(ia, end, val)] + rest
    return None


def _assemble(answer, words, lookup_all, is_link, indicator_types):
    """Find outer run + a charade inner block + insertion. Returns a placement or None."""
    n, N = len(words), len(answer)

    def is_con(k):
        try:
            ty = indicator_types(words[k].text) or set()
        except Exception:
            ty = set()
        return "container" in ty or "insertion" in ty

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    val_cache = {}

    def values(a, b):
        if (a, b) not in val_cache:
            val_cache[(a, b)] = _run_values(words, a, b, lookup_all)
        return val_cache[(a, b)]

    # Enumerate the insertion: inner = answer[p:p+L], outer = answer[:p]+answer[p+L:].
    for p in range(0, N):
        for L in range(1, N - p + 1):
            if not (p > 0 and p + L < N):
                # a TRUE container: the outer must straddle the inner on BOTH sides.
                # If the inner sits at the very start or end it is a charade (juxta-
                # position), not an insertion — leave those to the charade engines.
                continue
            inner = answer[p:p + L]
            outer = answer[:p] + answer[p + L:]
            for (oa, ob) in runs:
                if outer not in values(oa, ob):
                    continue
                # Inner is a charade: a contiguous block disjoint from the outer run that
                # segments into >= MIN_INNER_PIECES DB values concatenating to `inner`.
                for ia in range(n):
                    for ib in range(ia + 1, n + 1):
                        if not (ib <= oa or ia >= ob):    # disjoint from outer
                            continue
                        pieces = _segment_inner(words, ia, ib, inner, values)
                        if pieces is None or len(pieces) < MIN_INNER_PIECES:
                            continue
                        used = set(range(oa, ob))
                        used |= {k for (a, b, _) in pieces for k in range(a, b)}
                        residue = [k for k in range(n) if k not in used]
                        # PHRASE-AWARE residue split (was per-word is_con + links).
                        from core.engine_common import indicator_plus_links
                        sp = indicator_plus_links(words, residue, indicator_types,
                                                  ("container", "insertion"), is_link)
                        if sp is not None:
                            return {"p": p, "L": L, "outer": (oa, ob),
                                    "pieces": pieces, "con": sp[0], "links": sp[1]}
    return None


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    p, L = pl["p"], pl["L"]
    oa, ob = pl["outer"]
    outer_toks = words[oa:ob]
    outer = answer[:p] + answer[p + L:]

    # source 0 = outer; sources 1.. = inner charade pieces in clue order.
    outer_src = Source(
        clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in outer_toks), value=outer, mechanism="synonym")
    sources = [outer_src]
    inner_spans = []          # (start_offset_within_inner, length, source_index)
    off = 0
    for (a, b, val) in pl["pieces"]:
        toks = words[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=val, mechanism="synonym"))
        inner_spans.append((off, len(val), len(sources) - 1))
        off += len(val)

    # links: each answer letter -> its source. Inner region is p<pos<=p+L.
    links = []
    for pos in range(1, len(answer) + 1):
        if p < pos <= p + L:
            local = (pos - 1) - p
            si = next(idx for (start, length, idx) in inner_spans
                      if start <= local < start + length)
        else:
            si = 0
        links.append(Link(answer_pos=pos, source_index=si,
                          operation="container", clue_atom_id=None))

    annotations = []
    from core.engine_common import indicator_annotations
    annotations.extend(indicator_annotations(words, pl["con"], "container indicator"))
    for k in pl["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="container", solved_by="catalog")
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


def solve_container_inner_charade(ctx, defines, lookup_all, is_link, indicator_types,
                                  define_fallback=None, is_dbe=None):
    """Outer DB value wrapped around a charade inner. First clean PASS, else best parse,
    else None — same contract as the other container engines."""
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
        if len(words) < 3:                 # outer + >=2 inner pieces + indicator
            continue
        pl = _assemble(answer, words, lookup_all, is_link, indicator_types)
        if pl is None:
            continue
        parse = _build(ctx, split, words, answer, pl)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best
