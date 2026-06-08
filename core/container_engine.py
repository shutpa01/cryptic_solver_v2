"""Container engine — a plain container (one DB value inserted into another).

EVIDENCE-DRIVEN, links classified LAST (memory: feedback-never-preassign-links). The
answer is an insertion: OUTER with INNER inserted, where BOTH outer and inner are DB
values (synonym/abbreviation). e.g. LATEN = L(ATE)N? ; SHEPHERD = SHED around HERP? —
classic shape "A holding B" / "B in A".

It enumerates the insertion (where the inner sits in the answer), then looks for two
disjoint wordplay word-runs that PRODUCE the inner and the outer by DB lookup. Whatever
words are left must contain a CONTAINER indicator (container/insertion); the rest are
links (is_link or POS function/VERB/ADV); anything else leaves the parse unaccounted.

Gated: requires a container indicator. Definition decided upstream (def_pos). Per-piece
colour (the outer's two segments share its colour). Pure and DB-decoupled. This is the
sibling of anagram_container with BOTH components DB values instead of one anagram; it is
the foundation the container_charade engine will build on (a charade tail around this).
"""

from core import grammar
from core.wordplay import GLUE_POS
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
MAX_RUN = 5


def _run_values(words, a, b, lookup_all):
    """DB synonym/abbreviation values for the phrase words[a:b] — UNFILTERED (the
    container outer is split around the inner, so it is not a substring of the answer)."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out = []
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in out:
            out.append(v)
    return out


def _assemble(answer, words, postags, lookup_all, is_link, indicator_types):
    """Find outer/inner runs + insertion (both DB values). Returns a placement or None."""
    n, N = len(words), len(answer)

    def is_con(k):
        try:
            ty = indicator_types(words[k].text) or set()
        except Exception:
            ty = set()
        return "container" in ty or "insertion" in ty

    def residue_link(k):
        return (is_link and is_link(words[k].text)) or (postags[k] in GLUE_POS)

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    val_cache = {}

    def values(a, b):
        if (a, b) not in val_cache:
            val_cache[(a, b)] = _run_values(words, a, b, lookup_all)
        return val_cache[(a, b)]

    # Enumerate the insertion: inner = answer[p:p+L], outer = answer[:p]+answer[p+L:].
    for p in range(0, N):
        for L in range(1, N - p + 1):
            if p == 0 and p + L == N:
                continue                              # outer must be non-empty
            inner = answer[p:p + L]
            outer = answer[:p] + answer[p + L:]
            if not outer:
                continue
            for (ia, ib) in runs:
                if inner not in values(ia, ib):
                    continue
                for (oa, ob) in runs:
                    if not (ob <= ia or oa >= ib):    # runs must be disjoint
                        continue
                    if outer not in values(oa, ob):
                        continue
                    used = set(range(ia, ib)) | set(range(oa, ob))
                    residue = [k for k in range(n) if k not in used]
                    con_caps = [k for k in residue if is_con(k)]
                    for c in con_caps:
                        links, ok = [], True
                        for k in residue:
                            if k == c:
                                continue
                            if residue_link(k):
                                links.append(k)
                            else:
                                ok = False
                                break
                        if ok:
                            return {"p": p, "L": L, "inner": (ia, ib),
                                    "outer": (oa, ob), "con": [c], "links": links}
    return None


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    p, L = pl["p"], pl["L"]
    ia, ib = pl["inner"]
    oa, ob = pl["outer"]
    inner_toks = words[ia:ib]
    outer_toks = words[oa:ob]
    inner = answer[p:p + L]
    outer = answer[:p] + answer[p + L:]

    outer_src = Source(
        clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in outer_toks), value=outer, mechanism="synonym")
    inner_src = Source(
        clue_atom_ids=tuple(aid for t in inner_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in inner_toks), value=inner, mechanism="synonym")
    if oa < ia:
        sources = [outer_src, inner_src]; OUT, IN = 0, 1
    else:
        sources = [inner_src, outer_src]; OUT, IN = 1, 0

    links = []
    for pos in range(1, len(answer) + 1):
        si = IN if p < pos <= p + L else OUT
        links.append(Link(answer_pos=pos, source_index=si,
                          operation="container", clue_atom_id=None))

    annotations = []
    for k in pl["con"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="container indicator"))
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
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"


def solve_container(ctx, defines, lookup_all, is_link, indicator_types,
                    templates=None, define_fallback=None, is_dbe=None):
    """Full container solve — evidence-driven. First clean PASS, else best parse, else
    None. (`templates` accepted for call-site compatibility, unused for now.)"""
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
        if len(words) < 3:
            continue
        postags = grammar.wordplay_pos_tags(ctx, words)
        pl = _assemble(answer, words, postags, lookup_all, is_link, indicator_types)
        if pl is None:
            continue
        parse = _build(ctx, split, words, answer, pl)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best
