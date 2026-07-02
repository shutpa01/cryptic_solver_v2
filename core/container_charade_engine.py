"""Container+charade engine — a charade where ONE piece is a container.

EVIDENCE-DRIVEN, links classified LAST. The answer is tiled left-to-right by pieces:
  - value pieces: a DB synonym/abbreviation of a contiguous clue word-run, sitting at
    the current answer position (the ordinary charade piece);
  - exactly ONE container piece: a span of the answer = OUTER with INNER inserted, where
    OUTER and INNER are BOTH DB values of two disjoint clue word-runs (the outer is split
    around the inner, so looked up UNFILTERED).
Whatever clue words are left must contain a CONTAINER indicator; the rest are links;
anything else leaves the parse unaccounted.

Example: LURCHER = [LURE around CH] + R — outer LURE (Attraction), inner CH (church),
container indicator "outside", then the charade tail R (Rex); definition "dog".

Gated: only runs when a container indicator is present (it is the more specific, more
expensive engine, tried after the plain container). Pieces are drawn from ANY unused clue
run (a container breaks strict clue-order), so the strong filter is exact reconstruction
plus every clue word being accounted. Definition decided upstream. Pure and DB-decoupled.
"""

from core import grammar
from core.wordplay import GLUE_POS
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation", "raw")
MAX_RUN = 4


def _run_values(words, a, b, lookup_all):
    phrase = " ".join(words[k].text for k in range(a, b))
    out = []
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in out:
            out.append(v)
    return out


def _has_container_indicator(words, indicator_types):
    for t in words:
        try:
            ty = indicator_types(t.text) or set()
        except Exception:
            ty = set()
        if "container" in ty or "insertion" in ty:
            return True
    return False


def _assemble(answer, words, postags, lookup_all, is_link, indicator_types):
    n, N = len(words), len(answer)

    def is_con(k):
        try:
            ty = indicator_types(words[k].text) or set()
        except Exception:
            ty = set()
        return "container" in ty or "insertion" in ty

    def residue_link(k):
        return (is_link and is_link(words[k].text))

    all_runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    val_cache = {}

    def values(run):
        if run not in val_cache:
            val_cache[run] = _run_values(words, run[0], run[1], lookup_all)
        return val_cache[run]

    def avail(used):
        return [r for r in all_runs if not (set(range(r[0], r[1])) & used)]

    def finalize(pieces, used):
        if not any(p[0] == "container" for p in pieces):
            return None
        residue = [k for k in range(n) if k not in used]
        con = [k for k in residue if is_con(k)]
        if not con:
            return None
        c = con[0]
        links = []
        for k in residue:
            if k == c:
                continue
            if residue_link(k):
                links.append(k)
            else:
                return None
        return {"pieces": pieces, "con": [c], "links": links}

    def dfs(pos, used, pieces, con_used):
        if pos == N:
            return finalize(pieces, used)
        # 1. value piece — any unused run whose DB value sits at pos
        for r in avail(used):
            for v in values(r):
                if answer.startswith(v, pos):
                    res = dfs(pos + len(v), used | set(range(r[0], r[1])),
                              pieces + [("value", r, v)], con_used)
                    if res:
                        return res
        # 2. container piece (once) — span at pos = OUTER around INNER
        if not con_used:
            for L in range(2, N - pos + 1):
                span = answer[pos:pos + L]
                for q in range(1, L):
                    for Li in range(1, L - q + 1):
                        inner = span[q:q + Li]
                        outer = span[:q] + span[q + Li:]
                        if not outer:
                            continue
                        for ir in avail(used):
                            if inner not in values(ir):
                                continue
                            u2 = used | set(range(ir[0], ir[1]))
                            for orun in avail(u2):
                                if outer not in values(orun):
                                    continue
                                res = dfs(
                                    pos + L, u2 | set(range(orun[0], orun[1])),
                                    pieces + [("container", orun, ir, pos, L, q, Li,
                                               outer, inner)], True)
                                if res:
                                    return res
        return None

    return dfs(0, set(), [], False)


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, links = [], []
    pos = 0
    for piece in pl["pieces"]:
        if piece[0] == "value":
            _, (a, b), v = piece
            toks = words[a:b]
            si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                text=" ".join(t.text for t in toks), value=v, mechanism="synonym"))
            for _ in v:
                pos += 1
                links.append(Link(answer_pos=pos, source_index=si,
                                  operation="container_charade", clue_atom_id=None))
        else:
            _, (oa, ob), (ia, ib), spos, L, q, Li, outer, inner = piece
            outer_toks, inner_toks = words[oa:ob], words[ia:ib]
            o_si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
                text=" ".join(t.text for t in outer_toks), value=outer,
                mechanism="synonym"))
            i_si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in inner_toks for aid in t.atom_ids),
                text=" ".join(t.text for t in inner_toks), value=inner,
                mechanism="synonym"))
            for off in range(L):
                pos += 1
                si = i_si if q <= off < q + Li else o_si
                links.append(Link(answer_pos=pos, source_index=si,
                                  operation="container_charade", clue_atom_id=None))

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
                  definition=definition, operation="container_charade",
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
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"


def solve_container_charade(ctx, defines, lookup_all, is_link, indicator_types,
                            templates=None, define_fallback=None, is_dbe=None):
    """Full container+charade solve — evidence-driven. Gated on a container indicator
    in the wordplay (the more specific/expensive engine). First clean PASS, else best,
    else None."""
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 4:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    best = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 3 or not _has_container_indicator(words, indicator_types):
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
