"""Container-of-acrostic engine — an OUTER DB value wrapped around an INNER formed by
ACROSTIC letter-selection (the first or last letters of a contiguous word-run).

The container engines insert a DB *value*; the acrostic engine spells the WHOLE answer
from initials. Neither covers an inner that is an acrostic inserted into an outer:

  MESCAL = MEAL around SC
    "dinner" = MEAL (outer)              "during" = insertion indicator
    "initially Served Cold" -> S, C      "Spirit" = definition

A NEW stage (per the project rule: never edit a working engine to add a case). It is the
sibling of container_inner_charade — same true-container wrapping, but the inner is an
acrostic selection rather than a DB charade. ANSWER-DRIVEN (the selected initials must
equal the exact inner span) and gated on BOTH a container indicator and an acrostic
indicator. Per-letter provenance on the inner (§5.5). Pure and DB-decoupled.
"""

from core import selection, engine_common
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
_MODE_MECHANISM = {"first": "first_letter", "last": "last_letter"}
MAX_RUN = 5


def _run_values(words, a, b, lookup_all):
    """DB synonym/abbreviation values for words[a:b] — UNFILTERED (the outer is split
    around the inner, so it is not a contiguous substring of the answer)."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out = []
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in out:
            out.append(v)
    return out


def _assemble(ctx, answer, words, lookup_all, is_link, indicator_types):
    """Find outer run + an acrostic inner run + insertion, gated on both indicators."""
    n, N = len(words), len(answer)
    if indicator_types is None:
        return None
    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    val_cache = {}

    def values(a, b):
        if (a, b) not in val_cache:
            val_cache[(a, b)] = _run_values(words, a, b, lookup_all)
        return val_cache[(a, b)]

    for p in range(0, N):
        for L in range(1, N - p + 1):
            if not (p > 0 and p + L < N):
                continue                          # true container: outer straddles inner
            inner = answer[p:p + L]
            outer = answer[:p] + answer[p + L:]
            for (oa, ob) in runs:
                if outer not in values(oa, ob):
                    continue
                # inner = first/last letters of a contiguous L-word run, disjoint from outer
                for mode in ("first", "last"):
                    for ia in range(0, n - L + 1):
                        ib = ia + L
                        if not (ib <= oa or ia >= ob):
                            continue
                        run = words[ia:ib]
                        sel = selection.selected(ctx, run, mode)
                        if sel is None or "".join(c for c, _ in sel) != inner:
                            continue
                        used = set(range(oa, ob)) | set(range(ia, ib))
                        res = set(k for k in range(n) if k not in used)
                        acr = engine_common.find_typed_run(words, res, indicator_types,
                                                           "acrostic", 1)
                        if not acr:
                            continue
                        rem = res - set(acr)
                        con = (engine_common.find_typed_run(words, rem, indicator_types,
                                                            "container", 1)
                               or engine_common.find_typed_run(words, rem, indicator_types,
                                                               "insertion", 1))
                        if not con:
                            continue
                        rem2 = rem - set(con)
                        ok, links = True, []
                        for k in sorted(rem2):
                            if is_link and is_link(words[k].text):
                                links.append(k)
                            else:
                                ok = False
                                break
                        if ok:
                            return {"p": p, "L": L, "outer": (oa, ob),
                                    "inner_run": (ia, ib), "mode": mode, "sel": sel,
                                    "acr": acr, "con": con, "links": links}
    return None


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    p, L = pl["p"], pl["L"]
    oa, ob = pl["outer"]
    outer = answer[:p] + answer[p + L:]
    outer_toks = words[oa:ob]
    outer_src = Source(
        clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in outer_toks), value=outer,
        mechanism="synonym", source="db")

    ia, ib = pl["inner_run"]
    run = words[ia:ib]
    sel = pl["sel"]
    mech = _MODE_MECHANISM[pl["mode"]]
    # one Source per selected letter — same per-letter provenance as the acrostic engine
    inner_srcs = [Source(clue_atom_ids=tok.atom_ids, text=tok.text, value=char,
                         mechanism=mech, source="db")
                  for (char, _aid), tok in zip(sel, run)]
    sources = [outer_src] + inner_srcs            # outer = 0; inner letter j -> 1 + j

    links = []
    for pos in range(1, len(answer) + 1):
        if p < pos <= p + L:
            j = (pos - 1) - p
            _char, aid = sel[j]
            links.append(Link(answer_pos=pos, source_index=1 + j,
                              operation="acrostic", clue_atom_id=aid))
        else:
            links.append(Link(answer_pos=pos, source_index=0,
                              operation="container", clue_atom_id=None))

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    annotations = []
    acr_toks = [words[i] for i in pl["acr"]]
    annotations.append(Annotation(
        clue_atom_ids=tuple(aid for t in acr_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in acr_toks), role="indicator",
        note="acrostic indicator"))
    con_toks = [words[i] for i in pl["con"]]
    annotations.append(Annotation(
        clue_atom_ids=tuple(aid for t in con_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in con_toks), role="indicator",
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


def solve_container_acrostic(ctx, defines, lookup_all, is_link, indicator_types,
                             define_fallback=None, is_dbe=None):
    """Outer DB value wrapped around an acrostic inner. First clean PASS, else best
    parse, else None — same contract as the other container engines."""
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
        if len(words) < 4:        # outer + >=1 acrostic word + 2 indicators (min)
            continue
        pl = _assemble(ctx, answer, words, lookup_all, is_link, indicator_types)
        if pl is None:
            continue
        parse = _build(ctx, split, words, answer, pl)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best
