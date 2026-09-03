"""Reversal+container engine — an outer DB value wrapping an inner DB value that is
REVERSED before insertion: ARABS = AS ("like") circling reverse(BAR) ("watering hole",
"westerly") = A·RAB·S.

A NEW stage. The plain container engine inserts values as-is; the reversal engines
concatenate (whole-answer reversal / reversal-charade). Neither covers a reversed value
inserted INSIDE another, so this clue shape had no engine (the 'reversal_container'
catalog rows were orphaned — nothing consumed them).

EVIDENCE-DRIVEN and ANSWER-DRIVEN, links classified LAST (memory:
feedback-never-preassign-links). Gated on BOTH a container/insertion indicator AND a
reversal indicator, and the assembled letters must equal the answer EXACTLY, so it cannot
fabricate. Definition decided upstream (def_pos). Pure and DB-decoupled.
"""

from core import grammar
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation", "raw")
MAX_PIECE_WORDS = 4


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _value_candidates(words, a, b, lookup_all):
    """DB (value, mechanism) pairs for the run words[a:b] — UNFILTERED (a container piece
    is split around the other, so not a contiguous substring of the answer)."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out, seen = [], set()
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in seen:
            seen.add(v)
            out.append((v, mech))
    return out


def _verify_rev_insertion(vals_a, vals_b, answer):
    """Reconstruct the answer as an insertion where the INNER value is REVERSED before it
    is inserted into the outer. Returns (a_is_outer, p, Li, outer_val, inner_val_written)
    or None. inner occupies answer[p:p+Li]; outer = answer[:p]+answer[p+Li:]."""
    N = len(answer)
    for a_is_outer, outer_vals, inner_vals in ((True, vals_a, vals_b),
                                               (False, vals_b, vals_a)):
        for OUT in outer_vals:
            Lo = len(OUT)
            for IN in inner_vals:
                IN_r = IN[::-1]
                if IN_r == IN:                     # reversal must be non-trivial
                    continue
                Li = len(IN_r)
                if Lo < 1 or Li < 1 or Lo + Li != N:
                    continue
                # TRUE CONTAINER: the split must fall INSIDE the outer, so the outer
                # straddles the reversed inner on both sides. p==0 or p==Lo puts the
                # inner at an end — a reversal charade, with the container indicator
                # badged onto a parse where it does no work.
                for p in range(1, Lo):             # split point within the outer
                    inner = answer[p:p + Li]
                    outer = answer[:p] + answer[p + Li:]
                    if inner == IN_r and outer == OUT:
                        return (a_is_outer, p, Li, OUT, IN)
    return None


def _assemble(ctx, answer, words, lookup_all, is_link, indicator_types):
    n = len(words)

    def types(k):
        try:
            return indicator_types(words[k].text) or set()
        except Exception:
            return set()

    def is_con(k):
        return bool({"container", "insertion"} & types(k))

    def is_rev(k):
        return "reversal" in types(k)

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    from core.engine_common import has_typed_indicator
    if (not has_typed_indicator(words, indicator_types, ("container", "insertion"))
            or not has_typed_indicator(words, indicator_types, "reversal")):
        return None                                # gate (phrase-aware): both indicators required

    runs = [(a, b) for a in range(n)
            for b in range(a + 1, min(a + MAX_PIECE_WORDS, n) + 1)]
    best_nm = None                                 # best near-miss placement (fewest unaccounted)
    for (a1, b1) in runs:
        v1 = _value_candidates(words, a1, b1, lookup_all)
        if not v1:
            continue
        for (a2, b2) in runs:
            if not (b1 <= a2 or b2 <= a1):         # value runs must be disjoint
                continue
            v2 = _value_candidates(words, a2, b2, lookup_all)
            if not v2:
                continue
            ins = _verify_rev_insertion([v for v, _ in v1], [v for v, _ in v2], answer)
            if ins is None:
                continue
            a_is_outer, p, Li, outer_val, inner_val = ins
            if a_is_outer:
                outer_run, inner_run = (a1, b1), (a2, b2)
                outer_mech = next(m for v, m in v1 if v == outer_val)
                inner_mech = next(m for v, m in v2 if v == inner_val)
            else:
                outer_run, inner_run = (a2, b2), (a1, b1)
                outer_mech = next(m for v, m in v2 if v == outer_val)
                inner_mech = next(m for v, m in v1 if v == inner_val)
            used = set(range(*outer_run)) | set(range(*inner_run))
            remaining = [k for k in range(n) if k not in used]
            # PHRASE-AWARE indicator identification (was per-word is_con/is_rev, which
            # stranded half of a two-word indicator as an unaccounted word). Container
            # first; the reversal cover is computed on what remains AFTER the container
            # words, so an overlapping longer phrase row (ARABS: 'circling westerly' is
            # a reversal ROW while 'circling' is the container) cannot steal a word the
            # container role needs. Fallback to the full pool for a single word typed
            # both ways (old behaviour: it serves as the container, rev set stays empty).
            from core.engine_common import disjoint_typed_cover
            con_runs = disjoint_typed_cover(words, remaining, indicator_types,
                                            ("container", "insertion"))
            if not con_runs:
                continue
            con_ind = sorted({k for r in con_runs for k in r})
            rev_pool = [k for k in remaining if k not in con_ind]
            rev_runs = disjoint_typed_cover(words, rev_pool, indicator_types, "reversal")
            if not rev_runs:
                rev_runs = disjoint_typed_cover(words, remaining, indicator_types,
                                                "reversal")
            if not rev_runs:
                continue
            rev_ind = sorted({k for r in rev_runs for k in r})
            ind_set = set(con_ind) | set(rev_ind)
            # ONLY genuine links are annotated; any non-link residue is left UNACCOUNTED so
            # _verify NAMES it and marks a FAIL (near-miss). A clean placement (no unaccounted)
            # is used immediately; otherwise keep the best near-miss (fewest unaccounted) to
            # surface when none is clean. Pass-invariant (an unaccounted word fails _verify).
            links = [k for k in remaining if k not in ind_set and residue_link(k)]
            unacct = [k for k in remaining if k not in ind_set and not residue_link(k)]
            pl = {"outer_run": outer_run, "inner_run": inner_run,
                  "outer_val": outer_val, "inner_val": inner_val,
                  "outer_mech": outer_mech, "inner_mech": inner_mech,
                  "p": p, "Li": Li, "con": con_runs,
                  "rev": [r for r in rev_runs if not (set(r) & set(con_ind))],
                  "links": sorted(links)}
            if not unacct:
                return pl
            if best_nm is None or len(unacct) < best_nm[0]:
                best_nm = (len(unacct), pl)
    return best_nm[1] if best_nm else None


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    o0, o1 = pl["outer_run"]
    i0, i1 = pl["inner_run"]
    outer_toks, inner_toks = words[o0:o1], words[i0:i1]
    outer_src = Source(
        clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in outer_toks), value=pl["outer_val"],
        mechanism="synonym" if pl["outer_mech"] == "synonym" else "abbreviation")
    inner_src = Source(
        clue_atom_ids=tuple(aid for t in inner_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in inner_toks), value=pl["inner_val"],
        mechanism="synonym" if pl["inner_mech"] == "synonym" else "abbreviation")
    if o0 < i0:
        sources, OUT, IN = [outer_src, inner_src], 0, 1
    else:
        sources, OUT, IN = [inner_src, outer_src], 1, 0

    p, Li = pl["p"], pl["Li"]
    links = []
    for pos in range(1, len(answer) + 1):
        if p < pos <= p + Li:                      # inner span — reversed into the answer
            links.append(Link(answer_pos=pos, source_index=IN, operation="container",
                              clue_atom_id=None, transform="reversed"))
        else:
            links.append(Link(answer_pos=pos, source_index=OUT, operation="container",
                              clue_atom_id=None))

    annotations = []
    # pl["con"]/pl["rev"] are RUNS (each DB-typed as a whole phrase) — one annotation
    # per run, so role_validity validates the phrase row, never a component word.
    for run in pl["con"]:
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in run for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in run), role="indicator",
            note="container indicator"))
    for run in pl["rev"]:
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in run for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in run), role="indicator",
            note="reversal indicator"))
    for k in pl["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link", note="link word"))
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="reversal_container",
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


def solve_reversal_container(ctx, defines, lookup_all, is_link, indicator_types,
                             define_fallback=None, is_dbe=None):
    """Full reversal+container solve — evidence-driven. First clean PASS, else best parse,
    else None (abstain — not this shape, or an indicator/value is missing)."""
    from core.definition_engine import find_definitions

    answer = _answer(ctx)
    if len(answer) < 4:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    from core.engine_common import better_near_miss
    best = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 3:                          # outer + inner + an indicator, minimum
            continue
        pl = _assemble(ctx, answer, words, lookup_all, is_link, indicator_types)
        if pl is None:
            continue
        parse = _build(ctx, split, words, answer, pl)
        if parse.status == "pass":
            return parse
        best = better_near_miss(best, parse, ctx)
    return best
