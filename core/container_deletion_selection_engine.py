"""Container whose OUTER is a positional DELETION and whose INNER is a letter SELECTION —
the hardest container shape (TWO built pieces in one insertion), built deliberately TIGHT:

  RIVEN = RIEN around V
    "pal" = FRIEND; "discovered" = remove both ends (outer deletion) -> RIEN (the outer);
    "valuables" "primarily" = V (first letter, the inner); "smuggling" = container;
    "Split" = def.   RIEN around V = RI-V-EN = RIVEN.

Every other container engine builds AT MOST one piece (a charade/selection/deletion inner, or
a charade/reversed outer); none builds BOTH the outer and the inner. This one does, so it has
the largest search space and the highest false-pass risk — hence the gating is strict:

  * requires THREE distinct indicators present: a container/insertion indicator, a deletion
    indicator, AND a letter-selection indicator;
  * OUTER is answer-driven: a DB synonym/abbreviation V_o of a clue run, with the deletion op
    the deletion indicator's DB sub-type licenses (named sub-types only — NO generic widening
    when a named one exists), giving the EXACT outer string, len(V_o) > the outer;
  * INNER is answer-driven: the EXACT letters the selection indicator's rule takes from a
    single disjoint clue word;
  * true container (inner strictly interior); EVERY remaining word must be a DB link;
  * own _verify calls role_validity; returns ONLY a clean PASS.

A NEW bespoke stage (never edits a working engine). Per-letter provenance on the inner. Pure
and DB-decoupled.
"""

from core import deletion
from core.selection import select_span
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
MAX_RUN = 4
_OP_PHRASE = {"behead": "first letter", "curtail": "last letter",
              "outer": "both outer letters", "heartless": "central letter"}
_SEL_MECH = {"first": "first_letter", "last": "last_letter", "outer": "outer",
             "middle": "middle", "alternate": "alternate"}


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _run_values(words, a, b, lookup_all):
    phrase = " ".join(words[k].text for k in range(a, b))
    out, seen = [], set()
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in seen:
            seen.add(v)
            out.append((v, mech))
    return out


def _typed_run(words, n, wptypes, indicator_types, exclude):
    """Longest contiguous run (1..MAX_RUN words) not overlapping `exclude` whose joined text
    the DB types as one of `wptypes`."""
    best = None
    for L in range(min(MAX_RUN, n), 0, -1):
        for i in range(n - L + 1):
            idxs = tuple(range(i, i + L))
            if any(k in exclude for k in idxs):
                continue
            phrase = " ".join(words[k].text for k in idxs)
            try:
                ty = set(indicator_types(phrase) or ())
            except Exception:
                ty = set()
            if ty & wptypes:
                if best is None or L > len(best):
                    best = idxs
    return best


def _sel_run(words, n, selection_rules, exclude):
    """Longest run (1..MAX_RUN words) not overlapping `exclude` whose DB letter-selection
    rules are non-empty. Returns (run_indices, frozenset(rules)) or (None, None)."""
    best, best_rules = None, None
    for L in range(min(MAX_RUN, n), 0, -1):
        for i in range(n - L + 1):
            idxs = tuple(range(i, i + L))
            if any(k in exclude for k in idxs):
                continue
            phrase = " ".join(words[k].text for k in idxs)
            try:
                rules = set(selection_rules(phrase) or ())
            except Exception:
                rules = set()
            if rules and (best is None or L > len(best)):
                best, best_rules = idxs, rules
    return best, best_rules


def _del_ops(phrase, deletion_subtypes):
    """Positional deletion ops the indicator licenses. Named sub-types only when present (NO
    generic widening if a named one exists — keeps this tight); a purely generic removal
    widens to curtail/behead."""
    try:
        subs = set(deletion_subtypes(phrase) or ())
    except Exception:
        subs = set()
    ops, named = set(), False
    for s in subs:
        op = deletion.SUBTYPE_OP.get(s)
        if op:
            ops.add(op)
            named = True
    if not named:
        ops |= {"curtail", "behead"}
    return ops


def solve_container_deletion_selection(ctx, defines, lookup_all, is_link, indicator_types,
                                       deletion_subtypes, selection_rules,
                                       define_fallback=None, is_dbe=None):
    """Container with a deletion-built outer and a selection-built inner. Returns ONLY a clean
    PASS, else None (abstain)."""
    from core.definition_engine import find_definitions
    answer = _answer(ctx)
    if len(answer) < 4:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        n = len(words)
        if n < 5:           # outer + fodder + container + deletion + selection indicators, min
            continue
        parse = _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types,
                           deletion_subtypes, selection_rules)
        if parse is not None and parse.status == "pass":
            return parse
    return None


def _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types,
               deletion_subtypes, selection_rules):
    n, N = len(words), len(answer)
    # Claim the real container indicator FIRST, then deletion, then selection from what is
    # left — so a word that is incidentally typed 'insertion' (e.g. "primarily") cannot be
    # mistaken for the container when it is really the selection indicator.
    con_run = _typed_run(words, n, {"container", "insertion"}, indicator_types, set())
    if con_run is None:
        return None
    del_run = _typed_run(words, n, {"deletion"}, indicator_types, set(con_run))
    if del_run is None:
        return None
    sel_run, sel_rules = _sel_run(words, n, selection_rules, set(con_run) | set(del_run))
    if sel_run is None:
        return None
    ind = set(con_run) | set(del_run) | set(sel_run)
    del_ops = _del_ops(" ".join(words[k].text for k in del_run), deletion_subtypes)
    if not del_ops:
        return None

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)
            if not (set(range(a, b)) & ind)]
    val_cache = {}

    def values(run):
        if run not in val_cache:
            val_cache[run] = _run_values(words, run[0], run[1], lookup_all)
        return val_cache[run]

    # Enumerate the insertion: inner strictly interior, outer straddling on both sides.
    for p in range(1, N - 1):
        for L in range(1, N - p):
            if p + L >= N:
                continue
            inner = answer[p:p + L]
            outer = answer[:p] + answer[p + L:]
            if not outer:
                continue
            # OUTER = a positional deletion of a DB value of a clue run.
            for orun in runs:
                built_outer = None
                for V_o, omech in values(orun):
                    if len(V_o) <= len(outer):
                        continue
                    op = next((o for r, o in deletion.candidates(V_o, del_ops)
                               if r == outer), None)
                    if op is not None:
                        built_outer = (V_o, omech, op)
                        break
                if built_outer is None:
                    continue
                # INNER = the selection indicator's rule applied to a single disjoint word.
                for fi in range(n):
                    if fi in ind or fi in range(orun[0], orun[1]):
                        continue
                    for rule in sel_rules:
                        sel = next(((s, aids) for s, aids in select_span(ctx, words[fi], rule)
                                    if s.upper() == inner), None)
                        if sel is None:
                            continue
                        parse = _build(ctx, split, words, answer, orun, outer, built_outer,
                                       fi, rule, sel[1], inner, p, L, con_run, del_run,
                                       sel_run, ind, is_link)
                        if parse is not None and parse.status == "pass":
                            return parse
    return None


def _build(ctx, split, words, answer, orun, outer, built_outer, fi, rule, inner_atom_ids,
           inner, p, L, con_run, del_run, sel_run, ind, is_link):
    from collections import Counter
    from core.definition_engine import dbe_annotation
    V_o, omech, op = built_outer
    n = len(words)
    used = set(range(*orun)) | {fi} | ind
    links_idx, ok = [], True
    for k in range(n):
        if k in used:
            continue
        if is_link and is_link(words[k].text):
            links_idx.append(k)
        else:
            ok = False
            break
    if not ok:
        return None

    outer_toks = words[orun[0]:orun[1]]
    outer_src = Source(
        clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in outer_toks), value=V_o, mechanism=omech)
    inner_src = Source(clue_atom_ids=words[fi].atom_ids, text=words[fi].text, value=inner,
                       mechanism=_SEL_MECH.get(rule, rule))
    if orun[0] < fi:
        sources, OUT, IN = [outer_src, inner_src], 0, 1
    else:
        sources, OUT, IN = [inner_src, outer_src], 1, 0

    links = []
    for pos in range(1, len(answer) + 1):
        if p < pos <= p + L:
            aid = inner_atom_ids[pos - 1 - p]
            links.append(Link(answer_pos=pos, source_index=IN, operation="selection",
                              clue_atom_id=aid))
        else:
            links.append(Link(answer_pos=pos, source_index=OUT, operation="container",
                              clue_atom_id=None))

    where = _OP_PHRASE.get(op, op)
    removed = "".join((Counter(V_o) - Counter(outer)).elements())
    con_toks = [words[k] for k in con_run]
    del_toks = [words[k] for k in del_run]
    sel_toks = [words[k] for k in sel_run]
    annotations = [
        Annotation(clue_atom_ids=tuple(aid for t in con_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in con_toks), role="indicator",
                   note="container indicator"),
        Annotation(clue_atom_ids=tuple(aid for t in del_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in del_toks), role="indicator",
                   note="deletion: cut %s (%s) from %s -> %s" % (where, removed, V_o, outer)),
        Annotation(clue_atom_ids=tuple(aid for t in sel_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in sel_toks), role="indicator",
                   note="%s-letter selection indicator (of %s)" % (rule, words[fi].text)),
    ]
    for k in links_idx:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                                      role="link", note="link word"))
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition", source=split.source)
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="container_deletion_selection",
                  solved_by="catalog")
    parse.template_id = None
    parse.matched_signature = None
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """Three-state verdict: pass / pending (provisional def) / fail."""
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
    elif missing:
        parse.status = "fail"
    else:
        parse.status = "pending"
