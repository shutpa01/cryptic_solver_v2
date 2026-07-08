"""Charade with ONE container piece whose INNER is a letter-SELECTION — the insertion mirror
of the selection-DELETION engines (Times 29582 COPSHOP / TENDERHEARTED).

  TENDERHEARTED = TENDER + [HEATED around R]
    "offer" = TENDER (value tile)   "warm" = HEATED (outer)
    "last of" "our" -> R (last-letter selection, the inner)   "blankets" = insertion indicator
    "Easily touched" = definition
  COPSHOP = [COSH around P] + OP
    "truncheon" = COSH (outer)   "policeman's" "first" -> P (first-letter inner)
    "introduced to" = insertion indicator   "work" = OP (value tile)

We can already take a selected letter and DELETE it (anagram_selection_deletion,
container_deletion_selection). This is the SYMMETRIC source: take the selected letter and
INSERT it into a plain synonym, COMPOSED inside a charade. It is the exact sibling of
container_deletion_selection (which inserts a selection into a DELETED outer) and of
container_charade (one container piece, but a DB-value inner) — neither can reach a plain
outer wrapping a single-word selection inside a charade.

A NEW bespoke stage (never edits a working engine). It reuses core.selection.select_span
(answer-driven, per-letter sourced) and container_charade's OUTER-around-INNER charade tiling.
Built TIGHT because a single-letter insert is the riskiest thing we do:
  * requires a container/insertion indicator AND a letter-selection indicator (both DB-typed);
  * the selection indicator must be ADJACENT to the selected word, and the rule must be one the
    DB licenses for that indicator (named rule only — no widening);
  * INNER = the EXACT letters the rule takes from that single word, landing at its interior
    answer span; OUTER = an EXACT DB value of a disjoint run split around it (true container);
  * the rest of the answer is tiled by ordinary DB-value charade pieces; EVERY remaining word a
    DB link; answer-driven exact reconstruction; PASS-only; _verify calls role_validity.
Per-letter provenance on the inner. Pure and DB-decoupled.
"""

from core.selection import select_span, select_span_run
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation", "raw")
MAX_RUN = 4
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
    the DB types as one of `wptypes`. Returns the run indices (tuple) or None."""
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
            if ty & wptypes and (best is None or L > len(best)):
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


def solve_charade_container_selection(ctx, defines, lookup_all, is_link, indicator_types,
                                      selection_rules, define_fallback=None, is_dbe=None):
    """Charade with one container piece wrapping a single-word selection inner. PASS-only."""
    from core.definition_engine import find_definitions
    answer = _answer(ctx)
    if len(answer) < 4 or selection_rules is None:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    from core.engine_common import better_near_miss
    best_fail = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 4:        # outer + selected word + container ind + selection ind, min
            continue
        parse = _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types,
                           selection_rules)
        if parse is None:
            continue
        if parse.status == "pass":
            return parse
        best_fail = better_near_miss(best_fail, parse, ctx)
    return best_fail


def _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types,
               selection_rules):
    n, N = len(words), len(answer)
    # claim the container indicator FIRST, then the selection indicator from what is left.
    con_run = _typed_run(words, n, {"container", "insertion"}, indicator_types, set())
    if con_run is None:
        return None
    sel_run, sel_rules = _sel_run(words, n, selection_rules, set(con_run))
    if sel_run is None:
        return None
    ind = set(con_run) | set(sel_run)

    val_cache = {}

    def values(a, b):
        if (a, b) not in val_cache:
            val_cache[(a, b)] = _run_values(words, a, b, lookup_all)
        return val_cache[(a, b)]

    def adjacent(fi):
        # the selected word sits next to the selection indicator, allowing only LINK words
        # to intervene ("last of our": indicator "last", link "of", selected "our").
        if fi < sel_run[0]:
            gap = range(fi + 1, sel_run[0])
        elif fi > sel_run[-1]:
            gap = range(sel_run[-1] + 1, fi)
        else:
            return False
        return all(is_link and is_link(words[k].text) for k in gap)

    # candidate selected fodder: a word OR contiguous run (1..3 words, was one word only)
    # whose nearest edge is ADJACENT to the selection indicator, none of it an indicator.
    sel_runs_f = []
    for fa in range(n):
        for fb in range(fa + 1, min(fa + 3, n) + 1):
            r = range(fa, fb)
            if any(k in ind for k in r):
                break
            edge = fb - 1 if fb - 1 < sel_run[0] else fa
            if adjacent(edge):
                sel_runs_f.append((fa, fb))
    if not sel_runs_f:
        return None

    near = [None]        # best near-miss (side channel; dfs keeps its pass/None return protocol)

    def dfs(pos, used, pieces, con_used):
        if pos == N:
            if not con_used:
                return None
            return _finalize(ctx, split, words, answer, pieces, used, ind, con_run, sel_run,
                             is_link, near)
        # 1. plain value piece
        for a in range(n):
            if a in used or a in ind:
                continue
            for b in range(a + 1, min(a + MAX_RUN, n) + 1):
                if any(k in used or k in ind for k in range(a, b)):
                    break
                for v, mech in values(a, b):
                    if answer.startswith(v, pos):
                        r = dfs(pos + len(v), used | set(range(a, b)),
                                pieces + [("value", (a, b), v, mech)], con_used)
                        if r:
                            return r
        # 2. container piece (once): OUTER (DB value) around INNER (single-word selection)
        if not con_used:
            for L in range(2, N - pos + 1):
                span = answer[pos:pos + L]
                for q in range(1, L):
                    for Li in range(1, L - q + 1):
                        inner = span[q:q + Li]
                        outer = span[:q] + span[q + Li:]
                        if not outer:
                            continue
                        for (fa, fb) in sel_runs_f:
                            frun = set(range(fa, fb))
                            if frun & used:
                                continue
                            sel = None
                            for rule in sel_rules:
                                hit = next(((s, aids) for s, aids
                                            in select_span_run(ctx, words[fa:fb], rule)
                                            if s.upper() == inner), None)
                                if hit is not None:
                                    sel = (rule, hit[1])
                                    break
                            if sel is None:
                                continue
                            fi = (fa, fb)
                            u1 = used | frun
                            for oa in range(n):
                                if oa in u1 or oa in ind:
                                    continue
                                for ob in range(oa + 1, min(oa + MAX_RUN, n) + 1):
                                    if any(k in u1 or k in ind for k in range(oa, ob)):
                                        break
                                    omech = next((m for v, m in values(oa, ob) if v == outer),
                                                 None)
                                    if omech is None:
                                        continue
                                    r = dfs(pos + L, u1 | set(range(oa, ob)),
                                            pieces + [("container", (oa, ob), fi, sel[0],
                                                       outer, inner, sel[1], L, q, Li, omech)],
                                            True)
                                    if r:
                                        return r
        return None

    result = dfs(0, set(), [], False)
    if result is not None and result.status == "pass":
        return result
    return near[0]                                     # surface the best near-miss fail


def _finalize(ctx, split, words, answer, pieces, used, ind, con_run, sel_run, is_link, near):
    from core.engine_common import better_near_miss
    n = len(words)
    # ONLY genuine links are annotated; any non-link residue stays UNACCOUNTED so _verify NAMES
    # it and marks a FAIL. Build EVERY complete tiling and record the best as a near-miss (side
    # channel `near`), but PRESERVE the DFS control flow: return the parse (stopping the search)
    # ONLY when the residue is all links, else None (keep searching) — so on the clean path the
    # build is byte-identical and pass behaviour is unchanged.
    residue = [k for k in range(n) if k not in used and k not in ind]
    links = [k for k in residue if is_link and is_link(words[k].text)]
    parse = _build(ctx, split, words, answer, pieces, con_run, sel_run, links)
    near[0] = better_near_miss(near[0], parse, ctx)
    return parse if len(links) == len(residue) else None


def _build(ctx, split, words, answer, pieces, con_run, sel_run, links):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition", source=split.source)
    sources, links_out, pos = [], [], 0
    sel_note = None
    for piece in pieces:
        if piece[0] == "value":
            _, (a, b), v, mech = piece
            toks = words[a:b]
            si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                text=" ".join(t.text for t in toks), value=v, mechanism=mech))
            for _ in v:
                pos += 1
                links_out.append(Link(answer_pos=pos, source_index=si,
                                      operation="charade", clue_atom_id=None))
        else:
            _, (oa, ob), fi, rule, outer, inner, sel_aids, L, q, Li, omech = piece
            fa, fb = fi                          # the selected fodder RUN (was one word)
            outer_toks = words[oa:ob]
            o_si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
                text=" ".join(t.text for t in outer_toks), value=outer, mechanism=omech))
            i_si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in words[fa:fb] for aid in t.atom_ids),
                text=" ".join(t.text for t in words[fa:fb]),
                value=inner, mechanism=_SEL_MECH.get(rule, rule)))
            for off in range(L):
                pos += 1
                if q <= off < q + Li:
                    aid = sel_aids[off - q]
                    links_out.append(Link(answer_pos=pos, source_index=i_si,
                                          operation="selection", clue_atom_id=aid))
                else:
                    links_out.append(Link(answer_pos=pos, source_index=o_si,
                                          operation="container", clue_atom_id=None))
            sel_note = ("%s-letter selection indicator (of %s)"
                        % (rule, " ".join(t.text for t in words[fa:fb])))

    con_toks = [words[k] for k in con_run]
    sel_toks = [words[k] for k in sel_run]
    annotations = [
        Annotation(clue_atom_ids=tuple(aid for t in con_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in con_toks), role="indicator",
                   note="container indicator"),
        Annotation(clue_atom_ids=tuple(aid for t in sel_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in sel_toks), role="indicator",
                   note=sel_note or "letter-selection indicator"),
    ]
    for k in links:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                                      role="link", note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links_out, annotations=annotations,
                  definition=definition, operation="charade_container_selection",
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
