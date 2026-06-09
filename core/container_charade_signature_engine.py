"""Container+charade engine — CATALOG-DRIVEN (design §4), insertion-aware.

Signature-driven sibling of container_signature_engine, for a charade where ONE piece
is a container (LURCHER = [LURE around CH] + R; SPONSOR = [SONS around P] + OR). Its
signatures were mined unambiguously from the evidence engine's own placements
(core/_mine_container_charade.py reads pl['pieces']); each names, in clue order:

  CNT_F  a container-pair member — there are exactly TWO; together they are OUTER with
         INNER inserted (which is which is resolved by the verifier, not the signature)
  SYN_F / ABR_F  an ordinary charade value piece (a DB synonym/abbreviation), 0+ of them
  CON_I  the container / insertion indicator

Placement only FIXES which clue runs play which role (clue order, gaps -> links). A
reconstructor then tiles the ANSWER with the charade pieces and ONE container span in
any answer order (a container breaks clue order), keeping the arrangement that rebuilds
the answer exactly. More constrained than the evidence engine (the runs are fixed by the
signature), so it is additive/deterministic. Evidence preserved on a miss (design §2).
Definition decided upstream. Pure and DB-decoupled.
"""

from core import grammar, literals
from core.wordplay import GLUE_POS
from core.wfw_model import Source, Link, Annotation, Parse

# A value piece may be a DB synonym/abbreviation OR a curated literal (a short function
# word read as its own letters; core.literals). Gated to the lexicon and bound by exact
# reconstruction, so a constrained candidate, not a wildcard.
_VALUE_MECH = ("synonym", "abbreviation", "raw")


def _piece_mechanism(toks, value):
    """Label a placed value piece: 'raw' when it is the curated literal of its
    (single-word) phrase, else 'synonym'."""
    phrase = " ".join(t.text for t in toks)
    if literals.literal_value(phrase) == value:
        return "raw"
    return "synonym"


def _is_con_indicator(text, indicator_types):
    try:
        ty = indicator_types(text) or set()
    except Exception:
        ty = set()
    return "container" in ty or "insertion" in ty


def _value_candidates(words, a, b, lookup_all):
    """DB value strings (synonym/abbreviation) for the run words[a:b] — UNFILTERED."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out, seen = [], set()
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _container_spans(words, run_a, run_b, lookup_all, N, answer):
    """All (span, a_is_outer, p, Li, outer_val, inner_val) for the container pair —
    OUTER with INNER inserted at split p (span = OUT[:p] + IN + OUT[p:]).

    ANSWER-DRIVEN, like the evidence engine (container_charade_engine._assemble): we
    enumerate the answer's own substrings as candidate spans and derive OUT/IN from each
    split, keeping only those whose IN is a DB value of one run and OUT a DB value of the
    other. This is bounded by answer length (O(N^4), N small), never the synonym product
    OUT x IN x positions — which ballooned to ~GB on clues with common fodder words. The
    accepted set is identical: the reconstructor only ever uses a span that is an answer
    substring (it filters with answer.startswith), so generating non-answer spans was pure
    waste plus a memory spike."""
    vals_a = set(_value_candidates(words, run_a[0], run_a[1], lookup_all))
    vals_b = set(_value_candidates(words, run_b[0], run_b[1], lookup_all))
    out, seen = [], set()
    for s in range(N):
        for L in range(2, N - s + 1):
            span = answer[s:s + L]
            for p in range(0, L):                  # outer chars before the inner
                for Li in range(1, L - p + 1):     # inner length
                    IN = span[p:p + Li]
                    OUT = span[:p] + span[p + Li:]
                    if not OUT or not IN:
                        continue
                    for a_is_outer, oset, iset in ((True, vals_a, vals_b),
                                                   (False, vals_b, vals_a)):
                        if OUT in oset and IN in iset:
                            key = (span, a_is_outer, p, Li, OUT, IN)
                            if key not in seen:
                                seen.add(key)
                                out.append(key)
    return out


def _reconstruct(answer, val_runs, cnt_runs, words, lookup_all):
    """Tile the answer with the charade value pieces + one container span, in any order.
    Returns the arrangement (answer-order list of placed pieces) or None."""
    N = len(answer)
    nval = len(val_runs)
    val_cands = [_value_candidates(words, a, b, lookup_all) for (a, b) in val_runs]
    spans = _container_spans(words, cnt_runs[0], cnt_runs[1], lookup_all, N, answer)

    def dfs(pos, used_vals, used_cnt, placed):
        if pos == N:
            return placed if (used_cnt and len(used_vals) == nval) else None
        for i in range(nval):
            if i in used_vals:
                continue
            for v in val_cands[i]:
                if v and answer.startswith(v, pos):
                    r = dfs(pos + len(v), used_vals | {i}, used_cnt,
                            placed + [("value", val_runs[i], v, pos)])
                    if r:
                        return r
        if not used_cnt:
            for (span, a_is_outer, p, Li, OUT, IN) in spans:
                if answer.startswith(span, pos):
                    r = dfs(pos + len(span), used_vals, True,
                            placed + [("container", cnt_runs, a_is_outer, p, Li,
                                       OUT, IN, pos, len(span))])
                    if r:
                        return r
        return None

    return dfs(0, frozenset(), False, [])


def _place(slots, words, answer, postags, lookup_all, is_link, indicator_types):
    """Assign each slot a consecutive clue run in clue order (gaps allowed), validate
    CON_I, classify gaps as links LAST, then reconstruct. Returns a placement or None."""
    n = len(words)
    nslots = len(slots)

    def residue_link(k):
        return (is_link and is_link(words[k].text)) or (postags[k] in GLUE_POS)

    def finalize(assigned, gap_idxs):
        cnt_runs = [r for r, role in assigned if role == "CNT_F"]
        val_runs = [r for r, role in assigned if role in ("SYN_F", "ABR_F")]
        if len(cnt_runs) != 2:
            return None
        con_i = [k for r, role in assigned if role == "CON_I" for k in range(*r)]
        links = []
        for k in gap_idxs:
            if residue_link(k):
                links.append(k)
            else:
                return None
        arrangement = _reconstruct(answer, val_runs, cnt_runs, words, lookup_all)
        if arrangement is None:
            return None
        return {"val_runs": val_runs, "cnt_runs": cnt_runs, "con_i": sorted(con_i),
                "links": sorted(links), "arrangement": arrangement}

    def dfs(si, wi, assigned, gaps):
        if si == nslots:
            return finalize(assigned, gaps + list(range(wi, n)))
        slot = slots[si]
        nw = slot.n_words
        for j in range(wi, n - nw + 1):
            run = (j, j + nw)
            if slot.role == "CON_I" and not any(
                    _is_con_indicator(words[k].text, indicator_types)
                    for k in range(*run)):
                continue
            r = dfs(si + 1, j + nw, assigned + [(run, slot.role)],
                    gaps + list(range(wi, j)))
            if r:
                return r
        return None

    return dfs(0, 0, [], [])


def _try_template(ctx, answer, template, split, words, postags, lookup_all, is_link,
                  indicator_types):
    if split.where != template.def_pos:
        return None
    if template.fodder_word_count > len(words):
        return None
    roles = [s.role for s in template.slots]
    if not set(roles) <= {"SYN_F", "ABR_F", "CNT_F", "CON_I"}:
        return None
    if roles.count("CNT_F") != 2:
        return None
    if roles.count("CON_I") < 1:
        return None
    placement = _place(template.slots, words, answer, postags, lookup_all, is_link,
                       indicator_types)
    if placement is None:
        return None
    return _build(ctx, split, words, answer, placement, template)


def _build(ctx, split, words, answer, placement, template):
    """Assemble the Parse from the answer-order arrangement: one Source per charade
    piece, two for the container (outer + inner), per-letter links coloured by piece."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, links = [], []
    for elem in placement["arrangement"]:
        if elem[0] == "value":
            _, (a, b), v, spos = elem
            toks = words[a:b]
            si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                text=" ".join(t.text for t in toks), value=v,
                mechanism=_piece_mechanism(toks, v)))
            for off in range(len(v)):
                links.append(Link(answer_pos=spos + off + 1, source_index=si,
                                  operation="container_charade", clue_atom_id=None))
        else:
            _, (run_a, run_b), a_is_outer, p, Li, OUT, IN, spos, L = elem
            outer_run, inner_run = (run_a, run_b) if a_is_outer else (run_b, run_a)
            outer_toks = words[outer_run[0]:outer_run[1]]
            inner_toks = words[inner_run[0]:inner_run[1]]
            o_si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
                text=" ".join(t.text for t in outer_toks), value=OUT,
                mechanism=_piece_mechanism(outer_toks, OUT)))
            i_si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in inner_toks for aid in t.atom_ids),
                text=" ".join(t.text for t in inner_toks), value=IN,
                mechanism=_piece_mechanism(inner_toks, IN)))
            for off in range(L):
                in_inner = p <= off < p + Li
                links.append(Link(answer_pos=spos + off + 1,
                                  source_index=(i_si if in_inner else o_si),
                                  operation="container_charade", clue_atom_id=None))

    annotations = []
    for k in placement["con_i"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="container indicator"))
    for k in placement["links"]:
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
    parse.template_id = template.id
    parse.matched_signature = template.signature
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


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def solve_container_charade(ctx, defines, lookup_all, is_link, indicator_types,
                            templates, define_fallback=None, is_dbe=None):
    """Full container+charade solve — catalog-driven. Walk the container_charade
    signatures in priority order; for each, for each definition split at its edge,
    place the roles and reconstruct. First clean PASS (fewest residue), else best
    non-pass, else preserved fail-evidence."""
    from core.definition_engine import find_definitions

    answer = _answer(ctx)
    if len(answer) < 4 or not templates:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    prepared = []
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 3:
            continue
        postags = grammar.wordplay_pos_tags(ctx, words)
        prepared.append((split, words, postags))
    if not prepared:
        return None

    best_pass, best_key, best_other = None, None, None
    for template in templates:
        for split, words, postags in prepared:
            parse = _try_template(ctx, answer, template, split, words, postags,
                                  lookup_all, is_link, indicator_types)
            if parse is None:
                continue
            if parse.status == "pass":
                residue = sum(1 for a in parse.annotations if a.role == "link")
                if best_key is None or residue < best_key:
                    best_pass, best_key = parse, residue
                    if residue == 0:
                        return parse
            elif best_other is None:
                best_other = parse
    if best_pass is not None:
        return best_pass
    if best_other is not None:
        return best_other
    return _build_fail_evidence(ctx, split=prepared[0][0])


def _build_fail_evidence(ctx, split):
    """Preserve the evidence when no signature instantiated: keep the definition; no
    wordplay roles assigned by elimination."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    dbe = dbe_annotation(split)
    annotations = [dbe] if dbe is not None else []
    return Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                 sources=[], links=[], annotations=annotations,
                 definition=definition, operation="container_charade",
                 solved_by="catalog", status="fail",
                 warnings=["no container+charade signature matched this clue"])
