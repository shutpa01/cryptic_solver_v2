"""Container engine — CATALOG-DRIVEN (design §4), insertion-aware.

Signature-driven sibling of the other catalog engines, for a plain container (one DB
value inserted into another): BREAM = BEAM around R; TACTICS = TICS around ACT. Its
signatures were seeded from the working evidence solves (operation 'container'); each
names the value pieces and the container indicator in clue order:

  SYN_F / ABR_F  a value component (a DB synonym/abbreviation) — there are exactly TWO
  CON_I          the container / insertion indicator

Like anagram+container this is an INSERTION, so placement only FIXES which words play
which role; a separate verifier then reconstructs the answer by trying each value run as
the OUTER (the other inserted as the INNER) and keeping whichever rebuilds the answer
exactly. Gaps are classified as links LAST; a confirmed CON_I is required by placement.
Evidence preserved on a miss (design §2). Definition decided upstream. Pure and
DB-decoupled. This is the signature-driven replacement for the evidence container engine.
"""

from core import grammar, literals
from core.wordplay import GLUE_POS
from core.wfw_model import Source, Link, Annotation, Parse

# A value piece may be a DB synonym/abbreviation OR a curated literal (a short
# function word read as its own letters, e.g. "it" -> IT; core.literals). The literal
# is gated to the lexicon and still bound by exact reconstruction below, so it is a
# constrained candidate, not a wildcard.
_VALUE_MECH = ("synonym", "abbreviation", "raw")


def _piece_mechanism(toks, value):
    """Label a placed value piece: 'raw' when it is the curated literal of its
    (single-word) phrase, else 'synonym' (the engine does not further split
    synonym vs abbreviation in display)."""
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
    """DB value strings (synonym/abbreviation) for the run words[a:b] — UNFILTERED
    (the container outer is split around the inner, so not a substring of the answer)."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out, seen = [], set()
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _verify_insertion(run_a, run_b, vals_a, vals_b, answer):
    """Reconstruct the answer as an insertion of the two value runs, given each run's
    candidate value strings (a DB lookup for SYN_F/ABR_F, a letter selection for SEL_F).

    Returns (a_is_outer, p, Li, outer_val, inner_val) or None:
      inner occupies answer[p:p+Li]; outer = answer[:p]+answer[p+Li:].
      a_is_outer True  -> run_a is the outer, run_b the inner.
      a_is_outer False -> run_b is the outer, run_a the inner."""
    N = len(answer)
    for a_is_outer, outer_vals, inner_vals in ((True, vals_a, vals_b),
                                               (False, vals_b, vals_a)):
        for OUT in outer_vals:
            Lo = len(OUT)
            for IN in inner_vals:
                Li = len(IN)
                if Lo < 1 or Li < 1 or Lo + Li != N:
                    continue
                for p in range(0, Lo + 1):          # split point within the outer
                    inner = answer[p:p + Li]
                    outer = answer[:p] + answer[p + Li:]
                    if inner == IN and outer == OUT:
                        return (a_is_outer, p, Li, OUT, IN)
    return None


def _place(slots, words, answer, postags, lookup_all, is_link, indicator_types,
           ctx=None, sel=None):
    """Assign each slot a consecutive word-run in clue order (gaps allowed), validate
    the CON_I slot, classify gaps as links LAST, then verify the insertion.
    Returns {run_a, run_b, role_a, role_b, con_i, links, sel_indicator, sel_rule,
    insertion} or None.

    `sel`, when given, is (rule, indicator_indices) for a SEL_F value run: those words
    are accounted as the licensing selection indicator (not links), and a SEL_F run's
    candidate values come from core.selection.select_span(word, rule) — answer-driven,
    so only a selection that completes the insertion to the exact answer is kept."""
    from core.selection import select_span
    n = len(words)
    nslots = len(slots)
    sel_rule, sel_ind = (sel if sel else (None, ()))
    sel_ind = set(sel_ind)

    def residue_link(k):
        return (is_link and is_link(words[k].text))

    def run_values(run, role):
        if role == "SEL_F":
            if (run[1] - run[0]) != 1 or ctx is None or sel_rule is None:
                return []
            return [s for s, _ in select_span(ctx, words[run[0]], sel_rule)]
        return _value_candidates(words, run[0], run[1], lookup_all)

    def finalize(assigned, gap_idxs):
        val_runs = [(r, role) for r, role in assigned
                    if role in ("SYN_F", "ABR_F", "SEL_F")]
        if len(val_runs) != 2:
            return None
        con_i = [k for r, role in assigned if role == "CON_I" for k in range(*r)]
        links, indicator = [], []
        for k in gap_idxs:
            if k in sel_ind:
                indicator.append(k)
            elif residue_link(k):
                links.append(k)
            else:
                return None                          # a content word unaccounted
        if not sel_ind <= set(indicator):
            return None                              # licensed indicator must be accounted
        (run_a, role_a), (run_b, role_b) = val_runs
        ins = _verify_insertion(run_a, run_b, run_values(run_a, role_a),
                                run_values(run_b, role_b), answer)
        if ins is None:
            return None
        return {"run_a": run_a, "run_b": run_b, "role_a": role_a, "role_b": role_b,
                "con_i": sorted(con_i), "links": sorted(links),
                "sel_indicator": sorted(indicator), "sel_rule": sel_rule,
                "insertion": ins}

    def dfs(si, wi, assigned, gaps):
        if si == nslots:
            return finalize(assigned, gaps + list(range(wi, n)))
        slot = slots[si]
        nw = slot.n_words
        for j in range(wi, n - nw + 1):
            run = (j, j + nw)
            if any(k in sel_ind for k in range(*run)):
                continue                             # never build a piece from the indicator
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
    if not set(roles) <= {"SYN_F", "ABR_F", "CON_I", "SEL_F"}:
        return None
    if sum(roles.count(r) for r in ("SYN_F", "ABR_F", "SEL_F")) != 2:
        return None
    if roles.count("CON_I") < 1:
        return None
    # A SEL_F value run needs a licensing selection indicator; try each (rule + words).
    sel_options = [None]
    if "SEL_F" in roles:
        from core.selection_indicators import find_indicators
        inds = find_indicators(words)
        if not inds:
            return None
        sel_options = inds
    for sel in sel_options:
        placement = _place(template.slots, words, answer, postags, lookup_all, is_link,
                           indicator_types, ctx=ctx, sel=sel)
        if placement is not None:
            return _build(ctx, split, words, answer, placement, template)
    return None


def _build(ctx, split, words, answer, placement, template):
    """Assemble the Parse: one Source per value piece (outer + inner), per-letter links
    coloured by piece (inner span vs outer), indicator + link annotations."""
    from core.definition_engine import dbe_annotation
    a_is_outer, p, L, outer_val, inner_val = placement["insertion"]
    run_a = placement["run_a"]
    run_b = placement["run_b"]
    role_a = placement.get("role_a")
    role_b = placement.get("role_b")
    outer_run, inner_run, outer_role, inner_role = (
        (run_a, run_b, role_a, role_b) if a_is_outer
        else (run_b, run_a, role_b, role_a))
    outer_toks = words[outer_run[0]:outer_run[1]]
    inner_toks = words[inner_run[0]:inner_run[1]]

    def _value_source(toks, role, value):
        """A value piece. A SEL_F piece is the letters SELECTED from its single word;
        reference only those letters' atoms (so the render lights exactly the taken
        letters) and label it a selection."""
        if role == "SEL_F":
            from core.selection import select_span
            atom_ids = next((aid for s, aid in
                             select_span(ctx, toks[0], placement.get("sel_rule"))
                             if s == value), None)
            return Source(
                clue_atom_ids=atom_ids or tuple(aid for t in toks for aid in t.atom_ids),
                text=" ".join(t.text for t in toks), value=value,
                mechanism="selection")
        return Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=value,
            mechanism=_piece_mechanism(toks, value))

    outer_src = _value_source(outer_toks, outer_role, outer_val)
    inner_src = _value_source(inner_toks, inner_role, inner_val)
    # stable colour: sources in clue order
    if outer_run[0] < inner_run[0]:
        sources = [outer_src, inner_src]; OUT, IN = 0, 1
    else:
        sources = [inner_src, outer_src]; OUT, IN = 1, 0

    links = []
    for pos in range(1, len(answer) + 1):
        si = IN if p < pos <= p + L else OUT
        links.append(Link(answer_pos=pos, source_index=si,
                          operation="container", clue_atom_id=None))

    annotations = []
    for k in placement["con_i"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="container indicator"))
    sel_ind_idx = placement.get("sel_indicator") or []
    if sel_ind_idx:
        rule = placement.get("sel_rule")
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in sel_ind_idx for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in sel_ind_idx),
            role="indicator", note="selection indicator (%s)" % (rule or "selection")))
    for k in placement["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="container", solved_by="catalog")
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


def solve_container(ctx, defines, lookup_all, is_link, indicator_types,
                    templates, define_fallback=None, is_dbe=None):
    """Full container solve — catalog-driven. Walk the container signatures in priority
    order; for each, for each definition split at its edge, place the value roles + CON_I
    and verify the insertion. First clean PASS (fewest residue), else best non-pass, else
    preserved fail-evidence."""
    from core.definition_engine import find_definitions

    answer = _answer(ctx)
    if len(answer) < 3 or not templates:
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
    return _build_fail_evidence(ctx, answer, prepared[0][0])


def _build_fail_evidence(ctx, answer, split):
    """Preserve the evidence when no signature instantiated: keep the definition; no
    wordplay roles assigned by elimination (no clean insertion was found)."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    dbe = dbe_annotation(split)
    annotations = [dbe] if dbe is not None else []
    return Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                 sources=[], links=[], annotations=annotations,
                 definition=definition, operation="container",
                 solved_by="catalog", status="fail",
                 warnings=["no container signature matched this clue"])
