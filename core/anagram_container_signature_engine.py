"""Anagram+container engine — CATALOG-DRIVEN (design §4), insertion-aware.

Signature-driven sibling of the other catalog engines, for "a container where one
component is an anagram" (SANDWICHES = SANDS around anag(I CHEW); EXHORT = anag(HER TO)
around X). Its signatures were seeded from the working evidence solves (operation
'anagram_container'); each names four roles in clue order:

  ANA_F  the anagram component (the fodder run)
  SYN_F  the value component (a DB synonym/abbreviation; outer or inner)
  ANA_I  the anagram indicator
  CON_I  the container / insertion indicator

Unlike the concatenation engines, container is an INSERTION, so placement only FIXES
which words play which role; a separate verifier then reconstructs the answer by trying
the value as the OUTER (with the anagram inserted as the inner) and as the INNER (inside
the anagram outer), keeping whichever rebuilds the answer exactly. Gaps are classified as
links LAST; a confirmed ANA_I / CON_I is required by the placement. Evidence preserved on
a miss (design §2). Definition decided upstream. Pure and DB-decoupled.
"""

from core import grammar
from core.wordplay import GLUE_POS, fodder_letter_forms, is_anagram_indicator
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
ROLE_MECHANISM = {"ANA_F": "anagram_fodder", "SYN_F": "synonym",
                  "ABR_F": "abbreviation"}


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


def _verify_insertion(words, ana_run, val_run, answer, lookup_all, value_cands=None):
    """Reconstruct the answer as an insertion of ANA_F (anagram) and SYN_F (value).

    Returns (inner_anag, p, L, value, anag_value) or None:
      inner occupies answer[p:p+L]; outer = answer[:p]+answer[p+L:].
      inner_anag True  -> inner is the anagram (of the fodder), outer is the value.
      inner_anag False -> inner is the value, outer is the anagram (of the fodder).
      `value` = the value-component letters; `anag_value` = the anagram-component letters.
    `value_cands`, if given, overrides the DB lookup (used by the AI fallback)."""
    N = len(answer)
    forms = fodder_letter_forms(words[ana_run[0]:ana_run[1]])
    if value_cands is None:
        value_cands = _value_candidates(words, val_run[0], val_run[1], lookup_all)
    for F in forms:
        Lf = len(F)
        if Lf < 1 or Lf >= N:
            continue
        fsorted = sorted(F)
        for V in value_cands:
            Lv = len(V)
            if Lv < 1 or Lv + Lf != N:
                continue
            # Arrangement A: value is the OUTER, anagram is the INNER (length Lf).
            for p in range(0, Lv + 1):
                inner = answer[p:p + Lf]
                outer = answer[:p] + answer[p + Lf:]
                if outer == V and sorted(inner) == fsorted:
                    return (True, p, Lf, V, inner)
            # Arrangement B: value is the INNER (length Lv), anagram is the OUTER.
            for p in range(0, Lf + 1):
                inner = answer[p:p + Lv]
                outer = answer[:p] + answer[p + Lv:]
                if inner == V and sorted(outer) == fsorted:
                    return (False, p, Lv, V, outer)
    return None


def _place(slots, words, answer, postags, lookup_all, is_link, indicator_types,
           value_cands_fn=None):
    """Assign each slot a consecutive word-run in clue order (gaps allowed), validate
    the indicator slots, classify gaps as links LAST, then verify the insertion.
    Returns {ana_run, val_run, ana_i, con_i, links, insertion} or None."""
    n = len(words)
    nslots = len(slots)

    def residue_link(k):
        return (is_link and is_link(words[k].text)) or (postags[k] in GLUE_POS)

    def finalize(assigned, gap_idxs):
        ana_run = next((r for r, role in assigned if role == "ANA_F"), None)
        val_run = next((r for r, role in assigned if role in ("SYN_F", "ABR_F")), None)
        if ana_run is None or val_run is None:
            return None
        ana_i = [k for r, role in assigned if role == "ANA_I" for k in range(*r)]
        con_i = [k for r, role in assigned if role == "CON_I" for k in range(*r)]
        links = []
        for k in gap_idxs:
            if residue_link(k):
                links.append(k)
            else:
                return None                          # a content word unaccounted
        cands = value_cands_fn(val_run) if value_cands_fn else None
        ins = _verify_insertion(words, ana_run, val_run, answer, lookup_all, cands)
        if ins is None:
            return None
        return {"ana_run": ana_run, "val_run": val_run, "ana_i": sorted(ana_i),
                "con_i": sorted(con_i), "links": sorted(links), "insertion": ins}

    def dfs(si, wi, assigned, gaps):
        if si == nslots:
            return finalize(assigned, gaps + list(range(wi, n)))
        slot = slots[si]
        nw = slot.n_words
        for j in range(wi, n - nw + 1):
            run = (j, j + nw)
            if slot.role == "ANA_I" and not any(
                    is_anagram_indicator(words[k].text, indicator_types)
                    for k in range(*run)):
                continue
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
                  indicator_types, value_cands_fn=None):
    if split.where != template.def_pos:
        return None
    if template.fodder_word_count > len(words):
        return None
    roles = [s.role for s in template.slots]
    if not set(roles) <= {"ANA_F", "SYN_F", "ABR_F", "ANA_I", "CON_I"}:
        return None
    if roles.count("ANA_F") != 1:
        return None
    if sum(roles.count(r) for r in ("SYN_F", "ABR_F")) != 1:
        return None
    placement = _place(template.slots, words, answer, postags, lookup_all, is_link,
                       indicator_types, value_cands_fn)
    if placement is None:
        return None
    return _build(ctx, split, words, placement, template)


def _build(ctx, split, words, placement, template):
    """Assemble the Parse: one Source per component (anagram + value), per-letter links
    coloured by component (inner span vs outer), indicator + link annotations."""
    from core.definition_engine import dbe_annotation
    inner_anag, p, L, value, anag_value = placement["insertion"]
    ana_a, ana_b = placement["ana_run"]
    val_a, val_b = placement["val_run"]
    ana_toks = words[ana_a:ana_b]
    val_toks = words[val_a:val_b]

    ana_src = Source(clue_atom_ids=tuple(aid for t in ana_toks for aid in t.atom_ids),
                     text=" ".join(t.text for t in ana_toks), value=anag_value,
                     mechanism="anagram_fodder")
    val_src = Source(clue_atom_ids=tuple(aid for t in val_toks for aid in t.atom_ids),
                     text=" ".join(t.text for t in val_toks), value=value,
                     mechanism="synonym")
    # stable colour: sources in clue order
    if ana_a < val_a:
        sources = [ana_src, val_src]; ANA, VAL = 0, 1
    else:
        sources = [val_src, ana_src]; ANA, VAL = 1, 0

    # inner occupies answer[p:p+L]; the inner component is the anagram iff inner_anag.
    links = []
    for pos in range(1, len(ctx_answer(ctx)) + 1):
        in_inner = p < pos <= p + L
        is_anag_pos = (in_inner == inner_anag)
        si = ANA if is_anag_pos else VAL
        links.append(Link(answer_pos=pos, source_index=si,
                          operation="anagram_container", clue_atom_id=None,
                          transform="anagram_of" if is_anag_pos else None))

    annotations = []
    for k in placement["con_i"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="container indicator"))
    for k in placement["ana_i"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="anagram indicator"))
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
                  definition=definition, operation="anagram_container",
                  solved_by="catalog")
    parse.template_id = template.id
    parse.matched_signature = template.signature
    _verify(ctx, parse)
    return parse


def ctx_answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


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
    if any(getattr(s, "source", "db") == "pending" for s in parse.sources):
        warnings.append("a wordplay piece is provisional (queued for enrichment)")
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"


def solve_anagram_container(ctx, defines, lookup_all, is_link, indicator_types,
                            templates, define_fallback=None, is_dbe=None):
    """Full anagram+container solve — catalog-driven. Walk the anagram_container
    signatures in priority order; for each, for each definition split at its edge,
    place the roles and verify the insertion. First clean PASS (fewest residue), else
    best non-pass, else preserved fail-evidence."""
    from core.definition_engine import find_definitions

    answer = ctx_answer(ctx)
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
    return _build_fail_evidence(ctx, answer, prepared[0][0], lookup_all)


def _build_fail_evidence(ctx, answer, split, lookup_all):
    """Preserve the evidence when no signature instantiated: keep the definition and,
    if a contiguous wordplay run anagrams to a span of the answer, show it as the
    anagram-fodder candidate. No roles assigned by elimination."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    words = [t for t in split.wordplay_tokens if t.kind == "word"]
    sources = []
    n, N = len(words), len(answer)
    best = None
    for a in range(n):
        for b in range(a + 1, n + 1):
            for fl in fodder_letter_forms(words[a:b]):
                if 3 <= len(fl) <= N:
                    key = sorted(fl)
                    for start in range(0, N - len(fl) + 1):
                        sp = answer[start:start + len(fl)]
                        if sorted(sp) == key and (best is None or len(fl) > len(best[2])):
                            best = (a, b, sp)
                            break
    if best is not None:
        a, b, sp = best
        toks = words[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=sp,
            mechanism="anagram_fodder"))
    dbe = dbe_annotation(split)
    annotations = [dbe] if dbe is not None else []
    return Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                 sources=sources, links=[], annotations=annotations,
                 definition=definition, operation="anagram_container",
                 solved_by="catalog", status="fail",
                 warnings=["no anagram+container signature matched this clue "
                           "(the fodder below is a candidate, not a placement)"])
