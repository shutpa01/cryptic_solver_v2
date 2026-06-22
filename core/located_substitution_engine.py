"""Located-substitution engine — a synonym base with its FIRST or LAST letter
replaced by a clued value (DT 31272 DRESSES).

  DRESSES = TRESSES ("Hair") with the initial letter ("initially") cut and D
  ("diamonds") used instead → D + RESSES.

substitution_engine swaps one clued letter for another ANYWHERE in a base, and
needs BOTH letters to be clued word-values of the same length. It cannot reach
DRESSES because the removed letter (T) is not clued by any word — it is the base's
LOCATED first letter, named only by a position indicator ("initially"). This engine
fills exactly that gap: a base synonym, a LOCATED position (first/last) fixed by a
DB selection indicator, the located letter removed, and a DB value inserted there.

ANSWER-DRIVEN — the base with the located letter swapped for the value must EQUAL
the answer exactly. Gated on BOTH a substitution indicator AND a first/last
selection indicator. A DB deletion indicator present in the residue (e.g. "cut") is
accounted as the removal indicator — a genuine DB role, never assigned by
elimination. Residue otherwise must be DB link words; anything else leaves the
parse unaccounted (honest pending/fail, never a forced pass). _verify calls
role_validity so every recorded indicator/link role is DB-backed. Definition decided
upstream. Pure, DB-decoupled.
"""

from core.wfw_model import Source, Link, Annotation, Parse


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _values(word, lookup_all):
    out = []
    for v, m in lookup_all(word):
        u = (v or "").upper()
        if u and (u, m) not in out:
            out.append((u, m))
    return out


def solve_located_substitution(ctx, defines, lookup_all, synonyms_of, is_link,
                               indicator_types, selection_rules,
                               define_fallback=None, is_dbe=None):
    """Whole-answer located substitution. Abstains (None) unless a substitution
    indicator AND a first/last selection indicator are present and some base synonym,
    with its located (first/last) letter replaced by a clued value, equals the answer.
    First clean PASS, else best parse, else None."""
    from core.definition_engine import find_definitions
    answer = _answer(ctx)
    if len(answer) < 3:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    best = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 2:
            continue
        parse = _try_split(ctx, answer, split, words, lookup_all, synonyms_of,
                           is_link, indicator_types, selection_rules)
        if parse is None:
            continue
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best


def _try_split(ctx, answer, split, words, lookup_all, synonyms_of, is_link,
               indicator_types, selection_rules):
    n, N = len(words), len(answer)

    def types(k):
        try:
            return indicator_types(words[k].text) or set()
        except Exception:
            return set()

    def rules(k):
        try:
            return selection_rules(words[k].text) or set()
        except Exception:
            return set()

    sub_ind = [k for k in range(n) if "substitution" in types(k)]
    if not sub_ind:
        return None                              # gated: substitution indicator required
    # Location indicators that fix a FIRST or LAST position.
    loc = {}                                     # word index -> 'first' / 'last'
    for k in range(n):
        r = rules(k)
        if "first" in r:
            loc[k] = "first"
        elif "last" in r:
            loc[k] = "last"
    if not loc:
        return None                              # gated: a first/last location required

    val_cache = {}

    def values(k):
        if k not in val_cache:
            val_cache[k] = _values(words[k].text, lookup_all)
        return val_cache[k]

    best = None
    for li, where in loc.items():
        for vi in range(n):                      # the inserted value's word
            if vi == li or vi in sub_ind:
                continue
            for V, vmech in values(vi):
                for bstart in range(n):          # base span (contiguous)
                    for bend in range(bstart, n):
                        span = set(range(bstart, bend + 1))
                        if li in span or vi in span or span & set(sub_ind):
                            continue
                        phrase = " ".join(words[k].text for k in range(bstart, bend + 1))
                        for bval in (synonyms_of(phrase) or []):
                            B = "".join(c for c in (bval or "").upper() if c.isalpha())
                            if len(B) < 2:
                                continue
                            if where == "first":
                                cand = V + B[1:]
                                removed = B[0]
                            else:
                                cand = B[:-1] + V
                                removed = B[-1]
                            if cand != answer:
                                continue
                            used = span | {vi, li} | set(sub_ind)
                            parse = _build(ctx, split, words, answer, (bstart, bend),
                                           B, where, removed, (vi, V, vmech), li, sub_ind,
                                           used, is_link, types)
                            if parse is not None and parse.status == "pass":
                                return parse
                            if parse is not None and best is None:
                                best = parse      # keep best non-pass, keep searching
    return best


def _build(ctx, split, words, answer, base_span, B, where, removed, value, li,
           sub_ind, used, is_link, types):
    """Assemble the Parse. The base contributes all but its located letter; the value
    fills that end. A residue DB deletion indicator is accounted as the removal
    indicator (genuine DB role). Remaining residue must be DB links."""
    bs, be = base_span
    vi, V, vmech = value
    n = len(words)
    base_toks = words[bs:be + 1]

    # Residue classification: a deletion indicator -> the removal indicator (at most one);
    # a link word -> link; anything else -> unaccounted (left out, surfaced by _verify).
    rest = [k for k in range(n) if k not in used]
    del_ind, links = [], []
    took_del = False
    for k in rest:
        if not took_del and "deletion" in types(k):
            del_ind.append(k)
            took_del = True
        elif is_link and is_link(words[k].text):
            links.append(k)
        # else: leave unaccounted

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    base_source = Source(
        clue_atom_ids=tuple(aid for t in base_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in base_toks), value=B, mechanism="synonym")
    value_source = Source(clue_atom_ids=words[vi].atom_ids, text=words[vi].text,
                          value=V, mechanism=vmech)
    sources = [base_source, value_source]
    BASE_I, VAL_I = 0, 1

    # Links: one per answer letter. For 'first' the value sits at the front, then the
    # base remainder B[1:]; for 'last' the base remainder B[:-1] then the value.
    links_out = []
    if where == "first":
        for ci in range(len(V)):
            links_out.append(Link(answer_pos=ci + 1, source_index=VAL_I,
                                  operation="substitution"))
        for ci in range(len(B) - 1):
            links_out.append(Link(answer_pos=len(V) + ci + 1, source_index=BASE_I,
                                  operation="substitution"))
    else:
        for ci in range(len(B) - 1):
            links_out.append(Link(answer_pos=ci + 1, source_index=BASE_I,
                                  operation="substitution"))
        for ci in range(len(V)):
            links_out.append(Link(answer_pos=len(B) - 1 + ci + 1, source_index=VAL_I,
                                  operation="substitution"))

    annotations = []
    sub_toks = [words[i] for i in sub_ind]
    annotations.append(Annotation(
        clue_atom_ids=tuple(aid for t in sub_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in sub_toks), role="indicator",
        note="substitution: %s (%s) replaces the %s letter (%s) of %s"
             % (words[vi].text, V, where, removed, B)))
    annotations.append(Annotation(
        clue_atom_ids=words[li].atom_ids, text=words[li].text, role="indicator",
        note="selection indicator (%s)" % where))
    for k in del_ind:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="deletion indicator (removes the %s letter)"
                                           % where))
    for k in links:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link", note="link word"))
    from core.definition_engine import dbe_annotation
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links_out, annotations=annotations,
                  definition=definition, operation="substitution",
                  solved_by="catalog")
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
