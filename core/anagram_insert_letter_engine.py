"""Anagram containing a selected letter — an anagram of a fodder run with a single
SELECTED letter (the first or last letter of a clue word) inserted into it:

    "Source of pearls beginning to emerge in fancy story" = OYSTER
      def "Source of pearls"; anagram("fancy")(STORY) containing E
      (E = "beginning to emerge" = first letter of "emerge"); "in" = container.
      anagram(STORY + E) = OYSTER

A NEW stage. GATED on THREE indicators — an anagram indicator, a container/insertion
indicator, AND a first/last-letter selection indicator — and ANSWER-DRIVEN (the fodder
letters plus the one selected letter must anagram to the answer EXACTLY), so the triple
gate plus exact match make a false fire very unlikely. Definition decided upstream. Pure
and DB-decoupled.
"""

from collections import Counter

from core.wordplay import is_anagram_indicator, raw
from core.wfw_model import Source, Link, Annotation, Parse

MAX_FODDER = 6            # fodder run spans at most this many clue words


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def solve_anagram_insert_letter(ctx, defines, indicator_types, is_link,
                                define_fallback=None, is_dbe=None):
    from core.definition_engine import find_definitions
    answer = _answer(ctx)
    if len(answer) < 4:
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
        parse = _assemble(ctx, answer, split, words, indicator_types, is_link)
        if parse is not None:
            if parse.status == "pass":
                return parse
            if best is None:
                best = parse
    return best


def _assemble(ctx, answer, split, words, indicator_types, is_link):
    from core.selection_indicators import find_indicators
    n = len(words)

    def types(k):
        try:
            return indicator_types(words[k].text) or set()
        except Exception:
            return set()

    def is_con(k):
        return bool({"container", "insertion"} & types(k))

    def is_first_ind(k):
        return "acrostic" in types(k)

    anag = [k for k in range(n) if is_anagram_indicator(words[k].text, indicator_types)]
    con = [k for k in range(n) if is_con(k)]
    if not anag or not con:
        return None
    # selection indicators: first via 'acrostic' type or selection_indicators 'first';
    # last via selection_indicators 'last'.
    sel = find_indicators(words)                    # [(rule, (idx, ...)), ...]
    first_inds = [k for k in range(n) if is_first_ind(k)]
    first_inds += [i for r, idxs in sel if r == "first" for i in idxs]
    last_inds = [i for r, idxs in sel if r == "last" for i in idxs]
    if not first_inds and not last_inds:
        return None

    ans_sorted = sorted(answer)
    # try each fodder run + each source word + first/last selected letter
    for a in range(n):
        for b in range(a + 1, min(a + MAX_FODDER, n) + 1):
            fodder = list(range(a, b))
            fraw = "".join(raw(words[k].text) for k in fodder)
            if not fraw or len(fraw) + 1 != len(answer):
                continue
            for sk in range(n):
                if a <= sk < b:
                    continue                        # source word not in the fodder
                wl = raw(words[sk].text)
                if not wl:
                    continue
                for mode, inds in (("first", first_inds), ("last", last_inds)):
                    if not inds:
                        continue
                    L = wl[0] if mode == "first" else wl[-1]
                    if sorted(fraw + L) != ans_sorted:
                        continue
                    # the selection indicator: nearest of `inds` that is not fodder/source
                    sel_idx = next((i for i in inds if not (a <= i < b) and i != sk), None)
                    if sel_idx is None:
                        continue
                    pl = _account(n, fodder, sk, anag, con, sel_idx, is_con, is_link, words)
                    if pl is None:
                        continue
                    return _build(ctx, answer, split, words, fodder, sk, L, mode,
                                  pl, indicator_types)
    return None


def _account(n, fodder, sk, anag, con, sel_idx, is_con, is_link, words):
    """Assign every clue word a role: fodder, source(sk), anagram indicator, container
    indicator, selection indicator, or link. A bare unaccounted word -> reject."""
    used = set(fodder) | {sk, sel_idx}
    anag_run = [k for k in anag if k not in used]
    if not anag_run:
        return None
    con_run = [k for k in con if k not in used and k not in anag_run]
    if not con_run:
        return None
    used |= set(anag_run) | set(con_run)
    links = []
    for k in range(n):
        if k in used:
            continue
        if is_link and is_link(words[k].text):
            links.append(k)
        else:
            return None
    return {"anag": anag_run, "con": con_run, "sel": sel_idx, "links": links}


def _build(ctx, answer, split, words, fodder, sk, L, mode, pl, indicator_types):
    from core.definition_engine import dbe_annotation
    sources, remaining = [], []
    for k in fodder:
        wl = raw(words[k].text)
        remaining.append([len(sources), Counter(wl)])
        sources.append(Source(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                              value=wl, mechanism="anagram_fodder"))
    # the selected letter as its own source
    sel_src_i = len(sources)
    remaining.append([sel_src_i, Counter(L)])
    sources.append(Source(clue_atom_ids=words[sk].atom_ids, text=words[sk].text,
                          value=L, mechanism=("first_letter" if mode == "first"
                                              else "last_letter")))
    links = []
    for pos_i, ch in enumerate(answer, start=1):
        si = 0
        for entry in remaining:
            if entry[1].get(ch, 0) > 0:
                entry[1][ch] -= 1
                si = entry[0]
                break
        links.append(Link(answer_pos=pos_i, source_index=si, operation="anagram",
                          clue_atom_id=None, transform="anagram_of"))

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition", source=split.source)
    annotations = []
    for k in pl["anag"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                                      role="indicator", note="anagram indicator"))
    for k in pl["con"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                                      role="indicator", note="container indicator"))
    annotations.append(Annotation(
        clue_atom_ids=words[pl["sel"]].atom_ids, text=words[pl["sel"]].text,
        role="indicator", note="%s-letter indicator" % mode))
    for k in pl["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                                      role="link", note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="anagram", solved_by="catalog")
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully covered by the fodder")
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
