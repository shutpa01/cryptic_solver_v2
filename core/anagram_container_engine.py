"""Anagram+container engine — a container where one component is an anagram.

Two shapes, both an insertion (answer = outer with inner inserted) where exactly one
of outer/inner is an anagram and the other a DB value:
  - anagram INNER:  SANDWICHES = SANDS around anag(I CHEW);  PANGOLIN = PAIN around anag(LONG)
  - anagram OUTER:  EXHORT = anag(HER TO)=EHORT around X (kiss)

EVIDENCE-DRIVEN, links classified LAST (memory: feedback-never-preassign-links). It
does not pre-assign anything. It enumerates the insertion (where the inner sits in
the answer), then looks for wordplay word-runs that PRODUCE the inner and the outer —
one by the letter-match (anagram), one by a DB lookup (synonym/abbreviation, multi-
word phrases included). Whatever words are left must contain a container indicator
and an anagram indicator; the rest are links (is_link or POS function/VERB/ADV);
anything else leaves the parse unaccounted.

Gated: requires BOTH a container indicator and an anagram indicator. Definition
decided upstream (def_pos). Per-piece colour (the outer's two segments share its
colour). Pure and DB-decoupled.
"""

from core import grammar
from core.wordplay import GLUE_POS, fodder_letter_forms, adjacent_run
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
MAX_RUN = 5


def _run_letters(words, a, b):
    """As-written and contraction-stripped letters of words[a:b]."""
    return fodder_letter_forms(words[a:b])


def _run_values(words, a, b, lookup_all):
    """DB synonym/abbreviation values for the phrase words[a:b] — UNFILTERED (the
    container outer is split around the inner, so it is not a substring of answer)."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out = []
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in out:
            out.append(v)
    return out


def _produces(words, a, b, target, lookup_all):
    """How words[a:b] can produce `target`: returns set of {'anag','value'}."""
    ways = set()
    if any(sorted(s) == sorted(target) for s in _run_letters(words, a, b)):
        ways.add("anag")
    if target in _run_values(words, a, b, lookup_all):
        ways.add("value")
    return ways


def _assemble(answer, words, postags, lookup_all, is_link, indicator_types):
    """Find outer/inner runs + insertion. Returns a placement or None."""
    n, N = len(words), len(answer)

    def has(text, kind):
        try:
            return kind in (indicator_types(text) or set())
        except Exception:
            return False

    def is_con(k):
        return has(words[k].text, "container") or has(words[k].text, "insertion")

    def is_ana(k):
        return has(words[k].text, "anagram")

    def residue_link(k):
        return (is_link and is_link(words[k].text)) or (postags[k] in GLUE_POS)

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    best = None     # a confirmed (db) placement wins; provisional kept only as fallback

    # Enumerate the insertion: inner = answer[p:p+L], outer = answer[:p]+answer[p+L:].
    for p in range(0, N):
        for L in range(1, N - p + 1):
            if p == 0 and p + L == N:
                continue                              # outer must be non-empty
            inner = answer[p:p + L]
            outer = answer[:p] + answer[p + L:]
            if not outer:
                continue
            for (ia, ib) in runs:
                inner_ways = _produces(words, ia, ib, inner, lookup_all)
                if not inner_ways:
                    continue
                for (oa, ob) in runs:
                    if not (ob <= ia or oa >= ib):    # runs must be disjoint
                        continue
                    outer_ways = _produces(words, oa, ob, outer, lookup_all)
                    if not outer_ways:
                        continue
                    # exactly one component is an anagram, the other a DB value
                    for inner_anag in (True, False):
                        need_in = "anag" if inner_anag else "value"
                        need_out = "value" if inner_anag else "anag"
                        if need_in not in inner_ways or need_out not in outer_ways:
                            continue
                        used = set(range(ia, ib)) | set(range(oa, ob))
                        residue = [k for k in range(n) if k not in used]
                        con_caps = [k for k in residue if is_con(k)]
                        ana_caps = [k for k in residue if is_ana(k)]
                        # The container and anagram indicators must be DISTINCT words
                        # (a word like "about" is tagged both, but here it is one or
                        # the other). Pick a container word, then anagram words that
                        # are not it; the rest must be links.
                        anag_run = (ia, ib) if inner_anag else (oa, ob)
                        for c in con_caps:
                            ana = [k for k in ana_caps if k != c]
                            ana_source = "db"
                            if not ana:
                                # MISSING-INDICATOR FALLBACK: a container indicator is
                                # present and the anagram is proven by its letters, so
                                # the residue word adjacent to the anagram component is
                                # the anagram indicator — accept it provisionally.
                                ana = [k for k in adjacent_run(
                                    residue, anag_run[0], anag_run[1]) if k != c]
                                if not ana:
                                    continue
                                ana_source = "pending"
                            spoken = {c} | set(ana)
                            links, ok = [], True
                            for k in residue:
                                if k in spoken:
                                    continue
                                if residue_link(k):
                                    links.append(k)
                                else:
                                    ok = False
                                    break
                            if ok:
                                placement = {"p": p, "L": L, "inner": (ia, ib),
                                             "outer": (oa, ob),
                                             "inner_anag": inner_anag, "con": [c],
                                             "ana": ana, "ana_source": ana_source,
                                             "links": links}
                                if ana_source == "db":
                                    return placement     # confirmed — best, stop
                                if best is None:
                                    best = placement     # provisional — keep as fallback
    return best


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    p, L = pl["p"], pl["L"]
    ia, ib = pl["inner"]
    oa, ob = pl["outer"]
    inner_anag = pl["inner_anag"]
    inner_toks = words[ia:ib]
    outer_toks = words[oa:ob]
    inner = answer[p:p + L]
    outer = answer[:p] + answer[p + L:]

    outer_src = Source(
        clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in outer_toks), value=outer,
        mechanism="anagram_fodder" if not inner_anag else "synonym")
    inner_src = Source(
        clue_atom_ids=tuple(aid for t in inner_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in inner_toks), value=inner,
        mechanism="anagram_fodder" if inner_anag else "synonym")
    # source order in clue order, for stable colour
    if oa < ia:
        sources = [outer_src, inner_src]; OUT, IN = 0, 1
    else:
        sources = [inner_src, outer_src]; OUT, IN = 1, 0

    links = []
    for pos in range(1, len(answer) + 1):
        if p < pos <= p + L:
            si, anag = IN, inner_anag
        else:
            si, anag = OUT, (not inner_anag)
        links.append(Link(answer_pos=pos, source_index=si,
                          operation="anagram_container", clue_atom_id=None,
                          transform="anagram_of" if anag else None))

    annotations = []
    for k in pl["con"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="container indicator"))
    for k in pl["ana"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="anagram indicator",
                                      source=pl.get("ana_source", "db")))
    for k in pl["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="anagram_container",
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
    if any(a.role == "indicator" and getattr(a, "source", "db") == "pending"
           for a in parse.annotations):
        warnings.append("the anagram indicator is provisional (queued for enrichment)")
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"


def solve_anagram_container(ctx, defines, lookup_all, is_link, indicator_types,
                            templates=None, define_fallback=None, is_dbe=None):
    """Full anagram+container solve — evidence-driven. Returns the first clean PASS,
    else the best parse, else None."""
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
        if len(words) < 3:
            continue
        postags = grammar.pos_tags([t.text for t in words]) or [None] * len(words)
        pl = _assemble(answer, words, postags, lookup_all, is_link, indicator_types)
        if pl is None:
            continue
        parse = _build(ctx, split, words, answer, pl)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best
