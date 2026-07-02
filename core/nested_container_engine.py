"""Nested container — a container inside a container (Times 29575 VACUUM, FALLENANGEL).

  VACUUM      = VAM ("5am") ∋ [ CU ("Copper") ∋ U ("university") ]
                = VA-CU-U-M, "interrupts" + "guarding" the two insertions.
  FALLENANGEL = FL ("Florida") ∋ [ ALLEGE ("claim") ∋ NAN ("relative") ]
                = F-ALLE-NAN-GE-L, "in" + "keeping" the two insertions.

The plain container engines insert ONE DB value into another. Neither reaches a
DOUBLE insertion (an OUTER value wrapping a MIDDLE value that itself wraps an INNER
value). This engine does exactly that, and only that: THREE distinct DB values and
TWO container indicators, so it cannot intercept a single-container clue (which has
one indicator) or anything simpler.

ANSWER-DRIVEN. The answer is split as outer_left + [ mid_left + inner + mid_right ] +
outer_right, where OUTER = outer_left+outer_right, MIDDLE = mid_left+mid_right and
INNER = inner are each a DB synonym/abbreviation of a distinct clue word-run, and both
wraps are GENUINE (all four flank segments non-empty, so each value truly surrounds the
next). Residue must hold TWO distinct container indicators + DB link words; anything
else leaves the parse unaccounted. Returns ONLY a clean PASS (a deliberately
conservative compound engine — it never displaces a simpler engine's pending/fail).
_verify calls role_validity so every recorded indicator/link role is DB-backed.
Definition decided upstream. Pure, DB-decoupled.
"""

from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation", "raw")
MAX_RUN = 5


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _run_values(words, a, b, lookup_all):
    phrase = " ".join(words[k].text for k in range(a, b))
    out = []
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in out:
            out.append(v)
    return out


def solve_nested_container(ctx, defines, lookup_all, is_link, indicator_types,
                           define_fallback=None, is_dbe=None):
    """Double-insertion container. Returns ONLY a clean PASS, else None.

    Gated on >=2 container indicators; answer-driven; three distinct DB values. Like the
    other compound engines it surfaces nothing but a confirmed pass, so it can never
    displace a simpler engine's pending/fail with a speculative nested reading."""
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
        if len(words) < 3:
            continue
        parse = _try_split(ctx, answer, split, words, lookup_all, is_link,
                           indicator_types)
        if parse is not None and parse.status == "pass":
            return parse
    return None


def _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types):
    n, N = len(words), len(answer)

    def is_con(k):
        try:
            ty = indicator_types(words[k].text) or set()
        except Exception:
            ty = set()
        return "container" in ty or "insertion" in ty

    # Gate: at least two container indicators must be present.
    con_words = [k for k in range(n) if is_con(k)]
    if len(con_words) < 2:
        return None

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    # value string -> list of runs (a, b) that produce it
    val_runs = {}
    for (a, b) in runs:
        for v in _run_values(words, a, b, lookup_all):
            val_runs.setdefault(v, []).append((a, b))

    def runs_for(value):
        return val_runs.get(value, ())

    # Enumerate the two GENUINE insertions (all flank segments non-empty):
    #   answer = outer_left + X + outer_right        (OUTER = outer_left+outer_right)
    #   X      = mid_left  + inner + mid_right        (MIDDLE = mid_left+mid_right)
    for p1 in range(1, N - 1):                         # outer_left non-empty
        for L1 in range(1, N - p1):                    # outer_right non-empty
            outer = answer[:p1] + answer[p1 + L1:]
            X = answer[p1:p1 + L1]
            if not outer or len(X) < 3:                # X must hold middle+inner (>=1 each side)
                continue
            outer_rs = runs_for(outer)
            if not outer_rs:
                continue
            Lx = len(X)
            for p2 in range(1, Lx - 1):                # mid_left non-empty
                for L2 in range(1, Lx - p2):           # mid_right non-empty
                    inner = X[p2:p2 + L2]
                    middle = X[:p2] + X[p2 + L2:]
                    if not inner or not middle:
                        continue
                    mid_rs = runs_for(middle)
                    inner_rs = runs_for(inner)
                    if not mid_rs or not inner_rs:
                        continue
                    parse = _assemble(ctx, split, words, answer, is_link, is_con,
                                      (p1, L1, p2, L2), outer, middle, inner,
                                      outer_rs, mid_rs, inner_rs)
                    if parse is not None and parse.status == "pass":
                        return parse
    return None


def _assemble(ctx, split, words, answer, is_link, is_con, geom, outer, middle, inner,
              outer_rs, mid_rs, inner_rs):
    n = len(words)
    for o in outer_rs:
        for m in mid_rs:
            if _overlap(o, m):
                continue
            for i in inner_rs:
                if _overlap(o, i) or _overlap(m, i):
                    continue
                used = _span(o) | _span(m) | _span(i)
                residue = [k for k in range(n) if k not in used]
                cons = [k for k in residue if is_con(k)]
                if len(cons) < 2:
                    continue
                # Pick two distinct container indicators; the rest must be DB links.
                for ci in range(len(cons)):
                    for cj in range(len(cons)):
                        if ci == cj:
                            continue
                        c_out, c_mid = cons[ci], cons[cj]
                        spoken = {c_out, c_mid}
                        links, ok = [], True
                        for k in residue:
                            if k in spoken:
                                continue
                            if is_link and is_link(words[k].text):
                                links.append(k)
                            else:
                                ok = False
                                break
                        if not ok:
                            continue
                        parse = _build(ctx, split, words, answer, geom, outer, middle,
                                       inner, o, m, i, c_out, c_mid, links)
                        if parse is not None and parse.status == "pass":
                            return parse
    return None


def _overlap(r1, r2):
    return not (r1[1] <= r2[0] or r2[1] <= r1[0])


def _span(r):
    return set(range(r[0], r[1]))


def _build(ctx, split, words, answer, geom, outer, middle, inner, o_run, m_run, i_run,
           c_out, c_mid, links):
    from core.definition_engine import dbe_annotation
    p1, L1, p2, L2 = geom
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)

    def src(run, value):
        toks = words[run[0]:run[1]]
        return Source(clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                      text=" ".join(t.text for t in toks), value=value,
                      mechanism="synonym")

    sources = [src(o_run, outer), src(m_run, middle), src(i_run, inner)]
    OUT, MID, INN = 0, 1, 2

    # Map each answer position to its source. Layout (0-based answer offsets):
    #   [0, p1)            -> OUTER (outer_left)
    #   X = [p1, p1+L1):   mid_left -> MIDDLE, inner -> INNER, mid_right -> MIDDLE
    #   [p1+L1, N)         -> OUTER (outer_right)
    links_out = []
    for pos in range(len(answer)):
        if pos < p1 or pos >= p1 + L1:
            si = OUT
        else:
            xoff = pos - p1
            if xoff < p2:
                si = MID
            elif xoff < p2 + L2:
                si = INN
            else:
                si = MID
        links_out.append(Link(answer_pos=pos + 1, source_index=si,
                              operation="container"))

    annotations = [
        Annotation(clue_atom_ids=words[c_out].atom_ids, text=words[c_out].text,
                   role="indicator", note="container indicator"),
        Annotation(clue_atom_ids=words[c_mid].atom_ids, text=words[c_mid].text,
                   role="indicator", note="container indicator"),
    ]
    for k in links:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link", note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links_out, annotations=annotations,
                  definition=definition, operation="container", solved_by="catalog")
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
