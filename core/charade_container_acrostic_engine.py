"""Charade combining a CONTAINER piece and an ACROSTIC piece (DT 10076613 INDICATES).

  INDICATES = [INDIA around C] + TES
    "Asian country" = INDIA (outer)   "Clubs" = C (inner)   "in" = insertion indicator
      -> INDICA (the container piece, INDI-C-A)
    "initially try erotic strip" -> T, E, S = TES (acrostic piece)
    "shows" = definition

This falls exactly between two existing engines, neither of which can reach it:
  - charade_acrostic tiles PLAIN DB-value pieces + an acrostic, but cannot make a piece that
    is itself a CONTAINER (INDIA around C is not a plain DB value);
  - container_charade allows one CONTAINER piece + plain value pieces, but cannot supply an
    ACROSTIC piece (TES is not a DB value of any run).

A NEW bespoke stage (per the project rule: never edit a working engine to add a case). It
reuses the proven primitives — container_charade's OUTER-around-INNER enumeration and
charade_acrostic's first/last-letter selection. The answer is tiled left-to-right by:
  - value pieces:     a DB synonym/abbreviation/raw value of a contiguous clue word-run;
  - ONE container piece: a span = OUTER around INNER, both DB values of disjoint runs;
  - acrostic pieces:  the first (or last) letter of each of a run of >=2 consecutive words.
It REQUIRES >=1 container piece AND >=1 acrostic piece, so it can never intercept a plain
charade, a container_charade, or a charade_acrostic. ANSWER-DRIVEN (exact reconstruction),
gated on BOTH a container/insertion indicator AND an acrostic indicator in the residue, all
other words DB links. Returns ONLY a clean PASS (a conservative compound — never displaces a
simpler engine's pending/fail). _verify calls role_validity. Definition decided upstream.
Pure and DB-decoupled.
"""

from core import selection, engine_common
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation", "raw")
_MODE_MECHANISM = {"first": "first_letter", "last": "last_letter"}
MAX_RUN = 4            # max words in a plain value piece / container outer or inner run
MAX_ACRO = 6           # max words spanned by an acrostic piece


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


def _mech_for(words, a, b, value, lookup_all):
    """The DB mechanism (abbreviation/synonym/raw) that yields `value` for run words[a:b] —
    so the piece is labelled faithfully (Clubs->C is an abbreviation, not a synonym)."""
    phrase = " ".join(words[k].text for k in range(a, b))
    for val, mech in lookup_all(phrase):
        if (val or "").upper() == value and mech in _VALUE_MECH:
            return mech
    return "synonym"


def _has(words, indicator_types, *kinds):
    for t in words:
        try:
            ty = indicator_types(t.text) or set()
        except Exception:
            ty = set()
        if ty & set(kinds):
            return True
    return False


def solve_charade_container_acrostic(ctx, defines, lookup_all, is_link, indicator_types,
                                     define_fallback=None, is_dbe=None):
    """Charade with one container piece and >=1 acrostic piece. Returns ONLY a clean PASS."""
    from core.definition_engine import find_definitions
    answer = _answer(ctx)
    if len(answer) < 4 or indicator_types is None:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 4:                            # outer + inner + 2 acrostic words, min
            continue
        # cheap pre-gate: both an insertion and an acrostic indicator must be present
        if not (_has(words, indicator_types, "container", "insertion")
                and _has(words, indicator_types, "acrostic")):
            continue
        parse = _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types)
        if parse is not None and parse.status == "pass":
            return parse
    return None


def _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types):
    n, N = len(words), len(answer)
    val_cache = {}

    def values(a, b):
        if (a, b) not in val_cache:
            val_cache[(a, b)] = _run_values(words, a, b, lookup_all)
        return val_cache[(a, b)]

    def avail_runs(used):
        return [(a, b) for a in range(n) if a not in used
                for b in range(a + 1, min(a + MAX_RUN, n) + 1)
                if not (set(range(a, b)) & used)]

    # DFS tiling: value pieces + ONE container piece + >=1 acrostic piece.
    def dfs(pos, used, pieces, con_used, n_acro):
        if pos == N:
            if not con_used or n_acro < 1:
                return None
            return _finalize(ctx, split, words, answer, pieces, used, is_link,
                             indicator_types, lookup_all)
        # 1. plain value piece
        for a in range(n):
            if a in used:
                continue
            for b in range(a + 1, min(a + MAX_RUN, n) + 1):
                if any(k in used for k in range(a, b)):
                    break
                for v in values(a, b):
                    if answer.startswith(v, pos):
                        r = dfs(pos + len(v), used | set(range(a, b)),
                                pieces + [("value", (a, b), v)], con_used, n_acro)
                        if r:
                            return r
        # 2. container piece (at most once) — span at pos = OUTER around INNER
        if not con_used:
            for L in range(2, N - pos + 1):
                span = answer[pos:pos + L]
                for q in range(1, L):
                    for Li in range(1, L - q + 1):
                        inner = span[q:q + Li]
                        outer = span[:q] + span[q + Li:]
                        if not outer:
                            continue
                        for (ia, ib) in avail_runs(used):
                            if inner not in values(ia, ib):
                                continue
                            u2 = used | set(range(ia, ib))
                            for (oa, ob) in avail_runs(u2):
                                if outer not in values(oa, ob):
                                    continue
                                r = dfs(pos + L, u2 | set(range(oa, ob)),
                                        pieces + [("container", (oa, ob), (ia, ib),
                                                   L, q, Li, outer, inner)], True, n_acro)
                                if r:
                                    return r
        # 3. acrostic piece — first/last letter of each of a run of >=2 consecutive words
        for mode in ("first", "last"):
            for a in range(n):
                if a in used:
                    continue
                for b in range(a + 2, min(a + MAX_ACRO, n) + 1):
                    if any(k in used for k in range(a, b)):
                        break
                    sel = selection.selected(ctx, words[a:b], mode)
                    if sel is None:
                        continue
                    s = "".join(c for c, _ in sel)
                    if s and answer.startswith(s, pos):
                        r = dfs(pos + len(s), used | set(range(a, b)),
                                pieces + [("acro", (a, b), s, (mode, sel))],
                                con_used, n_acro + 1)
                        if r:
                            return r
        return None

    return dfs(0, set(), [], False, 0)


def _finalize(ctx, split, words, answer, pieces, used, is_link, indicator_types, lookup_all):
    n = len(words)
    residue = set(k for k in range(n) if k not in used)
    # both indicators must sit in the residue (disjoint runs); rest must be DB links.
    con_pos = engine_common.find_typed_run(words, residue, indicator_types, "insertion",
                                            min_length=1)
    if con_pos is None:
        con_pos = engine_common.find_typed_run(words, residue, indicator_types, "container",
                                               min_length=1)
    if con_pos is None:
        return None
    acro_pos = engine_common.find_typed_run(words, residue - set(con_pos), indicator_types,
                                            "acrostic", min_length=1)
    if acro_pos is None:
        return None
    spoken = set(con_pos) | set(acro_pos)
    links = []
    for k in sorted(residue):
        if k in spoken:
            continue
        if is_link and is_link(words[k].text):
            links.append(k)
        else:
            return None                               # unaccounted content word -> reject
    return _build(ctx, split, words, answer, pieces, con_pos, acro_pos, links, lookup_all)


def _build(ctx, split, words, answer, pieces, con_pos, acro_pos, links, lookup_all):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition", source=split.source)
    sources, links_out, pos = [], [], 0
    for piece in pieces:
        if piece[0] == "value":
            _, (a, b), v = piece
            toks = words[a:b]
            si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                text=" ".join(t.text for t in toks), value=v,
                mechanism=_mech_for(words, a, b, v, lookup_all)))
            for _ in v:
                pos += 1
                links_out.append(Link(answer_pos=pos, source_index=si,
                                      operation="charade", clue_atom_id=None))
        elif piece[0] == "container":
            _, (oa, ob), (ia, ib), L, q, Li, outer, inner = piece
            outer_toks, inner_toks = words[oa:ob], words[ia:ib]
            o_si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
                text=" ".join(t.text for t in outer_toks), value=outer,
                mechanism=_mech_for(words, oa, ob, outer, lookup_all)))
            i_si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in inner_toks for aid in t.atom_ids),
                text=" ".join(t.text for t in inner_toks), value=inner,
                mechanism=_mech_for(words, ia, ib, inner, lookup_all)))
            for off in range(L):
                pos += 1
                si = i_si if q <= off < q + Li else o_si
                links_out.append(Link(answer_pos=pos, source_index=si,
                                      operation="container", clue_atom_id=None))
        else:  # acrostic: one Source per selected letter, with per-letter provenance
            _, (a, b), s, (mode, sel) = piece
            mech = _MODE_MECHANISM[mode]
            for (char, aid), tok in zip(sel, words[a:b]):
                si = len(sources)
                sources.append(Source(clue_atom_ids=tok.atom_ids, text=tok.text,
                                      value=char, mechanism=mech, source="db"))
                pos += 1
                links_out.append(Link(answer_pos=pos, source_index=si,
                                      operation="acrostic", clue_atom_id=aid))

    con_toks = [words[i] for i in con_pos]
    acro_toks = [words[i] for i in acro_pos]
    annotations = [
        Annotation(clue_atom_ids=tuple(aid for t in con_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in con_toks), role="indicator",
                   note="container indicator"),
        Annotation(clue_atom_ids=tuple(aid for t in acro_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in acro_toks), role="indicator",
                   note="acrostic indicator"),
    ]
    for k in links:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                                      role="link", note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links_out, annotations=annotations,
                  definition=definition, operation="charade_container_acrostic",
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
