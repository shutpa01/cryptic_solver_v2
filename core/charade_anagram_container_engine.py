"""Charade + anagram-container — a charade where ONE piece is an anagram-container
(DT 31272 DOGFIGHT).

  DOGFIGHT = D ("Day") + [ anag("fog hit") with G inserted ]
           = D + OGFIGHT,  where OGFIGHT = anag(FOGHIT) "awful" containing G "hiding",
             and G = the first letter of "Germany('s)" named by "leader".

The plain anagram_container engine builds [anag ∋ value] as the WHOLE answer; it has no
charade prefix/suffix, so it cannot reach D + OGFIGHT. This engine adds exactly that:
the answer is tiled left-to-right by plain charade pieces (DB synonym/abbreviation
values) PLUS exactly ONE contiguous anagram-container span. It requires >=1 plain piece
AND exactly one anagram-container span, so it cannot intercept a plain charade (no AC)
or a plain anagram-container (no extra charade piece) — both already claimed earlier.

The anagram-container span is resolved like anagram_container_engine: an inner/outer
insertion where one side is an anagram of a fodder run (letter-proven) and the other is
the inserted VALUE. The value may be a DB synonym/abbreviation OR a first/last-letter
SELECTION of a single word licensed by a selection indicator (Germany's "leader" -> G).

ANSWER-DRIVEN throughout. Gated on BOTH an anagram indicator AND a container indicator.
_verify calls role_validity so every recorded indicator/link role is DB-backed.
Definition decided upstream. Pure, DB-decoupled.
"""

from core.wordplay import fodder_letter_forms
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
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


def _anag_ok(words, a, b, target):
    """True only for a GENUINE anagram: the fodder must be >=3 letters and produce the
    target by a real REARRANGEMENT (same multiset, different order). This rejects the
    degenerate 1-letter "anagram" (a -> A) and a no-op identity that would otherwise let
    this engine fabricate a charade+anagram-container reading of a plain container/charade
    clue (e.g. RUN ACROSS = RUN + 'a'->A + CROSS)."""
    if len(target) < 3:
        return False
    for s in fodder_letter_forms(words[a:b]):
        if len(s) == len(target) and s != target and sorted(s) == sorted(target):
            return True
    return False


def solve_charade_anagram_container(ctx, defines, lookup_all, is_link, indicator_types,
                                    selection_rules, define_fallback=None, is_dbe=None):
    """Charade with one anagram-container piece. Returns ONLY a clean PASS, else None.

    Deliberately conservative for a compound engine: it surfaces nothing unless it has a
    fully DB-grounded PASS, so it can never DISPLACE a simpler engine's pending or fail
    with a speculative charade+anagram-container reading (observed displacing a plain
    container/anagram pending). A genuine clue it cannot fully solve falls through to those
    engines unchanged. Gated on an anagram AND a container indicator; answer-driven."""
    from core.definition_engine import find_definitions
    answer = _answer(ctx)
    if len(answer) < 5:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 4:
            continue
        parse = _try_split(ctx, answer, split, words, lookup_all, is_link,
                           indicator_types, selection_rules)
        if parse is not None and parse.status == "pass":
            return parse
    return None


def _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types,
               selection_rules):
    from core.selection import select_span
    from core.selection_indicators import find_indicators
    n, N = len(words), len(answer)

    def has(text, kind):
        try:
            return kind in (indicator_types(text) or set())
        except Exception:
            return False

    # Gate: an anagram AND a container indicator must be present somewhere.
    if not any(has(t.text, "anagram") for t in words):
        return None
    if not any(has(t.text, "container") or has(t.text, "insertion") for t in words):
        return None

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    # Prefer the SHORTEST licensing selection indicator, so a one-word indicator
    # ("leader") wins over a longer phrase ("leader in") and the extra word ("in") is
    # left free to be the container's link — the more natural attribution.
    sel_options = sorted(find_indicators(words), key=lambda ri: len(ri[1]))

    val_cache = {}

    def values(run):
        if run not in val_cache:
            val_cache[run] = _run_values(words, run[0], run[1], lookup_all)
        return val_cache[run]

    def disjoint(used, run):
        return not (set(range(run[0], run[1])) & used)

    def ac_for_span(span, used):
        """Anagram-container placements for `span` using words disjoint from `used`.
        Yields dicts: inner/outer runs, which side is the anagram, the value side's
        source (db run, or a selection word + its licensing indicator)."""
        out = []
        M = len(span)
        for p in range(M):
            for L in range(1, M - p + 1):
                if p == 0 and p + L == M:
                    continue                         # outer must be non-empty
                inner = span[p:p + L]
                outer = span[:p] + span[p + L:]
                if not outer:
                    continue
                for (ia, ib) in runs:
                    if not disjoint(used, (ia, ib)):
                        continue
                    inner_anag_ok = _anag_ok(words, ia, ib, inner)
                    inner_vals = values((ia, ib))
                    for (oa, ob) in runs:
                        if not disjoint(used, (oa, ob)):
                            continue
                        if not (ob <= ia or oa >= ib):
                            continue                 # inner/outer runs disjoint
                        outer_anag_ok = _anag_ok(words, oa, ob, outer)
                        outer_vals = values((oa, ob))
                        for inner_anag in (True, False):
                            if inner_anag:
                                if not inner_anag_ok:
                                    continue
                                if outer not in outer_vals:
                                    continue
                                val_run, anag_run = (oa, ob), (ia, ib)
                                val_target = outer
                            else:
                                if not outer_anag_ok:
                                    continue
                                if inner not in inner_vals:
                                    continue
                                val_run, anag_run = (ia, ib), (oa, ob)
                                val_target = inner
                            out.append({"p": p, "L": L, "inner": (ia, ib),
                                        "outer": (oa, ob), "inner_anag": inner_anag,
                                        "anag_run": anag_run, "val_run": val_run,
                                        "val_target": val_target, "val_kind": "db",
                                        "val_word": None, "sel": None,
                                        "used": set(range(ia, ib)) | set(range(oa, ob))})
        # value via first/last SELECTION of a single word (Germany's "leader" -> G)
        for p in range(M):
            for L in range(1, M - p + 1):
                if p == 0 and p + L == M:
                    continue
                inner = span[p:p + L]
                outer = span[:p] + span[p + L:]
                if not outer:
                    continue
                for inner_anag in (True, False):
                    anag_target = inner if inner_anag else outer
                    val_target = outer if inner_anag else inner
                    for (aa, ab) in runs:            # the anagram fodder run
                        if not disjoint(used, (aa, ab)):
                            continue
                        if not _anag_ok(words, aa, ab, anag_target):
                            continue
                        for rule, idxs in sel_options:
                            if rule not in ("first", "last"):
                                continue
                            for j in range(n):       # the selection source word
                                if j in used or j in set(idxs):
                                    continue
                                if aa <= j < ab:
                                    continue         # not part of the fodder
                                for s, atom_ids in select_span(ctx, words[j], rule):
                                    if s.upper() != val_target:
                                        continue
                                    anag_run = (aa, ab)
                                    out.append({"p": p, "L": L,
                                                "inner": (aa, ab) if inner_anag else (j, j + 1),
                                                "outer": (j, j + 1) if inner_anag else (aa, ab),
                                                "inner_anag": inner_anag,
                                                "anag_run": anag_run, "val_run": (j, j + 1),
                                                "val_target": val_target, "val_kind": "sel",
                                                "val_word": (j, rule, atom_ids),
                                                "sel": (rule, tuple(idxs)),
                                                "used": set(range(aa, ab)) | {j}})
        return out

    # DFS: tile the answer with plain charade pieces + exactly one anagram-container span.
    def dfs(pos, used, used_sel, n_plain, ac, pieces):
        if pos == N:
            if ac is None or n_plain < 1:
                return None
            return _finalize(ctx, split, words, answer, pieces, ac, used, used_sel,
                             is_link, has)
        # plain charade piece
        for r in runs:
            if not disjoint(used, r):
                continue
            for v in values(r):
                if answer.startswith(v, pos):
                    res = dfs(pos + len(v), used | set(range(r[0], r[1])), used_sel,
                              n_plain + 1, ac, pieces + [("plain", r, v, None)])
                    if res:
                        return res
        # the single anagram-container span
        if ac is None:
            for M in range(4, N - pos + 1):
                span = answer[pos:pos + M]
                for acp in ac_for_span(span, used):
                    nsel = used_sel
                    if acp["sel"] is not None:
                        nsel = used_sel + [acp["sel"]]
                    res = dfs(pos + M, used | acp["used"], nsel, n_plain, acp,
                              pieces + [("ac", pos, M, acp)])
                    if res:
                        return res
        return None

    return dfs(0, set(), [], 0, None, [])


def _finalize(ctx, split, words, answer, pieces, ac, used, used_sel, is_link, has):
    n = len(words)
    # The anagram + container indicators (distinct) must be in the residue, plus any
    # selection indicator used by the value side; the rest must be DB link words.
    sel_words = set()
    for _rule, idxs in used_sel:
        sel_words.update(idxs)
    if sel_words & used:
        return None
    residue = [k for k in range(n) if k not in used and k not in sel_words]
    con = [k for k in residue if has(words[k].text, "container")
           or has(words[k].text, "insertion")]
    ana = [k for k in residue if has(words[k].text, "anagram")]
    for c in con:
        for a in ana:
            if a == c:
                continue
            spoken = {c, a}
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
            parse = _build(ctx, split, words, answer, pieces, ac, c, a,
                           used_sel, links)
            if parse is not None:
                return parse
    return None


def _build(ctx, split, words, answer, pieces, ac, con_idx, ana_idx, used_sel, links):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, links_out = [], []

    pos = 0                                          # running answer offset (0-based)
    for piece in pieces:
        if piece[0] == "plain":
            (a, b), v = piece[1], piece[2]
            toks = words[a:b]
            si = len(sources)
            sources.append(Source(
                clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                text=" ".join(t.text for t in toks), value=v, mechanism="synonym"))
            for ci in range(len(v)):
                links_out.append(Link(answer_pos=pos + ci + 1, source_index=si,
                                      operation="charade"))
            pos += len(v)
        else:  # ('ac', start, M, acp)
            start, M, acp = piece[1], piece[2], piece[3]
            _emit_ac(sources, links_out, words, answer, start, M, acp)
            pos = start + M

    annotations = [
        Annotation(clue_atom_ids=words[con_idx].atom_ids, text=words[con_idx].text,
                   role="indicator", note="container indicator"),
        Annotation(clue_atom_ids=words[ana_idx].atom_ids, text=words[ana_idx].text,
                   role="indicator", note="anagram indicator"),
    ]
    for rule, idxs in used_sel:
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in idxs for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in idxs), role="indicator",
            note="selection indicator (%s)" % rule))
    for k in links:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link", note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links_out, annotations=annotations,
                  definition=definition, operation="anagram_container",
                  solved_by="catalog")
    _verify(ctx, parse)
    return parse


def _emit_ac(sources, links_out, words, answer, start, M, acp):
    """Emit the anagram-container piece's sources + per-letter links over answer[start:start+M]."""
    span = answer[start:start + M]
    p, L = acp["p"], acp["L"]
    ia, ib = acp["inner"]
    oa, ob = acp["outer"]
    inner_anag = acp["inner_anag"]
    inner = span[p:p + L]
    outer = span[:p] + span[p + L:]
    inner_toks = words[ia:ib]
    outer_toks = words[oa:ob]

    outer_src = Source(
        clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in outer_toks), value=outer,
        mechanism="anagram_fodder" if not inner_anag else "synonym")
    # the inner: anagram fodder, a selection, or a DB value
    if inner_anag:
        inner_mech = "anagram_fodder"
    elif acp["val_kind"] == "sel" and acp["val_run"] == (ia, ib):
        inner_mech = "selection"
    else:
        inner_mech = "synonym"
    inner_src = Source(
        clue_atom_ids=(acp["val_word"][2] if (inner_mech == "selection")
                       else tuple(aid for t in inner_toks for aid in t.atom_ids)),
        text=" ".join(t.text for t in inner_toks), value=inner, mechanism=inner_mech)

    base = len(sources)
    if oa < ia:
        sources.extend([outer_src, inner_src]); OUT, IN = base, base + 1
    else:
        sources.extend([inner_src, outer_src]); IN, OUT = base, base + 1

    for off in range(M):
        pos = start + off + 1
        if p <= off < p + L:
            si, anag = IN, inner_anag
        else:
            si, anag = OUT, (not inner_anag)
        links_out.append(Link(answer_pos=pos, source_index=si,
                              operation="anagram_container",
                              transform="anagram_of" if anag else None))


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
