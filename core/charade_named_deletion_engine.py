"""Charade + named-letter deletion — a charade where ONE piece is a value with a SPECIFIC
named letter removed, the removed letter being a VERIFIED wordplay-table value of another
clue word. e.g.

    "Turn over motor, even losing volume" = CAREEN
      def "Turn over"; CAR = "motor"; EEN = "even" (the word's letters EVEN) losing V
      ("volume" = V, a wordplay-table abbreviation); "losing" = deletion indicator.
      CAR + EEN = CAREEN

Distinct from charade_deletion_engine (which removes a POSITIONAL letter — first/last/
outer/middle — reconstructed over all 26). Here the deleted letter is NAMED: it must be a
confirmed abbreviation/substitution of a clue word, so nothing is invented — every removed
letter is backed by the wordplay table, and the charade is ANSWER-DRIVEN (the pieces must
concatenate to the EXACT answer). A NEW stage (the working charade/deletion engines are
untouched). GATED on a deletion indicator. Pure and DB-decoupled.
"""

from core.wordplay import raw
from core.wfw_model import Source, Link, Annotation, Parse

MAX_RUN = 3               # a charade piece spans at most this many clue words
_MAX_DEL_LEN = 2          # a named deleted value: a letter or short abbreviation
_SUB_MECH = ("abbreviation", "substitution")
_MECH_PRI = {"literal": 0, "raw": 0, "abbreviation": 1, "synonym": 2}


def _piece_values(lookup_all, phrase):
    """(value, mechanism) for a run — its own letters (literal) first, then abbreviation,
    then synonym. The literal lets a clue word stand for its own letters (EVEN), which is
    safe here because a named deletion + exact answer reconstruction constrain it."""
    rows = [((v or "").upper(), m) for v, m in lookup_all(phrase) if v]
    lit = raw(phrase)
    if lit:
        rows.append((lit, "literal"))
    rows.sort(key=lambda x: (_MECH_PRI.get(x[1], 3), len(x[0])))
    out, seen = [], set()
    for vu, m in rows:
        if vu and vu not in seen:
            seen.add(vu)
            out.append((vu, m))
    return out


def _named_letters(word, all_values):
    """Short, VERIFIED letter values for a deletion-namer word — abbreviation/substitution
    only (never a synonym), so the deleted letter is always backed by the wordplay table."""
    own = raw(word)
    out, seen = [], set()
    for val, mech in all_values(word):
        if mech not in _SUB_MECH:
            continue
        vv = "".join(ch for ch in (val or "").upper() if ch.isalpha())
        if vv and len(vv) <= _MAX_DEL_LEN and vv != own and vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out


def solve_charade_named_deletion(ctx, defines, lookup_all, all_values, is_link,
                                 indicator_types, define_fallback=None, is_dbe=None):
    from core.definition_engine import find_definitions
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    best = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 3:                          # >=1 plain piece, the deletion piece, a namer
            continue
        parse = _assemble(ctx, answer, split, words, lookup_all, all_values, is_link,
                          indicator_types)
        if parse is not None:
            if parse.status == "pass":
                return parse
            if best is None:
                best = parse
    return best


def _assemble(ctx, answer, split, words, lookup_all, all_values, is_link, indicator_types):
    n, N = len(words), len(answer)

    def types(k):
        try:
            return indicator_types(words[k].text) or set()
        except Exception:
            return set()

    def is_del(k):
        return "deletion" in types(k)

    def is_glue(k):
        return bool((is_link and is_link(words[k].text)) or types(k))

    if not any(is_del(k) for k in range(n)):
        return None                                 # GATE: a deletion indicator is required

    # namer letters per word (verified wordplay-table values only)
    namers = {k: _named_letters(words[k].text, all_values) for k in range(n)}

    vcache, scache = {}, {}

    def ordered(a, b):
        if (a, b) not in vcache:
            vcache[(a, b)] = _piece_values(
                lookup_all, " ".join(words[k].text for k in range(a, b)))
        return vcache[(a, b)]

    def vset(a, b):
        if (a, b) not in scache:
            scache[(a, b)] = {v for v, _ in ordered(a, b)}
        return scache[(a, b)]

    def dfs(wi, pos, pieces, gaps, namer_used):
        if pos == N:
            return _finalize(ctx, answer, split, words, pieces,
                             gaps + list(range(wi, n)), namer_used, is_del, is_glue,
                             is_link)
        if wi >= n:
            return None
        # skip wi as a gap (glue / link / indicator / namer)
        r = dfs(wi + 1, pos, pieces, gaps + [wi], namer_used)
        if r is not None:
            return r
        for b in range(wi + 1, min(wi + MAX_RUN, n) + 1):
            # plain piece: a run value that is a prefix of the remaining answer
            for V, mech in ordered(wi, b):
                if V and answer.startswith(V, pos):
                    r = dfs(b, pos + len(V),
                            pieces + [(wi, b, V, "plain", mech, None)], gaps, namer_used)
                    if r is not None:
                        return r
            # named-deletion piece (only one): run value V_full = seg with a named letter
            # inserted; the named letter L is a verified value of a DISTINCT namer word.
            if namer_used is None:
                vs = vset(wi, b)
                for k in range(1, N - pos + 1):
                    seg = answer[pos:pos + k]
                    for nk in range(n):
                        if wi <= nk < b:
                            continue                # namer must not be the value run itself
                        for L in namers.get(nk, ()):
                            for ins in range(len(seg) + 1):
                                vfull = seg[:ins] + L + seg[ins:]
                                if vfull in vs:
                                    r = dfs(b, pos + k,
                                            pieces + [(wi, b, seg, "named_del", vfull,
                                                       (nk, L))], gaps, nk)
                                    if r is not None:
                                        return r
        return None

    return dfs(0, 0, [], [], None)


def _finalize(ctx, answer, split, words, pieces, gaps, namer_used, is_del, is_glue,
              is_link):
    if namer_used is None or len(pieces) < 2:
        return None
    if not any(p[3] == "named_del" for p in pieces):
        return None
    if not any(is_del(g) for g in gaps):
        return None                                 # the deletion indicator must be present
    namer_set = {namer_used}
    # every leftover word must be the namer, a deletion indicator, glue or a link
    for g in gaps:
        if g in namer_set or is_del(g) or is_glue(g):
            continue
        return None                                 # a bare content word is unaccounted

    sources, links, pos = [], [], 0
    for (a, b, seg, kind, info, namer) in pieces:
        toks = words[a:b]
        mech = "deletion" if kind == "named_del" else (info if info else "synonym")
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=seg, mechanism=mech, source="db"))
        si = len(sources) - 1
        for _ in range(len(seg)):
            links.append(Link(answer_pos=pos + 1, source_index=si,
                              operation="charade", clue_atom_id=None))
            pos += 1

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition", source=split.source)
    annotations = []
    # the deletion: name which letter was removed, from which word
    for (a, b, seg, kind, vfull, namer) in pieces:
        if kind == "named_del":
            nk, L = namer
            annotations.append(Annotation(
                clue_atom_ids=words[nk].atom_ids, text=words[nk].text, role="indicator",
                note="deleted letters: %s" % L))
    for g in gaps:
        if g == namer_used:
            continue                                # already annotated above
        if is_del(g):
            role, note = "indicator", "deletion indicator"
        elif is_link and is_link(words[g].text):
            role, note = "link", "link word"
        else:
            role, note = "indicator", "charade indicator"
        annotations.append(Annotation(clue_atom_ids=words[g].atom_ids,
                                      text=words[g].text, role=role, note=note))

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="charade_deletion",
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
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"
