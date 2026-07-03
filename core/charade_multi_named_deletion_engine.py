"""Charade + MULTI-word named-letter deletion — SIBLING of charade_named_deletion.

charade_named_deletion removes ONE named value supplied by ONE clue word (CAREEN = CAR +
EVEN["even"] losing V["volume"]). This sibling removes a string that is a CHARADE of named
letters from a CONTIGUOUS RUN of >=2 clue words. e.g.

    "Poet's slowish to drop a name" = DANTE
      def "Poet's"; slowish -> ANDANTE (a real DB synonym); "drop" = deletion indicator;
      "a name" -> A + N  (a=A, name=N, both VERIFIED abbreviations) = "AN", the deleted string;
      ANDANTE - AN = DANTE.

SAFETY (this is the indirect-deletion area that fabricated false PASSes when it deleted
arbitrary/synonym pieces — see the SHELVED charade_synonym_multi_deletion). Nothing is invented:
  - the base value (ANDANTE) must be a REAL DB value of the deleted-from run;
  - the deleted string is the EXACT concatenation of the namer run's ABBREVIATION / substitution
    values IN ORDER (never a synonym, nothing positional);
  - the namer run is >=2 CONTIGUOUS clue words, disjoint from the value run;
  - a deletion indicator must be present AND adjacent to the deletion piece;
  - every clue word is accounted (namer / deletion indicator / link / another piece);
  - answer-driven exact reconstruction.
Pure and DB-decoupled. A NEW stage — charade_named_deletion is untouched.
"""

from core.wordplay import raw
from core.wfw_model import Source, Link, Annotation, Parse

MAX_RUN = 3               # a charade piece spans at most this many clue words
MAX_NAMER = 3             # the deleted string is named by at most this many contiguous words
_MAX_DEL_TOTAL = 4        # total deleted letters (a short charade of named letters)
_MAX_NAMED_LEN = 2        # each namer word supplies a letter or short abbreviation
_SUB_MECH = ("abbreviation", "substitution")
_MECH_PRI = {"literal": 0, "raw": 0, "abbreviation": 1, "synonym": 2}


def _piece_values(lookup_all, phrase):
    """(value, mechanism) for a run — literal first, then abbreviation, then synonym."""
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
    """Short VERIFIED letter values for a namer word — abbreviation/substitution only (never a
    synonym), so every deleted letter is backed by the wordplay table. A genuine SINGLE-letter
    clue word IS that letter (a=A, I=I, o=O), which is how 'a name' -> A + N is named."""
    own = raw(word)
    out, seen = [], set()
    if own and len(own) == 1:                       # a single-letter word supplies its own letter
        out.append(own)
        seen.add(own)
    for val, mech in all_values(word):
        if mech not in _SUB_MECH:
            continue
        vv = "".join(ch for ch in (val or "").upper() if ch.isalpha())
        if vv and len(vv) <= _MAX_NAMED_LEN and vv != own and vv not in seen:
            seen.add(vv)
            out.append(vv)
    return out


def solve_charade_multi_named_deletion(ctx, defines, lookup_all, all_values, is_link,
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
        if len(words) < 3:                          # the deletion piece + >=2 namer words
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

    namers = {k: _named_letters(words[k].text, all_values) for k in range(n)}

    def named_charade_for(D, lo, hi):
        """A CONTIGUOUS run of >=2 clue words (disjoint from [lo,hi)) whose abbreviation values
        concatenate to exactly D. Returns [(word_idx, value), ...] or None."""
        def match(e, rem, acc):
            if not rem:
                return acc if len(acc) >= 2 else None
            if e >= n or (lo <= e < hi) or (e - (acc[0][0] if acc else e)) >= MAX_NAMER:
                return None
            for val in namers.get(e, ()):
                if rem.startswith(val):
                    res = match(e + 1, rem[len(val):], acc + [(e, val)])
                    if res is not None:
                        return res
            return None
        for s in range(n):
            if lo <= s < hi:
                continue
            res = match(s, D, [])
            if res is not None:
                return res
        return None

    vcache = {}

    def ordered(a, b):
        if (a, b) not in vcache:
            vcache[(a, b)] = _piece_values(
                lookup_all, " ".join(words[k].text for k in range(a, b)))
        return vcache[(a, b)]

    def dfs(wi, pos, pieces, gaps, del_used):
        if pos == N:
            return _finalize(ctx, answer, split, words, pieces,
                             gaps + list(range(wi, n)), del_used, is_del, is_glue, is_link)
        if wi >= n:
            return None
        # skip wi as a gap (namer / link / indicator)
        r = dfs(wi + 1, pos, pieces, gaps + [wi], del_used)
        if r is not None:
            return r
        for b in range(wi + 1, min(wi + MAX_RUN, n) + 1):
            # plain piece: a run value that is a prefix of the remaining answer
            for V, mech in ordered(wi, b):
                if V and answer.startswith(V, pos):
                    r = dfs(b, pos + len(V),
                            pieces + [(wi, b, V, "plain", mech, None)], gaps, del_used)
                    if r is not None:
                        return r
            # named MULTI-deletion piece (only one): the run has a REAL DB value V_full; remove a
            # CONTIGUOUS chunk D (named by a >=2-word charade) to leave the answer segment `seg`.
            if not del_used:
                for V_full, vmech in ordered(wi, b):
                    Lf = len(V_full)
                    if Lf <= N - pos:               # V_full must be LONGER than what it leaves
                        continue
                    for ins in range(Lf):
                        for dlen in range(1, min(_MAX_DEL_TOTAL, Lf - ins) + 1):
                            seg = V_full[:ins] + V_full[ins + dlen:]
                            if not seg or pos + len(seg) > N:
                                continue
                            if not answer.startswith(seg, pos):
                                continue
                            D = V_full[ins:ins + dlen]
                            nlist = named_charade_for(D, wi, b)
                            if nlist is None:
                                continue
                            r = dfs(b, pos + len(seg),
                                    pieces + [(wi, b, seg, "named_del", (V_full, vmech, D), nlist)],
                                    gaps, True)
                            if r is not None:
                                return r
        return None

    return dfs(0, 0, [], [], False)


def _finalize(ctx, answer, split, words, pieces, gaps, del_used, is_del, is_glue, is_link):
    if not del_used:
        return None
    dels = [p for p in pieces if p[3] == "named_del"]
    if len(dels) != 1:
        return None
    if not any(is_del(g) for g in gaps):
        return None                                 # the deletion indicator must be present

    namer_idx = {k for (_a, _b, _seg, _k, _info, nlist) in dels for (k, _v) in nlist}

    # every leftover word must be a namer, a deletion indicator, a link, or an indicator glue
    for g in gaps:
        if g in namer_idx or is_del(g) or is_glue(g):
            continue
        return None

    # ADJACENCY: the deletion indicator must attach to the deletion piece — walk out from the
    # piece run through deletion-EXPRESSION words only (links, namer words, deletion indicators)
    # and reach a deletion indicator. A stray content word blocks it.
    covered = {w for p in pieces for w in range(p[0], p[1])}
    dp = dels[0]

    def expr(w):
        return (w not in covered) and (is_del(w) or (w in namer_idx)
                                       or bool(is_link and is_link(words[w].text)))
    bound = False
    j = dp[1]
    while j < len(words) and expr(j):
        if is_del(j):
            bound = True
            break
        j += 1
    if not bound:
        j = dp[0] - 1
        while j >= 0 and expr(j):
            if is_del(j):
                bound = True
                break
            j -= 1
    if not bound:
        return None

    sources, links, pos = [], [], 0
    for (a, b, seg, kind, info, nlist) in pieces:
        toks = words[a:b]
        if kind == "named_del":
            V_full, vmech, D = info
            value, mech = V_full, vmech          # store the BASE value so the render shows -D
        else:
            value, mech = seg, (info if info else "synonym")
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=value, mechanism=mech, source="db"))
        si = len(sources) - 1
        for _ in range(len(seg)):                 # links cover the SURVIVING letters (seg)
            links.append(Link(answer_pos=pos + 1, source_index=si, operation="charade",
                              clue_atom_id=None))
            pos += 1

    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition", source=split.source)
    annotations = []
    # name each removed letter with the word that supplies it (a -> A, name -> N), role="deletion"
    # (as the plain named-deletion engine does — labelling it "indicator" made role_validity reject).
    for (a, b, seg, kind, info, nlist) in pieces:
        if kind == "named_del":
            for (k, val) in nlist:
                annotations.append(Annotation(
                    clue_atom_ids=words[k].atom_ids, text=words[k].text, role="deletion",
                    note="deleted letters: %s" % val))
    for g in gaps:
        if g in namer_idx:
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
