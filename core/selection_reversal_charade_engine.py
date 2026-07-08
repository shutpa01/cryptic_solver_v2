"""Selection+reversal charade — a charade mixing a SELECTION piece and a REVERSAL
piece (DT 31272 TRAIL).

  TRAIL = T (end of "accoun·t", last-letter selection) + RAIL (reverse of LIAR
  "storyteller", "returning").

charade_signature builds SEL_F selection pieces and reversal_charade builds REV_F
reversed pieces, but NEITHER combines a selection AND a reversal in one charade.
This engine does, and ONLY that: it requires AT LEAST ONE selection piece AND AT
LEAST ONE reversal piece, so it can never intercept a plain charade, a plain
reversal_charade, or a selection-only charade — each of which the earlier engines
already claim. It is wired AFTER the reversal family for the same reason.

EVIDENCE-DRIVEN and ANSWER-DRIVEN, modelled on reversal_charade_engine._assemble:
the answer is tiled left-to-right by pieces, each one of
  - FORWARD   — a DB synonym/abbreviation value of a word-run, placed as-is;
  - REVERSED  — a DB value reversed (≥1 required; a reversal indicator must sit in
                the residue, exactly as in reversal_charade_engine);
  - SELECTION — letters taken from ONE word by a named rule (first/last/outer/...),
                ≥1 required, licensed by a DB selection indicator (core.selection +
                core.selection_indicators). The indicator words are accounted as
                indicators, never as links.
Residue words that are not the used reversal/selection indicators must be DB link
words; anything else leaves the parse unaccounted and the placement is rejected.
_verify calls role_validity so every recorded indicator/link role is DB-backed
(never assigned by elimination). Definition decided upstream. Pure, DB-decoupled.
"""

from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
MAX_RUN = 4


def _run_values(words, a, b, lookup_all):
    phrase = " ".join(words[k].text for k in range(a, b))
    out = []
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in out:
            out.append(v)
    return out


def _has_reversal_indicator(words, indicator_types):
    # PHRASE-AWARE gate (was per-word): a multi-word DB indicator must open it too.
    from core.engine_common import has_typed_indicator
    return has_typed_indicator(words, indicator_types, "reversal")


def _assemble(ctx, answer, words, lookup_all, is_link, indicator_types, sel_options):
    """Tile the answer with forward / reversed / selection pieces (>=1 reversed AND
    >=1 selection), the used reversal/selection indicators + DB links filling the
    residue. Returns a placement dict or None.

    sel_options: [(rule, idx_tuple), ...] — the DB selection indicators present in
    `words`, each licensing one selection rule over the word indices it occupies.
    """
    from core.selection import select_span
    n, N = len(words), len(answer)

    def is_rev(k):
        try:
            return "reversal" in (indicator_types(words[k].text) or set())
        except Exception:
            return False

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    all_runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    val_cache = {}

    def values(run):
        if run not in val_cache:
            val_cache[run] = _run_values(words, run[0], run[1], lookup_all)
        return val_cache[run]

    def finalize(pieces, used, used_sel, nrev, nsel):
        if nrev < 1 or nsel < 1:
            return None
        # The licensing selection-indicator words must be accounted, and may not also
        # have been consumed as a piece word (a word is ONE thing).
        sel_ind_words = set()
        for _rule, idxs in used_sel:
            sel_ind_words.update(idxs)
        if sel_ind_words & used:
            return None
        residue = [k for k in range(n) if k not in used and k not in sel_ind_words]
        # PHRASE-AWARE residue split (was: one rev-typed WORD + links, which stranded
        # the other half of a two-word indicator and rejected a correct parse).
        from core.engine_common import indicator_plus_links
        split = indicator_plus_links(words, residue, indicator_types, "reversal",
                                     is_link)
        if split is None:
            return None                          # a content word unaccounted -> reject
        return {"pieces": pieces, "rev": split[0],
                "sel_ind": sorted(used_sel, key=lambda ri: ri[1]),
                "links": sorted(split[1])}

    def dfs(pos, used, used_sel, pieces, nrev, nsel):
        if pos == N:
            return finalize(pieces, used, used_sel, nrev, nsel)

        # forward / reversed pieces from any unused word-run
        for r in all_runs:
            run_set = frozenset(range(r[0], r[1]))
            if run_set & used:
                continue
            for v in values(r):
                if answer.startswith(v, pos):                       # forward
                    res = dfs(pos + len(v), used | run_set, used_sel,
                              pieces + [("fwd", r, v, None)], nrev, nsel)
                    if res:
                        return res
                rv = v[::-1]
                if rv != v and answer.startswith(rv, pos):          # reversed (no-op if palindrome)
                    res = dfs(pos + len(rv), used | run_set, used_sel,
                              pieces + [("rev", r, v, None)], nrev + 1, nsel)
                    if res:
                        return res

        # selection piece: a word OR contiguous run (was one word only), a rule
        # licensed by a selection indicator. WIDTH-FIRST: every single-word option is
        # tried before ANY multi-word run — the exact old exploration order — so the
        # widening can only add tilings after the old ones, never displace the tiling
        # the old search found first (IMPERIAL/DECREE regressed on that displacement).
        from core.selection import select_span_run
        for width in (1, 2, 3):
            for rule, idxs in sel_options:
                idx_set = frozenset(idxs)
                for j in range(n - width + 1):
                    run = frozenset(range(j, j + width))
                    if (run & used) or (run & idx_set):
                        continue
                    for s, atom_ids in select_span_run(ctx, words[j:j + width], rule):
                        if s and answer.startswith(s, pos):
                            res = dfs(pos + len(s), used | run,
                                      used_sel + [(rule, tuple(idxs))],
                                      pieces + [("sel", (j, j + width), s,
                                                 {"atom_ids": atom_ids, "rule": rule})],
                                      nrev, nsel + 1)
                            if res:
                                return res
        return None

    return dfs(0, frozenset(), [], [], 0, 0)


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, links, pos = [], [], 0
    for kind, payload, value, extra in pl["pieces"]:
        si = len(sources)
        if kind == "sel":
            atom_ids = extra["atom_ids"]
            ja, jb = payload                     # the selection RUN (was a single index)
            sources.append(Source(clue_atom_ids=atom_ids,
                                  text=" ".join(t.text for t in words[ja:jb]),
                                  value=value, mechanism="selection"))
            for ci in range(len(value)):
                pos += 1
                links.append(Link(answer_pos=pos, source_index=si,
                                  operation="charade",
                                  clue_atom_id=atom_ids[ci] if ci < len(atom_ids) else None))
            continue
        a, b = payload
        toks = words[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=value, mechanism="synonym"))
        transform = "reversed" if kind == "rev" else None
        for _ in range(len(value)):
            pos += 1
            links.append(Link(answer_pos=pos, source_index=si,
                              operation="reversal_charade", clue_atom_id=None,
                              transform=transform))

    annotations = []
    # ONE annotation per contiguous indicator run (joined phrase), so role_validity
    # validates the DB row ("picked up"), never a bare component word.
    from core.engine_common import contiguous_groups
    for grp in contiguous_groups(sorted(pl["rev"])):
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in grp for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in grp), role="indicator",
            note="reversal indicator"))
    for rule, idxs in pl["sel_ind"]:
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in idxs for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in idxs),
            role="indicator", note="selection indicator (%s)" % rule))
    for k in pl["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="reversal_charade",
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


def solve_selection_reversal_charade(ctx, defines, lookup_all, is_link,
                                     indicator_types, define_fallback=None,
                                     is_dbe=None):
    """Full selection+reversal charade solve — evidence-driven. Gated on BOTH a
    reversal indicator AND a selection indicator in the wordplay; requires the
    assembled tiling to use >=1 of each. First clean PASS, else best parse, else None.
    """
    from core.definition_engine import find_definitions
    from core.selection_indicators import find_indicators

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
        if len(words) < 2 or not _has_reversal_indicator(words, indicator_types):
            continue
        sel_options = find_indicators(words)
        if not sel_options:
            continue                              # selection requires an indicator
        pl = _assemble(ctx, answer, words, lookup_all, is_link, indicator_types,
                       sel_options)
        if pl is None:
            continue
        parse = _build(ctx, split, words, answer, pl)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best
