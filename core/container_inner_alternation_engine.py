"""Container with an ALTERNATION inner — an OUTER plain DB value wrapped around an INNER
that is the EVERY-OTHER (alternate) letters of a single clue word, licensed by an
alternation indicator:

  PRISONER = PRIER ("peeping Tom") around SON ("scor[n] at intervals" = s,o,n)
             = PRI-SON-ER

The plain container inserts a DB value as-is; container_acrostic inserts a first/last-letter
selection; container_inner_charade inserts a charade. None of them inserts an alternation
selection. This is the alternation sibling of container_inner_deletion (same outer, gating
and verifier; the inner is built by selection.select_span('alternate') instead of a deletion).

A NEW bespoke stage (never edits a working engine). ANSWER-DRIVEN: the OUTER must be an exact
plain DB value of a clue run, and the INNER answer span must equal the alternate letters of a
single disjoint clue word EXACTLY. GATED on BOTH a container/insertion indicator AND an
alternation indicator (distinct words). True container (the inner sits strictly interior so
the outer straddles it on both sides). All remaining words must be DB links. Per-letter
provenance on the inner (each inner answer letter <- the exact clue character it was taken
from). Own _verify calls role_validity. Returns ONLY a clean PASS. Pure and DB-decoupled.
"""

from core.selection import select_span, select_span_run
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation", "raw")
MAX_RUN = 4
# DB indicator types meaning "take alternate letters". WAS {alternation, alternating}
# only — which left the COMMON cues (even, evenly, odd, regularly, ... — typed
# parts/alternate or selection-licensed) invisible to this engine: the confirmed
# 2026-07-07 bug. Now unified with charade_alternation: a word/phrase is licensed if
# DB-typed one of these OR the selection layer licenses it for the 'alternate' rule.
_ALT_TYPES = {"alternation", "alternating", "alternate"}


def _alt_run(words, n, indicator_types, selection_rules, exclude):
    """Longest contiguous run (1..MAX_RUN) not overlapping `exclude` that is licensed as
    an alternate-letters indicator (type OR selection rule). Returns tuple or None."""
    best = None
    for L in range(min(MAX_RUN, n), 0, -1):
        for i in range(n - L + 1):
            idxs = tuple(range(i, i + L))
            if any(k in exclude for k in idxs):
                continue
            phrase = " ".join(words[k].text for k in idxs)
            ok = False
            try:
                ok = bool(set(indicator_types(phrase) or ()) & _ALT_TYPES)
            except Exception:
                pass
            if not ok and selection_rules is not None:
                try:
                    ok = "alternate" in (selection_rules(phrase) or ())
                except Exception:
                    pass
            if ok and (best is None or L > len(best)):
                best = idxs
    return best


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _run_values(words, a, b, lookup_all):
    """DB synonym/abbreviation (value, mechanism) for words[a:b] — UNFILTERED (the outer is
    split around the inner, so it is not a contiguous substring of the answer)."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out, seen = [], set()
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in seen:
            seen.add(v)
            out.append((v, mech))
    return out


def _typed_run(words, n, wptypes, indicator_types, exclude):
    """Longest contiguous run (1..MAX_RUN words) not overlapping `exclude` whose joined text
    the DB types as one of `wptypes`. Returns the run indices (tuple) or None."""
    best = None
    for L in range(min(MAX_RUN, n), 0, -1):
        for i in range(n - L + 1):
            idxs = tuple(range(i, i + L))
            if any(k in exclude for k in idxs):
                continue
            phrase = " ".join(words[k].text for k in idxs)
            try:
                ty = set(indicator_types(phrase) or ())
            except Exception:
                ty = set()
            if ty & wptypes:
                if best is None or L > len(best):
                    best = idxs
    return best


def solve_container_inner_alternation(ctx, defines, lookup_all, is_link, indicator_types,
                                      define_fallback=None, is_dbe=None,
                                      selection_rules=None):
    """Outer DB value wrapped around an alternation inner. Returns ONLY a clean PASS, else
    None (abstain)."""
    from core.definition_engine import find_definitions
    answer = _answer(ctx)
    if len(answer) < 4:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    from core.engine_common import better_near_miss
    best_fail = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        n = len(words)
        if n < 4:                       # outer + fodder + container ind + alternation ind, min
            continue
        parse = _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types,
                           selection_rules)
        if parse is None:
            continue
        if parse.status == "pass":
            return parse
        best_fail = better_near_miss(best_fail, parse, ctx)
    return best_fail


def _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types,
               selection_rules=None):
    n, N = len(words), len(answer)
    con_run = _typed_run(words, n, {"container", "insertion"}, indicator_types, set())
    if con_run is None:
        return None
    alt_run = _alt_run(words, n, indicator_types, selection_rules, set(con_run))
    if alt_run is None:
        return None
    ind = set(con_run) | set(alt_run)

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)
            if not (set(range(a, b)) & ind)]
    val_cache = {}

    def values(run):
        if run not in val_cache:
            val_cache[run] = _run_values(words, run[0], run[1], lookup_all)
        return val_cache[run]

    from core.engine_common import better_near_miss
    best_fail = None
    # Enumerate the insertion: inner = answer[p:p+L], outer = the rest; true container
    # (inner strictly interior so the outer straddles it on both sides).
    for p in range(1, N - 1):
        for L in range(1, N - p):
            if p + L >= N:
                continue
            inner = answer[p:p + L]
            outer = answer[:p] + answer[p + L:]
            if not outer:
                continue
            for orun in runs:
                outer_hit = next((m for v, m in values(orun) if v == outer), None)
                if outer_hit is None:
                    continue
                # the inner is the alternate letters of a clue word OR contiguous run
                # (1..3 words; was a single word only), disjoint from outer/indicators.
                for fa in range(n):
                  for fb in range(fa + 1, min(fa + 3, n) + 1):
                    frun = set(range(fa, fb))
                    if (frun & ind) or (frun & set(range(orun[0], orun[1]))):
                        break
                    sel = next(((s, aids) for s, aids
                                in select_span_run(ctx, words[fa:fb], "alternate")
                                if s.upper() == inner), None)
                    if sel is None:
                        continue
                    parse = _build(ctx, split, words, answer, orun, outer, outer_hit,
                                   (fa, fb), sel[1], inner, p, L, con_run, alt_run, ind,
                                   is_link)
                    if parse is None:
                        continue
                    if parse.status == "pass":
                        return parse
                    best_fail = better_near_miss(best_fail, parse, ctx)
    return best_fail


def _build(ctx, split, words, answer, orun, outer, outer_mech, fi, inner_atom_ids, inner,
           p, L, con_run, alt_run, ind, is_link):
    from core.definition_engine import dbe_annotation
    fa, fb = fi                                  # the fodder RUN (was one word)
    n = len(words)
    used = set(range(*orun)) | set(range(fa, fb)) | ind
    links_idx = []
    for k in range(n):
        if k in used:
            continue
        if is_link and is_link(words[k].text):
            links_idx.append(k)
        # else: non-link residue left unaccounted -> _verify marks a FAIL and names it
        # (near-miss). Cannot pass (an unaccounted word fails _verify), so pass-invariant.

    outer_toks = words[orun[0]:orun[1]]
    outer_src = Source(
        clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in outer_toks), value=outer, mechanism=outer_mech)
    inner_src = Source(
        clue_atom_ids=tuple(aid for t in words[fa:fb] for aid in t.atom_ids),
        text=" ".join(t.text for t in words[fa:fb]), value=inner, mechanism="alternate")
    if orun[0] < fa:
        sources, OUT, IN = [outer_src, inner_src], 0, 1
    else:
        sources, OUT, IN = [inner_src, outer_src], 1, 0

    links = []
    for pos in range(1, len(answer) + 1):
        if p < pos <= p + L:
            aid = inner_atom_ids[pos - 1 - p]      # the exact clue char this letter came from
            links.append(Link(answer_pos=pos, source_index=IN, operation="alternation",
                              clue_atom_id=aid))
        else:
            links.append(Link(answer_pos=pos, source_index=OUT, operation="container",
                              clue_atom_id=None))

    con_toks = [words[k] for k in con_run]
    alt_toks = [words[k] for k in alt_run]
    annotations = [
        Annotation(clue_atom_ids=tuple(aid for t in con_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in con_toks), role="indicator",
                   note="container indicator"),
        Annotation(clue_atom_ids=tuple(aid for t in alt_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in alt_toks), role="indicator",
                   note="alternation indicator (alternate letters of %s)"
                        % " ".join(t.text for t in words[fa:fb])),
    ]
    for k in links_idx:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                                      role="link", note="link word"))
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition", source=split.source)
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="container_inner_alternation",
                  solved_by="catalog")
    parse.template_id = None
    parse.matched_signature = None
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """Three-state verdict: pass / pending (provisional def) / fail (a clue word
    unaccounted, or a role the DB does not back)."""
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
    elif missing:
        parse.status = "fail"
    else:
        parse.status = "pending"
