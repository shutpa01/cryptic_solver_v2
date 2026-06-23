"""Reversed-outer container — a container whose OUTER is a reversed synonym, wrapped
around an inner that may itself be a charade (Times 31273 EMPEROR).

  EMPEROR = EOR (caviar->ROE, "flipping" = reversed) around MPER (politician->MP +
            monarch->ER, "tucked into") = E-MPER-OR, def "Ruler".

The plain container inserts a value as-is; reversal_container reverses the INNER before
inserting (ARABS = AS around rev(BAR)); container_inner_charade allows a charade inner but
a PLAIN outer. None reverses the OUTER. This engine does exactly that: the wrapping value is
a DB synonym/abbreviation REVERSED, and the inner is a charade of >=1 DB-value pieces.

ANSWER-DRIVEN: enumerate the insertion (inner = answer[p:p+L], outer = the rest); the OUTER
is valid only if reverse(outer) is a DB value of a clue run, and the inner must tile exactly
into DB-value pieces from OTHER clue runs. Gated on BOTH a reversal indicator AND a container
indicator (distinct words). Returns ONLY a clean PASS (conservative compound — never displaces
a simpler engine's pending/fail). _verify calls role_validity. Definition decided upstream.
"""

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


def solve_reversed_outer_container(ctx, defines, lookup_all, is_link, indicator_types,
                                   define_fallback=None, is_dbe=None):
    """Container with a reversed-synonym outer + charade inner. Returns ONLY a clean PASS."""
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

    def has(text, kind):
        try:
            return kind in (indicator_types(text) or set())
        except Exception:
            return False

    def is_rev(k):
        return has(words[k].text, "reversal")

    def is_con(k):
        return has(words[k].text, "container") or has(words[k].text, "insertion")

    if not any(is_rev(k) for k in range(n)) or not any(is_con(k) for k in range(n)):
        return None

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    run_vals = {r: _run_values(words, r[0], r[1], lookup_all) for r in runs}
    val_runs = {}                                  # value string -> [runs producing it]
    for r, vals in run_vals.items():
        for v in vals:
            val_runs.setdefault(v, []).append(r)

    def tile_inner(target, blocked):
        """Tilings of `target` by DB-value pieces from runs disjoint from `blocked` and
        each other (>=1 piece, left-to-right). Returns a list of [(run, value), ...]."""
        out = []

        def dfs(pos, used, pieces):
            if pos == len(target):
                if pieces:
                    out.append(list(pieces))
                return
            for r in runs:
                rs = set(range(r[0], r[1]))
                if rs & blocked or rs & used:
                    continue
                for v in run_vals[r]:
                    if target.startswith(v, pos):
                        dfs(pos + len(v), used | rs, pieces + [(r, v)])

        dfs(0, set(), [])
        return out

    # Enumerate the insertion: inner sits strictly inside the outer (both flanks non-empty).
    for p in range(1, N - 1):
        for L in range(1, N - p):
            inner = answer[p:p + L]
            outer = answer[:p] + answer[p + L:]
            if not inner or not outer:
                continue
            rev_outer = outer[::-1]
            if rev_outer == outer:
                continue                           # palindromic 'reversal' is a no-op
            for o in val_runs.get(rev_outer, ()):  # reverse(outer) is a DB value
                o_set = set(range(o[0], o[1]))
                for piece_runs in tile_inner(inner, o_set):
                    parse = _assemble(ctx, split, words, answer, p, L, o, rev_outer,
                                      piece_runs, indicator_types, is_link)
                    if parse is not None and parse.status == "pass":
                        return parse
    return None


def _con_run(words, positions, indicator_types):
    """Longest contiguous container/insertion indicator run within `positions`
    (multi-word aware: 'tucked into' is one insertion indicator). Returns list or None."""
    from core.engine_common import find_typed_run
    best = None
    for ty in ("insertion", "container"):
        r = find_typed_run(words, positions, indicator_types, ty, min_length=1)
        if r is not None and (best is None or len(r) > len(best)):
            best = r
    return best


def _assemble(ctx, split, words, answer, p, L, o_run, outer_value, piece_runs,
              indicator_types, is_link):
    from core.engine_common import find_typed_run
    n = len(words)
    used = set(range(o_run[0], o_run[1]))
    for (r, _v) in piece_runs:
        used |= set(range(r[0], r[1]))
    residue = set(k for k in range(n) if k not in used)
    # the reversal and container indicators may each span several words
    rev_pos = find_typed_run(words, residue, indicator_types, "reversal", min_length=1)
    if rev_pos is None:
        return None
    con_pos = _con_run(words, residue - set(rev_pos), indicator_types)
    if con_pos is None:
        return None
    spoken = set(rev_pos) | set(con_pos)
    links = []
    for k in sorted(residue):
        if k in spoken:
            continue
        if is_link and is_link(words[k].text):
            links.append(k)
        else:
            return None
    return _build(ctx, split, words, answer, p, L, o_run, outer_value,
                  piece_runs, rev_pos, con_pos, links)


def _build(ctx, split, words, answer, p, L, o_run, outer_value, piece_runs, rev_pos,
           con_pos, links):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    N = len(answer)
    # outer source (the reversed synonym): covers the answer positions OUTSIDE [p, p+L)
    o_toks = words[o_run[0]:o_run[1]]
    outer_src = Source(clue_atom_ids=tuple(aid for t in o_toks for aid in t.atom_ids),
                       text=" ".join(t.text for t in o_toks), value=outer_value,
                       mechanism="synonym")
    sources = [outer_src]
    OUT = 0
    # inner charade pieces, in answer order (they tile answer[p:p+L] left-to-right)
    inner_src_index = {}
    for i, (r, v) in enumerate(piece_runs):
        toks = words[r[0]:r[1]]
        inner_src_index[i] = len(sources)
        sources.append(Source(clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                              text=" ".join(t.text for t in toks), value=v,
                              mechanism="synonym"))

    links_out = []
    # map answer positions: outside [p,p+L) -> OUTER (reversed); inside -> the inner pieces
    inner_pos = p
    piece_spans = []                               # (start, end, src_index) within answer
    cur = p
    for i, (r, v) in enumerate(piece_runs):
        piece_spans.append((cur, cur + len(v), inner_src_index[i]))
        cur += len(v)
    for pos in range(N):
        if p <= pos < p + L:
            si = next(s for (a, b, s) in piece_spans if a <= pos < b)
            links_out.append(Link(answer_pos=pos + 1, source_index=si,
                                  operation="container"))
        else:
            links_out.append(Link(answer_pos=pos + 1, source_index=OUT,
                                  operation="container", transform="reversed"))

    rev_toks = [words[k] for k in rev_pos]
    con_toks = [words[k] for k in con_pos]
    annotations = [
        Annotation(clue_atom_ids=tuple(aid for t in rev_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in rev_toks),
                   role="indicator", note="reversal indicator"),
        Annotation(clue_atom_ids=tuple(aid for t in con_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in con_toks),
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
