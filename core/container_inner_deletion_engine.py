"""Container with a positional-DELETION inner — an OUTER plain DB value wrapped around an
INNER that is a single DB value with a POSITIONAL deletion applied (curtail/behead/outer/
heartless) before insertion:

  ASPIC     = AC ("Bill"/account) around SPI (SPIN "turn", "briefly" -> curtail)
              = A-SPI-C
  LIMBURGER = LIMBER ("Flexible") around URG (URGE "fancy", "for the most part" -> curtail)
              = LIMB-URG-ER

The plain container inserts a DB value as-is; container_acrostic inserts a first/last-letter
selection; container_inner_charade inserts a charade. None of them deletes letters from the
inner value before inserting it. container_deletion is the OTHER order — it deletes from the
WHOLE assembled container, not from the inner piece — so it cannot reach these (LIMBER ∋ URGE
= LIMBURGEER, curtailed = LIMBURGEE, not LIMBURGER).

A NEW bespoke stage (never edits a working engine). It LEVERAGES the existing deletion math
(core.deletion) and the synonym lookup, adding only its own answer-driven verifier. The
deletion op is read from the deletion indicator's DB sub-type (a generic/unnamed removal
widens to the common positional ops curtail/behead, exactly as container_deletion does).

ANSWER-DRIVEN: the OUTER must be an exact plain DB value of a clue run, and the INNER answer
span must equal op(V) EXACTLY for a DB value V of a disjoint clue run with len(V) > the span
(so a real deletion happened). GATED on BOTH a container/insertion indicator AND a deletion
indicator (distinct words). True container (the inner sits strictly interior so the outer
straddles it on both sides). All remaining words must be DB links. Own _verify calls
role_validity. Returns ONLY a clean PASS, so it can never displace a simpler engine's
pending/fail. Pure and DB-decoupled.
"""

from core import deletion
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation", "raw")
MAX_RUN = 4

# Plain-English description of which letters each deletion op removes (for the explanation).
_OP_PHRASE = {"behead": "first letter", "curtail": "last letter",
              "outer": "both outer letters", "heartless": "central letter"}


def _answer(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def _run_values(words, a, b, lookup_all):
    """DB synonym/abbreviation (value, mechanism) for words[a:b] — UNFILTERED (the outer is
    split around the inner, and the inner value is PRE-deletion, so neither is a contiguous
    substring of the answer)."""
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


def _del_ops(phrase, deletion_subtypes):
    """The positional deletion ops a deletion indicator licenses, from its DB sub-types via
    SUBTYPE_OP. A generic/unnamed removal widens to the common positional ops (curtail,
    behead) — answer-driven matching decides which actually fits (same rule as
    container_deletion._del_ops)."""
    try:
        subs = set(deletion_subtypes(phrase) or ())
    except Exception:
        subs = set()
    ops, named = set(), False
    for s in subs:
        op = deletion.SUBTYPE_OP.get(s)
        if op:
            ops.add(op)
            named = True
    if not named:
        ops |= {"curtail", "behead"}
    return ops


def solve_container_inner_deletion(ctx, defines, lookup_all, is_link, indicator_types,
                                   deletion_subtypes, define_fallback=None, is_dbe=None):
    """Outer DB value wrapped around a positionally-deleted inner value. Returns ONLY a clean
    PASS, else None (abstain)."""
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
        n = len(words)
        if n < 4:                       # outer + inner + container ind + deletion ind, min
            continue
        parse = _try_split(ctx, answer, split, words, lookup_all, is_link,
                           indicator_types, deletion_subtypes)
        if parse is not None and parse.status == "pass":
            return parse
    return None


def _try_split(ctx, answer, split, words, lookup_all, is_link, indicator_types,
               deletion_subtypes):
    n, N = len(words), len(answer)
    con_run = _typed_run(words, n, {"container", "insertion"}, indicator_types, set())
    if con_run is None:
        return None
    del_run = _typed_run(words, n, {"deletion"}, indicator_types, set(con_run))
    if del_run is None:
        return None
    ind = set(con_run) | set(del_run)
    ops = _del_ops(" ".join(words[k].text for k in del_run), deletion_subtypes)
    if not ops:
        return None

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)
            if not (set(range(a, b)) & ind)]
    val_cache = {}

    def values(run):
        if run not in val_cache:
            val_cache[run] = _run_values(words, run[0], run[1], lookup_all)
        return val_cache[run]

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
                for irun in runs:
                    if set(range(*irun)) & set(range(*orun)):
                        continue                       # inner run disjoint from outer run
                    for V, vmech in values(irun):
                        if len(V) <= L:
                            continue                   # a real deletion shortens V to len L
                        op = next((o for r, o in deletion.candidates(V, ops)
                                   if r == inner), None)
                        if op is None:
                            continue
                        parse = _build(ctx, split, words, answer, orun, outer, outer_hit,
                                       irun, V, vmech, inner, p, L, op, con_run, del_run,
                                       ind, is_link)
                        if parse is not None and parse.status == "pass":
                            return parse
    return None


def _build(ctx, split, words, answer, orun, outer, outer_mech, irun, V, vmech, inner, p, L,
           op, con_run, del_run, ind, is_link):
    from core.definition_engine import dbe_annotation
    n = len(words)
    used = set(range(*orun)) | set(range(*irun)) | ind
    links_idx, ok = [], True
    for k in range(n):
        if k in used:
            continue
        if is_link and is_link(words[k].text):
            links_idx.append(k)
        else:
            ok = False
            break
    if not ok:
        return None

    outer_toks = words[orun[0]:orun[1]]
    inner_toks = words[irun[0]:irun[1]]
    outer_src = Source(
        clue_atom_ids=tuple(aid for t in outer_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in outer_toks), value=outer, mechanism=outer_mech)
    inner_src = Source(
        clue_atom_ids=tuple(aid for t in inner_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in inner_toks), value=V, mechanism=vmech)
    if orun[0] < irun[0]:
        sources, OUT, IN = [outer_src, inner_src], 0, 1
    else:
        sources, OUT, IN = [inner_src, outer_src], 1, 0

    links = []
    for pos in range(1, len(answer) + 1):
        si = IN if p < pos <= p + L else OUT
        links.append(Link(answer_pos=pos, source_index=si, operation="container",
                          clue_atom_id=None))

    where = _OP_PHRASE.get(op, op)
    del_toks = [words[k] for k in del_run]
    con_toks = [words[k] for k in con_run]
    # Colon-form note so the renderer surfaces the detail; contains "deletion" so
    # role_validity checks the indicator against the DB deletion type.
    del_note = "deletion: cut %s from %s -> %s" % (where, V, inner)
    annotations = [
        Annotation(clue_atom_ids=tuple(aid for t in con_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in con_toks), role="indicator",
                   note="container indicator"),
        Annotation(clue_atom_ids=tuple(aid for t in del_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in del_toks), role="indicator",
                   note=del_note),
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
                  definition=definition, operation="container_inner_deletion",
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
    elif missing or parse.definition is None:
        parse.status = "fail"
    else:
        parse.status = "pending"
