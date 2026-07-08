"""Container-with-charade-OUTER engine — an OUTER that is itself a CHARADE of two or more
DB values, wrapped around a single INNER value.

The exact MIRROR of container_inner_charade (outer single value around a charade inner).
The plain container inserts one value into one outer; container_inner_charade lets the
INNER be a charade; this one lets the OUTER be a charade:

  DUB = (D + B) around U
    "Germany" = D   "Britain" = B   (the two-piece outer charade, in clue order)
    "university" = U (inner)         "in" = insertion indicator   "Call" = definition

The two outer pieces flank the inner (D before it, B after it). They are read in CLUE
order as a charade and may have a link word between them ("and"); the inner word may sit
anywhere in the clue. ANSWER-DRIVEN: the assembled outer-charade with the inner inserted
must equal the answer exactly. Gated on a container/insertion indicator, links classified
LAST. A NEW isolated stage (never edit a working engine); pure and DB-decoupled.
"""

from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation", "raw")
MAX_RUN = 5
MIN_OUTER_PIECES = 2          # 1 outer piece is already the plain container's job


def _run_values(words, a, b, lookup_all):
    """DB synonym/abbreviation values for words[a:b] — UNFILTERED (the outer pieces are
    matched by exact prefix against the outer string, so no answer-substring filter)."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out = []
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in out:
            out.append(v)
    return out


def _tile_outer(words, target, values, exclude, start_wi):
    """Tile `target` as a CHARADE of clue-word-run values: pieces in increasing clue order
    (gaps between runs allowed — those words are links/indicator), each run up to MAX_RUN
    words, none overlapping `exclude` (the inner run). Returns [(a, b, value), ...] whose
    values concatenate to `target`, or None."""
    if target == "":
        return []
    n = len(words)
    for a in range(start_wi, n):
        if a in exclude:
            continue
        for b in range(a + 1, min(a + MAX_RUN, n) + 1):
            if any(k in exclude for k in range(a, b)):
                continue
            for val in values(a, b):
                if target.startswith(val):
                    rest = _tile_outer(words, target[len(val):], values, exclude, b)
                    if rest is not None:
                        return [(a, b, val)] + rest
    return None


def _assemble(answer, words, lookup_all, is_link, indicator_types):
    """Find a single inner value inserted into a charade outer. Returns a placement or None."""
    n, N = len(words), len(answer)

    def is_con(k):
        try:
            ty = indicator_types(words[k].text) or set()
        except Exception:
            ty = set()
        return "container" in ty or "insertion" in ty

    def residue_link(k):
        return bool(is_link and is_link(words[k].text))

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    val_cache = {}

    def values(a, b):
        if (a, b) not in val_cache:
            val_cache[(a, b)] = _run_values(words, a, b, lookup_all)
        return val_cache[(a, b)]

    # inner = a single DB value from one clue-word run, placed STRICTLY interior so the
    # outer straddles it on both sides (a true container, not juxtaposition).
    for (ia, ib) in runs:
        for vin in values(ia, ib):
            L = len(vin)
            for p in range(1, N - L):                 # p>=1 and p+L<=N-1 -> interior
                if answer[p:p + L] != vin:
                    continue
                outer_str = answer[:p] + answer[p + L:]
                pieces = _tile_outer(words, outer_str, values, set(range(ia, ib)), 0)
                if pieces is None or len(pieces) < MIN_OUTER_PIECES:
                    continue
                used = set(range(ia, ib))
                used |= {k for (a, b, _) in pieces for k in range(a, b)}
                residue = [k for k in range(n) if k not in used]
                # PHRASE-AWARE residue split (was per-word is_con + links).
                from core.engine_common import indicator_plus_links
                sp = indicator_plus_links(words, residue, indicator_types,
                                          ("container", "insertion"), is_link)
                if sp is not None:
                    return {"p": p, "L": L, "inner": (ia, ib), "inner_val": vin,
                            "pieces": pieces, "con": sp[0], "links": sp[1]}
    return None


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    p, L = pl["p"], pl["L"]
    ia, ib = pl["inner"]
    inner_toks = words[ia:ib]

    # source 0 = inner; sources 1.. = outer charade pieces in clue order.
    inner_src = Source(
        clue_atom_ids=tuple(aid for t in inner_toks for aid in t.atom_ids),
        text=" ".join(t.text for t in inner_toks), value=pl["inner_val"],
        mechanism="synonym")
    sources = [inner_src]
    outer_spans = []          # (start_offset_within_outer_str, length, source_index)
    off = 0
    for (a, b, val) in pl["pieces"]:
        toks = words[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=val, mechanism="synonym"))
        outer_spans.append((off, len(val), len(sources) - 1))
        off += len(val)

    # links: inner region (p<pos<=p+L) -> source 0; else map the answer letter back to its
    # offset in outer_str (before = [0,p), after = [p, len(outer_str))) and the piece there.
    links = []
    for pos in range(1, len(answer) + 1):
        if p < pos <= p + L:
            si = 0
        else:
            idx0 = pos - 1
            outer_off = idx0 if idx0 < p else idx0 - L
            si = next(s for (start, length, s) in outer_spans
                      if start <= outer_off < start + length)
        links.append(Link(answer_pos=pos, source_index=si,
                          operation="container", clue_atom_id=None))

    annotations = []
    from core.engine_common import indicator_annotations
    annotations.extend(indicator_annotations(words, pl["con"], "container indicator"))
    for k in pl["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="container", solved_by="catalog")
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


def solve_container_outer_charade(ctx, defines, lookup_all, is_link, indicator_types,
                                  define_fallback=None, is_dbe=None):
    """Charade outer (2+ DB values) wrapped around a single inner value. First clean PASS,
    else best parse, else None — same contract as the other container engines."""
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
        if len(words) < 3:                 # inner + >=2 outer pieces + indicator
            continue
        pl = _assemble(answer, words, lookup_all, is_link, indicator_types)
        if pl is None:
            continue
        parse = _build(ctx, split, words, answer, pl)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best
