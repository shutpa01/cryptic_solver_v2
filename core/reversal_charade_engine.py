"""Reversal+charade engine — a charade where the reversal applies.

EVIDENCE-DRIVEN, links classified LAST (memory: feedback-never-preassign-links). The answer
is tiled left-to-right by charade value pieces drawn from ANY unused clue word-run (a
reversal, like a container, can break strict clue order), where EACH piece is placed either
FORWARD (its DB value sits at the current position) or REVERSED (its DB value, reversed, sits
there). At least ONE piece must be reversed — otherwise it is a plain charade, not this.

This single order-free tiler covers both sub-forms the signature never encodes:
  (b) one piece reversed:   AFAR  = A + rev(RAF "service")
  (a) whole charade reversed: ERATO = rev(ARE "are") + rev(OT "old books")  (clue order flips)

ANSWER-DRIVEN: a reversed piece is found by testing answer.startswith(value[::-1], pos) over a
run's DB values — bounded by the value count, never a product, so none of the container
family's memory cost. Whatever words are left must contain a reversal indicator; the rest are
links; anything else leaves the parse unaccounted. Gated: requires a reversal indicator.
Definition decided upstream. Pure and DB-decoupled. Built on the plain reversal primitive
(core.reversal_engine) extended with a charade tail, mirroring container_charade_engine.
"""

from core import grammar
from core.wordplay import GLUE_POS
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
    for t in words:
        try:
            ty = indicator_types(t.text) or set()
        except Exception:
            ty = set()
        if "reversal" in ty:
            return True
    return False


def _assemble(answer, words, postags, lookup_all, is_link, indicator_types):
    """Tile the answer with forward/reversed charade pieces (>=1 reversed) + a reversal
    indicator in the residue. Returns a placement or None."""
    n, N = len(words), len(answer)

    def is_rev(k):
        try:
            ty = indicator_types(words[k].text) or set()
        except Exception:
            ty = set()
        return "reversal" in ty

    def residue_link(k):
        return (is_link and is_link(words[k].text)) or (postags[k] in GLUE_POS)

    all_runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    val_cache = {}

    def values(run):
        if run not in val_cache:
            val_cache[run] = _run_values(words, run[0], run[1], lookup_all)
        return val_cache[run]

    def avail(used):
        return [r for r in all_runs if not (set(range(r[0], r[1])) & used)]

    def finalize(pieces, used, nrev):
        if nrev < 1:
            return None
        residue = [k for k in range(n) if k not in used]
        rev = [k for k in residue if is_rev(k)]
        if not rev:
            return None
        c = rev[0]
        links = []
        for k in residue:
            if k == c:
                continue
            if residue_link(k):
                links.append(k)
            else:
                return None
        return {"pieces": pieces, "rev": [c], "links": links}

    def dfs(pos, used, pieces, nrev):
        if pos == N:
            return finalize(pieces, used, nrev)
        for r in avail(used):
            run_set = set(range(r[0], r[1]))
            for v in values(r):
                if answer.startswith(v, pos):                     # forward piece
                    res = dfs(pos + len(v), used | run_set,
                              pieces + [("fwd", r, v)], nrev)
                    if res:
                        return res
                rv = v[::-1]
                if rv != v and answer.startswith(rv, pos):        # reversed piece
                    res = dfs(pos + len(rv), used | run_set,
                              pieces + [("rev", r, v)], nrev + 1)
                    if res:
                        return res
        return None

    return dfs(0, set(), [], 0)


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, links = [], []
    pos = 0
    for kind, (a, b), v in pl["pieces"]:
        toks = words[a:b]
        si = len(sources)
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=v, mechanism="synonym"))
        transform = "reversed" if kind == "rev" else None
        for _ in range(len(v)):
            pos += 1
            links.append(Link(answer_pos=pos, source_index=si,
                              operation="reversal_charade", clue_atom_id=None,
                              transform=transform))

    annotations = []
    for k in pl["rev"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="indicator",
                                      note="reversal indicator"))
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


def solve_reversal_charade(ctx, defines, lookup_all, is_link, indicator_types,
                           templates=None, define_fallback=None, is_dbe=None):
    """Full reversal+charade solve — evidence-driven. Gated on a reversal indicator in the
    wordplay. First clean PASS, else best parse, else None."""
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 4:
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
        postags = grammar.wordplay_pos_tags(ctx, words)
        pl = _assemble(answer, words, postags, lookup_all, is_link, indicator_types)
        if pl is None:
            continue
        parse = _build(ctx, split, words, answer, pl)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best
