"""Reversal of a curtailed synonym — the whole answer is ONE synonym that has letters
REMOVED (a deletion) and is REVERSED:

    "Chunk of wood cut after turning" = SLAB
      def "Chunk"; "wood" = BALSA; "cut" = deletion (curtail -> BALS);
      "after turning" = reversal -> SLAB.       reverse(curtail(BALSA)) = SLAB

A NEW bespoke stage — it never edits the working reversal or deletion engines. It LEVERAGES
the existing deletion op-math (core.deletion) and the synonym lookup, adding only its own
answer-driven verifier. ANSWER-DRIVEN: the synonym, after the deletion its indicator licenses
and a reversal, must equal the answer EXACTLY (both orders tried: delete-then-reverse and
reverse-then-delete). GATED on BOTH a reversal indicator AND a deletion indicator, plus a DB
synonym/abbreviation source, so it cannot fabricate. Single-piece (whole-answer) form: every
other wordplay word must be a link.
"""

from core import deletion
from core.wfw_model import Source, Link, Annotation, Parse

_MAX_RUN = 4

# Plain-English description of which letters each deletion op removes (for the explanation).
# Carries its own article so the note reads naturally after "cut " (no fixed "the ").
_OP_PHRASE = {"behead": "the first letter", "curtail": "the last letter",
              "outer": "both outer letters", "heartless": "the central letter",
              "empty": "all but the outer letters", "internal": "an interior letter"}


def _typed_run(words, n, wptype, indicator_types, exclude):
    """Longest contiguous run (1.._MAX_RUN words) not overlapping `exclude` whose joined text
    the DB types as `wptype`. Returns the run indices (tuple) or None."""
    best = None
    for L in range(min(_MAX_RUN, n), 0, -1):
        for i in range(n - L + 1):
            idxs = tuple(range(i, i + L))
            if any(k in exclude for k in idxs):
                continue
            phrase = " ".join(words[k].text for k in idxs)
            try:
                if wptype in (indicator_types(phrase) or ()):
                    if best is None or L > len(best):
                        best = idxs
            except Exception:
                pass
    return best


def _ops_for(deletion_subtypes, text):
    """The deletion ops a deletion indicator licenses: its DB subtypes mapped via SUBTYPE_OP.
    A generic/unspecified subtype widens to ALL ops (answer-driven decides which fits)."""
    try:
        subs = set(deletion_subtypes(text) or ())
    except Exception:
        subs = set()
    ops = {deletion.SUBTYPE_OP[s] for s in subs if s in deletion.SUBTYPE_OP}
    if not ops or (subs - set(deletion.SUBTYPE_OP)):     # generic/unknown subtype -> widen
        ops |= set(deletion._OP_FUNCS) | {"internal"}
    return ops


def _transform(v, answer, ops):
    """Does `v` become `answer` by a deletion (one of `ops`) plus a reversal, in either order?
    Returns (op, order, intermediate) or None."""
    for res, op in deletion.candidates(v, ops):          # delete THEN reverse
        if res[::-1] == answer:
            return (op, "delete_then_reverse", res)
    rv = v[::-1]                                          # reverse THEN delete
    for res, op in deletion.candidates(rv, ops):
        if res == answer:
            return (op, "reverse_then_delete", rv)
    return None


def solve_reversal_deletion(ctx, defines, lookup_all, is_link, indicator_types,
                            deletion_subtypes, define_fallback=None, is_dbe=None):
    """Solve a clue whose whole answer is one synonym, deleted and reversed. Returns the first
    clean PASS, else the best parse, else None (abstain)."""
    from core.definition_engine import find_definitions
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 2:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None

    best = None
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        n = len(words)
        if n < 3:                                        # synonym + deletion ind + reversal ind
            continue
        rev_run = _typed_run(words, n, "reversal", indicator_types, set())
        if rev_run is None:
            continue
        del_run = _typed_run(words, n, "deletion", indicator_types, set(rev_run))
        if del_run is None:
            continue
        ind = set(rev_run) | set(del_run)
        ops = _ops_for(deletion_subtypes, " ".join(words[k].text for k in del_run))

        for fi in range(n):
            if fi in ind or (is_link and is_link(words[fi].text)):
                continue
            others = [k for k in range(n) if k not in ind and k != fi]
            if any(not (is_link and is_link(words[k].text)) for k in others):
                continue                                 # whole-answer form: the rest are links
            for val, mech in lookup_all(words[fi].text):
                if mech not in ("synonym", "abbreviation"):
                    continue
                v = "".join(c for c in (val or "").upper() if c.isalpha())
                if len(v) < 2:
                    continue
                hit = _transform(v, answer, ops)
                if hit is None:
                    continue
                op, order, mid = hit
                parse = _build(ctx, split, words, fi, v, mech, del_run, rev_run,
                               others, op, order, mid, answer)
                if parse.status == "pass":
                    return parse
                if best is None:
                    best = parse
    return best


def _build(ctx, split, words, fi, value, mech, del_run, rev_run, link_idx, op, order, mid,
           answer):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition", source=split.source)
    src = Source(clue_atom_ids=words[fi].atom_ids, text=words[fi].text, value=value,
                 mechanism=mech)
    links = [Link(answer_pos=i + 1, source_index=0, operation="reversal",
                  clue_atom_id=None, transform=None) for i in range(len(answer))]

    from collections import Counter
    del_toks = [words[k] for k in del_run]
    rev_toks = [words[k] for k in rev_run]
    where = _OP_PHRASE.get(op, op)
    # Notes are in the "deletion:" / "reversal:" colon form so the renderer surfaces the
    # detail (a plain note's detail is dropped). Name the EXACT letter(s) cut and from where.
    if order == "delete_then_reverse":
        removed = "".join((Counter(value) - Counter(mid)).elements())
        del_note = "deletion: cut %s (%s) from %s → %s" % (where, removed, value, mid)
        rev_note = "reversal: %s → %s" % (mid, answer)
    else:
        rv = value[::-1]
        removed = "".join((Counter(rv) - Counter(answer)).elements())
        del_note = ("deletion: cut %s (%s) from %s (reversed) → %s"
                    % (where, removed, rv, answer))
        rev_note = "reversal: %s → %s" % (value, rv)
    annotations = [
        Annotation(clue_atom_ids=tuple(aid for t in del_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in del_toks), role="indicator", note=del_note),
        Annotation(clue_atom_ids=tuple(aid for t in rev_toks for aid in t.atom_ids),
                   text=" ".join(t.text for t in rev_toks), role="indicator", note=rev_note),
    ]
    for k in link_idx:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                                      role="link", note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=[src], links=links, annotations=annotations,
                  definition=definition, operation="reversal_deletion", solved_by="catalog")
    parse.template_id = None
    parse.matched_signature = None
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """Three-state verdict: pass / pending (provisional def) / fail (a clue word unaccounted,
    or a role the DB does not back)."""
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully covered")
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
