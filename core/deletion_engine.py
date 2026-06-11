"""Deletion engine — a plain deletion (the WHOLE answer is one DB value with letters
removed). EVIDENCE-DRIVEN, links classified LAST (memory: feedback-never-preassign-links).

Two structurally distinct, honestly-attributed forms (v1 found that conflating them
produced letter-correct but wrong-explanation parses — see deletion-engine-v1-finding):

  FORM A  POSITIONAL deletion — a deletion indicator names WHICH letters go: a fused word
          (beheaded/curtailed/heartless/mostly...) or a removal word + a position noun
          ("without leader"=behead, "losing heart"=heartless). answer = op(V) for a single
          DB value V of one fodder run. e.g. TAU = curtail(TAUT); HOPI = outer(CHOPIN).

  FORM B  NAMED deletion — the removed letters are NAMED by ANOTHER clue word (a DB value)
          with a generic removal word marking the cut. answer = V minus R, R = a DB value
          of a second run. e.g. LOTTO = BLOTTO("drunk") minus B("bishop"); OATH = LOATH
          minus L("large"); DELIBES = DELIBERATES minus RATE("speed").

A bare removal word with NO position noun and NO named-letter source does NOT make a
solve — that was v1's mis-attribution bug. ANSWER-DRIVEN / OOM-safe throughout: per fodder
run we enumerate its BOUNDED DB values and apply O(L) ops / removed_runs, never a product.
Definition decided upstream (def_pos). Pure and DB-decoupled. The single-piece base the
charade+deletion / container+deletion engines will build on.
"""

from core import grammar, deletion
from core.wordplay import GLUE_POS
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
MAX_RUN = 4


def _run_values(words, a, b, lookup_all):
    phrase = " ".join(words[k].text for k in range(a, b))
    out, seen = [], set()
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in seen:
            seen.add(v)
            out.append((v, mech))
    return out


_OP_NOTE = {
    "behead": "drops the first letter",
    "curtail": "drops the last letter",
    "outer": "drops both end letters",
    "heartless": "drops the central letter",
    "empty": "keeps only the first and last letters",
}


def _assemble(answer, words, postags, lookup_all, is_link, indicator_types):
    """Find the deletion (Form A or Form B) that yields the answer. Returns a placement
    or None. Longest fodder first (fewest leftovers)."""
    n = len(words)

    def residue_link(k):
        return (is_link and is_link(words[k].text))

    runs = sorted([(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)],
                  key=lambda r: -(r[1] - r[0]))

    for (a, b) in runs:
        values = _run_values(words, a, b, lookup_all)
        if not values:
            continue
        src_idx = set(range(a, b))
        residue0 = [k for k in range(n) if k not in src_idx]
        res_items = [(k, words[k].text) for k in residue0]

        # FORM A — positional indicator fixes the op
        op, ind_set = deletion.positional_op(res_items)
        if op:
            for (V, mech) in values:
                if len(V) <= len(answer):
                    continue
                if deletion.apply_op(op, V) == answer:
                    links = [k for k in residue0 if k not in ind_set]
                    if all(residue_link(k) for k in links):
                        return {"form": "A", "run": (a, b), "value": V, "mech": mech,
                                "op": op, "ind": sorted(ind_set), "links": links}

        # FORM B — removed letters NAMED by another run, generic removal word marks the cut
        rem_all = deletion.is_removal(res_items)
        if rem_all:
            for (V, mech) in values:
                if len(V) <= len(answer):
                    continue
                cut_runs = deletion.removed_runs(V, answer)
                if not cut_runs:
                    continue
                for (na, nb) in runs:
                    if set(range(na, nb)) & src_idx:
                        continue
                    nvals = {v for v, _ in _run_values(words, na, nb, lookup_all)}
                    R = next((r for r in cut_runs if r in nvals), None)
                    if R is None:
                        continue
                    used = src_idx | set(range(na, nb))
                    rind = sorted(k for k in rem_all if k not in used)
                    if not rind:
                        continue                       # need a removal word as the indicator
                    ind = set(rind)
                    links = [k for k in range(n) if k not in used and k not in ind]
                    if all(residue_link(k) for k in links):
                        return {"form": "B", "run": (a, b), "value": V, "mech": mech,
                                "removed": R, "named": (na, nb), "ind": sorted(ind),
                                "links": links}
    return None


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    a, b = pl["run"]
    toks = words[a:b]
    # value = the FULL pre-deletion string; transform 'deletion' on the links carries the
    # operation (as reversal does with 'reversed').
    fodder = Source(
        clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
        text=" ".join(t.text for t in toks), value=pl["value"], mechanism=pl["mech"])
    sources = [fodder]
    links = [Link(answer_pos=pos, source_index=0, operation="deletion",
                  clue_atom_id=None, transform="deletion")
             for pos in range(1, len(answer) + 1)]

    annotations = []
    if pl["form"] == "A":
        note = "deletion: %s (%s)" % (pl["op"], _OP_NOTE.get(pl["op"], ""))
        for k in pl["ind"]:
            annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                          text=words[k].text, role="indicator", note=note))
    else:
        na, nb = pl["named"]
        named_toks = words[na:nb]
        named_text = " ".join(t.text for t in named_toks)
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for t in named_toks for aid in t.atom_ids),
            text=named_text, role="deletion",
            note="deleted: %r → %s" % (named_text, pl["removed"])))
        for k in pl["ind"]:
            annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                          text=words[k].text, role="indicator",
                                          note="deletion indicator"))
    for k in pl["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link", note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="deletion", solved_by="catalog")
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


def solve_deletion(ctx, defines, lookup_all, is_link, indicator_types,
                   templates=None, define_fallback=None, is_dbe=None):
    """Full plain-deletion solve — evidence-driven. First clean PASS, else best parse,
    else None. (`templates` accepted for call-site compatibility, unused for now.)"""
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
        if len(words) < 2:
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
