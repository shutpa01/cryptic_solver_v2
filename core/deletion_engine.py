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


def _contiguous_groups(idxs):
    """Maximal runs of consecutive integers within the sorted index list."""
    groups, cur = [], []
    for k in idxs:
        if cur and k == cur[-1] + 1:
            cur.append(k)
        else:
            if cur:
                groups.append(cur)
            cur = [k]
    if cur:
        groups.append(cur)
    return groups


def _indicator_runs(words, residue_idx, del_subtypes):
    """Every DB-typed deletion indicator run among the residue — ALL contiguous sub-runs
    (<= MAX_RUN words), not just the longest, so a specific op carried by a component word
    is never masked by a longer phrase the DB types generically ("without leader" is typed
    generic, but its "leader" still pins behead). Returns a list of (seg, spec_ops,
    is_generic):
      seg        = the run's word indices.
      spec_ops   = canonical ops the run names (DB subtype in SUBTYPE_OP) — a co-present
                   generic subtype does NOT suppress them ('mostly' = {general, tail} still
                   pins curtail).
      is_generic = the run carries a generic subtype (general/removal/deletion/NULL): a
                   plain removal whose removed letters are NAMED by another word (Form B).
    No hardcoded vocabulary — the words and their sub-typing live entirely in the DB."""
    out = []
    for group in _contiguous_groups(sorted(residue_idx)):
        L = len(group)
        for length in range(1, min(MAX_RUN, L) + 1):
            for s in range(0, L - length + 1):
                seg = group[s:s + length]
                subs = del_subtypes(" ".join(words[k].text for k in seg))
                if not subs:
                    continue
                spec = {deletion.SUBTYPE_OP[x] for x in subs if x in deletion.SUBTYPE_OP}
                generic = any(x not in deletion.SUBTYPE_OP for x in subs)
                out.append((seg, spec, generic))
    return out


def _assemble(answer, words, postags, lookup_all, is_link, del_subtypes):
    """Find the deletion (Form A or Form B) that yields the answer. Returns a placement
    or None. Longest fodder first (fewest leftovers). The deletion indicator and which
    letters it removes come from the DB (see _indicator_runs)."""
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
        ind_runs = _indicator_runs(words, residue0, del_subtypes)
        ind_all = {k for seg, _, _ in ind_runs for k in seg}

        # FORM A — a positional indicator pins the op. The WHOLE deletion expression (every
        # deletion-typed residue word, e.g. "without"+"leader") is the indicator; the rest
        # must be links. Answer-driven among the pinned ops.
        pinned = {op for _, spec, _ in ind_runs for op in spec}
        ops = set(pinned)
        if {"behead", "curtail"} <= ops:                # two ends named -> drop both
            ops.add("outer")
        if ops:
            links = [k for k in residue0 if k not in ind_all]
            if all(residue_link(k) for k in links):
                for (V, mech) in values:
                    if len(V) <= len(answer):
                        continue
                    for op in ops:
                        if deletion.apply_op(op, V) == answer:
                            return {"form": "A", "run": (a, b), "value": V, "mech": mech,
                                    "op": op, "ind": sorted(ind_all), "links": links}

        # FORM B — removed letters NAMED by another run; a GENERIC removal run marks the cut.
        # The named source may itself be a deletion-typed word (it is used here as a value,
        # e.g. "right" -> R), so only the chosen removal run is reserved as the indicator.
        for seg, _, is_generic in ind_runs:
            if not is_generic:
                continue
            rem_idx = set(seg)
            for (V, mech) in values:
                if len(V) <= len(answer):
                    continue
                cut_runs = deletion.removed_runs(V, answer)
                if not cut_runs:
                    continue
                for (na, nb) in runs:
                    nset = set(range(na, nb))
                    if nset & src_idx or nset & rem_idx:
                        continue
                    nvals = {v for v, _ in _run_values(words, na, nb, lookup_all)}
                    R = next((r for r in cut_runs if r in nvals), None)
                    if R is None:
                        continue
                    used = src_idx | nset | rem_idx
                    links = [k for k in range(n) if k not in used]
                    if all(residue_link(k) for k in links):
                        return {"form": "B", "run": (a, b), "value": V, "mech": mech,
                                "removed": R, "named": (na, nb), "ind": sorted(rem_idx),
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


def solve_deletion(ctx, defines, lookup_all, is_link, del_subtypes,
                   templates=None, define_fallback=None, is_dbe=None):
    """Full plain-deletion solve — evidence-driven. First clean PASS, else best parse,
    else None. `del_subtypes(phrase)` returns the DB deletion subtypes for a phrase (the
    indicator vocabulary, formerly hardcoded, now read from the DB). (`templates` accepted
    for call-site compatibility, unused for now.)"""
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
        pl = _assemble(answer, words, postags, lookup_all, is_link, del_subtypes)
        if pl is None:
            continue
        parse = _build(ctx, split, words, answer, pl)
        if parse.status == "pass":
            return parse
        if best is None:
            best = parse
    return best
