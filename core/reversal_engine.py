"""Reversal engine — a plain reversal (the WHOLE answer is one DB value, reversed).

EVIDENCE-DRIVEN, links classified LAST (memory: feedback-never-preassign-links). The
answer = reverse(V), where V is a SINGLE DB value (synonym/abbreviation) of one contiguous
wordplay word-run, signalled by a reversal indicator. Strictly single-piece: a charade of
pieces (one or all reversed) is reversal_charade, a different engine — this one solves only
the pure case, e.g. DESSERTS = reverse(STRESSED), SMART = reverse(TRAMS).

ANSWER-DRIVEN: target = reverse(answer); the engine asks "is target a DB value of some
wordplay run?" — one lookup + a membership test, no enumeration and no product, so it has
none of the memory cost that bit the container family. Whatever clue words are left must
contain a reversal indicator (reversal); the rest are links (is_link or POS function word);
anything else leaves the parse unaccounted.

Gated: requires a reversal indicator. Definition decided upstream (def_pos). Pure and
DB-decoupled. This is the single-piece base the reversal_charade engine will build on (a
charade of reversed/with-reversal pieces).
"""

from core import grammar
from core.wordplay import GLUE_POS
from core.wfw_model import Source, Link, Annotation, Parse

_VALUE_MECH = ("synonym", "abbreviation")
MAX_RUN = 4


def _run_values(words, a, b, lookup_all):
    """DB (value, mechanism) pairs for the phrase words[a:b]."""
    phrase = " ".join(words[k].text for k in range(a, b))
    out, seen = [], set()
    for val, mech in lookup_all(phrase):
        v = (val or "").upper()
        if mech in _VALUE_MECH and v and v not in seen:
            seen.add(v)
            out.append((v, mech))
    return out


def _assemble(answer, words, postags, lookup_all, is_link, indicator_types):
    """Find the single fodder run whose DB value reverses to the answer, plus a reversal
    indicator. Returns a placement or None. Answer-driven: target = reverse(answer)."""
    n = len(words)
    target = answer[::-1]
    if target == answer:
        return None                                   # palindrome: no real reversal

    from core.engine_common import has_typed_indicator, indicator_plus_links
    if not has_typed_indicator(words, indicator_types, "reversal"):
        return None                                   # gate: need a reversal indicator

    runs = [(a, b) for a in range(n) for b in range(a + 1, min(a + MAX_RUN, n) + 1)]
    runs.sort(key=lambda r: -(r[1] - r[0]))            # longest fodder first (fewer leftovers)
    for (a, b) in runs:
        mech = None
        for v, m in _run_values(words, a, b, lookup_all):
            if v == target:
                mech = m
                break
        if mech is None:
            continue
        used = set(range(a, b))
        residue = [k for k in range(n) if k not in used]
        # PHRASE-AWARE residue split (was per-word is_rev + links, which stranded the
        # other half of a two-word indicator like "picked up" and killed the parse).
        split = indicator_plus_links(words, residue, indicator_types, "reversal", is_link)
        if split is not None:
            return {"run": (a, b), "value": target, "mech": mech,
                    "rev": split[0], "links": split[1]}
    return None


def _build(ctx, split, words, answer, pl):
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    a, b = pl["run"]
    toks = words[a:b]
    fodder = Source(
        clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
        text=" ".join(t.text for t in toks), value=pl["value"], mechanism=pl["mech"])
    sources = [fodder]

    # answer = reverse(value): every answer letter is the single fodder source, reversed.
    links = [Link(answer_pos=pos, source_index=0, operation="reversal",
                  clue_atom_id=None, transform="reversed")
             for pos in range(1, len(answer) + 1)]

    annotations = []
    # ONE annotation per contiguous indicator run, carrying the JOINED phrase — so
    # role_validity validates "picked up" (the DB row), never a bare component word.
    from core.engine_common import contiguous_groups
    for grp in contiguous_groups(sorted(pl["rev"])):
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for k in grp for aid in words[k].atom_ids),
            text=" ".join(words[k].text for k in grp), role="indicator",
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
                  definition=definition, operation="reversal", solved_by="catalog")
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


def solve_reversal(ctx, defines, lookup_all, is_link, indicator_types,
                   templates=None, define_fallback=None, is_dbe=None):
    """Full reversal solve — evidence-driven. First clean PASS, else best parse, else None.
    (`templates` accepted for call-site compatibility, unused for now.)"""
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
        if len(words) < 2:                            # fodder + indicator, minimum
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
