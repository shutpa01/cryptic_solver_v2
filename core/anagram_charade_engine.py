"""Anagram+charade engine — a charade where ONE piece is an anagram (compound).

CATALOG-DRIVEN like the charade engine (copied, not shared — the 13-isolated-engines
rule), extended so a slot can be an anagram span:

  CUTHBERT = anag("angry") of BUTCHER + T (cross)   [ANA_F + ABR_F], def "saint"
  ARCADE   = A + anag("wrong") of CARD + E (East)    [ABR_F + ANA_F + ABR_F]

The pieces concatenate left-to-right to the answer (a charade). Each SYN_F/ABR_F slot
is filled from the DB lookup (known letters, must sit at its position, exactly as in
the charade engine). An ANA_F slot is filled differently: it occupies a SPAN of the
answer whose letters equal the fodder word's letters (a multiset match), wherever the
tiling places it.

The anagram indicator is NOT a catalog slot here (the mined signatures don't carry
one), so it is found among the wordplay words: a confirmed anagram indicator licenses
the ANA piece; link words are skipped. Definition comes from the signature's def_pos.

Per-piece COLOUR (each piece a colour, the anagram span coloured as one piece).
Pure and DB-decoupled. Evidence-preserving; no roles assigned on a fail.
"""

from core import contractions
from core.wfw_model import Source, Link, Annotation, Parse

# SYN_F/ABR_F are value-slots filled from the DB lookup (like charade).
ROLE_MECHANISM = {"SYN_F": "synonym", "ABR_F": "abbreviation"}


def _raw(text):
    return "".join(c for c in (text or "").upper() if c.isalpha())


def _is_anag_indicator(text, indicator_types):
    try:
        return "anagram" in (indicator_types(text) or set())
    except Exception:
        return False


def _value_candidates(role, phrase, answer, lookup):
    """Candidate values for a SYN_F/ABR_F slot — role-pure, answer-aware (as charade)."""
    mech = ROLE_MECHANISM.get(role)
    if mech is None:
        return []
    out, seen = [], set()
    for value, m in lookup(phrase, answer):
        v = (value or "").upper()
        if m == mech and v and v in answer and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _reconstruct(slot_infos, answer):
    """Tile the answer left-to-right across the slots. A value-slot tries its
    candidate strings; an anagram-slot consumes a span of the fodder's length whose
    letters match the fodder (multiset). Returns the chosen value per slot or None."""
    n = len(slot_infos)
    chosen = [None] * n

    def search(i, pos):
        if i == n:
            return pos == len(answer)
        kind, _toks, data = slot_infos[i]
        if kind == "val":
            for v in data:
                if answer.startswith(v, pos):
                    chosen[i] = v
                    if search(i + 1, pos + len(v)):
                        return True
        else:  # anagram span
            flen = len(data)
            span = answer[pos:pos + flen]
            if len(span) == flen and sorted(span) == sorted(data):
                chosen[i] = span
                if search(i + 1, pos + flen):
                    return True
        return False

    return list(chosen) if search(0, 0) else None


def _slot_infos(template, fodder_pool, answer, lookup):
    """Map the content words onto the signature's slots and build per-slot fill
    info. Returns [(kind, tokens, data), ...] or None if a slot cannot be filled.
    kind 'val' -> data is the candidate value list; 'ana' -> data is fodder letters."""
    total = sum(s.n_words for s in template.slots)
    if len(fodder_pool) != total:
        return None
    infos, pos = [], 0
    for slot in template.slots:
        toks = fodder_pool[pos:pos + slot.n_words]
        pos += slot.n_words
        phrase = " ".join(t.text for t in toks)
        if slot.role == "ANA_F":
            letters = "".join(_raw(t.text) for t in toks)
            if not letters:
                return None
            infos.append(("ana", toks, letters))
        else:
            cands = _value_candidates(slot.role, phrase, answer, lookup)
            if not cands:
                return None
            infos.append(("val", toks, cands))
    return infos


def _try_template(ctx, answer, template, split, lookup, is_link, indicator_types):
    """Try to instantiate one anagram_charade signature on one definition split."""
    if split.where != template.def_pos:
        return None
    if not any(s.role == "ANA_F" for s in template.slots):
        return None
    wordplay = list(split.wordplay_tokens)
    # The anagram piece needs a licensing indicator that is NOT a slot word. Try each
    # confirmed anagram indicator as THE indicator; the rest (minus links) are slots.
    indicators = [t for t in wordplay if _is_anag_indicator(t.text, indicator_types)]
    if not indicators:
        return None
    for ind in indicators:
        rest = [t for t in wordplay if t is not ind]
        fodder_pool = [t for t in rest if not is_link(t.text)]
        link_tokens = [t for t in rest if is_link(t.text)]
        infos = _slot_infos(template, fodder_pool, answer, lookup)
        if infos is None:
            continue
        values = _reconstruct(infos, answer)
        if values is None:
            continue
        return _build(ctx, answer, split, infos, values, ind, link_tokens, template)
    return None


def _build(ctx, answer, split, slot_infos, values, indicator, link_tokens, template):
    """Assemble the Parse: one Source per piece (anagram span included), per-letter
    links coloured by piece, the indicator and link words as annotations."""
    definition = Source(
        clue_atom_ids=split.def_atom_ids, text=split.phrase,
        value=ctx.answer_text, mechanism="definition", source=split.source)

    sources, links, pos = [], [], 0
    for si, ((kind, toks, _data), value) in enumerate(zip(slot_infos, values)):
        atom_ids = tuple(aid for t in toks for aid in t.atom_ids)
        text = " ".join(t.text for t in toks)
        mech = "anagram_fodder" if kind == "ana" else ROLE_MECHANISM[
            [s for s in template.slots][si].role]
        sources.append(Source(clue_atom_ids=atom_ids, text=text, value=value,
                              mechanism=mech))
        for _ in value:
            pos += 1
            links.append(Link(answer_pos=pos, source_index=si,
                              operation="anagram_charade", clue_atom_id=None,
                              transform="anagram_of" if kind == "ana" else None))

    annotations = [Annotation(clue_atom_ids=indicator.atom_ids, text=indicator.text,
                              role="indicator", note="anagram indicator")]
    annotations += [Annotation(clue_atom_ids=t.atom_ids, text=t.text,
                               role="link", note="link word") for t in link_tokens]
    from core.definition_engine import dbe_annotation
    dbe = dbe_annotation(split)        # by-example marker (peeled from wordplay)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(
        clue_text=ctx.clue_text, answer_text=ctx.answer_text,
        sources=sources, links=links, annotations=annotations,
        definition=definition, operation="anagram_charade", solved_by="catalog")
    parse.template_id = template.id
    parse.matched_signature = template.signature
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """Three-state verdict (as charade): pass / pending (provisional def or piece) /
    fail (a clue word unaccounted)."""
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully tiled by the pieces")
    missing = parse.unexplained_words(ctx)
    if missing:
        warnings.append("these clue words are unaccounted for: "
                        + ", ".join(repr(m) for m in missing))
    if parse.definition is None:
        warnings.append("no definition found")
    elif getattr(parse.definition, "source", "db") == "pending":
        warnings.append("the definition is provisional (queued for enrichment)")
    if any(getattr(s, "source", "db") == "pending" for s in parse.sources):
        warnings.append("a wordplay piece is provisional (queued for enrichment)")

    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing:
        parse.status = "fail"
    else:
        parse.status = "pending"


def solve_anagram_charade(ctx, defines, lookup, is_link, indicator_types, templates,
                          define_fallback=None, is_dbe=None):
    """Full anagram+charade solve — catalog-driven. Returns the first clean PASS,
    else the best parse found (pending/fail), else None."""
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3 or not templates:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None

    best = None
    for template in templates:                       # priority order
        for split in splits:
            parse = _try_template(ctx, answer, template, split, lookup, is_link,
                                  indicator_types)
            if parse is None:
                continue
            if parse.status == "pass":
                return parse
            if best is None:
                best = parse
    return best
