"""Charade engine — CATALOG-DRIVEN with links-as-residue (design §4 + the binding
never-preassign-links rule reconciled).

This is the redesign's signature-driven charade engine. It walks the mined charade
signatures (catalog_templates, via core.catalog_loader) in priority order and tries
to INSTANTIATE each one:

  for each charade signature (most frequent first):
    - the signature says which edge the definition sits on (def_pos); use only the
      definition splits at THAT edge (the definition is decided upstream, never here),
    - PLACE the signature's typed slots onto disjoint word-runs of the wordplay in
      clue order, GAPS ALLOWED between and around them (a slot consumes its n_words
      consecutive words),
    - fill each slot STRICTLY by role (SYN_F -> a synonym, ABR_F -> an abbreviation;
      never the word's own raw letters — that is the free-tiling the catalog replaces),
    - the filled pieces must concatenate left-to-right to the EXACT answer,
    - the GAP words left over are classified LAST: a function/connective word (is_link,
      or POS ADP/PART/AUX/DET/CCONJ/SCONJ/VERB/ADV) is a link; anything else leaves the
      parse unaccounted and that placement is rejected. Links are NEVER pre-stripped.

The difference from the earlier catalog-driven engine (commit fb0119d4) is exactly the
placement step: that version pre-stripped link words by is_link before mapping the
rest onto slots (`fodder = [t for t in wordplay if not is_link(t)]`), which is the
pre-assignment the user rejected. Here links fall out as residue, classified only once
a complete placement exists — both the catalog design and the links-last rule honoured.

ISOLATED per the 13-engines decision: its own assembly + its own verifier, no shared
operation substrate. Definition decided upstream. Pure and DB-decoupled.
"""

from core import grammar
from core.wordplay import GLUE_POS, raw
from core.wfw_model import Source, Link, Annotation, Parse


# A charade slot's role dictates the ONLY mechanism allowed to fill it — the
# constraint that makes the breakdown true rather than coincidental, and the source
# of the displayed label (NOT a guess from which table a value fell out of).
#   SYN_F -> a synonym; ABR_F -> an abbreviation; LIT_F -> the word's OWN letters
#   (a literal, e.g. "in" -> IN). A literal is safe here ONLY because a signature
#   pins which slot is the literal — there is no free raw-tiling of arbitrary words.
ROLE_MECHANISM = {
    "SYN_F": "synonym",
    "ABR_F": "abbreviation",
    "LIT_F": "raw",
}


def _role_candidates(role, phrase, answer, lookup):
    """Values that may fill a slot of `role`, drawn from `phrase`. ROLE-PURE: a SYN_F
    slot accepts only synonyms, an ABR_F slot only abbreviations, a LIT_F slot only
    the phrase's own letters. The placement still requires the value to land at the
    exact answer position, so a literal is constrained, not a wildcard."""
    if role == "LIT_F":
        lit = raw(phrase)
        return [lit] if lit else []
    mech_wanted = ROLE_MECHANISM.get(role)
    if mech_wanted is None:
        return []
    out, seen = [], set()
    for value, mech in lookup(phrase, answer):
        v = (value or "").upper()
        if mech == mech_wanted and v and v in answer and v not in seen:
            out.append(v)
            seen.add(v)
    return out


def _place(slots, words, answer, postags, lookup, is_link):
    """Place the typed slots onto disjoint word-runs in clue order (gaps allowed),
    filling each by role so the pieces concatenate to EXACTLY the answer; classify
    the gap words as links LAST. Returns {pieces, links} or None.

    pieces: [(start, end, role, value)] in slot order. links: gap word indices.
    """
    n, N = len(words), len(answer)
    nslots = len(slots)

    def residue_link(k):
        return (is_link and is_link(words[k].text)) or (postags[k] in GLUE_POS)

    def finalize(pieces, gap_idxs):
        links = []
        for k in gap_idxs:
            if residue_link(k):
                links.append(k)
            else:
                return None                  # a content word unaccounted -> reject
        return {"pieces": pieces, "links": links}

    def dfs(si, wi, pos, pieces, gaps):
        if si == nslots:
            if pos != N:
                return None
            return finalize(pieces, gaps + list(range(wi, n)))
        slot = slots[si]
        nw = slot.n_words
        for j in range(wi, n - nw + 1):           # slot starts at j; wi..j are gaps
            phrase = " ".join(words[k].text for k in range(j, j + nw))
            for val in _role_candidates(slot.role, phrase, answer, lookup):
                if answer.startswith(val, pos):
                    r = dfs(si + 1, j + nw, pos + len(val),
                            pieces + [(j, j + nw, slot.role, val)],
                            gaps + list(range(wi, j)))
                    if r:
                        return r
        return None

    return dfs(0, 0, 0, [], [])


def _try_template(ctx, answer, template, split, words, postags, lookup, is_link):
    """Instantiate one signature on one definition split. Parse or None."""
    if split.where != template.def_pos:
        return None
    if template.fodder_word_count > len(words):
        return None
    if any(s.role not in ROLE_MECHANISM for s in template.slots):
        return None                              # role this engine can't fill
    placement = _place(template.slots, words, answer, postags, lookup, is_link)
    if placement is None:
        return None
    return _build(ctx, split, words, placement, template)


def _build(ctx, split, words, placement, template):
    """Assemble the wfw_model.Parse from a matched, placed signature."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, links, pos = [], [], 0
    for si, (a, b, role, value) in enumerate(placement["pieces"]):
        toks = words[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=value,
            mechanism=ROLE_MECHANISM[role]))
        for _ in value:
            pos += 1
            links.append(Link(answer_pos=pos, source_index=si,
                              operation="charade", clue_atom_id=None))
    annotations = [Annotation(clue_atom_ids=words[k].atom_ids, text=words[k].text,
                              role="link", note="link word")
                   for k in placement["links"]]
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="charade", solved_by="catalog")
    parse.template_id = template.id
    parse.matched_signature = template.signature
    _verify_charade(ctx, parse)
    return parse


def _verify_charade(ctx, parse):
    """Three-state verdict: pass / pending (provisional def or piece) / fail."""
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


def solve_charade(ctx, defines, lookup, is_link, templates, define_fallback=None,
                  is_dbe=None):
    """Full charade solve — catalog-driven. Walk the charade signatures in priority
    order; for each, try every confirmed definition split at the signature's def edge:
    place the typed slots on the wordplay (gaps -> links, classified last), fill by
    role, verify the pieces concatenate to the answer.

    Among PASSing placements, prefer the one with the FEWEST residue (link) words —
    i.e. the parse that accounts for the most clue words as pieces, breaking ties by
    signature priority (templates are iterated in priority order, so the earlier wins
    a tie). This stops a coarser signature from dropping a content word that POS
    mis-tags as a link (e.g. "beastly home" -> SETT: minimise-residue keeps the
    two-word piece instead of dropping "beastly"). A zero-residue pass is optimal and
    returns immediately. Else the best non-pass parse; else a fail-evidence parse."""
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 2 or not templates:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None

    # Pre-tag each split's wordplay words once (POS used for residue classification).
    prepared = []
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 2:
            continue
        postags = grammar.pos_tags([t.text for t in words]) or [None] * len(words)
        prepared.append((split, words, postags))
    if not prepared:
        return None

    best_pass, best_pass_residue, best_other = None, None, None
    for template in templates:                       # priority order
        for split, words, postags in prepared:
            parse = _try_template(ctx, answer, template, split, words, postags,
                                  lookup, is_link)
            if parse is None:
                continue
            if parse.status == "pass":
                residue = sum(1 for a in parse.annotations if a.role == "link")
                if best_pass is None or residue < best_pass_residue:
                    best_pass, best_pass_residue = parse, residue
                    if residue == 0:
                        return parse             # cannot account for more words
            elif best_other is None:
                best_other = parse
    if best_pass is not None:
        return best_pass
    if best_other is not None:
        return best_other

    return _build_fail_evidence(ctx, answer, prepared[0][0], lookup)


def _build_fail_evidence(ctx, answer, split, lookup):
    """Preserve the evidence when no signature instantiated: keep the definition and
    show, for each wordplay word, a candidate value it COULD contribute. A FAIL
    asserts nothing about structure, so it assigns NO roles — function words are left
    unaccounted, never relabelled links by elimination (feedback-no-role-on-fail)."""
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, unresolved = [], []
    for token in split.wordplay_tokens:
        cands = []
        for value, mech in lookup(token.text, answer):
            v = (value or "").upper()
            if v and v in answer and v not in {c for c, _ in cands}:
                cands.append((v, mech))
        if cands:
            value, mech = cands[0]
            sources.append(Source(clue_atom_ids=token.atom_ids, text=token.text,
                                  value=value, mechanism=mech))
        else:
            unresolved.append(token.text)
    warnings = ["no charade signature matched this clue "
                "(pieces below are candidates, not a placement)"]
    if unresolved:
        warnings.append("these clue words are unaccounted for: "
                        + ", ".join(repr(u) for u in unresolved))
    from core.definition_engine import dbe_annotation
    dbe = dbe_annotation(split)
    annotations = [dbe] if dbe is not None else []
    return Parse(
        clue_text=ctx.clue_text, answer_text=ctx.answer_text,
        sources=sources, links=[], annotations=annotations,
        definition=definition, operation="charade", solved_by="catalog",
        status="fail", warnings=warnings)
