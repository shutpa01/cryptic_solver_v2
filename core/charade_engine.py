"""Charade engine — CATALOG-DRIVEN (design §4).

A charade reads in clue order: each fodder piece contributes letters and the
pieces join end-to-end to spell the answer. There is no reordering without a
positional indicator, so the slot order is fixed by clue order.

This engine does NOT free-tile the answer. It walks the mined charade signatures
(catalog_templates, via core.catalog_loader) in priority order and tries to
INSTANTIATE each one (design §4 "Catalog engine (the core)"):

  for each charade signature (most frequent first):
    - the signature says where the definition sits (def_pos start/end); split it
      off THAT edge and require the DB to confirm it defines the answer,
    - the remaining wordplay words map one-to-one onto the signature's typed
      slots (link words skipped, recorded as annotations),
    - each slot is filled STRICTLY by its role: SYN_F -> a synonym only,
      ABR_F -> an abbreviation only (no raw-letter wildcard, no role mixing),
    - the filled pieces must concatenate left-to-right to the EXACT answer,
    - emit the per-letter wfw_model record and verify.

A clue matching NO signature is a catalog-creation gap (design §8), not a
free-tile fail. Evidence is preserved on every branch (the definition and the
candidate pieces are shown); a no-result branch never returns None when a
definition was found (design §2).

ISOLATED per the 13-engines decision (memory: catalog-13-isolated-engines): this
engine carries its OWN copy of the assembly logic and its OWN verifier, sharing
no operation substrate with the other engines. Paired with its own bespoke screen.

Pure and DB-decoupled: takes a wfw_atoms context, the loaded charade `templates`,
and injected predicates (defines / lookup / is_link); returns a wfw_model.Parse
or None.
"""

from core.wfw_model import Source, Link, Annotation, Parse


# A charade slot's role dictates the ONLY mechanism allowed to fill it. This is
# the constraint that makes the breakdown true rather than coincidental.
ROLE_MECHANISM = {
    "SYN_F": "synonym",
    "ABR_F": "abbreviation",
}


def _role_candidates(role, phrase, answer, lookup):
    """The values that may fill a slot of `role`, drawn from `phrase`.

    Answer-aware and UNCAPPED (the answer itself is the filter), but ROLE-PURE:
    a SYN_F slot accepts only synonyms, an ABR_F slot only abbreviations. The
    word's own raw letters are NOT a candidate — that wildcard is exactly the
    free-tiling the catalog-driven engine exists to replace.
    """
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


def _reconstruct(cand_lists, answer):
    """Choose one value per slot (in order) so they concatenate to EXACTLY answer.

    Returns the chosen values [v0, v1, ...] or None. Pieces read left-to-right in
    slot (= clue) order; there is no reordering in a charade.
    """
    n = len(cand_lists)
    chosen = [None] * n

    def search(i, pos):
        if i == n:
            return pos == len(answer)
        for v in cand_lists[i]:
            if answer.startswith(v, pos):
                chosen[i] = v
                if search(i + 1, pos + len(v)):
                    return True
        return False

    return list(chosen) if search(0, 0) else None


def _slot_groups(template, fodder_tokens):
    """Partition the fodder tokens into the template's slots by n_words.

    Returns [(slot, [tokens]), ...] in slot order, or None if the token count
    does not match the signature's fodder-word count exactly.
    """
    if len(fodder_tokens) != template.fodder_word_count:
        return None
    groups, pos = [], 0
    for slot in template.slots:
        groups.append((slot, fodder_tokens[pos:pos + slot.n_words]))
        pos += slot.n_words
    return groups


def _try_template(ctx, answer, template, split, lookup, is_link):
    """Try to instantiate one signature on one definition split. Parse or None."""
    if split.where != template.def_pos:
        return None
    fodder = [t for t in split.wordplay_tokens if not is_link(t.text)]
    links = [t for t in split.wordplay_tokens if is_link(t.text)]
    groups = _slot_groups(template, fodder)
    if groups is None:
        return None

    cand_lists = []
    for slot, toks in groups:
        phrase = " ".join(t.text for t in toks)
        cands = _role_candidates(slot.role, phrase, answer, lookup)
        if not cands:
            return None
        cand_lists.append(cands)

    values = _reconstruct(cand_lists, answer)
    if values is None:
        return None
    return _build(ctx, answer, split, groups, values, links, template)


def _build(ctx, answer, split, groups, values, link_tokens, template):
    """Assemble the wfw_model.Parse from a matched, reconstructed signature."""
    definition = Source(
        clue_atom_ids=split.def_atom_ids, text=split.phrase,
        value=ctx.answer_text, mechanism="definition", source=split.source)

    sources, links = [], []
    pos = 0
    for si, ((slot, toks), value) in enumerate(zip(groups, values)):
        atom_ids = tuple(aid for t in toks for aid in t.atom_ids)
        text = " ".join(t.text for t in toks)
        sources.append(Source(clue_atom_ids=atom_ids, text=text,
                              value=value, mechanism=ROLE_MECHANISM[slot.role]))
        for _ in value:
            pos += 1
            links.append(Link(answer_pos=pos, source_index=si,
                              operation="charade", clue_atom_id=None))

    annotations = [Annotation(clue_atom_ids=t.atom_ids, text=t.text,
                              role="link", note="link word")
                   for t in link_tokens]
    from core.definition_engine import dbe_annotation
    dbe = dbe_annotation(split)        # by-example marker (peeled from wordplay)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(
        clue_text=ctx.clue_text, answer_text=ctx.answer_text,
        sources=sources, links=links, annotations=annotations,
        definition=definition, operation="charade", solved_by="catalog")
    # Diagnostic provenance: which mined signature produced this (design §10 wants
    # template attribution recorded; the Parse model carries it as plain attrs the
    # storage layer can read without a schema change).
    parse.template_id = template.id
    parse.matched_signature = template.signature
    return parse


def _verify_charade(ctx, parse):
    """Engine-level verification — rules SPECIFIC to charade, three-state verdict
    (memory: core-cascade-stop-and-verdict-rules). No shared grand verifier.

      pass    — every answer letter tiled by a DB-confirmed piece, every clue word
                accounted for, definition DB-confirmed.
      pending — fully tiled but a piece or the definition is provisional (a queued
                enrichment candidate).
      fail    — tiled assembly exists but a clue word is left unaccounted and is not
                a queueable gap.
    """
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

    provisional_piece = any(getattr(s, "source", "db") == "pending"
                            for s in parse.sources)
    if provisional_piece:
        warnings.append("a wordplay piece is provisional (queued for enrichment)")

    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing:
        parse.status = "fail"           # unaccounted words are not a queueable gap
    else:
        parse.status = "pending"        # only provisional pieces/definition remain


def solve_charade(ctx, defines, lookup, is_link, templates, define_fallback=None,
                  is_dbe=None):
    """Full charade solve — catalog-driven.

    1. Find the edge definitions the DB confirms (the Haiku fallback fires only on
       a complete DB miss and yields a provisional split — it is NOT the primary
       definition mechanism here; the definition position comes from the signature).
    2. Walk the charade signatures in priority order. For each, try every confirmed
       definition split at the signature's def_pos edge: lay the wordplay words on
       the typed slots, fill each by role, and verify the pieces concatenate to the
       answer.
    3. Return the first clean PASS; else the best parse found (pending/fail) so the
       evidence is shown; else a fail-evidence parse when a definition was found but
       no signature instantiated (catalog-creation gap, §8); else None.
    """
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 2 or not templates:
        return None

    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None

    best = None
    for template in templates:                       # priority order
        for split in splits:
            parse = _try_template(ctx, answer, template, split, lookup, is_link)
            if parse is None:
                continue
            _verify_charade(ctx, parse)
            if parse.status == "pass":
                return parse
            if best is None:
                best = parse
    if best is not None:
        return best

    # No signature instantiated. NEVER discard the evidence (design §2): a
    # definition was found, so return a FAIL parse that keeps it (still queued)
    # and shows the candidate letters each wordplay word could contribute. This is
    # a catalog-creation gap (§8), surfaced honestly — nothing fabricated, no
    # placement claimed (no links, uncoloured tiles).
    return _build_fail_evidence(ctx, answer, splits[0], lookup)


def _build_fail_evidence(ctx, answer, split, lookup):
    """Preserve the evidence when no signature instantiated.

    Keeps the definition (shown and queued) and records, for each wordplay word,
    the candidate synonym/abbreviation values it COULD contribute — so the user
    sees how far we got and which word defeated assembly.

    A FAIL asserts NOTHING about structure, so it assigns NO roles. In particular
    it does NOT stamp function words as link words: a link word is residue that may
    be named only when the operation is COMPLETE, and nothing is placed on a fail.
    Relabeling a word a "link" by elimination falsely shows it as explained when it
    may be an unfound wordplay piece (e.g. CATERWAUL's "with" = W). So any word with
    no resolvable candidate is left honestly UNACCOUNTED — never relabeled.
    (memory: feedback-no-role-assignment-on-fail; mirrors the hidden engine's rule.)
    """
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
    # A definition-by-example marker IS a confirmed role attached to the definition
    # (not relabel-by-elimination), so it is shown and accounted even on a fail.
    from core.definition_engine import dbe_annotation
    dbe = dbe_annotation(split)
    annotations = [dbe] if dbe is not None else []
    return Parse(
        clue_text=ctx.clue_text, answer_text=ctx.answer_text,
        sources=sources, links=[], annotations=annotations,
        definition=definition, operation="charade", solved_by="catalog",
        status="fail", warnings=warnings)
