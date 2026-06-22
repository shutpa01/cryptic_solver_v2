"""Anagram+charade engine — CATALOG-DRIVEN with links-as-residue (design §4).

Signature-driven sibling of the charade signature engine, for the compound type
"a charade where ONE piece is an anagram". It walks the mined anagram_charade
signatures (catalog_templates, via core.catalog_loader) in priority order and tries
to INSTANTIATE each one:

  for each anagram_charade signature (most frequent first), for each definition split
  at the signature's def edge:
    - PLACE the signature's typed slots onto disjoint word-runs of the wordplay in
      clue order, GAPS ALLOWED between and around them (a slot consumes its n_words
      consecutive words),
    - fill each slot STRICTLY by role:
        ANA_F -> the run's letters (as written, or contraction-stripped "Lionel's"
                 ->LIONEL) must ANAGRAM to the SPAN of the answer at the current
                 position (sorted-equal, not an exact reversal); the piece contributes
                 that span,
        SYN_F -> a synonym of the run that sits at the current answer position,
        ABR_F -> an abbreviation of the run that sits at the current answer position,
    - the filled pieces must concatenate left-to-right to the EXACT answer,
    - the GAP words left over are classified LAST: among them the anagram INDICATOR is
      a confirmed anagram indicator (or, if the DB doesn't know it, the run adjacent to
      the anagram piece accepted PROVISIONALLY -> pending, the missing-indicator
      fallback); the remaining residue words are links (is_link or POS function word);
      anything else leaves the parse unaccounted and that placement is rejected. Links
      and the indicator are NEVER pre-stripped.

The anagram_charade signatures carry no indicator slot (mined without one), so unlike
the pure-anagram signature engine the indicator is found in the residue, exactly as
the evidence engine did. ANA_F here anagrams to a SPAN (the pieces concatenate to the
answer), not to the whole answer — that span attribution is the only structural
difference from a pure anagram, and a single-ANA_F signature (a disguised pure anagram)
is skipped: this engine requires exactly one anagram piece AND at least one charade
piece, so a true compound.

Among PASSing placements, prefer the FEWEST residue (link) words, then signature
priority. Records the matched signature (template_id) — design §10.

ISOLATED per the 13-engines decision: its own assembly + its own verifier, no shared
operation substrate; the colour/letter assignment mirrors the evidence anagram+charade
engine (one Source per piece, the anagram span coloured as one). Definition decided
upstream. Pure and DB-decoupled.
"""

from core import grammar
from core.wordplay import GLUE_POS, fodder_letter_forms, is_anagram_indicator, \
    adjacent_run
from core.wfw_model import Source, Link, Annotation, Parse


# A slot's role dictates the ONLY mechanism allowed to fill it — the constraint that
# makes the breakdown true rather than coincidental, and the source of the displayed
# label. ANA_F is the anagram piece; SYN_F a synonym; ABR_F an abbreviation. (No LIT_F
# — anagram_charade signatures were mined without literal slots.)
ROLE_MECHANISM = {
    "ANA_F": "anagram_fodder",
    "SYN_F": "synonym",
    "ABR_F": "abbreviation",
}


def _role_candidates(role, phrase, answer, lookup, suggest_piece=None):
    """Values that may fill a SYN_F / ABR_F slot drawn from `phrase` — ROLE-PURE: a
    SYN_F slot accepts only synonyms, an ABR_F slot only abbreviations. (ANA_F is not
    handled here; the anagram fodder is matched against the answer span in _place.)

    `suggest_piece`, when supplied (the Haiku fallback), is consulted ONLY for a SYNONYM
    slot the DB could not fill — its value is PROVISIONAL (the caller marks the piece
    pending and queues it), so it never becomes a silent pass."""
    mech_wanted = ROLE_MECHANISM.get(role)
    if mech_wanted not in ("synonym", "abbreviation"):
        return []
    out, seen = [], set()
    for value, mech in lookup(phrase, answer):
        v = (value or "").upper()
        if mech == mech_wanted and v and v in answer and v not in seen:
            out.append(v)
            seen.add(v)
    # Haiku fallback for a missing SYNONYM piece. Consulted when the DB gave NOTHING,
    # OR only weak single-letter candidates — a stray 1-letter synonym (often junk)
    # must not suppress the real multi-letter piece.
    if suggest_piece is not None and mech_wanted == "synonym" and (
            not out or all(len(v) == 1 for v in out)):
        v = (suggest_piece(phrase, answer) or "").upper()
        if v and v in answer and v not in out:
            out.append(v)
    return out


def _place(slots, words, answer, postags, lookup, is_link, indicator_types,
           suggest_piece=None):
    """Place the typed slots onto disjoint word-runs in clue order (gaps allowed),
    filling each by role so the pieces concatenate to EXACTLY the answer; classify the
    gap words as indicator/links LAST. Returns {pieces, indicator, links,
    indicator_source} or None.

    pieces: [(start, end, role, value)] in slot order — the ANA_F piece's value is the
    answer span it anagrams to.
    """
    n, N = len(words), len(answer)
    nslots = len(slots)

    def residue_link(k):
        return (is_link and is_link(words[k].text))

    def is_ind_word(k):
        return is_anagram_indicator(words[k].text, indicator_types)

    def finalize(pieces, gap_idxs):
        anag = next(((a, b) for a, b, role, _ in pieces if role == "ANA_F"), None)
        if anag is None:
            return None
        # The indicator is residue: a confirmed anagram indicator among the gaps, else
        # the MISSING-INDICATOR FALLBACK — the gap run adjacent to the anagram piece,
        # accepted provisionally and queued for enrichment (the anagram is otherwise
        # proven by the letter-match).
        indicator = [k for k in gap_idxs if is_ind_word(k)]
        source = "db"
        if not indicator:
            indicator = adjacent_run(gap_idxs, anag[0], anag[1])
            if not indicator:
                return None
            source = "pending"
        ind_set = set(indicator)
        links = []
        for k in gap_idxs:
            if k in ind_set:
                continue
            if residue_link(k):
                links.append(k)
            else:
                return None                          # a content word unaccounted
        return {"pieces": pieces, "indicator": sorted(indicator),
                "links": links, "indicator_source": source}

    def dfs(si, wi, pos, pieces, gaps):
        if si == nslots:
            if pos != N:
                return None
            return finalize(pieces, gaps + list(range(wi, n)))
        slot = slots[si]
        nw = slot.n_words
        for j in range(wi, n - nw + 1):               # slot starts at j; wi..j are gaps
            if slot.role == "ANA_F":
                for fl in fodder_letter_forms(words[j:j + nw]):
                    span = answer[pos:pos + len(fl)]
                    if len(span) == len(fl) and sorted(span) == sorted(fl) \
                            and span[::-1] != fl:    # exact reversal isn't an anagram
                        r = dfs(si + 1, j + nw, pos + len(fl),
                                pieces + [(j, j + nw, "ANA_F", span)],
                                gaps + list(range(wi, j)))
                        if r:
                            return r
            else:
                phrase = " ".join(words[k].text for k in range(j, j + nw))
                for val in _role_candidates(slot.role, phrase, answer, lookup,
                                            suggest_piece):
                    if answer.startswith(val, pos):
                        r = dfs(si + 1, j + nw, pos + len(val),
                                pieces + [(j, j + nw, slot.role, val)],
                                gaps + list(range(wi, j)))
                        if r:
                            return r
        return None

    return dfs(0, 0, 0, [], [])


def _try_template(ctx, answer, template, split, words, postags, lookup, is_link,
                  indicator_types, suggest_piece=None):
    """Instantiate one signature on one definition split. Parse or None."""
    if split.where != template.def_pos:
        return None
    if template.fodder_word_count > len(words):
        return None
    roles = [s.role for s in template.slots]
    if not set(roles) <= set(ROLE_MECHANISM):
        return None                                  # a role this engine can't fill
    if roles.count("ANA_F") != 1:
        return None                                  # exactly one anagram piece
    if all(r == "ANA_F" for r in roles):
        return None                                  # a disguised pure anagram, skip
    placement = _place(template.slots, words, answer, postags, lookup, is_link,
                       indicator_types, suggest_piece)
    if placement is None:
        return None
    return _build(ctx, split, words, placement, template, lookup)


def _build(ctx, split, words, placement, template, lookup):
    """Assemble the wfw_model.Parse from a matched, placed signature. One Source per
    piece (the anagram span coloured as one); per-letter links coloured by piece;
    indicator + link words and the by-example marker as annotations. A SYNONYM piece
    whose value is NOT a DB synonym for its phrase came from the Haiku fallback — it is
    marked source='pending' so the parse goes pending and the piece is queued."""
    from core.definition_engine import dbe_annotation
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    sources, links, pos = [], [], 0
    for si, (a, b, role, value) in enumerate(placement["pieces"]):
        toks = words[a:b]
        phrase = " ".join(t.text for t in toks)
        origin = "db"
        if role == "SYN_F":
            db_vals = {(v or "").upper() for v, m in lookup(phrase, answer)
                       if m == "synonym"}
            if value not in db_vals:        # came from the Haiku fallback
                origin = "pending"
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=phrase, value=value,
            mechanism=ROLE_MECHANISM[role], source=origin))
        for _ in value:
            pos += 1
            links.append(Link(answer_pos=pos, source_index=si,
                              operation="anagram_charade", clue_atom_id=None,
                              transform="anagram_of" if role == "ANA_F" else None))
    ind_toks = [words[k] for k in placement["indicator"]]
    annotations = []
    if ind_toks:
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
            text=" ".join(t.text for t in ind_toks), role="indicator",
            note="anagram indicator",
            source=placement.get("indicator_source", "db")))
    for k in placement["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="anagram_charade",
                  solved_by="catalog")
    parse.template_id = template.id
    parse.matched_signature = template.signature
    _verify(ctx, parse)
    return parse


def _verify(ctx, parse):
    """Three-state verdict: pass / pending (provisional def or indicator) / fail.
    Identical logic to the evidence anagram+charade engine — only the placement that
    produced the parse is catalog-driven."""
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
    if any(a.role == "indicator" and getattr(a, "source", "db") == "pending"
           for a in parse.annotations):
        warnings.append("the anagram indicator is provisional (queued for enrichment)")
    if any(getattr(s, "source", "db") == "pending" for s in parse.sources):
        warnings.append("a wordplay piece is provisional (queued for enrichment)")
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


def solve_anagram_charade(ctx, defines, lookup, is_link, indicator_types, templates,
                          define_fallback=None, is_dbe=None, suggest_piece=None):
    """Full anagram+charade solve — catalog-driven. Walk the anagram_charade signatures
    in priority order; for each, try every confirmed definition split at the signature's
    def edge: place the typed slots on the wordplay (gaps -> indicator/links classified
    last), fill by role, verify the pieces concatenate to the answer.

    Among PASSing placements, pick the FEWEST residue (link) words, then signature
    priority (templates iterated in priority order). A zero-residue pass returns
    immediately. Else the best non-pass parse, else None."""
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3 or not templates:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None

    prepared = []
    for split in splits:
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 2:
            continue
        postags = grammar.wordplay_pos_tags(ctx, words)
        prepared.append((split, words, postags))
    if not prepared:
        return None

    best_pass = best_pending = best_other = None
    pass_key = pend_key = None
    for template in templates:                        # priority order
        for split, words, postags in prepared:
            parse = _try_template(ctx, answer, template, split, words, postags,
                                  lookup, is_link, indicator_types, suggest_piece)
            if parse is None:
                continue
            residue = sum(1 for a in parse.annotations if a.role == "link")
            if parse.status == "pass":
                if pass_key is None or residue < pass_key:
                    best_pass, pass_key = parse, residue
                    if residue == 0:
                        return parse                 # nothing could beat zero residue
            elif parse.status == "pending":
                if pend_key is None or residue < pend_key:
                    best_pending, pend_key = parse, residue
            elif best_other is None:
                best_other = parse
    if best_pass is not None:
        return best_pass
    if best_pending is not None:
        return best_pending
    if best_other is not None:
        return best_other
    # No signature instantiated. Do NOT return None and discard what was found —
    # preserve the evidence (design §2 / §5.9) so the gap is visible and the separate
    # signature-creation process has the pieces to work from.
    return _build_fail_evidence(ctx, answer, prepared[0][0], lookup, indicator_types)


def _build_fail_evidence(ctx, answer, split, lookup, indicator_types):
    """Preserve the evidence when no anagram_charade signature instantiated. A FAIL
    asserts nothing about structure, so it assigns NO link/indicator roles by
    elimination (feedback-no-role-on-fail). It keeps the definition and shows, for
    each wordplay word, the evidence collected:
      - a candidate VALUE it could contribute (a synonym/abbreviation that is a
        substring of the answer), and
      - any contiguous run of wordplay words whose letters ANAGRAM to a span of the
        answer — the anagram-fodder evidence distinctive to this type, marked as a
        candidate, not a committed piece.
    Both are surfaced as candidate sources so the gap can name its pieces."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    tokens = [t for t in split.wordplay_tokens if t.kind == "word"]
    sources, unresolved = [], []

    # Anagram-fodder candidate FIRST: the longest contiguous run whose letters anagram to a
    # span of the answer. Words INSIDE it are accounted as fodder, so they must be excluded
    # from the value/unaccounted check below — otherwise a fodder word is shown as fodder
    # AND reported unaccounted (the contradiction this fixes).
    anag = _anagram_fodder_candidate(tokens, answer)
    fodder_idx = set(range(anag[0], anag[1])) if anag is not None else set()

    # Value candidates: per NON-fodder word, a synonym/abbreviation that sits inside the
    # answer. A word with neither a value nor a fodder role is genuinely unaccounted.
    for i, token in enumerate(tokens):
        if i in fodder_idx:
            continue                              # accounted as anagram fodder below
        cand = None
        for value, mech in lookup(token.text, answer):
            v = (value or "").upper()
            if v and v in answer:
                cand = (v, mech)
                break
        if cand is not None:
            sources.append(Source(clue_atom_ids=token.atom_ids, text=token.text,
                                  value=cand[0], mechanism=cand[1]))
        else:
            unresolved.append(token.text)

    if anag is not None:
        a, b, letters = anag
        toks = tokens[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=letters,
            mechanism="anagram_fodder"))

    warnings = ["no anagram_charade signature matched this clue "
                "(pieces below are candidates, not a placement)"]
    if unresolved:
        warnings.append("these clue words are unaccounted for: "
                        + ", ".join(repr(u) for u in unresolved))
    dbe = dbe_annotation(split)
    annotations = [dbe] if dbe is not None else []
    return Parse(
        clue_text=ctx.clue_text, answer_text=ctx.answer_text,
        sources=sources, links=[], annotations=annotations,
        definition=definition, operation="anagram_charade", solved_by="catalog",
        status="fail", warnings=warnings)


def _anagram_fodder_candidate(tokens, answer):
    """The longest contiguous run of wordplay tokens whose letters anagram to some
    contiguous span of the answer (sorted-equal, not an exact reversal). Returns
    (start, end, span_letters) or None. Pure evidence — no placement is asserted."""
    n, N = len(tokens), len(answer)
    best = None
    for a in range(n):
        for b in range(a + 1, n + 1):
            for fl in fodder_letter_forms(tokens[a:b]):
                L = len(fl)
                if L < 3 or L > N:
                    continue
                key = sorted(fl)
                for start in range(0, N - L + 1):
                    span = answer[start:start + L]
                    if sorted(span) == key and span[::-1] != fl:
                        if best is None or L > len(best[2]):
                            best = (a, b, span)
    return best
