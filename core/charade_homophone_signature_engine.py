"""Charade + homophone engine — WORDPLAY-FIRST, answer-driven, definition-as-residue.

A charade whose pieces concatenate to the answer, one piece a HOM_F homophone (an
answer SPAN that SOUNDS like a clue word, or a synonym of one): MIDDLEWEIGHT = MIDDLE
[intermediate] + WEIGHT [sounds like WAIT = delay].

Selection (no arbitrary tie-breaks):
  1. ANCHOR on the homophone indicator (DB-typed). No indicator -> not this type.
  2. The definition is a contiguous edge block; the wordplay is the rest and MUST
     contain the indicator. For each signature x edge split, solve the wordplay against
     the KNOWN answer: pieces (HOM_F by sound, SYN_F/ABR_F by DB value) concatenating to
     the exact answer, the indicator and any leftover words accounted (link words only).
  3. The leftover edge is the definition. DB-confirmed -> a real definition; else, if
     POS-coherent, a provisional residue.
  4. Among ALL complete parses, pick the one explaining the MOST PIECES (most answer
     letters from DB-confirmed pieces, a DB-confirmed definition preferred, fewest
     leftover links, fewest provisional pieces). The single principled selector.
  5. If NOTHING parses completely, do NOT discard — return the best PARTIAL by the same
     most-pieces measure, so the pieces found are preserved as evidence.

Enrichment (design §5.9): a missing-DB-synonym homophone source is filled by a
homophone-aware Haiku suggestion (the SOUND re-verified against the dictionary, and only
for spans of >=2 letters so a single answer letter cannot be passed off as a homophone),
marked PROVISIONAL -> PENDING and the synonym queued. ISOLATED per the 13-engines rule.
"""

from core import grammar, engine_common
from core.wordplay import raw
from core.wfw_model import Source, Link, Annotation, Parse


ROLE_MECHANISM = {
    "SYN_F": "synonym",
    "ABR_F": "abbreviation",
    "LIT_F": "raw",
}

_CONTENT = {"VERB", "NOUN", "ADJ", "PROPN", "NUM", "ADV"}
_TRAIL_BAD = {"ADP", "DET", "CCONJ", "SCONJ", "PART", "AUX"}


class _Stop(Exception):
    """Unwind the DFS once a complete placement is found (the AI pass needs only one)."""


def _letters(s):
    return "".join(c for c in (s or "").upper() if c.isalpha())


def _pos_coherent(def_idx, postags):
    """Does this leftover edge read like a definition? Needs >=1 content word and must
    not end on a dangling function word. Permissive when POS is unavailable."""
    if not postags:
        return True
    tags = [postags[i] for i in def_idx]
    if not any(t in _CONTENT for t in tags):
        return False
    if tags and tags[-1] in _TRAIL_BAD:
        return False
    return True


def _role_candidates(role, phrase, answer, lookup):
    """Substring-of-answer values that may fill a SYN_F/ABR_F/LIT_F slot, role-pure."""
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


def _hom_candidates(phrase, answer, pos, sounds_alike, synonyms_of, suggest_hom=None):
    """Spans of the answer at `pos` that `phrase` — itself or via a synonym — sounds
    like. Returns [(span, sound_source, provisional)]. A span equal to the source's OWN
    letters is rejected (literal, not homophone). The Haiku fallback is consulted ONLY
    for a span of >=2 letters the DB could not source, and its word's SOUND is
    re-verified here — a single answer letter (A~'ay') is never passed off as a
    homophone via an invented synonym (the ASHORE 'the->AY' fault)."""
    N = len(answer)
    out = []
    syns = None
    for L in range(1, N - pos + 1):
        span = answer[pos:pos + L]
        if _letters(phrase) != span and sounds_alike(phrase, span):
            out.append((span, phrase, False))
            continue
        if syns is None:
            syns = synonyms_of(phrase) or []
        matched = False
        for syn in syns:
            if _letters(syn) != span and sounds_alike(syn, span):
                out.append((span, syn, False))
                matched = True
                break
        if matched:
            continue
        if suggest_hom is not None and L >= 2 and sounds_alike(span, span):
            w = suggest_hom(phrase, span)
            if w and _letters(w) != span and sounds_alike(w, span):
                out.append((span, w.upper(), True))
    return out


def _enumerate(slots, words, answer, ind_set, lookup, is_link, sounds_alike,
               synonyms_of, suggest_hom, stop_first=False):
    """DFS placing the slots on `words` (gaps allowed) so pieces concatenate to EXACTLY
    the answer. Returns (completes, best_partial):
      completes   = list of {pieces, links} where every word is a piece/indicator/link;
      best_partial= (covered_letters, n_pieces, pieces) — the deepest prefix any
                    placement reached, kept so a non-solving clue still yields evidence.
    With stop_first, returns as soon as one complete placement exists (the AI pass)."""
    n, N = len(words), len(answer)
    nslots = len(slots)
    completes = []
    best = [None]

    def consider(pos, pieces):
        if not pieces:
            return
        cand = (pos, len(pieces))
        if best[0] is None or cand > (best[0][0], best[0][1]):
            best[0] = (pos, len(pieces), list(pieces))

    def finalize(pieces, gap_idxs):
        links = []
        for k in gap_idxs:
            if k in ind_set:
                continue
            if is_link and is_link(words[k].text):
                links.append(k)
            else:
                return None
        return {"pieces": pieces, "links": links}

    def dfs(si, wi, pos, pieces, gaps):
        consider(pos, pieces)
        if si == nslots:
            if pos == N:
                fin = finalize(pieces, gaps + list(range(wi, n)))
                if fin is not None:
                    completes.append(fin)
                    if stop_first:
                        raise _Stop()
            return
        slot = slots[si]
        nw = slot.n_words
        for j in range(wi, n - nw + 1):
            span_idxs = range(j, j + nw)
            if any(k in ind_set for k in span_idxs):
                continue
            phrase = " ".join(words[k].text for k in span_idxs)
            if slot.role == "HOM_F":
                for span, ss, prov in _hom_candidates(phrase, answer, pos, sounds_alike,
                                                      synonyms_of, suggest_hom):
                    dfs(si + 1, j + nw, pos + len(span),
                        pieces + [(j, j + nw, slot.role, span, ss, prov)],
                        gaps + list(range(wi, j)))
            else:
                for val in _role_candidates(slot.role, phrase, answer, lookup):
                    if answer.startswith(val, pos):
                        dfs(si + 1, j + nw, pos + len(val),
                            pieces + [(j, j + nw, slot.role, val, None, False)],
                            gaps + list(range(wi, j)))

    try:
        dfs(0, 0, 0, [], [])
    except _Stop:
        pass
    return completes, best[0]


def _piece_sources(wp_words, pieces):
    """Build the Source list + per-letter Links for a list of placed pieces."""
    sources, links, pos = [], [], 0
    for si, (a, b, role, value, ss, prov) in enumerate(pieces):
        toks = wp_words[a:b]
        mech = "homophone" if role == "HOM_F" else ROLE_MECHANISM[role]
        src = Source(clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                     text=" ".join(t.text for t in toks), value=value,
                     mechanism=mech, source=("pending" if prov else "db"))
        if role == "HOM_F" and prov:
            src.enrich_synonym = (" ".join(t.text for t in toks), ss)
        sources.append(src)
        transform = ('sounds like "%s"' % ss) if role == "HOM_F" else None
        for _ in value:
            pos += 1
            links.append(Link(answer_pos=pos, source_index=si,
                              operation="charade_homophone", clue_atom_id=None,
                              transform=transform))
    return sources, links


def _indicator_annotation(wp_words, ind_set):
    if not ind_set:
        return None
    toks = [wp_words[k] for k in sorted(ind_set)]
    return Annotation(clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                      text=" ".join(t.text for t in toks), role="indicator",
                      note="homophone indicator")


def _build(ctx, answer, words, def_idx, def_source, wp_words, placement, template,
           ind_set):
    """Assemble a complete Parse from a placement + its residue definition."""
    definition = Source(
        clue_atom_ids=tuple(aid for i in def_idx for aid in words[i].atom_ids),
        text=" ".join(words[i].text for i in def_idx),
        value=ctx.answer_text, mechanism="definition", source=def_source)
    sources, links = _piece_sources(wp_words, placement["pieces"])
    annotations = []
    ind = _indicator_annotation(wp_words, ind_set)
    if ind is not None:
        annotations.append(ind)
    for k in placement["links"]:
        annotations.append(Annotation(clue_atom_ids=wp_words[k].atom_ids,
                                      text=wp_words[k].text, role="link",
                                      note="link word"))
    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="charade_homophone",
                  solved_by="catalog")
    parse.template_id = template.id
    parse.matched_signature = template.signature
    _verify(ctx, parse)
    return parse


def _build_partial(ctx, answer, wp_words, ind_set, pieces):
    """The most-pieces PARTIAL kept when nothing parsed completely: show the pieces that
    DID place (a prefix of the answer) and the indicator, name the unresolved tail, and
    assert NO definition and NO link roles by elimination. Evidence, not a solve."""
    sources, links = _piece_sources(wp_words, pieces)
    annotations = []
    ind = _indicator_annotation(wp_words, ind_set)
    if ind is not None:
        annotations.append(ind)
    covered = len(links)
    remaining = answer[covered:]
    warnings = ["no complete charade+homophone parse; best partial shown — pieces "
                "account for %d of %d answer letters, '%s' and the definition unresolved"
                % (covered, len(answer), remaining)]
    return Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                 sources=sources, links=links, annotations=annotations,
                 definition=None, operation="charade_homophone", solved_by="catalog",
                 status="fail", warnings=warnings)


def _verify(ctx, parse):
    """Three-state verdict: pass / pending (provisional residue def or piece) / fail."""
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
        warnings.append("the definition is the leftover edge but is not DB-confirmed "
                        "(queued for enrichment)")
    if any(getattr(s, "source", "db") == "pending" for s in parse.sources):
        warnings.append("a homophone piece is provisional (queued for enrichment)")
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


def _rank(parse):
    """The ONE selector — most pieces: a DB-confirmed definition first, then the most
    answer letters from DB-confirmed pieces, then fewest leftover links, then fewest
    provisional pieces. Higher is better."""
    dbdef = 1 if (parse.definition is not None
                  and getattr(parse.definition, "source", "db") == "db") else 0
    confirmed = sum(len(s.value) for s in parse.sources
                    if s.mechanism != "definition"
                    and getattr(s, "source", "db") != "pending")
    links = sum(1 for a in parse.annotations if a.role == "link")
    prov = sum(1 for s in parse.sources if getattr(s, "source", "db") == "pending")
    return (dbdef, confirmed, -links, -prov)


def _collect(ctx, answer, words, ind_set, postags, templates, defines, lookup, is_link,
             sounds_alike, synonyms_of, suggest_hom, stop_first=False):
    """Every signature x edge split: solve the wordplay, validate the residue
    definition. Returns (complete_parses, best_partial) where best_partial =
    (covered, n_pieces, wp_words, ind_set_local, pieces)."""
    n = len(words)
    completes = []
    gpartial = None
    for template in templates:
        edge = template.def_pos
        for d in range(1, n):
            if edge == "start":
                def_idx, wp_idx = list(range(0, d)), list(range(d, n))
            else:
                def_idx, wp_idx = list(range(n - d, n)), list(range(0, n - d))
            if not wp_idx or (ind_set & set(def_idx)):
                continue
            if not (ind_set <= set(wp_idx)):
                continue
            wp_words = [words[i] for i in wp_idx]
            local_ind = {wp_idx.index(i) for i in ind_set}
            cps, part = _enumerate(template.slots, wp_words, answer, local_ind, lookup,
                                   is_link, sounds_alike, synonyms_of, suggest_hom,
                                   stop_first=stop_first)
            for placement in cps:
                def_phrase = " ".join(words[i].text for i in def_idx)
                if defines(def_phrase, answer):
                    ds = "db"
                elif _pos_coherent(def_idx, postags):
                    ds = "pending"
                else:
                    continue
                completes.append(_build(ctx, answer, words, def_idx, ds, wp_words,
                                        placement, template, local_ind))
            if stop_first and completes:
                return completes, gpartial
            if part is not None:
                cov, npc, pcs = part
                if gpartial is None or (cov, npc) > (gpartial[0], gpartial[1]):
                    gpartial = (cov, npc, wp_words, local_ind, pcs)
    return completes, gpartial


def solve_charade_homophone(ctx, defines, lookup, is_link, indicator_types,
                            templates, sounds_alike, synonyms_of,
                            define_fallback=None, is_dbe=None, suggest_hom=None):
    """Full solve. Pass 1 DB-only: among ALL complete parses, return the most-pieces
    winner. Pass 2 (only if Pass 1 found nothing, and a fallback is supplied): a missing
    synonym source resolved provisionally -> pending. Else the best PARTIAL by the same
    most-pieces measure; else None. `define_fallback`/`is_dbe` accepted for call-site
    compatibility, unused (this engine derives its own definition)."""
    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3 or not templates or indicator_types is None \
            or sounds_alike is None:
        return None
    words = [t for t in ctx.clue_tokens if t.kind == "word"]
    n = len(words)
    if n < 3:
        return None
    ind = engine_common.find_typed_run(words, list(range(n)), indicator_types,
                                       "homophone", min_length=1)
    if not ind:
        return None
    ind_set = set(ind)
    postags = grammar.pos_tags([t.text for t in words])

    completes, gpartial = _collect(ctx, answer, words, ind_set, postags, templates,
                                   defines, lookup, is_link, sounds_alike, synonyms_of,
                                   None)
    if completes:
        return max(completes, key=_rank)

    if suggest_hom is not None:
        cps2, _ = _collect(ctx, answer, words, ind_set, postags, templates, defines,
                           lookup, is_link, sounds_alike, synonyms_of, suggest_hom,
                           stop_first=True)
        if cps2:
            return max(cps2, key=_rank)

    if gpartial is not None:
        _, _, wp_words, local_ind, pieces = gpartial
        return _build_partial(ctx, answer, wp_words, local_ind, pieces)
    return None
