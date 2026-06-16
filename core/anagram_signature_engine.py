"""Anagram engine — CATALOG-DRIVEN with links-as-residue (design §4).

Signature-driven sibling of the charade signature engine, for pure anagrams. It walks
the mined anagram signatures (ANA_F fodder + optional ANA_I indicator, in various
orders/word-counts, def at one edge) in priority order and instantiates each:

  for each anagram signature (priority order), for each definition split at its edge:
    - PLACE the typed slots on the wordplay words in clue order, GAPS ALLOWED (a slot
      consumes its n_words consecutive words);
    - ANA_F slot: the fodder words, whose letters (as written, or contraction-stripped
      "Lionel's"->LIONEL) must ANAGRAM to the whole answer (sorted-equal, not a
      self-anagram, not an exact reversal);
    - ANA_I slot: the indicator words, which must be a CONFIRMED anagram indicator;
    - the GAP words are classified LAST: function/connective (is_link or POS function)
      -> link; anything else leaves the parse unaccounted and the placement is rejected.
      Links are NEVER pre-stripped.

Among passing placements, prefer the FEWEST residue (link) words, then signature
priority. Records the matched signature (template_id) — design §10. The definition is
the shared stage (find_definitions); the engine only explains the wordplay.

ISOLATED per the 13-engines decision: own assembly + own verifier; the colour/letter
assignment mirrors the evidence anagram engine (per-fodder-word colour). Pure and
DB-decoupled.
"""

from collections import Counter
from itertools import combinations

from core import contractions, grammar
from core.wordplay import GLUE_POS, FUNCTION_POS, raw, is_anagram_indicator
from core.wfw_model import Source, Link, Annotation, Parse


def _fodder_forms(words, a, b):
    """The fodder letters of words[a:b], as written and contraction-stripped."""
    as_written = "".join(raw(words[k].text) for k in range(a, b))
    stripped = "".join(raw(contractions.strip_suffixes(words[k].text))
                       for k in range(a, b))
    return [f for f in (as_written, stripped) if f]


def _place(slots, words, answer, postags, is_link, indicator_types):
    """Place ANA_F / ANA_I slots on the words (gaps allowed); classify gaps as links
    LAST. Returns {fodder:(a,b,letters), indicator:[idx], links:[idx]} or None."""
    n, N = len(words), len(answer)
    nslots = len(slots)

    def residue_link(k):
        return (is_link and is_link(words[k].text))

    def is_ind_word(k):
        return is_anagram_indicator(words[k].text, indicator_types)

    def finalize(placed, gap_idxs):
        fodder = next((p for p in placed if p[2] == "ANA_F"), None)
        if fodder is None:
            return None
        a, b, _ = fodder
        # INTERIOR-LINK EXCLUSION: a link word inside the fodder span may be a joiner
        # whose letters are NOT fodder ("tea AND chips" -> TEACHIPS). Try excluding
        # subsets of interior link words, FEWEST exclusions first (so a link that IS
        # fodder is kept by default); the exact letter-match decides. The excluded
        # words become links (residue), never silently dropped.
        span = list(range(a, b))
        link_in = [k for k in span if is_link and is_link(words[k].text)]
        kept_idx, letters, excl_idx = None, None, []
        for r in range(0, len(link_in) + 1):
            for excl in combinations(link_in, r):
                es = set(excl)
                kept = [k for k in span if k not in es]
                if not kept:
                    continue
                aw = "".join(raw(words[k].text) for k in kept)
                st = "".join(raw(contractions.strip_suffixes(words[k].text))
                             for k in kept)
                f = next((x for x in (aw, st) if len(x) == N
                          and sorted(x) == sorted(answer) and x[::-1] != answer),
                         None)
                if f is not None:
                    kept_idx, letters, excl_idx = kept, f, list(excl)
                    break
            if kept_idx is not None:
                break
        if kept_idx is None:
            return None                              # fodder does not anagram answer
        ind_slots = [p for p in placed if p[2] == "ANA_I"]
        indicator = []
        for (ia, ib, _) in ind_slots:
            # Confirm the indicator at the word OR phrase level — a multi-word indicator
            # ("could make") is not a per-word indicator, only as the whole phrase.
            if not (any(is_ind_word(k) for k in range(ia, ib))
                    or is_anagram_indicator(
                        " ".join(words[k].text for k in range(ia, ib)), indicator_types)):
                return None                          # ANA_I run not a confirmed indicator
            indicator.extend(range(ia, ib))
        links = []
        for k in list(gap_idxs) + excl_idx:
            if residue_link(k):
                links.append(k)
            else:
                return None                          # content word unaccounted
        return {"fodder_idx": kept_idx, "letters": letters,
                "indicator": sorted(indicator), "links": sorted(links),
                "ind_confirmed": True}

    def dfs(si, wi, placed, gaps):
        if si == nslots:
            return finalize(placed, gaps + list(range(wi, n)))
        nw = slots[si].n_words
        role = slots[si].role
        for j in range(wi, n - nw + 1):
            r = dfs(si + 1, j + nw, placed + [(j, j + nw, role)],
                    gaps + list(range(wi, j)))
            if r:
                return r
        return None

    return dfs(0, 0, [], [])


def _try_template(ctx, answer, template, split, words, postags, is_link,
                  indicator_types):
    if split.where != template.def_pos:
        return None
    roles = {s.role for s in template.slots}
    if not roles <= {"ANA_F", "ANA_I"} or "ANA_F" not in roles:
        return None
    if sum(s.n_words for s in template.slots) > len(words):
        return None
    placement = _place(template.slots, words, answer, postags, is_link,
                       indicator_types)
    if placement is None:
        return None
    return _build(ctx, answer, split, words, placement, template)


def _build(ctx, answer, split, words, placement, template):
    """Assemble the Parse: one Source per fodder word (colour), each answer letter
    attributed to a fodder word that supplied it; indicator + links as annotations."""
    from core.definition_engine import dbe_annotation
    letters = placement["letters"]
    fodder_tokens = [words[k] for k in placement["fodder_idx"]]
    # per-fodder-word letters (split the matched letters back across the words by the
    # same as-written/stripped choice that produced `letters`)
    per_word, used = [], 0
    for t in fodder_tokens:
        w_letters = raw(t.text)
        if used + len(w_letters) > len(letters) or \
                letters[used:used + len(w_letters)] != w_letters:
            w_letters = raw(contractions.strip_suffixes(t.text))
        per_word.append(w_letters)
        used += len(w_letters)

    sources, remaining = [], []
    for t, wl in zip(fodder_tokens, per_word):
        remaining.append([len(sources), Counter(wl)])
        sources.append(Source(clue_atom_ids=t.atom_ids, text=t.text, value=wl,
                              mechanism="anagram_fodder"))
    links = []
    for pos_i, ch in enumerate(answer, start=1):
        si = 0
        for entry in remaining:
            if entry[1].get(ch, 0) > 0:
                entry[1][ch] -= 1
                si = entry[0]
                break
        links.append(Link(answer_pos=pos_i, source_index=si, operation="anagram",
                          clue_atom_id=None, transform="anagram_of"))

    ind_toks = [words[k] for k in placement["indicator"]]
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    ind_confirmed = placement.get("ind_confirmed", True)
    annotations = []
    if ind_toks:
        annotations.append(Annotation(
            clue_atom_ids=tuple(aid for t in ind_toks for aid in t.atom_ids),
            text=" ".join(t.text for t in ind_toks), role="indicator",
            note="anagram indicator", source="db" if ind_confirmed else "pending"))
    for k in placement["links"]:
        annotations.append(Annotation(clue_atom_ids=words[k].atom_ids,
                                      text=words[k].text, role="link",
                                      note="link word"))
    dbe = dbe_annotation(split)
    if dbe is not None:
        annotations.append(dbe)

    parse = Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                  sources=sources, links=links, annotations=annotations,
                  definition=definition, operation="anagram", solved_by="catalog")
    parse.template_id = template.id if template is not None else None
    parse.matched_signature = template.signature if template is not None else None
    _verify(ctx, parse, bool(ind_toks), ind_confirmed)
    return parse


def _verify(ctx, parse, has_indicator, ind_confirmed=True):
    warnings = []
    if not parse.is_complete():
        warnings.append("the answer letters are not fully covered by the fodder")
    missing = parse.unexplained_words(ctx)
    if missing:
        warnings.append("these clue words are unaccounted for: "
                        + ", ".join(repr(m) for m in missing))
    if not has_indicator:
        warnings.append("no anagram indicator found")
    elif not ind_confirmed:
        warnings.append("the anagram indicator is provisional (queued for enrichment)")
    if parse.definition is None:
        warnings.append("no definition found")
    elif getattr(parse.definition, "source", "db") == "pending":
        warnings.append("the definition is provisional (queued for enrichment)")
    parse.warnings = warnings
    if not warnings:
        parse.status = "pass"
    elif missing or not has_indicator or not parse.is_complete():
        parse.status = "fail"
    else:
        parse.status = "pending"


def _find_fodder(words, answer, is_link):
    """The LONGEST contiguous word run whose letters (with interior link words optionally
    excluded) anagram the WHOLE answer. Returns (a, b, fodder_idx, letters, excl) or None.
    This is the certain anchor — the wordplay is whatever spells the answer."""
    n, N = len(words), len(answer)
    best = None
    for a in range(n):
        for b in range(a + 1, n + 1):
            span = list(range(a, b))
            link_in = [k for k in span if is_link and is_link(words[k].text)]
            for r in range(0, len(link_in) + 1):
                hit = None
                for excl in combinations(link_in, r):
                    kept = [k for k in span if k not in set(excl)]
                    if not kept:
                        continue
                    aw = "".join(raw(words[k].text) for k in kept)
                    st = "".join(raw(contractions.strip_suffixes(words[k].text))
                                 for k in kept)
                    f = next((x for x in (aw, st) if len(x) == N
                              and sorted(x) == sorted(answer) and x[::-1] != answer), None)
                    if f is not None:
                        hit = (a, b, kept, f, list(excl))
                        break
                if hit:
                    if best is None or len(hit[2]) > len(best[2]):
                        best = hit
                    break
    return best


def _fodder_anchored(ctx, answer, words, postags, defines, is_link, indicator_types):
    """FODDER-ANCHORED anagram (the design's reliable path). The fodder is certain (it
    anagrams the answer); the leftover splits into the indicator (adjacent to the fodder)
    and the definition (far edge), with spaCy POS peeling FUNCTION words off both as
    links so neither swallows a link word. A DB anagram indicator is used as-is; an
    unknown one is SUGGESTED provisionally. A DB-confirmed definition + indicator pass;
    anything provisional is pending (queued). No Haiku — the structure is deterministic."""
    n = len(words)

    def is_fn(k):
        return (is_link and is_link(words[k].text))

    def is_ind(k):
        return is_anagram_indicator(words[k].text, indicator_types)

    def peel(idxs):
        """Drop OUTER function words (links); keep the inner content core."""
        lo, hi, links = 0, len(idxs) - 1, []
        while lo <= hi and is_fn(idxs[lo]):
            links.append(idxs[lo]); lo += 1
        while hi >= lo and is_fn(idxs[hi]):
            links.append(idxs[hi]); hi -= 1
        return idxs[lo:hi + 1], links

    fod = _find_fodder(words, answer, is_link)
    if fod is None:
        return None
    fa, fb, fodder_idx, letters, excl = fod
    left, right = list(range(0, fa)), list(range(fb, n))
    links = list(excl)

    def_idx, ind_idx = None, None
    if left and right:                       # fodder in the middle: one side each
        lcore, ll = peel(left); rcore, rl = peel(right); links += ll + rl
        ldef = bool(lcore) and defines(" ".join(words[k].text for k in lcore), answer)
        rdef = bool(rcore) and defines(" ".join(words[k].text for k in rcore), answer)
        if rdef and not ldef:
            def_idx, ind_idx = rcore, lcore
        elif ldef and not rdef:
            def_idx, ind_idx = lcore, rcore
        else:
            # Fodder in the middle and NEITHER side is a DB definition: we cannot tell
            # which flank is the definition and which is the indicator without guessing
            # (a link like "causing" can look like an indicator). Don't guess — abstain.
            return None
    else:                                    # fodder at an edge: one leftover run holds both
        # The leftover (in clue order) holds the definition at the FAR edge from the fodder
        # and the indicator adjacent to it. Place the boundary using the DEFINITION LOOKUP
        # (the strong signal we already have) rather than a function-word boundary that may
        # not exist — e.g. "Rugby leader possibly | could make" is all content words, so the
        # old link-boundary scan swallowed everything into the indicator and left no
        # definition. Search boundaries (LONGEST definition first) for one whose far side is
        # a confirmed definition; the near side is then the indicator (provisional if the DB
        # does not confirm it).
        side = right or left                 # leftover run, in clue order
        fodder_at_left = not left            # fodder precedes the leftover -> ind is nearest
        chosen = None
        for d in range(len(side) - 1, 0, -1):    # definition length, longest first
            if fodder_at_left:               # def at far end (suffix); ind = prefix
                def_part, ind_part = side[len(side) - d:], side[:len(side) - d]
            else:                            # def at far start (prefix); ind = suffix
                def_part, ind_part = side[:d], side[d:]
            dcore, dl = peel(def_part)
            icore, il = peel(ind_part)
            if not dcore or not icore:
                continue
            if not defines(" ".join(words[x].text for x in sorted(dcore)), answer):
                continue
            chosen = (dcore, icore, dl + il)
            break                            # longest confirmed definition wins
        if chosen is not None:
            dcore, icore, extra_links = chosen
            links += extra_links
            ind_idx, def_idx = icore, dcore
        else:
            # No definition-confirmed boundary: fall back to the original function-word
            # heuristic (a link-word boundary with a not-yet-DB-confirmed definition).
            from_fodder = side if right else list(reversed(side))
            i = 0
            while i < len(from_fodder) and is_fn(from_fodder[i]):
                links.append(from_fodder[i]); i += 1
            ind = []
            while i < len(from_fodder) and not is_fn(from_fodder[i]):
                ind.append(from_fodder[i]); i += 1
            while i < len(from_fodder) and is_fn(from_fodder[i]):
                links.append(from_fodder[i]); i += 1
            rest = from_fodder[i:]
            dcore, dl = peel(rest); links += dl
            ind_idx, def_idx = ind, dcore

    if not ind_idx or not def_idx:
        return None
    ind_confirmed = (any(is_ind(k) for k in ind_idx)
                     or is_anagram_indicator(
                         " ".join(words[k].text for k in sorted(ind_idx)), indicator_types))
    def_idx = sorted(def_idx)
    def_phrase = " ".join(words[k].text for k in def_idx)
    def_confirmed = defines(def_phrase, answer)

    placement = {"fodder_idx": fodder_idx, "letters": letters,
                 "indicator": sorted(ind_idx), "links": sorted(set(links)),
                 "ind_confirmed": ind_confirmed}
    from core.definition_engine import DefinitionSplit
    split = DefinitionSplit(
        phrase=def_phrase, where="end",
        def_atom_ids=tuple(aid for k in def_idx for aid in words[k].atom_ids),
        def_tokens=[words[k] for k in def_idx],
        wordplay_tokens=[words[k] for k in range(n) if k not in def_idx],
        source="db" if def_confirmed else "pending")
    return _build(ctx, answer, split, words, placement, None)


def solve_anagram(ctx, defines, is_link, indicator_types, templates,
                  define_fallback=None, is_dbe=None):
    """Full anagram solve — catalog-driven. Returns the first clean PASS (fewest
    residue), else the best parse, else None."""
    from core.definition_engine import find_definitions

    answer = "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")
    if len(answer) < 3 or not templates:
        return None
    splits = list(find_definitions(ctx, defines, define_fallback=define_fallback,
                                   is_dbe=is_dbe))
    if not splits:
        return None
    from core.definition_engine import _split_from_indices
    allwords = [t for t in ctx.clue_tokens if t.kind == "word"]
    alltexts = [t.text for t in allwords]
    prepared = []
    for split in splits:
        # Keep a grammatically-bound multi-word definition WHOLE (spaCy subtree) before
        # anything is split off as wordplay — so "Avoided dealing with" stays the
        # definition instead of leaking "dealing"/"with" into the indicator/links.
        def_atoms = set(split.def_atom_ids)
        def_idx = {i for i, t in enumerate(allwords)
                   if any(aid in def_atoms for aid in t.atom_ids)}
        grown = grammar.phrase_extent(alltexts, def_idx)
        # The extent is GRAMMATICAL, so it would absorb a word that modifies the
        # definition seed even when that word is the ANAGRAM INDICATOR (e.g. "running
        # shoes": "running" modifies "shoes" but IS the indicator). Never let the
        # definition swallow this engine's own operator — keep it for the wordplay.
        # (Seed words in def_idx are always retained.)
        def _is_anag_ind(i):
            try:
                return "anagram" in (indicator_types(allwords[i].text) or set())
            except Exception:
                return False
        if grown:
            grown = {i for i in grown if i in def_idx or not _is_anag_ind(i)}
        # A definition is a CONTIGUOUS span. If removing an INTERIOR anagram indicator
        # from the grown extent left a gap (e.g. "no top [generating] urges" -> the
        # non-contiguous "no top urges"), the grow is not a valid definition — keep the
        # original split rather than emit a non-contiguous definition.
        contiguous = bool(grown) and grown == set(range(min(grown), max(grown) + 1))
        if grown and contiguous and grown != def_idx:
            split = _split_from_indices(allwords, grown, split.where,
                                        source=split.source)
        words = [t for t in split.wordplay_tokens if t.kind == "word"]
        if len(words) < 1:
            continue
        postags = grammar.wordplay_pos_tags(ctx, words)
        prepared.append((split, words, postags))
    if not prepared:
        return None

    best_pass, best_key, best_other = None, None, None
    for template in templates:                       # priority order
        for split, words, postags in prepared:
            parse = _try_template(ctx, answer, template, split, words, postags,
                                  is_link, indicator_types)
            if parse is None:
                continue
            if parse.status == "pass":
                residue = sum(1 for a in parse.annotations if a.role == "link")
                if best_key is None or residue < best_key:
                    best_pass, best_key = parse, residue
                    if residue == 0:
                        return parse
            elif best_other is None:
                best_other = parse
    if best_pass is not None:
        return best_pass
    # FODDER-ANCHORED fallback: the fodder is certain (it anagrams the answer), so the
    # indicator (adjacent) and definition (far edge) fall out, with spaCy peeling link
    # words off both. Reliable where the deterministic slot-placement missed (e.g. the
    # indicator is not in the DB, or a spurious definition split misled it). Run over the
    # WHOLE clue (the definition is NOT pre-removed — it falls out here).
    allwords = [t for t in ctx.clue_tokens if t.kind == "word"]
    if len(allwords) >= 3:
        allpos = grammar.wordplay_pos_tags(ctx, allwords)
        fp = _fodder_anchored(ctx, answer, allwords, allpos, defines, is_link,
                              indicator_types)
        if fp is not None and fp.status in ("pass", "pending"):
            return fp
    # A DB-template non-pass (best_other) is only returned when its DEFINITION is
    # DB-confirmed — otherwise it is the noisy guess (wrong def/indicator) that the
    # fodder-anchored path deliberately abstained from, so prefer honest fail-evidence.
    if best_other is not None and \
            getattr(best_other.definition, "source", "db") == "db":
        return best_other
    # No signature instantiated. Do NOT return None and discard what was found —
    # preserve the evidence (design §2 / §5.9) so the gap is visible and the separate
    # signature-creation process has the fodder to work from.
    return _build_fail_evidence(ctx, answer, prepared[0][0])


def _build_fail_evidence(ctx, answer, split):
    """Preserve the evidence when no anagram signature instantiated. A FAIL asserts
    nothing about structure, so it assigns NO indicator/link roles by elimination
    (feedback-no-role-on-fail). It keeps the definition and surfaces the anagram-fodder
    candidate: the contiguous run of wordplay words whose letters anagram to the WHOLE
    answer (preferred), or failing that the longest run anagramming to a span — marked
    a candidate, not a committed piece."""
    from core.definition_engine import dbe_annotation
    definition = Source(clue_atom_ids=split.def_atom_ids, text=split.phrase,
                        value=ctx.answer_text, mechanism="definition",
                        source=split.source)
    words = [t for t in split.wordplay_tokens if t.kind == "word"]
    sources = []
    cand = _fodder_candidate(words, answer)
    if cand is not None:
        a, b, letters = cand
        toks = words[a:b]
        sources.append(Source(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks), value=letters,
            mechanism="anagram_fodder"))
    warnings = ["no anagram signature matched this clue "
                "(the fodder below is a candidate, not a placement)"]
    dbe = dbe_annotation(split)
    annotations = [dbe] if dbe is not None else []
    return Parse(clue_text=ctx.clue_text, answer_text=ctx.answer_text,
                 sources=sources, links=[], annotations=annotations,
                 definition=definition, operation="anagram", solved_by="catalog",
                 status="fail", warnings=warnings)


def _fodder_candidate(words, answer):
    """The contiguous run of wordplay words whose letters anagram to the WHOLE answer
    (preferred, longest such run), else the longest run anagramming to a span of the
    answer (sorted-equal, not an exact reversal). Returns (start, end, letters) or
    None. Pure evidence — no placement asserted."""
    n, N = len(words), len(answer)
    whole, span_best = None, None
    for a in range(n):
        for b in range(a + 1, n + 1):
            for fl in _fodder_forms(words, a, b):
                L = len(fl)
                if L < 3 or L > N:
                    continue
                key = sorted(fl)
                if L == N and key == sorted(answer) and answer[::-1] != fl:
                    if whole is None or (b - a) > (whole[1] - whole[0]):
                        whole = (a, b, answer)
                    continue
                for start in range(0, N - L + 1):
                    sp = answer[start:start + L]
                    if sorted(sp) == key and sp[::-1] != fl:
                        if span_best is None or L > len(span_best[2]):
                            span_best = (a, b, sp)
                        break
    return whole or span_best
