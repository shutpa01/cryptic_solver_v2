"""Hidden-word engine — built on the shared character atomiser.

A hidden clue conceals the answer in a run of consecutive clue letters, e.g.
de[BRIE]fs -> BRIE. This engine finds that run and records, for EVERY answer
letter, the EXACT clue character it came from (the per-character provenance the
whole redesign exists to produce). Forward and reversed are both handled.

Pure: takes a wfw_atoms context, returns a wfw_model.Parse or None. Definition
and indicator are separate shared concerns; this engine reports the wordplay and
the host word(s), and accepts optional annotations from the caller.

Paired with core/hidden_screen.py — its own bespoke screen, built together with
this engine (per the per-clue-type UI rule). No universal renderer.
"""

from core.wfw_model import Source, Link, Parse
from core import engine_common


def _letter_atoms(atoms):
    """The (normalized_char, atom_id) stream of just the letter atoms."""
    return [(a.normalized, a.atom_id) for a in atoms if a.kind == "letter"]


def find_hidden(ctx, annotations=None, definition=None, search_atom_ids=None):
    """Find the answer hidden in the clue's letter stream.

    Returns a Parse whose links map each answer letter to the exact clue
    character atom that produced it, or None if the answer is not hidden.

    `search_atom_ids`, when given, restricts the host search to those clue
    atoms — the wordplay region handed over by the definition engine — so the
    answer is never found hiding inside its own definition words.
    """
    ans_stream = _letter_atoms(ctx.answer_atoms)
    target = "".join(ch for ch, _ in ans_stream)
    if len(target) < 3:
        return None

    clue_stream = _letter_atoms(ctx.clue_atoms)
    if search_atom_ids is not None:
        allowed = set(search_atom_ids)
        clue_stream = [(ch, aid) for ch, aid in clue_stream if aid in allowed]
    clue_chars = "".join(ch for ch, _ in clue_stream)

    for reverse in (False, True):
        needle = target[::-1] if reverse else target
        start = clue_chars.find(needle)
        while start != -1:
            end = start + len(needle)
            window = clue_stream[start:end]          # (char, atom_id) run
            if _is_valid_hidden(start, end, clue_chars, window, ctx):
                return _build(ctx, ans_stream, window, reverse,
                              annotations, definition)
            start = clue_chars.find(needle, start + 1)
    return None


def _is_valid_hidden(start, end, clue_chars, window, ctx):
    """Reject the two non-hidden cases: the whole clue, or one whole word."""
    if start == 0 and end == len(clue_chars):
        return False                                  # the entire clue
    host_tokens = _host_tokens(ctx, [aid for _, aid in window])
    if len(host_tokens) == 1:
        tok = host_tokens[0]
        tok_letters = sum(1 for aid in tok.atom_ids
                          if _atom_kind(ctx, aid) == "letter")
        if len(window) == tok_letters:
            return False                              # a whole single word
    return True


def _atom_kind(ctx, atom_id):
    for a in ctx.clue_atoms:
        if a.atom_id == atom_id:
            return a.kind
    return None


def _host_tokens(ctx, atom_ids):
    """The clue tokens (words) that the hidden run touches, in order."""
    wanted = set(atom_ids)
    return [t for t in ctx.clue_tokens
            if t.kind == "word" and wanted.intersection(t.atom_ids)]


def _build(ctx, ans_stream, window, reverse, annotations, definition):
    # window is the clue letters that spell the answer, in CLUE order. When the
    # answer is reversed, the first answer letter comes from the LAST window
    # character, so align by reversing the window for the mapping.
    mapping = list(reversed(window)) if reverse else list(window)

    host_tokens = _host_tokens(ctx, [aid for _, aid in window])
    host_text = " ".join(t.text for t in host_tokens)
    host_atom_ids = tuple(aid for t in host_tokens for aid in t.atom_ids)

    target = "".join(ch for ch, _ in ans_stream)
    source = Source(clue_atom_ids=host_atom_ids, text=host_text,
                    value=target,
                    mechanism="hidden_reversed" if reverse else "hidden")

    links = []
    for pos, ((_, _src_aid), (ans_char, _)) in enumerate(zip(mapping, ans_stream), start=1):
        links.append(Link(answer_pos=pos, source_index=0, operation="hidden",
                          clue_atom_id=mapping[pos - 1][1],
                          transform="reversed" if reverse else None))

    return Parse(
        clue_text=ctx.clue_text, answer_text=ctx.answer_text,
        sources=[source], links=links,
        annotations=list(annotations or []),
        definition=definition,
        operation="hidden_reversed" if reverse else "hidden",
        solved_by="hidden")


def solve_hidden(ctx, defines, indicator_types=None, is_link=None,
                 define_fallback=None):
    """Full hidden solve.

    Hidden is one of the simplest clue types, so the HIDDEN ELEMENT itself —
    the answer appearing as a run of consecutive clue letters — is what
    identifies the clue as hidden. We do NOT lead with an indicator. Order:

    1. Find the hidden host run (the trigger). If there is none, it is not a
       hidden clue.
    2. Identify the definition: a contiguous edge run the def engine confirms
       (universal, shared), or — when no DB definition exists but the indicator is
       pinned — the leftover edge run (edge-anchored, provisional).
    3. The indicator is the grammatically-bound PHRASE: when the region left after
       host + definition is a single contiguous run, that whole run is the
       indicator (DB-confirmed if it contains a known hidden indicator, else
       provisional + queued). Multi-run regions fall back to per-token indicator
       recognition, with link words classified LAST and only once the operation is
       complete (host + definition + indicator all present). Anything still
       unaccounted is surfaced, never relabelled by elimination.

    Injected, DB-decoupled:
      defines(phrase, answer_letters) -> bool
      indicator_types(word) -> set of wordplay-type strings   (optional)
      is_link(word) -> bool                                    (optional)
      define_fallback(ctx) -> DefinitionSplit | None           (optional)
          The AI definition fallback, consulted ONLY when the DB defines nothing.
          Its split is flagged source='pending' so the caller can queue it for
          verification and the screen can badge it provisional.

    Returns a Parse or None.
    """
    from core.definition_engine import find_definitions
    from core.wfw_model import Source

    # (2) candidate definitions, longest first; plus the no-definition option so
    # a missing definition never blocks a genuine hidden find. The host search
    # is restricted to the wordplay region so the answer is never found hiding
    # inside its own definition words.
    db_splits = list(find_definitions(ctx, defines))
    candidates = list(db_splits)
    if not db_splits and define_fallback is not None:
        fb = define_fallback(ctx)        # Haiku fallback only on a DB miss
        if fb is not None:
            candidates.append(fb)
    candidates.append(None)
    best_fail = None
    for split in candidates:
        if split is None:
            parse = find_hidden(ctx)                       # (1) trigger, whole clue
        else:
            definition = Source(clue_atom_ids=split.def_atom_ids,
                                text=split.phrase, value=ctx.answer_text,
                                mechanism="definition", source=split.source)
            parse = find_hidden(ctx, definition=definition,
                                search_atom_ids=split.wordplay_atom_ids)
        if parse is None:
            continue
        # Indicator. Recognise the LONGEST DB-known phrase among the leftovers
        # first (a multi-word indicator like "a little" the table holds), then the
        # per-token pass for anything else. No gating: whatever the indicators
        # table knows is used — most-specific (longest) match wins, exactly as
        # definition recognition already works.
        _classify_multiword_indicator(ctx, parse, indicator_types)
        _classify_indicator(ctx, parse, indicator_types, is_link)
        # The indicator is the grammatically-bound PHRASE, not a fragment. Once the
        # definition is known, the region left after host + definition, when it is a
        # single contiguous run, IS the indicator as a whole ("Contents of", "It's
        # part of", "confined by") — DB-confirmed if it contains a known hidden
        # indicator, otherwise provisional and queued for enrichment. Multi-run
        # regions keep the per-token indicator/link split (e.g. TIGER).
        if parse.definition is not None:
            _consolidate_single_run_indicator(ctx, parse, indicator_types)
        # Grow a DB-confirmed definition to its full grammatical extent, knowing the
        # wordplay (host + indicator), so bound function words ("in the") join the
        # definition instead of being stranded (ANDEAN -> "in the mountains").
        if split is not None and parse.definition is not None:
            _extend_parse_definition(ctx, parse, split)
        # REVERSED hidden ONLY: account the reversal indicator ("up", "elevated", "on
        # reflection") that the hidden-only passes leave stranded. Gated to the reversed
        # operation so forward hidden is byte-for-byte unchanged; runs before the edge-
        # definition and link steps so a multi-word reversal phrase is taken whole.
        if parse.operation == "hidden_reversed":
            _classify_reversal_indicator(ctx, parse, indicator_types)
        # Edge-anchored definition: indicator pinned but no DB definition -> the
        # single contiguous run of real leftover words at a clue EDGE IS the
        # definition (provisional, queued); then re-consolidate the indicator.
        if parse.definition is None and any(a.role == "indicator"
                                            for a in parse.annotations):
            _maybe_add_edge_definition(ctx, parse)
            if parse.definition is not None:
                _consolidate_single_run_indicator(ctx, parse, indicator_types)
        # LINK WORDS — assigned ONLY when the operation is COMPLETE: host +
        # definition + indicator all present. A link word is pure residue; if the
        # definition OR the indicator is missing we assign NO links and surface the
        # gap honestly (the rule set at the start).
        if (parse.definition is not None
                and any(a.role == "indicator" for a in parse.annotations)):
            _classify_links(ctx, parse, is_link)
        # Verify against the hidden-specific rules; a clean PASS wins immediately.
        _verify_hidden(ctx, parse)
        if parse.status == "pass":
            return parse
        if best_fail is None:
            best_fail = parse
    # No candidate was a clean PASS: return the best PENDING (candidates are tried
    # best-first, so this is the most-explained parse — run found, with whatever
    # piece is still queued for enrichment clearly warned). Hidden never fails.
    return best_fail


def _leftover_word_indices(ctx, parse):
    """Word positions not yet accounted for by the host, the definition, or an
    existing annotation — the only place a fallback indicator may be carved."""
    words = [t for t in ctx.clue_tokens if t.kind == "word"]
    accounted = engine_common.accounted_atom_ids(parse)
    return {i for i, t in enumerate(words)
            if not any(aid in accounted for aid in t.atom_ids)}


def _classify_multiword_indicator(ctx, parse, indicator_types):
    """Recognise a MULTI-WORD hidden indicator the DB knows as a phrase (e.g. an
    enriched "a little"). Scans contiguous runs of leftover words, LONGEST first,
    and attaches the first whose joined phrase the DB types as 'hidden'. Runs
    before the per-token pass so the most-specific known phrase wins; brings
    indicator recognition up to the multi-word standard definitions already enjoy.
    A no-op when no leftover run is a known phrase, so single-word clues are
    handled by the per-token pass exactly as before."""
    if indicator_types is None:
        return
    from core.wfw_model import Annotation
    words = [t for t in ctx.clue_tokens if t.kind == "word"]
    accounted = engine_common.accounted_atom_ids(parse)
    leftset = {i for i, t in enumerate(words)
               if not any(aid in accounted for aid in t.atom_ids)}
    n = len(words)
    for length in range(len(leftset), 1, -1):          # multi-word only (>=2)
        for start in range(0, n - length + 1):
            run = list(range(start, start + length))
            if not all(i in leftset for i in run):
                continue
            phrase = " ".join(words[i].text for i in run)
            if "hidden" in (indicator_types(phrase) or set()):
                toks = [words[i] for i in run]
                parse.annotations.append(Annotation(
                    clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                    text=" ".join(t.text for t in toks),
                    role="indicator", note="hidden indicator"))
                return


def _classify_reversal_indicator(ctx, parse, indicator_types):
    """A REVERSED hidden (operation 'hidden_reversed') carries a REVERSAL indicator
    ("up", "elevated", "on reflection", "in retreat") on top of the hidden indicator.
    Every other indicator pass only recognises 'hidden'-typed words, so the reversal
    indicator is left unaccounted and the clue cannot pass — and adding it to the DB
    did nothing because nothing queried 'reversal'. This recogniser fills that gap.

    Scans contiguous runs of STILL-UNACCOUNTED words, LONGEST first, and attaches the
    first whose joined phrase the DB types as 'reversal' (so 'on reflection' is taken
    whole, not split). Additive and GATED to the reversed case by the caller — forward
    hidden never reaches here, so its behaviour is unchanged. Runs BEFORE link
    classification so the phrase's function words are still free to join it."""
    if indicator_types is None:
        return
    from core.wfw_model import Annotation
    words = [t for t in ctx.clue_tokens if t.kind == "word"]
    accounted = engine_common.accounted_atom_ids(parse)
    leftset = {i for i, t in enumerate(words)
               if not any(aid in accounted for aid in t.atom_ids)}
    if not leftset:
        return

    def is_rev(text):
        try:
            return "reversal" in (indicator_types(text) or set())
        except Exception:
            return False

    n = len(words)
    for length in range(len(leftset), 0, -1):
        for start in range(0, n - length + 1):
            run = list(range(start, start + length))
            if not all(i in leftset for i in run):
                continue
            phrase = " ".join(words[i].text for i in run)
            if is_rev(phrase):
                toks = [words[i] for i in run]
                parse.annotations.append(Annotation(
                    clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
                    text=phrase, role="indicator", note="reversal indicator"))
                return


def _consolidate_single_run_indicator(ctx, parse, indicator_types):
    """Record a DB-known multi-word hidden-indicator PHRASE as one annotation when the
    whole leftover run (after host + definition) is exactly that phrase — e.g. an
    enriched "contents of". It fires ONLY when the entire run's joined text is typed
    'hidden' in the DB; it does NOT grab the run because one word in it is a known
    indicator, and it does NOT invent a provisional indicator from an untyped run
    (both were role-by-elimination). When the run is not a known phrase it does
    nothing, leaving the per-token indicator pass (which grows across adjacent
    hidden-typed words) and the link/unaccounted classification to do the honest job.
    A multi-run region is left to the per-token pass."""
    from core.wfw_model import Annotation
    words = [t for t in ctx.clue_tokens if t.kind == "word"]
    host = set(parse.sources[0].clue_atom_ids)
    defids = set(parse.definition.clue_atom_ids) if parse.definition else set()
    region = [i for i, t in enumerate(words)
              if not any(aid in host for aid in t.atom_ids)
              and not any(aid in defids for aid in t.atom_ids)]
    if not region:
        return
    if region != list(range(region[0], region[-1] + 1)):
        return                                   # multi-run -> per-token handles it
    toks = [words[i] for i in region]
    phrase = " ".join(t.text for t in toks)

    # Only consolidate when the WHOLE leftover run is itself a DB-known hidden
    # indicator phrase. Never grab the run on the strength of a single known word,
    # and never invent an indicator from an untyped run — that is role-by-elimination
    # (feedback-no-role-on-fail / links-from-list-only). When the phrase is not known,
    # do nothing: the per-token indicator pass (which already grows across adjacent
    # hidden-typed words) keeps the words the DB types, and the link / unaccounted
    # classification handles the rest honestly.
    try:
        if "hidden" not in (indicator_types(phrase) or set()):
            return
    except Exception:
        return

    region_atoms = {aid for t in toks for aid in t.atom_ids}
    parse.annotations = [a for a in parse.annotations
                         if not (a.role == "indicator"
                                 and any(aid in region_atoms for aid in a.clue_atom_ids))]
    parse.annotations.append(Annotation(
        clue_atom_ids=tuple(sorted(region_atoms)),
        text=phrase, role="indicator", note="hidden indicator", source="db"))


def _maybe_add_edge_definition(ctx, parse):
    """Edge-anchored definition. With the host and indicator pinned, a hidden clue
    is indicator + fodder + definition, so a single contiguous run of leftover real
    words sitting at a clue EDGE must be the definition. Take it as a provisional
    definition (queued for verification). Abstains when leftovers sit at BOTH edges
    or mid-clue — then we cannot disambiguate, so we never guess."""
    from core.wfw_model import Source
    words = [t for t in ctx.clue_tokens if t.kind == "word"]
    leftover = sorted(_leftover_word_indices(ctx, parse))
    if not leftover:
        return
    if leftover != list(range(leftover[0], leftover[-1] + 1)):
        return                                   # not a single contiguous run
    if leftover[0] != 0 and leftover[-1] != len(words) - 1:
        return                                   # not at an edge
    toks = [words[i] for i in leftover]
    parse.definition = Source(
        clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
        text=" ".join(t.text for t in toks), value=ctx.answer_text,
        mechanism="definition", source="pending")


def _verify_hidden(ctx, parse):
    """Engine-level verification — rules SPECIFIC to hidden, run inside the
    engine (no shared grand verifier).

    The cascade rule (2026-06-02): finding the answer as a contiguous run of clue
    letters is conclusive that the clue IS hidden, so hidden is TERMINAL whenever
    it fires and NEVER returns 'fail'. The verdict is therefore two-state here:

      - 'pass'    when everything is DB-confirmed: contiguous run, every clue word
                  accounted for, a DB-confirmed indicator, and a DB-confirmed
                  definition.
      - 'pending' when the run is found but a required piece is missing or only
                  provisional (a queued enrichment candidate): no/provisional
                  indicator, no/provisional definition, or unaccounted words.

    (A clue with no contiguous run never reaches here — find_hidden returns None
    and the cascade moves on; that absence is not a hidden 'fail'.)

    On a pending verdict we keep all the genuine pieces and explain what is queued.
    """
    warnings = []

    # 1. contiguity — the links must map to a single unbroken letter run.
    if not _is_contiguous_run(ctx, parse):
        warnings.append("the answer is not a single unbroken run of clue letters")

    # 2. every clue word accounted for.
    w_unaccounted = engine_common.unaccounted_words_warning(ctx, parse)
    if w_unaccounted:
        warnings.append(w_unaccounted)

    # 3. a hidden indicator present and DB-confirmed.
    indicators = [a for a in parse.annotations if a.role == "indicator"]
    if not indicators:
        warnings.append("no hidden indicator found")
    elif any(getattr(a, "source", "db") == "pending" for a in indicators):
        warnings.append("the hidden indicator is provisional (queued for enrichment)")

    # 4. a definition present and DB-confirmed.
    w_def = engine_common.definition_warning(parse)
    if w_def:
        warnings.append(w_def)

    # 5. remaining words must be known link words — already enforced by
    #    classification (anything not host/def/indicator/link is 'unaccounted',
    #    caught by rule 2), so no separate check is needed here.

    from core import role_validity
    warnings += role_validity.unbacked_roles(parse)   # indicator/links must be DB-backed
    parse.warnings = warnings
    # Hidden never fails: a clean parse passes, any queueable gap is pending.
    parse.status = "pass" if not warnings else "pending"


def _is_contiguous_run(ctx, parse):
    """True if the host links cover a single unbroken run in the clue's
    LETTER stream (spaces/punctuation between words are allowed; missing letters
    are not). Forward or reversed."""
    letter_index = {}
    pos = 0
    for a in ctx.clue_atoms:
        if a.kind == "letter":
            letter_index[a.atom_id] = pos
            pos += 1
    positions = [letter_index.get(l.clue_atom_id) for l in parse.links]
    if any(p is None for p in positions):
        return False
    positions.sort()
    return positions == list(range(positions[0], positions[0] + len(positions)))


def _extend_parse_definition(ctx, parse, split):
    """Grow parse.definition outward via the universal grammar-extent test,
    treating host + already-identified indicator words as the wordplay."""
    from core.definition_engine import extend_definition
    from core.wfw_model import Source
    used = set(parse.sources[0].clue_atom_ids)
    for a in parse.annotations:
        if a.role == "indicator":
            used.update(a.clue_atom_ids)
    grown = extend_definition(ctx, split, used)
    if grown.def_atom_ids != parse.definition.clue_atom_ids:
        parse.definition = Source(clue_atom_ids=grown.def_atom_ids,
                                  text=grown.phrase, value=ctx.answer_text,
                                  mechanism="definition",
                                  source=parse.definition.source)


def _classify_indicator(ctx, parse, indicator_types, is_link):
    """Identify the indicator phrase — the operative words — FIRST.

    - A word that is a hidden indicator and NOT also a link word is a genuine
      ('pure') hidden indicator.
    - A word that sits in BOTH lists (e.g. "in") joins the indicator when it is
      adjacent to a genuine hidden indicator, forming a contiguous phrase
      ("hidden in"). Requires a genuine indicator to be present.

    Link words are NOT touched here; they are considered only after the
    definition has been grown to its full grammatical extent.
    """
    from core.wfw_model import Annotation

    accounted = set(parse.sources[0].clue_atom_ids)
    if parse.definition:
        accounted.update(parse.definition.clue_atom_ids)

    words = [t for t in ctx.clue_tokens if t.kind == "word"]
    leftovers = [t for t in words
                 if not any(aid in accounted for aid in t.atom_ids)]
    if not leftovers:
        return

    def types_of(t):
        return set(indicator_types(t.text)) if indicator_types else set()

    def linky(t):
        return bool(is_link and is_link(t.text))

    hidden_capable = {t.index for t in leftovers if "hidden" in types_of(t)}
    pure_indicator = {t.index for t in leftovers
                      if "hidden" in types_of(t) and not linky(t)}

    indicator_idx = set(pure_indicator)
    if indicator_idx:
        changed = True
        while changed:
            changed = False
            for idx in list(indicator_idx):
                for adj in (idx - 1, idx + 1):
                    if adj in hidden_capable and adj not in indicator_idx:
                        indicator_idx.add(adj)
                        changed = True

    by_index = {t.index: t for t in leftovers}
    for group in engine_common.contiguous_groups(sorted(indicator_idx)):
        toks = [by_index[i] for i in group]
        parse.annotations.append(Annotation(
            clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
            text=" ".join(t.text for t in toks),
            role="indicator", note="hidden indicator"))


def _classify_links(ctx, parse, is_link):
    """After the definition is grown, classify whatever still remains: a link
    word if the data confirms it, otherwise left unexplained (surfaced honestly,
    never relabelled by elimination). Shared with every other engine via
    engine_common.classify_links (the unaccounted return is unused here — hidden
    surfaces a gap through verification, it does not abstain)."""
    engine_common.classify_links(ctx, parse, is_link)

