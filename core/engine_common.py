"""Shared engine primitives — the type-AGNOSTIC steps every isolated catalog
engine repeats, centralised so a fix lands in ONE place instead of N.

This is deliberately small and unopinionated. It holds ONLY the steps that are
genuinely identical across engines:

  - contiguous_groups   : group sorted ints into consecutive runs
  - accounted_atom_ids  : the clue atoms already explained by a parse
  - unaccounted_words_warning / definition_warning : the two completeness
        warnings whose wording is shared (each engine still ORDERS its own
        warning list and decides its own pass/pending/fail verdict)
  - classify_links      : turn confirmed link words into Link annotations and
        report any still-unaccounted content words (the caller decides what to
        do with them — surface, or abstain)
  - find_typed_run      : the longest contiguous run of candidate word positions
        whose joined phrase the DB types as a given wordplay type

It does NOT try to unify the genuinely type-specific logic: how an engine finds
its source (hidden run / acrostic letters / homophone source), how it grows its
indicator, or its verdict rules. Forcing those together would distort the
engines and re-create the blast-radius problem isolation was meant to solve.

Each function preserves the EXACT behaviour of the code it replaces; the
A/B harness (core/_ab_engine_refactor.py) proves byte-for-byte identical output.
"""
from core.wfw_model import Annotation


def contiguous_groups(indices):
    """Group a sorted list of ints into runs of consecutive values.
    [1,2,3,7,8] -> [[1,2,3],[7,8]]. Input is sorted by the caller."""
    groups, run = [], []
    for i in indices:
        if run and i == run[-1] + 1:
            run.append(i)
        else:
            if run:
                groups.append(run)
            run = [i]
    if run:
        groups.append(run)
    return groups


def accounted_atom_ids(parse):
    """The set of clue CharAtom ids already explained by this parse: every
    source word, the definition, and every existing annotation."""
    accounted = set()
    for s in parse.sources:
        accounted.update(s.clue_atom_ids)
    if parse.definition:
        accounted.update(parse.definition.clue_atom_ids)
    for a in parse.annotations:
        accounted.update(a.clue_atom_ids)
    return accounted


def unaccounted_words_warning(ctx, parse):
    """The shared 'unaccounted clue words' completeness warning, or None when
    every clue word has a role. Identical wording across engines."""
    missing = parse.unexplained_words(ctx)
    if not missing:
        return None
    return ("these clue words are unaccounted for: "
            + ", ".join(repr(m) for m in missing))


def definition_warning(parse):
    """The shared definition completeness warning, or None when the definition
    is present and DB-confirmed. Identical wording across engines."""
    if parse.definition is None:
        return "no definition found"
    src = getattr(parse.definition, "source", "db")
    if src == "pending":
        return "the definition is provisional (queued for enrichment)"
    if src == "andlit":
        return ("&lit: the whole clue is both the definition and the wordplay "
                "(confirm the definition reading)")
    return None


def better_near_miss(cur, new, ctx):
    """Pick the more-complete NEAR-MISS fail between two non-pass parses, so an answer-driven
    engine can surface its best partial (a complete assembly blocked only by leftover words)
    instead of silently abstaining. Ranks by fewest unaccounted clue words, then by whether the
    answer letters are fully covered. Used by the conservative compound engines that formerly
    returned None on an unclassified residue word (memory: near-miss reporting)."""
    if new is None:
        return cur
    if cur is None:
        return new
    try:
        n_new, n_cur = len(new.unexplained_words(ctx)), len(cur.unexplained_words(ctx))
        if n_new != n_cur:
            return new if n_new < n_cur else cur
        return new if (new.is_complete() and not cur.is_complete()) else cur
    except Exception:
        return cur


def classify_links(ctx, parse, is_link):
    """Classify the still-unaccounted clue words: a confirmed link word becomes a
    'link' Annotation; anything else is left for the caller. Returns the list of
    word tokens that are NEITHER accounted NOR confirmed links — the genuinely
    unaccounted content words. The caller decides whether to surface them (hidden:
    pending) or abstain (acrostic: not a clean solve).

    Links are taken ONLY from the link list (is_link), never by elimination — the
    binding rule, enforced in one place for every engine that calls this.
    """
    accounted = accounted_atom_ids(parse)
    unaccounted = []
    for t in (w for w in ctx.clue_tokens if w.kind == "word"):
        if any(aid in accounted for aid in t.atom_ids):
            continue
        if is_link and is_link(t.text):
            parse.annotations.append(Annotation(
                clue_atom_ids=t.atom_ids, text=t.text,
                role="link", note="link word"))
        else:
            unaccounted.append(t)
    return unaccounted


def typed_runs(word_tokens, positions, indicator_types, wptype, max_run=4):
    """Every contiguous sub-run of `positions` (indices into word_tokens) whose JOINED
    surface text the DB types as `wptype` — longest first, then earliest.

    The phrase-aware complement of a per-word check: a multi-word indicator ("picked
    up", "taken aback") is stored in the DB as ONE row, so testing words one at a time
    can never see it. That single-word defect silently killed otherwise-correct parses
    across the reversal family (2026-07-07, AROMA); residue/gate checks must use runs.

    `wptype` may be one type name or a set of acceptable type names (e.g. the container
    engines accept both 'container' and 'insertion')."""
    if indicator_types is None:
        return []
    wanted = {wptype} if isinstance(wptype, str) else set(wptype)

    def typed(text):
        try:
            return bool(wanted & (indicator_types(text) or set()))
        except Exception:
            return False

    out = []
    for group in contiguous_groups(sorted(positions)):
        for length in range(min(len(group), max_run), 0, -1):
            for s in range(0, len(group) - length + 1):
                seg = group[s:s + length]
                phrase = " ".join(word_tokens[i].text for i in seg)
                if typed(phrase):
                    out.append(list(seg))
    out.sort(key=lambda seg: (-len(seg), seg[0]))
    return out


def indicator_plus_links(word_tokens, residue, indicator_types, wptype, is_link):
    """Classify a residue as ONE DB-typed indicator (single word OR contiguous phrase)
    plus DB link words. Tries every typed run (longest first) and returns
    (indicator_positions, link_positions) for the first split where every remaining
    residue word is a DB link; None when no split works. Nothing is assigned by
    elimination — the indicator run must be DB-typed and the links must be DB links."""
    for run in typed_runs(word_tokens, residue, indicator_types, wptype):
        rest = [k for k in residue if k not in run]
        if all(bool(is_link and is_link(word_tokens[k].text)) for k in rest):
            return (run, rest)
    return None


def has_typed_indicator(word_tokens, indicator_types, wptype, max_run=4):
    """Phrase-aware gate: True when any single word OR contiguous phrase of
    `word_tokens` is DB-typed `wptype`. Replaces per-word any() gates, which miss an
    indicator stored only as a multi-word phrase."""
    return bool(typed_runs(word_tokens, range(len(word_tokens)), indicator_types,
                           wptype, max_run=max_run))


def classify_glue_run(word_tokens, run, indicator_types, wptype, is_link):
    """Segment one CONTIGUOUS leftover run into DB link words and (possibly multi-word)
    `wptype` indicator phrases. Returns [("indicator"|"link", [positions...]), ...] or
    None when the run cannot be fully segmented. Longest indicator phrase first at each
    step (with backtracking), so "picked up" is one indicator, not 'up' + a stranded
    'picked'. Strictly WIDER than the per-word glue check it replaces: every old
    single-word segmentation is still reachable."""
    if indicator_types is None:
        return None
    run = list(run)
    m = len(run)
    wanted = {wptype} if isinstance(wptype, str) else set(wptype)

    def typed(a, b):
        phrase = " ".join(word_tokens[run[k]].text for k in range(a, b))
        try:
            return bool(wanted & (indicator_types(phrase) or set()))
        except Exception:
            return False

    def seg(i):
        if i == m:
            return []
        for j in range(min(m, i + 4), i, -1):        # longest indicator phrase first
            if typed(i, j):
                rest = seg(j)
                if rest is not None:
                    return [("indicator", [run[k] for k in range(i, j)])] + rest
        if is_link and is_link(word_tokens[run[i]].text):
            rest = seg(i + 1)
            if rest is not None:
                return [("link", [run[i]])] + rest
        return None

    return seg(0)


def disjoint_typed_cover(word_tokens, positions, indicator_types, wptype):
    """A DISJOINT, longest-first selection of DB-typed runs within `positions` — for
    engines that treat EVERY typed leftover word as an indicator. Phrase runs are
    preferred, single typed words kept, and each returned run is DB-typed as a WHOLE
    (so an annotation per run is role_validity-safe; blindly merging adjacent separate
    indicators would fabricate an untyped phrase)."""
    chosen, taken = [], set()
    for run in typed_runs(word_tokens, positions, indicator_types, wptype):
        if not (set(run) & taken):
            chosen.append(run)
            taken.update(run)
    return chosen


def indicator_annotations(word_tokens, positions, note):
    """ONE Annotation per contiguous run of `positions`, carrying the JOINED phrase —
    so role_validity validates the DB row ("picked up"), never a component word.
    Replaces the per-word `for k in pl[...]` annotation loops."""
    out = []
    for grp in contiguous_groups(sorted(positions)):
        out.append(Annotation(
            clue_atom_ids=tuple(aid for k in grp for aid in word_tokens[k].atom_ids),
            text=" ".join(word_tokens[k].text for k in grp),
            role="indicator", note=note))
    return out


def find_typed_run(word_tokens, candidate_positions, indicator_types, wptype,
                   min_length=1):
    """The LONGEST contiguous run of positions drawn from `candidate_positions`
    (positional indices into `word_tokens`) whose joined surface text the DB types
    as `wptype`. Ties broken by earliest start. Returns the list of positions, or
    None if no qualifying run exists.

    `candidate_positions` need not be contiguous: runs are formed from the
    contiguous stretches WITHIN the candidate set, so an indicator can only be a
    block of adjacent leftover words, never a jump across an accounted word.
    """
    if indicator_types is None:
        return None

    def typed(text):
        try:
            return wptype in (indicator_types(text) or set())
        except Exception:
            return False

    best = None
    for group in contiguous_groups(sorted(candidate_positions)):
        for length in range(len(group), min_length - 1, -1):
            for s in range(0, len(group) - length + 1):
                seg = group[s:s + length]
                phrase = " ".join(word_tokens[i].text for i in seg)
                if typed(phrase):
                    if best is None or length > len(best):
                        best = seg
    return best
