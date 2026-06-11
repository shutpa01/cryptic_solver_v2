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
    if getattr(parse.definition, "source", "db") == "pending":
        return "the definition is provisional (queued for enrichment)"
    return None


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
