"""Universal definition engine — the one shared definition finder.

By cryptic convention a clue is a definition at ONE EDGE (start or end) plus
wordplay in the middle. Finding that definition is the same job for every
wordplay type, so it is built ONCE here and fed to each type engine; the type
engine then explains only the wordplay. (Double-definition and cryptic-definition
are their own clue types and do not use this edge-split engine.)

Built on the shared character atomiser (wfw_atoms): it returns the exact clue
atom ids of the definition span and of the remaining wordplay, so a type engine
and its screen work in the same character coordinates as everything else.

Decoupled from any particular database: the caller injects
    defines(phrase, answer_letters) -> bool
(live wiring passes the reference DB's check; tests pass a fake).
"""

import os

from dataclasses import dataclass, field

# The NO-DEFINITION FLOOR (provisional longest-edge "definition" offered when neither the DB
# nor Haiku confirms one) is ON by default. It only ever produces source='pending' splits,
# so it can never create or destroy a confirmed PASS. Two safeguards (added 2026-06-24) keep
# it honest instead of letting it fabricate nonsense:
#   - a floor (source='pending') definition is DROPPED from any FAIL parse in
#     engine_registry._finish, so a "forced definition with no wordplay" is never shown;
#   - on a KEPT near-solve (status 'pending') the floor guess renders as "unidentified
#     definition - not confirmed", never as if it were the real definition.
# A hand-set (forced) definition is source='manual' and is never touched by the floor.
# Set env DEF_FLOOR=0 to turn the floor off (used to A/B this change).
_DEF_FLOOR_ENABLED = os.environ.get("DEF_FLOOR", "1") == "1"


@dataclass
class DefinitionSplit:
    phrase: str                  # the defining words, as written
    where: str                   # 'start' or 'end'
    def_atom_ids: tuple          # clue char atom ids of the definition span
    def_indices: tuple = ()                                # word positions of the definition
    def_tokens: list = field(default_factory=list)        # the definition word tokens
    wordplay_tokens: list = field(default_factory=list)   # remaining word tokens
    wordplay_atom_ids: tuple = ()                          # their char atom ids
    source: str = "db"                                     # 'db' or 'pending'
    dbe_tokens: list = field(default_factory=list)        # definition-by-example
                                                          #   indicator word(s) ("perhaps",
                                                          #   "possibly") peeled off the
                                                          #   wordplay edge by the def
    by_example: bool = False                              # the definition is by example


def _word_tokens(ctx):
    return [t for t in ctx.clue_tokens if t.kind == "word"]


def _answer_letters(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def andlit_split(ctx):
    """The &lit (all-in-one) split: the WHOLE clue is BOTH the definition AND the
    wordplay. def and wordplay overlap on every word, so an answer-driven wordplay
    engine (acrostic / alternation / ...) can use every word as fodder while the whole
    clue stands as the definition. source='pending' — the wordplay is verified
    mechanically, but 'the surface ALSO reads as a definition' is a human judgement, so
    an &lit is never auto-confirmed (the same rule as a cryptic definition)."""
    words = _word_tokens(ctx)
    atoms = tuple(aid for t in words for aid in t.atom_ids)
    return DefinitionSplit(
        phrase=ctx.clue_text, where="all", def_atom_ids=atoms,
        def_indices=tuple(range(len(words))), def_tokens=list(words),
        wordplay_tokens=list(words), wordplay_atom_ids=atoms, source="andlit")


def find_definitions(ctx, defines, max_window=8, extend=False,
                     wordplay_indices=None, define_fallback=None, is_dbe=None,
                     andlit=False):
    """All edge definition splits whose phrase `defines` the answer.

    Tries the longest edge windows first (a longer real definition beats a
    shorter coincidental one), leaving at least one wordplay word. Returns a
    list of DefinitionSplit, best (longest) first; empty if none define.

    When `extend` is true, a grammar-EXTENT pass grows each confirmed definition
    outward toward the wordplay, absorbing grammatically-bound function words the
    DB could not confirm on their own (e.g. "mountains" -> "in the mountains" for
    ANDEAN). `wordplay_indices` (word positions that carry a wordplay role) are
    never absorbed; pass them when known so the extent stops at the wordplay.

    `define_fallback`, when supplied, is the shared Haiku definition fallback
    (core.definition_fallback.make_fallback): a callable define_fallback(ctx) ->
    DefinitionSplit | None. It is consulted ONLY when the reference DB confirms
    NO definition, and the split it returns is flagged source='pending' so the
    engine renders it provisionally and the registry queues it for enrichment.
    This lives here, in the one definition stage, so every engine that calls it
    inherits the fallback without re-implementing it. (Hidden/DD do not pass it
    and so are unaffected — they keep their own existing handling.)
    """
    words = _word_tokens(ctx)
    answer = _answer_letters(ctx)
    n = len(words)
    if n < 2 or not answer:
        return []

    upper = min(max_window, n - 1)               # leave >=1 wordplay word
    out = []
    for size in range(upper, 0, -1):
        # start edge
        phrase = " ".join(t.text for t in words[:size])
        if defines(phrase, answer):
            out.append(_split_from_indices(words, set(range(size)), "start"))
        # end edge
        phrase = " ".join(t.text for t in words[n - size:])
        if defines(phrase, answer):
            out.append(_split_from_indices(words, set(range(n - size, n)), "end"))

    # Haiku fallback — ONLY on a complete DB miss. Provisional, queued downstream.
    if not out and define_fallback is not None:
        fb = define_fallback(ctx)
        if fb is not None:
            out = [fb]

    # NO-DEFINITION FLOOR — when NEITHER the DB nor Haiku supplies a definition, offer
    # every edge window as an UNCONFIRMED definition (source='pending'). The engine that
    # reconstructs the answer from the REST proves the leftover edge IS the definition:
    # so a solvable wordplay is still shown (pending) and the residue edge is queued for
    # you to add, instead of the whole clue vanishing. Only fires on a total def miss, so
    # it never changes a confirmed solve.
    if not out and _DEF_FLOOR_ENABLED:
        from core import grammar
        postags = grammar.pos_tags([t.text for t in words])
        out = _residue_edge_splits(words, max_window, postags)

    if extend and out:
        out = [_extend_split(words, s, wordplay_indices) for s in out]

    # Definition-by-example: a DBE indicator ("perhaps", "possibly") sitting on the
    # wordplay edge next to the definition is peeled off here — it marks the
    # definition as by-example and must not be left for the wordplay to grab as an
    # operation indicator (memory: feedback-definition-by-example).
    if is_dbe is not None:
        out = [_peel_dbe(words, s, is_dbe) for s in out]
        # A DBE marker can also sit INSIDE the wordplay, attached to a fodder word
        # ("evergreen, say" -> TREE by example), not just at the definition edge. Offer
        # ADDITIONAL splits with such a marker peeled (the unpeeled split is kept, so a
        # clue where the word is real fodder is unaffected). Engines account the marker via
        # dbe_annotation(split), exactly as for a definition-edge DBE.
        extra = []
        for s in out:
            extra += _peel_wordplay_dbe(words, s, is_dbe)
        out = out + extra

    # &lit (all-in-one): offer the whole-clue overlap split so a clue whose wordplay IS
    # its definition can solve. Only when NO DB-confirmed edge definition exists (a
    # confirmed edge def means it is an ordinary clue, not &lit). Added LAST so a genuine
    # edge-def solve is always preferred; the &lit split is pending, so it never
    # auto-confirms. Opt-in (andlit=True) per engine — only the precise, answer-driven,
    # indicator-gated engines should accept it.
    if andlit and not any(s.source == "db" for s in out):
        out = out + [andlit_split(ctx)]
    return out


def _peel_wordplay_dbe(words, split, is_dbe):
    """Variant splits with ONE contiguous definition-by-example run peeled from INSIDE the
    wordplay and recorded on dbe_tokens. Additive: returns a list (possibly empty); the
    caller keeps the original. Skipped when the split already carries a DBE marker (the
    definition-edge case, handled by _peel_dbe). Always leaves >=1 wordplay word."""
    wp = split.wordplay_tokens
    if not wp or len(wp) < 2 or getattr(split, "dbe_tokens", None):
        return []
    variants, seen = [], set()
    m = len(wp)
    for k in range(_DBE_MAX_WORDS, 0, -1):
        for i in range(0, m - k + 1):
            run = wp[i:i + k]
            if not _is_dbe_phrase(is_dbe, " ".join(t.text for t in run)):
                continue
            new_wp = wp[:i] + wp[i + k:]
            if not new_wp:
                continue
            key = tuple(id(t) for t in run)
            if key in seen:
                continue
            seen.add(key)
            variants.append(DefinitionSplit(
                phrase=split.phrase, where=split.where,
                def_atom_ids=split.def_atom_ids, def_indices=split.def_indices,
                def_tokens=split.def_tokens, wordplay_tokens=new_wp,
                wordplay_atom_ids=tuple(aid for t in new_wp for aid in t.atom_ids),
                source=split.source, dbe_tokens=list(run), by_example=True))
    return variants


def dbe_annotation(split):
    """The Annotation for a split's definition-by-example indicator, or None — the
    one place engines build it, so the DBE marker is recorded consistently and the
    word is accounted (kept out of the wordplay)."""
    from core.wfw_model import Annotation
    toks = getattr(split, "dbe_tokens", None)
    if not toks:
        return None
    return Annotation(
        clue_atom_ids=tuple(aid for t in toks for aid in t.atom_ids),
        text=" ".join(t.text for t in toks),
        role="indicator", note="definition by example")


_DBE_MAX_WORDS = 3       # longest DBE indicator phrase to peel ("for example" = 2)


def _is_dbe_phrase(is_dbe, phrase):
    try:
        return bool(is_dbe(phrase))
    except Exception:
        return False


def _peel_dbe(words, split, is_dbe):
    """Peel a definition-by-example indicator ("perhaps", "say", "for example", "e g",
    ...) sitting at the boundary between the definition and the wordplay, recording it
    on the split (dbe_tokens, by_example) so it is accounted as a by-example marker and
    not handed to the wordplay as an operation indicator (memory:
    feedback-definition-by-example).

    Two positions are handled, both at the definition's wordplay-facing edge:
      A. the indicator was ABSORBED INTO the definition window — its inner-edge word(s)
         (e.g. "skeletons perhaps" | British -> def "skeletons" + DBE "perhaps");
      B. the indicator sits in the WORDPLAY adjacent to the definition (e.g.
         skeletons | "e g" British -> def "skeletons" + DBE "e g").
    Multi-word indicators ("for example", "e g") are matched, longest first. Always
    leaves >=1 definition word and >=1 wordplay word; a no-op when the boundary word(s)
    are not a DBE indicator."""
    if not split.def_indices:
        return split
    def_idx = sorted(split.def_indices)
    def_set = set(def_idx)
    n = len(words)

    # --- Case A: DBE indicator is the definition's inner-edge run -------------------
    # start-def: inner edge = the LAST def words; end-def: the FIRST def words.
    inner = list(reversed(def_idx)) if split.where == "start" else list(def_idx)
    for k in range(min(_DBE_MAX_WORDS, len(def_idx) - 1), 0, -1):
        peel = sorted(inner[:k])
        if peel != list(range(peel[0], peel[-1] + 1)):
            continue                              # must be a contiguous run
        phrase = " ".join(words[i].text for i in peel)
        if _is_dbe_phrase(is_dbe, phrase):
            new_def = [i for i in def_idx if i not in set(peel)]
            if new_def:
                return _dbe_split(words, split, new_def,
                                  [words[i] for i in peel])

    # --- Case B: DBE indicator on the wordplay edge adjacent to the definition ------
    if len(split.wordplay_tokens) < 2:
        return split
    wp_ids = {id(t) for t in split.wordplay_tokens}
    for k in range(_DBE_MAX_WORDS, 0, -1):
        if split.where == "start":
            idxs = sorted(max(def_idx) + j for j in range(1, k + 1))
        else:
            idxs = sorted(min(def_idx) - j for j in range(1, k + 1))
        if idxs[0] < 0 or idxs[-1] >= n or set(idxs) & def_set:
            continue
        toks = [words[i] for i in idxs]
        if not all(id(t) in wp_ids for t in toks):
            continue
        phrase = " ".join(t.text for t in toks)
        if _is_dbe_phrase(is_dbe, phrase):
            peeled = {id(t) for t in toks}
            new_wp = [t for t in split.wordplay_tokens if id(t) not in peeled]
            if new_wp:
                return DefinitionSplit(
                    phrase=split.phrase, where=split.where,
                    def_atom_ids=split.def_atom_ids, def_indices=split.def_indices,
                    def_tokens=split.def_tokens, wordplay_tokens=new_wp,
                    wordplay_atom_ids=tuple(aid for t in new_wp for aid in t.atom_ids),
                    source=split.source, dbe_tokens=toks, by_example=True)
    return split


def _dbe_split(words, split, new_def_idx, dbe_toks):
    """A split with the definition shrunk to new_def_idx and dbe_toks recorded as the
    by-example marker (the peeled words are NOT returned to the wordplay)."""
    new_def_idx = sorted(new_def_idx)
    new_def_tokens = [words[i] for i in new_def_idx]
    return DefinitionSplit(
        phrase=" ".join(t.text for t in new_def_tokens),
        where=split.where,
        def_atom_ids=tuple(aid for t in new_def_tokens for aid in t.atom_ids),
        def_indices=tuple(new_def_idx),
        def_tokens=new_def_tokens,
        wordplay_tokens=split.wordplay_tokens,
        wordplay_atom_ids=split.wordplay_atom_ids,
        source=split.source, dbe_tokens=dbe_toks, by_example=True)


def _extend_split(words, split, wordplay_indices):
    """Apply the grammar-extent pass to one split, returning the grown split."""
    from core import grammar
    surfaces = [t.text for t in words]
    def_idx = set(split.def_indices)
    wp_idx = (set(wordplay_indices) if wordplay_indices is not None
              else {i for i in range(len(words)) if i not in def_idx})
    grown = grammar.extend_definition_indices(surfaces, def_idx, wp_idx)
    if grown == def_idx:
        return split
    return _split_from_indices(words, grown, split.where, source=split.source)


def extend_definition(ctx, split, used_atom_ids):
    """Grow a confirmed definition outward once the engine knows the wordplay.

    `used_atom_ids` is the set of clue atom ids that already carry a wordplay
    role (host letters + indicator). Any word that is NEITHER the definition nor
    used by the wordplay is a candidate the grammar pass may absorb into the
    definition (e.g. "in", "the" for ANDEAN). Returns a possibly-grown split.

    This is the universal definition-extent test, applied at the right moment:
    after roles are known, so the wordplay words are never swallowed.
    """
    words = _word_tokens(ctx)
    used = set(used_atom_ids)
    wp_idx = {i for i, t in enumerate(words)
              if any(aid in used for aid in t.atom_ids)}
    return _extend_split(words, split, wp_idx)


# A definition is a coherent phrase: it may not DANGLE on a linking/function word at
# either end, nor be made up only of function words. (A leading determiner is fine —
# "A term" — so DET is barred at the end but allowed at the start.)
_DEF_BAD_END = frozenset({"ADP", "CCONJ", "SCONJ", "PART", "DET"})
_DEF_BAD_START = frozenset({"ADP", "CCONJ", "SCONJ", "PART"})
_DEF_FUNC = frozenset({"ADP", "PART", "AUX", "DET", "CCONJ", "SCONJ"})


def _def_span_grammatical(postags, lo, hi):
    """True if the definition span words[lo:hi] is a grammatically plausible phrase: it does
    not dangle on a linking/function word at either edge and is not entirely function words
    ("A term for", "for Oxbridge festival", "A" are rejected). Best-effort: with no POS
    (postags empty) everything passes, preserving prior behaviour."""
    if not postags:
        return True
    span = postags[lo:hi]
    if not span:
        return False
    if span[-1] in _DEF_BAD_END or span[0] in _DEF_BAD_START:
        return False
    return not all(p in _DEF_FUNC for p in span)


def _residue_edge_splits(words, max_window=8, postags=None):
    """Every GRAMMATICALLY-PLAUSIBLE edge-window definition candidate (both edges, longest
    first), marked source='pending'. The NO-DEFINITION FLOOR: handed to the engines only
    when no real definition was found, so the one whose REST reconstructs the answer
    surfaces the leftover edge as the (provisional) definition. Windows that do not read as
    a coherent phrase are dropped so a nonsense definition is never proposed. Leaves >=1
    wordplay word."""
    n = len(words)
    if n < 2:
        return []
    upper = min(max_window, n - 1)
    out = []
    for size in range(upper, 0, -1):
        if _def_span_grammatical(postags, 0, size):
            out.append(_split_from_indices(words, set(range(size)), "start",
                                           source="pending"))
        if _def_span_grammatical(postags, n - size, n):
            out.append(_split_from_indices(words, set(range(n - size, n)), "end",
                                           source="pending"))
    return out


def _split_from_indices(words, def_indices, where, source="db"):
    """Build a DefinitionSplit from the set of definition word positions."""
    def_idx = sorted(def_indices)
    def_set = set(def_idx)
    def_tokens = [words[i] for i in def_idx]
    wp_tokens = [t for i, t in enumerate(words) if i not in def_set]
    return DefinitionSplit(
        phrase=" ".join(t.text for t in def_tokens),
        where=where,
        def_atom_ids=tuple(aid for t in def_tokens for aid in t.atom_ids),
        def_indices=tuple(def_idx),
        def_tokens=def_tokens,
        wordplay_tokens=wp_tokens,
        wordplay_atom_ids=tuple(aid for t in wp_tokens for aid in t.atom_ids),
        source=source)
