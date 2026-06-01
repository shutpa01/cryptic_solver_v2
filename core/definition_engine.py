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

from dataclasses import dataclass, field


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


def _word_tokens(ctx):
    return [t for t in ctx.clue_tokens if t.kind == "word"]


def _answer_letters(ctx):
    return "".join(a.normalized for a in ctx.answer_atoms if a.kind == "letter")


def find_definitions(ctx, defines, max_window=5, extend=False,
                     wordplay_indices=None):
    """All edge definition splits whose phrase `defines` the answer.

    Tries the longest edge windows first (a longer real definition beats a
    shorter coincidental one), leaving at least one wordplay word. Returns a
    list of DefinitionSplit, best (longest) first; empty if none define.

    When `extend` is true, a grammar-EXTENT pass grows each confirmed definition
    outward toward the wordplay, absorbing grammatically-bound function words the
    DB could not confirm on their own (e.g. "mountains" -> "in the mountains" for
    ANDEAN). `wordplay_indices` (word positions that carry a wordplay role) are
    never absorbed; pass them when known so the extent stops at the wordplay.
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

    if extend and out:
        out = [_extend_split(words, s, wordplay_indices) for s in out]
    return out


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
