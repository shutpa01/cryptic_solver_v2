"""Effective operation for a parse, read from WHAT EACH PIECE DOES.

Clues filed in /hs are stored with operation 'manual', so the op name says
nothing; the pool used to classify them by tile SHAPE alone. That (2026-10-31):
  - rejected every pure /hs anagram (one fodder source = one tile, 'atoms_1'),
  - rejected every /hs hidden word (one raw source = one tile),
  - accepted charade/reorder/split clues whose anagram-fodder tile showed the
    answer letters ALREADY UNSCRAMBLED (ACCUSER: ACUSER around C),
  - accepted clues containing homophone / spoonerism pieces (sound family is
    excluded from the game).

effective_op(parse) returns an op name the existing builders already handle
(FIXED_KIND in seed_pool, the dispatch in convert.convert_clue), so the pool
and the converter always agree. Shared by both — never classify in one only.
"""
from crypticker.convert import build_hidden

SOUND_MECHS = {"homophone", "spoonerism"}
FIXED_OPS = {"anagram", "anagram_charade", "anagram_container", "hidden",
             "hidden_reversed", "acrostic", "alternation", "double_definition"}


def _used_sources(parse):
    used = {l.source_index for l in parse.links if l.source_index is not None}
    return [parse.sources[i] for i in sorted(used) if 0 <= i < len(parse.sources)]


def effective_op(parse):
    op = parse.operation or ""
    if op in FIXED_OPS:                     # the engine named a fixed kind: trust it
        return op
    srcs = _used_sources(parse)
    mechs = [s.mechanism for s in srcs]
    if any(m in SOUND_MECHS for m in mechs):
        return "homophone"                  # excluded by seed_pool SOUND_MARKERS
    if mechs and all(m == "anagram_fodder" for m in mechs):
        return "anagram"
    if "anagram_fodder" in mechs:
        return "anagram_charade"            # mixed board (anagram_hybrid)
    if len(srcs) == 1 and mechs[0] == "raw":
        # one literal run of clue letters: a hidden word, only if the answer
        # really sits unbroken in the clue (forwards or backwards)
        if build_hidden(parse.clue_text, parse.answer_text, False):
            return "hidden"
        if build_hidden(parse.clue_text, parse.answer_text, True):
            return "hidden_reversed"
    return op
