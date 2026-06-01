"""Prove the WFW model holds the two reference clues, with colour mapping.

Builds CATALOGUE (anagram) and INSTIGATE (charade) BY HAND on core/wfw_model,
using the real character numbering from core/wfw_atoms. No solver is involved —
this proves only that the model can record what the user asked for: every answer
tile traces to the source word that made it, so tile and word share a colour.

Run:  python -m core.prove_wfw_model
"""

from core.wfw_atoms import build_wfw_atom_context
from core.wfw_model import Source, Link, Annotation, Parse

# Eight terminal-ish colours, one per source word.
_PALETTE = ["blue", "green", "orange", "pink", "cyan", "yellow", "red", "purple"]


def clue_word_atom_ids(ctx, word_text):
    """The clue CharAtom ids of the first token whose text matches word_text."""
    for tok in ctx.clue_tokens:
        if tok.text.lower().strip(",.?!\"'") == word_text.lower():
            return tok.atom_ids
    raise ValueError("word not found in clue: %r" % word_text)


def phrase_atom_ids(ctx, *words):
    """Combined clue atom ids for a run of words, e.g. ('being','rewritten')."""
    ids = ()
    for w in words:
        ids += clue_word_atom_ids(ctx, w)
    return ids


def show(parse):
    print("=" * 68)
    print("CLUE:   %s" % parse.clue_text)
    print("ANSWER: %s   (%s)" % (parse.answer_text, parse.operation))
    print("complete (every answer letter sourced once):", parse.is_complete())
    # Colour per source.
    colour = {i: _PALETTE[i % len(_PALETTE)] for i in range(len(parse.sources))}
    print("\nsource words:")
    for i, s in enumerate(parse.sources):
        print("   [%-7s] %-10r -> %-9s (%s)" % (colour[i], s.text, s.value, s.mechanism))
    if parse.definition:
        print("   [definition] %r covers the whole answer" % parse.definition.text)
    print("\nanswer tiles:")
    letters = parse.answer_letters()
    by_pos = {l.answer_pos: l for l in parse.links}
    row_tiles, row_cols = [], []
    for pos in range(1, len(letters) + 1):
        link = by_pos[pos]
        row_tiles.append(" %s " % letters[pos - 1])
        row_cols.append(("%s" % colour[link.source_index]).center(3))
    print("   " + "|".join(row_tiles))
    print("   " + " ".join(row_cols))


def build_catalogue():
    """'Record a go at clue being rewritten' -> CATALOGUE.
    Definition: 'Record'. Fodder a+go+at+clue = AGOATCLUE, anagram of CATALOGUE
    ('being rewritten' is the anagram indicator). Each answer letter is assigned
    to the fodder word it came from, so every tile gets a colour."""
    clue_text = "Record a go at clue being rewritten"
    ctx = build_wfw_atom_context(clue_text, "CATALOGUE")
    # Four fodder words, each a colour source (CLUE shown blue per the spec).
    sources = [
        Source(clue_word_atom_ids(ctx, "clue"), "clue", "CLUE", "anagram_fodder"),
        Source(clue_word_atom_ids(ctx, "a"),    "a",    "A",    "anagram_fodder"),
        Source(clue_word_atom_ids(ctx, "go"),   "go",   "GO",   "anagram_fodder"),
        Source(clue_word_atom_ids(ctx, "at"),   "at",   "AT",   "anagram_fodder"),
    ]
    # CATALOGUE letters -> source index (the colours in the spec):
    #  C->CLUE A->A T->AT A->AT L->CLUE O->GO G->GO U->CLUE E->CLUE
    assign = [0, 1, 3, 3, 0, 2, 2, 0, 0]
    links = [Link(answer_pos=i + 1, source_index=assign[i], operation="anagram",
                  transform="anagram_of") for i in range(9)]
    annotations = [
        Annotation(phrase_atom_ids(ctx, "being", "rewritten"),
                   "being rewritten", "indicator", "anagram indicator"),
    ]
    return Parse(clue_text=clue_text, answer_text="CATALOGUE", sources=sources,
                 links=links, annotations=annotations,
                 definition=Source(clue_word_atom_ids(ctx, "record"),
                                   "Record", "CATALOGUE", "definition"),
                 operation="anagram", solved_by="catalog")


def build_instigate():
    """'Popular street with one barrier to start' -> INSTIGATE.
    IN(Popular) + ST(street) + I(one) + GATE(barrier); 'to start' defines."""
    clue_text = "Popular street with one barrier to start"
    ctx = build_wfw_atom_context(clue_text, "INSTIGATE")
    sources = [
        Source(clue_word_atom_ids(ctx, "popular"), "Popular", "IN",   "synonym"),
        Source(clue_word_atom_ids(ctx, "street"),  "street",  "ST",   "abbreviation"),
        Source(clue_word_atom_ids(ctx, "one"),     "one",     "I",    "abbreviation"),
        Source(clue_word_atom_ids(ctx, "barrier"), "barrier", "GATE", "synonym"),
    ]
    # INSTIGATE: I N S T I G A T E
    #   IN  -> pos 1,2 (src 0); ST -> 3,4 (src1); I -> 5 (src2); GATE -> 6,7,8,9 (src3)
    assign = [0, 0, 1, 1, 2, 3, 3, 3, 3]
    links = [Link(answer_pos=i + 1, source_index=assign[i], operation="charade")
             for i in range(9)]
    annotations = [
        Annotation(clue_word_atom_ids(ctx, "with"), "with", "link",
                   "joining word"),
    ]
    return Parse(clue_text=clue_text, answer_text="INSTIGATE", sources=sources,
                 links=links, annotations=annotations,
                 definition=Source(phrase_atom_ids(ctx, "to", "start"),
                                   "to start", "INSTIGATE", "definition"),
                 operation="charade", solved_by="catalog")


if __name__ == "__main__":
    show(build_catalogue())
    show(build_instigate())
