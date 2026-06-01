"""WFW parse model — the coloured word-for-word record.

Sits on top of core/wfw_atoms.py, which numbers every character of the clue and
the answer. A solved clue is recorded as:

- a list of Sources: the clue words (or spans) that contribute answer letters.
  Each Source is the unit of COLOUR — one colour per source word, shown on both
  the clue word and the answer tiles it produced.
- a list of Links: ONE per answer letter, saying which Source produced that
  letter (its colour) and, for letter-selection operations, the exact clue
  character it came from.
- a Definition: a separate Source covering the whole answer by meaning. It
  overlaps the wordplay deliberately (the answer is both defined and built), so
  completeness is checked within the wordplay layer, not across both.

This is the user's model exactly: every answer tile points back to the source
word that made it, so tile and word share a colour.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Source:
    """One clue word/span that contributes answer letters (a colour unit)."""
    clue_atom_ids: tuple      # the clue CharAtom ids forming this word/span
    text: str                 # surface as written, e.g. "barrier"
    value: str                # letters produced, e.g. "GATE"
    mechanism: str            # synonym | abbreviation | raw | hidden |
                              #   first_letter | homophone | anagram_fodder
    source: str = "db"        # provenance of this piece. For a DEFINITION source:
                              #   'db' = confirmed by the reference DB;
                              #   'pending' = supplied provisionally (AI definition
                              #   fallback or edge-anchored) and queued for human
                              #   enrichment (badged, cleared once verified).


@dataclass
class Annotation:
    """A clue word that contributes NO answer letters but still has a job:
    a link/joining word, or an indicator that signals the operation. Recorded so
    every clue word is explained — nothing is left silently unaccounted."""
    clue_atom_ids: tuple
    text: str                 # surface as written, e.g. "with" / "being rewritten"
    role: str                 # 'link' | 'indicator'
    note: str = ""            # e.g. 'joining word' / 'anagram indicator'
    source: str = "db"        # 'db' = role confirmed by the reference DB;
                              # 'pending' = supplied provisionally (e.g. a whole
                              # bound indicator phrase) and queued for enrichment.


@dataclass
class Link:
    """One answer letter and where it came from — the unit of colour on a tile.

    answer_pos is 1-based (matches wfw_atoms letter_position). source_index
    points into Parse.sources. clue_atom_id pins the exact clue character for
    letter-selection operations (hidden, acrostic, reversal, deletion, and the
    per-letter anagram assignment); it is None for meaning-based pieces
    (synonym, abbreviation) whose letters come from the DB value, not from a
    clue character.
    """
    answer_pos: int
    source_index: int
    operation: str                       # charade | container | anagram | ...
    clue_atom_id: Optional[str] = None
    transform: Optional[str] = None      # None | reversed | anagram_of | ...


@dataclass
class Parse:
    clue_text: str
    answer_text: str
    sources: list = field(default_factory=list)   # list[Source]  (wordplay pieces)
    links: list = field(default_factory=list)      # list[Link], one per answer letter
    annotations: list = field(default_factory=list)  # list[Annotation] link/indicator words
    definition: Optional[Source] = None            # definition layer (whole answer)
    operation: str = ""                            # top-level operation label
    confidence: int = 0
    solved_by: str = ""                            # hidden | dd | catalog | cd
    status: str = "pass"                           # 'pass' | 'fail' (engine verdict)
    warnings: list = field(default_factory=list)   # list[str], plain-English on fail

    def answer_letters(self) -> str:
        return "".join(c for c in self.answer_text.upper() if c.isalpha())

    def is_complete(self) -> bool:
        """Every answer letter has exactly one wordplay link — no gap, no
        double. The definition layer is intentionally excluded from this check."""
        n = len(self.answer_letters())
        positions = sorted(link.answer_pos for link in self.links)
        return positions == list(range(1, n + 1))

    def unexplained_words(self, ctx) -> list:
        """Clue words with NO recorded role — the gap the user must never see.

        Every clue word must be a piece (source), the definition, a link, or an
        indicator. Returns the surface text of any word that is none of these.
        `ctx` is a wfw_atoms context for this clue.
        """
        accounted = set()
        for s in self.sources:
            accounted.update(s.clue_atom_ids)
        if self.definition:
            accounted.update(self.definition.clue_atom_ids)
        for a in self.annotations:
            accounted.update(a.clue_atom_ids)
        missing = []
        for tok in ctx.clue_tokens:
            if tok.kind != "word":
                continue                       # punctuation carries no role
            if not any(aid in accounted for aid in tok.atom_ids):
                missing.append(tok.text)
        return missing

    def enumeration(self) -> str:
        """The (n) / (4,3) enumeration from the answer's word lengths."""
        counts = [sum(1 for c in w if c.isalpha())
                  for w in self.answer_text.split()]
        counts = [c for c in counts if c]
        return "(%s)" % ",".join(str(c) for c in counts) if counts else ""
