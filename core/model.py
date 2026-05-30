"""Provenance model — the word-for-word record a solved clue produces.

Deliberately flat and small. A solved clue is a list of Pieces plus a list of
Provenance entries saying which answer letters each piece produced and by what
operation. The common case is flat; a genuinely nested clue (a container whose
inner is itself an anagram) can hang a child ParseResult off a Provenance entry,
but we do not build that until a clue needs it.

No solving machinery here — this is only the record.
"""

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Piece:
    """A contribution from one or more clue atoms."""
    atoms: list          # indices of the clue atoms this came from
    source_text: str     # the clue word(s), e.g. "Bloke"
    value: str           # the letters it produced, e.g. "MAN"
    mechanism: str       # synonym | abbreviation | raw | first_letter | ...


@dataclass
class Provenance:
    """answer[slot_start:slot_end] was produced by `piece` via `operation`.

    slot_end is exclusive. `transform` records a per-piece change such as a
    reversal; `child` lets a span be explained by a nested ParseResult for the
    rare compound case (left None otherwise).
    """
    slot_start: int
    slot_end: int
    piece: Piece
    operation: str                      # charade | container | anagram | ...
    transform: Optional[str] = None     # None | "reversed" | "anagram_of"
    child: Optional["ParseResult"] = None


@dataclass
class ParseResult:
    answer: str                         # bare answer letters, upper-case
    definition: str                     # the defining phrase from the clue
    pieces: list = field(default_factory=list)        # list[Piece]
    provenance: list = field(default_factory=list)    # list[Provenance]
    operation: str = ""                 # top-level operation label
    confidence: int = 0
    reasons: list = field(default_factory=list)       # list[(text, delta)]
    solved_by: str = ""                 # 'hidden' | 'dd' | 'catalog' | 'cd'

    def covers_answer(self) -> bool:
        """Completeness: every answer letter is sourced exactly once.

        This is the word-for-word guarantee — a parse with a gap or an overlap
        is not a complete account.
        """
        covered = [False] * len(self.answer)
        for pr in self.provenance:
            if not (0 <= pr.slot_start < pr.slot_end <= len(covered)):
                return False
            for i in range(pr.slot_start, pr.slot_end):
                if covered[i]:
                    return False        # overlap
                covered[i] = True
        return all(covered)

    def render(self) -> str:
        """A plain word-for-word line: which letters came from where."""
        parts = []
        for pr in sorted(self.provenance, key=lambda p: p.slot_start):
            seg = self.answer[pr.slot_start:pr.slot_end]
            tag = f" ({pr.transform})" if pr.transform else ""
            parts.append(f'{seg} <- "{pr.piece.source_text}"{tag}')
        return " | ".join(parts) + f"  =  {self.answer}"
