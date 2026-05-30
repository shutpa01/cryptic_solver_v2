"""Definition extraction — stage 3.

Splits the clue into a definition (a phrase at the START or END of the clue,
by cryptic convention) and the remaining wordplay words, so the router and the
catalog engine work on the wordplay only.

core/ stays decoupled: takes a `defines` predicate -- defines(phrase, answer)
-> bool -- supplied by the caller (live wiring passes the reference DB's check;
tests pass a fake). Returns a list of DefinitionSplit candidates, longest
window first (a more specific definition preferred), or an empty list.
"""

from dataclasses import dataclass
from .atomiser import AtomisedClue


@dataclass
class DefinitionSplit:
    phrase: str          # the defining phrase, as written
    where: str           # 'start' or 'end'
    wordplay: list       # the remaining Atoms, in clue order


def extract_definitions(clue: AtomisedClue, defines, max_window: int = 4) -> list:
    atoms = clue.atoms
    answer = clue.answer_letters
    n = len(atoms)
    if n < 2 or not answer:
        return []

    # Windows must leave at least one word for the wordplay.
    upper = min(max_window, n - 1)
    candidates = []
    for size in range(upper, 0, -1):            # longest first
        start_phrase = " ".join(a.surface for a in atoms[:size])
        if defines(start_phrase, answer):
            candidates.append(
                DefinitionSplit(start_phrase, "start", list(atoms[size:])))
        end_phrase = " ".join(a.surface for a in atoms[n - size:])
        if defines(end_phrase, answer):
            candidates.append(
                DefinitionSplit(end_phrase, "end", list(atoms[:n - size])))
    return candidates
