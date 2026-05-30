"""Double-definition check — stage 2.

A double-definition clue is two definitions side by side, each independently
defining the answer (e.g. "Bear endure" -> both mean ENDURE). There is no
wordplay: both halves are definitions.

core/ stays decoupled from any particular database: this takes a `defines`
predicate -- defines(phrase, answer) -> bool -- which the caller supplies
(the live wiring passes the reference DB's check; tests pass a fake). Takes an
AtomisedClue, returns a ParseResult or None.
"""

from .atomiser import AtomisedClue
from .model import Piece, Provenance, ParseResult


def find_dd(clue: AtomisedClue, defines) -> "ParseResult | None":
    answer = clue.answer_letters
    atoms = clue.atoms
    if len(answer) < 2 or len(atoms) < 2:
        return None

    # Split the clue into a LEFT phrase and a RIGHT phrase at every point;
    # both halves must independently define the answer. Every atom falls in
    # one half or the other, so the whole clue is always accounted for.
    for k in range(1, len(atoms)):
        left = " ".join(a.surface for a in atoms[:k])
        right = " ".join(a.surface for a in atoms[k:])
        if defines(left, answer) and defines(right, answer):
            return _build(clue, answer, left, right)
    return None


def _build(clue: AtomisedClue, answer: str, left: str, right: str):
    # No wordplay: both halves define the whole answer. One provenance entry
    # spans the whole answer, tagged as a double definition.
    piece = Piece(atoms=[a.index for a in clue.atoms],
                  source_text=f"{left}  /  {right}",
                  value=answer, mechanism="double_definition")
    prov = [Provenance(0, len(answer), piece, "double_definition")]
    return ParseResult(answer=answer, definition=f"{left}  /  {right}",
                       pieces=[piece], provenance=prov,
                       operation="double_definition", solved_by="dd")
