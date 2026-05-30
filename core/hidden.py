"""Hidden-word check — the first solving stage a clue meets.

A clue is a hidden-word clue when the answer's letters appear contiguously,
forwards or reversed, inside the clue's letters — within a single word (but not
the whole word, e.g. OTIC in "nOTICed") or spanning several words (but not all
of them). Pure: no database, no network.

Takes an AtomisedClue, returns a ParseResult or None. The definition (which
words define the answer) is a separate, shared concern and is not set here.
"""

from .atomiser import AtomisedClue
from .model import Piece, Provenance, ParseResult


def find_hidden(clue: AtomisedClue):
    target = clue.answer_letters
    if len(target) < 3:
        return None

    # Clue letter-stream, with a map from each stream position back to its atom.
    chars = []
    pos_atom = []
    for atom in clue.atoms:
        for ch in atom.text.upper():
            if ch.isalpha():
                chars.append(ch)
                pos_atom.append(atom.index)
    stream = "".join(chars)

    for reverse in (False, True):
        needle = target[::-1] if reverse else target
        start = stream.find(needle)
        while start != -1:
            end = start + len(needle)
            spanned = sorted(set(pos_atom[start:end]))
            if _is_valid_hidden(start, end, stream, spanned, clue):
                return _build(clue, target, spanned, reverse)
            start = stream.find(needle, start + 1)
    return None


def _is_valid_hidden(start, end, stream, spanned, clue) -> bool:
    """Reject the two non-hidden cases: the whole clue, and a whole single word."""
    if start == 0 and end == len(stream):
        return False                                  # the entire clue
    if len(spanned) == 1:
        atom = clue.atoms[spanned[0]]
        atom_len = sum(1 for c in atom.text if c.isalpha())
        if end - start == atom_len:
            return False                              # the whole single word
    return True


def _build(clue: AtomisedClue, target: str, spanned: list, reverse: bool):
    source_text = " ".join(clue.atoms[i].surface for i in spanned)
    mechanism = "hidden_reversed" if reverse else "hidden"
    piece = Piece(atoms=spanned, source_text=source_text,
                  value=target, mechanism=mechanism)
    prov = [Provenance(0, len(target), piece, "hidden",
                       transform="reversed" if reverse else None)]
    # Confidence and definition are set by later shared stages; not here.
    return ParseResult(answer=target, definition="", pieces=[piece],
                       provenance=prov, operation=mechanism, solved_by="hidden")
