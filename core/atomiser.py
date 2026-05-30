"""Atomiser — front-end input preparation for the redesigned solver.

Breaks a raw clue and answer into clean units ONCE, so no downstream stage
re-derives them (the "atomise the clue and answer once" principle). Pure: no
database, no solving, no network.

- The clue becomes a list of Atoms, one per word.
- The answer becomes a list of LetterSlots, one per answer letter. Each slot
  has a `source` that solving fills in later — this is the word-for-word
  substrate (left None here; the atomiser does not solve).

Deliberately minimal. Sub-word atomisation (splitting contractions and
possessives, e.g. "he's" -> "he" + "'s") is a known future extension and is
NOT done here yet; see documents/SOLVER_REDESIGN.md section 3.1.
"""

from dataclasses import dataclass, field
import unicodedata


# Smart punctuation -> plain equivalents.
_SMART = {
    "‘": "'", "’": "'",      # ' '
    "“": '"', "”": '"',      # " "
    "–": "-", "—": "-",      # en/em dash
    "…": "...",                    # ellipsis
}

_EDGE_PUNCT = " .,;:!?\"'()[]{}-‘’“”"


@dataclass
class Atom:
    """One clue word."""
    index: int          # 0-based position among the atoms
    surface: str        # the word as written, e.g. "Bloke"
    text: str           # normalised, lower-case, edge-punctuation stripped


@dataclass
class LetterSlot:
    """One answer letter. `source` is filled during solving (provenance)."""
    index: int          # 0-based position in the answer
    letter: str         # single upper-case letter
    source: object = None


@dataclass
class AtomisedClue:
    clue_text: str          # original clue, as given
    answer_text: str        # original answer, as given
    atoms: list = field(default_factory=list)    # list[Atom]
    answer: list = field(default_factory=list)   # list[LetterSlot]

    @property
    def answer_letters(self) -> str:
        """The bare answer letters, upper-case, no spaces/hyphens."""
        return "".join(s.letter for s in self.answer)

    @property
    def words(self) -> list:
        """The atom surface words, in order."""
        return [a.surface for a in self.atoms]


def _normalise(text: str) -> str:
    """Strip accents and convert smart quotes/dashes to plain equivalents."""
    for smart, plain in _SMART.items():
        text = text.replace(smart, plain)
    # Decompose accents (a-grave -> a + combining mark) and drop the marks.
    decomposed = unicodedata.normalize("NFKD", text)
    return "".join(c for c in decomposed if not unicodedata.combining(c))


def atomise(clue_text: str, answer_text: str) -> AtomisedClue:
    """Atomise a clue and its answer once.

    The clue is normalised and split on whitespace into Atoms (pure-punctuation
    tokens are dropped). The answer is reduced to its bare upper-case letters,
    one LetterSlot each.
    """
    normalised = _normalise(clue_text)

    atoms = []
    for raw in normalised.split():
        text = raw.lower().strip(_EDGE_PUNCT)
        if not text:
            continue            # pure punctuation carries no role
        atoms.append(Atom(index=len(atoms), surface=raw, text=text))

    letters = [c for c in answer_text.upper() if c.isalpha()]
    answer = [LetterSlot(index=i, letter=ch) for i, ch in enumerate(letters)]

    return AtomisedClue(
        clue_text=clue_text,
        answer_text=answer_text,
        atoms=atoms,
        answer=answer,
    )
