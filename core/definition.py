"""Definition extraction — stage 3.

Splits the clue into a definition (a phrase at the START or END of the clue,
by cryptic convention) and the remaining wordplay words, so the catalog engine
works on the wordplay only.

core/ stays decoupled: takes a `defines` predicate -- defines(phrase, answer)
-> bool -- and an optional `ai_define` fallback -- ai_define(clue_text, answer)
-> phrase|None -- both supplied by the caller (live wiring passes the reference
DB's check and the separated AI assist; tests pass fakes).

The DB path is tried first. Only if it finds nothing is the AI fallback used,
and its result is flagged source="ai" so downstream knows it is a guess.
"""

from dataclasses import dataclass
from .atomiser import AtomisedClue

_EDGE = " .,;:!?\"'()[]{}-"


@dataclass
class DefinitionSplit:
    phrase: str          # the defining phrase, as written
    where: str           # 'start' or 'end'
    wordplay: list       # the remaining Atoms, in clue order
    source: str = "db"   # 'db' or 'ai'


def _norm(text):
    return " ".join(t for t in (w.strip(_EDGE) for w in text.lower().split()) if t)


def extract_definitions(clue: AtomisedClue, defines, ai_define=None,
                        max_window: int = 4) -> list:
    atoms = clue.atoms
    answer = clue.answer_letters
    n = len(atoms)
    if n < 2 or not answer:
        return []

    upper = min(max_window, n - 1)              # leave >=1 word for wordplay
    candidates = []
    for size in range(upper, 0, -1):            # longest first
        start_phrase = " ".join(a.surface for a in atoms[:size])
        if defines(start_phrase, answer):
            candidates.append(DefinitionSplit(start_phrase, "start", list(atoms[size:])))
        end_phrase = " ".join(a.surface for a in atoms[n - size:])
        if defines(end_phrase, answer):
            candidates.append(DefinitionSplit(end_phrase, "end", list(atoms[:n - size])))
    if candidates:
        return candidates

    # --- AI fallback: only when the DB found nothing ---
    if ai_define is None:
        return []
    phrase = ai_define(clue.clue_text, clue.answer_text)
    if not phrase:
        return []
    split = _locate_edge(atoms, phrase)
    return [split] if split else []


def _locate_edge(atoms, phrase):
    """Find which edge window the AI phrase corresponds to (so we get the
    wordplay words), or None if it is not a contiguous edge run leaving
    wordplay behind."""
    n = len(atoms)
    target = _norm(phrase)
    if not target:
        return None
    for size in range(1, n):                    # leave >=1 wordplay word
        if _norm(" ".join(a.surface for a in atoms[:size])) == target:
            return DefinitionSplit(phrase, "start", list(atoms[size:]), source="ai")
        if _norm(" ".join(a.surface for a in atoms[n - size:])) == target:
            return DefinitionSplit(phrase, "end", list(atoms[:n - size]), source="ai")
    return None
