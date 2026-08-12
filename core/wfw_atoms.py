"""Character atom foundation for WFW proof records.

This module is deliberately small: it does not solve clues.  It creates the
lowest-level durable tokens that later working blocks, transformations, and
answer placements must reference.
"""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class CharAtom:
    atom_id: str
    stream: str
    index: int
    char: str
    normalized: str
    kind: str
    letter_position: int | None = None

    def as_dict(self):
        return {
            "atom_id": self.atom_id,
            "stream": self.stream,
            "index": self.index,
            "char": self.char,
            "normalized": self.normalized,
            "kind": self.kind,
            "letter_position": self.letter_position,
        }


@dataclass(frozen=True)
class OriginalToken:
    token_id: str
    stream: str
    index: int
    text: str
    start_atom: str
    end_atom: str
    atom_ids: tuple[str, ...]
    kind: str

    def as_dict(self):
        return {
            "token_id": self.token_id,
            "stream": self.stream,
            "index": self.index,
            "text": self.text,
            "start_atom": self.start_atom,
            "end_atom": self.end_atom,
            "atom_ids": list(self.atom_ids),
            "kind": self.kind,
        }


@dataclass(frozen=True)
class WFWAtomContext:
    clue_text: str
    answer_text: str
    clue_atoms: tuple[CharAtom, ...]
    answer_atoms: tuple[CharAtom, ...]
    clue_tokens: tuple[OriginalToken, ...]
    answer_tokens: tuple[OriginalToken, ...]
    direction: Optional[str] = None    # 'across' | 'down' | None (unknown). Carried so
                                       # orientation-dependent indicators (e.g. a down-only
                                       # charade-positional like 'supporting') can gate on it.

    @property
    def answer_letter_atoms(self):
        return tuple(atom for atom in self.answer_atoms
                     if atom.letter_position is not None)

    def as_dict(self):
        return {
            "clue_text": self.clue_text,
            "answer_text": self.answer_text,
            "clue_atoms": [atom.as_dict() for atom in self.clue_atoms],
            "answer_atoms": [atom.as_dict() for atom in self.answer_atoms],
            "clue_tokens": [token.as_dict() for token in self.clue_tokens],
            "answer_tokens": [token.as_dict() for token in self.answer_tokens],
            "direction": self.direction,
        }


def context_from_dict(d):
    """Rebuild a WFWAtomContext from its as_dict() form — the exact inverse of
    WFWAtomContext.as_dict(). This restores a PERSISTED atomisation without
    re-running the atomiser, so the atoms (and their ids) the stored provenance
    references are preserved verbatim even if the atomiser code later changes."""
    def _atom(a):
        return CharAtom(
            atom_id=a["atom_id"], stream=a["stream"], index=a["index"],
            char=a["char"], normalized=a["normalized"], kind=a["kind"],
            letter_position=a.get("letter_position"))

    def _tok(t):
        return OriginalToken(
            token_id=t["token_id"], stream=t["stream"], index=t["index"],
            text=t["text"], start_atom=t["start_atom"], end_atom=t["end_atom"],
            atom_ids=tuple(t["atom_ids"]), kind=t["kind"])

    return WFWAtomContext(
        clue_text=d.get("clue_text", ""),
        answer_text=d.get("answer_text", ""),
        clue_atoms=tuple(_atom(a) for a in d.get("clue_atoms", [])),
        answer_atoms=tuple(_atom(a) for a in d.get("answer_atoms", [])),
        clue_tokens=tuple(_tok(t) for t in d.get("clue_tokens", [])),
        answer_tokens=tuple(_tok(t) for t in d.get("answer_tokens", [])),
        direction=d.get("direction"),
    )


def build_wfw_atom_context(clue_text, answer_text, direction=None):
    """Return stable character atoms and original tokens for clue and answer.

    `direction` ('across' | 'down' | None), when known, is carried on the context so
    orientation-dependent indicators can gate on it (e.g. a down-only charade-positional
    indicator). None means unknown — such indicators then simply do not fire."""
    clue_atoms = atomize_characters("clue", clue_text, number_letters=False)
    answer_atoms = atomize_characters("answer", answer_text,
                                      number_letters=True)
    return WFWAtomContext(
        clue_text=clue_text or "",
        answer_text=answer_text or "",
        clue_atoms=clue_atoms,
        answer_atoms=answer_atoms,
        clue_tokens=tokenize_original("clue", clue_atoms),
        answer_tokens=tokenize_original("answer", answer_atoms),
        direction=(direction or None),
    )


def atomize_characters(stream, text, number_letters=False):
    """Create one stable atom per original character.

    Answer letter positions are one-based and count only alphabetic
    characters.  Spaces, hyphens, apostrophes, and punctuation stay in the
    atom stream for display and provenance but do not consume answer letter
    positions.
    """
    atoms = []
    letter_position = 0
    for index, char in enumerate(text or ""):
        kind = classify_char(char)
        pos = None
        if number_letters and kind == "letter":
            letter_position += 1
            pos = letter_position
        atoms.append(CharAtom(
            atom_id="%s_char_%04d" % (stream, index),
            stream=stream,
            index=index,
            char=char,
            normalized=normalize_char(char),
            kind=kind,
            letter_position=pos,
        ))
    return tuple(atoms)


def tokenize_original(stream, atoms):
    """Tokenise original text without discarding character atoms.

    Words may contain apostrophes and hyphens because the original token must
    remain intact until a later WFW transformation explicitly splits it.
    Standalone punctuation marks become their own tokens.
    """
    tokens = []
    i = 0
    while i < len(atoms):
        atom = atoms[i]
        if atom.kind == "space":
            i += 1
            continue
        start = i
        if _is_word_atom(atom):
            i += 1
            while i < len(atoms) and _is_word_atom(atoms[i]):
                i += 1
            kind = "word"
        else:
            i += 1
            kind = atom.kind
        group = atoms[start:i]
        # A run of only hyphen/quote characters (a standalone dash " – " used as a
        # clue separator, or a lone quote) is NOT a word — it is punctuation. Only a
        # group containing a letter or digit is a word ("well-known", "that's"); this
        # keeps lone dashes from becoming stray "word" tokens that strand as
        # unaccounted and break fodder contiguity.
        if kind == "word" and not any(a.kind in ("letter", "digit") for a in group):
            kind = "punctuation"
        index = len(tokens)
        tokens.append(OriginalToken(
            token_id="%s_tok_%04d" % (stream, index),
            stream=stream,
            index=index,
            text="".join(a.char for a in group),
            start_atom=group[0].atom_id,
            end_atom=group[-1].atom_id,
            atom_ids=tuple(a.atom_id for a in group),
            kind=kind,
        ))
    return tuple(tokens)


def classify_char(char):
    if char.isalpha():
        return "letter"
    if char.isdigit():
        return "digit"
    if char.isspace():
        return "space"
    if char in {"'", '"', "`",
                "‘", "’",       # ‘ ’ typographic single quotes / apostrophe
                "“", "”",       # “ ” typographic double quotes
                "ʼ"}:                # ʼ modifier letter apostrophe
        return "quote"
    if char in {"-", "\u2010", "\u2011", "\u2012", "\u2013", "\u2014"}:
        return "hyphen"
    if char in ".,;:!?()[]{}":
        return "punctuation"
    return "symbol"


def normalize_char(char):
    """The atom's COMPARABLE form: upper-case, with any diacritic FOLDED to the base
    letter (fiancée -> FIANCEE, gratiné -> GRATINE, Señor -> SENOR).

    `char` keeps the original verbatim, so display and reconstruct() are unaffected —
    this field exists only to be compared, and every consumer compares it against
    answer letters, which are plain ASCII. Without the fold, an accented clue word can
    never letter-match its answer: alternate letters of FIANCÉE derived INÉ against
    tiles INE, so the /hs selection gate refused every possible value and the clue was
    unfileable (telegraph 31316 1a FINE). Same double-bind for hidden runs, anagram
    fodder and acrostics through an accented word.

    This is the cryptic convention already stated by core.wordplay.raw(); the fold is
    inlined rather than imported to keep this foundation module dependency-free.

    CANONICAL decomposition (NFD), deliberately NOT compatibility (NFKD). Over the whole
    603k-clue corpus the two differ on exactly one character: 'º' (MASCULINE ORDINAL
    INDICATOR), a mis-scraped degree sign in "90º". Python calls it alphabetic, so it is
    already a letter atom; NFKD would fold it to a real letter O and hand the engines a
    letter source that is not in the clue. NFD leaves it alone — inert, matching nothing,
    exactly as before. æ ð ø ł œ likewise decompose under neither and stay inert.

    Strict no-op for plain ASCII: NFD leaves unaccented characters unchanged. A char that
    decomposes to several base letters falls back to a plain upper-case, preserving the
    one-atom-one-character invariant that selection and the hidden-run scan rely on.
    """
    if not char.isalpha():
        return char
    folded = "".join(c for c in unicodedata.normalize("NFD", char)
                     if not unicodedata.combining(c))
    return folded.upper() if len(folded) == 1 else char.upper()


def fold_letters(text):
    """`text` as bare comparable letters: diacritics folded, upper-case, A-Z only.

    The string-level counterpart of normalize_char, and the ONE definition of "the
    letters of this word/answer" for the web layer — mirrored character-for-character by
    foldLetters() in the /hs grid JS. Three subtly different hand-rolled versions of this
    used to disagree (one kept É, one deleted it and silently shifted every later letter's
    parity, one folded it), which is what made an accented clue unfileable.

    Dropping to A-Z cannot fabricate a letter, only withhold one; it is checked never to
    shorten any answer in the corpus.
    """
    return "".join(c for c in unicodedata.normalize("NFD", text or "").upper()
                   if "A" <= c <= "Z")


def reconstruct(atoms):
    return "".join(atom.char for atom in atoms)


def _is_word_atom(atom):
    return atom.kind in {"letter", "digit", "quote", "hyphen"}
