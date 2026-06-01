"""Regression tests for the WFW character atom foundation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from core.wfw_atoms import (
    build_wfw_atom_context,
    reconstruct,
)


def run_tests():
    ctx = build_wfw_atom_context("Tarzan?", "APE")
    assert reconstruct(ctx.clue_atoms) == "Tarzan?"
    assert [token.text for token in ctx.clue_tokens] == ["Tarzan", "?"]
    assert ctx.clue_tokens[1].kind == "punctuation"
    assert [atom.letter_position for atom in ctx.answer_letter_atoms] == [1, 2, 3]

    possessive = build_wfw_atom_context(
        "president's chair", "CHAIRS")
    assert [token.text for token in possessive.clue_tokens] == [
        "president's", "chair"]
    assert reconstruct(possessive.clue_atoms) == "president's chair"

    quoted = build_wfw_atom_context('for "wealth"', "RICHES")
    assert [token.text for token in quoted.clue_tokens] == [
        "for", '"wealth"']
    assert quoted.clue_tokens[1].atom_ids[0] == "clue_char_0004"
    assert reconstruct(quoted.clue_atoms) == 'for "wealth"'

    golden = build_wfw_atom_context(
        "Go back upset across island after information on adopting elderly dog",
        "GOLDEN RETRIEVER")
    assert reconstruct(golden.answer_atoms) == "GOLDEN RETRIEVER"
    letters = golden.answer_letter_atoms
    assert len(letters) == 15
    assert [atom.char for atom in letters[:6]] == list("GOLDEN")
    assert [atom.letter_position for atom in letters] == list(range(1, 16))
    assert any(atom.char == " " and atom.letter_position is None
               for atom in golden.answer_atoms)

    king = build_wfw_atom_context("borders of KING'S", "KG")
    assert [token.text for token in king.clue_tokens] == [
        "borders", "of", "KING'S"]
    assert reconstruct(king.clue_atoms) == "borders of KING'S"

    print("WFW atom regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
