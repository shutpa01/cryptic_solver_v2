"""Regression tests for WFW working blocks and transformations."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.wfw_atoms import build_wfw_atom_context
from signature_solver.wfw_working import (
    anagram_block,
    container_block,
    controller_atom_ids,
    place_anagram,
    place_charade,
    place_container,
    reverse_block,
    source_block_from_tokens,
    trim_block,
)


def run_tests():
    ctx = build_wfw_atom_context(
        "State education rejected by Bordeaux, nearly", "DECLARE")

    assert [token.text for token in ctx.clue_tokens] == [
        "State", "education", "rejected", "by", "Bordeaux", ",", "nearly"
    ]

    education = source_block_from_tokens(
        ctx, "piece_education_ed", (1,), "ED", "abbreviation")
    assert education.value == "ED"
    assert education.source_token_ids == ("clue_tok_0001",)

    rejected_atoms = controller_atom_ids(ctx, (2,))
    de, reverse = reverse_block(
        education, "piece_education_reversed", rejected_atoms)
    assert de.value == "DE"
    assert reverse.operation == "reversal"
    assert reverse.input_block_ids == ("piece_education_ed",)
    assert reverse.controller_atom_ids == rejected_atoms

    bordeaux = source_block_from_tokens(
        ctx, "piece_bordeaux_claret", (4,), "CLARET", "synonym")
    nearly_atoms = controller_atom_ids(ctx, (6,))
    clare, trim = trim_block(
        bordeaux, "piece_bordeaux_nearly", "last", nearly_atoms)
    assert clare.value == "CLARE"
    assert trim.operation == "trim_last"
    assert trim.input_block_ids == ("piece_bordeaux_claret",)
    assert trim.removed_char_atom_ids == ("piece_bordeaux_claret_char_0005",)
    assert trim.controller_atom_ids == nearly_atoms

    placements = place_charade(ctx.answer_atoms, (de, clare))
    assert [p.answer_position for p in placements] == list(range(1, 8))
    assert "".join(p.answer_letter for p in placements) == "DECLARE"
    assert [p.source_block_id for p in placements[:2]] == [
        "piece_education_reversed", "piece_education_reversed"]
    assert [p.source_block_id for p in placements[2:]] == [
        "piece_bordeaux_nearly"] * 5

    try:
        place_charade(ctx.answer_atoms, (clare, de))
    except ValueError as exc:
        assert "CLAREDE != DECLARE" in str(exc)
    else:
        raise AssertionError("wrong order must not publish a placement")

    real = build_wfw_atom_context("See each lake during hike", "REALISE")
    hike = source_block_from_tokens(
        real, "piece_hike_rise", (4,), "RISE", "synonym")
    each = source_block_from_tokens(
        real, "piece_each_ea", (1,), "EA", "abbreviation")
    lake = source_block_from_tokens(
        real, "piece_lake_l", (2,), "L", "abbreviation")
    during = controller_atom_ids(real, (3,))
    output, transform, insert_pos = container_block(
        hike, (each, lake), "piece_realise", "REALISE", during)
    assert output.value == "REALISE"
    assert transform.operation == "container"
    assert transform.input_block_ids == (
        "piece_hike_rise", "piece_each_ea", "piece_lake_l")
    assert insert_pos == 1
    placements = place_container(real.answer_atoms, hike, (each, lake), insert_pos)
    assert "".join(p.answer_letter for p in placements) == "REALISE"
    assert [p.source_block_id for p in placements] == [
        "piece_hike_rise",
        "piece_each_ea",
        "piece_each_ea",
        "piece_lake_l",
        "piece_hike_rise",
        "piece_hike_rise",
        "piece_hike_rise",
    ]

    selected = build_wfw_atom_context(
        "Deletes broadcast covering Conservative getting appointed",
        "SELECTED")
    deletes = source_block_from_tokens(
        selected, "piece_deletes", (0,), "DELETES", "anagram_fodder")
    conservative = source_block_from_tokens(
        selected, "piece_conservative", (3,), "C", "abbreviation")
    broadcast = controller_atom_ids(selected, (1,))
    output, transform = anagram_block(
        (deletes, conservative), "piece_selected", "SELECTED", broadcast)
    assert output.value == "SELECTED"
    assert transform.operation == "anagram"
    assert transform.controller_atom_ids == broadcast
    placements = place_anagram(
        selected.answer_atoms, (deletes, conservative))
    assert "".join(p.answer_letter for p in placements) == "SELECTED"
    assert placements[4].source_block_id == "piece_conservative"

    print("WFW working-block regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
    place_anagram,
