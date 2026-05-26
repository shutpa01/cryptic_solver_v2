"""Regression tests for WFW-native working-out materialisation."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.clue_context import ClueContext, SpanAnnotation, build_clue_context
from signature_solver.token_parse_assembler import assemble_token_parses
from signature_solver.wfw_atoms import build_wfw_atom_context
from signature_solver.wfw_native_assembly import materialise_wfw_assembly
from signature_solver.wfw_unified_solver import solve_wfw_unified


def _with_annotations(context, annotations):
    return ClueContext(
        clue_text=context.clue_text,
        normalized_clue=context.normalized_clue,
        answer=context.answer,
        tokens=context.tokens,
        spans=context.spans,
        definition_candidates=context.definition_candidates,
        wordplay_windows=context.wordplay_windows,
        annotations=tuple(annotations),
        pos_spans=context.pos_spans,
        pos_model_status=context.pos_model_status,
    )


def _first_assembly(clue, answer, db):
    result = solve_wfw_unified(clue, answer, db=db)
    assert result.wfw_assemblies, "expected WFW native assembly"
    assembly = result.wfw_assemblies[0]
    assert assembly.status == "materialised", assembly.objections
    return assembly.as_dict()


def _placed_answer(assembly):
    return "".join(
        placement["answer_letter"] for placement in assembly["placements"])


def run_tests():
    db = RefDB()

    charade = _first_assembly(
        "Sink India before noon, or capitulate?", "GIVEIN", db)
    assert charade["operation"] == "charade"
    assert [block["value"] for block in charade["working_blocks"]] == [
        "GIVE", "I", "N"]
    assert _placed_answer(charade) == "GIVEIN"

    reversal = _first_assembly(
        "Uncultivated land with space to the west", "MOOR", db)
    assert reversal["operation"] == "reversal"
    assert [block["value"] for block in reversal["working_blocks"]] == [
        "ROOM", "MOOR"]
    assert reversal["transformations"][0]["operation"] == "reversal"
    assert _placed_answer(reversal) == "MOOR"

    anagram = _first_assembly(
        "Electronic device disturbed mother's nap", "SMARTPHONE", db)
    assert anagram["operation"] == "anagram"
    assert anagram["working_blocks"][0]["value"] == "MOTHERSNAP"
    assert anagram["transformations"][0]["operation"] == "anagram"
    assert _placed_answer(anagram) == "SMARTPHONE"

    hidden = _first_assembly(
        "Turned over some carriages, salvaging weapon", "ASSEGAI", db)
    assert hidden["operation"] == "hidden_reversed"
    assert hidden["working_blocks"][0]["value"] == "CARRIAGESSALVAGING"
    assert hidden["transformations"][0]["operation"] == "hidden_reversed"
    assert _placed_answer(hidden) == "ASSEGAI"

    elgar = _first_assembly(
        "Article in Madrid newspaper about composer", "ELGAR", db)
    assert elgar["operation"] == "reversal_charade"
    assert [block["value"] for block in elgar["working_blocks"]] == [
        "EL", "GAR"]
    assert any(
        transform["operation"] == "reversal"
        for transform in elgar["transformations"]
    )
    assert _placed_answer(elgar) == "ELGAR"

    dd = _first_assembly("Personal hint", "INTIMATE", db)
    assert dd["operation"] == "double_definition"
    assert [block["text"] for block in dd["working_blocks"]] == [
        "Personal", "hint"]
    assert all(
        transform["operation"] == "defines_whole_answer"
        for transform in dd["transformations"]
    )

    hom_ctx = build_clue_context("heard site", "SIGHT", db, annotate=True)
    hom_ctx = _with_annotations(hom_ctx, [
        SpanAnnotation((0, 1), "heard", "HOM_I", (True,), "test"),
        SpanAnnotation((1, 2), "site", "HOM_F", ("SIGHT",), "test"),
    ])
    hom_parse = assemble_token_parses(hom_ctx)[0]
    hom = materialise_wfw_assembly(
        build_wfw_atom_context("heard site", "SIGHT"), hom_ctx, hom_parse)
    hom_payload = hom.as_dict()
    assert hom_payload["status"] == "materialised"
    assert hom_payload["transformations"][0]["operation"] == "homophone"
    assert _placed_answer(hom_payload) == "SIGHT"

    sub_ctx = build_clue_context(
        "Get up with daughter, not son, and travel.",
        "RIDE", db, annotate=True)
    sub_parse = assemble_token_parses(sub_ctx)[0]
    sub = materialise_wfw_assembly(
        build_wfw_atom_context(
            "Get up with daughter, not son, and travel.", "RIDE"),
        sub_ctx,
        sub_parse,
    )
    sub_payload = sub.as_dict()
    assert sub_payload["status"] == "materialised"
    assert sub_payload["transformations"][0]["operation"] == "substitution"
    assert [block["value"] for block in sub_payload["working_blocks"]] == [
        "RISE", "D", "S", "RIDE"]
    assert _placed_answer(sub_payload) == "RIDE"

    print("WFW native assembly regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
