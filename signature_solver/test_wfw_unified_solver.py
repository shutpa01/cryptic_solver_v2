"""Design-contract tests for the WFW-native unified solver."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.wfw_unified_solver import (
    OPERATION_FAMILIES,
    solve_wfw_unified,
)


def run_tests():
    bare = solve_wfw_unified("Tarzan?", "APE", db=None, assemble=False)
    assert bare.as_dict()["schema"] == "wfw_unified_solver:v1"
    assert bare.legacy_role == "evidence_provider_only"
    assert bare.stages[0].name == "wfw_atomization"
    assert len(bare.atom_context.clue_atoms) == len("Tarzan?")
    assert [token.text for token in bare.atom_context.clue_tokens] == [
        "Tarzan", "?"]
    assert any(
        node.kind == "SURFACE_TOKEN"
        and node.value == "?"
        and node.role == "possible_definition_qualifier"
        for node in bare.candidate_nodes
    )
    assert set(OPERATION_FAMILIES).issubset(set(bare.operation_families))
    assert "cryptic_definition_triage" in bare.operation_families
    assert "spoonerism" in bare.operation_families

    db = RefDB()
    result = solve_wfw_unified(
        "Search advanced into blacksmith's workshop", "FORAGE", db=db)
    assert result.stages[0].name == "wfw_atomization"
    assert result.clue_context is not None
    assert result.grammar_evidence.status in {
        "available", "model_unavailable", "not_requested", "unusable"
    } or result.grammar_evidence.status.startswith("available")
    assert any(
        node.kind == "DEF_BLOCK"
        and node.role == "defines_whole_answer"
        and node.value == "Search"
        for node in result.candidate_nodes
    )
    assert any(
        node.kind == "SURFACE_TOKEN"
        and node.value == "into"
        and node.role == "connector_candidate_requires_license"
        for node in result.candidate_nodes
    )
    assert any(item.atom_ids for item in result.evidence)
    assert any(
        stage.name == "answer_guided_assembly"
        for stage in result.stages
    )
    assert any(
        stage.name == "verification"
        for stage in result.stages
    )

    payload = result.as_dict()
    assert payload["legacy_role"] == "evidence_provider_only"
    assert payload["atom_context"]["clue_text"] == (
        "Search advanced into blacksmith's workshop")
    assert len(payload["atom_context"]["clue_atoms"]) == len(
        "Search advanced into blacksmith's workshop")
    assert payload["candidate_nodes"]
    assert payload["operation_families"] == list(OPERATION_FAMILIES)
    assert payload["wfw_assemblies"], (
        "unified result must preserve WFW working-out, not just token parse")
    assembly = payload["wfw_assemblies"][0]
    assert assembly["status"] == "materialised"
    assert assembly["operation"] == "container"
    assert [block["value"] for block in assembly["working_blocks"]] == [
        "FORGE", "A", "FORAGE"]
    assert assembly["transformations"][0]["operation"] == "container"
    assert len(assembly["placements"]) == len("FORAGE")
    assert "".join(
        placement["answer_letter"] for placement in assembly["placements"]
    ) == "FORAGE"
    assert assembly["placements"][3]["source_block_id"] == "wfw_src_inner"

    print("WFW unified solver contract passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
