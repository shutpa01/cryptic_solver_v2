"""Regression tests for the first GT2 candidate-generator integration."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver import gt2_candidate_generator as gt2
from signature_solver.clue_context import build_clue_context
from signature_solver.db import RefDB
from signature_solver.pos_span_model import POSSpan
from signature_solver.solver import (
    _normalize_clue,
    extract_definition_candidates,
    solve_clue,
)


class FakePOSModel:
    def extract_spans(self, normalized_text, clue_tokens):
        return (
            POSSpan(
                start=3,
                end=5,
                text="blacksmith's workshop",
                normalized="blacksmith's workshop",
                label="noun_chunk",
                root_index=4,
                root_text="workshop",
            ),
        )


def run_tests():
    db = RefDB()
    clue = "Search advanced into blacksmith's workshop"
    answer = "FORAGE"

    old_verify = gt2._verify_span_value
    gt2._verify_span_value = (
        lambda clue_text, phrase, value, ans, role_hint=None:
        phrase.lower() == "blacksmith's workshop" and value == "FORGE"
    )
    try:
        words = _normalize_clue(clue).strip().split()
        candidates = extract_definition_candidates(words, answer, db)
        ctx = build_clue_context(clue, answer, db)
        bundles = gt2.generate_gt2_candidates(
            clue, answer, db, candidates, clue_context=ctx)

        bundle = next(
            (b for b in bundles
             if b.overlay_synonyms == {"blacksmith's workshop": ["FORGE"]}),
            None)
        assert bundle is not None, "expected FORAGE GT2 bundle"
        assert bundle.overlay_synonyms == {"blacksmith's workshop": ["FORGE"]}
        assert bundle.operation == "container"
        assert any(
            n.kind == "RELATION_BLOCK"
            and n.span == (2, 3)
            and n.span_space == "full_clue_tokens"
            for n in bundle.nodes
        )
        assert any(
            n.kind == "SOURCE_BLOCK"
            and n.text == "blacksmith's workshop"
            and n.span == (3, 5)
            and n.span_space == "full_clue_tokens"
            for n in bundle.nodes
        )
        assert any(g["kind"] == "missing_span_value" for g in bundle.gaps)

        sr = solve_clue(clue, answer, db)
        assert sr.high_confidence, "WFW solver should verify the clue"
        assert hasattr(sr, "clue_context")
        assert sr.solver_authority == "wfw_unified"
        assert hasattr(sr, "wfw_unified_result")
        assert getattr(sr, "token_parses", None), (
            "WFW-solved FORAGE should carry a tokenised parse")
        forage_parse = sr.token_parses[0].as_dict()
        assert forage_parse["operation"] == "container"
        assert any(
            block["span"] == [3, 5]
            and block["value"] == "FORGE"
            for block in forage_parse["blocks"]
        )
    finally:
        gt2._verify_span_value = old_verify

    old_verify = gt2._verify_span_value
    gt2._verify_span_value = (
        lambda clue_text, phrase, value, ans, role_hint=None: False
    )
    try:
        words = _normalize_clue(clue).strip().split()
        candidates = extract_definition_candidates(words, answer, db)
        ctx = build_clue_context(
            clue, answer, db, pos_model=FakePOSModel())
        bundles = gt2.generate_gt2_candidates(
            clue, answer, db, candidates, clue_context=ctx)
        assert len(bundles) >= 1, (
            "POS chunk plus answer complement should preserve the phrase atom")
        assert any(
            bundle.gaps
            and bundle.gaps[0]["kind"] == "missing_span_value"
            and bundle.gaps[0]["text"] == "blacksmith's workshop"
            for bundle in bundles
        )
    finally:
        gt2._verify_span_value = old_verify

    old_verify = gt2._verify_span_value
    gt2._verify_span_value = (
        lambda clue_text, phrase, value, ans, role_hint=None:
        phrase.lower() == "group of three" and value == "TRIO"
    )
    try:
        tripoli = solve_clue(
            "Group of three left one securing parking in capital",
            "Tripoli", db)
        assert getattr(tripoli, "token_parses", None), (
            "GT2 span enrichment should let atomic assembly verify TRIPOLI")
        assert tripoli.span_value_candidates == [("Group of three", "TRIO")]
        tripoli_parse = tripoli.token_parses[0].as_dict()
        assert tripoli_parse["operation"] == "container_charade"
        assert any(
            block["span"] == [0, 3]
            and block["value"] == "TRIPO"
            for block in tripoli_parse["blocks"]
        )
    finally:
        gt2._verify_span_value = old_verify

    imam = solve_clue(
        "Religious leader's Mass supporting current postgraduate",
        "IMAM", db)
    assert imam.high_confidence, "IMAM baseline should solve"
    imam_nodes = [
        node.as_dict()
        for bundle in getattr(imam, "gt2_evidence_bundles", [])
        for node in bundle.nodes
    ]
    assert any(
        node["kind"] == "OP_BLOCK"
        and node["operation"] == "reversed"
        and node["value"] == "AM"
        for node in imam_nodes
    ), "IMAM should preserve MA -> AM as transform evidence"

    elgar = solve_clue(
        "Article in Madrid newspaper about composer",
        "ELGAR", db)
    assert elgar.high_confidence, "ELGAR baseline should solve"
    elgar_nodes = [
        node.as_dict()
        for bundle in getattr(elgar, "gt2_evidence_bundles", [])
        for node in bundle.nodes
    ]
    assert any(
        node["text"] == "Article in Madrid"
        and node["value"] == "EL"
        and node["span"] == [0, 3]
        and node["span_space"] == "full_clue_tokens"
        and "phrase_span" in node["evidence"]
        for node in elgar_nodes
    ), "ELGAR should preserve Article in Madrid -> EL phrase evidence"

    print("GT2 candidate-generator regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
