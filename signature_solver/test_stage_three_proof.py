"""Regression tests for the Stage Three PASS/REVIEW proof gate."""

import os
import sys
from types import SimpleNamespace

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.stage_three_proof import (
    PASS,
    REVIEW,
    build_stage_three_proof,
)
from signature_solver.stage_two_casefile import build_stage_two_casefile


def _check(proof, name):
    return next(item for item in proof.checks if item.name == name)


def _purpose(proof, text):
    return next(item for item in proof.word_purposes if item["text"] == text)


def run_tests():
    db = RefDB()

    clean_case = SimpleNamespace(
        clue_text="Quiet answer",
        answer="PA",
        definition_candidates=({
            "text": "answer",
            "span": (1, 2),
            "wordplay_span": (0, 1),
            "wordplay_text": "Quiet",
            "boundary_status": "complete_edge_phrase",
            "objections": [],
        },),
        working_pairs=(),
        assemblies=({
            "kind": "charade",
            "status": "answer_fit",
            "output": "PA",
            "parts": ({
                "kind": "source",
                "text": "Quiet",
                "span": (0, 1),
                "value": "P",
            }, {
                "kind": "source",
                "text": "a",
                "span": (2, 3),
                "value": "A",
            }),
        },),
        enrichment_candidates=(),
        unresolved_words=(),
    )
    clean = build_stage_three_proof(clean_case)
    assert clean.status == PASS
    assert all(check.status == PASS for check in clean.checks)
    assert _check(clean, "span_integrity").status == PASS
    assert _check(clean, "assembly_order").status == PASS
    assert [link["letter"] for link in clean.atomic_links] == list("PA")

    overlap_case = SimpleNamespace(
        clue_text="Quiet answer",
        answer="PA",
        definition_candidates=clean_case.definition_candidates,
        working_pairs=(),
        assemblies=({
            "kind": "charade",
            "status": "answer_fit",
            "output": "PA",
            "parts": ({
                "kind": "source",
                "text": "answer",
                "span": (1, 2),
                "value": "P",
            }, {
                "kind": "source",
                "text": "a",
                "span": (2, 3),
                "value": "A",
            }),
        },),
        enrichment_candidates=(),
        unresolved_words=(),
    )
    overlap = build_stage_three_proof(overlap_case)
    assert overlap.status == REVIEW
    assert _check(overlap, "span_integrity").status == REVIEW

    weak_definition_case = SimpleNamespace(
        clue_text="maybe answer",
        answer="PA",
        definition_candidates=({
            "text": "maybe answer",
            "span": (0, 2),
            "wordplay_span": (0, 2),
            "wordplay_text": "maybe answer",
            "boundary_status": "partial_phrase_hit",
            "objections": ["db_hit_does_not_cover_full_phrase"],
        },),
        source_candidates=(),
        working_pairs=(),
        assemblies=(),
        enrichment_candidates=(),
        unresolved_words=(),
    )
    weak_definition = build_stage_three_proof(weak_definition_case)
    assert _check(weak_definition, "definition_evidence").status == REVIEW

    out_of_order_case = SimpleNamespace(
        clue_text="Bee first",
        answer="AB",
        definition_candidates=(),
        source_candidates=(),
        working_pairs=(),
        assemblies=({
            "kind": "charade",
            "status": "answer_fit",
            "output": "AB",
            "parts": ({
                "kind": "source",
                "text": "first",
                "span": (1, 2),
                "value": "A",
            }, {
                "kind": "source",
                "text": "Bee",
                "span": (0, 1),
                "value": "B",
            }),
        },),
        enrichment_candidates=(),
        unresolved_words=(),
    )
    out_of_order = build_stage_three_proof(out_of_order_case)
    assert _check(out_of_order, "assembly_order").status == REVIEW

    unsupported_source_case = SimpleNamespace(
        clue_text="Quiet answer",
        answer="PA",
        definition_candidates=clean_case.definition_candidates,
        source_candidates=({
            "text": "Quiet",
            "span": (0, 1),
            "value": "Q",
        },),
        working_pairs=(),
        assemblies=({
            "kind": "charade",
            "status": "answer_fit",
            "output": "PA",
            "parts": ({"text": "Quiet", "span": (0, 1), "value": "P"},),
        },),
        enrichment_candidates=(),
        unresolved_words=(),
    )
    unsupported_source = build_stage_three_proof(unsupported_source_case)
    assert _check(unsupported_source, "source_evidence").status == REVIEW

    bad_pair_case = SimpleNamespace(
        clue_text="Bad mix answer",
        answer="AB",
        definition_candidates=clean_case.definition_candidates,
        working_pairs=({
            "kind": "anagram_pair",
            "source_text": "bad",
            "source_span": (0, 1),
            "indicator_text": "mix",
            "indicator_span": (1, 2),
            "input": "BAD",
            "output": "AB",
            "answer_span": (0, 2),
            "status": "answer_fit",
        },),
        assemblies=(),
        enrichment_candidates=(),
        unresolved_words=(),
    )
    bad_pair = build_stage_three_proof(bad_pair_case)
    assert _check(bad_pair, "mechanism_rules").status == REVIEW

    detached_pair_case = SimpleNamespace(
        clue_text="Bad source words mix answer",
        answer="BAD",
        definition_candidates=clean_case.definition_candidates,
        source_candidates=(),
        working_pairs=({
            "kind": "anagram_pair",
            "source_text": "Bad",
            "source_span": (0, 1),
            "indicator_text": "mix",
            "indicator_span": (3, 4),
            "input": "BAD",
            "output": "BAD",
            "answer_span": (0, 3),
            "status": "answer_fit",
        },),
        assemblies=(),
        enrichment_candidates=(),
        unresolved_words=(),
    )
    detached_pair = build_stage_three_proof(detached_pair_case)
    assert _check(detached_pair, "operation_attachment").status == REVIEW

    tijuana_case = build_stage_two_casefile(
        "Note Spanish male taken with a Mexican city",
        "TIJUANA",
        db,
    )
    tijuana = build_stage_three_proof(tijuana_case)
    assert tijuana.status == REVIEW
    assert _check(tijuana, "answer_assembly").status == PASS
    assert _check(tijuana, "source_evidence").status == PASS
    assert _check(tijuana, "definition_evidence").status == PASS
    assert _check(tijuana, "atomic_coverage").status == PASS
    assert _check(tijuana, "word_purpose_coverage").status == PASS
    assert _check(tijuana, "word_purpose_candidates").status == REVIEW
    assert [link["letter"] for link in tijuana.atomic_links] == list("TIJUANA")
    assert any(item["text"] == "taken" for item in tijuana.unresolved_items)
    assert _purpose(tijuana, "Note")["purpose"] == "answer_source"
    assert _purpose(tijuana, "Mexican")["purpose"] == "definition_phrase_member"
    assert _purpose(tijuana, "taken")["purpose"] == "operation_indicator_candidate"
    assert _purpose(tijuana, "with")["purpose"] == "structural_separator_candidate"
    assert any(
        item["text"] == "taken"
        and item["kind"] == "mechanism_indicator_evidence"
        for item in tijuana.purpose_requests
    )
    assert any(
        item["text"] == "with"
        and item["kind"] == "grammar_separator_evidence"
        for item in tijuana.purpose_requests
    )

    roc_case = build_stage_two_casefile(
        "Large reptile heading off fabulous bird",
        "ROC",
        db,
    )
    roc = build_stage_three_proof(roc_case)
    assert roc.status == REVIEW
    assert _check(roc, "answer_assembly").status == PASS
    assert _check(roc, "operation_evidence").status == PASS
    assert _check(roc, "operation_attachment").status == PASS
    assert _check(roc, "mechanism_rules").status == PASS
    assert _check(roc, "word_purpose_coverage").status == PASS
    assert _check(roc, "word_purpose_candidates").status == REVIEW
    assert _check(roc, "conditional_facts").status == REVIEW
    assert any(
        item["kind"] == "source_phrase_widening"
        and item["text"] == "Large reptile"
        and item["value"] == "CROC"
        for item in roc.required_enrichments
    )
    assert any(
        item["kind"] == "trim_first_pair"
        and item["input"] == "CROC"
        and item["output"] == "ROC"
        for item in roc.transformations
    )

    totally_case = build_stage_two_casefile(
        "Small child linked to friend in an absolute way",
        "TOTALLY",
        db,
    )
    totally = build_stage_three_proof(totally_case)
    assert totally.status == REVIEW
    assert _check(totally, "answer_assembly").status == PASS
    assert _check(totally, "atomic_coverage").status == PASS
    assert _check(totally, "word_purpose_coverage").status == REVIEW
    assert [link["letter"] for link in totally.atomic_links] == list("TOTALLY")
    assert any(item["text"] == "linked" for item in totally.unresolved_items)
    assert _purpose(totally, "linked")["purpose"] == "unresolved_purpose"
    assert _purpose(totally, "to")["purpose"] == "structural_separator_candidate"
    assert _purpose(totally, "in")["purpose"] == "definition_phrase_marker"
    assert _purpose(totally, "absolute")["purpose"] == "definition_phrase_member"
    assert any(
        item["text"] == "linked"
        and item["kind"] == "word_purpose_evidence"
        for item in totally.purpose_requests
    )

    hasbeen_case = build_stage_two_casefile(
        "One no longer relevant base sadly in western half of north London suburb",
        "HASBEEN",
        db,
    )
    hasbeen = build_stage_three_proof(hasbeen_case)
    assert hasbeen.status == REVIEW
    assert _check(hasbeen, "answer_assembly").status == REVIEW
    assert _check(hasbeen, "atomic_coverage").status == REVIEW
    assert _check(hasbeen, "operation_attachment").status == PASS
    assert _check(hasbeen, "mechanism_rules").status == PASS
    assert _check(hasbeen, "word_purpose_coverage").status == PASS
    assert _check(hasbeen, "conditional_facts").status == REVIEW
    assert hasbeen.atomic_links == ()
    assert any(
        item["kind"] == "conditional_source_gap"
        and item["text"] == "north London suburb"
        and item["value"] == "HENDON"
        for item in hasbeen.required_enrichments
    )
    assert any(
        item["kind"] == "anagram_pair"
        and item["source_text"] == "base"
        and item["indicator_text"] == "sadly"
        and item["output"] == "ASBE"
        for item in hasbeen.transformations
    )
    assert _purpose(hasbeen, "One")["purpose"] == "definition_phrase_candidate"
    assert _purpose(hasbeen, "base")["purpose"] == "answer_source"
    assert _purpose(hasbeen, "sadly")["purpose"] == "operation_indicator"
    assert _purpose(hasbeen, "western")["purpose"] == (
        "operation_indicator_modifier_candidate")
    assert _purpose(hasbeen, "north")["purpose"] == "conditional_source_candidate"
    assert _purpose(hasbeen, "in")["purpose"] == "operation_indicator_candidate"
    assert _purpose(hasbeen, "of")["purpose"] == "operation_indicator_candidate"
    assert _check(hasbeen, "word_purpose_candidates").status == REVIEW
    assert any(
        item["text"] == "north London suburb"
        and item["kind"] == "conditional_source_evidence"
        and [atom["text"] for atom in item["atoms"]] == [
            "north", "London", "suburb"]
        for item in hasbeen.purpose_requests
    )
    assert any(
        item["text"] == "One no longer relevant"
        and item["kind"] == "definition_phrase_evidence"
        and [atom["text"] for atom in item["atoms"]] == [
            "One", "no", "longer", "relevant"]
        for item in hasbeen.purpose_requests
    )

    levies_case = build_stage_two_casefile(
        "Taxes giving liberal European struggles",
        "LEVIES",
        db,
    )
    levies = build_stage_three_proof(levies_case)
    assert _check(levies, "answer_assembly").status == PASS
    assert _purpose(levies, "giving")["purpose"] == (
        "definition_separator_candidate")
    assert any(
        item["text"] == "giving"
        and item["kind"] == "definition_separator_evidence"
        for item in levies.purpose_requests
    )

    print("Stage Three proof contract passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
