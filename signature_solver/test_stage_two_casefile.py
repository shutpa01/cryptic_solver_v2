"""Regression tests for the read-only Stage Two case file."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.stage_two_casefile import build_stage_two_casefile


def _has_source(case, text, value):
    return any(
        item["text"] == text and item["value"] == value
        for item in case.source_candidates
    )


def _has_enrichment(case, kind, text, value):
    return any(
        item["kind"] == kind
        and item["text"] == text
        and item["value"] == value
        for item in case.enrichment_candidates
    )


def _unresolved_texts(case):
    return {item["text"] for item in case.unresolved_words}


def run_tests():
    db = RefDB()

    tijuana = build_stage_two_casefile(
        "Note Spanish male taken with a Mexican city",
        "TIJUANA",
        db,
    )
    assert tijuana.status == "answer_fit_needs_review"
    assert any(
        item["text"] == "Mexican city"
        and item["boundary_status"] == "complete_edge_phrase"
        for item in tijuana.definition_candidates
    )
    assert _has_source(tijuana, "Spanish male", "JUAN")
    assert not any(item["text"] == "city" for item in tijuana.source_candidates)
    assert any(
        assembly["kind"] == "charade"
        and [part["value"] for part in assembly["parts"]] == [
            "TI", "JUAN", "A"]
        for assembly in tijuana.assemblies
    )
    assert "taken" in _unresolved_texts(tijuana)

    roc = build_stage_two_casefile(
        "Large reptile heading off fabulous bird",
        "ROC",
        db,
    )
    assert any(
        item["text"] == "fabulous bird"
        for item in roc.definition_candidates
    )
    assert any(
        pair["kind"] == "trim_first_pair"
        and pair["source_text"] == "reptile"
        and pair["indicator_text"] == "heading off"
        and pair["input"] == "CROC"
        and pair["output"] == "ROC"
        for pair in roc.working_pairs
    )
    assert _has_enrichment(roc, "source_phrase_widening",
                           "Large reptile", "CROC")

    totally = build_stage_two_casefile(
        "Small child linked to friend in an absolute way",
        "TOTALLY",
        db,
    )
    assert _has_source(totally, "Small child", "TOT")
    assert any(
        assembly["kind"] == "charade"
        and assembly["output"] == "TOTALLY"
        and [part["value"] for part in assembly["parts"]] == ["TOT", "ALLY"]
        for assembly in totally.assemblies
    )
    assert "linked" in _unresolved_texts(totally)
    assert not _has_enrichment(totally, "source_phrase_widening",
                               "Small child", "TOT")

    hasbeen = build_stage_two_casefile(
        "One no longer relevant base sadly in western half of north London suburb",
        "HASBEEN",
        db,
    )
    assert hasbeen.status == "conditional_needs_enrichment"
    assert any(
        pair["kind"] == "anagram_pair"
        and pair["source_text"] == "base"
        and pair["indicator_text"] == "sadly"
        and pair["output"] == "ASBE"
        for pair in hasbeen.working_pairs
    )
    assert _has_enrichment(hasbeen, "definition_gap",
                           "One no longer relevant", "HASBEEN")
    assert _has_enrichment(hasbeen, "conditional_source_gap",
                           "north London suburb", "HENDON")
    assert any(
        assembly["kind"] == "conditional"
        and assembly["detail"] == "H(ASBE)EN = HASBEEN"
        for assembly in hasbeen.assemblies
    )

    sparrowhawk = build_stage_two_casefile(
        "Rash park (wow) disturbed bird of prey",
        "SPARROWHAWK",
        db,
    )
    assert not any(
        item["kind"] == "source_phrase_widening"
        and item["text"] == "disturbed bird"
        for item in sparrowhawk.enrichment_candidates
    )

    quiet_trip = build_stage_two_casefile(
        "Quiet trip before games producing intense arguments",
        "SHOUTINGMATCHES",
        db,
    )
    assert not _has_enrichment(quiet_trip, "source_phrase_widening",
                               "Quiet trip", "OUTING")

    levies = build_stage_two_casefile(
        "Taxes giving liberal European struggles",
        "LEVIES",
        db,
    )
    assert not any(
        item["kind"] == "source_phrase_widening"
        and item["text"] == "European struggles"
        for item in levies.enrichment_candidates
    )

    inept = build_stage_two_casefile(
        "Clumsy figure missing first point",
        "INEPT",
        db,
    )
    assert not any(
        item["kind"] == "source_phrase_widening"
        and item["text"] == "first point"
        for item in inept.enrichment_candidates
    )

    godown = build_stage_two_casefile(
        "Leave a college sink",
        "GODOWN",
        db,
    )
    assert not any(
        item["kind"] == "source_phrase_widening"
        and item["text"] == "college sink"
        for item in godown.enrichment_candidates
    )

    primordial = build_stage_two_casefile(
        "Very old, private soldiers wearing overly solemn ring",
        "PRIMORDIAL",
        db,
    )
    assert not _has_enrichment(primordial, "source_phrase_widening",
                               "overly solemn ring", "DIAL")

    icon = build_stage_two_casefile(
        "Religious painting carefully symbol cut",
        "ICON",
        db,
    )
    assert not _has_enrichment(icon, "source_phrase_widening",
                               "carefully symbol", "ICON")

    toyish = build_stage_two_casefile(
        "This boy dropping book to dance is playful",
        "TOYISH",
        db,
    )
    assert not _has_enrichment(toyish, "definition_gap",
                               "This boy dropping book", "TOYISH")

    stipulation = build_stage_two_casefile(
        "Sunlit patio needing repair? It's requirement",
        "STIPULATION",
        db,
    )
    assert not _has_enrichment(
        stipulation,
        "definition_gap",
        "Sunlit patio needing repair?",
        "STIPULATION",
    )

    print("Stage Two case file contract passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
