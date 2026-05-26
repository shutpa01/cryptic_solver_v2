"""Regression tests for canonical clue token/span context."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.clue_context import (
    build_clue_context,
    with_wordplay_annotations,
)
from signature_solver.db import RefDB
from signature_solver.pos_span_model import POSSpan
from signature_solver.wfw_unified_solver import solve_wfw_unified


class FakePOSModel:
    def extract_spans(self, normalized_text, clue_tokens):
        assert normalized_text == "Search advanced into blacksmith's workshop"
        return (
            POSSpan(
                start=3,
                end=5,
                text="blacksmith's workshop",
                normalized="blacksmiths workshop",
                label="noun_chunk",
                root_index=4,
                root_text="workshop",
                pos_tags=("NOUN", "NOUN"),
                dependencies=("compound", "ROOT"),
                source="test_pos_model",
            ),
        )


class EdgePhrasePOSModel:
    def extract_spans(self, normalized_text, clue_tokens):
        if normalized_text == "Note Spanish male taken with a Mexican city":
            return (
                POSSpan(
                    start=5,
                    end=8,
                    text="a Mexican city",
                    normalized="a mexican city",
                    label="noun_chunk",
                    root_index=7,
                    root_text="city",
                ),
            )
        if normalized_text == "Luxury dwelling place offers shelter":
            return (
                POSSpan(
                    start=0,
                    end=3,
                    text="Luxury dwelling place",
                    normalized="luxury dwelling place",
                    label="noun_chunk",
                    root_index=2,
                    root_text="place",
                ),
            )
        return ()


class FakeDefinitionDB:
    def is_definition_of(self, phrase, answer):
        return (
            phrase.lower(), answer.upper()
        ) in {
            ("mexican city", "TIJUANA"),
            ("luxury dwelling", "HOUSE"),
        }


def run_tests():
    db = RefDB()
    ctx = build_clue_context(
        "Search advanced into blacksmith's workshop",
        "FORAGE", db, annotate=True)

    assert [t.text for t in ctx.tokens] == [
        "Search", "advanced", "into", "blacksmith's", "workshop"
    ]
    assert ctx.atom_context is not None
    assert ctx.atom_context.clue_text == "Search advanced into blacksmith's workshop"
    assert ctx.atom_context.answer_text == "FORAGE"
    assert ctx.tokens[0].atom_ids == tuple(
        atom.atom_id for atom in ctx.atom_context.clue_atoms[0:6])
    assert len(ctx.atom_context.answer_letter_atoms) == 6
    assert ctx.tokens[3].start_char < ctx.tokens[3].end_char
    assert ctx.span_text(3, 5) == "blacksmith's workshop"
    assert any(
        span.start == 3 and span.end == 5 and span.text == "blacksmith's workshop"
        for span in ctx.spans
    )
    candidate = next(
        candidate for candidate in ctx.definition_candidates
        if candidate.definition_span.as_tuple() == (0, 1)
        and candidate.wordplay_span.as_tuple() == (1, 5)
    )
    assert candidate.definition_span.as_tuple() == (0, 1)
    assert candidate.wordplay_span.as_tuple() == (1, 5)
    assert candidate.as_legacy_tuple() == (
        "Search", ["advanced", "into", "blacksmith's", "workshop"])

    assert any(
        annotation.span == (1, 2)
        and annotation.token == "ABR_F"
        and "A" in annotation.values
        for annotation in ctx.annotations
    )
    assert any(
        annotation.span == (2, 3)
        and annotation.token == "CON_I"
        for annotation in ctx.annotations
    )

    edge_ctx = build_clue_context(
        "Note Spanish male taken with a Mexican city",
        "TIJUANA",
        FakeDefinitionDB(),
        annotate=False,
        pos_model=EdgePhrasePOSModel())
    mexican_candidates = [
        candidate for candidate in edge_ctx.definition_candidates
        if candidate.definition_span.text == "Mexican city"
    ]
    assert mexican_candidates
    assert all(
        candidate.wordplay_span.text == "Note Spanish male taken with a"
        for candidate in mexican_candidates
    )
    assert all(
        candidate.boundary_status == "complete_edge_phrase"
        for candidate in mexican_candidates
    )
    assert edge_ctx.annotations == ()
    edge_wordplay_ctx = with_wordplay_annotations(edge_ctx, db)
    assert not any(
        annotation.text == "city"
        for annotation in edge_wordplay_ctx.annotations
    )
    assert any(
        annotation.text == "Spanish male"
        and annotation.token == "SYN_F"
        and "JUAN" in annotation.values
        for annotation in edge_wordplay_ctx.annotations
    )

    partial_ctx = build_clue_context(
        "Luxury dwelling place offers shelter",
        "HOUSE",
        FakeDefinitionDB(),
        annotate=False,
        pos_model=EdgePhrasePOSModel())
    partial = next(
        candidate for candidate in partial_ctx.definition_candidates
        if candidate.definition_span.text == "Luxury dwelling place")
    assert partial.db_definition_span.text == "Luxury dwelling"
    assert partial.wordplay_span.text == "offers shelter"
    assert partial.boundary_status == "partial_phrase_hit"
    assert "db_hit_does_not_cover_full_phrase" in partial.objections

    wfw_result = solve_wfw_unified(
        ctx.clue_text, ctx.answer, db, assemble=False, clue_context=ctx)
    assert wfw_result.clue_context is ctx
    assert wfw_result.atom_context is ctx.atom_context
    assert any(
        stage.stage_id == "stage_02_context"
        and "reused shared" in stage.detail
        for stage in wfw_result.stages
    )

    pos_ctx = build_clue_context(
        "Search advanced into blacksmith's workshop",
        "FORAGE", db, annotate=False, pos_model=FakePOSModel())
    assert pos_ctx.pos_model_status == "available:injected"
    assert pos_ctx.pos_spans[0].start == 3
    assert pos_ctx.pos_spans[0].end == 5
    assert pos_ctx.pos_spans[0].label == "noun_chunk"
    assert any(
        span.start == 3
        and span.end == 5
        and span.kind == "pos_noun_chunk"
        for span in pos_ctx.spans
    )

    elgar_ctx = build_clue_context(
        "Article in Madrid newspaper about composer",
        "ELGAR", db, annotate=True)
    assert any(
        annotation.span == (0, 3)
        and annotation.token == "ABR_F"
        and "EL" in annotation.values
        for annotation in elgar_ctx.annotations
    )

    nudists_ctx = build_clue_context(
        "Detectives probing crazy people with no clothes on",
        "NUDISTS", db, annotate=True)
    assert any(
        candidate.definition_span.as_tuple() == (3, 8)
        and candidate.definition_span.text == "people with no clothes on"
        and candidate.wordplay_span.as_tuple() == (0, 3)
        for candidate in nudists_ctx.definition_candidates
    )

    game_shows_ctx = build_clue_context(
        "Programmes in Georgia question occupying military dining hall",
        "game shows", db, annotate=True)
    assert any(
        candidate.definition_span.as_tuple() == (0, 1)
        and candidate.definition_span.text == "Programmes"
        for candidate in game_shows_ctx.definition_candidates
    )
    assert any(
        annotation.span == (5, 8)
        and annotation.text == "military dining hall"
        and annotation.token == "SYN_F"
        and "MESS" in annotation.values
        and annotation.source.startswith("pos_span_lift")
        for annotation in game_shows_ctx.annotations
    )

    print("Clue context regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
