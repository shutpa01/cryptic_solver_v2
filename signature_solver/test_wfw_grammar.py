"""Regression tests for WFW grammar evidence alignment."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.clue_context import build_clue_context
from signature_solver.db import RefDB
from signature_solver.pos_span_model import POSSpan
from signature_solver.wfw_atoms import build_wfw_atom_context
from signature_solver.wfw_grammar import grammar_evidence_from_clue_context


class FakePOSModel:
    def extract_spans(self, normalized_text, clue_tokens):
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


def run_tests():
    clue = "Search advanced into blacksmith's workshop"
    atom_ctx = build_wfw_atom_context(clue, "FORAGE")
    clue_ctx = build_clue_context(
        clue, "FORAGE", RefDB(), annotate=False, pos_model=FakePOSModel())

    evidence = grammar_evidence_from_clue_context(atom_ctx, clue_ctx)
    assert evidence.status == "available"
    assert len(evidence.spans) == 1
    span = evidence.spans[0]
    assert span.label == "noun_chunk"
    assert span.token_indices == (3, 4)
    assert span.token_ids == ("clue_tok_0003", "clue_tok_0004")
    assert span.root_token_index == 4
    assert span.root_token_id == "clue_tok_0004"
    assert span.pos_tags == ("NOUN", "NOUN")
    assert span.dependencies == ("compound", "ROOT")
    assert "clue_char_0021" in span.atom_ids

    punct_clue = "State education rejected by Bordeaux, nearly"
    punct_atom_ctx = build_wfw_atom_context(punct_clue, "DECLARE")
    punct_clue_ctx = build_clue_context(
        punct_clue, "DECLARE", RefDB(), annotate=False)
    punct_evidence = grammar_evidence_from_clue_context(
        punct_atom_ctx, punct_clue_ctx)
    assert punct_evidence.status != "unusable"

    print("WFW grammar evidence regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
