"""Regression tests for adapting WFW proof records to the designed display."""

import json
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.db import RefDB
from signature_solver.stage_three_proof import build_stage_three_proof
from signature_solver.stage_two_casefile import build_stage_two_casefile
from signature_solver.wfw_display_adapter import display_from_stage_three_proof
from signature_solver.wfw_display_adapter import display_from_wfw_proof_attempt
from signature_solver.wfw_display_adapter import display_from_missing_wfw
from signature_solver.wfw_proof import build_wfw_proof_from_obase
from signature_solver.wfw_proof_store import write_wfw_proof_attempt
from signature_solver.wfw_proof_store import get_latest_wfw_proof_attempt


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB = os.path.join(ROOT, "data", "clues_master.db")


def _load_17a():
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """SELECT c.id, c.source, c.puzzle_number, c.clue_text,
                      c.answer, c.definition, c.ai_explanation,
                      se.definition_text, se.components
               FROM clues c
               JOIN structured_explanations se ON se.clue_id = c.id
               WHERE c.source = 'telegraph'
                 AND c.puzzle_number = '31243'
                 AND c.clue_number = '17'
                 AND c.direction = 'across'
               ORDER BY se.id DESC
               LIMIT 1"""
        ).fetchone()
    finally:
        conn.close()


def _load_clue(clue_number, direction):
    conn = sqlite3.connect(DB, timeout=30)
    conn.row_factory = sqlite3.Row
    try:
        return conn.execute(
            """SELECT c.id, c.source, c.puzzle_number, c.clue_text,
                      c.answer, c.definition, c.ai_explanation,
                      se.definition_text, se.components
               FROM clues c
               JOIN structured_explanations se ON se.clue_id = c.id
               WHERE c.source = 'telegraph'
                 AND c.puzzle_number = '31243'
                 AND c.clue_number = ?
                 AND c.direction = ?
               ORDER BY se.id DESC
               LIMIT 1""",
            (clue_number, direction),
        ).fetchone()
    finally:
        conn.close()


def run_tests():
    row = _load_17a()
    assert row is not None
    proof = build_wfw_proof_from_obase(
        row["clue_text"],
        row["answer"],
        json.loads(row["components"]),
        ai_explanation=row["ai_explanation"] or "",
        definition_text=row["definition_text"] or row["definition"],
    )

    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    write_wfw_proof_attempt(
        row["id"], row["source"], row["puzzle_number"], proof, conn=conn)
    attempt = get_latest_wfw_proof_attempt(row["id"], conn=conn)
    display = display_from_wfw_proof_attempt(attempt)

    assert display is not None
    assert display["answer"] == "DECLARE"
    assert display["operations"][0]["operation"] == "charade"
    assert "ED reversed = DE" in display["operations"][0]["detail"]
    assert [link["letter"] for link in display["answer_links"]] == list("DECLARE")
    assert display["answer_links"][0]["source_text"] == "education"
    assert display["answer_links"][2]["source_text"] == "Bordeaux,"
    assert display["blocks"][0]["kind"] == "DEF_BLOCK"
    assert display["blocks"][1]["kind"] == "SOURCE_BLOCK"
    assert any(block["text"] == "rejected" for block in display["blocks"])
    assert any(block["text"] == "nearly" for block in display["blocks"])

    ideas = _load_clue("3", "down")
    hidden_proof = build_wfw_proof_from_obase(
        ideas["clue_text"],
        ideas["answer"],
        json.loads(ideas["components"]),
        ai_explanation=ideas["ai_explanation"] or "",
        definition_text=ideas["definition_text"] or ideas["definition"],
    )
    write_wfw_proof_attempt(
        ideas["id"], ideas["source"], ideas["puzzle_number"],
        hidden_proof, conn=conn)
    hidden_attempt = get_latest_wfw_proof_attempt(ideas["id"], conn=conn)
    hidden_display = display_from_wfw_proof_attempt(hidden_attempt)
    assert hidden_display["blocks"][0]["role"] == "hidden_fodder"
    assert hidden_display["hidden_segments"]["display_hidden"] == "id eas"
    assert [link["letter"] for link in hidden_display["answer_links"]] == list("IDEAS")

    originated = _load_clue("1", "across")
    originated_proof = build_wfw_proof_from_obase(
        originated["clue_text"],
        originated["answer"],
        json.loads(originated["components"]),
        ai_explanation=originated["ai_explanation"] or "",
        definition_text=(
            originated["definition_text"] or originated["definition"]
        ),
    )
    assert originated_proof.status == "wfw_proven"
    write_wfw_proof_attempt(
        originated["id"], originated["source"], originated["puzzle_number"],
        originated_proof, conn=conn)
    originated_attempt = get_latest_wfw_proof_attempt(
        originated["id"], conn=conn)
    originated_display = display_from_wfw_proof_attempt(originated_attempt)
    block_texts = [block["text"] for block in originated_display["blocks"]]
    assert "Started" in block_texts
    assert "with" in block_texts
    assert "developing" in block_texts
    assert block_texts == [
        "Started", "to", "grin", "with", "idea", "developing",
    ]

    selected = _load_clue("11", "across")
    review_proof = build_wfw_proof_from_obase(
        selected["clue_text"],
        selected["answer"],
        json.loads(selected["components"]),
        ai_explanation=selected["ai_explanation"] or "",
        definition_text=selected["definition_text"] or selected["definition"],
    )
    assert review_proof.status == "wfw_review"
    write_wfw_proof_attempt(
        selected["id"], selected["source"], selected["puzzle_number"],
        review_proof, conn=conn)
    review_display = display_from_wfw_proof_attempt(
        get_latest_wfw_proof_attempt(selected["id"], conn=conn))
    assert review_display is not None
    assert review_display["status"] == "wfw_review"
    assert any(
        block["role"] == "unaccounted"
        for block in review_display["blocks"]
    )

    missing = display_from_missing_wfw(
        "Residents name a town in Cornwall? Not at first", "NATIVES")
    assert missing["status"] == "wfw_review"
    assert missing["operations"][0]["operation"] == "review"
    assert any(block["role"] == "unaccounted" for block in missing["blocks"])

    ref_db = RefDB()
    tijuana_proof = build_stage_three_proof(build_stage_two_casefile(
        "Note Spanish male taken with a Mexican city",
        "TIJUANA",
        ref_db,
    ))
    tijuana_display = display_from_stage_three_proof(tijuana_proof)
    assert tijuana_display["status"] == "wfw_review"
    assert tijuana_display["operations"][0]["detail"] == (
        "TI + JUAN + A = TIJUANA")
    assert [link["letter"] for link in tijuana_display["answer_links"]] == list(
        "TIJUANA")
    assert all("source_input_value" in link
               for link in tijuana_display["answer_links"])
    assert any(block["kind"] == "DEF_BLOCK"
               and block["text"] == "Mexican city"
               for block in tijuana_display["blocks"])
    assert any(block["kind"] == "REVIEW_BLOCK"
               and block["text"] == "taken"
               and block["role"] == "unaccounted"
               for block in tijuana_display["blocks"])
    assert any(item["type"] == "purpose"
               and item["word"] == "taken"
               and item["kind"] == "mechanism_indicator_evidence"
               for item in tijuana_display["missing_enrichments"])
    assert any(item["type"] == "purpose"
               and item["word"] == "with"
               and item["kind"] == "grammar_separator_evidence"
               for item in tijuana_display["missing_enrichments"])
    assert "required_enrichments" not in tijuana_display

    roc_proof = build_stage_three_proof(build_stage_two_casefile(
        "Large reptile heading off fabulous bird",
        "ROC",
        ref_db,
    ))
    roc_display = display_from_stage_three_proof(roc_proof)
    assert roc_display["status"] == "wfw_review"
    assert any(item["word"] == "Large reptile"
               and item["synonym"] == "CROC"
               for item in roc_display["missing_enrichments"])
    assert any(item["type"] == "purpose"
               and item["word"] == "Large"
               and item["kind"] == "source_phrase_evidence"
               for item in roc_display["missing_enrichments"])
    assert any(block["kind"] == "OP_BLOCK"
               and block["text"] == "heading off"
               for block in roc_display["blocks"])

    uncages_proof = build_stage_three_proof(build_stage_two_casefile(
        "Releases new actor Nicolas in America",
        "UNCAGES",
        ref_db,
    ))
    uncages_display = display_from_stage_three_proof(uncages_proof)
    uncages_words = [
        block["text"] for block in uncages_display["blocks"]
        if block.get("text")
    ]
    assert uncages_words == [
        "Releases", "new", "actor", "Nicolas", "in", "America",
    ]
    assert uncages_display["blocks"][0]["kind"] == "DEF_BLOCK"
    assert all(
        block["role"] == "unaccounted"
        for block in uncages_display["blocks"][1:]
    )

    hasbeen_proof = build_stage_three_proof(build_stage_two_casefile(
        "One no longer relevant base sadly in western half of north London suburb",
        "HASBEEN",
        ref_db,
    ))
    hasbeen_display = display_from_stage_three_proof(hasbeen_proof)
    assert hasbeen_display["status"] == "wfw_review"
    assert any(block["kind"] == "SOURCE_BLOCK"
               and block["text"] == "base"
               and block["input_value"] == "BASE"
               and block["value"] == "ASBE"
               for block in hasbeen_display["blocks"])
    assert any(block["kind"] == "OP_BLOCK"
               and block["text"] == "sadly"
               for block in hasbeen_display["blocks"])
    assert any(item["type"] == "purpose"
               and item["word"] == "One no longer relevant"
               and item["kind"] == "definition_phrase_evidence"
               for item in hasbeen_display["missing_enrichments"])
    assert any(item["type"] == "purpose"
               and item["word"] == "north London suburb"
               and item["kind"] == "conditional_source_evidence"
               for item in hasbeen_display["missing_enrichments"])

    print("WFW display adapter regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
