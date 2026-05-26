"""Regression tests for WFW parse formatting."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.clue_context import (
    ClueContext,
    SpanAnnotation,
    build_clue_context,
)
from signature_solver.db import RefDB
from signature_solver.token_parse_assembler import assemble_token_parses
from signature_solver.wfw_formatter import format_token_parse_for_wfw


def _with_annotation(context, annotation):
    return ClueContext(
        clue_text=context.clue_text,
        normalized_clue=context.normalized_clue,
        answer=context.answer,
        tokens=context.tokens,
        spans=context.spans,
        definition_candidates=context.definition_candidates,
        wordplay_windows=context.wordplay_windows,
        annotations=context.annotations + (annotation,),
    )


def run_tests():
    db = RefDB()
    ctx = build_clue_context(
        "Search advanced into blacksmith's workshop",
        "FORAGE", db, annotate=True)
    ctx = _with_annotation(ctx, SpanAnnotation(
        span=(3, 5),
        text="blacksmith's workshop",
        token="SYN_F",
        values=("FORGE",),
        source="gt2_span_value",
    ))
    parse = assemble_token_parses(ctx)[0]
    wfw = format_token_parse_for_wfw(ctx, parse)

    assert wfw["tokens"][0]["roles"][0]["kind"] == "DEF_BLOCK"
    assert wfw["tokens"][1]["roles"][0]["role"] == "inner"
    assert wfw["tokens"][2]["roles"][0]["role"] == "container_indicator"
    assert wfw["tokens"][3]["roles"][0]["role"] == "outer"
    assert wfw["tokens"][4]["roles"][0]["role"] == "outer"

    links = wfw["answer_links"]
    assert [link["letter"] for link in links] == list("FORAGE")
    assert links[0]["source_text"] == "blacksmith's workshop"
    assert links[3]["source_text"] == "advanced"
    assert links[3]["source_span"] == [1, 2]
    assert links[4]["source_text"] == "blacksmith's workshop"

    comb_pos_ctx = build_clue_context(
        "horse taking initially mango", "COMB", db, annotate=True)
    comb_pos_ctx = _with_annotation(comb_pos_ctx, SpanAnnotation(
        span=(0, 1), text="horse", token="SYN_F",
        values=("COB",), source="test"))
    comb_pos_ctx = _with_annotation(comb_pos_ctx, SpanAnnotation(
        span=(1, 2), text="taking", token="CON_I",
        values=(True,), source="test"))
    comb_pos_ctx = _with_annotation(comb_pos_ctx, SpanAnnotation(
        span=(2, 3), text="initially", token="POS_I_FIRST",
        values=(True,), source="test"))
    comb_pos_parse = assemble_token_parses(comb_pos_ctx)[0]
    comb_pos_wfw = format_token_parse_for_wfw(comb_pos_ctx, comb_pos_parse)
    assert comb_pos_wfw["operations"][0]["operation"] == "container"
    assert [link["letter"] for link in comb_pos_wfw["answer_links"]] == list("COMB")

    code_ctx = build_clue_context(
        "Set of expectations about lyric poem", "CODE", db, annotate=True)
    code_parse = assemble_token_parses(code_ctx)[0]
    code_wfw = format_token_parse_for_wfw(code_ctx, code_parse)
    assert code_wfw["answer_links"][0]["source_text"] == "about"
    assert code_wfw["answer_links"][1]["source_text"] == "lyric poem"
    assert code_wfw["answer_links"][3]["source_text"] == "lyric poem"

    hidden_ctx = build_clue_context(
        "Turned over some carriages, salvaging weapon",
        "ASSEGAI", db, annotate=True)
    hidden_parse = assemble_token_parses(hidden_ctx)[0]
    hidden_wfw = format_token_parse_for_wfw(hidden_ctx, hidden_parse)
    assert hidden_wfw["hidden_segments"]["prefix"] == "CARR"
    assert hidden_wfw["hidden_segments"]["hidden"] == "IAGESSA"
    assert hidden_wfw["hidden_segments"]["suffix"] == "LVAGING"
    assert hidden_wfw["hidden_segments"]["display_prefix"] == "carr"
    assert hidden_wfw["hidden_segments"]["display_hidden"] == "iages, sa"
    assert hidden_wfw["hidden_segments"]["display_suffix"] == "lvaging"
    assert hidden_wfw["hidden_segments"]["reversed"] is True
    assert [link["source_value_index"]
            for link in hidden_wfw["answer_links"]] == [10, 9, 8, 7, 6, 5, 4]

    clang_ctx = build_clue_context(
        "Loud noise from family dog's bottom", "CLANG", db, annotate=True)
    clang_parse = assemble_token_parses(clang_ctx)[0]
    clang_wfw = format_token_parse_for_wfw(clang_ctx, clang_parse)
    assert clang_wfw["operations"][0]["operation"] == "positional_charade"
    assert [link["source_text"] for link in clang_wfw["answer_links"]] == [
        "family", "family", "family", "family", "dog's"]

    dd_ctx = build_clue_context("Personal hint", "INTIMATE", db, annotate=True)
    dd_parse = assemble_token_parses(dd_ctx)[0]
    dd_wfw = format_token_parse_for_wfw(dd_ctx, dd_parse)
    assert dd_wfw["operations"][0]["operation"] == "double_definition"
    assert dd_wfw["tokens"][0]["roles"][0]["role"] == "dd_0"
    assert dd_wfw["tokens"][1]["roles"][0]["role"] == "dd_1"
    assert all(
        link["source_role"] == "double_definition"
        and "source_text" not in link
        for link in dd_wfw["answer_links"])

    pos_ctx = build_clue_context("bird initially", "B", db, annotate=True)
    pos_ctx = _with_annotation(pos_ctx, SpanAnnotation(
        span=(1, 2),
        text="initially",
        token="POS_I_FIRST",
        values=(True,),
        source="test",
    ))
    pos_parse = assemble_token_parses(pos_ctx)[0]
    pos_wfw = format_token_parse_for_wfw(pos_ctx, pos_parse)
    assert pos_wfw["operations"][0]["operation"] == "positional_charade"
    assert pos_wfw["answer_links"][0]["source_text"] == "bird"
    assert pos_wfw["answer_links"][0]["source_role"] == "piece_0"

    acro_ctx = build_clue_context(
        "initially big apple", "BA", db, annotate=True)
    acro_ctx = _with_annotation(acro_ctx, SpanAnnotation(
        span=(0, 1), text="initially", token="POS_I_FIRST",
        values=(True,), source="test"))
    acro_parse = assemble_token_parses(acro_ctx)[0]
    acro_wfw = format_token_parse_for_wfw(acro_ctx, acro_parse)
    assert [link["source_text"] for link in acro_wfw["answer_links"]] == [
        "big", "apple"]

    hom_ctx = build_clue_context("heard site", "SIGHT", db, annotate=True)
    hom_ctx = _with_annotation(hom_ctx, SpanAnnotation(
        span=(0, 1),
        text="heard",
        token="HOM_I",
        values=(True,),
        source="test",
    ))
    hom_ctx = _with_annotation(hom_ctx, SpanAnnotation(
        span=(1, 2),
        text="site",
        token="HOM_F",
        values=("SIGHT",),
        source="test",
    ))
    hom_parse = assemble_token_parses(hom_ctx)[0]
    hom_wfw = format_token_parse_for_wfw(hom_ctx, hom_parse)
    assert hom_wfw["operations"][0]["operation"] == "homophone"
    assert hom_wfw["answer_links"][0]["source_text"] == "site"

    del_ctx = build_clue_context("beheaded scare", "CARE", db, annotate=True)
    del_ctx = _with_annotation(del_ctx, SpanAnnotation(
        span=(0, 1),
        text="beheaded",
        token="POS_I_TRIM_FIRST",
        values=(True,),
        source="test",
    ))
    del_parse = assemble_token_parses(del_ctx)[0]
    del_wfw = format_token_parse_for_wfw(del_ctx, del_parse)
    assert del_wfw["operations"][0]["operation"] == "deletion"
    assert del_wfw["answer_links"][0]["source_text"] == "scare"

    rev_ctx = build_clue_context("back pan one", "NAPI", db, annotate=True)
    rev_ctx = _with_annotation(rev_ctx, SpanAnnotation(
        span=(0, 1), text="back", token="REV_I",
        values=(True,), source="test"))
    rev_ctx = _with_annotation(rev_ctx, SpanAnnotation(
        span=(1, 2), text="pan", token="SYN_F",
        values=("PAN",), source="test"))
    rev_ctx = _with_annotation(rev_ctx, SpanAnnotation(
        span=(2, 3), text="one", token="ABR_F",
        values=("I",), source="test"))
    rev_parse = assemble_token_parses(rev_ctx)[0]
    rev_wfw = format_token_parse_for_wfw(rev_ctx, rev_parse)
    assert rev_wfw["operations"][0]["operation"] == "reversal_charade"
    assert [link["letter"] for link in rev_wfw["answer_links"]] == list("NAPI")

    ana_ctx = build_clue_context("disturbed act one", "CATI", db, annotate=True)
    ana_ctx = _with_annotation(ana_ctx, SpanAnnotation(
        span=(0, 1), text="disturbed", token="ANA_I",
        values=(True,), source="test"))
    ana_ctx = _with_annotation(ana_ctx, SpanAnnotation(
        span=(2, 3), text="one", token="ABR_F",
        values=("I",), source="test"))
    ana_parse = assemble_token_parses(ana_ctx)[0]
    ana_wfw = format_token_parse_for_wfw(ana_ctx, ana_parse)
    assert ana_wfw["operations"][0]["operation"] == "anagram_charade"
    assert [link["letter"] for link in ana_wfw["answer_links"]] == list("CATI")

    delc_ctx = build_clue_context(
        "beheaded scare one", "CAREI", db, annotate=True)
    delc_ctx = _with_annotation(delc_ctx, SpanAnnotation(
        span=(0, 1), text="beheaded", token="POS_I_TRIM_FIRST",
        values=(True,), source="test"))
    delc_ctx = _with_annotation(delc_ctx, SpanAnnotation(
        span=(2, 3), text="one", token="ABR_F",
        values=("I",), source="test"))
    delc_parse = assemble_token_parses(delc_ctx)[0]
    delc_wfw = format_token_parse_for_wfw(delc_ctx, delc_parse)
    assert delc_wfw["operations"][0]["operation"] == "deletion_charade"
    assert [link["letter"] for link in delc_wfw["answer_links"]] == list("CAREI")

    conc_ctx = build_clue_context(
        "horse taking minutes one", "COMBI", db, annotate=True)
    conc_ctx = _with_annotation(conc_ctx, SpanAnnotation(
        span=(0, 1), text="horse", token="SYN_F",
        values=("COB",), source="test"))
    conc_ctx = _with_annotation(conc_ctx, SpanAnnotation(
        span=(1, 2), text="taking", token="CON_I",
        values=(True,), source="test"))
    conc_ctx = _with_annotation(conc_ctx, SpanAnnotation(
        span=(2, 3), text="minutes", token="ABR_F",
        values=("M",), source="test"))
    conc_ctx = _with_annotation(conc_ctx, SpanAnnotation(
        span=(3, 4), text="one", token="ABR_F",
        values=("I",), source="test"))
    conc_parse = assemble_token_parses(conc_ctx)[0]
    conc_wfw = format_token_parse_for_wfw(conc_ctx, conc_parse)
    assert conc_wfw["operations"][0]["operation"] == "container_charade"
    assert [link["letter"] for link in conc_wfw["answer_links"]] == list("COMBI")

    sub_ctx = build_clue_context(
        "Get up with daughter, not son, and travel.",
        "RIDE", db, annotate=True)
    sub_parse = assemble_token_parses(sub_ctx)[0]
    sub_wfw = format_token_parse_for_wfw(sub_ctx, sub_parse)
    assert sub_wfw["operations"][0]["operation"] == "substitution"
    assert [link["letter"] for link in sub_wfw["answer_links"]] == list("RIDE")
    assert [link["source_text"] for link in sub_wfw["answer_links"]] == [
        "Get up", "Get up", "daughter,", "Get up"]

    print("WFW formatter regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
