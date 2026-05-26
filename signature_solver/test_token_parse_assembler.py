"""Regression tests for tokenised parse assembly."""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from signature_solver.clue_context import (
    ClueContext,
    SpanAnnotation,
    build_clue_context,
)
from signature_solver.db import RefDB
from signature_solver.solver import solve_clue
from signature_solver.token_parse_assembler import assemble_token_parses


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
    )


def run_tests():
    db = RefDB()

    comb = build_clue_context(
        "Groom horse, taking minutes", "COMB", db, annotate=True)
    parses = assemble_token_parses(comb)
    assert parses, "expected a tokenised container parse for COMB"
    parse = parses[0]
    assert parse.operation == "container"
    blocks = {block.block_id: block for block in parse.blocks}
    assert blocks["def_0"].span == (0, 1)
    assert blocks["src_outer"].span == (1, 2)
    assert blocks["src_outer"].value == "COB"
    assert blocks["op_0"].span == (2, 3)
    assert blocks["src_inner"].span == (3, 4)
    assert blocks["src_inner"].value == "M"
    assert parse.operations[0].output == "COMB"

    comb_pos = build_clue_context(
        "horse taking initially mango", "COMB", db, annotate=True)
    comb_pos = _with_annotations(comb_pos, [
        SpanAnnotation((0, 1), "horse", "SYN_F", ("COB",), "test"),
        SpanAnnotation((1, 2), "taking", "CON_I", (True,), "test"),
        SpanAnnotation((2, 3), "initially", "POS_I_FIRST", (True,), "test"),
        SpanAnnotation((3, 4), "mango", "POS_F", ("MANGO",), "test"),
    ])
    parses = assemble_token_parses(comb_pos)
    assert parses and parses[0].operation == "container"
    blocks = {block.block_id: block for block in parses[0].blocks}
    assert blocks["src_inner"].value == "M"
    assert blocks["op_inner_pos"].role == "inner_positional_indicator"

    forage = build_clue_context(
        "Search advanced into blacksmith's workshop",
        "FORAGE", db, annotate=True)
    parses = assemble_token_parses(forage)
    assert parses, "FORAGE should assemble from preserved span evidence"
    parse = parses[0]
    blocks = {block.block_id: block for block in parse.blocks}
    assert blocks["def_0"].span == (0, 1)
    assert blocks["src_inner"].span == (1, 2)
    assert blocks["op_0"].span == (2, 3)
    assert blocks["src_outer"].span == (3, 5)
    assert parse.operations[0].detail == "FORGE contains A = FORAGE"

    sr = solve_clue("Groom horse, taking minutes", "COMB", db)
    assert getattr(sr, "token_parses", None), "solve result should carry token parses"
    assert sr.token_parses[0].as_dict()["blocks"][1]["span"] == [1, 2]
    assert getattr(sr, "wfw_token_parses", None), (
        "solve result should carry WFW token parse data")
    assert sr.wfw_token_parses[0]["answer_links"][0]["source_text"] == "horse,"

    givein = build_clue_context(
        "Sink India before noon, or capitulate?", "GIVEIN", db,
        annotate=True)
    parses = assemble_token_parses(givein)
    assert parses and parses[0].operation == "charade"
    assert any(
        block.text == "before"
        and block.role == "link"
        for block in parses[0].blocks
    )

    code = build_clue_context(
        "Set of expectations about lyric poem", "CODE", db, annotate=True)
    parses = assemble_token_parses(code)
    assert parses and parses[0].operation == "charade"
    blocks = {block.block_id: block for block in parses[0].blocks}
    assert blocks["src_0"].span == (3, 4)
    assert blocks["src_0"].value == "C"
    assert blocks["src_1"].span == (4, 6)
    assert blocks["src_1"].text == "lyric poem"
    assert blocks["src_1"].value == "ODE"

    incomplete_code = _with_annotations(
        code,
        [
            ann for ann in code.annotations
            if not (ann.span == (4, 6) and ann.token == "SYN_F")
        ],
    )
    incomplete_parses = assemble_token_parses(incomplete_code)
    assert incomplete_parses
    assert incomplete_parses[0].confidence == (
        "mechanically_verified_surface_gaps")

    moor = build_clue_context(
        "Uncultivated land with space to the west", "MOOR", db,
        annotate=True)
    parses = assemble_token_parses(moor)
    assert parses and parses[0].operation == "reversal"
    assert parses[0].operations[0].detail == "ROOM reversed = MOOR"

    smart = build_clue_context(
        "Electronic device disturbed mother's nap", "SMARTPHONE", db,
        annotate=True)
    parses = assemble_token_parses(smart)
    assert parses and parses[0].operation == "anagram"

    inferred_def = build_clue_context(
        "Useful for traditional dancers same ploy to vary",
        "MAYPOLES", db, annotate=True)
    inferred_parses = assemble_token_parses(inferred_def)
    assert inferred_parses
    inferred_blocks = {block.block_id: block for block in inferred_parses[0].blocks}
    assert inferred_blocks["def_0"].role == "inferred_definition"
    assert inferred_parses[0].confidence == (
        "mechanically_verified_inferred_definition")

    hidden = build_clue_context(
        "Turned over some carriages, salvaging weapon", "ASSEGAI", db,
        annotate=True)
    parses = assemble_token_parses(hidden)
    assert parses and parses[0].operation == "hidden_reversed"

    hidden_phrase_indicator = build_clue_context(
        "Snooker balls put in stored sets", "REDS", db, annotate=True)
    parses = assemble_token_parses(hidden_phrase_indicator)
    assert parses and parses[0].operation == "hidden"
    hidden_blocks = {block.block_id: block for block in parses[0].blocks}
    assert hidden_blocks["op_0"].text == "put in"

    clang = build_clue_context(
        "Loud noise from family dog's bottom", "CLANG", db, annotate=True)
    parses = assemble_token_parses(clang)
    assert parses and parses[0].operation == "positional_charade"
    clang_blocks = {block.block_id: block for block in parses[0].blocks}
    assert clang_blocks["src_0"].text == "family"
    assert clang_blocks["src_0"].value == "CLAN"
    assert clang_blocks["src_1"].text == "dog's"
    assert clang_blocks["src_1"].value == "G"
    assert any(block.text == "from" and block.role == "link"
               for block in parses[0].blocks)

    dd = solve_clue("Personal hint", "INTIMATE", db)
    assert dd.token_parses and dd.token_parses[0].operation == "double_definition"
    dd_blocks = {block.block_id: block for block in dd.token_parses[0].blocks}
    assert dd_blocks["dd_0"].span == (0, 1)
    assert dd_blocks["dd_0"].text == "Personal"
    assert dd_blocks["dd_1"].span == (1, 2)
    assert dd_blocks["dd_1"].text == "hint"

    ram = solve_clue("Sheep's memory", "RAM", db)
    assert ram.token_parses and ram.token_parses[0].operation == "double_definition"
    ram_blocks = {block.block_id: block for block in ram.token_parses[0].blocks}
    assert ram_blocks["dd_0"].text == "Sheep's"
    assert ram_blocks["dd_1"].text == "memory"
    assert ram_blocks["dd_1"].token == "ABR_F"

    cryptic_qualifier = build_clue_context("Standard at sea?", "ENSIGN", db,
                                           annotate=True)
    parses = assemble_token_parses(cryptic_qualifier)
    assert parses and parses[0].operation == "double_definition"
    assert parses[0].confidence == "mechanically_verified"
    cq_blocks = {block.block_id: block for block in parses[0].blocks}
    assert cq_blocks["dd_0"].text == "Standard"
    assert cq_blocks["dd_1"].text == "at sea?"

    whole_answer_shortcut = build_clue_context(
        "Principal cable", "LEAD", db, annotate=True)
    parses = assemble_token_parses(whole_answer_shortcut)
    assert not any(parse.operation == "charade" for parse in parses), (
        "a single whole-answer synonym is not a WFW charade proof")

    positional = build_clue_context("bird initially", "B", db, annotate=True)
    positional = _with_annotations(positional, [
        SpanAnnotation((0, 1), "bird", "POS_F", ("BIRD",), "test"),
        SpanAnnotation((1, 2), "initially", "POS_I_FIRST", (True,), "test"),
    ])
    parses = assemble_token_parses(positional)
    assert parses and parses[0].operation == "positional_charade"
    pos_blocks = {block.block_id: block for block in parses[0].blocks}
    assert pos_blocks["src_0"].value == "B"
    assert pos_blocks["op_0"].role == "piece_0_positional_indicator"

    alternate_even = build_clue_context(
        "Hancock regularly employed individual in dock",
        "ACCUSED", db, annotate=True)
    parses = assemble_token_parses(alternate_even)
    assert parses and parses[0].operation == "positional_charade"
    alt_blocks = {block.block_id: block for block in parses[0].blocks}
    assert alt_blocks["src_0"].text == "Hancock"
    assert alt_blocks["src_0"].value == "ACC"
    assert alt_blocks["op_0"].text == "regularly"

    acrostic = build_clue_context(
        "initially big apple", "BA", db, annotate=True)
    acrostic = _with_annotations(acrostic, [
        SpanAnnotation((0, 1), "initially", "POS_I_FIRST", (True,), "test"),
        SpanAnnotation((1, 2), "big", "POS_F", ("BIG",), "test"),
        SpanAnnotation((2, 3), "apple", "POS_F", ("APPLE",), "test"),
    ])
    parses = assemble_token_parses(acrostic)
    assert parses and parses[0].operation == "positional_charade"
    acrostic_blocks = {block.block_id: block for block in parses[0].blocks}
    assert acrostic_blocks["src_0"].value == "B"
    assert acrostic_blocks["src_1"].value == "A"

    hom = build_clue_context("heard site", "SIGHT", db, annotate=True)
    hom = _with_annotations(hom, [
        SpanAnnotation((0, 1), "heard", "HOM_I", (True,), "test"),
        SpanAnnotation((1, 2), "site", "HOM_F", ("SIGHT",), "test"),
    ])
    parses = assemble_token_parses(hom)
    assert parses and parses[0].operation == "homophone"
    hom_blocks = {block.block_id: block for block in parses[0].blocks}
    assert hom_blocks["src_0"].text == "site"
    assert hom_blocks["op_0"].role == "homophone_indicator"

    deletion = build_clue_context("beheaded scare", "CARE", db, annotate=True)
    deletion = _with_annotations(deletion, [
        SpanAnnotation((0, 1), "beheaded", "POS_I_TRIM_FIRST", (True,), "test"),
        SpanAnnotation((1, 2), "scare", "POS_F", ("SCARE",), "test"),
    ])
    parses = assemble_token_parses(deletion)
    assert parses and parses[0].operation == "deletion"
    del_blocks = {block.block_id: block for block in parses[0].blocks}
    assert del_blocks["src_0"].value == "CARE"
    assert del_blocks["op_0"].role == "deletion_indicator"

    rev_charade = build_clue_context("back pan one", "NAPI", db, annotate=True)
    rev_charade = _with_annotations(rev_charade, [
        SpanAnnotation((0, 1), "back", "REV_I", (True,), "test"),
        SpanAnnotation((1, 2), "pan", "SYN_F", ("PAN",), "test"),
        SpanAnnotation((2, 3), "one", "ABR_F", ("I",), "test"),
    ])
    parses = assemble_token_parses(rev_charade)
    assert parses and parses[0].operation == "reversal_charade"
    rev_blocks = {block.block_id: block for block in parses[0].blocks}
    assert rev_blocks["src_0"].value == "NAP"
    assert rev_blocks["op_0"].role == "piece_0_reversal_indicator"
    assert rev_blocks["src_1"].value == "I"

    ana_charade = build_clue_context(
        "disturbed act one", "CATI", db, annotate=True)
    ana_charade = _with_annotations(ana_charade, [
        SpanAnnotation((0, 1), "disturbed", "ANA_I", (True,), "test"),
        SpanAnnotation((1, 2), "act", "ANA_F", ("ACT",), "test"),
        SpanAnnotation((2, 3), "one", "ABR_F", ("I",), "test"),
    ])
    parses = assemble_token_parses(ana_charade)
    assert parses and parses[0].operation == "anagram_charade"
    ana_blocks = {block.block_id: block for block in parses[0].blocks}
    assert ana_blocks["src_0"].value == "CAT"
    assert ana_blocks["op_0"].role == "piece_0_anagram_indicator"
    assert ana_blocks["src_1"].value == "I"

    ana_deleted_syn = build_clue_context(
        "Welcomes largely old set at work", "GREETS", db, annotate=True)
    parses = assemble_token_parses(ana_deleted_syn)
    assert parses and parses[0].operation == "anagram_charade"
    ads_blocks = {block.block_id: block for block in parses[0].blocks}
    assert ads_blocks["def_0"].text == "Welcomes"
    assert ads_blocks["src_0"].text == "old"
    assert ads_blocks["src_0"].input_value == "GREY"
    assert ads_blocks["src_0"].value == "GRE"
    assert ads_blocks["op_0"].text == "largely"
    assert ads_blocks["src_1"].text == "set"
    assert ads_blocks["src_1"].value == "ETS"
    assert ads_blocks["op_1"].text == "work"

    split_ana = build_clue_context(
        "Ordinary kid and fan somehow is halfway decent",
        "OFAKIND", db, annotate=True)
    parses = assemble_token_parses(split_ana)
    assert parses and parses[0].operation == "anagram_charade"
    split_blocks = {block.block_id: block for block in parses[0].blocks}
    assert split_blocks["src_0"].text == "Ordinary"
    assert split_blocks["src_0"].value == "O"
    assert split_blocks["src_1"].text == "kid fan"
    assert split_blocks["src_1"].value == "FAKIND"

    link_inside_anagram = build_clue_context(
        "Started to grin, with idea developing",
        "ORIGINATED", db, annotate=True)
    parses = assemble_token_parses(link_inside_anagram)
    assert parses and parses[0].operation == "anagram"
    lia_blocks = {block.block_id: block for block in parses[0].blocks}
    assert lia_blocks["src_0"].text == "to grin, idea"
    assert lia_blocks["src_0"].value == "TOGRINIDEA"

    del_charade = build_clue_context(
        "beheaded scare one", "CAREI", db, annotate=True)
    del_charade = _with_annotations(del_charade, [
        SpanAnnotation((0, 1), "beheaded", "POS_I_TRIM_FIRST", (True,), "test"),
        SpanAnnotation((1, 2), "scare", "POS_F", ("SCARE",), "test"),
        SpanAnnotation((2, 3), "one", "ABR_F", ("I",), "test"),
    ])
    parses = assemble_token_parses(del_charade)
    assert parses and parses[0].operation == "deletion_charade"
    delc_blocks = {block.block_id: block for block in parses[0].blocks}
    assert delc_blocks["src_0"].value == "CARE"
    assert delc_blocks["op_0"].role == "piece_0_deletion_indicator"
    assert delc_blocks["src_1"].value == "I"

    subtractive_deletion = build_clue_context(
        "Broad Irish wit left to go", "WIDE", db, annotate=True)
    parses = assemble_token_parses(subtractive_deletion)
    assert parses and parses[0].operation == "deletion_charade"
    subdel_blocks = {block.block_id: block for block in parses[0].blocks}
    assert subdel_blocks["src_0"].text == "Irish wit"
    assert subdel_blocks["src_0"].input_value == "WILDE"
    assert subdel_blocks["src_0"].value == "WIDE"
    assert subdel_blocks["removed_0"].text == "left"
    assert subdel_blocks["removed_0"].value == "L"
    assert subdel_blocks["op_0"].text == "to go"

    dismissed_leader = build_clue_context(
        "Loathsome, rotten, broody leader is dismissed",
        "OFFENSIVE", db, annotate=True)
    parses = assemble_token_parses(dismissed_leader)
    assert parses and parses[0].operation == "deletion_charade"
    dl_blocks = {block.block_id: block for block in parses[0].blocks}
    assert dl_blocks["src_0"].value == "OFF"
    assert dl_blocks["src_1"].input_value == "PENSIVE"
    assert dl_blocks["src_1"].value == "ENSIVE"
    assert dl_blocks["op_1"].text == "leader is dismissed"

    half_charade = build_clue_context(
        "Meat - try half of it", "HEART", db, annotate=True)
    parses = assemble_token_parses(half_charade)
    assert parses and parses[0].operation == "positional_charade"
    half_blocks = {block.block_id: block for block in parses[0].blocks}
    assert half_blocks["src_0"].text == "try"
    assert half_blocks["src_0"].value == "HEAR"
    assert half_blocks["src_1"].text == "it"
    assert half_blocks["src_1"].value == "T"
    assert half_blocks["op_1"].text == "half"

    supporting_charade = build_clue_context(
        "Religious leader's Mass supporting current postgraduate",
        "IMAM", db, annotate=True)
    parses = assemble_token_parses(supporting_charade)
    assert parses and parses[0].operation == "supporting_charade"
    support_blocks = {block.block_id: block for block in parses[0].blocks}
    assert support_blocks["def_0"].text == "Religious leader's"
    assert support_blocks["src_0"].text == "current"
    assert support_blocks["src_0"].value == "I"
    assert support_blocks["src_1"].text == "postgraduate"
    assert support_blocks["src_1"].value == "MA"
    assert support_blocks["src_2"].text == "Mass"
    assert support_blocks["src_2"].value == "M"
    assert any(block.text == "supporting" for block in parses[0].blocks)

    alternate_phrase = build_clue_context(
        "Employ funster on and off", "USE", db, annotate=True)
    parses = assemble_token_parses(alternate_phrase)
    assert parses and parses[0].operation == "positional_charade"
    altp_blocks = {block.block_id: block for block in parses[0].blocks}
    assert altp_blocks["def_0"].text == "Employ"
    assert altp_blocks["src_0"].text == "funster"
    assert altp_blocks["src_0"].value == "USE"
    assert altp_blocks["op_0"].text == "on and off"

    gutted_outer = build_clue_context(
        "American elector gutted seeing exploiter",
        "USER", db, annotate=True)
    parses = assemble_token_parses(gutted_outer)
    assert parses and parses[0].operation == "positional_charade"
    go_blocks = {block.block_id: block for block in parses[0].blocks}
    assert go_blocks["def_0"].text == "exploiter"
    assert go_blocks["src_0"].text == "American"
    assert go_blocks["src_0"].value == "US"
    assert go_blocks["src_1"].text == "elector"
    assert go_blocks["src_1"].value == "ER"
    assert go_blocks["op_1"].text == "gutted"
    assert go_blocks["op_1"].token == "POS_I_OUTER"
    assert any(
        block.text == "seeing" and block.role == "link"
        for block in parses[0].blocks
    )
    assert not any(
        block.text == "seeing" and block.kind == "OP_BLOCK"
        for block in parses[0].blocks
    )

    grouped_anagram = build_clue_context(
        "Revolutionary at ordinary port in the Black Sea",
        "YALTA", db, annotate=True)
    parses = assemble_token_parses(grouped_anagram)
    assert parses and parses[0].operation == "anagram_charade"
    ga_blocks = {block.block_id: block for block in parses[0].blocks}
    assert ga_blocks["def_0"].text == "port in the Black Sea"
    assert ga_blocks["src_0"].text == "at ordinary"
    assert ga_blocks["src_0"].input_value == "ATLAY"
    assert ga_blocks["src_0"].value == "YALTA"

    half_synonym = build_clue_context(
        "Damage area for selling goods not half", "MAR", db, annotate=True)
    parses = assemble_token_parses(half_synonym)
    assert parses and parses[0].operation == "positional_charade"
    hs_blocks = {block.block_id: block for block in parses[0].blocks}
    assert hs_blocks["def_0"].text == "Damage"
    assert hs_blocks["src_0"].text == "area for selling goods"
    assert hs_blocks["src_0"].input_value == "MARKET"
    assert hs_blocks["src_0"].value == "MAR"

    trim_first_charade = build_clue_context(
        "Exclusive English literature topped list",
        "ELITIST", db, annotate=True)
    parses = assemble_token_parses(trim_first_charade)
    assert parses and parses[0].operation == "deletion_charade"
    trim_blocks = {block.block_id: block for block in parses[0].blocks}
    assert trim_blocks["src_0"].value == "E"
    assert trim_blocks["src_1"].value == "LIT"
    assert trim_blocks["src_2"].input_value == "LIST"
    assert trim_blocks["src_2"].value == "IST"
    assert trim_blocks["op_2"].text == "topped"

    con_charade = build_clue_context(
        "horse taking minutes one", "COMBI", db, annotate=True)
    con_charade = _with_annotations(con_charade, [
        SpanAnnotation((0, 1), "horse", "SYN_F", ("COB",), "test"),
        SpanAnnotation((1, 2), "taking", "CON_I", (True,), "test"),
        SpanAnnotation((2, 3), "minutes", "ABR_F", ("M",), "test"),
        SpanAnnotation((3, 4), "one", "ABR_F", ("I",), "test"),
    ])
    parses = assemble_token_parses(con_charade)
    assert parses and parses[0].operation == "container_charade"
    conc_blocks = {block.block_id: block for block in parses[0].blocks}
    assert conc_blocks["src_0"].value == "COMB"
    inner_block = conc_blocks.get("src_0_inner") or conc_blocks["src_0_inner_0"]
    assert inner_block.value == "M"
    assert conc_blocks["op_0"].role == "piece_0_container_indicator"
    assert conc_blocks["src_1"].value == "I"

    nested_container = build_clue_context(
        "Boxer outlaw intoxicated, drinking in the morning with European",
        "BANTAMWEIGHT", db, annotate=True)
    parses = assemble_token_parses(nested_container)
    assert parses and parses[0].operation == "container_charade"
    nested_blocks = {block.block_id: block for block in parses[0].blocks}
    assert nested_blocks["src_0"].text == "outlaw"
    assert nested_blocks["src_0"].value == "BAN"
    assert nested_blocks["src_1"].text == "intoxicated,"
    assert nested_blocks["src_1"].value == "TAMWEIGHT"
    assert nested_blocks["op_1"].text == "drinking"
    assert nested_blocks["src_1_inner_0"].text == "in the morning"
    assert nested_blocks["src_1_inner_0"].value == "AM"
    assert nested_blocks["src_1_inner_1"].text == "with"
    assert nested_blocks["src_1_inner_1"].value == "W"
    assert nested_blocks["src_1_inner_2"].text == "European"
    assert nested_blocks["src_1_inner_2"].value == "E"

    shell_container = build_clue_context(
        "Small company importing revolutionary wound plaster",
        "STUCCO", db, annotate=True)
    parses = assemble_token_parses(shell_container)
    assert parses and parses[0].operation == "container_charade"
    shell_blocks = {block.block_id: block for block in parses[0].blocks}
    assert shell_blocks["def_0"].text == "plaster"
    assert shell_blocks["src_0_shell_0"].text == "Small"
    assert shell_blocks["src_0_shell_0"].value == "S"
    assert shell_blocks["src_0_shell_1"].text == "company"
    assert shell_blocks["src_0_shell_1"].value == "CO"
    assert shell_blocks["op_0"].text == "importing"
    assert shell_blocks["src_0_inner"].text == "wound"
    assert shell_blocks["src_0_inner"].value == "TUC"
    assert shell_blocks["op_0_inner"].text == "revolutionary"

    game_shows = build_clue_context(
        "Programmes in Georgia question occupying military dining hall",
        "game shows", db, annotate=True)
    parses = assemble_token_parses(game_shows)
    assert parses and parses[0].operation == "container_charade"
    gs_blocks = {block.block_id: block for block in parses[0].blocks}
    assert gs_blocks["def_0"].text == "Programmes"
    assert gs_blocks["src_0"].text == "Georgia"
    assert gs_blocks["src_0"].value == "GA"
    if "src_1_shell_0" in gs_blocks:
        assert gs_blocks["src_1_shell_0"].text == "military dining hall"
        assert gs_blocks["src_1_shell_0"].value == "MESS"
        assert gs_blocks["src_1_inner"].text == "question"
        assert gs_blocks["src_1_inner"].value == "HOW"
    else:
        assert gs_blocks["src_1"].text == "military dining hall"
        assert gs_blocks["src_1"].input_value == "MESS"
        assert gs_blocks["src_1"].value == "MESHOWS"
        assert gs_blocks["src_1_inner_0"].text == "question"
        assert gs_blocks["src_1_inner_0"].value == "HOW"
    assert gs_blocks["op_1"].text == "occupying"
    assert any(block.text == "in" and block.role == "link"
               for block in parses[0].blocks)

    pang = build_clue_context(
        "Sudden twinge leads to Annie needing to have PG Tips",
        "PANG", db, annotate=True)
    parses = assemble_token_parses(pang)
    assert parses and parses[0].operation == "container_charade"
    pang_blocks = {block.block_id: block for block in parses[0].blocks}
    assert pang_blocks["def_0"].text == "Sudden twinge"
    assert pang_blocks["src_0"].text == "PG"
    assert pang_blocks["src_0"].value == "PANG"
    assert pang_blocks["src_0"].input_value == "PG"
    if "src_0_inner" in pang_blocks:
        assert pang_blocks["src_0_inner"].text == "Annie needing"
        assert pang_blocks["src_0_inner"].value == "AN"
    else:
        assert pang_blocks["src_0_inner_0"].text == "Annie"
        assert pang_blocks["src_0_inner_0"].value == "A"
        assert pang_blocks["src_0_inner_1"].text == "needing"
        assert pang_blocks["src_0_inner_1"].value == "N"

    estimates = build_clue_context(
        "Guesses friend enters sites getting renovated",
        "ESTIMATES", db, annotate=True)
    parses = assemble_token_parses(estimates)
    assert parses and parses[0].operation == "container_charade"
    est_blocks = {block.block_id: block for block in parses[0].blocks}
    assert est_blocks["def_0"].text == "Guesses"
    if "src_0_shell_0" in est_blocks:
        assert est_blocks["src_0_shell_0"].text == "sites"
        assert est_blocks["src_0_shell_0"].value == "ESTIS"
        assert est_blocks["op_0_shell_0"].text == "renovated"
        assert est_blocks["src_0_inner"].text == "friend"
        assert est_blocks["src_0_inner"].value == "MATE"
    else:
        assert est_blocks["src_0"].text == "sites"
        assert est_blocks["src_0"].input_value == "ESTIS"
        assert est_blocks["src_0"].value == "ESTIMATES"
        assert est_blocks["op_0_outer"].text == "renovated"
        assert est_blocks["src_0_inner"].text == "friend"
        assert est_blocks["src_0_inner"].value == "MATE"
    assert est_blocks["op_0"].text == "enters"
    assert any(block.text == "getting" and block.role == "link"
               for block in parses[0].blocks)

    pedestrian = build_clue_context(
        "Dull, annoying person ringing editor then managed to grab the setter",
        "PEDESTRIAN", db, annotate=True)
    parses = assemble_token_parses(pedestrian)
    assert parses and parses[0].operation == "container_charade"
    ped_blocks = {block.block_id: block for block in parses[0].blocks}
    assert ped_blocks["src_0"].text == "annoying person"
    assert ped_blocks["src_0"].input_value == "PEST"
    assert ped_blocks["src_0"].value == "PEDEST"
    ped_inner_0 = ped_blocks.get("src_0_inner") or ped_blocks["src_0_inner_0"]
    assert ped_inner_0.text == "editor"
    assert ped_inner_0.value == "ED"
    assert ped_blocks["op_0"].text == "ringing"
    assert ped_blocks["src_1"].text == "managed"
    assert ped_blocks["src_1"].input_value == "RAN"
    assert ped_blocks["src_1"].value == "RIAN"
    ped_inner_1 = ped_blocks.get("src_1_inner") or ped_blocks["src_1_inner_0"]
    assert ped_inner_1.text in ("setter", "the setter")
    assert ped_inner_1.value == "I"
    assert ped_blocks["op_1"].text == "grab"

    residue = build_clue_context(
        "What remains of troops you heard breaching flank",
        "RESIDUE", db, annotate=True)
    parses = assemble_token_parses(residue)
    assert parses and parses[0].operation == "container_charade"
    res_blocks = {block.block_id: block for block in parses[0].blocks}
    assert res_blocks["def_0"].text == "What remains"
    assert res_blocks["src_0"].text == "troops"
    assert res_blocks["src_0"].value == "RE"
    if "src_1_shell_0" in res_blocks:
        assert res_blocks["src_1_shell_0"].text == "flank"
        assert res_blocks["src_1_shell_0"].value == "SIDE"
        assert res_blocks["src_1_inner"].text == "you"
        assert res_blocks["src_1_inner"].value == "U"
    else:
        assert res_blocks["src_1"].text == "flank"
        assert res_blocks["src_1"].input_value == "SIDE"
        assert res_blocks["src_1"].value == "SIDUE"
        res_inner = res_blocks.get("src_1_inner") or res_blocks["src_1_inner_0"]
        assert res_inner.text == "you"
        assert res_inner.value == "U"
    assert res_blocks["op_1"].text == "breaching"

    kneading = build_clue_context(
        "Reportedly requiring massaging",
        "KNEADING", db, annotate=True)
    parses = assemble_token_parses(kneading)
    assert parses and parses[0].operation == "homophone"
    knead_blocks = {block.block_id: block for block in parses[0].blocks}
    assert knead_blocks["def_0"].text == "massaging"
    assert knead_blocks["src_0"].text == "requiring"
    assert knead_blocks["src_0"].value == "NEEDING"
    assert knead_blocks["op_0"].text == "Reportedly"

    substitution = build_clue_context(
        "Get up with daughter, not son, and travel.",
        "RIDE", db, annotate=True)
    parses = assemble_token_parses(substitution)
    assert parses and parses[0].operation == "substitution"
    sub_blocks = {block.block_id: block for block in parses[0].blocks}
    assert sub_blocks["def_0"].text == "travel."
    assert sub_blocks["src_base"].text == "Get up"
    assert sub_blocks["src_base"].value == "RISE"
    assert sub_blocks["src_insert"].text == "daughter,"
    assert sub_blocks["src_insert"].value == "D"
    assert sub_blocks["op_0"].text == "not"
    assert sub_blocks["src_remove"].text == "son,"
    assert sub_blocks["src_remove"].value == "S"

    print("Token parse assembler regression passed")
    return True


if __name__ == "__main__":
    try:
        ok = run_tests()
    except AssertionError as exc:
        print("FAIL:", exc)
        sys.exit(1)
    sys.exit(0 if ok else 1)
