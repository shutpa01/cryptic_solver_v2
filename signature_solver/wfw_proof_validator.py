"""Structural validation for unified WFW proof records.

This is deliberately independent of any one puzzle.  A proof may only publish
as WFW-proven when the record shows how answer letters were made and when every
displayed mechanism is actually used by the materialised assembly.
"""
from __future__ import annotations


def validate_unified_wfw_proof(proof):
    """Return objection codes for a unified WFW proof dict."""
    if not proof or proof.get("schema") != "wfw_unified_proof:v1":
        return ()

    objections = []
    token_parse = proof.get("token_parse") or {}
    assembly = proof.get("assembly") or {}
    blocks = token_parse.get("blocks") or []
    operation = assembly.get("operation") or token_parse.get("operation")

    if not token_parse:
        objections.append("v4_missing_token_parse")
    if not assembly or assembly.get("status") != "materialised":
        objections.append("v4_missing_materialised_assembly")

    if any(block.get("token") == "SURFACE_GAP" for block in blocks):
        objections.append("v4_surface_gap_in_proven_parse")

    if operation != "double_definition":
        if not any(block.get("kind") == "DEF_BLOCK" for block in blocks):
            objections.append("v4_missing_definition_block")
        objections.extend(_validate_source_definition_separation(
            blocks, operation))
        objections.extend(_validate_answer_placements(proof, assembly))
        objections.extend(_validate_source_contribution(blocks, assembly))

    objections.extend(_validate_operation_blocks(proof, blocks, assembly))

    return tuple(dict.fromkeys(objections))


def _validate_source_definition_separation(blocks, operation):
    if operation in {
            "hidden",
            "hidden_reversed",
            "reversal",
            "positional_charade",
            "acrostic",
    }:
        return ()
    definition_spans = [
        tuple(block.get("span") or ())
        for block in blocks
        if block.get("kind") == "DEF_BLOCK"
    ]
    objections = []
    for source in blocks:
        if source.get("kind") != "SOURCE_BLOCK":
            continue
        source_span = tuple(source.get("span") or ())
        if not source_span:
            continue
        if any(_spans_overlap(source_span, definition_span)
               for definition_span in definition_spans):
            objections.append(
                "v4_source_definition_overlap:%s" % (
                    source.get("text") or source.get("block_id") or "source"
                )
            )
    return tuple(objections)


def _spans_overlap(left, right):
    if len(left) != 2 or len(right) != 2:
        return False
    return max(left[0], right[0]) < min(left[1], right[1])


def _validate_answer_placements(proof, assembly):
    answer = _clean_letters(proof.get("answer") or "")
    placements = assembly.get("placements") or []
    if not answer:
        return ("v4_missing_answer",)
    if not placements:
        return ("v4_missing_answer_placements",)

    objections = []
    by_position = {}
    for placement in placements:
        position = placement.get("answer_position")
        if position in by_position:
            objections.append("v4_duplicate_answer_position")
        by_position[position] = placement
        if not placement.get("source_block_id"):
            objections.append("v4_answer_letter_missing_source_block")
        if not placement.get("source_char_atom_id"):
            objections.append("v4_answer_letter_missing_source_char")

    expected_positions = set(range(1, len(answer) + 1))
    actual_positions = set(by_position)
    if actual_positions != expected_positions:
        objections.append("v4_answer_positions_not_fully_covered")
        return tuple(objections)

    made = "".join(
        (by_position[pos].get("answer_letter") or "").upper()
        for pos in sorted(expected_positions)
    )
    if made != answer:
        objections.append("v4_answer_letters_do_not_match")
    return tuple(objections)


def _validate_source_contribution(parse_blocks, assembly):
    contributing = _contributing_working_block_ids(assembly)
    objections = []
    for block in parse_blocks:
        if block.get("kind") != "SOURCE_BLOCK":
            continue
        role = block.get("role") or ""
        if role.startswith("dd_"):
            continue
        work_id = "wfw_%s" % _clean_id(block.get("block_id") or "")
        if work_id not in contributing:
            objections.append(
                "v4_source_block_not_used:%s" % (block.get("text") or work_id)
            )
    return tuple(objections)


def _contributing_working_block_ids(assembly):
    placements = assembly.get("placements") or []
    contributing = {
        placement.get("source_block_id")
        for placement in placements
        if placement.get("source_block_id")
    }
    working_by_id = {
        block.get("block_id"): block
        for block in assembly.get("working_blocks") or []
        if block.get("block_id")
    }
    transformations = assembly.get("transformations") or []

    changed = True
    while changed:
        changed = False
        for block_id in list(contributing):
            block = working_by_id.get(block_id) or {}
            for parent_id in block.get("parent_block_ids") or []:
                if parent_id not in contributing:
                    contributing.add(parent_id)
                    changed = True
        for transform in transformations:
            if transform.get("output_block_id") not in contributing:
                continue
            for input_id in transform.get("input_block_ids") or []:
                if input_id not in contributing:
                    contributing.add(input_id)
                    changed = True
    return contributing


def _validate_operation_blocks(proof, parse_blocks, assembly):
    op_blocks = [
        block for block in parse_blocks
        if block.get("kind") in ("OP_BLOCK", "RELATION_BLOCK")
    ]
    if not op_blocks:
        return ()

    used_controller_atoms = {
        atom_id
        for transform in assembly.get("transformations") or []
        for atom_id in transform.get("controller_atom_ids") or []
    }
    objections = []
    for block in op_blocks:
        block_id = block.get("block_id") or ""
        role = block.get("role") or ""
        token = block.get("token") or ""
        if block_id.startswith("cov_"):
            objections.append(
                "v4_operation_block_not_used:%s" % (block.get("text") or block_id)
            )
            continue
        if not _operation_block_is_tied_to_piece(block_id, role, token):
            objections.append(
                "v4_operation_block_has_no_piece:%s" % (
                    block.get("text") or block_id
                )
            )
        atom_ids = _atom_ids_for_block(proof, block)
        if atom_ids and used_controller_atoms and not (
                atom_ids & used_controller_atoms):
            objections.append(
                "v4_operation_controller_not_used:%s" % (
                    block.get("text") or block_id
                )
            )
        expected_output = _expected_operation_output_block(parse_blocks, role)
        if expected_output and atom_ids:
            controller_transforms = [
                transform for transform in assembly.get("transformations") or []
                if atom_ids & set(transform.get("controller_atom_ids") or [])
            ]
            if controller_transforms and not any(
                    transform.get("output_block_id") == expected_output
                    for transform in controller_transforms):
                objections.append(
                    "v4_operation_output_mismatch:%s" % (
                        block.get("text") or block_id
                    )
                )
    return tuple(objections)


def _expected_operation_output_block(parse_blocks, op_role):
    piece = _piece_prefix(op_role)
    if not piece:
        return None
    for block in parse_blocks:
        if block.get("kind") != "SOURCE_BLOCK":
            continue
        if block.get("role") == piece:
            return "wfw_%s" % _clean_id(block.get("block_id") or "")
    return None


def _piece_prefix(role):
    parts = (role or "").split("_")
    for idx, part in enumerate(parts[:-1]):
        if part == "piece" and parts[idx + 1].isdigit():
            return "piece_%s" % parts[idx + 1]
    return None


def _operation_block_is_tied_to_piece(block_id, role, token):
    if "piece_" in role:
        return True
    if block_id.startswith("op_"):
        return True
    if token in {"ANA_I", "REV_I", "CON_I", "HOM_I", "SUB_I"}:
        return True
    if token.startswith("POS_I_"):
        return True
    return False


def _atom_ids_for_block(proof, block):
    span = block.get("span")
    text_key = _token_key(block.get("text") or "")
    atom_ids = _atom_ids_for_span(proof, span)
    if atom_ids and _span_text_key(proof, span) == text_key:
        return atom_ids
    if not text_key:
        return atom_ids
    tokens = (proof.get("atom_context") or {}).get("clue_tokens") or []
    for token in tokens:
        if _token_key(token.get("text") or "") == text_key:
            return set(token.get("atom_ids") or [])
    phrase_atoms = _atom_ids_for_phrase_text(tokens, text_key)
    if phrase_atoms:
        return phrase_atoms
    return atom_ids


def _atom_ids_for_phrase_text(tokens, text_key):
    if not text_key:
        return set()
    for start in range(len(tokens)):
        atom_ids = []
        parts = []
        for token in tokens[start:]:
            key = _token_key(token.get("text") or "")
            if not key:
                continue
            parts.append(key)
            atom_ids.extend(token.get("atom_ids") or [])
            joined = "".join(parts)
            if joined == text_key:
                return set(atom_ids)
            if not text_key.startswith(joined):
                break
    return set()


def _atom_ids_for_span(proof, span):
    if not span or len(span) != 2:
        return set()
    tokens = (proof.get("atom_context") or {}).get("clue_tokens") or []
    atom_ids = set()
    for token in tokens:
        index = token.get("index")
        if index is None or not (span[0] <= index < span[1]):
            continue
        atom_ids.update(token.get("atom_ids") or [])
    return atom_ids


def _span_text_key(proof, span):
    if not span or len(span) != 2:
        return ""
    tokens = (proof.get("atom_context") or {}).get("clue_tokens") or []
    return _token_key(" ".join(
        token.get("text") or ""
        for token in tokens
        if token.get("index") is not None
        and span[0] <= token.get("index") < span[1]
    ))


def _clean_id(value):
    return "".join(
        char.lower() if char.isalnum() else "_"
        for char in (value or "")
    ).strip("_")


def _clean_letters(value):
    return "".join(char.upper() for char in (value or "") if char.isalpha())


def _token_key(value):
    return "".join(char.upper() for char in (value or "") if char.isalnum())
