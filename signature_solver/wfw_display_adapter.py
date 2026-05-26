"""Adapt WFW proof records to the existing clue-page WFW display contract."""
from __future__ import annotations

import re

from .wfw_atoms import build_wfw_atom_context
from .wfw_proof_validator import validate_unified_wfw_proof


def display_from_wfw_proof_attempt(attempt_row):
    """Return the designed clue-page WFW shape for any WFW attempt.

    The template already has a visual language for WFW.  This adapter keeps
    that contract and changes only where the data comes from.
    """
    if not attempt_row:
        return None
    proof = attempt_row.get("proof") or {}
    proof_attempt = proof.get("proof_attempt") or {}
    if not proof:
        return None
    if proof.get("schema") == "wfw_manual_correction:v1":
        return _display_from_manual_correction(attempt_row, proof)
    if proof.get("schema") == "wfw_unified_proof:v1":
        return _display_from_unified_proof(attempt_row, proof)
    if proof.get("schema") == "stage_three_proof:v1":
        display = display_from_stage_three_proof(proof)
        if display is not None:
            display["status"] = (
                attempt_row.get("status")
                or proof.get("status")
                or display.get("status")
            )
        return display

    proposals = proof_attempt.get("proposals") or []
    proposal_by_index = {
        _piece_index(proposal.get("proposal_id")): proposal
        for proposal in proposals
        if _piece_index(proposal.get("proposal_id")) is not None
    }
    atom_context = proof.get("atom_context") or {}
    clue_tokens = atom_context.get("clue_tokens") or []
    clue_atoms = {
        atom.get("atom_id"): atom
        for atom in atom_context.get("clue_atoms") or []
    }

    blocks = _source_blocks(proposals, clue_tokens)
    blocks.extend(_operation_blocks(
        proof_attempt.get("transformations") or [], clue_atoms, clue_tokens))
    blocks.extend(_coverage_blocks(
        proof.get("coverage") or {}, {block["text"] for block in blocks}))
    blocks = sorted(blocks, key=_block_sort_key)

    answer = proof.get("answer") or ""
    answer_links = _answer_links(
        proof_attempt.get("placements") or [], proposal_by_index)
    if not answer_links:
        answer_links = _plain_answer_links(answer)

    display = {
        "status": attempt_row.get("status") or proof.get("status"),
        "objections": proof.get("objections") or [],
        "clue_text": proof.get("clue_text") or "",
        "answer": answer,
        "tokens": _display_tokens(clue_tokens),
        "blocks": blocks,
        "operations": [_operation_summary(proof_attempt)],
        "answer_links": answer_links,
        "proof_source": attempt_row.get("proof_source"),
        "proof_row_id": attempt_row.get("id"),
    }
    hidden_segments = _hidden_segments(display, proof_attempt)
    if hidden_segments:
        display["hidden_segments"] = hidden_segments
    return display


def _normalise_anagram_display_roles(blocks, answer_links):
    """For anagram proofs, normalise all SOURCE_BLOCK fodder roles to
    'anagram_fodder' so they render as a unified group rather than
    as separate charade-coloured pieces.

    Called only from display_from_stage_three_proof after blocks and
    answer_links are fully built.  Mutates both lists in place.
    Display-only: does not affect proof status, solver confidence,
    or stored evidence.
    """
    # Only applies when there is at least one anagram indicator block.
    # _stage_three_operation_name returns "anagram" for both pure anagram
    # and anagram_charade proofs; checking for the indicator block directly
    # is the most reliable gate.
    has_anagram_indicator = any(
        block.get("role") == "anagram_indicator"
        for block in blocks
    )
    if not has_anagram_indicator:
        return

    # Normalise SOURCE_BLOCK fodder roles and collect their spans.
    # Only SOURCE_BLOCKs with a piece_N role and a recognised fodder
    # token are changed.  "source_review" blocks (failed evidence) are
    # not touched.
    fodder_tokens = {"ANA_F"}
    normalised_spans = set()
    for block in blocks:
        if block.get("kind") != "SOURCE_BLOCK":
            continue
        role = block.get("role") or ""
        if not role.startswith("piece_"):
            continue
        if block.get("token") not in fodder_tokens:
            continue
        block["role"] = "anagram_fodder"
        span = block.get("span")
        if span and len(span) == 2:
            normalised_spans.add(tuple(span))

    if not normalised_spans:
        return

    # Update answer_links whose source_span belongs to a normalised block.
    # Matching is by span, not by role name, because atomic_links use
    # piece_INDEX (position in assembly.parts) while display blocks use
    # piece_IDX (position in the full proof blocks list, which includes
    # DEF_BLOCKs and OP_BLOCKs).  These indices differ and role-name
    # matching would therefore be unreliable.
    for link in answer_links:
        span = link.get("source_span")
        if span and len(span) == 2 and tuple(span) in normalised_spans:
            link["source_role"] = "anagram_fodder"


def _dedupe_stage_three_display_blocks(blocks):
    """Remove duplicate display blocks that share the same token span.

    stage_three_proof._blocks can yield both an OP_BLOCK (from
    operation_candidates) and a REVIEW_BLOCK (from unresolved_words)
    for the same span.  When multiple blocks share a span, only the
    highest-priority one is kept.

    Blocks with no span are always kept.  Display-only: does not affect
    proof status, solver confidence, or stored evidence.
    """
    def _priority(block):
        kind = block.get("kind") or ""
        role = block.get("role") or ""
        if kind == "DEF_BLOCK":
            return 100
        if kind == "SOURCE_BLOCK" and role == "source_review":
            return 90
        if kind == "SOURCE_BLOCK":
            return 80
        if kind == "OP_BLOCK":
            return 70
        if role.endswith("_indicator"):
            return 70
        if role == "review_indicator_candidate":
            return 60
        if role == "review_separator_candidate":
            return 50
        if kind == "LINK_BLOCK":
            return 30
        if role == "unaccounted":
            return 10
        return 20

    seen = {}
    no_span = []
    for block in blocks:
        span = block.get("span")
        if not span or len(span) != 2:
            no_span.append(block)
            continue
        key = tuple(span)
        if key not in seen or _priority(block) > _priority(seen[key]):
            seen[key] = block

    blocks[:] = list(seen.values()) + no_span


def display_from_stage_three_proof(proof):
    """Return the existing clue-page WFW display shape from Stage Three.

    Stage Three remains the verifier.  This function is only the final adapter:
    it turns PASS/REVIEW, blocks, placements, and gaps into the display contract
    already used by ``partials/atomic_parse.html``.
    """
    if hasattr(proof, "as_dict"):
        proof = proof.as_dict()
    proof = dict(proof or {})
    if proof.get("schema") != "stage_three_proof:v1":
        return None

    clue_text = proof.get("clue_text") or ""
    answer = proof.get("answer") or ""
    context = build_wfw_atom_context(clue_text, answer)
    word_purpose_by_index = {
        wp["index"]: wp
        for wp in (proof.get("word_purposes") or [])
        if wp.get("index") is not None
    }
    status = "wfw_proven" if proof.get("status") == "PASS" else "wfw_review"
    failed_checks = [
        check for check in proof.get("checks") or []
        if check.get("status") != "PASS"
    ]
    review_messages = [_stage_three_review_message(check)
                       for check in failed_checks]
    review_messages = [message for message in review_messages if message]

    blocks = []
    for idx, block in enumerate(proof.get("blocks") or []):
        blocks.append(
            _stage_three_display_block(
                block, idx, answer, word_purpose_by_index))
    blocks.extend(
        _stage_three_missing_word_blocks(
            context, blocks, word_purpose_by_index))

    answer_links = [
        {
            "answer_index": link.get("answer_index"),
            "letter": link.get("letter") or "",
            "source_block": link.get("source_block")
                            or link.get("source_role"),
            "source_span": link.get("source_span"),
            "source_text": link.get("source_text") or "",
            "source_role": link.get("source_role"),
            "source_value": link.get("source_value") or "",
            "source_input_value": link.get("source_input_value") or "",
            "source_value_index": link.get("source_value_index"),
        }
        for link in proof.get("atomic_links") or []
    ]
    answer_links = [
        link for link in answer_links
        if link.get("answer_index") is not None
    ]
    if answer_links:
        answer_links = sorted(answer_links, key=lambda link: link["answer_index"])
    else:
        answer_links = _plain_answer_links(answer)

    _normalise_anagram_display_roles(blocks, answer_links)

    _dedupe_stage_three_display_blocks(blocks)

    operation_detail = _stage_three_operation_detail(proof)
    return {
        "status": status,
        "objections": [check.get("name") for check in failed_checks],
        "review_messages": review_messages,
        "clue_text": clue_text,
        "answer": answer,
        "tokens": [
            {
                "index": token.index,
                "text": token.text,
                "normalized": token.text.upper(),
                "roles": [],
            }
            for token in context.clue_tokens
        ],
        "blocks": sorted(blocks, key=_block_sort_key),
        "operations": [{
            "operation": _stage_three_operation_name(proof),
            "detail": operation_detail,
        }],
        "answer_links": answer_links,
        "proof_source": "stage_three",
        "proof_row_id": None,
        "missing_enrichments": _stage_three_missing_enrichments(proof),
    }


def _review_block_hint(wp):
    """Return a short hint string for a REVIEW block, or empty string."""
    if not wp:
        return ""
    purpose = wp.get("purpose") or ""
    status = wp.get("status") or ""
    if purpose == "operation_indicator_candidate":
        return "Indicator?"
    if purpose == "operation_indicator_modifier_candidate":
        return "Indicator modifier?"
    if purpose == "structural_separator_candidate":
        return "Separator?"
    if purpose == "definition_separator_candidate":
        return "Def separator?"
    if status == "unresolved":
        return "No role found"
    return ""


def _stage_three_display_block(block, idx, answer,
                               word_purpose_by_index=None):
    kind = block.get("kind") or "REVIEW_BLOCK"
    role = block.get("role")
    wp = None
    if kind == "DEF_BLOCK":
        role = role or "definition"
    elif kind == "SOURCE_BLOCK":
        if block.get("evidence_status") == "failed":
            role = "source_review"
        else:
            role = role or "piece_%d" % idx
    elif kind == "OP_BLOCK":
        role = _stage_three_indicator_role(block)
    elif kind == "REVIEW_BLOCK":
        span = block.get("span")
        wp = None
        if word_purpose_by_index and span and len(span) == 2:
            for word_idx in range(span[0], span[1]):
                wp = word_purpose_by_index.get(word_idx)
                if wp:
                    break
        purpose = (wp or {}).get("purpose") or ""
        if purpose in ("operation_indicator_candidate",
                       "operation_indicator_modifier_candidate"):
            role = "review_indicator_candidate"
        elif purpose in ("structural_separator_candidate",
                         "definition_separator_candidate"):
            role = "review_separator_candidate"
        else:
            role = "unaccounted"
    return {
        "block_id": block.get("block_id") or "stage_three_%02d" % idx,
        "kind": kind,
        "role": role,
        "text": block.get("text") or "",
        "value": (
            answer if kind == "DEF_BLOCK"
            else "%s - unverified" % block.get("value")
            if role == "source_review" and block.get("value")
            else block.get("value") or _review_block_hint(wp)
        ),
        "token": block.get("token"),
        "evidence_status": block.get("evidence_status"),
        "evidence_reason": block.get("evidence_reason"),
        "input_value": block.get("input_value") or "",
        "span": block.get("span"),
    }


def _stage_three_indicator_role(block):
    token = block.get("token")
    if token == "ANA_I":
        return "anagram_indicator"
    if token == "REV_I":
        return "reversal_indicator"
    if token == "CON_I":
        return "container_indicator"
    if token == "DEL_I":
        return "deletion_indicator"
    if token == "HID_I":
        return "hidden_indicator"
    if token == "HOM_I":
        return "homophone_indicator"
    if isinstance(token, str) and token.startswith("POS_I"):
        return "positional_indicator"
    role = block.get("role") or "indicator"
    if not role.endswith("_indicator"):
        role = "%s_indicator" % role
    return role


def _stage_three_missing_word_blocks(context, blocks,
                                     word_purpose_by_index=None):
    covered = set()
    for block in blocks:
        span = block.get("span")
        if not span or len(span) != 2:
            continue
        covered.update(range(span[0], span[1]))

    # Stage Three spans number only word tokens. The display atom context also
    # includes standalone punctuation tokens, so build the matching word index.
    word_seq = {}
    seq = 0
    for token in context.clue_tokens:
        if token.kind == "word":
            word_seq[token.index] = seq
            seq += 1

    missing = []
    for token in context.clue_tokens:
        if token.kind != "word":
            continue
        ws = word_seq.get(token.index, token.index)
        if ws in covered:
            continue
        wp = (word_purpose_by_index or {}).get(ws)
        purpose = (wp or {}).get("purpose") or ""
        if purpose in ("operation_indicator_candidate",
                       "operation_indicator_modifier_candidate"):
            missing_role = "review_indicator_candidate"
        elif purpose in ("structural_separator_candidate",
                         "definition_separator_candidate"):
            missing_role = "review_separator_candidate"
        else:
            missing_role = "unaccounted"
        missing.append({
            "block_id": "stage_three_unaccounted_%s" % ws,
            "kind": "REVIEW_BLOCK",
            "role": missing_role,
            "text": token.text,
            "value": _review_block_hint(wp),
            "input_value": "",
            "span": [ws, ws + 1],
        })
    return missing


def _stage_three_operation_name(proof):
    for block in proof.get("blocks") or []:
        token = block.get("token")
        if token == "ANA_I":
            return "anagram"
        if token == "REV_I":
            return "reversal"
        if token == "CON_I":
            return "container"
        if token == "DEL_I":
            return "deletion"
        if token == "HID_I":
            return "hidden"
        if token == "HOM_I":
            return "homophone"
        if isinstance(token, str) and token.startswith("POS_I"):
            return "positional"
    transformations = proof.get("transformations") or []
    if transformations:
        kinds = [
            item.get("kind") for item in transformations
            if item.get("kind")
        ]
        if kinds:
            return " + ".join(dict.fromkeys(kinds))
    if proof.get("atomic_links"):
        return "charade"
    return "review"


def _stage_three_operation_detail(proof):
    checks = proof.get("checks") or []
    assembly = next(
        (check for check in checks if check.get("name") == "answer_assembly"),
        None,
    )
    if assembly and assembly.get("detail"):
        return assembly["detail"]
    failed = [
        check.get("detail") for check in checks
        if check.get("status") != "PASS" and check.get("detail")
    ]
    return "; ".join(failed[:2])


def _stage_three_review_message(check):
    name = check.get("name")
    if name == "source_evidence":
        evidence = check.get("evidence") or []
        failed = [
            "%s -> %s (%s)" % (
                item.get("text") or "?",
                item.get("value") or "?",
                item.get("evidence_reason") or "not verified",
            )
            for item in evidence
            if item.get("evidence_status") == "failed"
        ]
        if failed:
            return "Unverified source derivation: %s" % "; ".join(failed[:3])
    if name in ("surface_coverage", "word_purpose_coverage"):
        evidence = check.get("evidence") or []
        words = [item.get("text") for item in evidence if item.get("text")]
        if words:
            return "These clue words need a clear purpose before publication: %s" % (
                ", ".join(words[:4]))
    if name == "conditional_facts":
        return "This needs reviewed enrichment before it can be published as WFW."
    if name == "answer_assembly":
        return "The answer assembly is not yet fully verified."
    if name == "definition_evidence":
        return "The definition still needs accepted evidence."
    detail = check.get("detail")
    if detail:
        return detail
    return (name or "").replace("_", " ")


def _stage_three_missing_enrichments(proof):
    missing = []
    for item in proof.get("required_enrichments") or []:
        kind = item.get("kind") or "enrichment"
        text = item.get("text") or item.get("word") or item.get("definition") or ""
        value = item.get("value") or item.get("answer") or item.get("synonym") or ""
        if kind == "definition_gap":
            missing.append({
                "type": "definition",
                "definition": text,
                "answer": value,
                "kind": kind,
            })
        else:
            missing.append({
                "type": "synonym",
                "word": text,
                "synonym": value,
                "kind": kind,
            })
    for item in proof.get("purpose_requests") or []:
        missing.append({
            "type": "purpose",
            "word": item.get("text") or "",
            "kind": item.get("kind") or "word_purpose_evidence",
            "purpose": item.get("purpose") or "",
            "needed_evidence": item.get("needed_evidence") or "",
            "reason": item.get("reason") or "",
            "atoms": item.get("atoms") or [],
        })
    return missing


def _display_from_manual_correction(attempt_row, proof):
    atom_context = proof.get("atom_context") or {}
    clue_tokens = atom_context.get("clue_tokens") or []
    answer = proof.get("answer") or ""
    working_blocks = proof.get("working_blocks") or []
    block_by_id = {block.get("block_id"): block for block in working_blocks}

    blocks = []
    definition = proof.get("definition") or {}
    if definition:
        blocks.append({
            "block_id": "wfw_manual_definition",
            "kind": "DEF_BLOCK",
            "role": "definition",
            "text": definition.get("text") or "",
            "value": answer,
            "span": definition.get("span"),
        })
    for idx, block in enumerate(working_blocks):
        blocks.append({
            "block_id": block.get("block_id") or "manual_piece_%02d" % idx,
            "kind": "SOURCE_BLOCK",
            "role": "piece_%d" % idx,
            "text": block.get("text") or "",
            "value": block.get("value") or "",
            "span": _span_from_token_ids(clue_tokens, block.get("source_token_ids")),
        })

    answer_links = []
    for placement in proof.get("placements") or []:
        source_id = placement.get("source_block_id")
        source = block_by_id.get(source_id, {})
        piece_i = _piece_index(source_id)
        if piece_i is None:
            piece_i = _manual_piece_index(source_id)
        answer_links.append({
            "answer_index": (placement.get("answer_position") or 1) - 1,
            "letter": placement.get("answer_letter") or "",
            "source_block": source_id,
            "source_span": _span_from_token_ids(
                clue_tokens, source.get("source_token_ids")),
            "source_text": source.get("text") or "",
            "source_role": (
                "piece_%d" % piece_i if piece_i is not None else "piece"
            ),
            "source_value": source.get("value") or "",
            "source_value_index": placement.get("source_index"),
        })

    detail = " + ".join(
        block.get("value") or "" for block in working_blocks
    )
    if detail:
        detail = "%s = %s" % (detail, answer)

    missing = proof.get("missing_enrichments") or []
    objections = []
    if missing:
        objections = [
            "%s:%s" % (
                item.get("type"),
                item.get("definition") or item.get("word") or "",
            )
            for item in missing
        ]

    return {
        "status": "wfw_proven",
        "objections": objections,
        "clue_text": proof.get("clue_text") or "",
        "answer": answer,
        "tokens": _display_tokens(clue_tokens),
        "blocks": sorted(blocks, key=_block_sort_key),
        "operations": [{
            "operation": proof.get("operation") or "manual_correction",
            "detail": detail,
        }],
        "answer_links": sorted(answer_links, key=lambda link: link["answer_index"]),
        "proof_source": attempt_row.get("proof_source"),
        "proof_row_id": attempt_row.get("id"),
        "missing_enrichments": missing,
    }


def _display_from_unified_proof(attempt_row, proof):
    atom_context = proof.get("atom_context") or {}
    clue_tokens = atom_context.get("clue_tokens") or []
    answer = proof.get("answer") or ""
    token_parse = proof.get("token_parse") or {}
    assembly = proof.get("assembly") or {}
    working_blocks = assembly.get("working_blocks") or []
    transformations = assembly.get("transformations") or []
    block_by_id = {block.get("block_id"): block for block in working_blocks}
    parse_blocks = token_parse.get("blocks") or []
    parse_by_work_id = {
        "wfw_%s" % _clean_id(block.get("block_id") or ""): block
        for block in parse_blocks
    }

    blocks = []
    for block in parse_blocks:
        if block.get("kind") == "DEF_BLOCK":
            blocks.append({
                "block_id": block.get("block_id"),
                "kind": "DEF_BLOCK",
                "role": block.get("role") or "definition",
                "text": block.get("text") or "",
                "value": answer,
                "span": block.get("span"),
            })
        elif block.get("kind") in ("LINK_BLOCK", "OP_BLOCK", "RELATION_BLOCK"):
            blocks.append({
                "block_id": block.get("block_id"),
                "kind": block.get("kind"),
                "role": block.get("role"),
                "text": block.get("text") or "",
                "value": block.get("value") or "",
                "input_value": block.get("input_value") or "",
                "span": block.get("span"),
            })

    for idx, block in enumerate(working_blocks):
        if block.get("kind") not in ("source_piece", "transformed_piece"):
            continue
        parse_block = parse_by_work_id.get(block.get("block_id"), {})
        role = parse_block.get("role") or block.get("mechanism") or "piece_%d" % idx
        display_bits = _deletion_display_bits(
            parse_block.get("input_value") or "",
            block.get("value") or "",
        )
        blocks.append({
            "block_id": block.get("block_id") or "wfw_piece_%02d" % idx,
            "kind": "SOURCE_BLOCK",
            "role": role,
            "text": block.get("text") or parse_block.get("text") or "",
            "value": block.get("value") or "",
            "input_value": parse_block.get("input_value") or "",
            "kept_prefix": display_bits.get("kept_prefix", ""),
            "deleted_text": display_bits.get("deleted_text", ""),
            "kept_suffix": display_bits.get("kept_suffix", ""),
            "span": (
                parse_block.get("span")
                or _span_from_token_ids(clue_tokens, block.get("source_token_ids"))
            ),
        })

    answer_links = []
    for placement in assembly.get("placements") or []:
        source_id = placement.get("source_block_id")
        source = block_by_id.get(source_id, {})
        parse_block = parse_by_work_id.get(source_id, {})
        answer_links.append({
            "answer_index": (placement.get("answer_position") or 1) - 1,
            "letter": placement.get("answer_letter") or "",
            "source_block": source_id,
            "source_span": (
                parse_block.get("span")
                or _span_from_token_ids(
                    clue_tokens, source.get("source_token_ids"))
            ),
            "source_text": source.get("text") or parse_block.get("text") or "",
            "source_role": parse_block.get("role") or source.get("mechanism"),
            "source_value": source.get("value") or parse_block.get("value") or "",
            "source_input_value": parse_block.get("input_value") or "",
            "source_value_index": placement.get("source_index"),
        })

    operation_detail = _unified_operation_detail(token_parse, transformations)
    validation_objections = list(validate_unified_wfw_proof(proof))
    proof_objections = list(proof.get("objections") or [])
    objections = list(dict.fromkeys(proof_objections + validation_objections))
    review_messages = _review_messages_for_unified_proof(
        {**proof, "objections": objections}, token_parse)
    status = attempt_row.get("status") or proof.get("status")
    if validation_objections:
        status = "wfw_review"
    return {
        "status": status,
        "objections": objections,
        "review_messages": review_messages,
        "clue_text": proof.get("clue_text") or "",
        "answer": answer,
        "tokens": _display_tokens(clue_tokens),
        "blocks": sorted(blocks, key=_block_sort_key),
        "operations": [{
            "operation": assembly.get("operation") or token_parse.get("operation"),
            "detail": operation_detail,
        }],
        "answer_links": (
            sorted(answer_links, key=lambda link: link["answer_index"])
            if answer_links else _plain_answer_links(answer)
        ),
        "proof_source": attempt_row.get("proof_source"),
        "proof_row_id": attempt_row.get("id"),
    }


def _review_messages_for_unified_proof(proof, token_parse):
    messages = []
    objections = set(proof.get("objections") or [])
    blocks = token_parse.get("blocks") or []

    if "definition_not_db_verified" in objections:
        definitions = [
            block.get("text") for block in blocks
            if block.get("kind") == "DEF_BLOCK"
            and block.get("role") == "inferred_definition"
            and block.get("text")
        ]
        if definitions:
            messages.append(
                "Wordplay assembles, but the definition still needs DB evidence: %s"
                % ", ".join(definitions[:2]))
        else:
            messages.append(
                "Wordplay assembles, but the definition still needs DB evidence")

    if "surface_roles_not_verified" in objections:
        surface_words = [
            block.get("text") for block in blocks
            if block.get("token") == "SURFACE_GAP" and block.get("text")
        ]
        if surface_words:
            messages.append(
                "These surface words need a WFW role before publication: %s"
                % ", ".join(surface_words[:4]))
        else:
            messages.append(
                "Some clue words still need a WFW role before publication")

    if "definition_facts_need_review" in objections:
        messages.append(
            "Double-definition evidence needs review before publication")

    if not messages:
        for objection in proof.get("objections") or []:
            messages.append(objection.replace("_", " "))
    return messages


def display_from_missing_wfw(clue_text, answer, reason="missing_wfw_proposal"):
    """Return the same WFW display shape when no proof proposal exists yet."""
    context = build_wfw_atom_context(clue_text or "", answer or "")
    blocks = [
        {
            "block_id": "wfw_unaccounted_%s" % token.index,
            "kind": "REVIEW_BLOCK",
            "role": "unaccounted",
            "text": token.text,
            "value": "",
            "span": [token.index, token.index + 1],
        }
        for token in context.clue_tokens
        if token.kind == "word"
    ]
    return {
        "status": "wfw_review",
        "objections": [reason],
        "clue_text": clue_text or "",
        "answer": answer or "",
        "tokens": [
            {
                "index": token.index,
                "text": token.text,
                "normalized": token.text.upper(),
                "roles": [],
            }
            for token in context.clue_tokens
        ],
        "blocks": blocks,
        "operations": [{
            "operation": "review",
            "detail": "WFW has no structured proposal for this clue yet",
        }],
        "answer_links": _plain_answer_links(answer or ""),
        "proof_source": "missing",
        "proof_row_id": None,
    }


def _display_tokens(clue_tokens):
    return [
        {
            "index": token.get("index"),
            "text": token.get("text"),
            "normalized": (token.get("text") or "").upper(),
            "roles": [],
        }
        for token in clue_tokens
    ]


def _source_blocks(proposals, clue_tokens):
    blocks = []
    for proposal in proposals:
        piece_i = _piece_index(proposal.get("proposal_id"))
        if piece_i is None:
            continue
        token_indices = proposal.get("source_token_indices") or []
        span = None
        text = proposal.get("clue_word") or ""
        if token_indices:
            start = min(token_indices)
            end = max(token_indices) + 1
            span = [start, end]
            selected = [
                token.get("text") or ""
                for token in clue_tokens
                if start <= token.get("index", -1) < end
                and token.get("kind") == "word"
            ]
            if selected:
                text = " ".join(selected)
        blocks.append({
            "block_id": "wfw_piece_%s" % piece_i,
            "kind": "SOURCE_BLOCK",
            "role": _proposal_role(proposal),
            "text": text,
            "value": proposal.get("proposed_value") or "",
            "span": span,
        })
    return blocks


def _operation_blocks(transformations, clue_atoms, clue_tokens):
    blocks = []
    atom_to_token = {}
    for token in clue_tokens:
        for atom_id in token.get("atom_ids") or []:
            atom_to_token[atom_id] = token.get("index")
    for i, transform in enumerate(transformations):
        operation = transform.get("operation") or "operation"
        controller_ids = transform.get("controller_atom_ids") or []
        text = _text_from_atom_ids(controller_ids, clue_atoms)
        token_indices = sorted({
            atom_to_token[atom_id]
            for atom_id in controller_ids
            if atom_id in atom_to_token and atom_to_token[atom_id] is not None
        })
        span = None
        if token_indices:
            span = [min(token_indices), max(token_indices) + 1]
        if not text:
            text = operation.replace("_", " ")
        blocks.append({
            "block_id": "wfw_operation_%s" % i,
            "kind": "OP_BLOCK",
            "role": _operation_role(operation),
            "text": text,
            "value": "",
            "span": span,
        })
    return blocks


def _coverage_blocks(coverage, existing_texts):
    blocks = []
    for item in coverage.get("roles") or []:
        text = item.get("text") or ""
        for role in item.get("roles") or []:
            role_name = role.get("role")
            if role_name == "definition":
                blocks.append({
                    "block_id": "wfw_definition_%s" % item.get("index"),
                    "kind": "DEF_BLOCK",
                    "role": "definition",
                    "text": text,
                    "value": "",
                    "span": [item.get("index"), item.get("index") + 1],
                })
            elif role_name == "link" and text not in existing_texts:
                blocks.append({
                    "block_id": "wfw_link_%s" % item.get("index"),
                    "kind": "LINK_BLOCK",
                    "role": "link",
                    "text": text,
                    "value": "",
                    "span": [item.get("index"), item.get("index") + 1],
                })
    for item in coverage.get("uncovered") or []:
        blocks.append({
            "block_id": "wfw_unaccounted_%s" % item.get("index"),
            "kind": "REVIEW_BLOCK",
            "role": "unaccounted",
            "text": item.get("text") or "",
            "value": "",
            "span": [item.get("index"), item.get("index") + 1],
        })
    return blocks


def _operation_summary(proof_attempt):
    transformations = proof_attempt.get("transformations") or []
    details = [
        transform.get("detail")
        for transform in transformations
        if transform.get("detail")
    ]
    return {
        "operation": proof_attempt.get("operation") or "unknown",
        "detail": "; ".join(details),
    }


def _block_sort_key(block):
    span = block.get("span")
    if span:
        return (span[0], span[1], block.get("block_id") or "")
    return (9999, 9999, block.get("block_id") or "")


def _unified_operation_detail(token_parse, transformations):
    operations = token_parse.get("operations") or []
    details = [op.get("detail") for op in operations if op.get("detail")]
    if details:
        return "; ".join(details)
    details = [
        transform.get("detail")
        for transform in transformations
        if transform.get("detail")
    ]
    return "; ".join(details)


def _clean_id(value):
    return re.sub(r"[^A-Za-z0-9_]+", "_", value or "").strip("_")


def _deletion_display_bits(input_value, output_value):
    input_clean = _clean_letters(input_value)
    output_clean = _clean_letters(output_value)
    if not input_clean or not output_clean or input_clean == output_clean:
        return {}
    if input_clean.startswith(output_clean):
        return {
            "kept_prefix": output_clean,
            "deleted_text": input_clean[len(output_clean):],
            "kept_suffix": "",
        }
    if input_clean.endswith(output_clean):
        return {
            "kept_prefix": "",
            "deleted_text": input_clean[:len(input_clean) - len(output_clean)],
            "kept_suffix": output_clean,
        }
    start = input_clean.find(output_clean)
    if start >= 0:
        end = start + len(output_clean)
        return {
            "kept_prefix": input_clean[:end],
            "deleted_text": input_clean[end:],
            "kept_suffix": "",
        }
    return {}


def _answer_links(placements, proposal_by_index):
    links = []
    for placement in placements:
        piece_i = _piece_index(placement.get("source_block_id"))
        proposal = proposal_by_index.get(piece_i, {})
        links.append({
            "answer_index": (placement.get("answer_position") or 1) - 1,
            "letter": placement.get("answer_letter") or "",
            "source_block": "wfw_piece_%s" % piece_i if piece_i is not None else None,
            "source_span": None,
            "source_text": proposal.get("clue_word") or "",
            "source_role": (
                _proposal_role(proposal)
                if piece_i is not None else None
            ),
            "source_value": proposal.get("proposed_value") or "",
            "source_value_index": placement.get("source_index"),
        })
    return sorted(links, key=lambda link: link["answer_index"])


def _plain_answer_links(answer):
    return [
        {
            "answer_index": idx,
            "letter": letter,
            "source_block": None,
            "source_span": None,
            "source_text": "",
            "source_role": None,
            "source_value": "",
            "source_value_index": None,
        }
        for idx, letter in enumerate(_clean_letters(answer))
    ]


def _span_from_token_ids(clue_tokens, token_ids):
    if not token_ids:
        return None
    ids = set(token_ids)
    indices = [
        token.get("index")
        for token in clue_tokens
        if token.get("token_id") in ids
    ]
    indices = [idx for idx in indices if idx is not None]
    if not indices:
        return None
    return [min(indices), max(indices) + 1]


def _piece_index(value):
    if not value:
        return None
    match = re.search(r"piece_(\d+)", value)
    if not match:
        return None
    return int(match.group(1))


def _manual_piece_index(value):
    if not value:
        return None
    match = re.search(r"manual_piece_(\d+)", value)
    if not match:
        return None
    return int(match.group(1))


def _operation_role(operation):
    if operation == "reversal":
        return "reversal_indicator"
    if operation in ("trim_last", "trim_first", "deletion"):
        return "deletion_indicator"
    if operation == "anagram":
        return "anagram_indicator"
    if operation == "container":
        return "container_indicator"
    return "%s_indicator" % operation


def _proposal_role(proposal):
    if (proposal.get("mechanism") or "").lower() == "hidden":
        return "hidden_fodder"
    return proposal.get("proposal_id") or "piece"


def _hidden_segments(display, proof_attempt):
    if proof_attempt.get("operation") != "hidden":
        return None
    source = next(
        (block for block in display.get("blocks") or []
         if block.get("role") == "hidden_fodder"),
        None,
    )
    if not source:
        return None
    answer = _clean_letters(display.get("answer"))
    source_value = _clean_letters(source.get("text"))
    start = source_value.find(answer)
    reversed_hidden = False
    if start < 0:
        reversed_answer = answer[::-1]
        start = source_value.find(reversed_answer)
        reversed_hidden = start >= 0
    if start < 0:
        return None
    end = start + len(answer)
    display_prefix, display_hidden, display_suffix = _display_hidden_segments(
        source.get("text") or "", start, end)
    return {
        "source_block": source.get("block_id"),
        "source_text": source.get("text"),
        "source_value": source_value,
        "prefix": source_value[:start],
        "hidden": source_value[start:end],
        "suffix": source_value[end:],
        "display_prefix": display_prefix,
        "display_hidden": display_hidden,
        "display_suffix": display_suffix,
        "answer_order": answer,
        "reversed": reversed_hidden,
    }


def _display_hidden_segments(text, clean_start, clean_end):
    letters_seen = 0
    char_start = None
    char_end = None
    for idx, char in enumerate(text or ""):
        if not char.isalpha():
            continue
        if letters_seen == clean_start and char_start is None:
            char_start = idx
        letters_seen += 1
        if letters_seen == clean_end:
            char_end = idx + 1
            break
    if char_start is None or char_end is None:
        return "", text or "", ""
    return text[:char_start], text[char_start:char_end], text[char_end:]


def _clean_letters(value):
    return "".join(char.upper() for char in (value or "") if char.isalpha())


def _text_from_atom_ids(atom_ids, clue_atoms):
    chars = [
        clue_atoms[atom_id].get("char") or ""
        for atom_id in atom_ids
        if atom_id in clue_atoms
    ]
    return "".join(chars).strip()
