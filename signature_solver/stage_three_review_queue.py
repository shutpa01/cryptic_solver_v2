"""Read-only queue payloads for Stage Three proof gaps.

Stage Three does not write review rows.  These helpers shape failed proof
checks and evidence requests into records that can be written by an outer
admin/run layer to ``atomic_parse_review_items``.
"""
from __future__ import annotations


MECHANICAL_CHECKS = {
    "definition_evidence",
    "answer_assembly",
    "source_evidence",
    "assembly_order",
    "operation_evidence",
    "operation_attachment",
    "mechanism_rules",
    "atomic_coverage",
    "span_integrity",
}


def review_items_from_stage_three_proof(proof, clue_id=None):
    """Return queue-shaped review items for a StageThreeProof."""
    if hasattr(proof, "as_dict"):
        proof = proof.as_dict()
    proof = dict(proof or {})
    if proof.get("schema") != "stage_three_proof:v1":
        return ()

    items = []
    failed_checks = [
        check for check in proof.get("checks") or []
        if check.get("status") != "PASS"
    ]
    mechanical_failed = [
        check for check in failed_checks
        if check.get("name") in MECHANICAL_CHECKS
    ]
    has_action_requests = bool(
        (proof.get("purpose_requests") or ())
        or (proof.get("required_enrichments") or ())
    )
    if mechanical_failed:
        items.append(_mechanical_review_item(proof, mechanical_failed, clue_id))
    else:
        for check in failed_checks:
            if not _check_needs_review_item(check, has_action_requests):
                continue
            items.append(_check_review_item(proof, check, clue_id))
    required_enrichments = tuple(proof.get("required_enrichments") or ())
    if not mechanical_failed:
        for request in proof.get("purpose_requests") or []:
            if _purpose_request_covered_by_enrichment(
                    request, required_enrichments):
                continue
            items.append(_purpose_review_item(proof, request, clue_id))
    for request in required_enrichments:
        if _pending_enrichment_from_request(
                request,
                proof.get("answer") or "",
                proof.get("clue_text") or "",
                None,
                None):
            continue
        items.append(_enrichment_review_item(proof, request, clue_id))
    return tuple(_dedupe_review_items(items))


def _check_needs_review_item(check, has_action_requests):
    name = check.get("name")
    if name in {"mechanism_rules", "operation_attachment"}:
        return True
    if name == "assembly_order":
        return "not justified by left-to-right" in (check.get("detail") or "")
    if has_action_requests:
        return False
    return True


def _mechanical_review_item(proof, failed_checks, clue_id):
    names = [check.get("name") for check in failed_checks if check.get("name")]
    return {
        "clue_id": clue_id,
        "review_type": "stage_three:mechanical_proof",
        "summary": "Mechanical proof needs work: %s" % ", ".join(names[:5]),
        "payload": {
            "schema": "stage_three_review_item:v1",
            "source": "stage_three_mechanical_gate",
            "clue_text": proof.get("clue_text") or "",
            "answer": proof.get("answer") or "",
            "failed_checks": failed_checks,
        },
    }


def _purpose_request_covered_by_enrichment(request, enrichments):
    request_span = tuple(request.get("span") or ())
    if not request_span:
        return False
    for enrichment in enrichments:
        enrichment_span = tuple(enrichment.get("span") or ())
        if not enrichment_span:
            continue
        if (enrichment_span[0] <= request_span[0]
                and enrichment_span[1] >= request_span[1]):
            return True
    return False


def pending_enrichments_from_stage_three_proof(
        proof, clue_text=None, source=None, puzzle_number=None):
    """Return pending_enrichments-shaped rows for concrete DB facts only."""
    if hasattr(proof, "as_dict"):
        proof = proof.as_dict()
    proof = dict(proof or {})
    if proof.get("schema") != "stage_three_proof:v1":
        return ()

    clue_text = clue_text if clue_text is not None else proof.get("clue_text")
    rows = []
    for request in proof.get("required_enrichments") or []:
        row = _pending_enrichment_from_request(
            request, proof.get("answer") or "", clue_text or "",
            source, puzzle_number)
        if row:
            rows.append(row)
    return tuple(_dedupe_pending_enrichments(rows))


def _pending_enrichment_from_request(
        request, answer, clue_text, source, puzzle_number):
    kind = request.get("kind")
    text = request.get("text") or request.get("word") or ""
    value = request.get("value") or request.get("answer") or ""
    if not text or not value:
        return None
    if kind == "definition_gap":
        row_type = "definition"
        letters = answer or value
    elif kind in {"source_phrase_widening", "conditional_source_gap"}:
        row_type = "synonym"
        letters = value
    else:
        return None
    return {
        "type": row_type,
        "word": text,
        "letters": _clean_letters(letters),
        "answer": answer,
        "clue_text": clue_text,
        "source": source,
        "puzzle_number": puzzle_number,
        "stage_three_kind": kind,
    }


def _check_review_item(proof, check, clue_id):
    name = check.get("name") or "stage_three_check"
    return {
        "clue_id": clue_id,
        "review_type": "stage_three:%s" % name,
        "summary": "%s: %s" % (
            name.replace("_", " "),
            check.get("detail") or "needs review",
        ),
        "payload": {
            "schema": "stage_three_review_item:v1",
            "source": "stage_three_check",
            "clue_text": proof.get("clue_text") or "",
            "answer": proof.get("answer") or "",
            "check": check,
        },
    }


def _purpose_review_item(proof, request, clue_id):
    text = request.get("text") or ""
    kind = request.get("kind") or "word_purpose_evidence"
    return {
        "clue_id": clue_id,
        "review_type": "stage_three:%s" % kind,
        "summary": "Purpose evidence needed for %s" % text,
        "payload": {
            "schema": "stage_three_review_item:v1",
            "source": "stage_three_purpose_request",
            "clue_text": proof.get("clue_text") or "",
            "answer": proof.get("answer") or "",
            "request": request,
        },
    }


def _enrichment_review_item(proof, request, clue_id):
    text = request.get("text") or request.get("word") or ""
    kind = request.get("kind") or "enrichment"
    return {
        "clue_id": clue_id,
        "review_type": "stage_three:%s" % kind,
        "summary": "Enrichment needed for %s" % text,
        "payload": {
            "schema": "stage_three_review_item:v1",
            "source": "stage_three_required_enrichment",
            "clue_text": proof.get("clue_text") or "",
            "answer": proof.get("answer") or "",
            "request": request,
        },
    }


def _dedupe_review_items(items):
    seen = set()
    out = []
    for item in items:
        payload = item.get("payload") or {}
        request = payload.get("request") or payload.get("check") or {}
        key = (
            item.get("clue_id"),
            item.get("review_type"),
            request.get("kind") or request.get("name"),
            request.get("text") or item.get("summary"),
            tuple(request.get("span") or ()),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out


def _dedupe_pending_enrichments(rows):
    seen = set()
    out = []
    for row in rows:
        key = (
            row.get("type"),
            (row.get("word") or "").lower(),
            _clean_letters(row.get("letters")),
        )
        if key in seen:
            continue
        seen.add(key)
        out.append(row)
    return out


def _clean_letters(value):
    return "".join(char for char in (value or "").upper() if char.isalpha())
