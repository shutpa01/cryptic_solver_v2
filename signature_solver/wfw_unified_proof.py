"""Persistable WFW proof records from the unified solver.

This module is intentionally small: it does not translate WFW back into an
old prose explanation.  It packages the unified solver's token parse and
materialised working blocks so the clue page can render the WFW result
directly.
"""
from __future__ import annotations

from .wfw_proof_validator import validate_unified_wfw_proof


def build_wfw_proof_from_unified_result(wfw_result):
    """Return a proof-store dict for a WFWUnifiedResult."""
    if wfw_result is None:
        return None
    data = wfw_result.as_dict()
    token_parses = data.get("token_parses") or []
    assemblies = data.get("wfw_assemblies") or []
    proof = _first_valid_proof(data, token_parses, assemblies)
    if proof is not None:
        return proof

    assembly = _first_materialised_assembly(assemblies)
    token_parse = _matching_parse(token_parses, assembly) if assembly else None

    return _build_proof(data, token_parse, assembly)


def _first_valid_proof(data, token_parses, assemblies):
    for assembly in assemblies:
        if assembly.get("status") != "materialised":
            continue
        token_parse = _matching_parse(token_parses, assembly)
        proof = _build_proof(data, token_parse, assembly)
        if proof.get("status") == "wfw_proven":
            return proof
    return None


def _build_proof(data, token_parse, assembly):
    objections = []
    if not token_parse:
        objections.append("missing_token_parse")
    if not assembly:
        objections.append("missing_materialised_assembly")
    if token_parse and token_parse.get("confidence") == (
            "mechanically_verified_inferred_definition"):
        objections.append("definition_not_db_verified")
    if token_parse and token_parse.get("confidence") == (
            "mechanically_verified_surface_gaps"):
        objections.append("surface_roles_not_verified")
    if token_parse and token_parse.get("confidence") == (
            "mechanically_verified_definition_gaps"):
        objections.append("definition_facts_need_review")

    proof = {
        "schema": "wfw_unified_proof:v1",
        "source": "wfw_unified_solver",
        "status": "wfw_review",
        "clue_text": data.get("clue_text") or "",
        "answer": data.get("answer") or "",
        "atom_context": data.get("atom_context") or {},
        "grammar_evidence": data.get("grammar_evidence"),
        "token_parse": token_parse,
        "assembly": assembly,
        "stages": data.get("stages") or [],
        "objections": objections,
        "legacy_role": data.get("legacy_role"),
    }

    objections.extend(validate_unified_wfw_proof(proof))

    status = "wfw_review"
    if not objections and (assembly or {}).get("status") == "materialised":
        status = "wfw_proven"
    proof["status"] = status
    proof["objections"] = list(dict.fromkeys(objections))

    return proof


def _first_materialised_assembly(assemblies):
    for assembly in assemblies:
        if assembly.get("status") == "materialised":
            return assembly
    return assemblies[0] if assemblies else None


def _matching_parse(token_parses, assembly):
    parse_id = assembly.get("parse_id") if assembly else None
    for token_parse in token_parses:
        if token_parse.get("parse_id") == parse_id:
            return token_parse
    return token_parses[0] if token_parses else None
