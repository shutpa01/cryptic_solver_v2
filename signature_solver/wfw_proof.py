"""WFW proof-gated record builder.

This is the first single-system shape: evidence can come from obase, grammar,
or future sources, but the record status is controlled by WFW proof.
"""
from __future__ import annotations

from dataclasses import dataclass

from .wfw_atoms import build_wfw_atom_context
from .wfw_grammar import grammar_evidence_from_clue_context
from .wfw_obase_bridge import prove_obase_charade
from .wfw_obase_bridge import prove_obase_anagram
from .wfw_obase_bridge import prove_obase_container
from .wfw_obase_bridge import prove_obase_hidden


@dataclass(frozen=True)
class WFWProofRecord:
    status: str
    clue_text: str
    answer: str
    atom_context: object
    grammar_evidence: object | None
    proof_attempt: object | None
    source: str
    coverage: object | None = None
    objections: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "status": self.status,
            "clue_text": self.clue_text,
            "answer": self.answer,
            "atom_context": self.atom_context.as_dict(),
            "grammar_evidence": (
                self.grammar_evidence.as_dict()
                if self.grammar_evidence is not None else None
            ),
            "proof_attempt": (
                self.proof_attempt.as_dict()
                if self.proof_attempt is not None else None
            ),
            "source": self.source,
            "coverage": self.coverage,
            "objections": list(self.objections),
        }


def build_wfw_proof_from_obase(clue_text, answer, components,
                               ai_explanation="", clue_context=None,
                               definition_text=None):
    """Build a WFW proof record from obase structured evidence."""
    atom_context = build_wfw_atom_context(clue_text, answer)
    grammar_evidence = None
    if clue_context is not None:
        grammar_evidence = grammar_evidence_from_clue_context(
            atom_context, clue_context)

    components = components or {}
    operation = (components.get("assembly") or {}).get("op") or ""
    if operation == "charade":
        attempt = prove_obase_charade(
            atom_context, components, ai_explanation=ai_explanation)
        coverage = _coverage(atom_context, attempt, definition_text)
        status, objections = _status_with_coverage(attempt, coverage)
        return WFWProofRecord(
            status=status,
            clue_text=clue_text,
            answer=answer,
            atom_context=atom_context,
            grammar_evidence=grammar_evidence,
            proof_attempt=attempt,
            source="obase_structured",
            coverage=coverage,
            objections=objections,
        )
    if operation == "container":
        attempt = prove_obase_container(
            atom_context, components, ai_explanation=ai_explanation)
        coverage = _coverage(atom_context, attempt, definition_text)
        status, objections = _status_with_coverage(attempt, coverage)
        return WFWProofRecord(
            status=status,
            clue_text=clue_text,
            answer=answer,
            atom_context=atom_context,
            grammar_evidence=grammar_evidence,
            proof_attempt=attempt,
            source="obase_structured",
            coverage=coverage,
            objections=objections,
        )
    if operation == "anagram":
        attempt = prove_obase_anagram(
            atom_context, components, ai_explanation=ai_explanation)
        coverage = _coverage(atom_context, attempt, definition_text)
        status, objections = _status_with_coverage(attempt, coverage)
        return WFWProofRecord(
            status=status,
            clue_text=clue_text,
            answer=answer,
            atom_context=atom_context,
            grammar_evidence=grammar_evidence,
            proof_attempt=attempt,
            source="obase_structured",
            coverage=coverage,
            objections=objections,
        )
    if operation == "hidden":
        attempt = prove_obase_hidden(
            atom_context, components, ai_explanation=ai_explanation)
        coverage = _coverage(atom_context, attempt, definition_text)
        status, objections = _status_with_coverage(attempt, coverage)
        return WFWProofRecord(
            status=status,
            clue_text=clue_text,
            answer=answer,
            atom_context=atom_context,
            grammar_evidence=grammar_evidence,
            proof_attempt=attempt,
            source="obase_structured",
            coverage=coverage,
            objections=objections,
        )

    return WFWProofRecord(
        status="wfw_review",
        clue_text=clue_text,
        answer=answer,
        atom_context=atom_context,
        grammar_evidence=grammar_evidence,
        proof_attempt=None,
        source="obase_structured",
        coverage=None,
        objections=("unsupported_operation:%s" % (operation or "unknown"),),
    )


def _status_with_coverage(attempt, coverage):
    objections = tuple(attempt.objections or ()) + tuple(
        "uncovered_word:%s" % item["text"]
        for item in coverage.get("uncovered", [])
    )
    if attempt.status == "proven" and not coverage.get("uncovered"):
        return "wfw_proven", objections
    return "wfw_review", objections


def _coverage(context, attempt, definition_text):
    roles = {}

    for proposal in getattr(attempt, "proposals", ()) or ():
        for idx in proposal.source_token_indices:
            roles.setdefault(idx, []).append({
                "role": "source",
                "text": context.clue_tokens[idx].text,
                "value": proposal.proposed_value,
                "proposal_id": proposal.proposal_id,
            })

    atom_to_token = {}
    for token in context.clue_tokens:
        for atom_id in token.atom_ids:
            atom_to_token[atom_id] = token.index
    for transform in getattr(attempt, "transformations", ()) or ():
        token_indices = sorted({
            atom_to_token[atom_id]
            for atom_id in transform.controller_atom_ids
            if atom_id in atom_to_token
        })
        for idx in token_indices:
            roles.setdefault(idx, []).append({
                "role": _operation_role(transform.operation),
                "text": context.clue_tokens[idx].text,
                "value": "",
                "operation": transform.operation,
            })

    for idx in _find_token_indices(context, definition_text or ""):
        roles.setdefault(idx, []).append({
            "role": "definition",
            "text": context.clue_tokens[idx].text,
            "value": "",
        })

    for token in context.clue_tokens:
        if token.kind != "word":
            continue
        if token.index in roles:
            continue
        if _token_key(token.text) in _LINK_WORDS:
            roles.setdefault(token.index, []).append({
                "role": "link",
                "text": token.text,
                "value": "",
            })

    uncovered = [
        {
            "index": token.index,
            "text": token.text,
        }
        for token in context.clue_tokens
        if token.kind == "word" and token.index not in roles
    ]
    return {
        "roles": [
            {
                "index": idx,
                "text": context.clue_tokens[idx].text,
                "roles": token_roles,
            }
            for idx, token_roles in sorted(roles.items())
        ],
        "uncovered": uncovered,
    }


def _find_token_indices(context, phrase):
    phrase_keys = [
        _token_key(token.text)
        for token in build_wfw_atom_context(phrase or "", "").clue_tokens
        if _token_key(token.text)
    ]
    if not phrase_keys:
        return ()
    clue_keys = [_token_key(token.text) for token in context.clue_tokens]
    n = len(phrase_keys)
    for start in range(0, len(clue_keys) - n + 1):
        if clue_keys[start:start + n] == phrase_keys:
            return tuple(range(start, start + n))
    return ()


def _operation_role(operation):
    if operation == "reversal":
        return "reversal_indicator"
    if operation in {"trim_last", "trim_first", "deletion"}:
        return "deletion_indicator"
    if operation == "anagram":
        return "anagram_indicator"
    if operation == "container":
        return "container_indicator"
    if operation in {"hidden", "hidden_reversed"}:
        return "hidden_indicator"
    return "%s_indicator" % operation


def _token_key(text):
    return "".join(char.upper() for char in (text or "") if char.isalnum())


_LINK_WORDS = {
    "A", "AN", "AND", "AS", "AT", "BE", "BY", "FOR", "FROM", "HAS", "HAVE",
    "IN", "INTO", "IS", "OF", "ON", "OR", "THE", "TO", "WITH",
    "COVERING", "GETTING",
}
