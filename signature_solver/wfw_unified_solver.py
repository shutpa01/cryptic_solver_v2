"""WFW-native solver contract.

This module is the architectural entry point for the new signature solver.  It
does not accept a flattened ``SignatureResult`` as its input.  The solve starts
with WFW character atoms, then carries grammar evidence, surface candidates,
mechanical candidates, working blocks, assembly, and verification as separate
data layers.
"""
from __future__ import annotations

from dataclasses import dataclass

from .clue_context import build_clue_context, with_wordplay_annotations
from .wfw_atoms import WFWAtomContext, build_wfw_atom_context
from .wfw_grammar import WFWGrammarEvidence, grammar_evidence_from_clue_context


OPERATION_FAMILIES = (
    "charade",
    "anagram",
    "hidden",
    "reversal",
    "container",
    "deletion",
    "substitution",
    "selection",
    "homophone",
    "spoonerism",
    "double_definition",
    "cryptic_definition_triage",
)

CONNECTOR_LIKE_WORDS = {
    "about", "after", "and", "before", "behind", "by", "for", "following",
    "in", "into", "on", "with",
}


@dataclass(frozen=True)
class WFWStageRecord:
    stage_id: str
    name: str
    status: str
    detail: str = ""

    def as_dict(self):
        return {
            "stage_id": self.stage_id,
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class WFWEvidenceItem:
    evidence_id: str
    kind: str
    source: str
    token_ids: tuple[str, ...] = ()
    atom_ids: tuple[str, ...] = ()
    values: tuple = ()
    status: str = "candidate"
    detail: str = ""

    def as_dict(self):
        return {
            "evidence_id": self.evidence_id,
            "kind": self.kind,
            "source": self.source,
            "token_ids": list(self.token_ids),
            "atom_ids": list(self.atom_ids),
            "values": list(self.values),
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class WFWCandidateNode:
    node_id: str
    kind: str
    token_ids: tuple[str, ...] = ()
    atom_ids: tuple[str, ...] = ()
    role: str | None = None
    value: str | None = None
    status: str = "candidate"
    detail: str = ""

    def as_dict(self):
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "token_ids": list(self.token_ids),
            "atom_ids": list(self.atom_ids),
            "role": self.role,
            "value": self.value,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class WFWCandidateEdge:
    edge_id: str
    kind: str
    from_node_id: str
    to_node_id: str
    status: str = "candidate"
    detail: str = ""

    def as_dict(self):
        return {
            "edge_id": self.edge_id,
            "kind": self.kind,
            "from_node_id": self.from_node_id,
            "to_node_id": self.to_node_id,
            "status": self.status,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class WFWUnifiedResult:
    """Authoritative WFW solve record.

    ``legacy_role`` is deliberately constrained: older obase machinery may
    contribute evidence, but the WFW record is the authority.
    """

    clue_text: str
    answer: str
    atom_context: WFWAtomContext
    stage_one_context: object | None
    clue_context: object | None
    grammar_evidence: WFWGrammarEvidence
    stages: tuple[WFWStageRecord, ...]
    evidence: tuple[WFWEvidenceItem, ...]
    candidate_nodes: tuple[WFWCandidateNode, ...]
    candidate_edges: tuple[WFWCandidateEdge, ...]
    operation_families: tuple[str, ...]
    token_parses: tuple = ()
    wfw_displays: tuple = ()
    wfw_assemblies: tuple = ()
    status: str = "review"
    legacy_role: str = "evidence_provider_only"

    def as_dict(self):
        return {
            "schema": "wfw_unified_solver:v1",
            "status": self.status,
            "legacy_role": self.legacy_role,
            "clue_text": self.clue_text,
            "answer": self.answer,
            "atom_context": self.atom_context.as_dict(),
            "stage_one_context": (
                self.stage_one_context.as_dict()
                if self.stage_one_context is not None else None
            ),
            "clue_context": (
                self.clue_context.as_dict()
                if self.clue_context is not None else None
            ),
            "grammar_evidence": self.grammar_evidence.as_dict(),
            "stages": [stage.as_dict() for stage in self.stages],
            "evidence": [item.as_dict() for item in self.evidence],
            "candidate_nodes": [
                node.as_dict() for node in self.candidate_nodes
            ],
            "candidate_edges": [
                edge.as_dict() for edge in self.candidate_edges
            ],
            "operation_families": list(self.operation_families),
            "token_parses": [
                parse.as_dict() if hasattr(parse, "as_dict") else parse
                for parse in self.token_parses
            ],
            "wfw_displays": list(self.wfw_displays),
            "wfw_assemblies": [
                assembly.as_dict()
                if hasattr(assembly, "as_dict") else assembly
                for assembly in self.wfw_assemblies
            ],
        }


def solve_wfw_unified(clue_text, answer, db=None, assemble=True,
                      manual_roles=None, clue_context=None,
                      stage_one_context=None):
    """Build a WFW-native solve record from the published clue text.

    The first irreversible contract is ordering: character atoms are built
    before any normalized clue context, DB evidence, grammar evidence, or
    mechanical assembly is considered.
    """
    atom_context = (
        getattr(clue_context, "atom_context", None)
        if clue_context is not None else None
    )
    if atom_context is None:
        atom_context = build_wfw_atom_context(clue_text, answer)
    stages = [
        WFWStageRecord(
            "stage_01_atoms",
            "wfw_atomization",
            "complete",
            "preserved original clue and answer character atoms",
        )
    ]

    if clue_context is not None:
        stages.append(WFWStageRecord(
            "stage_02_context",
            "surface_context_and_db_evidence",
            "complete",
            "reused shared token spans, definition candidates, and annotations",
        ))
    elif db is not None:
        stage_one_context = build_clue_context(
            clue_text, answer, db, annotate=False,
            manual_roles=None)
        clue_context = with_wordplay_annotations(stage_one_context, db)
        if manual_roles:
            from .clue_context import (
                _manual_role_annotations,
                with_added_annotations,
            )
            clue_context = with_added_annotations(
                clue_context,
                _manual_role_annotations(stage_one_context, manual_roles))
        stages.append(WFWStageRecord(
            "stage_02_context",
            "surface_context_and_db_evidence",
            "complete",
            "built token spans, definition candidates, and annotations",
        ))
    else:
        stages.append(WFWStageRecord(
            "stage_02_context",
            "surface_context_and_db_evidence",
            "skipped",
            "no DB supplied",
        ))

    grammar_evidence = (
        grammar_evidence_from_clue_context(atom_context, clue_context)
        if clue_context is not None
        else WFWGrammarEvidence(status="not_requested", spans=())
    )
    stages.append(WFWStageRecord(
        "stage_03_grammar",
        "grammar_triage_evidence",
        grammar_evidence.status,
        "grammar evidence is stored before mechanical solve attempts",
    ))

    evidence = _evidence_from_context(atom_context, clue_context)
    candidate_nodes, candidate_edges = _candidate_graph_from_context(
        atom_context, clue_context)
    stages.append(WFWStageRecord(
        "stage_04_candidate_graph",
        "candidate_graph",
        "complete",
        "surface, definition, connector, qualifier, and operation candidates",
    ))

    token_parses = ()
    wfw_displays = ()
    wfw_assemblies = ()
    if assemble and clue_context is not None:
        token_parses, wfw_displays, wfw_assemblies = (
            _assemble_native_candidates(atom_context, clue_context)
        )
        stages.append(WFWStageRecord(
            "stage_05_assembly",
            "answer_guided_assembly",
            "complete" if token_parses else "review",
            "assembly attempts are separate from stored evidence",
        ))
    else:
        stages.append(WFWStageRecord(
            "stage_05_assembly",
            "answer_guided_assembly",
            "skipped",
            "assembly disabled or no clue context",
        ))

    stages.append(WFWStageRecord(
        "stage_06_verification",
        "verification",
        "complete" if token_parses else "review",
        "verification is downstream of assembly, not a prose fallback",
    ))

    return WFWUnifiedResult(
        clue_text=clue_text,
        answer=answer,
        atom_context=atom_context,
        stage_one_context=stage_one_context,
        clue_context=clue_context,
        grammar_evidence=grammar_evidence,
        stages=tuple(stages),
        evidence=tuple(evidence),
        candidate_nodes=tuple(candidate_nodes),
        candidate_edges=tuple(candidate_edges),
        operation_families=OPERATION_FAMILIES,
        token_parses=tuple(token_parses),
        wfw_displays=tuple(wfw_displays),
        wfw_assemblies=tuple(wfw_assemblies),
        status="solved" if token_parses else "review",
    )


def _evidence_from_context(atom_context, clue_context):
    if clue_context is None:
        return ()
    token_map = _clue_context_to_wfw_token_map(atom_context, clue_context)
    items = []
    for idx, annotation in enumerate(clue_context.annotations):
        token_ids, atom_ids = _span_provenance(atom_context, token_map,
                                               annotation.span)
        items.append(WFWEvidenceItem(
            evidence_id="evidence_%04d" % idx,
            kind=annotation.token,
            source=annotation.source,
            token_ids=token_ids,
            atom_ids=atom_ids,
            values=tuple(annotation.values),
            status="candidate",
            detail=annotation.text,
        ))
    return tuple(items)


def _candidate_graph_from_context(atom_context, clue_context):
    nodes = []
    edges = []

    for token in atom_context.clue_tokens:
        role = None
        if token.text in {"?", "!"}:
            role = "possible_definition_qualifier"
        elif token.text.lower() in CONNECTOR_LIKE_WORDS:
            role = "connector_candidate_requires_license"
        nodes.append(WFWCandidateNode(
            node_id="surface_%s" % token.token_id,
            kind="SURFACE_TOKEN",
            token_ids=(token.token_id,),
            atom_ids=token.atom_ids,
            role=role,
            value=token.text,
            detail="original published token",
        ))

    if clue_context is not None:
        token_map = _clue_context_to_wfw_token_map(atom_context, clue_context)
        for idx, candidate in enumerate(clue_context.definition_candidates):
            token_ids, atom_ids = _span_provenance(
                atom_context, token_map, candidate.definition_span.as_tuple())
            def_id = "definition_candidate_%04d" % idx
            nodes.append(WFWCandidateNode(
                node_id=def_id,
                kind="DEF_BLOCK",
                token_ids=token_ids,
                atom_ids=atom_ids,
                role="defines_whole_answer",
                value=candidate.def_phrase,
                detail="definition candidate defines the whole answer",
            ))
            nodes.append(WFWCandidateNode(
                node_id="wordplay_window_%04d" % idx,
                kind="WORDPLAY_WINDOW",
                token_ids=_span_provenance(
                    atom_context, token_map,
                    candidate.wordplay_span.as_tuple())[0],
                atom_ids=_span_provenance(
                    atom_context, token_map,
                    candidate.wordplay_span.as_tuple())[1],
                role="candidate_wordplay_material",
                value=candidate.wordplay_span.text,
            ))
            edges.append(WFWCandidateEdge(
                edge_id="def_to_answer_%04d" % idx,
                kind="DEFINES_WHOLE_ANSWER",
                from_node_id=def_id,
                to_node_id="answer",
                status="candidate",
            ))

    nodes.append(WFWCandidateNode(
        node_id="answer",
        kind="ANSWER_BLOCK",
        token_ids=tuple(token.token_id for token in atom_context.answer_tokens),
        atom_ids=tuple(atom.atom_id for atom in atom_context.answer_atoms),
        role="whole_answer",
        value=atom_context.answer_text,
        status="given",
    ))

    for family in OPERATION_FAMILIES:
        nodes.append(WFWCandidateNode(
            node_id="operation_family_%s" % family,
            kind="OPERATION_FAMILY",
            role=family,
            status="available",
            detail="implemented as data-bearing WFW relationship",
        ))

    return tuple(nodes), tuple(edges)


def _assemble_native_candidates(atom_context, clue_context):
    try:
        from .token_parse_assembler import assemble_token_parses
        from .wfw_formatter import format_token_parse_for_wfw
        from .wfw_native_assembly import materialise_wfw_assembly
    except Exception:
        return (), (), ()

    parses = tuple(assemble_token_parses(clue_context))
    displays = tuple(
        format_token_parse_for_wfw(clue_context, parse)
        for parse in parses
    )
    assemblies = tuple(
        materialise_wfw_assembly(atom_context, clue_context, parse)
        for parse in parses
    )
    return parses, displays, assemblies


def _clue_context_to_wfw_token_map(atom_context, clue_context):
    mapping = {}
    wfw_index = 0
    for clue_token in clue_context.tokens:
        target = _token_key(clue_token.text)
        if not target:
            continue
        while (wfw_index < len(atom_context.clue_tokens)
               and not _token_key(atom_context.clue_tokens[wfw_index].text)):
            wfw_index += 1
        if wfw_index >= len(atom_context.clue_tokens):
            break
        if _token_key(atom_context.clue_tokens[wfw_index].text) == target:
            mapping[clue_token.index] = wfw_index
            wfw_index += 1
    return mapping


def _span_provenance(atom_context, token_map, span):
    start, end = span
    wfw_indices = [
        token_map[idx] for idx in range(start, end)
        if idx in token_map
    ]
    tokens = tuple(atom_context.clue_tokens[idx] for idx in wfw_indices)
    token_ids = tuple(token.token_id for token in tokens)
    atom_ids = tuple(atom_id for token in tokens for atom_id in token.atom_ids)
    return token_ids, atom_ids


def _token_key(text):
    return "".join(char.upper() for char in (text or "") if char.isalnum())
