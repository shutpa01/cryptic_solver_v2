"""Bridge obase structured output into WFW proof attempts.

Obase is treated as an evidence proposer.  This module may use obase's stored
pieces as candidate working blocks, but WFW proof is only successful when the
pieces, transformations, and answer placement reconcile mechanically.
"""
from __future__ import annotations

from dataclasses import dataclass
import re

from .wfw_working import (
    WFWTransformation,
    anagram_block,
    container_block,
    controller_atom_ids,
    hidden_block,
    place_hidden,
    place_anagram,
    place_charade,
    place_container,
    reverse_block,
    source_block_from_tokens,
    trim_block,
)


@dataclass(frozen=True)
class ObasePieceProposal:
    proposal_id: str
    clue_word: str
    proposed_value: str
    mechanism: str
    base_value: str
    transform: str | None
    source_token_indices: tuple[int, ...]
    objections: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "proposal_id": self.proposal_id,
            "clue_word": self.clue_word,
            "proposed_value": self.proposed_value,
            "mechanism": self.mechanism,
            "base_value": self.base_value,
            "transform": self.transform,
            "source_token_indices": list(self.source_token_indices),
            "objections": list(self.objections),
        }


@dataclass(frozen=True)
class WFWProofAttempt:
    status: str
    operation: str
    proposals: tuple[ObasePieceProposal, ...]
    blocks: tuple
    transformations: tuple[WFWTransformation, ...]
    placements: tuple
    objections: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "status": self.status,
            "operation": self.operation,
            "proposals": [proposal.as_dict() for proposal in self.proposals],
            "blocks": [block.as_dict() for block in self.blocks],
            "transformations": [
                transform.as_dict() for transform in self.transformations
            ],
            "placements": [
                placement.as_dict() for placement in self.placements
            ],
            "objections": list(self.objections),
        }


def prove_obase_charade(context, components, ai_explanation=""):
    """Try to prove an obase charade through WFW working blocks."""
    components = components or {}
    assembly = components.get("assembly") or {}
    if assembly.get("op") != "charade":
        return WFWProofAttempt(
            status="unsupported",
            operation=assembly.get("op") or "",
            proposals=(),
            blocks=(),
            transformations=(),
            placements=(),
            objections=("only_charade_supported_in_this_slice",),
        )

    pieces = components.get("ai_pieces") or components.get("pieces") or []
    proposals = tuple(
        _proposal_from_piece(context, idx, piece)
        for idx, piece in enumerate(pieces)
    )
    proposal_objections = tuple(
        objection
        for proposal in proposals
        for objection in proposal.objections
    )
    if proposal_objections:
        return WFWProofAttempt(
            status="unproven",
            operation="charade",
            proposals=proposals,
            blocks=(),
            transformations=(),
            placements=(),
            objections=proposal_objections,
        )

    indicators = _indicator_map(context, ai_explanation)
    blocks = []
    transformations = []
    output_blocks = []
    for proposal in proposals:
        base = source_block_from_tokens(
            context,
            "obase_%s_base" % proposal.proposal_id,
            proposal.source_token_indices,
            proposal.base_value,
            proposal.mechanism,
        )
        blocks.append(base)
        out = base
        if proposal.transform == "reversal":
            atoms = indicators.get("reversal", ())
            out, transform = reverse_block(
                base, "obase_%s_reversed" % proposal.proposal_id, atoms)
            blocks.append(out)
            transformations.append(transform)
        elif proposal.transform == "trim_last":
            atoms = indicators.get("trim_last", ())
            out, transform = trim_block(
                base, "obase_%s_trim_last" % proposal.proposal_id,
                "last", atoms)
            blocks.append(out)
            transformations.append(transform)
        elif proposal.transform is not None:
            return WFWProofAttempt(
                status="unsupported",
                operation="charade",
                proposals=proposals,
                blocks=tuple(blocks),
                transformations=tuple(transformations),
                placements=(),
                objections=("unsupported_transform:%s" % proposal.transform,),
            )
        if out.value != proposal.proposed_value:
            return WFWProofAttempt(
                status="rejected",
                operation="charade",
                proposals=proposals,
                blocks=tuple(blocks),
                transformations=tuple(transformations),
                placements=(),
                objections=(
                    "proposal_value_mismatch:%s:%s!=%s" % (
                        proposal.proposal_id, out.value,
                        proposal.proposed_value),
                ),
            )
        output_blocks.append(out)

    try:
        placements = place_charade(context.answer_atoms, tuple(output_blocks))
    except ValueError as exc:
        return WFWProofAttempt(
            status="rejected",
            operation="charade",
            proposals=proposals,
            blocks=tuple(blocks),
            transformations=tuple(transformations),
            placements=(),
            objections=(str(exc),),
        )

    return WFWProofAttempt(
        status="proven",
        operation="charade",
        proposals=proposals,
        blocks=tuple(blocks),
        transformations=tuple(transformations),
        placements=placements,
    )


def prove_obase_container(context, components, ai_explanation=""):
    """Try to prove an obase container through WFW working blocks."""
    components = components or {}
    assembly = components.get("assembly") or {}
    if assembly.get("op") != "container":
        return WFWProofAttempt(
            status="unsupported",
            operation=assembly.get("op") or "",
            proposals=(),
            blocks=(),
            transformations=(),
            placements=(),
            objections=("only_container_supported_here",),
        )

    pieces = components.get("ai_pieces") or components.get("pieces") or []
    proposals = tuple(
        _proposal_from_piece(context, idx, piece)
        for idx, piece in enumerate(pieces)
    )
    if len(proposals) < 2:
        return WFWProofAttempt(
            status="unproven",
            operation="container",
            proposals=proposals,
            blocks=(),
            transformations=(),
            placements=(),
            objections=("container_needs_outer_and_inner",),
        )
    proposal_objections = tuple(
        objection
        for proposal in proposals
        for objection in proposal.objections
    )
    if proposal_objections:
        return WFWProofAttempt(
            status="unproven",
            operation="container",
            proposals=proposals,
            blocks=(),
            transformations=(),
            placements=(),
            objections=proposal_objections,
        )

    indicators = _indicator_map(context, ai_explanation)
    outer = _block_from_proposal(context, proposals[0])
    inner_blocks = tuple(
        _block_from_proposal(context, proposal)
        for proposal in proposals[1:]
    )
    blocks = (outer,) + inner_blocks
    try:
        output, transform, insert_pos = container_block(
            outer,
            inner_blocks,
            "obase_container_output",
            context.answer_text,
            indicators.get("container", ()),
        )
        placements = place_container(
            context.answer_atoms, outer, inner_blocks, insert_pos)
    except ValueError as exc:
        return WFWProofAttempt(
            status="rejected",
            operation="container",
            proposals=proposals,
            blocks=blocks,
            transformations=(),
            placements=(),
            objections=(str(exc),),
        )

    return WFWProofAttempt(
        status="proven",
        operation="container",
        proposals=proposals,
        blocks=blocks + (output,),
        transformations=(transform,),
        placements=placements,
    )


def prove_obase_anagram(context, components, ai_explanation=""):
    """Try to prove an obase anagram through WFW fodder mapping."""
    components = components or {}
    assembly = components.get("assembly") or {}
    if assembly.get("op") != "anagram":
        return WFWProofAttempt(
            status="unsupported",
            operation=assembly.get("op") or "",
            proposals=(),
            blocks=(),
            transformations=(),
            placements=(),
            objections=("only_anagram_supported_here",),
        )
    pieces = components.get("ai_pieces") or components.get("pieces") or []
    proposals = tuple(
        _proposal_from_piece(context, idx, piece)
        for idx, piece in enumerate(pieces)
    )
    proposal_objections = tuple(
        objection
        for proposal in proposals
        for objection in proposal.objections
    )
    if proposal_objections:
        return WFWProofAttempt(
            status="unproven",
            operation="anagram",
            proposals=proposals,
            blocks=(),
            transformations=(),
            placements=(),
            objections=proposal_objections,
        )
    fodder_blocks = tuple(
        _block_from_proposal(context, proposal)
        for proposal in proposals
    )
    indicators = _indicator_map(context, ai_explanation)
    try:
        output, transform = anagram_block(
            fodder_blocks,
            "obase_anagram_output",
            context.answer_text,
            indicators.get("anagram", ()),
        )
        placements = place_anagram(context.answer_atoms, fodder_blocks)
    except ValueError as exc:
        return WFWProofAttempt(
            status="rejected",
            operation="anagram",
            proposals=proposals,
            blocks=fodder_blocks,
            transformations=(),
            placements=(),
            objections=(str(exc),),
        )
    return WFWProofAttempt(
        status="proven",
        operation="anagram",
        proposals=proposals,
        blocks=fodder_blocks + (output,),
        transformations=(transform,),
        placements=placements,
    )


def prove_obase_hidden(context, components, ai_explanation=""):
    """Try to prove a hidden answer directly inside clue source words."""
    components = components or {}
    assembly = components.get("assembly") or {}
    if assembly.get("op") != "hidden":
        return WFWProofAttempt(
            status="unsupported",
            operation=assembly.get("op") or "",
            proposals=(),
            blocks=(),
            transformations=(),
            placements=(),
            objections=("only_hidden_supported_here",),
        )
    pieces = components.get("ai_pieces") or components.get("pieces") or []
    if not pieces:
        return WFWProofAttempt(
            status="unproven",
            operation="hidden",
            proposals=(),
            blocks=(),
            transformations=(),
            placements=(),
            objections=("hidden_needs_source_piece",),
        )
    proposal = _proposal_from_piece(context, 0, pieces[0])
    if proposal.objections:
        return WFWProofAttempt(
            status="unproven",
            operation="hidden",
            proposals=(proposal,),
            blocks=(),
            transformations=(),
            placements=(),
            objections=proposal.objections,
        )
    source = source_block_from_tokens(
        context,
        "obase_%s_base" % proposal.proposal_id,
        proposal.source_token_indices,
        _clean_source_tokens(context, proposal.source_token_indices),
        proposal.mechanism,
    )
    indicators = _indicator_map(context, ai_explanation)
    try:
        output, transform, start, reversed_hidden = hidden_block(
            source,
            "obase_%s_hidden" % proposal.proposal_id,
            context.answer_text,
            indicators.get("hidden", ()),
        )
        placements = place_hidden(
            context.answer_atoms, source, start, reversed_hidden)
    except ValueError as exc:
        return WFWProofAttempt(
            status="rejected",
            operation="hidden",
            proposals=(proposal,),
            blocks=(source,),
            transformations=(),
            placements=(),
            objections=(str(exc),),
        )
    return WFWProofAttempt(
        status="proven",
        operation="hidden",
        proposals=(proposal,),
        blocks=(source, output),
        transformations=(transform,),
        placements=placements,
    )


def _proposal_from_piece(context, idx, piece):
    clue_word = str(piece.get("clue_word") or piece.get("source") or "")
    proposed_value = _clean_letters(piece.get("letters") or piece.get("value"))
    mechanism = str(piece.get("mechanism") or piece.get("type") or "")
    base_value, transform = _parse_mechanism(mechanism, proposed_value)
    token_indices = _find_token_indices(context, clue_word)
    objections = []
    if not clue_word:
        objections.append("missing_clue_word:%d" % idx)
    if not proposed_value:
        objections.append("missing_value:%d" % idx)
    if not token_indices:
        objections.append("source_not_found:%s" % clue_word)
    return ObasePieceProposal(
        proposal_id="piece_%d" % idx,
        clue_word=clue_word,
        proposed_value=proposed_value,
        mechanism=mechanism,
        base_value=base_value,
        transform=transform,
        source_token_indices=tuple(token_indices),
        objections=tuple(objections),
    )


def _block_from_proposal(context, proposal):
    return source_block_from_tokens(
        context,
        "obase_%s_base" % proposal.proposal_id,
        proposal.source_token_indices,
        proposal.base_value,
        proposal.mechanism,
    )


def _parse_mechanism(mechanism, proposed_value):
    parts = mechanism.split(":")
    if len(parts) >= 2 and parts[0] == "reversal":
        return _clean_letters(parts[1]), "reversal"
    if len(parts) >= 3 and parts[0] == "deletion":
        detail = parts[2].lower()
        if "last" in detail or "end" in detail:
            return _clean_letters(parts[1]), "trim_last"
    return proposed_value, None


def _clean_source_tokens(context, token_indices):
    return _clean_letters("".join(
        context.clue_tokens[idx].text
        for idx in token_indices
    ))


def _find_token_indices(context, phrase):
    phrase_tokens = [
        _token_key(token.text)
        for token in _phrase_tokens(phrase)
        if _token_key(token.text)
    ]
    if not phrase_tokens:
        return ()
    clue_keys = [_token_key(token.text) for token in context.clue_tokens]
    n = len(phrase_tokens)
    for start in range(0, len(clue_keys) - n + 1):
        if clue_keys[start:start + n] == phrase_tokens:
            return tuple(range(start, start + n))
    # Try ignoring punctuation in the proposal, e.g. "Bordeaux," -> Bordeaux.
    phrase_words = [key for key in phrase_tokens if re.search(r"[A-Z0-9]", key)]
    if phrase_words != phrase_tokens:
        n = len(phrase_words)
        for start in range(0, len(clue_keys) - n + 1):
            if clue_keys[start:start + n] == phrase_words:
                return tuple(range(start, start + n))
    return ()


def _phrase_tokens(text):
    from .wfw_atoms import build_wfw_atom_context

    return build_wfw_atom_context(text or "", "").clue_tokens


def _indicator_map(context, ai_explanation):
    indicators = {}
    for kind, quoted in re.findall(r"\[(\w+):\s*\"([^\"]+)\"\]",
                                   ai_explanation or ""):
        indices = _find_token_indices(context, quoted)
        if indices:
            indicators[kind] = controller_atom_ids(context, indices)

    for idx, token in enumerate(context.clue_tokens):
        key = _token_key(token.text)
        if key in {"NEARLY", "ALMOST", "SHORT", "SHORTLY"}:
            indicators.setdefault("trim_last",
                                  controller_atom_ids(context, (idx,)))
        if key in {"RESTRICTS", "CONCEALS", "CONCEALING", "HIDES", "HIDING"}:
            indicators.setdefault("hidden", controller_atom_ids(context, (idx,)))
    return indicators


def _token_key(text):
    return "".join(char.upper() for char in (text or "") if char.isalnum())


def _clean_letters(value):
    return "".join(char.upper() for char in (value or "") if char.isalpha())
