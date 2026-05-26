"""Build WFW-native records from human corrections.

Human correction is not an old prose explanation.  It is a first-class WFW
record: definition span, source pieces, mechanisms, answer placements, and DB
gaps are all preserved as data.
"""
from __future__ import annotations

from dataclasses import dataclass

from .wfw_atoms import build_wfw_atom_context
from .wfw_working import place_charade, source_block_from_tokens


@dataclass(frozen=True)
class ManualPiece:
    clue_span: tuple[int, int]
    value: str
    mechanism: str = "synonym"
    label: str | None = None


def build_manual_wfw_correction(clue_text, answer, definition_span, pieces,
                                operation="charade", db=None,
                                source="manual_wfw_correction"):
    """Return a durable WFW correction record from human-selected spans.

    Spans use WFW original-token indices: ``(start, end)`` with ``end``
    exclusive.  For ``Sit across top mount`` the correction is:

    - definition span ``(0, 2)``: ``Sit across``
    - piece ``(2, 3)`` -> ``BEST``
    - piece ``(3, 4)`` -> ``RIDE``
    """
    if operation != "charade":
        raise ValueError("manual correction currently supports charade")
    atom_context = build_wfw_atom_context(clue_text, answer)
    normalised_pieces = tuple(_coerce_piece(p) for p in pieces)
    definition = _span_record(atom_context, "definition", definition_span)

    working_blocks = tuple(
        source_block_from_tokens(
            atom_context,
            "manual_piece_%02d" % idx,
            range(piece.clue_span[0], piece.clue_span[1]),
            piece.value,
            piece.mechanism,
        )
        for idx, piece in enumerate(normalised_pieces)
    )
    placements = place_charade(atom_context.answer_atoms, working_blocks)
    missing = _missing_enrichments(
        atom_context, definition, normalised_pieces, answer, db)

    status = "wfw_manual_proven"
    if missing:
        status = "wfw_manual_proven_with_db_gaps"

    return {
        "schema": "wfw_manual_correction:v1",
        "source": source,
        "status": status,
        "operation": operation,
        "clue_text": clue_text,
        "answer": answer,
        "atom_context": atom_context.as_dict(),
        "definition": definition,
        "working_blocks": [block.as_dict() for block in working_blocks],
        "placements": [placement.as_dict() for placement in placements],
        "missing_enrichments": missing,
    }


def _coerce_piece(piece):
    if isinstance(piece, ManualPiece):
        return piece
    return ManualPiece(
        clue_span=tuple(piece["clue_span"]),
        value=piece["value"],
        mechanism=piece.get("mechanism", "synonym"),
        label=piece.get("label"),
    )


def _span_record(atom_context, role, span):
    start, end = span
    tokens = atom_context.clue_tokens[start:end]
    atom_ids = tuple(atom_id for token in tokens for atom_id in token.atom_ids)
    return {
        "role": role,
        "span": [start, end],
        "text": " ".join(token.text for token in tokens),
        "token_ids": [token.token_id for token in tokens],
        "atom_ids": list(atom_ids),
    }


def _missing_enrichments(atom_context, definition, pieces, answer, db):
    if db is None:
        return []
    missing = []
    definition_text = definition["text"]
    answer_clean = _clean_letters(answer)
    if not db.is_definition_of(definition_text, answer_clean):
        missing.append({
            "type": "definition",
            "definition": definition_text,
            "answer": answer_clean,
        })
    for piece in pieces:
        source_text = _span_record(
            atom_context, "piece", piece.clue_span)["text"]
        value = _clean_letters(piece.value)
        if piece.mechanism == "synonym" and not db.is_definition_of(
                source_text, value):
            missing.append({
                "type": "synonym",
                "word": source_text,
                "synonym": value,
            })
    return missing


def _clean_letters(value):
    return "".join(char.upper() for char in (value or "") if char.isalpha())
