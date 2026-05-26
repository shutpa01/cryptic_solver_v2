"""WFW-native DB gap collection.

This module does not solve, score, or approve clues.  It inspects the WFW
evidence layer and reports reference facts that are missing or only licensed
by a loose reverse lookup.
"""
from __future__ import annotations

from dataclasses import dataclass

from .clue_context import build_clue_context
from .db import RefDB
from .wfw_unified_solver import solve_wfw_unified


@dataclass(frozen=True)
class WFWGap:
    gap_type: str
    word: str
    value: str
    reason: str
    clue_text: str
    answer: str
    span: tuple[int, int] | None = None

    def as_dict(self):
        return {
            "type": self.gap_type,
            "word": self.word,
            "value": self.value,
            "reason": self.reason,
            "clue_text": self.clue_text,
            "answer": self.answer,
            "span": list(self.span) if self.span else None,
        }


def collect_wfw_gaps(clue_text, answer, db=None, wfw_result=None):
    """Return WFW DB gaps for a clue without changing solver state."""
    db = db or RefDB()
    answer_clean = _clean_value(answer)
    result = wfw_result or solve_wfw_unified(clue_text, answer_clean, db=db)
    context = result.clue_context or build_clue_context(
        clue_text, answer_clean, db=db, annotate=True)

    gaps = []
    gaps.extend(_reverse_only_definition_gaps(context, db, answer_clean))
    gaps.extend(_definition_candidate_gaps(result, db, answer_clean))
    gaps.extend(_inferred_definition_gaps(result, db, answer_clean))
    gaps.extend(_surface_role_gaps(result, answer_clean))
    gaps.extend(_whole_answer_shortcut_gaps(result, context, db, answer_clean))
    return _dedupe_gaps(gaps)


def _reverse_only_definition_gaps(context, db, answer):
    gaps = []
    for candidate in context.definition_candidates:
        phrase = candidate.def_phrase
        if _has_direct_fact(db, phrase, answer):
            continue
        if not db.is_definition_of(phrase, answer):
            continue
        gaps.append(WFWGap(
            gap_type="synonym",
            word=phrase,
            value=answer,
            reason="reverse_only_definition_lookup",
            clue_text=context.clue_text,
            answer=answer,
            span=candidate.definition_span.as_tuple(),
        ))
    return gaps


def _whole_answer_shortcut_gaps(result, context, db, answer):
    gaps = []
    for token_parse in _primary_token_parses(result):
        source_blocks = [
            block for block in token_parse.blocks
            if block.kind == "SOURCE_BLOCK"
        ]
        if len(source_blocks) != 1:
            continue
        source = source_blocks[0]
        if _clean_value(source.value) != answer:
            continue
        definition_blocks = [
            block for block in token_parse.blocks
            if block.kind == "DEF_BLOCK"
        ]
        for definition in definition_blocks:
            if _has_direct_fact(db, definition.text, answer):
                continue
            gaps.append(WFWGap(
                gap_type="synonym",
                word=definition.text,
                value=answer,
                reason="whole_answer_shortcut_needs_direct_second_definition",
                clue_text=context.clue_text,
                answer=answer,
                span=definition.span,
            ))
    return gaps


def _inferred_definition_gaps(result, db, answer):
    gaps = []
    for token_parse in _primary_token_parses(result):
        if getattr(token_parse, "confidence", "") != (
                "mechanically_verified_inferred_definition"):
            continue
        for block in token_parse.blocks:
            if block.kind != "DEF_BLOCK":
                continue
            if block.role != "inferred_definition":
                continue
            if _has_direct_fact(db, block.text, answer):
                continue
            gaps.append(WFWGap(
                gap_type="synonym",
                word=block.text,
                value=answer,
                reason="inferred_definition_needs_db_fact",
                clue_text=result.clue_text,
                answer=answer,
                span=block.span,
            ))
    return gaps


def _definition_candidate_gaps(result, db, answer):
    gaps = []
    for token_parse in _primary_token_parses(result):
        if getattr(token_parse, "confidence", "") != (
                "mechanically_verified_definition_gaps"):
            continue
        for block in token_parse.blocks:
            if block.token != "DEF_CANDIDATE":
                continue
            if _has_direct_fact(db, block.text, answer):
                continue
            gaps.append(WFWGap(
                gap_type="synonym",
                word=block.text,
                value=answer,
                reason="definition_candidate_needs_direct_fact",
                clue_text=result.clue_text,
                answer=answer,
                span=block.span,
            ))
    return gaps


def _surface_role_gaps(result, answer):
    gaps = []
    for token_parse in _primary_token_parses(result):
        if getattr(token_parse, "confidence", "") != (
                "mechanically_verified_surface_gaps"):
            continue
        for block in token_parse.blocks:
            if block.token != "SURFACE_GAP":
                continue
            gaps.append(WFWGap(
                gap_type="surface_role",
                word=block.text,
                value="SURFACE",
                reason="surface_role_needs_review",
                clue_text=result.clue_text,
                answer=answer,
                span=block.span,
            ))
    return gaps


def _primary_token_parses(result):
    parses = tuple(result.token_parses or ())
    return parses[:1]


def _has_direct_fact(db, phrase, answer):
    answer_clean = _clean_value(answer)
    phrase_clean = (phrase or "").lower().strip(".,;:!?\"'()-").strip()
    for value in db.get_synonyms(phrase_clean):
        if _clean_value(value) == answer_clean:
            return True
    return False


def _dedupe_gaps(gaps):
    by_key = {}
    order = []
    for gap in gaps:
        key = (
            gap.gap_type,
            gap.word.lower(),
            _clean_value(gap.value),
        )
        if key not in by_key:
            by_key[key] = gap
            order.append(key)
            continue
        previous = by_key[key]
        reasons = tuple(
            reason for reason in (previous.reason, gap.reason)
            if reason
        )
        merged_reason = ";".join(dict.fromkeys(reasons))
        by_key[key] = WFWGap(
            gap_type=previous.gap_type,
            word=previous.word,
            value=previous.value,
            reason=merged_reason,
            clue_text=previous.clue_text,
            answer=previous.answer,
            span=previous.span,
        )
    return tuple(by_key[key] for key in order)


def _clean_value(value):
    return "".join(char.upper() for char in (value or "") if char.isalpha())
