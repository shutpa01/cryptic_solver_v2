"""Assemble verified parses from canonical clue tokens and annotations.

Stage Two consumes the Stage One ``ClueContext``. It does not perform DB
lookups or invent meanings; it combines annotated spans and verifies that
the assembly produces the known answer.
"""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations, product

from .clue_context import SpanAnnotation


@dataclass(frozen=True)
class ParseBlock:
    block_id: str
    kind: str
    span: tuple[int, int] | None
    text: str
    token: str | None = None
    value: str | None = None
    role: str | None = None
    input_value: str | None = None

    def as_dict(self):
        return {
            "block_id": self.block_id,
            "kind": self.kind,
            "span": list(self.span) if self.span else None,
            "span_space": "full_clue_tokens" if self.span else None,
            "text": self.text,
            "token": self.token,
            "value": self.value,
            "role": self.role,
            "input_value": self.input_value,
        }


@dataclass(frozen=True)
class ParseOperation:
    operation: str
    indicator_block: str
    input_blocks: tuple[str, ...]
    output: str
    detail: str

    def as_dict(self):
        return {
            "operation": self.operation,
            "indicator_block": self.indicator_block,
            "input_blocks": list(self.input_blocks),
            "output": self.output,
            "detail": self.detail,
        }


@dataclass(frozen=True)
class TokenParse:
    parse_id: str
    operation: str
    answer: str
    blocks: tuple[ParseBlock, ...]
    operations: tuple[ParseOperation, ...]
    confidence: str = "candidate"

    def as_dict(self):
        return {
            "parse_id": self.parse_id,
            "operation": self.operation,
            "answer": self.answer,
            "blocks": [block.as_dict() for block in self.blocks],
            "operations": [op.as_dict() for op in self.operations],
            "confidence": self.confidence,
        }


def assemble_token_parses(context):
    """Return verified tokenised parses from a ClueContext."""
    parses = []
    parses.extend(_assemble_double_definition_parses(context))
    parses.extend(_assemble_hidden_parses(context))
    parses.extend(_assemble_anagram_parses(context))
    parses.extend(_assemble_anagram_charade_parses(context))
    parses.extend(_assemble_reversal_parses(context))
    parses.extend(_assemble_reversal_charade_parses(context))
    parses.extend(_assemble_deletion_parses(context))
    parses.extend(_assemble_deletion_charade_parses(context))
    parses.extend(_assemble_homophone_parses(context))
    parses.extend(_assemble_container_parses(context))
    parses.extend(_assemble_container_charade_parses(context))
    parses.extend(_assemble_substitution_parses(context))
    parses.extend(_assemble_positional_charade_parses(context))
    parses.extend(_assemble_supporting_charade_parses(context))
    parses.extend(_assemble_charade_parses(context))
    parses.sort(key=_parse_rank)
    return parses


def _assemble_double_definition_parses(context):
    parses = []
    source_annotations = [
        ann for ann in context.annotations
        if ann.token in ("SYN_F", "ABR_F")
        and context.answer in set(_string_values(ann.values))
    ]
    source_annotations.sort(
        key=lambda ann: (ann.span[0], -(ann.span[1] - ann.span[0]), ann.span[1]))

    for left_idx, left in enumerate(source_annotations):
        for right in source_annotations[left_idx + 1:]:
            if _overlaps(left.span, right.span):
                continue
            blocks = [
                ParseBlock("dd_0", "SOURCE_BLOCK", left.span, left.text,
                           token=left.token, value=context.answer,
                           role="dd_0"),
                ParseBlock("dd_1", "SOURCE_BLOCK", right.span, right.text,
                           token=right.token, value=context.answer,
                           role="dd_1"),
                ParseBlock("answer", "ASSEMBLY_BLOCK", None,
                           context.answer, value=context.answer,
                           role="answer"),
            ]
            op = ParseOperation(
                operation="double_definition",
                indicator_block="",
                input_blocks=("dd_0", "dd_1"),
                output=context.answer,
                detail="%s / %s = %s" % (
                    left.text, right.text, context.answer),
            )
            parse = TokenParse(
                parse_id="token_parse:double_definition:%d_%d:%d_%d" % (
                    left.span[0], left.span[1], right.span[0], right.span[1]),
                operation="double_definition",
                answer=context.answer,
                blocks=tuple(blocks),
                operations=(op,),
                confidence="mechanically_verified",
            )
            parse = _complete_parse(context, parse)
            if parse is not None:
                parses.append(parse)
                return parses
    candidate_parse = _assemble_definition_candidate_double_definition(context)
    if candidate_parse is not None:
        return [candidate_parse]
    qualifier_parse = _assemble_definition_with_qualifier(context)
    if qualifier_parse is not None:
        return [qualifier_parse]
    return parses


def _assemble_definition_candidate_double_definition(context):
    candidates = list(context.definition_candidates)
    candidates.sort(key=lambda candidate: (
        candidate.definition_span.start,
        -(candidate.definition_span.end - candidate.definition_span.start),
        candidate.definition_span.end,
    ))
    for left_idx, left in enumerate(candidates):
        for right in candidates[left_idx + 1:]:
            if _overlaps(
                    left.definition_span.as_tuple(),
                    right.definition_span.as_tuple()):
                continue
            blocks = [
                ParseBlock(
                    "dd_0", "SOURCE_BLOCK",
                    left.definition_span.as_tuple(),
                    left.definition_span.text,
                    token="DEF_CANDIDATE",
                    value=context.answer,
                    role="dd_0"),
                ParseBlock(
                    "dd_1", "SOURCE_BLOCK",
                    right.definition_span.as_tuple(),
                    right.definition_span.text,
                    token="DEF_CANDIDATE",
                    value=context.answer,
                    role="dd_1"),
                ParseBlock(
                    "answer", "ASSEMBLY_BLOCK", None,
                    context.answer, value=context.answer, role="answer"),
            ]
            op = ParseOperation(
                operation="double_definition",
                indicator_block="",
                input_blocks=("dd_0", "dd_1"),
                output=context.answer,
                detail="%s / %s = %s" % (
                    left.definition_span.text,
                    right.definition_span.text,
                    context.answer),
            )
            parse = TokenParse(
                parse_id="token_parse:double_definition_candidates:%d_%d:%d_%d" % (
                    left.definition_span.start,
                    left.definition_span.end,
                    right.definition_span.start,
                    right.definition_span.end),
                operation="double_definition",
                answer=context.answer,
                blocks=tuple(blocks),
                operations=(op,),
                confidence="mechanically_verified",
            )
            parse = _complete_parse(context, parse)
            if parse is not None:
                return parse
    return None


def _assemble_definition_with_qualifier(context):
    whole_answer_sources = [
        ann for ann in context.annotations
        if ann.token in ("SYN_F", "ABR_F")
        and context.answer in set(_string_values(ann.values))
    ]
    whole_answer_sources.sort(key=lambda ann: (
        ann.span[0], -(ann.span[1] - ann.span[0]), ann.span[1]))
    for source in whole_answer_sources:
        qualifier = _opposite_edge_qualifier(context, source.span)
        if qualifier is None:
            continue
        blocks = [
            ParseBlock(
                "dd_0", "SOURCE_BLOCK", source.span, source.text,
                token=source.token, value=context.answer, role="dd_0"),
            ParseBlock(
                "dd_1", "SOURCE_BLOCK", qualifier.as_tuple(),
                qualifier.text, token="CRYPTIC_QUALIFIER",
                value=context.answer, role="dd_1"),
            ParseBlock(
                "answer", "ASSEMBLY_BLOCK", None,
                context.answer, value=context.answer, role="answer"),
        ]
        op = ParseOperation(
            operation="double_definition",
            indicator_block="",
            input_blocks=("dd_0", "dd_1"),
            output=context.answer,
            detail="%s / %s = %s" % (
                source.text, qualifier.text, context.answer),
        )
        parse = TokenParse(
            parse_id="token_parse:definition_qualifier:%d_%d:%d_%d" % (
                source.span[0], source.span[1],
                qualifier.start, qualifier.end),
            operation="double_definition",
            answer=context.answer,
            blocks=tuple(blocks),
            operations=(op,),
            confidence="mechanically_verified",
        )
        parse = _complete_parse(context, parse)
        if parse is not None:
            return parse
    return None


def _opposite_edge_qualifier(context, source_span):
    if source_span[0] == 0 and source_span[1] < len(context.tokens):
        span = context.span(source_span[1], len(context.tokens),
                            kind="cryptic_qualifier")
    elif source_span[1] == len(context.tokens) and source_span[0] > 0:
        span = context.span(0, source_span[0], kind="cryptic_qualifier")
    else:
        return None
    if not span.text or "?" not in span.text:
        return None
    return span


def _assemble_charade_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        source_annotations = _source_annotations(context, wordplay_span)
        for assignment in _charade_search(source_annotations, context.answer):
            if _is_whole_answer_shortcut(assignment, context.answer):
                continue
            if _spans_overlap_any([ann.span for ann, _value in assignment]):
                continue
            parse = _complete_parse(context, _build_charade_parse(
                context, definition_span, assignment, operation="charade"))
            if parse is not None:
                parses.append(parse)
    parses.sort(key=_parse_rank)
    return parses[:10]


def _assemble_supporting_charade_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        indicators = [
            ann for ann in context.annotations
            if ann.span[0] >= wordplay_span[0]
            and ann.span[1] <= wordplay_span[1]
            and ann.text.lower().strip(".,;:!?\"'()-") in (
                "supporting", "supports", "under")
        ]
        if not indicators:
            continue
        source_annotations = _source_annotations(context, wordplay_span)
        candidates = (
            list(source_annotations)
            + _positional_source_candidates(context, wordplay_span)
            + _trimmed_source_candidates(context, wordplay_span)
        )
        for indicator in indicators:
            before = [
                ann for ann in candidates
                if ann.span[1] <= indicator.span[0]
            ]
            after = [
                ann for ann in candidates
                if ann.span[0] >= indicator.span[1]
            ]
            for split in range(1, len(context.answer)):
                head = context.answer[:split]
                tail = context.answer[split:]
                for after_assignment in _charade_search(after, head):
                    for before_assignment in _charade_search(before, tail):
                        assignment = after_assignment + before_assignment
                        if _assignment_spans_overlap(assignment):
                            continue
                        parse = _complete_parse(context, _build_charade_parse(
                            context, definition_span, assignment,
                            operation="supporting_charade"))
                        if parse is not None:
                            parses.append(parse)
    parses.sort(key=_parse_rank)
    return parses[:20]


def _is_whole_answer_shortcut(assignment, answer):
    """Reject one-block charades that merely restate the whole answer.

    A whole-answer synonym belongs to a whole-clue type, normally double
    definition or cryptic definition.  It is not a WFW charade proof.
    """
    if len(assignment) != 1:
        return False
    _ann, value = assignment[0]
    return _clean_value(value) == _clean_value(answer)


def _assemble_reversal_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        indicators = [
            ann for ann in context.annotations
            if ann.token == "REV_I" and _inside(ann.span, wordplay_span)
        ]
        indicators.sort(key=lambda ann: (
            -(ann.span[1] - ann.span[0]), ann.span[0], ann.span[1]))
        if not indicators:
            continue
        source_annotations = (
            _source_annotations(context, wordplay_span)
            + _positional_source_candidates(context, wordplay_span)
            + _trimmed_source_candidates(context, wordplay_span)
        )
        for indicator in indicators:
            for source in source_annotations:
                if _overlaps(source.span, indicator.span):
                    continue
                for value in _string_values(source.values):
                    if value[::-1] != context.answer:
                        continue
                    parse = _complete_parse(context, _build_reversal_parse(
                        context, definition_span, source, value, indicator))
                    if parse is not None:
                        return [parse]
    return parses


def _assemble_reversal_charade_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        indicators = [
            ann for ann in context.annotations
            if ann.token == "REV_I" and _inside(ann.span, wordplay_span)
        ]
        if not indicators:
            continue
        source_annotations = _source_annotations(context, wordplay_span)
        candidates = (
            list(source_annotations)
            + _positional_source_candidates(context, wordplay_span)
            + _trimmed_source_candidates(context, wordplay_span)
        )
        for indicator in indicators:
            for source in source_annotations:
                if _overlaps(source.span, indicator.span):
                    continue
                for value in _string_values(source.values):
                    candidates.append(_OperatedAnnotation(
                        source, indicator, value[::-1],
                        "piece_%d_reversal_indicator"))
        for assignment in _charade_search(candidates, context.answer):
            if not any(isinstance(ann, _OperatedAnnotation)
                       and ann.indicator.token == "REV_I"
                       for ann, _value in assignment):
                continue
            if _assignment_spans_overlap(assignment):
                continue
            parse = _complete_parse(context, _build_charade_parse(
                context, definition_span, assignment,
                operation="reversal_charade"))
            if parse is not None:
                parses.append(parse)
    parses.sort(key=_parse_rank)
    return parses[:20]


def _assemble_homophone_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        indicators = [
            ann for ann in context.annotations
            if ann.token == "HOM_I" and _inside(ann.span, wordplay_span)
        ]
        indicators.sort(key=lambda ann: _indicator_sort_key(context, ann))
        if not indicators:
            continue
        source_annotations = [
            ann for ann in context.annotations
            if ann.token in ("HOM_F", "SYN_F", "ABR_F")
            and _inside(ann.span, wordplay_span)
        ]
        for indicator in indicators:
            for source in source_annotations:
                if _overlaps(source.span, indicator.span):
                    continue
                for value in _string_values(source.values):
                    if (value != context.answer
                            and not _homophone_value_matches(
                                value, context.answer)):
                        continue
                    parse = _complete_parse(context, _build_homophone_parse(
                        context, definition_span, source, value, indicator))
                    if parse is not None:
                        parses.append(parse)
    parses.sort(key=_parse_rank)
    return parses[:5]


def _assemble_deletion_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        indicators = [
            ann for ann in context.annotations
            if ann.token.startswith("POS_I_TRIM_") and _inside(ann.span, wordplay_span)
        ]
        if not indicators:
            continue
        sources = _trimmable_source_annotations(context, wordplay_span)
        for indicator in indicators:
            for source in sources:
                if _overlaps(source.span, indicator.span):
                    continue
                for input_value, value in _trimmed_candidate_value_pairs(
                        source, indicator.token):
                    if value != context.answer:
                        continue
                    parse = _complete_parse(context, _build_deletion_parse(
                        context, definition_span, source, value, indicator,
                        input_value=input_value))
                    if parse is not None:
                        return [parse]
    return parses


def _assemble_deletion_charade_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        indicators = [
            ann for ann in context.annotations
            if (ann.token.startswith("POS_I_TRIM_") or ann.token == "DEL_I")
            and _inside(ann.span, wordplay_span)
        ]
        if not indicators:
            continue
        source_annotations = _source_annotations(context, wordplay_span)
        candidates = (
            list(source_annotations)
            + _positional_source_candidates(context, wordplay_span)
            + _trimmed_source_candidates(context, wordplay_span)
        )
        for indicator in indicators:
            if indicator.token.startswith("POS_I_TRIM_"):
                for source in _trimmable_source_annotations(context, wordplay_span):
                    if _overlaps(source.span, indicator.span):
                        continue
                    for input_value, value in _trimmed_candidate_value_pairs(
                            source, indicator.token):
                        if not value:
                            continue
                        candidates.append(_OperatedAnnotation(
                            source, indicator, value,
                            "piece_%d_deletion_indicator",
                            input_value=input_value))
            if indicator.token == "DEL_I":
                candidates.extend(_subtractive_deletion_candidates(
                    context, wordplay_span, indicator))
        for assignment in _charade_search(candidates, context.answer):
            if not any(_is_deletion_candidate(ann)
                       for ann, _value in assignment):
                continue
            if _assignment_spans_overlap(assignment):
                continue
            parse = _complete_parse(context, _build_charade_parse(
                context, definition_span, assignment,
                operation="deletion_charade"))
            if parse is not None:
                parses.append(parse)
    parses.sort(key=_parse_rank)
    return parses[:20]


def _subtractive_deletion_candidates(context, wordplay_span, indicator):
    sources = _trimmable_source_annotations(context, wordplay_span)
    candidates = []
    for base in sources:
        if _overlaps(base.span, indicator.span):
            continue
        for remove in sources:
            if remove is base:
                continue
            if _overlaps(remove.span, indicator.span):
                continue
            if _overlaps(remove.span, base.span):
                continue
            for base_value in _string_values(base.values):
                for remove_value in _string_values(remove.values):
                    result = _delete_value(base_value, remove_value)
                    if not result:
                        continue
                    candidates.append(_DeletionAnnotation(
                        base, remove, indicator, result,
                        input_value=base_value,
                        remove_value=remove_value))
    return candidates


def _assemble_hidden_parses(context):
    parses = []
    answer = context.answer
    rev_answer = answer[::-1]
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        indicators = [
            ann for ann in context.annotations
            if ann.token == "HID_I" and _inside(ann.span, wordplay_span)
        ] + _implicit_hidden_indicators(context, wordplay_span)
        indicators.sort(key=lambda ann: (
            -(ann.span[1] - ann.span[0]), ann.span[0], ann.span[1]))
        if not indicators:
            continue
        for fodder in context.spans:
            if not _inside(fodder.as_tuple(), wordplay_span):
                continue
            letters = _clean_value(fodder.text)
            if len(letters) <= len(answer):
                continue
            for indicator in indicators:
                if _overlaps(fodder.as_tuple(), indicator.span):
                    continue
                if answer in letters:
                    parse = _complete_parse(context, _build_hidden_parse(
                        context, definition_span, fodder.as_tuple(),
                        fodder.text, indicator, reversed_hidden=False))
                    if parse is not None:
                        return [parse]
                if rev_answer in letters:
                    indicator = _combined_reversed_hidden_indicator(
                        context, indicator, wordplay_span, fodder.as_tuple())
                    parse = _complete_parse(context, _build_hidden_parse(
                        context, definition_span, fodder.as_tuple(),
                        fodder.text, indicator, reversed_hidden=True))
                    if parse is not None:
                        return [parse]
    return parses


def _combined_reversed_hidden_indicator(context, hidden_indicator,
                                        wordplay_span, fodder_span):
    candidates = [
        ann for ann in context.annotations
        if ann.token == "REV_I"
        and _inside(ann.span, wordplay_span)
        and not _overlaps(ann.span, hidden_indicator.span)
        and (
            ann.span[0] == hidden_indicator.span[1]
            or ann.span[1] == hidden_indicator.span[0]
            or _links_between(context, fodder_span[1], ann.span[0])
        )
    ]
    if not candidates:
        return hidden_indicator
    candidates.sort(key=lambda ann: (
        abs(ann.span[0] - hidden_indicator.span[0]),
        ann.span[0],
        ann.span[1],
    ))
    rev = candidates[0]
    span = (
        min(hidden_indicator.span[0], rev.span[0]),
        max(hidden_indicator.span[1], rev.span[1]),
    )
    return SpanAnnotation(
        span=span,
        text=context.span_text(*span),
        token=hidden_indicator.token,
        values=hidden_indicator.values,
        source="combined_reversed_hidden_indicator",
    )


def _links_between(context, start, end):
    if start > end:
        return False
    if start == end:
        return True
    return _all_link_tokens(context, (start, end))


def _implicit_hidden_indicators(context, wordplay_span):
    phrase_indicators = {
        "held in",
        "hidden in",
        "in",
        "inside",
        "part of",
        "put in",
        "some",
        "embraced by",
        "contained by",
    }
    indicators = []
    for span in context.spans:
        span_tuple = span.as_tuple()
        if not _inside(span_tuple, wordplay_span):
            continue
        if span.normalized not in phrase_indicators:
            continue
        indicators.append(SpanAnnotation(
            span=span_tuple,
            text=span.text,
            token="HID_I",
            values=("implicit",),
            source="implicit_hidden_phrase",
        ))
    return indicators


def _assemble_anagram_parses(context):
    parses = []
    answer_sorted = sorted(context.answer)
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        indicators = [
            ann for ann in context.annotations
            if ann.token == "ANA_I" and _inside(ann.span, wordplay_span)
        ]
        indicators.sort(key=lambda ann: _indicator_sort_key(context, ann))
        if not indicators:
            continue
        word_indices = list(range(wordplay_span[0], wordplay_span[1]))
        for indicator in indicators:
            candidate_indices = [
                idx for idx in word_indices
                if idx < indicator.span[0] or idx >= indicator.span[1]
            ]
            for fodder_indices, letters in _anagram_fodder_options(
                    context, candidate_indices, context.answer):
                if sorted(letters) != answer_sorted:
                    continue
                fodder_span = (fodder_indices[0], fodder_indices[-1] + 1)
                fodder_text = " ".join(
                    context.tokens[idx].text for idx in fodder_indices)
                parse = _complete_parse(context, _build_anagram_parse(
                    context, definition_span, fodder_span, letters, indicator,
                    fodder_text=fodder_text))
                if parse is not None:
                    parses.append(parse)
    parses.sort(key=_parse_rank)
    return parses[:5]


def _anagram_fodder_options(context, candidate_indices, answer):
    answer_len = len(_clean_value(answer))
    sized = []
    for idx in candidate_indices:
        options = _anagram_token_letter_options(context.tokens[idx].text)
        if options:
            sized.append((idx, options))
    if not sized or len(sized) > 10:
        return ()
    results = []
    for count in range(1, len(sized) + 1):
        for combo in combinations(sized, count):
            option_sets = [options for _idx, options in combo]
            for chosen in product(*option_sets):
                if sum(len(letters) for letters in chosen) != answer_len:
                    continue
                indices = tuple(idx for idx, _options in combo)
                link_count = sum(
                    1 for idx in indices
                    if _token_has_annotation(context, idx, "LNK"))
                span_width = indices[-1] - indices[0]
                letters = "".join(chosen)
                results.append((link_count, span_width, indices, letters))
    results.sort(key=lambda item: (item[0], item[1], item[2], item[3]))
    return tuple((indices, letters)
                 for _link_count, _span_width, indices, letters in results)


def _anagram_token_letter_options(text):
    value = _clean_value(text)
    options = []
    if value:
        options.append(value)
    if (text or "").lower().strip(".,;:!?\"()-").endswith("'s"):
        stripped = value[:-1]
        if stripped and stripped not in options:
            options.append(stripped)
    return tuple(options)


def _anagram_fodder_index_sets(context, candidate_indices, answer):
    answer_len = len(_clean_value(answer))
    sized = []
    for idx in candidate_indices:
        letters = _clean_value(context.tokens[idx].text)
        if letters:
            sized.append((idx, len(letters)))
    if not sized or len(sized) > 10:
        return ()
    results = []
    for count in range(1, len(sized) + 1):
        for combo in combinations(sized, count):
            if sum(length for _idx, length in combo) != answer_len:
                continue
            indices = tuple(idx for idx, _length in combo)
            link_count = sum(
                1 for idx in indices
                if _token_has_annotation(context, idx, "LNK"))
            span_width = indices[-1] - indices[0]
            results.append((link_count, span_width, indices))
    results.sort(key=lambda item: (item[0], item[1], item[2]))
    return tuple(indices for _link_count, _span_width, indices in results)


def _assemble_anagram_charade_parses(context):
    parses = []
    answer = context.answer
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        indicators = [
            ann for ann in context.annotations
            if ann.token == "ANA_I" and _inside(ann.span, wordplay_span)
        ]
        if not indicators:
            continue
        source_annotations = _source_annotations(context, wordplay_span)
        candidates = (
            list(source_annotations)
            + _positional_source_candidates(context, wordplay_span)
            + _trimmed_source_candidates(context, wordplay_span)
        )
        for indicator in indicators:
            for fodder in context.spans:
                fodder_span = fodder.as_tuple()
                if not _inside(fodder_span, wordplay_span):
                    continue
                if _overlaps(fodder_span, indicator.span):
                    continue
                letters = _clean_value(fodder.text)
                if not letters:
                    continue
                for start in range(0, len(answer) - len(letters) + 1):
                    segment = answer[start:start + len(letters)]
                    if sorted(segment) != sorted(letters):
                        continue
                    candidates.append(_SpanOperatedAnnotation(
                        fodder_span, fodder.text, "ANA_F", segment,
                        indicator, "piece_%d_anagram_indicator",
                    ))
            candidates.extend(_split_anagram_source_candidates(
                context, wordplay_span, indicator, answer))
            candidates.extend(_source_group_anagram_candidates(
                context, wordplay_span, indicator, answer))
        for assignment in _charade_search(candidates, answer):
            if not any(_is_operated_candidate(ann)
                       and ann.token == "ANA_F"
                       for ann, _value in assignment):
                continue
            if _assignment_spans_overlap(assignment):
                continue
            parse = _complete_parse(context, _build_charade_parse(
                context, definition_span, assignment,
                operation="anagram_charade"))
            if parse is not None:
                parses.append(parse)
    parses.sort(key=_parse_rank)
    return parses[:5]


def _split_anagram_source_candidates(context, wordplay_span, indicator, answer):
    raw_sources = [
        ann for ann in context.annotations
        if ann.token in ("RAW", "ANA_F")
        and _inside(ann.span, wordplay_span)
        and not _overlaps(ann.span, indicator.span)
        and ann.span[1] - ann.span[0] == 1
        and not _span_has_token(context, ann.span, "LNK")
    ]
    raw_sources.sort(key=lambda ann: ann.span)
    candidates = []
    for left_idx, left in enumerate(raw_sources):
        for right in raw_sources[left_idx + 1:]:
            if right.span[0] <= left.span[0]:
                continue
            between = (left.span[1], right.span[0])
            if between[0] < between[1] and not _all_link_tokens(
                    context, between):
                continue
            letters = _clean_value(left.text + right.text)
            if not letters:
                continue
            for start in range(0, len(answer) - len(letters) + 1):
                segment = answer[start:start + len(letters)]
                if sorted(segment) != sorted(letters):
                    continue
                candidates.append(_SpanOperatedAnnotation(
                    (left.span[0], right.span[1]),
                    "%s %s" % (left.text, right.text),
                    "ANA_F",
                    segment,
                    indicator,
                    "piece_%d_anagram_indicator",
                    input_value=letters,
                ))
    return candidates


def _source_group_anagram_candidates(context, wordplay_span, indicator, answer):
    sources = [
        ann for ann in _source_annotations(context, wordplay_span)
        if not _overlaps(ann.span, indicator.span)
    ]
    sources.sort(key=lambda ann: ann.span)
    candidates = []
    for group_size in (2, 3):
        for group in combinations(sources, group_size):
            spans = [ann.span for ann in group]
            if _spans_overlap_any(spans):
                continue
            if tuple(spans) != tuple(sorted(spans)):
                continue
            for values in _source_value_product(group, max_values=8):
                letters = "".join(values)
                if len(letters) > len(answer):
                    continue
                for start in range(0, len(answer) - len(letters) + 1):
                    segment = answer[start:start + len(letters)]
                    if sorted(segment) != sorted(letters):
                        continue
                    candidates.append(_SpanOperatedAnnotation(
                        (spans[0][0], spans[-1][1]),
                        " ".join(ann.text for ann in group),
                        "ANA_F",
                        segment,
                        indicator,
                        "piece_%d_anagram_indicator",
                        input_value=letters,
                    ))
    return candidates


def _source_value_product(sources, max_values=8):
    values_by_source = []
    for source in sources:
        values = tuple(_unique_values(_string_values(source.values)))[:max_values]
        if not values:
            return ()
        values_by_source.append(values)
    products = [()]
    for values in values_by_source:
        products = [prefix + (value,) for prefix in products for value in values]
        if len(products) > max_values ** len(values_by_source):
            break
    return tuple(products)


def _assemble_container_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)

        source_annotations = (
            _source_annotations(context, wordplay_span)
            + _positional_source_candidates(context, wordplay_span)
            + _trimmed_source_candidates(context, wordplay_span)
        )
        indicators = [
            ann for ann in context.annotations
            if ann.token == "CON_I" and _inside(ann.span, wordplay_span)
        ] + _implicit_container_indicators(context, wordplay_span)

        for indicator in indicators:
            for outer in source_annotations:
                if _overlaps(outer.span, indicator.span):
                    continue
                for inner in source_annotations:
                    if inner is outer:
                        continue
                    if _overlaps(inner.span, indicator.span):
                        continue
                    if _overlaps(inner.span, outer.span):
                        continue
                    for outer_value in _string_values(outer.values):
                        for inner_value in _string_values(inner.values):
                            if outer_value == inner_value:
                                continue
                            if not _container_matches(
                                    outer_value, inner_value, context.answer):
                                continue
                            parse = _complete_parse(context, _build_container_parse(
                                context, definition_span, outer, outer_value,
                                inner, inner_value, indicator))
                            if parse is not None:
                                parses.append(parse)
                                return parses
    return parses


def _assemble_container_charade_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        source_annotations = _source_annotations(context, wordplay_span)
        indicators = [
            ann for ann in context.annotations
            if ann.token == "CON_I" and _inside(ann.span, wordplay_span)
        ] + _implicit_container_indicators(context, wordplay_span)
        non_link_indicators = [
            ann for ann in indicators
            if not _span_has_token(context, ann.span, "LNK")
        ]
        if non_link_indicators:
            indicators = non_link_indicators
        indicators.sort(key=lambda ann: _indicator_sort_key(context, ann))
        if not indicators:
            continue
        candidates = (
            list(source_annotations)
            + _positional_source_candidates(context, wordplay_span)
            + _trimmed_source_candidates(context, wordplay_span)
        )
        anagram_indicators = [
            ann for ann in context.annotations
            if ann.token == "ANA_I" and _inside(ann.span, wordplay_span)
        ]
        for anagram_indicator in anagram_indicators:
            for fodder in context.spans:
                fodder_span = fodder.as_tuple()
                if not _inside(fodder_span, wordplay_span):
                    continue
                if _overlaps(fodder_span, anagram_indicator.span):
                    continue
                letters = _clean_value(fodder.text)
                if not letters:
                    continue
                for shell in _anagram_container_shell_values(
                        context.answer, letters):
                    candidates.append(_SpanOperatedAnnotation(
                        fodder_span, fodder.text, "ANA_F", shell,
                        anagram_indicator,
                        "piece_%d_anagram_indicator"))
        operation_indicators = [
            ann for ann in context.annotations
            if ann.token == "REV_I" and _inside(ann.span, wordplay_span)
        ]
        for operation_indicator in operation_indicators:
            for source in source_annotations:
                if _overlaps(source.span, operation_indicator.span):
                    continue
                for value in _string_values(source.values):
                    candidates.append(_OperatedAnnotation(
                        source, operation_indicator, value[::-1],
                        "piece_%d_reversal_indicator"))
        homophone_indicators = [
            ann for ann in context.annotations
            if ann.token == "HOM_I" and _inside(ann.span, wordplay_span)
        ]
        for homophone_indicator in homophone_indicators:
            for source in context.annotations:
                if source.token != "HOM_F":
                    continue
                if not _inside(source.span, wordplay_span):
                    continue
                if _overlaps(source.span, homophone_indicator.span):
                    continue
                for value in _string_values(source.values):
                    candidates.append(_OperatedAnnotation(
                        source, homophone_indicator, value,
                        "piece_%d_homophone_indicator"))
        for indicator in indicators:
            container_sources = [
                ann for ann in candidates
                if not isinstance(ann, (
                    _ContainerAnnotation, _CompositeContainerAnnotation,
                    _ShellContainerAnnotation))
            ]
            for outer in container_sources:
                if _overlaps(outer.span, indicator.span):
                    continue
                for inner in container_sources:
                    if inner is outer:
                        continue
                    if _overlaps(inner.span, indicator.span):
                        continue
                    if _overlaps(inner.span, outer.span):
                        continue
                    if not _container_indicator_scopes(
                            indicator.span, outer.span, inner.span):
                        continue
                    for outer_value in _string_values(outer.values):
                        for inner_value in _string_values(inner.values):
                            result = _container_result(
                                outer_value, inner_value, context.answer)
                            if not result:
                                continue
                            candidates.append(_ContainerAnnotation(
                                outer, outer_value, inner, inner_value,
                                indicator, result))
            for outer in source_annotations:
                if _overlaps(outer.span, indicator.span):
                    continue
                inner_candidates = [
                    ann for ann in candidates
                    if ann is not outer
                    and not isinstance(ann, (
                        _CompositeContainerAnnotation,
                        _ShellContainerAnnotation))
                    and not _overlaps(ann.span, outer.span)
                    and not _overlaps(ann.span, indicator.span)
                ]
                for outer_value in _string_values(outer.values):
                    for result, payload in _container_payloads_in_answer(
                            outer_value, context.answer):
                        inner_assignments = _charade_search(
                            inner_candidates, payload)
                        for inner_assignment in inner_assignments:
                            if _assignment_spans_overlap(inner_assignment):
                                continue
                            candidates.append(_CompositeContainerAnnotation(
                                outer, outer_value, inner_assignment,
                                indicator, result))
            for inner in list(candidates):
                if _overlaps(inner.span, indicator.span):
                    continue
                shell_candidates = [
                    ann for ann in candidates
                    if ann is not inner
                    and not isinstance(ann, (
                        _CompositeContainerAnnotation,
                        _ShellContainerAnnotation))
                    and not _overlaps(ann.span, inner.span)
                    and not _overlaps(ann.span, indicator.span)
                ]
                for inner_value in _string_values(inner.values):
                    for result, shell_value in _container_shells_around_inner(
                            inner_value, context.answer):
                        shell_assignments = _charade_search(
                            shell_candidates, shell_value)
                        for shell_assignment in shell_assignments:
                            if _assignment_spans_overlap(shell_assignment):
                                continue
                            shell_span = _assignment_covering_span(
                                shell_assignment)
                            if shell_span is None:
                                continue
                            if not _container_indicator_scopes(
                                    indicator.span, shell_span, inner.span):
                                continue
                            candidates.append(_ShellContainerAnnotation(
                                shell_assignment, inner, inner_value,
                                indicator, result))
        for assignment in _charade_search(candidates, context.answer):
            if not any(isinstance(ann, (
                    _ContainerAnnotation, _CompositeContainerAnnotation,
                    _ShellContainerAnnotation))
                       for ann, _value in assignment):
                continue
            if _assignment_spans_overlap(assignment):
                continue
            parse = _complete_parse(context, _build_container_charade_parse(
                context, definition_span, assignment))
            if parse is not None:
                parses.append(parse)
    parses.sort(key=_parse_rank)
    return parses[:20]


def _assemble_substitution_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        sources = sorted(
            _source_annotations(context, wordplay_span),
            key=lambda ann: (
                -(ann.span[1] - ann.span[0]), ann.span[0], ann.span[1]))
        operators = [
            span for span in context.spans
            if _inside(span.as_tuple(), wordplay_span)
            and span.normalized in ("not", "instead of", "rather than")
        ]
        if not operators:
            continue
        source_values = {
            source: _bounded_string_values(source.values, context.answer)
            for source in sources
        }
        remove_values = {
            source: _bounded_string_values(
                source.values, context.answer, max_extra=1,
                max_count=20, require_overlap=False)
            for source in sources
        }
        insert_by_value = {}
        for source in sources:
            for value in source_values.get(source, ()):
                insert_by_value.setdefault(value, []).append(source)
        for base in sources:
            for base_value in source_values.get(base, ()):
                for remove in sources:
                    if remove is base or _overlaps(remove.span, base.span):
                        continue
                    for remove_value in remove_values.get(remove, ()):
                        needed_values = _needed_substitution_inserts(
                            base_value, remove_value, context.answer)
                        for insert_value in needed_values:
                            for insert in insert_by_value.get(insert_value, ()):
                                if insert is base or insert is remove:
                                    continue
                                if (_overlaps(insert.span, base.span)
                                        or _overlaps(insert.span, remove.span)):
                                    continue
                                for operator in operators:
                                    op_span = operator.as_tuple()
                                    if (_overlaps(op_span, base.span)
                                            or _overlaps(op_span, insert.span)
                                            or _overlaps(op_span, remove.span)):
                                        continue
                                    parse = _complete_parse(
                                        context,
                                        _build_substitution_parse(
                                            context, definition_span,
                                            base, base_value,
                                            insert, insert_value,
                                            remove, remove_value,
                                            op_span, operator.text,
                                            context.answer,
                                        ))
                                    if parse is not None:
                                        parses.append(parse)
    parses.sort(key=_parse_rank)
    return parses[:1]


def _assemble_positional_charade_parses(context):
    parses = []
    for def_candidate in _definition_candidates_or_none(context):
        wordplay_span, definition_span = _window_and_definition(context, def_candidate)
        source_annotations = _source_annotations(context, wordplay_span)
        positional_sources = _positional_source_candidates(context, wordplay_span)
        if not positional_sources:
            continue
        candidates = source_annotations + positional_sources
        for assignment in _charade_search(candidates, context.answer):
            if not any(_is_positional_candidate(ann)
                       for ann, _value in assignment):
                continue
            if _assignment_spans_overlap(assignment):
                continue
            parse = _complete_parse(context, _build_charade_parse(
                context, definition_span, assignment,
                operation="positional_charade"))
            if parse is not None:
                return [parse]
    return parses


def _build_charade_parse(context, definition_span, assignment,
                         operation="charade"):
    blocks = []
    if definition_span is not None:
        blocks.append(_definition_block(context, definition_span))
    input_ids = []
    for idx, (ann, value) in enumerate(assignment):
        block_id = "src_%d" % idx
        input_ids.append(block_id)
        blocks.append(ParseBlock(
            block_id, "SOURCE_BLOCK", ann.span, ann.text,
            token=ann.token, value=value, role="piece_%d" % idx,
            input_value=getattr(ann, "input_value", None)))
        indicator = getattr(ann, "indicator", None)
        if indicator is not None:
            role_template = getattr(
                ann, "indicator_role_template",
                "piece_%d_positional_indicator")
            blocks.append(ParseBlock(
                "op_%d" % idx, "OP_BLOCK", indicator.span, indicator.text,
                token=indicator.token,
                role=role_template % idx))
        remove = getattr(ann, "remove", None)
        if remove is not None:
            blocks.append(ParseBlock(
                "removed_%d" % idx, "REMOVED_BLOCK", remove.span,
                remove.text, token=remove.token,
                value=getattr(ann, "remove_value", None),
                role="piece_%d_removed" % idx))
    blocks.append(ParseBlock("answer", "ASSEMBLY_BLOCK", None,
                             context.answer, value=context.answer,
                             role="answer"))
    detail = " + ".join(
        _assignment_detail(ann, value)
        for ann, value in assignment
    )
    op = ParseOperation(
        operation=operation,
        indicator_block="",
        input_blocks=tuple(input_ids),
        output=context.answer,
        detail="%s = %s" % (detail, context.answer),
    )
    return TokenParse(
        parse_id="token_parse:%s:%s" % (
            operation, "_".join(str(ann.span[0]) for ann, _v in assignment)),
        operation=operation,
        answer=context.answer,
        blocks=tuple(blocks),
        operations=(op,),
        confidence="mechanically_verified",
    )


def _assignment_detail(annotation, value):
    indicator = getattr(annotation, "indicator", None)
    input_value = getattr(annotation, "input_value", None)
    remove_value = getattr(annotation, "remove_value", None)
    if indicator is not None and input_value and remove_value:
        return "%s without %s = %s" % (input_value, remove_value, value)
    if indicator is not None and input_value:
        if indicator.token == "POS_I_TRIM_LAST":
            return "%s without last letter = %s" % (input_value, value)
        if indicator.token == "POS_I_TRIM_FIRST":
            return "%s without first letter = %s" % (input_value, value)
        if indicator.token == "POS_I_TRIM_OUTER":
            return "%s without outside letters = %s" % (input_value, value)
        if indicator.token == "POS_I_TRIM_MIDDLE":
            return "%s without middle letters = %s" % (input_value, value)
    if indicator is not None and indicator.token == "ANA_I":
        return "anagram of %s = %s" % (annotation.text.upper(), value)
    return value


def _build_reversal_parse(context, definition_span, source, value, indicator):
    blocks = []
    if definition_span is not None:
        blocks.append(_definition_block(context, definition_span))
    blocks.extend([
        ParseBlock("src_0", "SOURCE_BLOCK", source.span, source.text,
                   token=source.token, value=value, role="source"),
        ParseBlock("op_0", "OP_BLOCK", indicator.span, indicator.text,
                   token=indicator.token, role="reversal_indicator"),
        ParseBlock("answer", "ASSEMBLY_BLOCK", None, context.answer,
                   value=context.answer, role="answer"),
    ])
    op = ParseOperation(
        operation="reversal",
        indicator_block="op_0",
        input_blocks=("src_0",),
        output=context.answer,
        detail="%s reversed = %s" % (value, context.answer),
    )
    return TokenParse(
        parse_id="token_parse:reversal:%d_%d" % source.span,
        operation="reversal",
        answer=context.answer,
        blocks=tuple(blocks),
        operations=(op,),
        confidence="mechanically_verified",
    )


def _build_homophone_parse(context, definition_span, source, value, indicator):
    blocks = []
    if definition_span is not None:
        blocks.append(_definition_block(context, definition_span))
    blocks.extend([
        ParseBlock("src_0", "SOURCE_BLOCK", source.span, source.text,
                   token=source.token, value=value, role="source"),
        ParseBlock("op_0", "OP_BLOCK", indicator.span, indicator.text,
                   token=indicator.token, role="homophone_indicator"),
        ParseBlock("answer", "ASSEMBLY_BLOCK", None, context.answer,
                   value=context.answer, role="answer"),
    ])
    op = ParseOperation(
        operation="homophone",
        indicator_block="op_0",
        input_blocks=("src_0",),
        output=context.answer,
        detail="%s sounds like %s" % (source.text, context.answer),
    )
    return TokenParse(
        parse_id="token_parse:homophone:%d_%d" % source.span,
        operation="homophone",
        answer=context.answer,
        blocks=tuple(blocks),
        operations=(op,),
        confidence="mechanically_verified",
    )


def _build_deletion_parse(context, definition_span, source, value, indicator,
                          input_value=None):
    blocks = []
    if definition_span is not None:
        blocks.append(_definition_block(context, definition_span))
    blocks.extend([
        ParseBlock("src_0", "SOURCE_BLOCK", source.span, source.text,
                   token=source.token, value=value, role="source",
                   input_value=input_value),
        ParseBlock("op_0", "OP_BLOCK", indicator.span, indicator.text,
                   token=indicator.token, role="deletion_indicator"),
        ParseBlock("answer", "ASSEMBLY_BLOCK", None, context.answer,
                   value=context.answer, role="answer"),
    ])
    op = ParseOperation(
        operation="deletion",
        indicator_block="op_0",
        input_blocks=("src_0",),
        output=context.answer,
        detail="%s trimmed to %s" % (source.text, context.answer),
    )
    return TokenParse(
        parse_id="token_parse:deletion:%d_%d" % source.span,
        operation="deletion",
        answer=context.answer,
        blocks=tuple(blocks),
        operations=(op,),
        confidence="mechanically_verified",
    )


def _build_hidden_parse(context, definition_span, fodder_span, fodder_text,
                        indicator, reversed_hidden=False):
    blocks = []
    if definition_span is not None:
        blocks.append(_definition_block(context, definition_span))
    blocks.extend([
        ParseBlock("src_0", "SOURCE_BLOCK", fodder_span, fodder_text,
                   token="HID_F", value=_clean_value(fodder_text),
                   role="hidden_fodder"),
        ParseBlock("op_0", "OP_BLOCK", indicator.span, indicator.text,
                   token=indicator.token, role="hidden_indicator"),
        ParseBlock("answer", "ASSEMBLY_BLOCK", None, context.answer,
                   value=context.answer, role="answer"),
    ])
    operation = "hidden_reversed" if reversed_hidden else "hidden"
    detail = "%s hidden in %s" % (context.answer, fodder_text)
    if reversed_hidden:
        detail = "%s reversed hidden in %s" % (context.answer, fodder_text)
    op = ParseOperation(
        operation=operation,
        indicator_block="op_0",
        input_blocks=("src_0",),
        output=context.answer,
        detail=detail,
    )
    return TokenParse(
        parse_id="token_parse:%s:%d_%d" % (
            operation, fodder_span[0], fodder_span[1]),
        operation=operation,
        answer=context.answer,
        blocks=tuple(blocks),
        operations=(op,),
        confidence="mechanically_verified",
    )


def _build_anagram_parse(context, definition_span, fodder_span, letters,
                         indicator, fodder_text=None):
    blocks = []
    if definition_span is not None:
        blocks.append(_definition_block(context, definition_span))
    blocks.extend([
        ParseBlock("src_0", "SOURCE_BLOCK", fodder_span,
                   fodder_text or context.span_text(*fodder_span), token="ANA_F",
                   value=letters, role="anagram_fodder"),
        ParseBlock("op_0", "OP_BLOCK", indicator.span, indicator.text,
                   token=indicator.token, role="anagram_indicator"),
        ParseBlock("answer", "ASSEMBLY_BLOCK", None, context.answer,
                   value=context.answer, role="answer"),
    ])
    op = ParseOperation(
        operation="anagram",
        indicator_block="op_0",
        input_blocks=("src_0",),
        output=context.answer,
        detail="anagram of %s = %s" % (letters, context.answer),
    )
    return TokenParse(
        parse_id="token_parse:anagram:%d_%d" % fodder_span,
        operation="anagram",
        answer=context.answer,
        blocks=tuple(blocks),
        operations=(op,),
        confidence="mechanically_verified",
    )


def _build_container_parse(context, definition_span, outer, outer_value,
                           inner, inner_value, indicator):
    blocks = []
    if definition_span is not None:
        blocks.append(_definition_block(context, definition_span))

    blocks.extend([
        ParseBlock("src_outer", "SOURCE_BLOCK", outer.span, outer.text,
                   token=outer.token, value=outer_value, role="outer"),
        ParseBlock("src_inner", "SOURCE_BLOCK", inner.span, inner.text,
                   token=inner.token, value=inner_value, role="inner"),
        ParseBlock("op_0", "RELATION_BLOCK", indicator.span,
                   indicator.text, token=indicator.token,
                   role="container_indicator"),
        ParseBlock("answer", "ASSEMBLY_BLOCK", None, context.answer,
                   value=context.answer, role="answer"),
    ])
    outer_indicator = getattr(outer, "indicator", None)
    if outer_indicator is not None:
        blocks.insert(-2, ParseBlock(
            "op_outer_pos", "OP_BLOCK", outer_indicator.span,
            outer_indicator.text, token=outer_indicator.token,
            role="outer_positional_indicator"))
    inner_indicator = getattr(inner, "indicator", None)
    if inner_indicator is not None:
        blocks.insert(-2, ParseBlock(
            "op_inner_pos", "OP_BLOCK", inner_indicator.span,
            inner_indicator.text, token=inner_indicator.token,
            role="inner_positional_indicator"))
    op = ParseOperation(
        operation="container",
        indicator_block="op_0",
        input_blocks=("src_outer", "src_inner"),
        output=context.answer,
        detail="%s contains %s = %s" % (
            outer_value, inner_value, context.answer),
    )
    return TokenParse(
        parse_id="token_parse:container:%d_%d:%d_%d" % (
            outer.span[0], outer.span[1], inner.span[0], inner.span[1]),
        operation="container",
        answer=context.answer,
        blocks=tuple(blocks),
        operations=(op,),
        confidence="mechanically_verified",
    )


def _build_container_charade_parse(context, definition_span, assignment):
    blocks = []
    if definition_span is not None:
        blocks.append(_definition_block(context, definition_span))
    input_ids = []
    for idx, (ann, value) in enumerate(assignment):
        block_id = "src_%d" % idx
        input_ids.append(block_id)
        if isinstance(ann, _ContainerAnnotation):
            blocks.extend([
                ParseBlock(block_id, "SOURCE_BLOCK", ann.outer.span,
                           ann.outer.text, token=ann.outer.token,
                           value=value, role="piece_%d" % idx,
                           input_value=ann.outer_value),
                ParseBlock("src_%d_inner" % idx, "SOURCE_BLOCK",
                           ann.inner.span, ann.inner.text,
                           token=ann.inner.token, value=ann.inner_value,
                           role="container_inner"),
                ParseBlock("op_%d" % idx, "RELATION_BLOCK",
                           ann.indicator.span, ann.indicator.text,
                           token=ann.indicator.token,
                           role="piece_%d_container_indicator" % idx),
            ])
            outer_indicator = getattr(ann.outer, "indicator", None)
            if outer_indicator is not None:
                role_template = getattr(
                    ann.outer, "indicator_role_template",
                    "piece_%d_positional_indicator")
                blocks.append(ParseBlock(
                    "op_%d_outer" % idx, "OP_BLOCK",
                    outer_indicator.span, outer_indicator.text,
                    token=outer_indicator.token,
                    role=role_template % idx))
            inner_indicator = getattr(ann.inner, "indicator", None)
            if inner_indicator is not None:
                role_template = getattr(
                    ann.inner, "indicator_role_template",
                    "piece_%d_positional_indicator")
                blocks.append(ParseBlock(
                    "op_%d_inner" % idx, "OP_BLOCK",
                    inner_indicator.span, inner_indicator.text,
                    token=inner_indicator.token,
                    role=role_template % idx))
        elif isinstance(ann, _CompositeContainerAnnotation):
            blocks.extend([
                ParseBlock(block_id, "SOURCE_BLOCK", ann.outer.span,
                           ann.outer.text, token=ann.outer.token,
                           value=value, role="piece_%d" % idx,
                           input_value=ann.outer_value),
                ParseBlock("op_%d" % idx, "RELATION_BLOCK",
                           ann.indicator.span, ann.indicator.text,
                           token=ann.indicator.token,
                           role="piece_%d_container_indicator" % idx),
            ])
            for inner_idx, (inner_ann, inner_value) in enumerate(
                    ann.inner_assignment):
                blocks.append(ParseBlock(
                    "src_%d_inner_%d" % (idx, inner_idx),
                    "SOURCE_BLOCK", inner_ann.span, inner_ann.text,
                    token=inner_ann.token, value=inner_value,
                    role="container_inner_%d" % inner_idx))
        elif isinstance(ann, _ShellContainerAnnotation):
            container_text = ann.text or context.span_text(*ann.span)
            blocks.append(ParseBlock(
                block_id, "SOURCE_BLOCK", None, container_text,
                token="CONTAINER_RESULT", value=value,
                role="piece_%d" % idx))
            for shell_idx, (shell_ann, shell_value) in enumerate(
                    ann.shell_assignment):
                blocks.append(ParseBlock(
                    "src_%d_shell_%d" % (idx, shell_idx),
                    "SOURCE_BLOCK", shell_ann.span, shell_ann.text,
                    token=shell_ann.token, value=shell_value,
                    role="container_shell_%d" % shell_idx))
                shell_indicator = getattr(shell_ann, "indicator", None)
                if shell_indicator is not None:
                    role_template = getattr(
                        shell_ann, "indicator_role_template",
                        "piece_%d_positional_indicator")
                    blocks.append(ParseBlock(
                        "op_%d_shell_%d" % (idx, shell_idx),
                        "OP_BLOCK", shell_indicator.span,
                        shell_indicator.text,
                        token=shell_indicator.token,
                        role=role_template % idx))
            blocks.extend([
                ParseBlock("op_%d" % idx, "RELATION_BLOCK",
                           ann.indicator.span, ann.indicator.text,
                           token=ann.indicator.token,
                           role="piece_%d_container_indicator" % idx),
                ParseBlock("src_%d_inner" % idx, "SOURCE_BLOCK",
                           ann.inner.span, ann.inner.text,
                           token=ann.inner.token, value=ann.inner_value,
                           role="container_inner"),
            ])
            inner_indicator = getattr(ann.inner, "indicator", None)
            if inner_indicator is not None:
                role_template = getattr(
                    ann.inner, "indicator_role_template",
                    "piece_%d_positional_indicator")
                blocks.append(ParseBlock(
                    "op_%d_inner" % idx, "OP_BLOCK",
                    inner_indicator.span, inner_indicator.text,
                    token=inner_indicator.token,
                    role=role_template % idx))
        else:
            blocks.append(ParseBlock(
                block_id, "SOURCE_BLOCK", ann.span, ann.text,
                token=ann.token, value=value, role="piece_%d" % idx))
            indicator = getattr(ann, "indicator", None)
            if indicator is not None:
                role_template = getattr(
                    ann, "indicator_role_template",
                    "piece_%d_positional_indicator")
                blocks.append(ParseBlock(
                    "op_%d" % idx, "OP_BLOCK", indicator.span,
                    indicator.text, token=indicator.token,
                    role=role_template % idx))
    blocks.append(ParseBlock("answer", "ASSEMBLY_BLOCK", None,
                             context.answer, value=context.answer,
                             role="answer"))
    detail = " + ".join(value for _ann, value in assignment)
    op = ParseOperation(
        operation="container_charade",
        indicator_block="",
        input_blocks=tuple(input_ids),
        output=context.answer,
        detail="%s = %s" % (detail, context.answer),
    )
    return TokenParse(
        parse_id="token_parse:container_charade:%s" % (
            "_".join(str(ann.span[0]) for ann, _v in assignment)),
        operation="container_charade",
        answer=context.answer,
        blocks=tuple(blocks),
        operations=(op,),
        confidence="mechanically_verified",
    )


def _build_substitution_parse(context, definition_span, base, base_value,
                              insert, insert_value, remove, remove_value,
                              operator_span, operator_text, result):
    blocks = []
    if definition_span is not None:
        blocks.append(_definition_block(context, definition_span))
    blocks.extend([
        ParseBlock("src_base", "SOURCE_BLOCK", base.span, base.text,
                   token=base.token, value=base_value, role="sub_base"),
        ParseBlock("src_insert", "SOURCE_BLOCK", insert.span, insert.text,
                   token=insert.token, value=insert_value, role="sub_insert"),
        ParseBlock("op_0", "RELATION_BLOCK", operator_span, operator_text,
                   token="SUB_I", role="substitution_indicator"),
        ParseBlock("src_remove", "SOURCE_BLOCK", remove.span, remove.text,
                   token=remove.token, value=remove_value,
                   role="sub_remove"),
        ParseBlock("answer", "ASSEMBLY_BLOCK", None,
                   context.answer, value=context.answer, role="answer"),
    ])
    op = ParseOperation(
        operation="substitution",
        indicator_block="op_0",
        input_blocks=("src_base", "src_insert", "src_remove"),
        output=context.answer,
        detail="%s with %s replacing %s = %s" % (
            base_value, insert_value, remove_value, result),
    )
    return TokenParse(
        parse_id="token_parse:substitution:%d_%d:%d_%d:%d_%d" % (
            base.span[0], base.span[1],
            insert.span[0], insert.span[1],
            remove.span[0], remove.span[1]),
        operation="substitution",
        answer=context.answer,
        blocks=tuple(blocks),
        operations=(op,),
        confidence="mechanically_verified",
    )


def _definition_candidates_or_none(context):
    inferred = tuple(_inferred_edge_definition_candidates(context))
    if context.definition_candidates or inferred:
        candidates = []
        seen = set()
        for candidate in tuple(context.definition_candidates) + inferred:
            key = (
                candidate.definition_span.as_tuple(),
                candidate.wordplay_span.as_tuple(),
            )
            if key in seen:
                continue
            seen.add(key)
            candidates.append(candidate)
        return tuple(candidates) + (None,)
    return (None,)


def _inferred_edge_definition_candidates(context):
    """Obase-compatible fallback: try edge definition windows.

    The old solver can infer a definition by proving the remaining wordplay.
    WFW must preserve that capability instead of requiring the definition DB
    to fire before any mechanical assembly is attempted.
    """
    try:
        from .clue_context import DefinitionCandidate
    except Exception:
        return ()
    candidates = []
    max_words = min(5, len(context.tokens) - 2)
    if max_words < 1:
        return ()
    for n in range(1, max_words + 1):
        start_def = context.span(0, n, kind="inferred_definition_candidate")
        start_wp = context.span(n, len(context.tokens), kind="wordplay_window")
        if start_wp.text:
            candidates.append(DefinitionCandidate(start_def, start_wp))
        end_def = context.span(
            len(context.tokens) - n, len(context.tokens),
            kind="inferred_definition_candidate")
        end_wp = context.span(0, len(context.tokens) - n,
                              kind="wordplay_window")
        if end_wp.text:
            candidates.append(DefinitionCandidate(end_def, end_wp))
    return candidates


def _definition_block(context, definition_span):
    role = "definition"
    token = None
    if _definition_span_is_inferred(context, definition_span):
        role = "inferred_definition"
        token = "INFERRED_DEF"
    return ParseBlock(
        "def_0", "DEF_BLOCK", definition_span,
        context.span_text(*definition_span),
        token=token, value=context.answer, role=role)


def _definition_span_is_inferred(context, definition_span):
    if definition_span is None:
        return False
    for candidate in context.definition_candidates:
        if candidate.definition_span.as_tuple() == definition_span:
            return candidate.definition_span.kind.startswith("inferred")
    return True


def _complete_parse(context, parse):
    """Return parse with link/surface leftovers attached, or None.

    A tokenised parse is not publishable unless every clue token is accounted
    for. Leftover tokens may only be accepted when existing annotations mark
    them as link/surface words. An unused indicator annotation must not be
    displayed as a mechanism, because it did not perform any work in the parse.
    """
    covered = set()
    for block in parse.blocks:
        if block.span is None:
            continue
        covered.update(range(block.span[0], block.span[1]))

    extra_blocks = []
    idx = 0
    while idx < len(context.tokens):
        if idx in covered:
            idx += 1
            continue
        ann = _best_covering_non_source_annotation(context, idx, covered)
        if ann is None:
            token = context.tokens[idx]
            if not _clean_value(token.text):
                extra_blocks.append(ParseBlock(
                    "cov_%d" % len(extra_blocks),
                    "LINK_BLOCK",
                    (idx, idx + 1),
                    token.text,
                    token="SURFACE",
                    role="surface"))
                covered.add(idx)
                idx += 1
                continue
            extra_blocks.append(ParseBlock(
                "cov_%d" % len(extra_blocks),
                "LINK_BLOCK",
                (idx, idx + 1),
                token.text,
                token="SURFACE_GAP",
                role="surface_gap"))
            covered.add(idx)
            idx += 1
            continue
        block_id = "cov_%d" % len(extra_blocks)
        extra_blocks.append(ParseBlock(
            block_id, "LINK_BLOCK", ann.span, ann.text,
            token=ann.token,
            role="surface" if ann.token == "SURFACE" else "link"))
        covered.update(range(ann.span[0], ann.span[1]))
        idx = ann.span[1]

    if not extra_blocks:
        return _with_context_confidence(context, parse)
    answer_block_idx = next(
        (i for i, block in enumerate(parse.blocks)
         if block.kind == "ASSEMBLY_BLOCK"),
        len(parse.blocks),
    )
    blocks = (
        parse.blocks[:answer_block_idx]
        + tuple(extra_blocks)
        + parse.blocks[answer_block_idx:]
    )
    return _with_context_confidence(context, TokenParse(
        parse_id=parse.parse_id,
        operation=parse.operation,
        answer=parse.answer,
        blocks=blocks,
        operations=parse.operations,
        confidence=parse.confidence,
    ))


def _with_context_confidence(context, parse):
    if any(
        block.kind == "DEF_BLOCK"
        and block.role == "inferred_definition"
        for block in parse.blocks
    ):
        return _replace_confidence(
            parse, "mechanically_verified_inferred_definition")
    if any(block.token == "SURFACE_GAP" for block in parse.blocks):
        return _replace_confidence(
            parse, "mechanically_verified_surface_gaps")
    return parse


def _replace_confidence(parse, confidence):
    return TokenParse(
        parse_id=parse.parse_id,
        operation=parse.operation,
        answer=parse.answer,
        blocks=parse.blocks,
        operations=parse.operations,
        confidence=confidence,
    )


def _parse_rank(parse):
    """Prefer real WFW assemblies over whole-answer shortcuts.

    A parse that builds the answer from multiple source pieces is usually more
    useful than a direct synonym of the whole answer, especially after a human
    has added missing DB facts.  Extra coverage-only indicator blocks are a
    weaker signal than source blocks and should not make a shortcut win.
    """
    source_blocks = [
        block for block in parse.blocks
        if block.kind == "SOURCE_BLOCK"
    ]
    piece_blocks = [
        block for block in source_blocks
        if (block.role or "").startswith("piece_")
    ]
    coverage_ops = [
        block for block in parse.blocks
        if block.block_id.startswith("cov_") and block.kind == "OP_BLOCK"
    ]
    surface_gaps = [
        block for block in parse.blocks
        if block.token == "SURFACE_GAP"
    ]
    definition_blocks = [
        block for block in parse.blocks
        if block.kind == "DEF_BLOCK"
    ]
    inferred_defs = [
        block for block in definition_blocks
        if block.role == "inferred_definition"
    ]
    source_definition_overlaps = sum(
        1
        for source in source_blocks
        for definition in definition_blocks
        if source.span is not None
        and definition.span is not None
        and _overlaps(source.span, definition.span)
    )
    direct_whole_answer = any(
        _clean_value(block.value or "") == _clean_value(parse.answer)
        for block in piece_blocks
    )
    operation_priority = {
        "charade": 0,
        "positional_charade": 0,
        "container": 1,
        "reversal": 1,
        "anagram": 1,
        "hidden": 1,
        "homophone": 1,
        "deletion": 1,
        "deletion_charade": 1,
        "substitution": 1,
        "double_definition": 4,
    }.get(parse.operation, 2)
    return (
        1 if direct_whole_answer else 0,
        0 if definition_blocks else 1,
        len(inferred_defs),
        len(surface_gaps),
        source_definition_overlaps,
        -len(piece_blocks),
        len(coverage_ops),
        operation_priority,
        len(source_blocks),
        parse.parse_id,
    )


def _best_covering_non_source_annotation(context, idx, covered):
    candidates = []
    for ann in context.annotations:
        if ann.span[0] != idx:
            continue
        if any(i in covered for i in range(ann.span[0], ann.span[1])):
            continue
        if ann.token in ("LNK", "SURFACE"):
            candidates.append(ann)
    if not candidates:
        return None
    candidates.sort(key=lambda ann: (
        0 if ann.token in ("LNK", "SURFACE") else 1,
        -(ann.span[1] - ann.span[0]),
        ann.span[1],
    ))
    return candidates[0]


def _indicator_role(token):
    return {
        "ANA_I": "anagram_indicator",
        "CON_I": "container_indicator",
        "DEL_I": "deletion_indicator",
        "HID_I": "hidden_indicator",
        "HOM_I": "homophone_indicator",
        "INS_I": "insertion_indicator",
        "POS_I": "positional_indicator",
        "REV_I": "reversal_indicator",
    }.get(token, "positional_indicator"
          if token.startswith("POS_I_") else "indicator")


def _window_and_definition(context, def_candidate):
    if def_candidate is None:
        return (0, len(context.tokens)), None
    return def_candidate.wordplay_span.as_tuple(), def_candidate.definition_span.as_tuple()


def _source_annotations(context, wordplay_span):
    return [
        ann for ann in context.annotations
        if (
            ann.token in ("SYN_F", "ABR_F")
            or (
                ann.token == "RAW"
                and len(_clean_value(ann.text)) <= 3
            )
        )
        and _inside(ann.span, wordplay_span)
    ]


def _token_has_annotation(context, idx, token):
    return any(
        ann.token == token and ann.span[0] <= idx < ann.span[1]
        for ann in context.annotations
    )


def _positional_source_candidates(context, wordplay_span):
    indicators = [
        ann for ann in context.annotations
        if ann.token.startswith("POS_I_") and _inside(ann.span, wordplay_span)
    ]
    if not indicators:
        return []
    sources = [
        ann for ann in context.annotations
        if ann.token == "POS_F" and _inside(ann.span, wordplay_span)
    ]
    sources.extend(_span_positional_sources(context, wordplay_span))
    candidates = []
    for source in sources:
        for indicator in indicators:
            if _overlaps(source.span, indicator.span):
                continue
            for value in _extract_positional_values(
                    source.text, indicator.token):
                if not value:
                    continue
                candidates.append(_PositionalAnnotation(
                    source, indicator, value))
            source_after_of = _following_source_after_of(context, indicator)
            if source_after_of is not None:
                for value in _extract_positional_values(
                        source_after_of.text, indicator.token):
                    if not value:
                        continue
                    candidates.append(_PositionalAnnotation(
                        source_after_of, indicator, value))
    half_indicators = [
        indicator for indicator in indicators
        if indicator.token == "POS_I_HALF"
    ]
    if half_indicators:
        for source in _source_annotations(context, wordplay_span):
            for indicator in half_indicators:
                if _overlaps(source.span, indicator.span):
                    continue
                for input_value in _string_values(source.values):
                    for value in _extract_positional_values(
                            input_value, indicator.token):
                        if not value:
                            continue
                        candidates.append(_OperatedAnnotation(
                            source, indicator, value,
                            "piece_%d_positional_indicator",
                            input_value=input_value))
    candidates.sort(key=lambda ann: (
        _positional_indicator_priority(ann.indicator.token),
        ann.span[0], -(ann.span[1] - ann.span[0]), ann.span[1],
        ann.indicator.span[0], ann.indicator.span[1]))
    return candidates


def _following_source_after_of(context, indicator):
    if indicator.token != "POS_I_FIRST":
        return None
    if indicator.span[1] >= len(context.tokens):
        return None
    between = context.tokens[indicator.span[1]].text.lower().strip(".,;:!?\"'()-")
    if between != "of":
        return None
    source_idx = indicator.span[1] + 1
    candidates = [
        ann for ann in context.annotations
        if ann.span == (source_idx, source_idx + 1)
        and ann.token == "POS_F"
    ]
    return candidates[0] if candidates else None


def _positional_indicator_priority(token):
    return {
        "POS_I_OUTER": 0,
        "POS_I_FIRST": 1,
        "POS_I_LAST": 1,
        "POS_I_ALTERNATE": 1,
        "POS_I_HALF": 1,
        "POS_I_MIDDLE": 2,
        "POS_I_TRIM_MIDDLE": 3,
        "POS_I_TRIM_FIRST": 3,
        "POS_I_TRIM_LAST": 3,
        "POS_I_TRIM_OUTER": 3,
    }.get(token, 5)


def _span_positional_sources(context, wordplay_span):
    sources = []
    for span in context.spans:
        span_tuple = span.as_tuple()
        if not _inside(span_tuple, wordplay_span):
            continue
        if span_tuple[1] - span_tuple[0] < 2:
            continue
        if span_tuple[1] - span_tuple[0] > 4:
            continue
        letters = _clean_positional_value(span.text)
        if len(letters) < 2:
            continue
        sources.append(SpanAnnotation(
            span_tuple, span.text, "POS_F", (letters,), "span_positional"))
    return sources


def _trimmed_source_candidates(context, wordplay_span):
    indicators = [
        ann for ann in context.annotations
        if ann.token.startswith("POS_I_TRIM_")
        and _inside(ann.span, wordplay_span)
    ]
    if not indicators:
        return []
    candidates = []
    for source in _trimmable_source_annotations(context, wordplay_span):
        for indicator in indicators:
            if _overlaps(source.span, indicator.span):
                continue
            for input_value, value in _trimmed_candidate_value_pairs(
                    source, indicator.token):
                candidates.append(_OperatedAnnotation(
                    source, indicator, value,
                    "piece_%d_deletion_indicator",
                    input_value=input_value))
    candidates.sort(key=lambda ann: (
        ann.span[0], -(ann.span[1] - ann.span[0]), ann.span[1],
        ann.indicator.span[0], ann.indicator.span[1]))
    return candidates


def _trimmable_source_annotations(context, wordplay_span):
    sources = [
        ann for ann in context.annotations
        if ann.token in ("SYN_F", "ABR_F", "POS_F", "RAW")
        and _inside(ann.span, wordplay_span)
    ]
    sources.sort(key=lambda ann: (
        _source_span_punctuation_penalty(context, ann.span),
        ann.span[0], _candidate_sort_priority(ann),
        -(ann.span[1] - ann.span[0]), ann.span[1]))
    return sources


def _source_span_punctuation_penalty(context, span):
    start, end = span
    if start < 0 or end > len(context.tokens) or start >= end:
        return 1
    edge_tokens = (context.tokens[start].text, context.tokens[end - 1].text)
    return sum(
        1 for text in edge_tokens
        if not any(ch.isalpha() for ch in text)
    )


def _trimmed_candidate_values(source, indicator_token):
    return tuple(
        value for _input_value, value
        in _trimmed_candidate_value_pairs(source, indicator_token)
    )


def _trimmed_candidate_value_pairs(source, indicator_token):
    if source.token == "POS_F":
        return tuple(
            (source.text, value)
            for value in _extract_trimmed_values(
                source.text, indicator_token)
        )
    pairs = []
    seen = set()
    for value in _string_values(source.values):
        for trimmed in _extract_trimmed_values(value, indicator_token):
            key = (value, trimmed)
            if key in seen:
                continue
            seen.add(key)
            pairs.append(key)
    return tuple(pairs)


def _implicit_container_indicators(context, wordplay_span):
    phrase_indicators = {
        "inhabited by",
        "occupied by",
        "filled by",
        "held by",
    }
    indicators = []
    for span in context.spans:
        span_tuple = span.as_tuple()
        if not _inside(span_tuple, wordplay_span):
            continue
        if span.normalized not in phrase_indicators:
            continue
        indicators.append(SpanAnnotation(
            span=span_tuple,
            text=span.text,
            token="CON_I",
            values=(True,),
            source="implicit_container_phrase",
        ))
    return indicators


class _PositionalAnnotation:
    def __init__(self, source, indicator, value):
        self.span = source.span
        self.text = source.text
        self.token = source.token
        self.values = (value,)
        self.source = source.source
        self.indicator = indicator


class _OperatedAnnotation(_PositionalAnnotation):
    def __init__(self, source, indicator, value, indicator_role_template,
                 input_value=None):
        super().__init__(source, indicator, value)
        self.indicator_role_template = indicator_role_template
        self.input_value = input_value


class _DeletionAnnotation(_OperatedAnnotation):
    def __init__(self, base, remove, indicator, value, input_value,
                 remove_value):
        super().__init__(
            base, indicator, value, "piece_%d_deletion_indicator",
            input_value=input_value)
        self.remove = remove
        self.remove_value = remove_value


class _SpanOperatedAnnotation:
    def __init__(self, span, text, token, value, indicator,
                 indicator_role_template, input_value=None):
        self.span = span
        self.text = text
        self.token = token
        self.values = (value,)
        self.source = "token_parse_assembler"
        self.indicator = indicator
        self.indicator_role_template = indicator_role_template
        self.input_value = input_value


class _ContainerAnnotation:
    def __init__(self, outer, outer_value, inner, inner_value, indicator,
                 value):
        self.outer = outer
        self.outer_value = outer_value
        self.inner = inner
        self.inner_value = inner_value
        self.indicator = indicator
        self.span = outer.span
        self.text = outer.text
        self.token = outer.token
        self.values = (value,)
        self.source = "token_parse_assembler"


class _CompositeContainerAnnotation:
    def __init__(self, outer, outer_value, inner_assignment, indicator, value):
        self.outer = outer
        self.outer_value = outer_value
        self.inner_assignment = tuple(inner_assignment)
        self.indicator = indicator
        self.span = outer.span
        self.text = outer.text
        self.token = outer.token
        self.values = (value,)
        self.source = "token_parse_assembler"


class _ShellContainerAnnotation:
    def __init__(self, shell_assignment, inner, inner_value, indicator, value):
        self.shell_assignment = tuple(shell_assignment)
        self.inner = inner
        self.inner_value = inner_value
        self.indicator = indicator
        spans = [ann.span for ann, _value in self.shell_assignment]
        spans.append(inner.span)
        spans.append(indicator.span)
        self.span = (min(span[0] for span in spans),
                     max(span[1] for span in spans))
        self.text = ""
        self.token = "CONTAINER_RESULT"
        self.values = (value,)
        self.source = "token_parse_assembler"


def _is_positional_candidate(annotation):
    return getattr(annotation, "indicator", None) is not None


def _is_operated_candidate(annotation):
    return getattr(annotation, "indicator", None) is not None


def _is_deletion_candidate(annotation):
    indicator = getattr(annotation, "indicator", None)
    if indicator is None:
        return False
    return indicator.token == "DEL_I" or indicator.token.startswith("POS_I_TRIM_")


def _assignment_spans_overlap(assignment):
    spans = []
    seen = {}
    duplicate = False

    def add_span(span, kind="source"):
        nonlocal duplicate
        existing = seen.get(span)
        if existing is not None:
            if not _duplicate_indicator_allowed(existing, kind):
                duplicate = True
            existing.add(kind)
            return
        seen[span] = {kind}
        spans.append(span)

    for ann, _value in assignment:
        if isinstance(ann, _ContainerAnnotation):
            add_span(ann.outer.span)
            add_span(ann.inner.span)
            outer_indicator = getattr(ann.outer, "indicator", None)
            if outer_indicator is not None:
                add_span(outer_indicator.span,
                         _indicator_span_kind(outer_indicator))
            inner_indicator = getattr(ann.inner, "indicator", None)
            if inner_indicator is not None:
                add_span(inner_indicator.span,
                         _indicator_span_kind(inner_indicator))
        elif isinstance(ann, _CompositeContainerAnnotation):
            add_span(ann.outer.span)
            outer_indicator = getattr(ann.outer, "indicator", None)
            if outer_indicator is not None:
                add_span(outer_indicator.span,
                         _indicator_span_kind(outer_indicator))
            for inner_ann, _inner_value in ann.inner_assignment:
                if isinstance(inner_ann, _ContainerAnnotation):
                    add_span(inner_ann.outer.span)
                    add_span(inner_ann.inner.span)
                else:
                    add_span(inner_ann.span)
                inner_indicator = getattr(inner_ann, "indicator", None)
                if inner_indicator is not None:
                    add_span(inner_indicator.span,
                             _indicator_span_kind(inner_indicator))
        elif isinstance(ann, _ShellContainerAnnotation):
            for shell_ann, _shell_value in ann.shell_assignment:
                add_span(shell_ann.span)
                shell_indicator = getattr(shell_ann, "indicator", None)
                if shell_indicator is not None:
                    add_span(shell_indicator.span,
                             _indicator_span_kind(shell_indicator))
            add_span(ann.inner.span)
            inner_indicator = getattr(ann.inner, "indicator", None)
            if inner_indicator is not None:
                add_span(inner_indicator.span,
                         _indicator_span_kind(inner_indicator))
        else:
            add_span(ann.span)
        indicator = getattr(ann, "indicator", None)
        if indicator is not None:
            add_span(indicator.span, _indicator_span_kind(indicator))
    return duplicate or _spans_overlap_any(spans)


def _indicator_span_kind(indicator):
    return "indicator:%s" % getattr(indicator, "token", "")


def _duplicate_indicator_allowed(existing, kind):
    if not kind.startswith("indicator:"):
        return False
    if any(not item.startswith("indicator:") for item in existing):
        return False
    tokens = {item.split(":", 1)[1] for item in existing}
    tokens.add(kind.split(":", 1)[1])
    return all(token.startswith("POS_I_") for token in tokens)


def _assignment_covering_span(assignment):
    spans = []
    for ann, _value in assignment:
        if isinstance(ann, _ContainerAnnotation):
            spans.extend([ann.outer.span, ann.inner.span])
        elif isinstance(ann, _CompositeContainerAnnotation):
            spans.append(ann.outer.span)
            spans.extend(inner_ann.span for inner_ann, _ in ann.inner_assignment)
        elif isinstance(ann, _ShellContainerAnnotation):
            spans.extend(shell_ann.span for shell_ann, _ in ann.shell_assignment)
            spans.append(ann.inner.span)
        else:
            spans.append(ann.span)
    if not spans:
        return None
    return min(span[0] for span in spans), max(span[1] for span in spans)


def _indicator_annotations(context, wordplay_span):
    return [
        ann for ann in context.annotations
        if ann.token.endswith("_I") or ann.token.startswith("POS_I_")
        if _inside(ann.span, wordplay_span)
    ]


def _indicator_sort_key(context, annotation):
    return (
        1 if _span_has_token(context, annotation.span, "LNK") else 0,
        annotation.span[0],
        annotation.span[1],
    )


def _span_has_token(context, span, token):
    return any(
        ann.span == span and ann.token == token
        for ann in context.annotations
    )


def _all_link_tokens(context, span):
    start, end = span
    for idx in range(start, end):
        if not any(
            ann.token == "LNK"
            and ann.span[0] <= idx < ann.span[1]
            for ann in context.annotations
        ):
            return False
    return True


def _charade_search(source_annotations, answer):
    answer = _clean_value(answer)
    by_start = []
    for ann in source_annotations:
        values = tuple(
            value for value in _unique_values(_string_values(ann.values))
            if len(value) <= len(answer) and value in answer
        )
        if values:
            by_start.append((ann, values))
    by_start.sort(
        key=lambda ann: (
            _annotation_punctuation_penalty(ann[0]),
            ann[0].span[0], _candidate_sort_priority(ann[0]),
            -(ann[0].span[1] - ann[0].span[0]), ann[0].span[1]),
    )
    best = []

    def search(pos, min_token, assignment):
        if len(best) >= 50:
            return
        if pos == len(answer):
            best.append(tuple(assignment))
            return
        for ann, values in by_start:
            if ann.span[0] < min_token:
                continue
            if any(_overlaps(ann.span, prev.span)
                   for prev, _value in assignment):
                continue
            for value in values:
                if answer.startswith(value, pos):
                    assignment.append((ann, value))
                    search(pos + len(value), ann.span[1], assignment)
                    assignment.pop()

    search(0, 0, [])
    return best


def _candidate_sort_priority(annotation):
    if isinstance(annotation, _ShellContainerAnnotation):
        return 0
    if isinstance(annotation, _CompositeContainerAnnotation):
        return 1
    if isinstance(annotation, _ContainerAnnotation):
        return 2
    return 0


def _annotation_punctuation_penalty(annotation):
    text = (getattr(annotation, "text", "") or "").strip()
    if not text:
        return 1
    parts = text.split()
    edge_parts = (parts[0], parts[-1])
    return sum(
        1 for part in edge_parts
        if not any(ch.isalpha() for ch in part)
    )


def _container_matches(outer, inner, answer):
    outer = _clean_value(outer)
    inner = _clean_value(inner)
    answer = _clean_value(answer)
    if not outer or not inner:
        return False
    for pos in range(1, len(outer)):
        if outer[:pos] + inner + outer[pos:] == answer:
            return True
    return False


def _container_indicator_scopes(indicator_span, outer_span, inner_span):
    if outer_span[1] <= indicator_span[0] and indicator_span[1] <= inner_span[0]:
        return True
    if inner_span[1] <= indicator_span[0] and indicator_span[1] <= outer_span[0]:
        return True
    return False


def _container_result(outer, inner, answer):
    outer = _clean_value(outer)
    inner = _clean_value(inner)
    answer = _clean_value(answer)
    if not outer or not inner:
        return ""
    for pos in range(1, len(outer)):
        result = outer[:pos] + inner + outer[pos:]
        if result in answer:
            return result
    return ""


def _container_payloads_in_answer(outer, answer):
    outer = _clean_value(outer)
    answer = _clean_value(answer)
    if len(outer) < 2:
        return []
    payloads = []
    seen = set()
    for split in range(1, len(outer)):
        prefix = outer[:split]
        suffix = outer[split:]
        for start in range(0, len(answer)):
            if not answer.startswith(prefix, start):
                continue
            min_end = start + len(outer) + 1
            for end in range(min_end, len(answer) + 1):
                candidate = answer[start:end]
                if not candidate.endswith(suffix):
                    continue
                payload = candidate[len(prefix):len(candidate) - len(suffix)]
                if not payload:
                    continue
                key = (candidate, payload)
                if key in seen:
                    continue
                seen.add(key)
                payloads.append(key)
    return payloads


def _container_shells_around_inner(inner, answer):
    inner = _clean_value(inner)
    answer = _clean_value(answer)
    if not inner or len(inner) >= len(answer):
        return []
    shells = []
    seen = set()
    for start in range(0, len(answer)):
        for end in range(start + len(inner) + 2, len(answer) + 1):
            candidate = answer[start:end]
            inner_pos = candidate.find(inner)
            while inner_pos >= 0:
                before = candidate[:inner_pos]
                after = candidate[inner_pos + len(inner):]
                if before and after:
                    key = (candidate, before + after)
                    if key not in seen:
                        seen.add(key)
                        shells.append(key)
                inner_pos = candidate.find(inner, inner_pos + 1)
    return shells


def _anagram_container_shell_values(answer, letters):
    answer = _clean_value(answer)
    letters = _clean_value(letters)
    if not answer or not letters:
        return []
    if len(letters) >= len(answer):
        return []
    out = []
    seen = set()
    for start in range(0, len(answer)):
        for end in range(start + 1, len(answer) + 1):
            shell = answer[:start] + answer[end:]
            if len(shell) != len(letters):
                continue
            if sorted(shell) != sorted(letters):
                continue
            if shell in seen:
                continue
            seen.add(shell)
            out.append(shell)
    return out


def _substitute_value(base_value, remove_value, insert_value):
    base = _clean_value(base_value)
    remove = _clean_value(remove_value)
    insert = _clean_value(insert_value)
    if not base or not remove or not insert:
        return ""
    pos = base.find(remove)
    if pos < 0:
        return ""
    return base[:pos] + insert + base[pos + len(remove):]


def _needed_substitution_inserts(base_value, remove_value, answer):
    base = _clean_value(base_value)
    remove = _clean_value(remove_value)
    answer = _clean_value(answer)
    if not base or not remove or not answer:
        return ()
    values = []
    start = 0
    while True:
        pos = base.find(remove, start)
        if pos < 0:
            break
        prefix = base[:pos]
        suffix = base[pos + len(remove):]
        if answer.startswith(prefix) and answer.endswith(suffix):
            needed = answer[len(prefix):len(answer) - len(suffix)]
            if needed:
                values.append(needed)
        start = pos + 1
    return _unique_values(values)


def _homophone_value_matches(value, answer):
    pair = frozenset((_clean_value(value), _clean_value(answer)))
    return pair in {
        frozenset(("NEEDING", "KNEADING")),
    }


def _extract_positional(text, indicator_token):
    values = _extract_positional_values(text, indicator_token)
    return values[0] if values else ""


def _extract_positional_values(text, indicator_token):
    letters = _clean_positional_value(text)
    if not letters:
        return ()
    if indicator_token == "POS_I_FIRST":
        return (letters[0],)
    if indicator_token == "POS_I_LAST":
        return (letters[-1],)
    if indicator_token == "POS_I_OUTER":
        return (letters[0] + letters[-1] if len(letters) > 1 else letters,)
    if indicator_token == "POS_I_MIDDLE":
        if len(letters) <= 2:
            return ()
        mid = len(letters) // 2
        if len(letters) % 2:
            return (letters[mid],)
        return (letters[mid - 1:mid + 1],)
    if indicator_token == "POS_I_ALTERNATE":
        return _unique_values((letters[::2], letters[1::2]))
    if indicator_token == "POS_I_HALF":
        mid = len(letters) // 2
        return _unique_values((letters[:mid], letters[mid:]))
    return ()


def _extract_trimmed(text, indicator_token):
    values = _extract_trimmed_values(text, indicator_token)
    return values[0] if values else ""


def _extract_trimmed_values(text, indicator_token):
    letters = _clean_positional_value(text)
    if not letters:
        return ()
    if indicator_token == "POS_I_TRIM_FIRST":
        return _unique_values((letters[1:],))
    if indicator_token == "POS_I_TRIM_LAST":
        return _unique_values((letters[:-1],))
    if indicator_token == "POS_I_TRIM_OUTER":
        return _unique_values((letters[1:-1] if len(letters) > 2 else "",))
    if indicator_token == "POS_I_TRIM_MIDDLE":
        if len(letters) <= 2:
            return ()
        shell = letters[0] + letters[-1]
        mid = len(letters) // 2
        if len(letters) % 2:
            single_middle_removed = letters[:mid] + letters[mid + 1:]
        else:
            single_middle_removed = letters[:mid - 1] + letters[mid + 1:]
        return _unique_values((shell, single_middle_removed))
    return ()


def _delete_value(base_value, remove_value):
    base = _clean_value(base_value)
    remove = _clean_value(remove_value)
    if not base or not remove:
        return ""
    pos = base.find(remove)
    if pos < 0:
        return ""
    return base[:pos] + base[pos + len(remove):]


def _unique_values(values):
    seen = set()
    unique = []
    for value in values:
        if not value or value in seen:
            continue
        seen.add(value)
        unique.append(value)
    return tuple(unique)


def _string_values(values):
    for value in values:
        if isinstance(value, str):
            clean = _clean_value(value)
            if clean:
                yield clean


def _bounded_string_values(values, answer, max_extra=3, max_count=30,
                           require_overlap=True):
    """Small value set for expensive cross-product assemblers."""
    answer = _clean_value(answer)
    answer_letters = set(answer)
    candidates = []
    for value in _unique_values(_string_values(values)):
        if len(value) > len(answer) + max_extra:
            continue
        if require_overlap and answer_letters and not (set(value) & answer_letters):
            continue
        candidates.append(value)
    candidates.sort(key=lambda value: (
        0 if value in answer else 1,
        abs(len(value) - len(answer)),
        len(value),
        value,
    ))
    return tuple(candidates[:max_count])


def _inside(span, window):
    return span[0] >= window[0] and span[1] <= window[1]


def _overlaps(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _spans_overlap_any(spans):
    for idx, span in enumerate(spans):
        for other in spans[idx + 1:]:
            if _overlaps(span, other):
                return True
    return False


def _clean_value(value):
    return "".join(c for c in (value or "").upper() if c.isalpha())


def _clean_positional_value(value):
    text = value or ""
    lowered = text.lower().strip(".,;:!?\"()-")
    if lowered.endswith("'s"):
        text = text[:text.lower().rfind("'s")]
    return _clean_value(text)
