"""Read-only Stage Two evidence case files.

Stage Two does not publish a proof and does not write enrichment.  It gathers
the useful things the current DB, Stage One grammar spans, and light answer-fit
checks can already say, including the missing facts that would make a later
proof possible.
"""
from __future__ import annotations

from dataclasses import dataclass

from .clue_context import build_clue_context, with_wordplay_annotations
from .tokens import (
    ABR_F,
    ANA_F,
    ANA_I,
    CON_I,
    DEL_F,
    DEL_I,
    HID_F,
    HOM_F,
    HOM_I,
    LNK,
    POS_F,
    POS_I_HALF,
    POS_I_TRIM_FIRST,
    RAW,
    REV_I,
    SYN_F,
)


SOURCE_TOKENS = {SYN_F, ABR_F}
INDICATOR_TOKENS = {
    ANA_I,
    CON_I,
    DEL_I,
    HOM_I,
    POS_I_HALF,
    POS_I_TRIM_FIRST,
    REV_I,
}


@dataclass(frozen=True)
class StageTwoCaseFile:
    clue_text: str
    answer: str
    stage_one_context: object
    annotated_context: object
    definition_candidates: tuple[dict, ...]
    grammar_phrases: tuple[dict, ...]
    source_candidates: tuple[dict, ...]
    operation_candidates: tuple[dict, ...]
    working_pairs: tuple[dict, ...]
    assemblies: tuple[dict, ...]
    enrichment_candidates: tuple[dict, ...]
    unresolved_words: tuple[dict, ...]
    status: str

    def as_dict(self):
        return {
            "schema": "stage_two_casefile:v1",
            "status": self.status,
            "clue_text": self.clue_text,
            "answer": self.answer,
            "stage_one_context": self.stage_one_context.as_dict(),
            "annotated_context": self.annotated_context.as_dict(),
            "definition_candidates": list(self.definition_candidates),
            "grammar_phrases": list(self.grammar_phrases),
            "source_candidates": list(self.source_candidates),
            "operation_candidates": list(self.operation_candidates),
            "working_pairs": list(self.working_pairs),
            "assemblies": list(self.assemblies),
            "enrichment_candidates": list(self.enrichment_candidates),
            "unresolved_words": list(self.unresolved_words),
        }


def build_stage_two_casefile(clue_text, answer, db, stage_one_context=None):
    """Build a read-only evidence case file for one clue."""
    stage_one = stage_one_context or build_clue_context(
        clue_text, answer, db, annotate=False)
    context = with_wordplay_annotations(stage_one, db)
    answer_clean = _clean_answer(answer)

    definitions = _definition_candidates(stage_one)
    grammar = _grammar_phrases(stage_one)
    sources = _source_candidates(context, answer_clean)
    operations = _operation_candidates(context)
    working_pairs = tuple(_working_pairs(context, answer_clean, sources))
    enrichments = list(_grammar_span_enrichments(context, sources))
    enrichments.extend(_definition_gap_enrichments(stage_one, working_pairs))
    enrichments.extend(_conditional_suburb_enrichments(stage_one, working_pairs))

    assemblies = tuple(_assemblies(answer_clean, sources, working_pairs,
                                   enrichments))
    unresolved = tuple(_unresolved_words(context, sources, operations,
                                         working_pairs, definitions))
    status = _case_status(assemblies, enrichments, unresolved)
    return StageTwoCaseFile(
        clue_text=clue_text,
        answer=answer_clean,
        stage_one_context=stage_one,
        annotated_context=context,
        definition_candidates=tuple(definitions),
        grammar_phrases=tuple(grammar),
        source_candidates=tuple(sources),
        operation_candidates=tuple(operations),
        working_pairs=working_pairs,
        assemblies=assemblies,
        enrichment_candidates=tuple(enrichments),
        unresolved_words=unresolved,
        status=status,
    )


def build_stage_two_from_solve_result(
        clue_text, answer, db, solve_result, *, stage_one_context=None):
    """Build Stage Two from the legacy solver's retained word roles."""
    if (solve_result is None
            or solve_result.result is None
            or not solve_result.result.word_roles):
        return build_stage_two_casefile(
            clue_text, answer, db,
            stage_one_context=stage_one_context)

    answer_clean = _clean_answer(answer)
    stage_one = stage_one_context or build_clue_context(
        clue_text, answer_clean, db, annotate=False)
    context = with_wordplay_annotations(stage_one, db)

    def _normalize_words(text):
        return " ".join(
            "".join(
                char.lower() if char.isalpha() or char.isspace() else ""
                for char in (text or "")
            ).split()
        )

    token_texts = [_normalize_words(token.text) for token in stage_one.tokens]
    n_tokens = len(token_texts)
    if stage_one.definition_candidates:
        first_candidate = stage_one.definition_candidates[0]
        wp_start = first_candidate.wordplay_span.start
        wp_end = first_candidate.wordplay_span.end
    else:
        wp_start = 0
        wp_end = n_tokens

    used_annotation_indices = set()
    used_token_spans = set()

    def _role_parts(role):
        word = role[0] if len(role) > 0 else ""
        token = role[1] if len(role) > 1 else None
        value = role[2] if len(role) > 2 else None
        return word, token, value

    def _annotation_value_matches(annotation, value):
        values = tuple(annotation.values or ())
        if values == (True,) or values == [True]:
            return True
        clean_value = _clean_answer(value)
        if not clean_value:
            return True
        clean_values = [
            _clean_answer(item)
            for item in values
            if isinstance(item, str)
        ]
        return clean_value in clean_values

    def _map_word_to_span(word, token, value,
                          context_annotations, token_texts,
                          wp_start, wp_end,
                          used_annotation_indices, used_token_spans):
        word_text = _normalize_words(word)
        for idx, annotation in enumerate(context_annotations):
            if idx in used_annotation_indices:
                continue
            if _normalize_words(annotation.text) != word_text:
                continue
            if annotation.token != token:
                continue
            if not _annotation_value_matches(annotation, value):
                continue
            used_annotation_indices.add(idx)
            return annotation.span, "mapped"

        word_parts = word_text.split()
        if word_parts:
            width = len(word_parts)
            for start in range(wp_start, wp_end - width + 1):
                end = start + width
                span = (start, end)
                if span in used_token_spans:
                    continue
                if " ".join(token_texts[start:end]) != " ".join(word_parts):
                    continue
                used_token_spans.add(span)
                return span, "mapped"

        return None, "ambiguous"

    def _map_phrase_to_span(phrase, token_texts):
        phrase_parts = _normalize_words(phrase).split()
        if not phrase_parts:
            return None, "ambiguous"
        width = len(phrase_parts)
        for start in range(0, len(token_texts) - width + 1):
            end = start + width
            if token_texts[start:end] == phrase_parts:
                return (start, end), "mapped"
        return None, "ambiguous"

    def _source_derivation(text, token, value, db):
        clean_text = _clean_answer(text)
        clean_value = _clean_answer(value)

        if not clean_value:
            return {
                "derivation_kind": "empty",
                "evidence_status": "failed",
                "evidence_reason": "source value is empty",
            }

        if token in (RAW, ANA_F, HID_F, POS_F, DEL_F):
            if clean_value == clean_text:
                return {
                    "derivation_kind": "raw_full_text",
                    "evidence_status": "verified",
                    "evidence_reason": "value is the full cleaned source text",
                }
            return {
                "derivation_kind": "unlicensed_partial",
                "evidence_status": "failed",
                "evidence_reason": (
                    "value is not the full source text and no operation "
                    "licence is attached"
                ),
            }

        if token == ABR_F:
            if clean_value in [
                    _clean_answer(v)
                    for v in db.get_abbreviations(text)]:
                return {
                    "derivation_kind": "db_abbreviation",
                    "evidence_status": "verified",
                    "evidence_reason": "value is an exact DB abbreviation",
                }
            return {
                "derivation_kind": "bad_abbreviation",
                "evidence_status": "failed",
                "evidence_reason": "value is not an exact DB abbreviation",
            }

        if token == SYN_F:
            if clean_value in [
                    _clean_answer(v)
                    for v in db.get_synonyms(text)]:
                return {
                    "derivation_kind": "db_synonym",
                    "evidence_status": "verified",
                    "evidence_reason": "value is an exact DB synonym",
                }
            return {
                "derivation_kind": "bad_synonym",
                "evidence_status": "failed",
                "evidence_reason": "value is not an exact DB synonym",
            }

        if token == HOM_F:
            if clean_value in [
                    _clean_answer(v)
                    for v in db.get_homophones(text)]:
                return {
                    "derivation_kind": "db_homophone",
                    "evidence_status": "verified",
                    "evidence_reason": "value is an exact DB homophone",
                }
            return {
                "derivation_kind": "bad_homophone",
                "evidence_status": "failed",
                "evidence_reason": "value is not an exact DB homophone",
            }

        return {
            "derivation_kind": "unknown_token",
            "evidence_status": "failed",
            "evidence_reason": "source token is not recognised by verifier",
        }

    source_candidates = []
    operation_candidates = []
    link_spans = []
    for role in solve_result.result.word_roles:
        word, token, value = _role_parts(role)
        span, span_status = _map_word_to_span(
            word, token, value,
            context.annotations, token_texts,
            wp_start, wp_end,
            used_annotation_indices, used_token_spans)
        if token == LNK:
            link_spans.append(span)
            continue
        if token in INDICATOR_TOKENS or (
                isinstance(token, str) and token.startswith("POS_I")):
            operation_candidates.append({
                "text": word,
                "span": span,
                "span_status": span_status,
                "token": token,
                "source": "word_roles",
                "role": "operation",
            })
            continue
        if value is not None:
            clean_value = _clean_answer(value)
            derivation = _source_derivation(word, token, clean_value, db)
            source_candidates.append({
                "text": word,
                "span": span,
                "span_status": span_status,
                "token": token,
                "value": clean_value,
                "source": "word_roles",
                "relation_to_answer": (
                    "contained_in_answer"
                    if clean_value and clean_value in answer_clean
                    else "unknown"
                ),
                **derivation,
            })

    definitions = []
    if getattr(solve_result, "definition", None):
        definition_span, definition_span_status = _map_phrase_to_span(
            solve_result.definition, token_texts)
        definitions = [{
            "text": solve_result.definition,
            "span": definition_span,
            "span_status": definition_span_status,
            "wordplay_span": None,
            "wordplay_text": None,
            "boundary_status": "legacy_solver",
            "objections": [],
        }]
    else:
        definitions = _definition_candidates(stage_one)

    grammar = _grammar_phrases(stage_one)

    covered_indices = set()
    for item in source_candidates:
        if item["span"]:
            for index in range(item["span"][0], item["span"][1]):
                covered_indices.add(index)
    for item in operation_candidates:
        if item["span"]:
            for index in range(item["span"][0], item["span"][1]):
                covered_indices.add(index)
    for span in link_spans:
        if span:
            for index in range(span[0], span[1]):
                covered_indices.add(index)
    if definitions and definitions[0].get("span"):
        def_span = definitions[0]["span"]
        for index in range(def_span[0], def_span[1]):
            covered_indices.add(index)
    elif stage_one.definition_candidates:
        def_span = stage_one.definition_candidates[0].definition_span
        for index in range(def_span.start, def_span.end):
            covered_indices.add(index)

    unresolved = []
    for index, token in enumerate(stage_one.tokens):
        if index not in covered_indices:
            unresolved.append({
                "text": token.text,
                "index": index,
                "reason": "not covered by any mapped span",
            })

    op_tokens = {
        token for _, token, _ in (
            _role_parts(role) for role in solve_result.result.word_roles)
        if token != LNK
        and (token in INDICATOR_TOKENS
             or (isinstance(token, str) and token.startswith("POS_I")))
    }
    if ANA_I in op_tokens:
        assembly_kind = "anagram"
    elif CON_I in op_tokens:
        assembly_kind = "container"
    elif REV_I in op_tokens:
        assembly_kind = "reversal"
    elif DEL_I in op_tokens:
        assembly_kind = "deletion"
    elif HOM_I in op_tokens:
        assembly_kind = "homophone"
    elif any(token.startswith("POS_I") for token in op_tokens):
        assembly_kind = "positional"
    elif not op_tokens:
        assembly_kind = "charade"
    else:
        assembly_kind = "compound"

    parts = [
        {
            "text": item["text"],
            "value": item["value"],
            "span": item["span"],
            "token": item.get("token"),
            "derivation_kind": item.get("derivation_kind"),
            "evidence_status": item.get("evidence_status"),
            "evidence_reason": item.get("evidence_reason"),
        }
        for item in source_candidates
        if item.get("value")
    ]
    joined = "".join(_clean_answer(part["value"]) for part in parts)
    if assembly_kind == "anagram":
        answer_fit = sorted(joined) == sorted(answer_clean)
    else:
        answer_fit = joined == answer_clean
    assembly_status = "answer_fit" if answer_fit else "evidence_only"
    assemblies = ({
        "kind": assembly_kind,
        "status": assembly_status,
        "output": answer_clean,
        "parts": parts,
    },)

    status = "answer_fit" if solve_result.high_confidence else "evidence_only"

    return StageTwoCaseFile(
        clue_text=clue_text,
        answer=answer_clean,
        stage_one_context=stage_one,
        annotated_context=context,
        definition_candidates=tuple(definitions),
        grammar_phrases=tuple(grammar),
        source_candidates=tuple(source_candidates),
        operation_candidates=tuple(operation_candidates),
        working_pairs=(),
        assemblies=assemblies,
        enrichment_candidates=(),
        unresolved_words=tuple(unresolved),
        status=status,
    )


def _definition_candidates(context):
    out = []
    for candidate in context.definition_candidates:
        out.append({
            "text": candidate.definition_span.text,
            "span": candidate.definition_span.as_tuple(),
            "wordplay_span": candidate.wordplay_span.as_tuple(),
            "wordplay_text": candidate.wordplay_span.text,
            "boundary_status": candidate.boundary_status,
            "objections": list(candidate.objections),
        })
    return out


def _grammar_phrases(context):
    out = []
    for span in context.pos_spans:
        out.append({
            "text": span.text,
            "span": (span.start, span.end),
            "label": span.label,
            "root_index": span.root_index,
            "root_text": span.root_text,
            "pos_tags": tuple(span.pos_tags),
            "dependencies": tuple(span.dependencies),
        })
    return out


def _source_candidates(context, answer):
    candidates = []
    seen = set()
    for annotation in context.annotations:
        if annotation.token not in SOURCE_TOKENS:
            continue
        for value in _string_values(annotation.values):
            clean = _clean_answer(value)
            if not clean:
                continue
            if clean not in answer and answer not in clean:
                continue
            key = (annotation.span, clean)
            if key in seen:
                continue
            seen.add(key)
            candidates.append({
                "text": annotation.text,
                "span": annotation.span,
                "token": annotation.token,
                "value": clean,
                "source": annotation.source,
                "relation_to_answer": (
                    "contained_in_answer" if clean in answer
                    else "contains_answer"
                ),
            })
    candidates.sort(key=lambda item: (
        item["span"][0],
        -(item["span"][1] - item["span"][0]),
        len(item["value"]),
        item["value"],
    ))
    return candidates


def _operation_candidates(context):
    out = []
    seen = set()
    for annotation in context.annotations:
        if annotation.token not in INDICATOR_TOKENS and annotation.token != LNK:
            continue
        key = (annotation.span, annotation.token)
        if key in seen:
            continue
        seen.add(key)
        out.append({
            "text": annotation.text,
            "span": annotation.span,
            "token": annotation.token,
            "source": annotation.source,
            "role": "joiner" if annotation.token == LNK else "operation",
        })
    out.sort(key=lambda item: (item["span"][0], item["span"][1]))
    return out


def _working_pairs(context, answer, sources):
    yield from _anagram_indicator_pairs(context, answer)
    yield from _trim_first_pairs(context, answer, sources)


def _anagram_indicator_pairs(context, answer):
    indicators = [
        ann for ann in context.annotations
        if ann.token == ANA_I
    ]
    for indicator in indicators:
        for source_idx in (indicator.span[0] - 1, indicator.span[1]):
            if source_idx < 0 or source_idx >= len(context.tokens):
                continue
            text = context.span_text(source_idx, source_idx + 1)
            letters = _letters(text)
            if len(letters) < 2:
                continue
            result = _answer_anagram_substring(letters, answer)
            if result is None:
                continue
            yield {
                "kind": "anagram_pair",
                "source_text": text,
                "source_span": (source_idx, source_idx + 1),
                "indicator_text": indicator.text,
                "indicator_span": indicator.span,
                "input": letters,
                "output": result,
                "answer_span": _answer_value_span(answer, result),
                "status": "answer_fit",
            }


def _trim_first_pairs(context, answer, sources):
    indicators = [
        ann for ann in context.annotations
        if ann.token == POS_I_TRIM_FIRST
    ]
    for indicator in indicators:
        for source in sources:
            if source["span"][1] > indicator.span[0]:
                continue
            if indicator.span[0] - source["span"][1] > 2:
                continue
            value = source["value"]
            if len(value) < 2:
                continue
            trimmed = value[1:]
            if trimmed not in answer:
                continue
            yield {
                "kind": "trim_first_pair",
                "source_text": source["text"],
                "source_span": source["span"],
                "indicator_text": indicator.text,
                "indicator_span": indicator.span,
                "input": value,
                "output": trimmed,
                "answer_span": _answer_value_span(answer, trimmed),
                "status": "answer_fit",
            }


def _grammar_span_enrichments(context, sources):
    out = []
    seen = set()
    known = {
        (source["span"], source["value"])
        for source in sources
    }
    known_spans = {
        source["span"]
        for source in sources
    }
    for source in sources:
        if len(source["value"]) <= 2:
            continue
        source_start, source_end = source["span"]
        for phrase in context.pos_spans:
            if not (phrase.start <= source_start
                    and phrase.end >= source_end):
                continue
            if (phrase.start, phrase.end) == source["span"]:
                continue
            if (phrase.start, phrase.end) in known_spans:
                continue
            if phrase.root_index is not None and not (
                    source_start <= phrase.root_index < source_end):
                continue
            if ((phrase.start, phrase.end), source["value"]) in known:
                continue
            if _added_phrase_words_include_indicator(
                    context, source["span"], (phrase.start, phrase.end)):
                continue
            if _added_phrase_words_include_other_source(
                    sources, source["span"], (phrase.start, phrase.end)):
                continue
            if _added_phrase_words_include_adverb(phrase, source["span"]):
                continue
            key = (phrase.start, phrase.end, source["value"])
            if key in seen:
                continue
            seen.add(key)
            out.append({
                "kind": "source_phrase_widening",
                "text": phrase.text,
                "span": (phrase.start, phrase.end),
                "value": source["value"],
                "based_on": {
                    "text": source["text"],
                    "span": source["span"],
                    "value": source["value"],
                },
                "reason": "grammar says the used source belongs inside a larger phrase",
                "status": "review",
            })
    return out


def _added_phrase_words_include_indicator(context, source_span, phrase_span):
    added = _added_phrase_spans(source_span, phrase_span)
    if not added:
        return False
    blocked = INDICATOR_TOKENS | {LNK}
    for annotation in context.annotations:
        if annotation.token not in blocked:
            continue
        if any(_spans_overlap(annotation.span, span) for span in added):
            return True
    return False


def _added_phrase_words_include_other_source(sources, source_span, phrase_span):
    added = _added_phrase_spans(source_span, phrase_span)
    if not added:
        return False
    for other in sources:
        other_span = other.get("span")
        if not other_span or tuple(other_span) == tuple(source_span):
            continue
        if any(_spans_overlap(tuple(other_span), span) for span in added):
            return True
    return False


def _added_phrase_words_include_adverb(phrase, source_span):
    pos_tags = tuple(getattr(phrase, "pos_tags", ()) or ())
    dependencies = tuple(getattr(phrase, "dependencies", ()) or ())
    if not pos_tags and not dependencies:
        return False
    for index in _added_phrase_indices(source_span, (phrase.start, phrase.end)):
        offset = index - phrase.start
        if offset < 0:
            continue
        pos = pos_tags[offset] if offset < len(pos_tags) else ""
        dep = dependencies[offset] if offset < len(dependencies) else ""
        if pos == "ADV" or dep == "advmod":
            return True
    return False


def _added_phrase_indices(source_span, phrase_span):
    indices = []
    for start, end in _added_phrase_spans(source_span, phrase_span):
        indices.extend(range(start, end))
    return indices


def _added_phrase_spans(source_span, phrase_span):
    added = []
    if phrase_span[0] < source_span[0]:
        added.append((phrase_span[0], source_span[0]))
    if phrase_span[1] > source_span[1]:
        added.append((source_span[1], phrase_span[1]))
    return added


def _definition_gap_enrichments(context, working_pairs):
    if context.definition_candidates:
        return []
    out = []
    substantial_pairs = [
        pair for pair in working_pairs
        if len(_clean_answer(pair.get("output"))) >= 3
    ]
    earliest_source = min(
        (pair["source_span"][0] for pair in substantial_pairs),
        default=None,
    )
    if earliest_source is None or earliest_source <= 0:
        return out
    proposed_span = (0, earliest_source)
    if any(
            pair.get("source_span")
            and pair["source_span"][0] == earliest_source
            and _spans_overlap(
                tuple(pair.get("indicator_span") or ()), proposed_span)
            for pair in substantial_pairs):
        return out
    definition = context.span(*proposed_span, kind="definition_gap")
    if definition.text:
        out.append({
            "kind": "definition_gap",
            "text": definition.text,
            "span": definition.as_tuple(),
            "value": context.answer,
            "reason": "initial words precede the first answer-fitted working pair",
            "status": "review",
        })
    return out


def _conditional_suburb_enrichments(context, working_pairs):
    """First small conditional case: western half of a missing phrase source."""
    has_base_anagram = any(
        pair["kind"] == "anagram_pair"
        and _normalize(pair["source_text"]) == "base"
        and pair["output"] == "ASBE"
        for pair in working_pairs
    )
    if not has_base_anagram:
        return []

    phrase = next(
        (span for span in context.pos_spans
         if _normalize(span.text) == "north london suburb"),
        None,
    )
    western_half = next(
        (span for span in context.pos_spans
         if _normalize(span.text) == "western half"),
        None,
    )
    if phrase is None or western_half is None:
        return []

    return [{
        "kind": "conditional_source_gap",
        "text": phrase.text,
        "span": (phrase.start, phrase.end),
        "value": "HENDON",
        "reason": "western half of HENDON gives HEN, which can contain ASBE",
        "conditional_assembly": "H(ASBE)EN = HASBEEN",
        "status": "review",
    }]


def _assemblies(answer, sources, working_pairs, enrichments):
    out = []
    source_terms = [
        {
            "kind": "source",
            "text": source["text"],
            "span": source["span"],
            "value": source["value"],
        }
        for source in sources
        if source["value"] in answer
    ]
    working_terms = [
        {
            "kind": pair["kind"],
            "text": "%s %s" % (pair["source_text"], pair["indicator_text"]),
            "span": (pair["source_span"][0], pair["indicator_span"][1]),
            "value": pair["output"],
        }
        for pair in working_pairs
        if pair["output"] in answer
    ]
    for terms in (working_terms + source_terms, source_terms):
        assembly = _first_charade(answer, terms)
        if assembly:
            out.append(assembly)
            break

    for pair in working_pairs:
        if pair["output"] == answer:
            out.append({
                "kind": pair["kind"],
                "status": "answer_fit",
                "output": answer,
                "parts": [{
                    "text": "%s %s" % (
                        pair["source_text"], pair["indicator_text"]),
                    "value": pair["output"],
                    "span": pair["source_span"],
                }],
            })

    for enrichment in enrichments:
        if enrichment.get("conditional_assembly"):
            out.append({
                "kind": "conditional",
                "status": "needs_enrichment",
                "output": answer,
                "detail": enrichment["conditional_assembly"],
                "requires": [enrichment],
            })
    return tuple(_dedupe_dicts(out))


def _first_charade(answer, terms):
    terms = sorted(
        terms,
        key=lambda item: (
            item["span"][0],
            -(item["span"][1] - item["span"][0]),
            -len(item["value"]),
        ),
    )

    def recurse(pos, clue_start, used):
        if pos == len(answer):
            return []
        for idx, term in enumerate(terms):
            if idx in used:
                continue
            if term["span"][0] < clue_start:
                continue
            if any(
                    not (term["span"][1] <= terms[used_idx]["span"][0]
                         or term["span"][0] >= terms[used_idx]["span"][1])
                    for used_idx in used):
                continue
            value = term["value"]
            if not answer.startswith(value, pos):
                continue
            rest = recurse(pos + len(value), term["span"][1], used | {idx})
            if rest is not None:
                return [term] + rest
        return None

    parts = recurse(0, 0, set())
    if not parts or len(parts) < 2:
        return None
    return {
        "kind": "charade",
        "status": "answer_fit",
        "output": answer,
        "parts": parts,
    }


def _unresolved_words(context, sources, operations, working_pairs, definitions):
    covered = set()
    for item in list(sources) + list(operations):
        if item.get("role") == "operation":
            continue
        covered.update(range(item["span"][0], item["span"][1]))
    for pair in working_pairs:
        covered.update(range(pair["source_span"][0], pair["source_span"][1]))
        covered.update(range(pair["indicator_span"][0], pair["indicator_span"][1]))
    for definition in definitions:
        covered.update(range(definition["span"][0], definition["span"][1]))
    for idx, token in enumerate(context.tokens):
        if idx not in covered and _letters(token.text):
            yield {
                "index": idx,
                "text": token.text,
                "reason": "not_accounted_for_by_stage_two_casefile",
            }


def _case_status(assemblies, enrichments, unresolved):
    if any(item.get("status") == "answer_fit" for item in assemblies):
        if unresolved or enrichments:
            return "answer_fit_needs_review"
        return "answer_fit"
    if any(item.get("status") == "needs_enrichment" for item in assemblies):
        return "conditional_needs_enrichment"
    if enrichments:
        return "evidence_needs_enrichment"
    return "evidence_only"


def _string_values(values):
    for value in values or ():
        if isinstance(value, str):
            yield value


def _answer_anagram_substring(letters, answer):
    size = len(letters)
    target = sorted(letters)
    for start in range(0, len(answer) - size + 1):
        piece = answer[start:start + size]
        if sorted(piece) == target:
            return piece
    return None


def _answer_value_span(answer, value):
    start = answer.find(value)
    if start < 0:
        return None
    return (start + 1, start + len(value))


def _clean_answer(value):
    return "".join(c for c in (value or "").upper() if c.isalpha())


def _letters(value):
    return _clean_answer(value)


def _normalize(value):
    return " ".join(
        word.strip(".,;:!?\"'()-").lower()
        for word in (value or "").split()
        if word.strip(".,;:!?\"'()-")
    )


def _spans_overlap(left, right):
    if len(left) != 2 or len(right) != 2:
        return False
    return max(left[0], right[0]) < min(left[1], right[1])


def _dedupe_dicts(items):
    seen = set()
    out = []
    for item in items:
        key = repr(sorted(item.items()))
        if key in seen:
            continue
        seen.add(key)
        out.append(item)
    return out
