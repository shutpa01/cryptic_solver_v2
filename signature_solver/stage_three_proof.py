"""Stage Three proof gate for Stage Two case files.

Stage Three is read-only.  It does not solve from scratch, write enrichment, or
publish old WFW rows.  It turns a Stage Two evidence package into an admin-only
PASS/REVIEW verdict with named checks and answer-letter placements.
"""
from __future__ import annotations

from dataclasses import dataclass


PASS = "PASS"
REVIEW = "REVIEW"


@dataclass(frozen=True)
class StageThreeCheck:
    name: str
    status: str
    detail: str
    evidence: object | None = None

    def as_dict(self):
        out = {
            "name": self.name,
            "status": self.status,
            "detail": self.detail,
        }
        if self.evidence is not None:
            out["evidence"] = self.evidence
        return out


@dataclass(frozen=True)
class StageThreeProof:
    clue_text: str
    answer: str
    status: str
    checks: tuple[StageThreeCheck, ...]
    blocks: tuple[dict, ...]
    atomic_links: tuple[dict, ...]
    transformations: tuple[dict, ...]
    word_purposes: tuple[dict, ...]
    purpose_requests: tuple[dict, ...]
    unresolved_items: tuple[dict, ...]
    required_enrichments: tuple[dict, ...]
    source_candidates: tuple[dict, ...] = ()
    operation_candidates: tuple[dict, ...] = ()

    def as_dict(self):
        return {
            "schema": "stage_three_proof:v1",
            "clue_text": self.clue_text,
            "answer": self.answer,
            "status": self.status,
            "checks": [check.as_dict() for check in self.checks],
            "blocks": list(self.blocks),
            "atomic_links": list(self.atomic_links),
            "transformations": list(self.transformations),
            "word_purposes": list(self.word_purposes),
            "purpose_requests": list(self.purpose_requests),
            "unresolved_items": list(self.unresolved_items),
            "required_enrichments": list(self.required_enrichments),
            "source_candidates": list(self.source_candidates),
            "operation_candidates": list(self.operation_candidates),
        }


def build_stage_three_proof(casefile):
    """Return a PASS/REVIEW proof object for a StageTwoCaseFile."""
    answer = _clean_answer(casefile.answer)
    manual_assembly = getattr(casefile, "manual_assembly", None)
    assembly = manual_assembly or _best_answer_fit_assembly(
        casefile.assemblies, answer)
    # Preserve Stage Two source and operation candidates in the proof.
    # A source candidate is "used" only when it matches a selected assembly
    # part by span and value; if both sides have tokens, the tokens must match.
    _selected_parts = []
    if assembly:
        for _part in assembly.get("parts") or []:
            _sp = _part.get("span")
            _val = _clean_answer(_part.get("value") or "")
            _tok = _part.get("token")
            if _sp and len(_sp) == 2 and _val:
                _selected_parts.append((_sp[0], _sp[1], _val, _tok))

    _source_candidates_in = tuple(
        getattr(casefile, "source_candidates", ()) or ())
    _source_candidates_out = []
    for _sc in _source_candidates_in:
        _sc_sp = _sc.get("span")
        _sc_val = _clean_answer(_sc.get("value") or "")
        _sc_tok = _sc.get("token")
        _used = False
        if _sc_sp and len(_sc_sp) == 2 and _sc_val:
            for _p0, _p1, _p_val, _p_tok in _selected_parts:
                if _sc_sp[0] != _p0 or _sc_sp[1] != _p1 or _sc_val != _p_val:
                    continue
                if _sc_tok and _p_tok and _sc_tok != _p_tok:
                    continue
                _used = True
                break
        _source_candidates_out.append(
            dict(_sc, assembly_status="used" if _used else "candidate"))
    _source_candidates_out = tuple(_source_candidates_out)
    _operation_candidates_out = tuple(
        getattr(casefile, "operation_candidates", ()) or ())
    conditional_assemblies = tuple(
        item for item in casefile.assemblies
        if item.get("status") == "needs_enrichment"
        or item.get("kind") == "conditional"
    )
    enrichments = tuple(casefile.enrichment_candidates)
    unresolved = tuple(casefile.unresolved_words)
    _manual_roles = tuple(
        getattr(casefile, "manual_roles", None) or ()
    )
    _manual_def_candidates = tuple(
        getattr(casefile, "manual_definition_candidates", None) or ()
    )

    blocks = tuple(_blocks(casefile, assembly, conditional_assemblies))
    links = tuple(_atomic_links(answer, assembly))
    transformations = tuple(_transformations(casefile))
    word_purposes = tuple(
        _word_purposes(casefile, blocks, enrichments, _manual_roles))
    purpose_requests = tuple(_purpose_requests(word_purposes))
    checks = (
        _definition_check(casefile, _manual_def_candidates),
        _assembly_check(answer, assembly, conditional_assemblies),
        _source_check(casefile, assembly),
        _assembly_order_check(assembly),
        _operation_check(casefile),
        _operation_attachment_check(casefile),
        _mechanism_rules_check(casefile),
        _atomic_coverage_check(answer, links, assembly),
        _span_integrity_check(casefile, assembly),
        _word_purpose_check(word_purposes),
        _word_purpose_candidate_check(word_purposes),
        _conditional_facts_check(enrichments, conditional_assemblies),
    )
    status = PASS if all(check.status == PASS for check in checks) else REVIEW
    return StageThreeProof(
        clue_text=casefile.clue_text,
        answer=answer,
        status=status,
        checks=checks,
        blocks=blocks,
        atomic_links=links,
        transformations=transformations,
        word_purposes=word_purposes,
        purpose_requests=purpose_requests,
        unresolved_items=unresolved,
        required_enrichments=enrichments,
        source_candidates=_source_candidates_out,
        operation_candidates=_operation_candidates_out,
    )


def _definition_check(casefile, extra_candidates=()):
    all_candidates = list(casefile.definition_candidates) + list(extra_candidates)
    accepted = [
        item for item in all_candidates
        if _accepted_definition_candidate(item)
    ]
    if accepted:
        return StageThreeCheck(
            "definition_evidence",
            PASS,
            "accepted definition-answer evidence found",
            accepted,
        )
    gap_candidates = [
        item for item in all_candidates
        if item.get("boundary_status") == "manual_definition_gap"
    ]
    if gap_candidates:
        gaps = "; ".join(
            "%s -> %s not in DB" % (
                item.get("text", "?"), item.get("missing_answer", "?"))
            for item in gap_candidates
        )
        return StageThreeCheck(
            "definition_evidence",
            REVIEW,
            "definition gap: %s - add to definition_answers_augmented" % gaps,
            gap_candidates,
        )
    if all_candidates:
        return StageThreeCheck(
            "definition_evidence",
            REVIEW,
            "definition candidates exist but need stronger boundary evidence",
            all_candidates,
        )
    return StageThreeCheck(
        "definition_evidence",
        REVIEW,
        "no accepted definition-answer evidence found",
    )


def _accepted_definition_candidate(item):
    if item.get("objections"):
        return False
    if (item.get("boundary_status") == "legacy_solver"
            and item.get("span_status") == "mapped"
            and item.get("span")):
        return _span_is_edge(item.get("span"), item.get("clue_word_count"))
    return item.get("boundary_status") in {
        "complete_edge_phrase",
        "edge_db_hit_no_larger_pos_phrase",
        "manual_definition",
    }


def _span_is_edge(span, clue_word_count):
    if not span or len(span) != 2:
        return False
    start, end = span
    if start == 0:
        return True
    return clue_word_count is not None and end == clue_word_count


def _assembly_check(answer, assembly, conditional_assemblies):
    if assembly and assembly.get("status") == "manual_fit":
        if _clean_answer(assembly.get("output", "")) == answer:
            return StageThreeCheck(
                "answer_assembly",
                PASS,
                "%s (manual) = %s" % (assembly.get("output", ""), answer),
                assembly,
            )
        return StageThreeCheck(
            "answer_assembly",
            REVIEW,
            "manual assembly output %r does not match answer %r" % (
                assembly.get("output", ""), answer),
        )
    if assembly:
        values = "".join(part.get("value", "") for part in assembly.get("parts", ()))
        clean_values = _clean_answer(values)
        if assembly.get("kind") == "anagram":
            fits = sorted(clean_values) == sorted(answer)
        else:
            fits = clean_values == answer
        if fits:
            if assembly.get("kind") == "anagram":
                detail = "%s (anagram) = %s" % (clean_values, answer)
            else:
                detail = "%s = %s" % (
                    " + ".join(
                        part.get("value", "")
                        for part in assembly["parts"]),
                    answer,
                )
            return StageThreeCheck(
                "answer_assembly",
                PASS,
                detail,
                assembly,
            )
    if conditional_assemblies:
        return StageThreeCheck(
            "answer_assembly",
            REVIEW,
            "answer assembly is conditional on accepted enrichment",
            list(conditional_assemblies),
        )
    return StageThreeCheck(
        "answer_assembly",
        REVIEW,
        "no complete verified answer assembly found",
    )


def _source_check(casefile, assembly):
    if not assembly:
        _candidates = tuple(
            getattr(casefile, "source_candidates", ()) or ())
        if _candidates:
            _summary = "; ".join(
                "%s -> %s [%s]" % (
                    c.get("text", "?"),
                    c.get("value", "?"),
                    c.get("evidence_status") or "unknown",
                )
                for c in _candidates[:6]
            )
            return StageThreeCheck(
                "source_evidence",
                REVIEW,
                "no complete assembly found; source candidates exist but "
                "were not accepted as a complete assembly: %s" % _summary,
                list(_candidates),
            )
        return StageThreeCheck(
            "source_evidence",
            REVIEW,
            "no complete assembly source list to verify",
        )
    if assembly.get("status") == "manual_fit":
        return StageThreeCheck(
            "source_evidence",
            PASS,
            "manual assembly mechanically verified by assembly builder",
            list(assembly.get("parts", ())),
        )
    missing = [
        part for part in assembly.get("parts", ())
        if not part.get("text") or not part.get("value")
    ]
    if missing:
        return StageThreeCheck(
            "source_evidence",
            REVIEW,
            "one or more assembly parts lack source text or value",
            missing,
        )
    unverified = [
        part for part in assembly.get("parts", ())
        if part.get("evidence_status") is not None
        and part.get("evidence_status") != "verified"
    ]
    if unverified:
        return StageThreeCheck(
            "source_evidence",
            REVIEW,
            "one or more assembly parts lack verified source derivation",
            unverified,
        )
    unsupported = [
        part for part in assembly.get("parts", ())
        if not _part_has_stage_two_evidence(casefile, part)
    ]
    if unsupported:
        return StageThreeCheck(
            "source_evidence",
            REVIEW,
            "one or more assembly parts are not backed by Stage Two evidence",
            unsupported,
        )
    return StageThreeCheck(
        "source_evidence",
        PASS,
        "all assembly parts are backed by Stage Two source evidence",
        list(assembly.get("parts", ())),
    )


def _part_has_stage_two_evidence(casefile, part):
    sources = tuple(getattr(casefile, "source_candidates", ()) or ())
    pairs = tuple(getattr(casefile, "working_pairs", ()) or ())
    if not sources and not pairs:
        return True
    span = tuple(part.get("span") or ())
    value = _clean_answer(part.get("value"))
    text = _normalize(part.get("text"))
    for source in sources:
        if tuple(source.get("span") or ()) != span:
            continue
        if _clean_answer(source.get("value")) != value:
            continue
        return True
    for pair in pairs:
        pair_span = _pair_combined_span(pair)
        if pair_span and pair_span == span and _clean_answer(
                pair.get("output")) == value:
            return True
        if tuple(pair.get("source_span") or ()) == span and _clean_answer(
                pair.get("output")) == value:
            return True
        if text == _normalize("%s %s" % (
                pair.get("source_text", ""), pair.get("indicator_text", "")
        )) and _clean_answer(pair.get("output")) == value:
            return True
    return False


def _assembly_order_check(assembly):
    if not assembly:
        return StageThreeCheck(
            "assembly_order",
            REVIEW,
            "no assembly available for order verification",
        )
    if assembly.get("kind") != "charade":
        return StageThreeCheck(
            "assembly_order",
            PASS,
            "assembly does not require left-to-right charade order",
            assembly,
        )
    spans = [
        tuple(part.get("span") or ())
        for part in assembly.get("parts", ())
    ]
    bad_spans = [
        span for span in spans
        if len(span) != 2 or span[0] is None or span[1] is None
    ]
    if bad_spans:
        return StageThreeCheck(
            "assembly_order",
            REVIEW,
            "one or more charade parts lack a stable clue span",
            bad_spans,
        )
    out_of_order = []
    for idx, span in enumerate(spans[1:], start=1):
        previous = spans[idx - 1]
        if span[0] < previous[1]:
            out_of_order.append({
                "previous_span": previous,
                "span": span,
            })
    if out_of_order:
        return StageThreeCheck(
            "assembly_order",
            REVIEW,
            "charade pieces are not justified by left-to-right clue order",
            out_of_order,
        )
    return StageThreeCheck(
        "assembly_order",
        PASS,
        "charade pieces follow left-to-right clue order",
        assembly,
    )


def _pair_combined_span(pair):
    source_span = pair.get("source_span")
    indicator_span = pair.get("indicator_span")
    if (not source_span or len(source_span) != 2
            or not indicator_span or len(indicator_span) != 2):
        return None
    return (
        min(source_span[0], indicator_span[0]),
        max(source_span[1], indicator_span[1]),
    )


def _operation_check(casefile):
    if not casefile.working_pairs:
        return StageThreeCheck(
            "operation_evidence",
            PASS,
            "no operation pair required by the selected evidence",
        )
    unsupported = [
        pair for pair in casefile.working_pairs
        if not pair.get("source_text")
        or not pair.get("indicator_text")
        or not pair.get("output")
    ]
    if unsupported:
        return StageThreeCheck(
            "operation_evidence",
            REVIEW,
            "one or more working pairs lack source, indicator, or output",
            unsupported,
        )
    return StageThreeCheck(
        "operation_evidence",
        PASS,
        "working source/indicator pairs are preserved",
        list(casefile.working_pairs),
    )


def _operation_attachment_check(casefile):
    pairs = tuple(casefile.working_pairs)
    if not pairs:
        return StageThreeCheck(
            "operation_attachment",
            PASS,
            "no transformed working pair requires attachment verification",
        )
    objections = []
    for pair in pairs:
        objection = _operation_attachment_objection(pair)
        if objection:
            objections.append(objection)
    if objections:
        return StageThreeCheck(
            "operation_attachment",
            REVIEW,
            "one or more working-pair indicators are not securely attached",
            objections,
        )
    return StageThreeCheck(
        "operation_attachment",
        PASS,
        "working-pair indicators are securely attached to their sources",
        list(pairs),
    )


def _operation_attachment_objection(pair):
    source_span = tuple(pair.get("source_span") or ())
    indicator_span = tuple(pair.get("indicator_span") or ())
    if not _valid_span(source_span) or not _valid_span(indicator_span):
        return {
            "kind": pair.get("kind"),
            "reason": "missing stable source or indicator span",
            "pair": pair,
        }
    if _spans_overlap(source_span, indicator_span):
        return {
            "kind": pair.get("kind"),
            "reason": "source and indicator spans overlap",
            "pair": pair,
        }
    gap = _span_gap(source_span, indicator_span)
    kind = pair.get("kind")
    if kind == "anagram_pair" and gap > 0:
        return {
            "kind": kind,
            "reason": "anagram indicator is not adjacent to fodder",
            "source_span": source_span,
            "indicator_span": indicator_span,
            "pair": pair,
        }
    if kind == "trim_first_pair":
        if indicator_span[0] < source_span[1]:
            return {
                "kind": kind,
                "reason": "trim-first indicator should follow the source",
                "source_span": source_span,
                "indicator_span": indicator_span,
                "pair": pair,
            }
        if gap > 1:
            return {
                "kind": kind,
                "reason": "trim-first indicator is too far from the source",
                "source_span": source_span,
                "indicator_span": indicator_span,
                "pair": pair,
            }
    return None


def _mechanism_rules_check(casefile):
    pairs = tuple(casefile.working_pairs)
    if not pairs:
        return StageThreeCheck(
            "mechanism_rules",
            PASS,
            "no transformed working pair requires mechanism verification",
        )
    objections = []
    for pair in pairs:
        objection = _mechanism_objection(pair)
        if objection:
            objections.append(objection)
    if objections:
        return StageThreeCheck(
            "mechanism_rules",
            REVIEW,
            "one or more working-pair mechanisms failed strict verification",
            objections,
        )
    return StageThreeCheck(
        "mechanism_rules",
        PASS,
        "working-pair mechanisms are strictly verified",
        list(pairs),
    )


def _mechanism_objection(pair):
    kind = pair.get("kind")
    source = pair.get("source_text")
    indicator = pair.get("indicator_text")
    input_value = _clean_answer(pair.get("input"))
    output = _clean_answer(pair.get("output"))
    if not source or not indicator or not input_value or not output:
        return {
            "kind": kind,
            "reason": "missing source, indicator, input, or output",
            "pair": pair,
        }
    if kind == "anagram_pair":
        if sorted(input_value) != sorted(output):
            return {
                "kind": kind,
                "reason": "output is not an anagram of input",
                "input": input_value,
                "output": output,
                "pair": pair,
            }
        return None
    if kind == "trim_first_pair":
        if input_value[1:] != output:
            return {
                "kind": kind,
                "reason": "output is not input with first letter removed",
                "input": input_value,
                "output": output,
                "pair": pair,
            }
        return None
    return {
        "kind": kind,
        "reason": "mechanism kind is not yet verified by Stage Three",
        "pair": pair,
    }


def _atomic_coverage_check(answer, links, assembly):
    if not assembly:
        return StageThreeCheck(
            "atomic_coverage",
            REVIEW,
            "no answer-fit assembly available for answer-letter placement",
        )
    if len(links) != len(answer):
        return StageThreeCheck(
            "atomic_coverage",
            REVIEW,
            "answer letters are not fully covered by placed pieces",
            list(links),
        )
    made = "".join(link["letter"] for link in links)
    if made != answer:
        return StageThreeCheck(
            "atomic_coverage",
            REVIEW,
            "placed letters do not match the answer",
            list(links),
        )
    return StageThreeCheck(
        "atomic_coverage",
        PASS,
        "all answer letters are placed from preserved evidence",
        list(links),
    )


def _span_integrity_check(casefile, assembly):
    if not assembly:
        return StageThreeCheck(
            "span_integrity",
            REVIEW,
            "no assembly spans available to check",
        )
    definition_spans = [
        tuple(item.get("span") or ())
        for item in casefile.definition_candidates
        if _accepted_definition_candidate(item)
    ]
    part_spans = [
        tuple(part.get("span") or ())
        for part in assembly.get("parts", ())
    ]
    bad_spans = [
        span for span in part_spans
        if len(span) != 2 or span[0] is None or span[1] is None
    ]
    if bad_spans:
        return StageThreeCheck(
            "span_integrity",
            REVIEW,
            "one or more assembly parts lack a stable clue span",
            bad_spans,
        )
    overlaps_definition = [
        {
            "part_span": part_span,
            "definition_span": definition_span,
        }
        for part_span in part_spans
        for definition_span in definition_spans
        if _spans_overlap(part_span, definition_span)
    ]
    if overlaps_definition:
        return StageThreeCheck(
            "span_integrity",
            REVIEW,
            "source material overlaps the definition span",
            overlaps_definition,
        )
    overlaps_source = []
    for left_i, left in enumerate(part_spans):
        for right in part_spans[left_i + 1:]:
            if _spans_overlap(left, right):
                overlaps_source.append({
                    "left_span": left,
                    "right_span": right,
                })
    if overlaps_source:
        return StageThreeCheck(
            "span_integrity",
            REVIEW,
            "assembly source spans overlap each other",
            overlaps_source,
        )
    return StageThreeCheck(
        "span_integrity",
        PASS,
        "definition and source spans are stable and non-overlapping",
    )


def _word_purpose_check(word_purposes):
    unresolved = [
        item for item in word_purposes
        if item.get("purpose") == "unresolved_purpose"
        and item.get("status") == "unresolved"
    ]
    if unresolved:
        return StageThreeCheck(
            "word_purpose_coverage",
            REVIEW,
            "some clue words still need a grammatical or cryptic purpose",
            unresolved,
        )
    return StageThreeCheck(
        "word_purpose_coverage",
        PASS,
        "every clue word has a recorded purpose",
        list(word_purposes),
    )


def _word_purpose_candidate_check(word_purposes):
    candidates = [
        item for item in word_purposes
        if item.get("status") == "candidate"
    ]
    if candidates:
        return StageThreeCheck(
            "word_purpose_candidates",
            REVIEW,
            "some clue word purposes still depend on candidate evidence",
            candidates,
        )
    return StageThreeCheck(
        "word_purpose_candidates",
        PASS,
        "no clue word purpose depends on candidate evidence",
    )


def _purpose_requests(word_purposes):
    requests = []
    for item in word_purposes:
        if item.get("status") in ("verified", "manual"):
            continue
        purpose = item.get("purpose")
        request = _purpose_request_for(item)
        if request:
            requests.append({
                "kind": request["kind"],
                "text": item.get("text"),
                "span": (item.get("index"), item.get("index", 0) + 1),
                "purpose": purpose,
                "status": item.get("status"),
                "needed_evidence": request["needed_evidence"],
                "reason": request["reason"],
                "evidence": item.get("evidence") or [],
                "atoms": [{
                    "index": item.get("index"),
                    "text": item.get("text"),
                    "purpose": purpose,
                    "status": item.get("status"),
                }],
            })
    yield from _consolidated_purpose_requests(requests)


def _consolidated_purpose_requests(requests):
    if not requests:
        return
    current = None
    for request in requests:
        if current is None:
            current = dict(request)
            continue
        if _can_merge_purpose_request(current, request):
            current["text"] = "%s %s" % (current["text"], request["text"])
            current["span"] = (current["span"][0], request["span"][1])
            current["purpose"] = _merged_purpose_label(
                current["purpose"], request["purpose"])
            current["status"] = _merged_status(
                current["status"], request["status"])
            current["evidence"] = list(current.get("evidence") or []) + list(
                request.get("evidence") or [])
            current["atoms"] = list(current.get("atoms") or []) + list(
                request.get("atoms") or [])
            continue
        yield current
        current = dict(request)
    if current is not None:
        yield current


def _can_merge_purpose_request(left, right):
    return (
        left.get("kind") == right.get("kind")
        and left.get("needed_evidence") == right.get("needed_evidence")
        and left.get("reason") == right.get("reason")
        and left.get("span")
        and right.get("span")
        and left["span"][1] == right["span"][0]
    )


def _merged_purpose_label(left, right):
    if left == right:
        return left
    labels = []
    for value in (left, right):
        for label in str(value).split("+"):
            if label and label not in labels:
                labels.append(label)
    return "+".join(labels)


def _merged_status(left, right):
    if left == right:
        return left
    if "unresolved" in (left, right):
        return "unresolved"
    return "candidate"


def _purpose_request_for(item):
    purpose = item.get("purpose")
    if purpose == "operation_indicator_candidate":
        return {
            "kind": "mechanism_indicator_evidence",
            "needed_evidence": (
                "accepted mechanism showing this word controls a real "
                "letter operation"
            ),
            "reason": (
                "indicator-looking words do not pass unless attached to a "
                "verified source and output"
            ),
        }
    if purpose == "operation_indicator_modifier_candidate":
        return {
            "kind": "mechanism_indicator_evidence",
            "needed_evidence": (
                "accepted mechanism showing this modifier is part of the "
                "indicator phrase"
            ),
            "reason": (
                "grammar can attach a modifier to an indicator head, but the "
                "mechanism still needs proof"
            ),
        }
    if purpose == "structural_separator_candidate":
        return {
            "kind": "grammar_separator_evidence",
            "needed_evidence": (
                "grammar evidence showing this word separates definition and "
                "wordplay, or joins two verified wordplay pieces"
            ),
            "reason": (
                "separator words are grammatical jobs, not generic link-word "
                "allowances"
            ),
        }
    if purpose == "definition_separator_candidate":
        return {
            "kind": "definition_separator_evidence",
            "needed_evidence": (
                "grammar evidence showing this word separates definition "
                "from wordplay"
            ),
            "reason": (
                "definition separators are grammatical boundary jobs, not "
                "letter mechanisms"
            ),
        }
    if purpose == "definition_phrase_candidate":
        return {
            "kind": "definition_phrase_evidence",
            "needed_evidence": (
                "accepted definition evidence for the whole phrase, not just "
                "a plausible edge"
            ),
            "reason": "candidate definition phrases cannot override the DB gate",
        }
    if purpose == "phrase_widening_candidate":
        return {
            "kind": "source_phrase_evidence",
            "needed_evidence": (
                "accepted synonym/source evidence for the widened phrase"
            ),
            "reason": (
                "grammar can suggest a wider phrase, but Stage Three needs "
                "the phrase-to-value fact"
            ),
        }
    if purpose == "conditional_source_candidate":
        return {
            "kind": "conditional_source_evidence",
            "needed_evidence": (
                "accepted source fact for the conditional phrase before the "
                "assembly can pass"
            ),
            "reason": "conditional source facts must be enriched upstream",
        }
    if purpose == "unresolved_purpose":
        return {
            "kind": "word_purpose_evidence",
            "needed_evidence": (
                "a verified cryptic or grammatical purpose for this clue word"
            ),
            "reason": (
                "Stage Three cannot treat clue words as disposable surface text"
            ),
        }
    return None


def _conditional_facts_check(enrichments, conditional_assemblies):
    if conditional_assemblies or enrichments:
        return StageThreeCheck(
            "conditional_facts",
            REVIEW,
            "reviewed enrichment is required before this can pass",
            {
                "assemblies": list(conditional_assemblies),
                "enrichments": list(enrichments),
            },
        )
    return StageThreeCheck(
        "conditional_facts",
        PASS,
        "no conditional facts are required",
    )


def _best_answer_fit_assembly(assemblies, answer):
    for assembly in assemblies:
        if assembly.get("status") != "answer_fit":
            continue
        if _clean_answer(assembly.get("output")) != answer:
            continue
        if assembly.get("parts"):
            return assembly
    return None


def _blocks(casefile, assembly, conditional_assemblies):
    source_spans = set()
    operation_spans = set()
    _defs = tuple(
        item for item in (getattr(casefile, "definition_candidates", ()) or ())
        if _accepted_definition_candidate(item)
    )
    _pairs = tuple(getattr(casefile, "working_pairs", ()) or ())
    _ops = tuple(getattr(casefile, "operation_candidates", ()) or ())
    _unresolved = tuple(getattr(casefile, "unresolved_words", ()) or ())
    _manual_def_spans = set()
    for _mc in (getattr(casefile, "manual_definition_candidates", None) or ()):
        if _mc.get("boundary_status") != "manual_definition":
            continue
        _mc_span = _mc.get("span")
        if not _mc_span:
            continue
        yield {
            "kind": "DEF_BLOCK",
            "text": _mc.get("text", ""),
            "span": _mc_span,
            "value": casefile.answer,
            "status": "manual",
        }
        _manual_def_spans.add(tuple(_mc_span))
    if not _manual_def_spans:
        for definition in _defs:
            yield {
                "kind": "DEF_BLOCK",
                "text": definition["text"],
                "span": definition["span"],
                "value": casefile.answer,
                "status": "verified",
            }
    if assembly:
        for idx, part in enumerate(assembly.get("parts", ())):
            yield {
                "kind": "SOURCE_BLOCK",
                "role": "piece_%d" % idx,
                "text": part.get("text", ""),
                "span": part.get("span"),
                "value": part.get("value", ""),
                "token": part.get("token"),
                "derivation_kind": part.get("derivation_kind"),
                "evidence_status": part.get("evidence_status"),
                "evidence_reason": part.get("evidence_reason"),
                "container_role": part.get("container_role"),
                "status": "verified",
            }
            if part.get("span"):
                source_spans.add(tuple(part.get("span")))
    for pair in _pairs:
        source_span = pair.get("source_span")
        if source_span and tuple(source_span) not in source_spans:
            yield {
                "kind": "SOURCE_BLOCK",
                "role": "source",
                "text": pair.get("source_text", ""),
                "span": source_span,
                "value": pair.get("output", ""),
                "input_value": pair.get("input", ""),
                "status": pair.get("status"),
            }
            source_spans.add(tuple(source_span))
        yield {
            "kind": "OP_BLOCK",
            "role": pair.get("kind"),
            "text": pair.get("indicator_text", ""),
            "span": pair.get("indicator_span"),
            "value": pair.get("output", ""),
            "input_value": pair.get("input", ""),
            "source_text": pair.get("source_text", ""),
            "source_span": pair.get("source_span"),
            "status": "answer_fit",
        }
        if pair.get("indicator_span"):
            operation_spans.add(tuple(pair.get("indicator_span")))
    _anagram_assembly = (
        assembly is not None
        and assembly.get("kind") == "anagram"
        and assembly.get("status") == "answer_fit"
    )
    _selected_source_start = None
    _selected_source_end = None
    if _anagram_assembly:
        _part_spans = [
            p.get("span")
            for p in (assembly.get("parts") or [])
            if p.get("span") and len(p.get("span")) == 2
        ]
        if _part_spans:
            _selected_source_start = min(s[0] for s in _part_spans)
            _selected_source_end = max(s[1] for s in _part_spans)
    for operation in _ops:
        span = operation.get("span")
        if span and tuple(span) in operation_spans:
            continue
        _op_is_verified_ana_i = (
            _anagram_assembly
            and operation.get("token") == "ANA_I"
            and span is not None
            and len(span) == 2
            and _selected_source_start is not None
            and (
                span[1] == _selected_source_start
                or span[0] == _selected_source_end
            )
        )
        _op_status = (
            "verified"
            if _op_is_verified_ana_i
            else operation.get("span_status") or "candidate"
        )
        yield {
            "kind": "OP_BLOCK",
            "role": operation.get("role") or "operation",
            "text": operation.get("text", ""),
            "span": span,
            "token": operation.get("token"),
            "source": operation.get("source"),
            "status": _op_status,
        }
        if span:
            operation_spans.add(tuple(span))
    for item in conditional_assemblies:
        yield {
            "kind": "CONDITIONAL_ASSEMBLY_BLOCK",
            "text": item.get("detail", ""),
            "span": None,
            "value": item.get("output", ""),
            "status": "review",
            "requires": item.get("requires", ()),
        }
    for _mr in (getattr(casefile, "manual_roles", None) or ()):
        if _mr.get("source") != "manual":
            continue
        _mr_role = _mr.get("role", "")
        if _purpose_for_manual_role(_mr_role) != "operation_indicator":
            continue
        _mr_span = (_mr["index"], _mr["index"] + 1)
        if _mr_span in operation_spans:
            continue
        yield {
            "kind": "OP_BLOCK",
            "role": _mr_role,
            "text": _mr.get("text", ""),
            "span": list(_mr_span),
            "token": None,
            "source": "manual_role",
            "status": "verified",
        }
        operation_spans.add(_mr_span)

    _covered_indices = set()
    for _sp in source_spans:
        for _i in range(_sp[0], _sp[1]):
            _covered_indices.add(_i)
    for _sp in operation_spans:
        for _i in range(_sp[0], _sp[1]):
            _covered_indices.add(_i)
    for _sp in _manual_def_spans:
        for _i in range(_sp[0], _sp[1]):
            _covered_indices.add(_i)

    for item in _unresolved:
        if item.get("index") in _covered_indices:
            continue
        yield {
            "kind": "REVIEW_BLOCK",
            "role": "unresolved",
            "text": item.get("text", ""),
            "span": (item.get("index"), item.get("index", 0) + 1),
            "status": "review",
        }


def _atomic_links(answer, assembly):
    if not assembly:
        return ()
    if assembly.get("kind") == "mixed_anagram":
        _parts = list(assembly.get("parts", ()))
        _fixed_indexed = [
            (pi, p) for pi, p in enumerate(_parts)
            if p.get("token") != "ANA_F"
        ]
        _fodder_indexed = [
            (pi, p) for pi, p in enumerate(_parts)
            if p.get("token") == "ANA_F"
        ]
        _fixed_str = _clean_answer(
            "".join(p.get("value", "") for _, p in _fixed_indexed))

        _pool = []
        for _pi_overall, _part in _fodder_indexed:
            _val = _clean_answer(_part.get("value", ""))
            for _si, _letter in enumerate(_val):
                _pool.append({
                    "letter": _letter,
                    "source_text": _part.get("text", ""),
                    "source_span": _part.get("span"),
                    "source_role": "piece_%d" % _pi_overall,
                    "source_value": _val,
                    "source_value_index": _si,
                    "used": False,
                })

        flen = len(_fixed_str)
        if not _fixed_str:
            _fixed_answer_start = 0
            _fodder_answer_start = 0
            _fodder_portion = answer
            _fixed_indexed_to_emit = []
        elif answer[:flen] == _fixed_str:
            _fixed_answer_start = 0
            _fodder_answer_start = flen
            _fodder_portion = answer[flen:]
            _fixed_indexed_to_emit = _fixed_indexed
        elif answer[-flen:] == _fixed_str:
            _fixed_answer_start = len(answer) - flen
            _fodder_answer_start = 0
            _fodder_portion = answer[:-flen]
            _fixed_indexed_to_emit = _fixed_indexed
        else:
            return ()

        _links = []
        _ai = _fixed_answer_start
        for _pi_overall, _part in _fixed_indexed_to_emit:
            _val = _clean_answer(_part.get("value", ""))
            for _si, _letter in enumerate(_val):
                _links.append({
                    "answer_index": _ai,
                    "letter": _letter,
                    "source_text": _part.get("text", ""),
                    "source_span": _part.get("span"),
                    "source_role": "piece_%d" % _pi_overall,
                    "source_value": _val,
                    "source_value_index": _si,
                })
                _ai += 1

        for _offset, _aletter in enumerate(_fodder_portion):
            _matched = None
            for _entry in _pool:
                if not _entry["used"] and _entry["letter"] == _aletter:
                    _entry["used"] = True
                    _matched = _entry
                    break
            if _matched is None:
                return ()
            _links.append({
                "answer_index": _fodder_answer_start + _offset,
                "letter": _aletter,
                "source_text": _matched["source_text"],
                "source_span": _matched["source_span"],
                "source_role": _matched["source_role"],
                "source_value": _matched["source_value"],
                "source_value_index": _matched["source_value_index"],
            })

        _links.sort(key=lambda l: l["answer_index"])
        if len(_links) != len(answer):
            return ()
        return tuple(_links)

    if assembly.get("kind") != "anagram":
        links = []
        answer_index = 0
        for piece_index, part in enumerate(assembly.get("parts", ())):
            value = _clean_answer(part.get("value", ""))
            for source_index, letter in enumerate(value):
                links.append({
                    "answer_index": answer_index,
                    "letter": letter,
                    "source_text": part.get("text", ""),
                    "source_span": part.get("span"),
                    "source_role": "piece_%d" % piece_index,
                    "source_value": value,
                    "source_value_index": source_index,
                })
                answer_index += 1
        if answer_index != len(answer):
            return ()
        return tuple(links)

    # Anagram: match answer letters to source letters by letter value,
    # consuming from a pool so each source letter is used at most once.
    pool = []
    for piece_index, part in enumerate(assembly.get("parts", ())):
        value = _clean_answer(part.get("value", ""))
        for source_index, letter in enumerate(value):
            pool.append({
                "letter": letter,
                "source_text": part.get("text", ""),
                "source_span": part.get("span"),
                "source_role": "piece_%d" % piece_index,
                "source_value": value,
                "source_value_index": source_index,
                "used": False,
            })
    links = []
    for answer_index, answer_letter in enumerate(answer):
        matched = None
        for entry in pool:
            if not entry["used"] and entry["letter"] == answer_letter:
                entry["used"] = True
                matched = entry
                break
        if matched is None:
            return ()
        links.append({
            "answer_index": answer_index,
            "letter": answer_letter,
            "source_text": matched["source_text"],
            "source_span": matched["source_span"],
            "source_role": matched["source_role"],
            "source_value": matched["source_value"],
            "source_value_index": matched["source_value_index"],
        })
    return tuple(links)


def _transformations(casefile):
    for pair in casefile.working_pairs:
        yield {
            "kind": pair.get("kind"),
            "source_text": pair.get("source_text"),
            "source_span": pair.get("source_span"),
            "indicator_text": pair.get("indicator_text"),
            "indicator_span": pair.get("indicator_span"),
            "input": pair.get("input"),
            "output": pair.get("output"),
            "answer_span": pair.get("answer_span"),
            "status": pair.get("status"),
        }


def _manual_roles_by_index(manual_roles):
    """Return a dict mapping word index to manual role entry."""
    out = {}
    for entry in (manual_roles or ()):
        idx = entry.get("index")
        if idx is not None and idx not in out:
            out[idx] = entry
    return out


def _purpose_for_manual_role(role):
    """Map a clue_word_roles role string to a Stage Three purpose string."""
    if role in ("link", "surface", "charade_joiner"):
        return "structural_separator"
    if role == "definition":
        return "definition_phrase_member"
    if role in ("synonym", "synonym_source", "abbreviation",
                "abbreviation_source", "nato_phonetic",
                "literal_source", "letter_source", "roman_numeral",
                "single_letter", "positional_source", "reversal_source",
                "deletion_source", "hidden_source", "homophone_source",
                "container_frame", "container_content_source",
                "anagram_fodder"):
        return "answer_source"
    if role in ("anagram_indicator", "reversal_indicator",
                "container_indicator", "deletion_indicator",
                "hidden_indicator", "homophone_indicator",
                "first_letter_indicator", "last_letter_indicator",
                "letter_position_indicator", "alternating_indicator",
                "parts_indicator", "positional_indicator",
                "spoonerism_indicator"):
        return "operation_indicator"
    return None


def _word_purposes(casefile, blocks, enrichments, manual_roles=()):
    """Classify each clue word's current Stage Three purpose.

    This is intentionally conservative. It records what Stage Three can support
    now; it does not forgive leftover words merely because they look common.
    """
    annotated = getattr(casefile, "annotated_context", None)
    words = tuple(getattr(annotated, "words", ()) or _clue_words(
        casefile.clue_text))
    block_roles = _roles_by_token_index(blocks)
    definition_spans = tuple(
        tuple(block.get("span"))
        for block in blocks
        if block.get("kind") == "DEF_BLOCK"
        and block.get("span")
        and len(block.get("span")) == 2
    )
    definition_boundary_indices = _definition_boundary_indices(definition_spans)
    enrichment_roles = _enrichment_roles_by_token_index(enrichments)
    operation_roles = _operation_roles_by_token_index(
        getattr(casefile, "operation_candidates", ()) or ())
    operation_modifier_roles = _operation_modifier_roles_by_token_index(
        getattr(casefile, "grammar_phrases", ()) or (),
        operation_roles,
    )
    grammar = _grammar_by_token_index(
        getattr(casefile, "grammar_phrases", ()) or ())
    manual_by_index = _manual_roles_by_index(manual_roles)
    _source_block_spans = [
        tuple(block["span"])
        for block in blocks
        if block.get("kind") == "SOURCE_BLOCK"
        and block.get("span")
        and len(block.get("span")) == 2
    ]
    _wordplay_end = (
        max(s[1] for s in _source_block_spans)
        if _source_block_spans else None
    )
    _definition_start = (
        min(s[0] for s in definition_spans) if definition_spans else None
    )
    for index, word in enumerate(words):
        evidence = []
        role = None
        status = "candidate"
        if index in block_roles:
            block = block_roles[index]
            kind = block.get("kind")
            if kind == "DEF_BLOCK":
                role = _definition_purpose(index, word, definition_spans)
                status = "verified"
            elif kind == "SOURCE_BLOCK":
                role = "answer_source"
                status = "verified"
            elif kind == "OP_BLOCK":
                if block.get("status") in ("answer_fit", "verified"):
                    role = "operation_indicator"
                    status = "verified"
                else:
                    role = _purpose_for_operation_candidate(
                        block, index, definition_boundary_indices)
                    status = "candidate"
            elif kind == "REVIEW_BLOCK":
                role = "unresolved_purpose"
                status = "unresolved"
            elif kind == "CONDITIONAL_ASSEMBLY_BLOCK":
                role = "conditional_assembly"
            evidence.append({
                "source": "stage_three_block",
                "kind": kind,
                "text": block.get("text"),
            })
        if index in enrichment_roles:
            enrich = enrichment_roles[index]
            if role is None or role == "unresolved_purpose" or status == "candidate":
                role = _purpose_for_enrichment(enrich)
                status = "candidate"
            evidence.append({
                "source": "stage_two_enrichment",
                "kind": enrich.get("kind"),
                "text": enrich.get("text"),
                "value": enrich.get("value"),
            })
        if index in operation_roles:
            operation = operation_roles[index]
            if role is None or role == "unresolved_purpose":
                role = _purpose_for_operation_candidate(
                    operation, index, definition_boundary_indices)
                status = "candidate"
            evidence.append({
                "source": "stage_two_operation_candidate",
                "role": operation.get("role"),
                "token": operation.get("token"),
                "text": operation.get("text"),
            })
        if index in operation_modifier_roles:
            operation = operation_modifier_roles[index]
            if role is None or role == "unresolved_purpose":
                role = "operation_indicator_modifier_candidate"
                status = "candidate"
            evidence.append({
                "source": "grammar_operation_modifier",
                "phrase": operation.get("phrase"),
                "root_text": operation.get("root_text"),
                "root_token": operation.get("root_token"),
                "root_role": operation.get("root_role"),
                "text": word,
            })
        if index in grammar:
            evidence.extend(grammar[index])
        if (role is None or role == "unresolved_purpose") and index in manual_by_index:
            manual = manual_by_index[index]
            mapped = _purpose_for_manual_role(manual.get("role") or "")
            if mapped:
                role = mapped
                status = "manual"
                evidence.append({
                    "source": "manual_role",
                    "role": manual.get("role"),
                    "text": word,
                })
        if (
            role is None
            and _wordplay_end is not None
            and _definition_start is not None
            and _wordplay_end <= index < _definition_start
        ):
            role = "structural_separator"
            status = "verified"
        if role is None:
            role = "unresolved_purpose"
            status = "unresolved"
        yield {
            "index": index,
            "text": word,
            "purpose": role,
            "status": status,
            "evidence": evidence,
        }


def _roles_by_token_index(blocks):
    out = {}
    priority = {
        "DEF_BLOCK": 5,
        "SOURCE_BLOCK": 4,
        "OP_BLOCK": 3,
        "CONDITIONAL_ASSEMBLY_BLOCK": 2,
        "REVIEW_BLOCK": 1,
    }
    for block in blocks:
        span = block.get("span")
        if not span or len(span) != 2:
            continue
        for index in range(span[0], span[1]):
            current = out.get(index)
            if (current is None
                    or priority.get(block.get("kind"), 0)
                    > priority.get(current.get("kind"), 0)):
                out[index] = block
    return out


def _enrichment_roles_by_token_index(enrichments):
    out = {}
    for item in enrichments:
        span = item.get("span")
        if not span or len(span) != 2:
            continue
        for index in range(span[0], span[1]):
            out.setdefault(index, item)
    return out


def _operation_roles_by_token_index(operations):
    out = {}
    for item in operations:
        span = item.get("span")
        if not span or len(span) != 2:
            continue
        for index in range(span[0], span[1]):
            out.setdefault(index, item)
    return out


def _operation_modifier_roles_by_token_index(grammar_phrases, operation_roles):
    out = {}
    for phrase in grammar_phrases:
        root = phrase.get("root_index")
        if root is None or root not in operation_roles:
            continue
        span = phrase.get("span")
        if not span or len(span) != 2:
            continue
        root_operation = operation_roles[root]
        for index in range(span[0], span[1]):
            if index == root:
                continue
            out.setdefault(index, {
                "phrase": phrase.get("text"),
                "root_text": phrase.get("root_text"),
                "root_token": root_operation.get("token"),
                "root_role": root_operation.get("role"),
            })
    return out


def _grammar_by_token_index(grammar_phrases):
    out = {}
    for phrase in grammar_phrases:
        span = phrase.get("span")
        if not span or len(span) != 2:
            continue
        evidence = {
            "source": "grammar_phrase",
            "label": phrase.get("label"),
            "text": phrase.get("text"),
            "root_text": phrase.get("root_text"),
            "pos_tags": list(phrase.get("pos_tags") or ()),
            "dependencies": list(phrase.get("dependencies") or ()),
        }
        for index in range(span[0], span[1]):
            out.setdefault(index, []).append(evidence)
    return out


def _purpose_for_enrichment(item):
    kind = item.get("kind")
    if kind == "definition_gap":
        return "definition_phrase_candidate"
    if kind == "source_phrase_widening":
        return "phrase_widening_candidate"
    if kind == "conditional_source_gap":
        return "conditional_source_candidate"
    return "enrichment_candidate"


def _definition_boundary_indices(definition_spans):
    out = set()
    for start, end in definition_spans:
        out.add(end)
        out.add(start - 1)
    return out


def _purpose_for_operation_candidate(item, index=None,
                                     definition_boundary_indices=()):
    if item.get("role") == "joiner":
        return "structural_separator_candidate"
    if index in definition_boundary_indices:
        return "definition_separator_candidate"
    return "operation_indicator_candidate"


def _definition_purpose(index, word, definition_spans):
    marker_words = {
        "as", "for", "from", "in", "like", "of", "to",
    }
    for start, end in definition_spans:
        if index == start and end - start > 1 and word.lower() in marker_words:
            return "definition_phrase_marker"
    return "definition_phrase_member"


def _clue_words(clue_text):
    return tuple(
        word.strip(".,;:!?()[]{}\"'")
        for word in (clue_text or "").split()
        if word.strip(".,;:!?()[]{}\"'")
    )


def _clean_answer(value):
    return "".join(char for char in (value or "").upper() if char.isalpha())


def _valid_span(span):
    return (
        len(span) == 2
        and span[0] is not None
        and span[1] is not None
        and span[0] < span[1]
    )


def _span_gap(left, right):
    if left[1] <= right[0]:
        return right[0] - left[1]
    if right[1] <= left[0]:
        return left[0] - right[1]
    return 0


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
