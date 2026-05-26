"""GT2-style evidence-preserving candidate generation.

This module does not solve clues.  It emits constrained candidate bundles
that preserve spans, atoms, operations, gaps, and the overlay needed for the
existing solver to verify the parse through its normal paths.
"""
from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class GT2Node:
    node_id: str
    kind: str
    span: tuple[int, int] | None
    text: str
    span_space: str = "wordplay_tokens"
    value: str | None = None
    operation: str | None = None
    mechanisms: tuple[str, ...] = ()
    evidence: tuple[str, ...] = ()
    status: str = "candidate"

    def as_dict(self):
        return {
            "node_id": self.node_id,
            "kind": self.kind,
            "span": list(self.span) if self.span else None,
            "span_space": self.span_space if self.span else None,
            "text": self.text,
            "value": self.value,
            "operation": self.operation,
            "mechanisms": list(self.mechanisms),
            "evidence": list(self.evidence),
            "status": self.status,
        }


@dataclass(frozen=True)
class GT2Edge:
    edge_id: str
    kind: str
    from_node: str
    to_node: str
    evidence: tuple[str, ...] = ()
    scope_status: str = "known"
    confidence: str = "medium"
    notes: str = ""

    def as_dict(self):
        return {
            "edge_id": self.edge_id,
            "kind": self.kind,
            "from_node": self.from_node,
            "to_node": self.to_node,
            "evidence": list(self.evidence),
            "scope_status": self.scope_status,
            "confidence": self.confidence,
            "notes": self.notes,
        }


@dataclass(frozen=True)
class GT2CandidateBundle:
    candidate_id: str
    clue: str
    answer: str
    operation: str
    nodes: tuple[GT2Node, ...]
    edges: tuple[GT2Edge, ...]
    overlay_synonyms: dict[str, list[str]] = field(default_factory=dict)
    gaps: tuple[dict, ...] = ()
    verification: dict = field(default_factory=dict)

    def as_dict(self):
        return {
            "candidate_id": self.candidate_id,
            "clue": self.clue,
            "answer": self.answer,
            "operation": self.operation,
            "nodes": [n.as_dict() for n in self.nodes],
            "edges": [e.as_dict() for e in self.edges],
            "overlay_synonyms": self.overlay_synonyms,
            "gaps": list(self.gaps),
            "verification": self.verification,
        }


def generate_gt2_candidates(clue_text, answer, db, definition_candidates,
                            clue_context=None):
    """Generate evidence-preserving candidate bundles for failed clues.

    First production slice: answer-aware container complement recovery.
    If a container indicator and a short known inner atom exist, derive the
    missing outer shell from the answer, attach it to a preserved source span,
    and return an overlay bundle for the existing solver to verify.
    """
    bundles = []
    bundles.extend(_generate_container_complement_candidates(
        clue_text, answer, db, definition_candidates, clue_context))
    bundles.extend(_generate_container_charade_span_candidates(
        clue_text, answer, db, definition_candidates, clue_context))
    return bundles


def generate_gt2_evidence_bundles(clue_text, answer, db,
                                  definition_candidates,
                                  solve_result=None,
                                  clue_context=None):
    """Emit read-only GT2 evidence bundles from existing parser evidence.

    This is the broad adapter layer: it converts current word/phrase analysis
    and the accepted solver result, when present, into GT2 nodes. It does not
    add overlays and does not affect solver selection.
    """
    answer = _clean_value(answer)
    bundles = []
    wp_windows = definition_candidates or [(None, clue_text.strip().split())]
    for idx, (def_phrase, wp_words) in enumerate(wp_windows[:3]):
        bundles.append(_build_analysis_bundle(
            idx, clue_text, answer, db, def_phrase, wp_words, solve_result,
            clue_context))
    return bundles


def bundle_actually_used(sr_obj, bundle):
    """Confirm a solver result used the overlay source/value from a bundle."""
    if sr_obj is None or sr_obj.result is None:
        return False
    targets = {
        (_clean_text(phrase), _clean_value(values[0]))
        for phrase, values in bundle.overlay_synonyms.items()
        if values
    }
    if not targets:
        return False
    for word, _tok, val, *_ in sr_obj.result.word_roles:
        if (_clean_text(word), _clean_value(val or "")) in targets:
            return True
    return False


def annotations_from_bundle(bundle):
    """Convert verified GT2 overlay gaps into ClueContext annotations."""
    try:
        from .clue_context import SpanAnnotation
    except Exception:
        return ()
    annotations = []
    for gap in bundle.gaps:
        if gap.get("kind") != "missing_span_value":
            continue
        if gap.get("span_space") != "full_clue_tokens":
            continue
        span = gap.get("span")
        value = gap.get("value")
        text = gap.get("text", "")
        if (not isinstance(span, list) or len(span) != 2
                or not value):
            continue
        annotations.append(SpanAnnotation(
            span=(span[0], span[1]),
            text=text,
            token="SYN_F",
            values=(_clean_value(value),),
            source=gap.get("source") or "gt2_bundle",
        ))
    return tuple(annotations)


def _build_analysis_bundle(idx, clue_text, answer, db, def_phrase, wp_words,
                           solve_result, clue_context=None):
    from .word_analyzer import analyze_phrases
    from .tokens import (
        ABR_F, ANA_F, CON_I, DBE_MARKER, DEL_F, DEL_I, HID_F, HID_I,
        HOM_F, HOM_I, LNK, POS_F, RAW, REV_I, SYN_F,
    )

    analyses, phrases = analyze_phrases(wp_words, answer, db)
    wp_offset = _wordplay_offset(clue_context, wp_words)
    nodes = [
        GT2Node("answer", "ASSEMBLY_BLOCK", None, answer,
                value=answer, status="inferred")
    ]
    edges = []

    if def_phrase:
        def_node = GT2Node(
            "def_0", "DEF_BLOCK",
            _definition_span(clue_text, def_phrase, wp_words, clue_context),
            def_phrase, span_space="full_clue_tokens", value=answer,
            evidence=("definition_candidate",), status="observed")
        nodes.append(def_node)
        edges.append(GT2Edge(
            "edge_def", "DEFINES", "def_0", "answer",
            evidence=("definition_candidate",), confidence="strong"))

    source_tokens = {SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, DEL_F, POS_F}
    relation_tokens = {CON_I}
    op_tokens = {REV_I, DEL_I, HID_I, HOM_I}

    def add_role_node(node_id, span, text, tok, values, status):
        node_span = _full_span_from_wordplay(span, wp_offset)
        span_space = "full_clue_tokens" if node_span else "wordplay_tokens"
        if node_span is None:
            node_span = span
        kind = "SOURCE_BLOCK"
        operation = None
        mechanisms = (tok,)
        if tok in relation_tokens:
            kind = "RELATION_BLOCK"
            operation = "container"
            mechanisms = ()
        elif tok in op_tokens:
            kind = "OP_BLOCK"
            operation = _operation_for_token(tok)
            mechanisms = ()
        elif tok.startswith("POS_I_"):
            kind = "LOCATOR_BLOCK"
            operation = _operation_for_token(tok)
            mechanisms = ()
        elif tok == LNK:
            kind = "CONNECTOR_BLOCK"
            mechanisms = ()
        elif tok == DBE_MARKER:
            kind = "DEF_MODIFIER_BLOCK"
            mechanisms = ()
        value = None
        if values and values != [True] and tok in source_tokens:
            value = values[0]
        evidence = ("word_analyzer",)
        if len(span or ()) == 2 and span[1] - span[0] > 1:
            evidence = evidence + ("phrase_span",)
        nodes.append(GT2Node(
            node_id, kind, node_span, text, span_space=span_space,
            value=value, operation=operation, mechanisms=mechanisms,
            evidence=evidence, status=status))
        if kind == "SOURCE_BLOCK":
            edges.append(GT2Edge(
                "edge_%s_contrib" % node_id, "CONTRIBUTES_TO", node_id,
                "answer", evidence=evidence,
                scope_status="candidate_until_verification",
                confidence="medium"))

    for i, wa in enumerate(analyses):
        for tok, vals in wa.roles.items():
            add_role_node("cand_w%d_%s" % (i, tok), (i, i + 1),
                          wp_words[i], tok, vals, "candidate")

    for (start, end), wa in phrases.items():
        text = " ".join(wp_words[start:end])
        for tok, vals in wa.roles.items():
            add_role_node("cand_p%d_%d_%s" % (start, end, tok),
                          (start, end), text, tok, vals, "candidate")

    if clue_context is not None and wp_offset is not None:
        for pos_idx, pos_span in enumerate(clue_context.pos_spans):
            local = (pos_span.start - wp_offset, pos_span.end - wp_offset)
            if local[0] < 0 or local[1] > len(wp_words):
                continue
            nodes.append(GT2Node(
                "pos_span_%d" % pos_idx, "SPAN_CANDIDATE",
                (pos_span.start, pos_span.end),
                pos_span.text, span_space="full_clue_tokens",
                mechanisms=("pos_%s" % pos_span.label,),
                evidence=("pos_model", clue_context.pos_model_status),
                status="candidate"))

    if solve_result is not None and solve_result.result is not None:
        accepted_nodes, accepted_edges = _accepted_result_nodes(
            wp_words, solve_result, wp_offset)
        nodes.extend(accepted_nodes)
        edges.extend(accepted_edges)

    verification = {
        "status": "mechanically_verified"
        if solve_result is not None and solve_result.high_confidence
        else "candidate",
        "accepted_form": (
            solve_result.result.explanation_parts[0]
            if solve_result is not None and solve_result.result else None),
        "unresolved_questions": [],
    }
    return GT2CandidateBundle(
        candidate_id="gt2:evidence:%d" % idx,
        clue=clue_text,
        answer=answer,
        operation="evidence_adapter",
        nodes=tuple(nodes),
        edges=tuple(edges),
        verification=verification,
    )


def _accepted_result_nodes(wp_words, solve_result, wp_offset=None):
    from .tokens import (
        ABR_F, ANA_F, CON_I, DBE_MARKER, DEL_F, DEL_I, HID_F, HID_I,
        HOM_F, HOM_I, LNK, POS_F, RAW, REV_I, SYN_F,
    )

    source_tokens = {SYN_F, ABR_F, ANA_F, RAW, HID_F, HOM_F, DEL_F, POS_F}
    nodes = []
    edges = []
    for idx, role in enumerate(solve_result.result.word_roles):
        word, tok, val, *rest = role
        meta = rest[0] if rest and isinstance(rest[0], dict) else {}
        local_span = _find_span(wp_words, word)
        span = _full_span_from_wordplay(local_span, wp_offset) or local_span
        span_space = "full_clue_tokens" if wp_offset is not None else "wordplay_tokens"
        node_id = "accepted_%d" % idx
        kind = "SOURCE_BLOCK"
        operation = None
        mechanisms = (tok,)
        if tok == CON_I:
            kind = "RELATION_BLOCK"
            operation = "container"
            mechanisms = ()
        elif tok in {REV_I, DEL_I, HID_I, HOM_I}:
            kind = "OP_BLOCK"
            operation = _operation_for_token(tok)
            mechanisms = ()
        elif tok.startswith("POS_I_"):
            kind = "LOCATOR_BLOCK"
            operation = _operation_for_token(tok)
            mechanisms = ()
        elif tok == LNK:
            kind = "CONNECTOR_BLOCK"
            mechanisms = ()
        elif tok == DBE_MARKER:
            kind = "DEF_MODIFIER_BLOCK"
            mechanisms = ()
        nodes.append(GT2Node(
            node_id, kind, span, word, span_space=span_space, value=val,
            operation=operation, mechanisms=mechanisms,
            evidence=("accepted_parse",), status="observed"))
        if tok in source_tokens:
            edges.append(GT2Edge(
                "edge_%s_contrib" % node_id, "CONTRIBUTES_TO",
                node_id, "answer", evidence=("accepted_parse",),
                confidence="strong"))

        if meta.get("transform"):
            transform_id = "accepted_%d_transform" % idx
            output = meta.get("reversed_to") or meta.get("derived")
            nodes.append(GT2Node(
                transform_id, "OP_BLOCK", span, word, span_space=span_space,
                value=output, operation=meta.get("transform"),
                evidence=("accepted_parse", "transform_metadata"),
                status="observed"))
            edges.append(GT2Edge(
                "edge_%s_operates" % transform_id, "OPERATES_ON",
                transform_id, node_id,
                evidence=("transform_metadata",),
                scope_status="known", confidence="strong",
                notes="%s -> %s" % (val, output)))
            if output:
                output_id = "accepted_%d_output" % idx
                nodes.append(GT2Node(
                    output_id, "SOURCE_BLOCK", None, output,
                    value=output, mechanisms=("transform_output",),
                    evidence=("accepted_parse", "transform_metadata"),
                    status="inferred"))
                edges.append(GT2Edge(
                    "edge_%s_output" % transform_id, "PRODUCES",
                    transform_id, output_id,
                    evidence=("transform_metadata",),
                    scope_status="known", confidence="strong"))
    return nodes, edges


def _operation_for_token(tok):
    if tok == "CON_I":
        return "container"
    if tok == "REV_I":
        return "reversal"
    if tok == "DEL_I":
        return "deletion"
    if tok == "HID_I":
        return "hidden"
    if tok == "HOM_I":
        return "homophone"
    if tok.startswith("POS_I_TRIM"):
        return "trim"
    if tok.startswith("POS_I_"):
        return "positional"
    return None


def _find_span(words, text):
    target = _clean_words(text.split())
    if not target:
        return None
    for i in range(len(words) - len(target) + 1):
        if _clean_words(words[i:i + len(target)]) == target:
            return (i, i + len(target))
    return None


def _generate_container_complement_candidates(
        clue_text, answer, db, definition_candidates, clue_context=None):
    if not definition_candidates:
        return []

    answer = _clean_value(answer)
    if len(answer) < 5:
        return []

    bundles = []
    validation_calls = 0
    max_validation_calls = 1

    for def_phrase, wp_words in definition_candidates:
        wp_offset = _wordplay_offset(clue_context, wp_words)
        if len(wp_words) < 3:
            continue

        indicator_spans = _container_indicator_spans(wp_words, db)
        if not indicator_spans:
            continue

        atom_candidates = _short_inner_atom_candidates(wp_words, answer, db)
        if not atom_candidates:
            continue

        for ind_span in indicator_spans[:3]:
            ind_idxs = set(range(ind_span[0], ind_span[1]))
            for inner_span, inner_value, inner_mechanism in atom_candidates[:8]:
                inner_idxs = set(range(inner_span[0], inner_span[1]))
                if inner_idxs & ind_idxs:
                    continue
                for pos in _occurrences(answer, inner_value):
                    shell = answer[:pos] + answer[pos + len(inner_value):]
                    if len(shell) < 3:
                        continue
                    if not db.is_real_word(shell):
                        continue
                    if shell[:pos] + inner_value + shell[pos:] != answer:
                        continue
                    used = ind_idxs | inner_idxs
                    for outer_span in _remaining_source_spans(
                            wp_words, used, clue_context, wp_offset):
                        phrase = " ".join(wp_words[outer_span[0]:outer_span[1]])
                        if _phrase_looks_operational(phrase, db):
                            continue
                        known = _span_value_known_to_db(phrase, shell, db)
                        evidence_source = "span_value_verified"
                        if not known:
                            if _span_is_pos_model_candidate(
                                    outer_span, clue_context, wp_offset):
                                known = True
                                evidence_source = (
                                    "pos_span_answer_complement")
                            else:
                                if validation_calls >= max_validation_calls:
                                    continue
                                validation_calls += 1
                                known = _verify_span_value(
                                    clue_text, phrase, shell, answer,
                                    role_hint="container outer/shell")
                        if not known:
                            continue
                        bundles.append(_build_container_bundle(
                            clue_text, answer, def_phrase,
                            _definition_span(
                                clue_text, def_phrase, wp_words,
                                clue_context),
                            wp_words, wp_offset,
                            outer_span, phrase, shell,
                            inner_span, inner_value, inner_mechanism,
                            ind_span, evidence_source=evidence_source))
                        if len(bundles) >= 3:
                            return bundles
    return bundles


def _build_container_bundle(clue_text, answer, def_phrase, def_span, wp_words,
                            wp_offset,
                            outer_span, outer_text, outer_value,
                            inner_span, inner_value, inner_mechanism,
                            indicator_span,
                            evidence_source="span_value_verified"):
    indicator_text = " ".join(wp_words[indicator_span[0]:indicator_span[1]])
    inner_text = " ".join(wp_words[inner_span[0]:inner_span[1]])
    outer_node_span = _full_span_from_wordplay(outer_span, wp_offset) or outer_span
    inner_node_span = _full_span_from_wordplay(inner_span, wp_offset) or inner_span
    indicator_node_span = (
        _full_span_from_wordplay(indicator_span, wp_offset) or indicator_span)
    wp_span_space = (
        "full_clue_tokens" if wp_offset is not None else "wordplay_tokens")
    cid = "gt2:container_complement:%s:%s:%s" % (
        outer_span[0], inner_span[0], indicator_span[0])
    nodes = (
        GT2Node("answer", "ASSEMBLY_BLOCK", None, answer,
                value=answer, status="inferred"),
        GT2Node("def_0", "DEF_BLOCK", def_span, def_phrase,
                span_space="full_clue_tokens", value=answer,
                evidence=("definition_candidate",), status="observed"),
        GT2Node("src_outer", "SOURCE_BLOCK", outer_node_span, outer_text,
                span_space=wp_span_space, value=outer_value,
                mechanisms=("synonym",),
                evidence=("answer_complement", evidence_source),
                status="inferred"),
        GT2Node("src_inner", "SOURCE_BLOCK", inner_node_span, inner_text,
                span_space=wp_span_space, value=inner_value,
                mechanisms=(inner_mechanism,),
                evidence=("db_atom", "answer_substring"),
                status="observed"),
        GT2Node("rel_0", "RELATION_BLOCK", indicator_node_span,
                indicator_text, span_space=wp_span_space,
                operation="container",
                evidence=("indicator_table",), status="observed"),
    )
    edges = (
        GT2Edge("edge_def", "DEFINES", "def_0", "answer",
                evidence=("definition_candidate",), confidence="strong"),
        GT2Edge("edge_outer", "CONTRIBUTES_TO", "src_outer", "answer",
                evidence=("answer_complement",), confidence="strong"),
        GT2Edge("edge_inner", "CONTRIBUTES_TO", "src_inner", "answer",
                evidence=("answer_substring",), confidence="strong"),
        GT2Edge("edge_contains_outer", "CONTAINS", "rel_0", "src_outer",
                scope_status="resolved_by_answer_mechanics",
                notes="outer shell derived by subtracting inner atom"),
        GT2Edge("edge_contains_inner", "CONTAINS", "rel_0", "src_inner",
                scope_status="resolved_by_answer_mechanics",
                notes="inner atom occurs inside answer"),
    )
    gap = {
        "kind": "missing_span_value",
        "span": list(outer_node_span),
        "span_space": wp_span_space,
        "text": outer_text,
        "value": outer_value,
        "source": "gt2_%s" % evidence_source,
    }
    verification = {
        "status": "candidate",
        "tested_forms": [
            "%s containing %s -> %s" % (
                outer_value, inner_value, answer)
        ],
        "accepted_form": None,
        "unresolved_questions": [
            "existing solver must verify bundle overlay through normal paths"
        ],
    }
    return GT2CandidateBundle(
        candidate_id=cid,
        clue=clue_text,
        answer=answer,
        operation="container",
        nodes=nodes,
        edges=edges,
        overlay_synonyms={outer_text: [outer_value]},
        gaps=(gap,),
        verification=verification,
    )


def _generate_container_charade_span_candidates(
        clue_text, answer, db, definition_candidates, clue_context=None):
    if not definition_candidates:
        return []
    answer = _clean_value(answer)
    bundles = []
    validation_calls = 0
    max_validation_calls = 2
    for def_phrase, wp_words in definition_candidates:
        wp_offset = _wordplay_offset(clue_context, wp_words)
        indicator_spans = _container_indicator_spans(wp_words, db)
        atom_candidates = _short_inner_atom_candidates(wp_words, answer, db)
        source_candidates = _short_source_atom_candidates(wp_words, answer, db)
        if not indicator_spans or not atom_candidates:
            continue
        for ind_span in indicator_spans[:3]:
            ind_idxs = set(range(ind_span[0], ind_span[1]))
            local_atom_candidates = sorted(
                atom_candidates,
                key=lambda item: _indicator_proximity_key(ind_span, item[0]))
            for inner_span, inner_value, inner_mechanism in local_atom_candidates[:10]:
                if set(range(inner_span[0], inner_span[1])) & ind_idxs:
                    continue
                used = ind_idxs | set(range(inner_span[0], inner_span[1]))
                for outer_span in _remaining_source_spans(
                        wp_words, used, clue_context, wp_offset):
                    if not _container_indicator_scopes(
                            ind_span, outer_span, inner_span):
                        continue
                    phrase = " ".join(wp_words[outer_span[0]:outer_span[1]])
                    if _phrase_looks_operational(phrase, db):
                        continue
                    other_sources = [
                        item for item in source_candidates
                        if not _spans_overlap(item[0], outer_span)
                        and not _spans_overlap(item[0], inner_span)
                        and not _spans_overlap(item[0], ind_span)
                    ]
                    for result_start in range(0, len(answer)):
                        for result_end in range(
                                result_start + len(inner_value) + 2,
                                len(answer) + 1):
                            result = answer[result_start:result_end]
                            outer_values = _container_outer_values_for_result(
                                result, inner_value)
                            if not outer_values:
                                continue
                            before = answer[:result_start]
                            after = answer[result_end:]
                            if not _can_cover_with_atoms(before, other_sources):
                                continue
                            if not _can_cover_with_atoms(after, other_sources):
                                continue
                            for outer_value in outer_values:
                                known = _span_value_known_to_db(
                                    phrase, outer_value, db)
                                evidence_source = "span_value_verified"
                                if not known:
                                    if validation_calls >= max_validation_calls:
                                        continue
                                    validation_calls += 1
                                    known = _verify_span_value(
                                        clue_text, phrase, outer_value, answer,
                                        role_hint="container outer/shell")
                                if not known:
                                    continue
                                bundles.append(_build_container_charade_bundle(
                                    clue_text, answer, def_phrase,
                                    _definition_span(
                                        clue_text, def_phrase, wp_words,
                                        clue_context),
                                    wp_words, wp_offset,
                                    outer_span, phrase, outer_value,
                                    inner_span, inner_value, inner_mechanism,
                                    ind_span, result,
                                    evidence_source=evidence_source))
                                if len(bundles) >= 3:
                                    return bundles
    return bundles


def _build_container_charade_bundle(
        clue_text, answer, def_phrase, def_span, wp_words, wp_offset,
        outer_span, outer_text, outer_value,
        inner_span, inner_value, inner_mechanism, indicator_span,
        result, evidence_source="span_value_verified"):
    bundle = _build_container_bundle(
        clue_text, answer, def_phrase, def_span, wp_words, wp_offset,
        outer_span, outer_text, outer_value,
        inner_span, inner_value, inner_mechanism, indicator_span,
        evidence_source=evidence_source)
    return GT2CandidateBundle(
        candidate_id="gt2:container_charade_span:%s:%s:%s" % (
            outer_span[0], inner_span[0], indicator_span[0]),
        clue=bundle.clue,
        answer=bundle.answer,
        operation="container_charade",
        nodes=bundle.nodes,
        edges=bundle.edges,
        overlay_synonyms=bundle.overlay_synonyms,
        gaps=bundle.gaps,
        verification={
            "status": "candidate",
            "tested_forms": [
                "%s containing %s -> %s inside %s" % (
                    outer_value, inner_value, result, answer)
            ],
            "accepted_form": None,
            "unresolved_questions": [
                "atomic assembler must verify the surrounding charade"
            ],
        },
    )


def _container_indicator_spans(words, db):
    out = []
    n = len(words)
    for span in (1, 2, 3):
        for i in range(n - span + 1):
            phrase = " ".join(words[i:i + span])
            ind_types = db.get_indicator_types(_clean_text(phrase))
            if any(t in ("container", "insertion") for t, _, _ in ind_types):
                out.append((i, i + span))
    return out


def _short_inner_atom_candidates(words, answer, db):
    out = []
    seen = set()
    for i, word in enumerate(words):
        vals = []
        vals.extend((a, "abbreviation") for a in db.get_abbreviations(_clean_text(word)))
        vals.extend((s, "synonym")
                    for s in db.get_synonyms_substring_of(_clean_text(word), answer))
        for val, mechanism in vals:
            clean = _clean_value(val)
            key = ((i, i + 1), clean)
            if (not clean or clean == answer or clean not in answer
                    or len(clean) > 3 or key in seen):
                continue
            out.append(((i, i + 1), clean, mechanism))
            seen.add(key)
    return out


def _indicator_proximity_key(indicator_span, source_span):
    if source_span[0] >= indicator_span[1]:
        distance = source_span[0] - indicator_span[1]
        side = 0
    elif source_span[1] <= indicator_span[0]:
        distance = indicator_span[0] - source_span[1]
        side = 1
    else:
        distance = 99
        side = 2
    return (distance, side, source_span[0], source_span[1])


def _short_source_atom_candidates(words, answer, db):
    out = []
    seen = set()
    answer = _clean_value(answer)
    for i, word in enumerate(words):
        vals = []
        vals.extend(db.get_abbreviations(_clean_text(word)))
        vals.extend(db.get_synonyms_substring_of(_clean_text(word), answer))
        for val in vals:
            clean = _clean_value(val)
            key = ((i, i + 1), clean)
            if not clean or clean not in answer or key in seen:
                continue
            out.append(((i, i + 1), clean))
            seen.add(key)
    return out


def _remaining_source_spans(words, used_idxs, clue_context=None,
                            wp_offset=None):
    """Prefer POS/chunk spans, then contiguous leftovers, then singles."""
    n = len(words)
    spans = []
    if clue_context is not None and wp_offset is not None:
        for pos_span in clue_context.pos_spans:
            local = (pos_span.start - wp_offset, pos_span.end - wp_offset)
            if local[0] < 0 or local[1] > n:
                continue
            idxs = set(range(local[0], local[1]))
            if idxs & used_idxs:
                continue
            if local not in spans:
                spans.append(local)
    for local in _semantic_source_spans(words):
        idxs = set(range(local[0], local[1]))
        if idxs & used_idxs:
            continue
        if local not in spans:
            spans.append(local)
    for span in range(min(4, n), 1, -1):
        for i in range(n - span + 1):
            idxs = set(range(i, i + span))
            if idxs & used_idxs:
                continue
            if (i, i + span) not in spans:
                spans.append((i, i + span))
    for i in range(n):
        if i not in used_idxs and (i, i + 1) not in spans:
            spans.append((i, i + 1))
    return spans


def _semantic_source_spans(words):
    spans = []
    clean = [_clean_text(word) for word in words]
    for idx in range(1, len(clean) - 1):
        if clean[idx] in {"of", "for"}:
            spans.append((idx - 1, idx + 2))
    return spans


def _container_indicator_scopes(indicator_span, outer_span, inner_span):
    if outer_span[1] <= indicator_span[0] and indicator_span[1] <= inner_span[0]:
        return True
    if inner_span[1] <= indicator_span[0] and indicator_span[1] <= outer_span[0]:
        return True
    return False


def _container_outer_values_for_result(result, inner):
    result = _clean_value(result)
    inner = _clean_value(inner)
    values = []
    if not result or not inner:
        return values
    start = 0
    while True:
        pos = result.find(inner, start)
        if pos < 0:
            break
        if pos > 0 and pos + len(inner) < len(result):
            values.append(result[:pos] + result[pos + len(inner):])
        start = pos + 1
    return values


def _can_cover_with_atoms(text, source_candidates):
    text = _clean_value(text)
    if not text:
        return True
    by_value = sorted(source_candidates, key=lambda item: len(item[1]),
                      reverse=True)

    def search(pos, used_spans):
        if pos == len(text):
            return True
        for span, value in by_value:
            if span in used_spans:
                continue
            if text.startswith(value, pos):
                if search(pos + len(value), used_spans | {span}):
                    return True
        return False

    return search(0, set())


def _spans_overlap(a, b):
    return a[0] < b[1] and b[0] < a[1]


def _span_is_pos_model_candidate(span, clue_context=None, wp_offset=None):
    if clue_context is None or wp_offset is None:
        return False
    full_span = _full_span_from_wordplay(span, wp_offset)
    if full_span is None:
        return False
    return any(
        pos_span.start == full_span[0] and pos_span.end == full_span[1]
        for pos_span in clue_context.pos_spans
    )


def _definition_span(clue_text, def_phrase, wp_words, clue_context=None):
    if clue_context is not None:
        span = clue_context.find_span_text(
            def_phrase, kind="definition_candidate")
        if span is not None:
            return span.as_tuple()
    clue_words = clue_text.strip().split()
    def_words = def_phrase.strip().split()
    if not clue_words or not def_words:
        return None
    if _clean_words(clue_words[:len(def_words)]) == _clean_words(def_words):
        return (0, len(def_words))
    if _clean_words(clue_words[-len(def_words):]) == _clean_words(def_words):
        return (len(clue_words) - len(def_words), len(clue_words))

    wp_clean = _clean_words(wp_words)
    for i in range(len(clue_words) + 1):
        remaining = clue_words[:i] + clue_words[i + len(def_words):]
        if _clean_words(remaining) == wp_clean:
            return (i, i + len(def_words))
    return None


def _wordplay_offset(clue_context, wp_words):
    if clue_context is None:
        return None
    span = clue_context.wordplay_span_for_words(wp_words)
    if span is None:
        return None
    return span.start


def _full_span_from_wordplay(span, wp_offset):
    if span is None or wp_offset is None:
        return None
    return (span[0] + wp_offset, span[1] + wp_offset)


def _occurrences(text, needle):
    start = 0
    while True:
        pos = text.find(needle, start)
        if pos < 0:
            return
        yield pos
        start = pos + 1


def _phrase_looks_operational(phrase, db):
    words = phrase.lower().split()
    if not words:
        return True
    for word in (words[0], words[-1]):
        clean = _clean_text(word)
        if db.is_link_word(clean):
            return True
        if db.get_indicator_types(clean):
            return True
    return False


def _span_value_known_to_db(phrase, value, db):
    value = _clean_value(value)
    if value in {_clean_value(v) for v in _semantic_span_values(phrase)}:
        return True
    vals = []
    vals.extend(db.get_abbreviations(_clean_text(phrase)))
    vals.extend(db.get_synonyms(_clean_text(phrase), max_len=len(value)))
    return value in {_clean_value(v) for v in vals}


def _semantic_span_values(phrase):
    clean = _clean_text(phrase)
    return {
        "group of two": ("PAIR", "DUO"),
        "group of three": ("TRIO",),
        "group of four": ("QUARTET",),
        "group of five": ("QUINTET",),
        "group of six": ("SEXTET",),
        "set of three": ("TRIO",),
    }.get(clean, ())


def _verify_span_value(clue_text, phrase, value, answer, role_hint=None):
    try:
        from .haiku_span_value import verify_span_value
    except Exception:
        return False
    return verify_span_value(clue_text, phrase, value, answer, role_hint)


def _clean_text(text):
    return (text or "").lower().strip(".,;:!?\"'()-")


def _clean_words(words):
    return [_clean_text(w) for w in words]


def _clean_value(value):
    return "".join(c for c in (value or "").upper() if c.isalpha())
