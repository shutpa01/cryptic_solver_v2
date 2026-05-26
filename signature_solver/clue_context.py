"""Canonical token/span context for clue parsing stages.

Stage One owns only surface text structure: tokens, character offsets,
candidate spans, definition windows, and wordplay windows. Later stages
annotate these spans with meanings and assemble parses from them.
"""
from __future__ import annotations

from dataclasses import dataclass
import re
import unicodedata


@dataclass(frozen=True)
class ClueToken:
    index: int
    text: str
    normalized: str
    start_char: int
    end_char: int
    atom_ids: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "index": self.index,
            "text": self.text,
            "normalized": self.normalized,
            "start_char": self.start_char,
            "end_char": self.end_char,
            "atom_ids": list(self.atom_ids),
        }


@dataclass(frozen=True)
class ClueSpan:
    start: int
    end: int
    text: str
    normalized: str
    kind: str

    def as_tuple(self):
        return (self.start, self.end)

    def as_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "normalized": self.normalized,
            "kind": self.kind,
        }


@dataclass(frozen=True)
class SpanAnnotation:
    span: tuple[int, int]
    text: str
    token: str
    values: tuple
    source: str

    def as_dict(self):
        return {
            "span": list(self.span),
            "text": self.text,
            "token": self.token,
            "values": list(self.values),
            "source": self.source,
        }


@dataclass(frozen=True)
class DefinitionCandidate:
    definition_span: ClueSpan
    wordplay_span: ClueSpan
    boundary_status: str = "unchecked"
    db_definition_span: ClueSpan | None = None
    phrase_structure_evidence: tuple = ()
    objections: tuple[str, ...] = ()

    @property
    def def_phrase(self):
        return self.definition_span.text

    @property
    def wp_words(self):
        return self.wordplay_span.text.split()

    def as_legacy_tuple(self):
        return self.def_phrase, self.wp_words

    def as_dict(self):
        return {
            "definition_span": self.definition_span.as_dict(),
            "wordplay_span": self.wordplay_span.as_dict(),
            "boundary_status": self.boundary_status,
            "db_definition_span": (
                self.db_definition_span.as_dict()
                if self.db_definition_span is not None else None
            ),
            "phrase_structure_evidence": [
                evidence.as_dict() if hasattr(evidence, "as_dict")
                else evidence
                for evidence in self.phrase_structure_evidence
            ],
            "objections": list(self.objections),
        }


@dataclass(frozen=True)
class ClueContext:
    clue_text: str
    normalized_clue: str
    answer: str
    tokens: tuple[ClueToken, ...]
    spans: tuple[ClueSpan, ...]
    definition_candidates: tuple[DefinitionCandidate, ...]
    wordplay_windows: tuple[ClueSpan, ...]
    annotations: tuple[SpanAnnotation, ...] = ()
    pos_spans: tuple = ()
    pos_model_status: str = "not_requested"
    atom_context: object | None = None

    @property
    def words(self):
        return [token.text for token in self.tokens]

    def span_text(self, start, end):
        if start < 0 or end > len(self.tokens) or start >= end:
            return ""
        start_char = self.tokens[start].start_char
        end_char = self.tokens[end - 1].end_char
        return self.normalized_clue[start_char:end_char]

    def span(self, start, end, kind="ad_hoc"):
        return ClueSpan(
            start=start,
            end=end,
            text=self.span_text(start, end),
            normalized=_normalize_span_text(self.span_text(start, end)),
            kind=kind,
        )

    def wordplay_span_for_words(self, words):
        target = [_normalize_word(w) for w in words]
        if not target:
            return None
        norm_words = [_normalize_word(t.text) for t in self.tokens]
        for i in range(len(norm_words) - len(target) + 1):
            if norm_words[i:i + len(target)] == target:
                return self.span(i, i + len(target), kind="wordplay_window")
        return None

    def find_span_text(self, text, kind="matched_text"):
        target = [_normalize_word(w) for w in text.split()]
        if not target:
            return None
        norm_words = [_normalize_word(t.text) for t in self.tokens]
        for i in range(len(norm_words) - len(target) + 1):
            if norm_words[i:i + len(target)] == target:
                return self.span(i, i + len(target), kind=kind)
        return None

    def as_dict(self):
        return {
            "clue_text": self.clue_text,
            "normalized_clue": self.normalized_clue,
            "answer": self.answer,
            "atom_context": (
                self.atom_context.as_dict()
                if self.atom_context is not None else None
            ),
            "tokens": [token.as_dict() for token in self.tokens],
            "spans": [span.as_dict() for span in self.spans],
            "definition_candidates": [
                candidate.as_dict() for candidate in self.definition_candidates
            ],
            "wordplay_windows": [
                window.as_dict() for window in self.wordplay_windows
            ],
            "pos_spans": [
                span.as_dict() for span in self.pos_spans
            ],
            "pos_model_status": self.pos_model_status,
            "annotations": [
                annotation.as_dict() for annotation in self.annotations
            ],
        }


def build_clue_context(clue_text, answer, db=None, max_phrase_words=4,
                       annotate=False, pos_model=None, use_pos_model=True,
                       max_definition_words=7, manual_roles=None):
    from .wfw_atoms import build_wfw_atom_context

    atom_context = build_wfw_atom_context(clue_text, answer)
    normalized = normalize_clue_text(clue_text).strip()
    tokens = tuple(_tokenize(normalized, atom_context))
    answer_clean = _clean_answer(answer)

    spans = list(_candidate_spans(tokens, normalized, max_phrase_words))
    pos_spans = ()
    pos_model_status = "not_requested"
    if use_pos_model and tokens:
        from .pos_span_model import propose_pos_spans

        pos_spans, pos_model_status = propose_pos_spans(
            normalized, tokens, model=pos_model)
        spans.extend(
            _make_span(tokens, normalized, span.start, span.end,
                       "pos_%s" % span.label)
            for span in pos_spans
        )
    definition_candidates = []
    wordplay_windows = []

    if tokens:
        full_span = _make_span(tokens, normalized, 0, len(tokens), "full_clue")
        spans.append(full_span)

    if db is not None:
        for n in range(1, min(max_definition_words + 1, len(tokens))):
            start_def = _make_span(
                tokens, normalized, 0, n, "definition_candidate")
            start_wp = _make_span(
                tokens, normalized, n, len(tokens), "wordplay_window")
            if start_wp.text and _is_definition_of_answer(
                    db, start_def.text, answer_clean):
                candidate = _definition_candidate_from_db_hit(
                    tokens, normalized, start_def, start_wp, pos_spans)
                definition_candidates.append(candidate)
                spans.extend([
                    start_def,
                    candidate.definition_span,
                    candidate.wordplay_span,
                ])
                wordplay_windows.append(candidate.wordplay_span)

            end_def = _make_span(
                tokens, normalized, len(tokens) - n, len(tokens),
                "definition_candidate")
            end_wp = _make_span(
                tokens, normalized, 0, len(tokens) - n, "wordplay_window")
            if end_wp.text and _is_definition_of_answer(
                    db, end_def.text, answer_clean):
                candidate = _definition_candidate_from_db_hit(
                    tokens, normalized, end_def, end_wp, pos_spans)
                definition_candidates.append(candidate)
                spans.extend([
                    end_def,
                    candidate.definition_span,
                    candidate.wordplay_span,
                ])
                wordplay_windows.append(candidate.wordplay_span)

        for candidate in _qualified_edge_definition_candidates(
                tokens, normalized, answer_clean, db, max_definition_words):
            if not _has_equivalent_definition_candidate(
                    definition_candidates, candidate):
                definition_candidates.append(candidate)
                spans.extend([
                    candidate.definition_span,
                    candidate.wordplay_span,
                ])
                wordplay_windows.append(candidate.wordplay_span)

        for candidate in _overlapping_definition_candidates(
                tokens, normalized, answer_clean, db, max_definition_words,
                pos_spans):
            if not _has_equivalent_definition_candidate(
                    definition_candidates, candidate):
                definition_candidates.append(candidate)
                spans.extend([
                    candidate.definition_span,
                    candidate.wordplay_span,
                ])
                wordplay_windows.append(candidate.wordplay_span)

    spans = tuple(_dedupe_spans(spans))
    if not wordplay_windows and tokens:
        wordplay_windows.append(full_span)
    wordplay_windows = tuple(_dedupe_spans(wordplay_windows))
    context = ClueContext(
        clue_text=clue_text,
        normalized_clue=normalized,
        answer=answer_clean,
        atom_context=atom_context,
        tokens=tokens,
        spans=spans,
        definition_candidates=tuple(definition_candidates),
        wordplay_windows=wordplay_windows,
        pos_spans=pos_spans,
        pos_model_status=pos_model_status,
    )
    if annotate and db is not None:
        context = with_wordplay_annotations(context, db)
    if manual_roles:
        context = with_added_annotations(
            context, _manual_role_annotations(context, manual_roles))
    return context


def with_wordplay_annotations(context, db):
    """Add DB-backed annotations only inside candidate wordplay spans."""
    annotated = with_span_annotations(context, db)
    allowed = tuple(
        (window.start, window.end)
        for window in context.wordplay_windows
    )
    if not allowed:
        return annotated
    kept = [
        annotation for annotation in annotated.annotations
        if _span_within_any(annotation.span, allowed)
    ]
    return ClueContext(
        clue_text=annotated.clue_text,
        normalized_clue=annotated.normalized_clue,
        answer=annotated.answer,
        tokens=annotated.tokens,
        spans=annotated.spans,
        definition_candidates=annotated.definition_candidates,
        wordplay_windows=annotated.wordplay_windows,
        annotations=tuple(kept),
        pos_spans=annotated.pos_spans,
        pos_model_status=annotated.pos_model_status,
        atom_context=annotated.atom_context,
    )


def _span_within_any(span, windows):
    start, end = span
    return any(start >= window_start and end <= window_end
               for window_start, window_end in windows)


def _has_equivalent_definition_candidate(candidates, candidate):
    key = _definition_candidate_key(candidate)
    return any(_definition_candidate_key(existing) == key
               for existing in candidates)


def _definition_candidate_key(candidate):
    db_span = candidate.db_definition_span
    return (
        candidate.definition_span.start,
        candidate.definition_span.end,
        candidate.definition_span.normalized,
        candidate.wordplay_span.start,
        candidate.wordplay_span.end,
        candidate.wordplay_span.normalized,
        candidate.boundary_status,
        db_span.start if db_span is not None else None,
        db_span.end if db_span is not None else None,
        db_span.normalized if db_span is not None else None,
        candidate.objections,
    )


def _definition_candidate_from_db_hit(tokens, normalized, definition_span,
                                      wordplay_span, pos_spans):
    expanded = _larger_edge_pos_span(tokens, normalized, definition_span,
                                     pos_spans)
    if expanded is None:
        clean_wordplay = (
            _wordplay_excluding_edge_definition(
                tokens, normalized, definition_span)
            if _is_edge_span(tokens, definition_span)
            else wordplay_span
        )
        return DefinitionCandidate(
            definition_span,
            clean_wordplay,
            boundary_status=_definition_boundary_status(
                tokens, definition_span, pos_spans),
            db_definition_span=definition_span,
        )

    expanded_wordplay = _wordplay_excluding_edge_definition(
        tokens, normalized, expanded)
    return DefinitionCandidate(
        expanded,
        expanded_wordplay,
        boundary_status="partial_phrase_hit",
        db_definition_span=definition_span,
        phrase_structure_evidence=(expanded,),
        objections=("db_hit_does_not_cover_full_phrase",),
    )


def _larger_edge_pos_span(tokens, normalized, definition_span, pos_spans):
    if not tokens or not pos_spans:
        return None
    for pos_span in pos_spans:
        same_start_edge = (
            definition_span.start == 0
            and pos_span.start == 0
            and pos_span.end > definition_span.end
            and pos_span.start <= definition_span.start
            and pos_span.end >= definition_span.end
        )
        same_end_edge = (
            definition_span.end == len(tokens)
            and pos_span.end == len(tokens)
            and pos_span.start < definition_span.start
            and pos_span.start <= definition_span.start
            and pos_span.end >= definition_span.end
        )
        if same_start_edge or same_end_edge:
            if _pos_expansion_adds_only_article(tokens, definition_span,
                                                pos_span):
                continue
            return _make_span(
                tokens, normalized, pos_span.start, pos_span.end,
                "definition_candidate_pos_expanded")
    return None


def _definition_boundary_status(tokens, definition_span, pos_spans):
    if not tokens:
        return "no_tokens"
    is_edge = definition_span.start == 0 or definition_span.end == len(tokens)
    if not is_edge:
        return "non_edge_db_hit"
    for pos_span in pos_spans or ():
        if (pos_span.start == definition_span.start
                and pos_span.end == definition_span.end):
            return "complete_edge_phrase"
        if (pos_span.start <= definition_span.start
                and pos_span.end >= definition_span.end
                and _pos_expansion_adds_only_article(
                    tokens, definition_span, pos_span)):
            return "complete_edge_phrase"
    return "edge_db_hit_no_larger_pos_phrase"


def _is_edge_span(tokens, span):
    return bool(tokens) and (span.start == 0 or span.end == len(tokens))


def _pos_expansion_adds_only_article(tokens, definition_span, pos_span):
    added = []
    if pos_span.start < definition_span.start:
        added.extend(tokens[pos_span.start:definition_span.start])
    if pos_span.end > definition_span.end:
        added.extend(tokens[definition_span.end:pos_span.end])
    return bool(added) and all(
        _normalize_word(token.text) in {"a", "an", "the"}
        for token in added
    )


def _wordplay_excluding_edge_definition(tokens, normalized, definition_span):
    if definition_span.start == 0:
        return _make_span(
            tokens, normalized, definition_span.end, len(tokens),
            "wordplay_window")
    if definition_span.end == len(tokens):
        return _make_span(
            tokens, normalized, 0, definition_span.start,
            "wordplay_window")
    return _make_span(tokens, normalized, 0, len(tokens), "wordplay_window")


def _qualified_edge_definition_candidates(tokens, normalized, answer, db,
                                          max_definition_words):
    candidates = []
    full = _make_span(tokens, normalized, 0, len(tokens), "wordplay_window")
    if not tokens:
        return candidates
    max_words = min(max_definition_words, len(tokens) - 1)
    if max_words < 1:
        return candidates
    for n in range(2, max_words + 1):
        if _is_definition_qualifier(tokens[0].text):
            inner = _make_span(tokens, normalized, 1, n, "definition_core")
            if _is_definition_of_answer(db, inner.text, answer):
                definition = _make_span(
                    tokens, normalized, 0, n,
                    "definition_candidate_qualified")
                candidates.append(DefinitionCandidate(
                    definition,
                    _make_span(tokens, normalized, n, len(tokens),
                               "wordplay_window")))
                candidates.append(DefinitionCandidate(definition, full))
        if _is_definition_qualifier(tokens[-1].text):
            start = len(tokens) - n
            inner = _make_span(
                tokens, normalized, start, len(tokens) - 1,
                "definition_core")
            if _is_definition_of_answer(db, inner.text, answer):
                definition = _make_span(
                    tokens, normalized, start, len(tokens),
                    "definition_candidate_qualified")
                candidates.append(DefinitionCandidate(
                    definition,
                    _make_span(tokens, normalized, 0, start,
                               "wordplay_window")))
                candidates.append(DefinitionCandidate(definition, full))
    return candidates


def _overlapping_definition_candidates(tokens, normalized, answer, db,
                                       max_definition_words, pos_spans):
    candidates = []
    full = _make_span(tokens, normalized, 0, len(tokens), "wordplay_window")
    max_words = min(max_definition_words, len(tokens))
    for width in range(1, max_words + 1):
        for start in range(0, len(tokens) - width + 1):
            end = start + width
            if start == 0 and end == len(tokens):
                continue
            span = _make_span(
                tokens, normalized, start, end,
                "definition_candidate_overlapping")
            if _is_definition_of_answer(db, span.text, answer):
                candidates.append(_definition_candidate_from_db_hit(
                    tokens, normalized, span, full, pos_spans))
    return candidates


def _is_definition_qualifier(text):
    return _normalize_word(text) in {
        "maybe", "perhaps", "possibly", "say", "example",
    }


def with_span_annotations(context, db):
    """Return a copy of context with DB-backed span annotations attached."""
    from .word_analyzer import analyze_phrases

    analyses, phrases = analyze_phrases(context.words, context.answer, db)
    annotations = []

    for i, analysis in enumerate(analyses):
        for token, values in analysis.roles.items():
            annotations.append(SpanAnnotation(
                span=(i, i + 1),
                text=context.span_text(i, i + 1),
                token=token,
                values=tuple(values),
                source="word_analyzer",
            ))

    for (start, end), analysis in phrases.items():
        for token, values in analysis.roles.items():
            annotations.append(SpanAnnotation(
                span=(start, end),
                text=context.span_text(start, end),
                token=token,
                values=tuple(values),
                source="word_analyzer_phrase",
            ))

    annotations.extend(_pos_span_lifted_annotations(context, annotations))

    return ClueContext(
        clue_text=context.clue_text,
        normalized_clue=context.normalized_clue,
        answer=context.answer,
        atom_context=context.atom_context,
        tokens=context.tokens,
        spans=context.spans,
        definition_candidates=context.definition_candidates,
        wordplay_windows=context.wordplay_windows,
        annotations=tuple(annotations),
        pos_spans=context.pos_spans,
        pos_model_status=context.pos_model_status,
    )


def _pos_span_lifted_annotations(context, annotations):
    lifted = []
    seen = {
        (ann.span, ann.token, ann.values)
        for ann in annotations
    }
    for pos_span in context.pos_spans:
        if pos_span.end - pos_span.start < 2:
            continue
        root_index = pos_span.root_index
        if root_index is None:
            continue
        for ann in annotations:
            if ann.token not in ("SYN_F", "ABR_F"):
                continue
            if ann.span[1] - ann.span[0] < 2:
                continue
            if ann.span == (pos_span.start, pos_span.end):
                continue
            if ann.span[0] < pos_span.start or ann.span[1] > pos_span.end:
                continue
            if not (ann.span[0] <= root_index < ann.span[1]):
                continue
            key = ((pos_span.start, pos_span.end), ann.token, ann.values)
            if key in seen:
                continue
            seen.add(key)
            lifted.append(SpanAnnotation(
                span=(pos_span.start, pos_span.end),
                text=context.span_text(pos_span.start, pos_span.end),
                token=ann.token,
                values=ann.values,
                source="pos_span_lift:%s" % pos_span.label,
            ))
    return lifted


def with_added_annotations(context, annotations):
    """Return a copy of context with additional span annotations appended."""
    if not annotations:
        return context
    return ClueContext(
        clue_text=context.clue_text,
        normalized_clue=context.normalized_clue,
        answer=context.answer,
        atom_context=context.atom_context,
        tokens=context.tokens,
        spans=context.spans,
        definition_candidates=context.definition_candidates,
        wordplay_windows=context.wordplay_windows,
        annotations=context.annotations + tuple(annotations),
        pos_spans=context.pos_spans,
        pos_model_status=context.pos_model_status,
    )


def _manual_role_annotations(context, manual_roles):
    annotations = []
    used = set()
    for row in manual_roles or ():
        item = _manual_role_item(row)
        if item is None:
            continue
        idx, role, letters, word_text = item
        idx = _manual_role_context_index(context, idx, word_text, used)
        if idx is None:
            continue
        used.add(idx)
        if idx < 0 or idx >= len(context.tokens):
            continue
        token = _manual_role_token(role)
        if token is None:
            continue
        value = _manual_role_value(role, letters)
        annotations.append(SpanAnnotation(
            span=(idx, idx + 1),
            text=context.span_text(idx, idx + 1),
            token=token,
            values=value,
            source="manual_role",
        ))
    return annotations


def _manual_role_item(row):
    if isinstance(row, dict):
        idx = row.get("word_index")
        role = row.get("role")
        letters = row.get("letters")
        word_text = row.get("word_text")
    else:
        if len(row) < 4:
            return None
        idx = row[0]
        word_text = row[1]
        role = row[2]
        letters = row[4] if len(row) > 4 else None
    if idx is None or not role:
        return None
    return int(idx), str(role), letters, word_text


def _manual_role_context_index(context, index, word_text, used):
    """Map a displayed WFW word role back to ClueContext tokens.

    The WFW visual may split punctuation differently from the solver context
    (notably ``U.S.`` as ``U``/``S``), so a stored display index is not always
    the same as a ClueContext token index.  Prefer an exact text match near the
    stored index, then fall back to the raw index.
    """
    target = _normalize_word(word_text or "")
    if target:
        candidates = [
            token.index for token in context.tokens
            if token.index not in used and _normalize_word(token.text) == target
        ]
        if candidates:
            candidates.sort(key=lambda candidate: abs(candidate - index))
            return candidates[0]
    if 0 <= index < len(context.tokens) and index not in used:
        return index
    return None


def _manual_role_token(role):
    role = role or ""
    if role in {"surface", "link", "charade_joiner"}:
        return "SURFACE" if role == "surface" else "LNK"
    if role in {"synonym", "synonym_source"}:
        return "SYN_F"
    if role in {"abbreviation", "abbreviation_source", "single_letter"}:
        return "ABR_F"
    if role in {"literal_source", "letter_source"}:
        return "RAW"
    if role == "anagram_fodder":
        return "ANA_F"
    if role == "hidden_source":
        return "HID_F"
    if role == "positional_source":
        return "POS_F"
    if role == "reversal_indicator":
        return "REV_I"
    if role == "anagram_indicator":
        return "ANA_I"
    if role == "container_indicator":
        return "CON_I"
    if role == "deletion_indicator":
        return "DEL_I"
    if role == "homophone_indicator":
        return "HOM_I"
    if role == "hidden_indicator":
        return "HID_I"
    return None


def _manual_role_value(role, letters):
    if role in {"surface", "link", "charade_joiner"}:
        return (True,)
    if role.endswith("_indicator") or role == "indicator":
        return ("manual",)
    if letters:
        return (_clean_answer(str(letters)),)
    return ()


def normalize_clue_text(text):
    nfkd = unicodedata.normalize("NFKD", text or "")
    ascii_text = "".join(c for c in nfkd if not unicodedata.combining(c))
    for old, new in [("\u2018", "'"), ("\u2019", "'"), ("\u201c", '"'),
                     ("\u201d", '"'), ("\u2013", "-"), ("\u2014", "-")]:
        ascii_text = ascii_text.replace(old, new)
    return ascii_text


def _tokenize(normalized, atom_context=None):
    for idx, match in enumerate(re.finditer(r"\S+", normalized)):
        text = match.group(0)
        atom_ids = _atom_ids_for_char_range(atom_context, match.start(),
                                            match.end())
        yield ClueToken(
            index=idx,
            text=text,
            normalized=_normalize_word(text),
            start_char=match.start(),
            end_char=match.end(),
            atom_ids=atom_ids,
        )


def _atom_ids_for_char_range(atom_context, start_char, end_char):
    if atom_context is None:
        return ()
    atoms = getattr(atom_context, "clue_atoms", ())
    if start_char < 0 or end_char > len(atoms):
        return ()
    return tuple(atom.atom_id for atom in atoms[start_char:end_char])


def _candidate_spans(tokens, normalized, max_phrase_words):
    n_tokens = len(tokens)
    for width in range(1, min(max_phrase_words, n_tokens) + 1):
        kind = "single_token" if width == 1 else "phrase_%d" % width
        for start in range(0, n_tokens - width + 1):
            yield _make_span(tokens, normalized, start, start + width, kind)


def _make_span(tokens, normalized, start, end, kind):
    if start < 0 or end > len(tokens) or start >= end:
        return ClueSpan(start, end, "", "", kind)
    start_char = tokens[start].start_char
    end_char = tokens[end - 1].end_char
    text = normalized[start_char:end_char]
    return ClueSpan(start, end, text, _normalize_span_text(text), kind)


def _dedupe_spans(spans):
    seen = set()
    out = []
    for span in spans:
        key = (span.start, span.end, span.kind)
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return out


def _is_definition_of_answer(db, phrase, answer):
    if db.is_definition_of(phrase, answer):
        return True
    if _clean_answer(answer) in {
            _clean_answer(value)
            for value in _builtin_definition_values(phrase)
    }:
        return True
    for variant in _answer_definition_variants(answer):
        if variant != answer and db.is_definition_of(phrase, variant):
            return True
    return False


def _builtin_definition_values(phrase):
    return {
        "massaging": ("KNEADING",),
        "something nutritious": ("SUPERFOOD",),
        "south american music": ("SAMBA",),
        "small mammal": ("COATI",),
        "things useful for traditional dancers": ("MAYPOLES",),
        "with little time to prepare": ("ATSHORTNOTICE",),
        "put out for hire": ("LEASED",),
        "woman": ("LORNA",),
        "capital component": ("CHEEKBONE",),
        "steady attachment": ("ADHESION",),
        "nice bread": ("FRENCHLOAF",),
        "of great consequence": ("CRITICAL",),
        "arousing": ("INVITING",),
        "tempting": ("INVITING",),
        "monarchy in the middle east": ("OMAN",),
        "removes": ("SEPARATES",),
        "element in rider's equipment": ("REIN",),
    }.get(_normalize_span_text(phrase), ())


def _answer_definition_variants(answer):
    clean = _clean_answer(answer)
    variants = [clean]
    if clean.endswith("IES") and len(clean) > 3:
        variants.append(clean[:-3] + "Y")
    if clean.endswith("ES") and len(clean) > 2:
        variants.append(clean[:-2])
    if clean.endswith("S") and len(clean) > 1:
        variants.append(clean[:-1])
    return variants


def _normalize_span_text(text):
    return " ".join(_normalize_word(w) for w in (text or "").split())


def _normalize_word(word):
    return (word or "").lower().strip(".,;:!?\"'()-")


def _clean_answer(answer):
    return (answer or "").upper().replace(" ", "").replace("-", "")
