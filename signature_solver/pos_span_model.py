"""POS/chunk model adapter for Stage One span proposal.

This module is deliberately model-backed. If no supported POS/chunking model
is installed, it reports that status and emits no spans rather than pretending
fixed windows are linguistic chunks.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import importlib.util


@dataclass(frozen=True)
class POSSpan:
    start: int
    end: int
    text: str
    normalized: str
    label: str
    root_index: int | None = None
    root_text: str | None = None
    pos_tags: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    source: str = "pos_model"
    confidence: str = "model"

    def as_dict(self):
        return {
            "start": self.start,
            "end": self.end,
            "text": self.text,
            "normalized": self.normalized,
            "label": self.label,
            "root_index": self.root_index,
            "root_text": self.root_text,
            "pos_tags": list(self.pos_tags),
            "dependencies": list(self.dependencies),
            "source": self.source,
            "confidence": self.confidence,
        }


def propose_pos_spans(normalized_text, clue_tokens, model=None):
    """Return ``(spans, status)`` from a real POS/chunk model.

    ``model`` is an injectable test seam with ``extract_spans(text, tokens)``.
    In production we currently support spaCy with an installed English model.
    """
    if model is not None:
        spans = tuple(model.extract_spans(normalized_text, clue_tokens))
        return _dedupe_pos_spans(spans), "available:injected"

    adapter, status = _default_adapter()
    if adapter is None:
        return (), status
    return _dedupe_pos_spans(adapter.extract_spans(
        normalized_text, clue_tokens)), status


@lru_cache(maxsize=1)
def _default_adapter():
    if importlib.util.find_spec("spacy") is None:
        return None, "unavailable:spacy_not_installed"
    try:
        import spacy
    except Exception as exc:
        return None, "unavailable:spacy_import_failed:%s" % type(exc).__name__

    for model_name in ("en_core_web_sm", "en_core_web_md", "en_core_web_lg"):
        if importlib.util.find_spec(model_name) is None:
            continue
        try:
            return _SpacyAdapter(spacy.load(model_name)), (
                "available:spacy:%s" % model_name)
        except Exception as exc:
            return None, (
                "unavailable:spacy_model_load_failed:%s:%s"
                % (model_name, type(exc).__name__))
    return None, "unavailable:spacy_english_model_not_installed"


class _SpacyAdapter:
    def __init__(self, nlp):
        self.nlp = nlp

    def extract_spans(self, normalized_text, clue_tokens):
        doc = self.nlp(normalized_text)
        spans = []
        for chunk in getattr(doc, "noun_chunks", ()):
            span = _char_span_to_clue_span(
                clue_tokens, chunk.start_char, chunk.end_char)
            if span is None or span[1] - span[0] < 2:
                continue
            spans.append(_make_pos_span(
                clue_tokens, span, "noun_chunk",
                root_token=getattr(chunk, "root", None),
                model_tokens=tuple(chunk)))

        for token in doc:
            if token.dep_ not in {"compound", "poss", "amod"}:
                continue
            head = token.head
            left = min(token.i, head.i)
            right = max(token.i, head.i) + 1
            if right - left < 2:
                continue
            chunk = doc[left:right]
            span = _char_span_to_clue_span(
                clue_tokens, chunk.start_char, chunk.end_char)
            if span is None:
                continue
            spans.append(_make_pos_span(
                clue_tokens, span, "compound_like",
                root_token=head, model_tokens=tuple(chunk)))
        return tuple(spans)


def _make_pos_span(clue_tokens, span, label, root_token=None,
                   model_tokens=()):
    start, end = span
    text = _span_text(clue_tokens, start, end)
    root_index = None
    root_text = None
    if root_token is not None:
        root_span = _char_span_to_clue_span(
            clue_tokens, root_token.idx, root_token.idx + len(root_token.text))
        if root_span is not None:
            root_index = root_span[0]
            root_text = clue_tokens[root_span[0]].text
    return POSSpan(
        start=start,
        end=end,
        text=text,
        normalized=" ".join(t.normalized for t in clue_tokens[start:end]),
        label=label,
        root_index=root_index,
        root_text=root_text,
        pos_tags=tuple(getattr(t, "pos_", "") for t in model_tokens),
        dependencies=tuple(getattr(t, "dep_", "") for t in model_tokens),
    )


def _char_span_to_clue_span(clue_tokens, start_char, end_char):
    covered = [
        token.index for token in clue_tokens
        if token.start_char >= start_char and token.end_char <= end_char
    ]
    if not covered:
        return None
    return min(covered), max(covered) + 1


def _span_text(clue_tokens, start, end):
    if start < 0 or end > len(clue_tokens) or start >= end:
        return ""
    return " ".join(token.text for token in clue_tokens[start:end])


def _dedupe_pos_spans(spans):
    seen = set()
    out = []
    for span in spans:
        key = (span.start, span.end, span.label)
        if key in seen:
            continue
        seen.add(key)
        out.append(span)
    return tuple(out)
