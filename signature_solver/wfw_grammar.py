"""First-class grammar evidence for WFW proof construction."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class WFWGrammarSpan:
    span_id: str
    label: str
    token_indices: tuple[int, ...]
    token_ids: tuple[str, ...]
    atom_ids: tuple[str, ...]
    root_token_index: int | None = None
    root_token_id: str | None = None
    root_text: str | None = None
    pos_tags: tuple[str, ...] = ()
    dependencies: tuple[str, ...] = ()
    source: str = "pos_model"
    confidence: str = "model"

    def as_dict(self):
        return {
            "span_id": self.span_id,
            "label": self.label,
            "token_indices": list(self.token_indices),
            "token_ids": list(self.token_ids),
            "atom_ids": list(self.atom_ids),
            "root_token_index": self.root_token_index,
            "root_token_id": self.root_token_id,
            "root_text": self.root_text,
            "pos_tags": list(self.pos_tags),
            "dependencies": list(self.dependencies),
            "source": self.source,
            "confidence": self.confidence,
        }


@dataclass(frozen=True)
class WFWGrammarEvidence:
    status: str
    spans: tuple[WFWGrammarSpan, ...]
    objections: tuple[str, ...] = ()

    def as_dict(self):
        return {
            "status": self.status,
            "spans": [span.as_dict() for span in self.spans],
            "objections": list(self.objections),
        }


def grammar_evidence_from_clue_context(atom_context, clue_context):
    """Map existing POS/chunk spans onto WFW original tokens and atoms."""
    token_map, objections = _align_tokens(atom_context, clue_context)
    if objections:
        return WFWGrammarEvidence(
            status="unusable",
            spans=(),
            objections=tuple(objections),
        )

    spans = []
    for idx, pos_span in enumerate(getattr(clue_context, "pos_spans", ())):
        if any(i not in token_map for i in range(pos_span.start, pos_span.end)):
            continue
        token_indices = tuple(
            token_map[i] for i in range(pos_span.start, pos_span.end))
        atom_tokens = tuple(atom_context.clue_tokens[i] for i in token_indices)
        atom_ids = tuple(
            atom_id
            for token in atom_tokens
            for atom_id in token.atom_ids
        )
        root_token_index = None
        root_token_id = None
        if pos_span.root_index is not None:
            root_token_index = token_map.get(pos_span.root_index)
            if root_token_index is not None:
                root_token_id = atom_context.clue_tokens[
                    root_token_index].token_id
        spans.append(WFWGrammarSpan(
            span_id="grammar_span_%04d" % idx,
            label=pos_span.label,
            token_indices=token_indices,
            token_ids=tuple(token.token_id for token in atom_tokens),
            atom_ids=atom_ids,
            root_token_index=root_token_index,
            root_token_id=root_token_id,
            root_text=pos_span.root_text,
            pos_tags=tuple(pos_span.pos_tags),
            dependencies=tuple(pos_span.dependencies),
            source=pos_span.source,
            confidence=pos_span.confidence,
        ))

    status = "available" if spans else getattr(
        clue_context, "pos_model_status", "not_requested")
    return WFWGrammarEvidence(status=status, spans=tuple(spans))


def _align_tokens(atom_context, clue_context):
    """Align ClueContext word tokens to WFW original tokens.

    WFW tokens preserve standalone punctuation. ClueContext usually attaches
    punctuation to word tokens, so alignment compares normalized alphanumeric
    text and skips punctuation-only WFW tokens.
    """
    mapping = {}
    objections = []
    wfw_index = 0
    for clue_token in clue_context.tokens:
        target = _token_key(clue_token.text)
        if not target:
            continue
        while (wfw_index < len(atom_context.clue_tokens)
               and not _token_key(atom_context.clue_tokens[wfw_index].text)):
            wfw_index += 1
        if wfw_index >= len(atom_context.clue_tokens):
            objections.append("missing_wfw_token:%s" % clue_token.text)
            break
        actual = _token_key(atom_context.clue_tokens[wfw_index].text)
        if actual != target:
            objections.append("token_alignment:%s!=%s" % (actual, target))
            break
        mapping[clue_token.index] = wfw_index
        wfw_index += 1
    return mapping, objections


def _token_key(text):
    return "".join(char.upper() for char in (text or "") if char.isalnum())
