"""Build weak operation-derived attachment labels for GT V2 R&D.

This is not solver code. It turns the grammar scaffold into an inspectable
JSONL/Markdown slice where each residue run gets a provisional attachment role
derived from the structured operation.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "grammar_feature_scaffold_2026-05-17.jsonl"
DEFAULT_JSONL = PROJECT_ROOT / "documents" / "operation_attachment_slice_2026-05-17.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "documents" / "operation_attachment_slice_2026-05-17.md"

DIRECTION_WORDS = {
    "back",
    "backing",
    "comeback",
    "east",
    "left",
    "north",
    "over",
    "raised",
    "rejected",
    "repelled",
    "returning",
    "returns",
    "revolting",
    "reverse",
    "reversed",
    "revolutionary",
    "rise",
    "rising",
    "up",
    "upset",
    "west",
}

LOCATOR_WORDS = {
    "centre",
    "end",
    "ends",
    "first",
    "front",
    "head",
    "heart",
    "initial",
    "intervals",
    "last",
    "middle",
    "outside",
    "tail",
    "top",
}

ORDER_WORDS = {
    "after",
    "before",
    "behind",
    "following",
    "on",
    "over",
    "under",
}

DBE_MARKERS = {
    "eg",
    "e.g",
    "example",
    "like",
    "maybe",
    "perhaps",
    "say",
}

CONTAINER_WORDS = {
    "about",
    "around",
    "eating",
    "engulfing",
    "holding",
    "in",
    "inside",
    "interrupting",
    "round",
    "swallowing",
    "with",
}

HIDDEN_WORDS = {
    "among",
    "bit",
    "cases",
    "concealing",
    "contains",
    "cover",
    "covers",
    "embraces",
    "entirely",
    "featuring",
    "hidden",
    "houses",
    "in",
    "inside",
    "jackets",
    "kept",
    "little",
    "nurses",
    "part",
    "partially",
    "partly",
    "sandwiches",
    "section",
    "some",
    "somewhat",
}

HOMOPHONE_WORDS = {
    "audience",
    "auditorium",
    "broadcast",
    "did",
    "hear",
    "heard",
    "hears",
    "like",
    "listeners",
    "mentioned",
    "picked",
    "radio",
    "reported",
    "reportedly",
    "say",
    "sound",
    "sounds",
    "speak",
    "speech",
    "spoken",
    "up",
    "we",
    "you",
}

ANAGRAM_WORDS = {
    "awkwardly",
    "bewildered",
    "broken",
    "broadcast",
    "change",
    "changing",
    "criminal",
    "confused",
    "cycling",
    "devised",
    "differently",
    "dressed",
    "eccentric",
    "excited",
    "fancy",
    "formulated",
    "ground",
    "mess",
    "new",
    "novel",
    "order",
    "out",
    "playing",
    "possibly",
    "reconstructed",
    "repairing",
    "replaced",
    "reviewed",
    "sadly",
    "shredded",
    "sozzled",
    "suspect",
    "trouble",
    "unusual",
    "unusually",
    "upset",
    "wobbling",
    "wrongly",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def wordplay_tokens(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in record.get("tokens", []) if t.get("label") in {"SOURCE", "RESIDUE"}]


def residue_runs(tokens: list[dict[str, Any]]) -> list[tuple[int, int]]:
    runs = []
    start = None
    for i, token in enumerate(tokens):
        if token.get("label") == "RESIDUE":
            if start is None:
                start = i
        elif start is not None:
            runs.append((start, i))
            start = None
    if start is not None:
        runs.append((start, len(tokens)))
    return runs


def source_spans(tokens: list[dict[str, Any]]) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i, token in enumerate(tokens):
        if token.get("label") == "SOURCE":
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(tokens)))
    return spans


def definition_spans(tokens: list[dict[str, Any]]) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i, token in enumerate(tokens):
        if token.get("label") == "DEF":
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(tokens)))
    return spans


def definition_span_records(tokens: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for span_index, span in enumerate(definition_spans(tokens)):
        records.append(
            {
                "span": list(span),
                "definition_span_id": span_index,
                "text": text_for(tokens, span),
            }
        )
    return records


def source_span_records(tokens: list[dict[str, Any]], pieces: list[dict[str, Any]]) -> list[dict[str, Any]]:
    records = []
    for span in source_spans(tokens):
        start, end = span
        span_ids = {
            tokens[i].get("source_span_id")
            for i in range(start, end)
            if tokens[i].get("source_span_id") is not None
        }
        piece_indices = sorted(
            {
                tokens[i].get("piece_index")
                for i in range(start, end)
                if tokens[i].get("piece_index") is not None
            }
        )
        piece_records = [
            pieces[piece_index]
            for piece_index in piece_indices
            if isinstance(piece_index, int) and 0 <= piece_index < len(pieces)
        ]
        records.append(
            {
                "span": list(span),
                "source_span_id": min(span_ids) if span_ids else None,
                "piece_indices": piece_indices,
                "mechanisms": sorted(
                    {
                        str(piece.get("mechanism"))
                        for piece in piece_records
                        if piece.get("mechanism") is not None
                    }
                ),
                "letters": [piece.get("letters") for piece in piece_records],
                "text": text_for(tokens, span),
            }
        )
    return records


def text_for(tokens: list[dict[str, Any]], span: tuple[int, int]) -> str:
    start, end = span
    return " ".join(t["text"] for t in tokens[start:end])


def clean_words(tokens: list[dict[str, Any]], span: tuple[int, int]) -> set[str]:
    start, end = span
    return {tokens[i].get("clean", "") for i in range(start, end)}


def adjacent_source_ids(tokens: list[dict[str, Any]], run: tuple[int, int]) -> dict[str, int | None]:
    start, end = run
    left = tokens[start - 1].get("source_span_id") if start > 0 and tokens[start - 1].get("label") == "SOURCE" else None
    right = tokens[end].get("source_span_id") if end < len(tokens) and tokens[end].get("label") == "SOURCE" else None
    return {"left": left, "right": right}


def parser_source_heads(
    tokens: list[dict[str, Any]],
    full_tokens: list[dict[str, Any]],
    run: tuple[int, int],
) -> list[dict[str, Any]]:
    start, end = run
    heads = []
    for i in range(start, end):
        token = tokens[i]
        grammar = token.get("grammar") or {}
        head_i = grammar.get("head_i")
        if head_i is None or head_i < 0 or head_i >= len(full_tokens):
            continue
        head = full_tokens[head_i]
        if head.get("label") == "SOURCE":
            heads.append(
                {
                    "token": token.get("text"),
                    "dep": grammar.get("dep"),
                    "head": head.get("text"),
                    "head_index": head_i,
                    "head_source_span_id": head.get("source_span_id"),
                }
            )
    return heads


def run_near_definition(
    tokens: list[dict[str, Any]],
    full_tokens: list[dict[str, Any]],
    run: tuple[int, int],
) -> bool:
    start, end = run
    for i in range(start, end):
        original_index = tokens[i].get("index")
        if original_index is None:
            continue
        for neighbour in (original_index - 1, original_index + 1):
            if 0 <= neighbour < len(full_tokens) and full_tokens[neighbour].get("label") == "DEF":
                return True
    return False


def weak_attachment_label(operation: str, words: set[str], near_definition: bool) -> str:
    if words & LOCATOR_WORDS:
        return "LOCATOR_SCOPE"
    if operation in {"reversal", "hidden_reversed"} and words & DIRECTION_WORDS:
        return "DIRECTION_OR_ORIENTATION"
    if operation == "homophone" and words & HOMOPHONE_WORDS:
        return "OPERATOR_SCOPE"
    if operation in {"anagram", "deletion+anagram"} and words & ANAGRAM_WORDS:
        return "OPERATOR_SCOPE"
    if operation in {"hidden", "hidden_reversed"} and words & HIDDEN_WORDS:
        return "OPERATOR_SCOPE"
    if near_definition and words & DBE_MARKERS:
        return "DEF_MODIFIER"
    if operation == "container" or words & CONTAINER_WORDS and operation in {"container", "charade"}:
        return "CONTAINER_RELATION"
    if operation == "charade" and words & ORDER_WORDS:
        return "ORDER_OR_POSITION"
    if words and all(w in {"a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of", "on", "the", "to", "with"} for w in words):
        return "CONNECTOR_OR_SURFACE"
    return "UNCLASSIFIED_RESIDUE"


def token_attachment_label(operation: str, word: str, near_definition: bool) -> str:
    words = {word}
    if word in LOCATOR_WORDS:
        return "LOCATOR_SCOPE"
    if operation in {"reversal", "hidden_reversed"} and word in DIRECTION_WORDS:
        return "DIRECTION_OR_ORIENTATION"
    if operation == "homophone" and word in HOMOPHONE_WORDS:
        return "OPERATOR_SCOPE"
    if operation in {"anagram", "deletion+anagram"} and word in ANAGRAM_WORDS:
        return "OPERATOR_SCOPE"
    if operation in {"hidden", "hidden_reversed"} and word in HIDDEN_WORDS:
        return "OPERATOR_SCOPE"
    if near_definition and word in DBE_MARKERS:
        return "DEF_MODIFIER"
    if operation == "container" or word in CONTAINER_WORDS and operation in {"container", "charade"}:
        return "CONTAINER_RELATION"
    if operation == "charade" and word in ORDER_WORDS:
        return "ORDER_OR_POSITION"
    if words and all(w in {"a", "an", "and", "as", "at", "by", "for", "from", "in", "is", "of", "on", "the", "to", "with"} for w in words):
        return "CONNECTOR_OR_SURFACE"
    return "UNCLASSIFIED_RESIDUE"


def token_annotations(
    operation: str,
    tokens: list[dict[str, Any]],
    run: tuple[int, int],
    near_definition: bool,
) -> list[dict[str, Any]]:
    start, end = run
    annotations = []
    for i in range(start, end):
        token = tokens[i]
        clean = token.get("clean", "")
        label = token_attachment_label(operation, clean, near_definition)
        annotations.append(
            {
                "text": token.get("text"),
                "clean": clean,
                "weak_attachment_label": label,
                "block_relationship": relationship_for_label(label),
            }
        )
    return annotations


def needs_split(annotations: list[dict[str, Any]]) -> bool:
    meaningful = {
        item["weak_attachment_label"]
        for item in annotations
        if item["weak_attachment_label"] != "CONNECTOR_OR_SURFACE"
    }
    return len(meaningful) > 1


def relationship_for_label(label: str) -> str:
    if label == "OPERATOR_SCOPE":
        return "OPERATES_ON"
    if label == "LOCATOR_SCOPE":
        return "LOCATES_WITHIN"
    if label == "ORDER_OR_POSITION":
        return "ORDERS"
    if label == "CONTAINER_RELATION":
        return "CONTAINS"
    if label == "SOURCE_INTERNAL":
        return "BELONGS_TO_SOURCE"
    if label == "DEF_MODIFIER":
        return "MODIFIES_DEFINITION"
    if label == "DIRECTION_OR_ORIENTATION":
        return "AWAITING_SCOPE"
    if label == "CONNECTOR_OR_SURFACE":
        return "SURFACE_CONNECTS"
    return "UNRESOLVED"


def scope_status_for_label(label: str) -> str:
    if label in {"DIRECTION_OR_ORIENTATION", "ORDER_OR_POSITION"}:
        return "answer_aware_scope_needed"
    if label in {"OPERATOR_SCOPE", "CONTAINER_RELATION", "LOCATOR_SCOPE"}:
        return "weakly_scoped_from_operation"
    if label == "SOURCE_INTERNAL":
        return "candidate_source_absorption"
    if label == "DEF_MODIFIER":
        return "definition_modifier_candidate"
    if label == "CONNECTOR_OR_SURFACE":
        return "surface_only_until_grammar_check"
    return "unresolved"


def grammar_evidence(
    tokens: list[dict[str, Any]],
    full_tokens: list[dict[str, Any]],
    run: tuple[int, int],
) -> list[dict[str, Any]]:
    start, end = run
    evidence = []
    for i in range(start, end):
        token = tokens[i]
        grammar = token.get("grammar") or {}
        head_i = grammar.get("head_i")
        if head_i is None or head_i < 0 or head_i >= len(full_tokens):
            head = None
        else:
            head = full_tokens[head_i]
        evidence.append(
            {
                "token": token.get("text"),
                "clean": token.get("clean"),
                "mid_pos": grammar.get("mid_pos"),
                "dep": grammar.get("dep"),
                "head_text": grammar.get("head_text"),
                "head_label": head.get("label") if head else None,
                "head_source_span_id": head.get("source_span_id") if head else None,
            }
        )
    return evidence


def build_slice(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for record in records:
        full_tokens = record.get("tokens", [])
        tokens = wordplay_tokens(record)
        spans = source_span_records(tokens, record.get("pieces", []))
        definitions = definition_span_records(full_tokens)
        for run in residue_runs(tokens):
            words = clean_words(tokens, run)
            near_definition = run_near_definition(tokens, full_tokens, run)
            label = weak_attachment_label(record.get("operation", ""), words, near_definition)
            annotations = token_annotations(record.get("operation", ""), tokens, run, near_definition)
            rows.append(
                {
                    "id": record.get("id"),
                    "source": record.get("source"),
                    "puzzle_number": record.get("puzzle_number"),
                    "clue": record.get("clue"),
                    "answer": record.get("answer"),
                    "operation": record.get("operation"),
                    "residue_run": {
                        "span": run,
                        "text": text_for(tokens, run),
                        "clean_words": sorted(words),
                    },
                    "weak_attachment_label": label,
                    "token_annotations": annotations,
                    "needs_split": needs_split(annotations),
                    "block_relationship": relationship_for_label(label),
                    "scope_status": scope_status_for_label(label),
                    "near_definition": near_definition,
                    "adjacent_source_span_ids": adjacent_source_ids(tokens, run),
                    "parser_source_heads": parser_source_heads(tokens, full_tokens, run),
                    "grammar_evidence": grammar_evidence(tokens, full_tokens, run),
                    "source_spans": spans,
                    "definition_spans": definitions,
                    "wordplay_labels": " ".join(t.get("label", "?")[0] for t in tokens),
                }
            )
    return rows


def write_outputs(rows: list[dict[str, Any]], jsonl_path: Path, report_path: Path) -> None:
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for row in rows:
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

    label_counts = Counter(row["weak_attachment_label"] for row in rows)
    by_operation: dict[str, Counter[str]] = defaultdict(Counter)
    for row in rows:
        by_operation[row["operation"]][row["weak_attachment_label"]] += 1

    lines = [
        "# Operation Attachment Slice",
        "",
        "Date: 2026-05-17",
        "",
        "This is a weakly-labelled R&D slice for residue attachment.",
        "Labels are provisional and operation-derived; they are not production truth.",
        "",
        f"Residue runs: `{len(rows)}`",
        "",
        "## Label Counts",
        "",
    ]
    for label, count in label_counts.most_common():
        lines.append(f"- `{label}`: {count}")

    lines.extend(["", "## Labels By Operation", ""])
    for operation in sorted(by_operation):
        items = ", ".join(f"`{label}`={count}" for label, count in by_operation[operation].most_common())
        lines.append(f"- `{operation}`: {items}")

    lines.extend(["", "## Examples", ""])
    for row in rows[:40]:
        lines.append(f"- `{row['clue']}` ({row['answer']})")
        lines.append(
            f"  Operation: `{row['operation']}`; label: `{row['weak_attachment_label']}`; relationship: `{row['block_relationship']}`"
        )
        lines.append(f"  Scope status: `{row['scope_status']}`")
        lines.append(f"  Residue: `{row['residue_run']['text']}`")
        if row.get("needs_split"):
            parts = ", ".join(
                f"{item['text']}->{item['weak_attachment_label']}"
                for item in row.get("token_annotations", [])
            )
            lines.append(f"  Needs split: `{parts}`")
        lines.append("  Sources: " + "; ".join(f"`{s['text']}`" for s in row["source_spans"]))
        if row.get("definition_spans"):
            lines.append(
                "  Definitions: " + "; ".join(f"`{s['text']}`" for s in row["definition_spans"])
            )
        if row["parser_source_heads"]:
            heads = ", ".join(f"{h['token']}->{h['head']}:{h['dep']}" for h in row["parser_source_heads"])
            lines.append(f"  Parser source heads: `{heads}`")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This slice is intended for manual inspection and later classifier experiments.",
            "The useful next move is to inspect one weak label at a time and decide whether the relationship is anatomically right.",
            "The `block_relationship` and `scope_status` fields are deliberately separate from the label so that operation, locator, and scope can be preserved without flattening.",
            "",
        ]
    )
    report_path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    records = load_records(args.input)
    rows = build_slice(records)
    write_outputs(rows, args.jsonl_out, args.report_out)
    print(f"records={len(records)}")
    print(f"residue_runs={len(rows)}")
    print(f"jsonl={args.jsonl_out}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
