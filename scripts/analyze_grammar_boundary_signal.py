"""Analyze grammar signal against supervised GT V2 block boundaries.

This script reads the grammar feature scaffold and asks whether spaCy features
help with block anatomy: source span cohesion, residue attachment, and glue word
ambiguity. It does not attempt to solve clues.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "grammar_feature_scaffold_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "grammar_boundary_signal_2026-05-17.md"


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def grammar(token: dict[str, Any]) -> dict[str, Any]:
    return token.get("grammar") or {}


def head_token(tokens: list[dict[str, Any]], token: dict[str, Any]) -> dict[str, Any] | None:
    head_i = grammar(token).get("head_i")
    if head_i is None or head_i < 0 or head_i >= len(tokens):
        return None
    return tokens[head_i]


def compact_counts(counter: Counter[str], n: int = 10) -> str:
    return ", ".join(f"`{key}`={count}" for key, count in counter.most_common(n)) or "`none`"


def same_source_span(token: dict[str, Any], head: dict[str, Any] | None) -> bool:
    if head is None:
        return False
    return (
        token.get("label") == "SOURCE"
        and head.get("label") == "SOURCE"
        and token.get("source_span_id") is not None
        and token.get("source_span_id") == head.get("source_span_id")
    )


def residue_text(record: dict[str, Any]) -> str:
    return " ".join(t["text"].lower() for t in record.get("tokens", []) if t.get("label") == "RESIDUE")


def labels(record: dict[str, Any]) -> str:
    return " ".join((t.get("label") or "?")[0] for t in record.get("tokens", []))


def example_lines(record: dict[str, Any], note: str) -> list[str]:
    return [
        f"- `{record['clue']}` ({record['answer']})",
        f"  Operation: `{record['operation']}`; {note}",
        f"  Residue: `{residue_text(record) or '<empty>'}`",
        f"  Labels: `{labels(record)}`",
    ]


def write_report(records: list[dict[str, Any]], path: Path) -> None:
    label_pos: dict[str, Counter[str]] = defaultdict(Counter)
    label_dep: dict[str, Counter[str]] = defaultdict(Counter)
    head_label_edges = Counter()
    source_head_edges = Counter()
    residue_head_edges = Counter()
    prep_head_edges = Counter()
    glue_head_edges = Counter()
    glue_dep_edges = Counter()
    operationish_edges = Counter()
    operation_prep_edges: dict[str, Counter[str]] = defaultdict(Counter)
    source_span_internal_heads = 0
    source_span_head_total = 0
    source_tokens = 0
    residue_tokens = 0
    records_with_grammar = 0
    boundary_cross_examples = []
    glue_examples = []
    direction_examples = []

    for record in records:
        tokens = record.get("tokens", [])
        if any("grammar" in t for t in tokens):
            records_with_grammar += 1
        for token in tokens:
            g = grammar(token)
            if not g:
                continue
            label = token.get("label") or "?"
            mid = g.get("mid_pos") or "?"
            dep = g.get("dep") or "?"
            label_pos[label][mid] += 1
            label_dep[label][dep] += 1
            head = head_token(tokens, token)
            head_label = head.get("label") if head else "<none>"
            edge = f"{label}->{head_label}"
            head_label_edges[edge] += 1

            if label == "SOURCE":
                source_tokens += 1
                if head and head.get("label") == "SOURCE":
                    source_head_edges[
                        "same-span" if same_source_span(token, head) else "other-source-span"
                    ] += 1
                    source_span_head_total += 1
                    if same_source_span(token, head):
                        source_span_internal_heads += 1
                else:
                    source_head_edges[f"outside->{head_label}"] += 1

            if label == "RESIDUE":
                residue_tokens += 1
                residue_head_edges[f"{token['clean']}->{head_label}"] += 1

            if mid == "P":
                prep_head_edges[f"{token['clean']}:{label}->{head_label}"] += 1
                operation_prep_edges[record["operation"]][f"{token['clean']}:{label}->{head_label}"] += 1

            if token.get("is_glue_word"):
                glue_head_edges[f"{token['clean']}:{label}->{head_label}"] += 1
                glue_dep_edges[f"{token['clean']}:{label}:{dep}"] += 1
                if len(glue_examples) < 8 and label == "RESIDUE" and head_label == "SOURCE":
                    glue_examples.append(
                        (
                            record,
                            f"`{token['text']}` heads to SOURCE `{head['text'] if head else '?'}` as `{dep}`",
                        )
                    )

            if token.get("is_operationish_word"):
                operationish_edges[f"{token['clean']}:{label}:{dep}->{head_label}"] += 1

            if label == "SOURCE" and head and head.get("label") == "RESIDUE":
                if len(boundary_cross_examples) < 8:
                    boundary_cross_examples.append(
                        (
                            record,
                            f"SOURCE `{token['text']}` depends on RESIDUE `{head['text']}` as `{dep}`",
                        )
                    )

            if token["clean"] in {"up", "back", "east", "reverse", "rising"} and len(direction_examples) < 8:
                direction_examples.append(
                    (
                        record,
                        f"direction `{token['text']}` labelled `{label}` heads to `{head['text'] if head else '?'}` / `{head_label}`",
                    )
                )

    internal_rate = source_span_internal_heads / source_span_head_total if source_span_head_total else 0.0

    lines = [
        "# Grammar Boundary Signal",
        "",
        "Date: 2026-05-17",
        "",
        "This report tests whether parser features give signal for GT V2 block anatomy.",
        "It uses the supervised scaffold and does not alter solver behaviour.",
        "",
        "## Coverage",
        "",
        f"- Records analysed: `{len(records)}`",
        f"- Records with grammar features: `{records_with_grammar}`",
        f"- SOURCE tokens: `{source_tokens}`",
        f"- RESIDUE tokens: `{residue_tokens}`",
        f"- SOURCE tokens whose parser head is another SOURCE token in the same source span: `{source_span_internal_heads}/{source_span_head_total}` (`{internal_rate:.0%}`)",
        "",
        "## POS By Supervised Label",
        "",
    ]

    for label in ["SOURCE", "RESIDUE", "DEF"]:
        lines.append(f"- `{label}`: {compact_counts(label_pos[label], 12)}")

    lines.extend(["", "## Dependency Labels By Supervised Label", ""])
    for label in ["SOURCE", "RESIDUE", "DEF"]:
        lines.append(f"- `{label}`: {compact_counts(label_dep[label], 12)}")

    lines.extend(["", "## Parser Head Crossings", ""])
    lines.append(compact_counts(head_label_edges, 20))

    lines.extend(["", "## SOURCE Head Pattern", ""])
    lines.append(compact_counts(source_head_edges, 20))

    lines.extend(["", "## RESIDUE To Head Label", ""])
    lines.append(compact_counts(residue_head_edges, 30))

    lines.extend(["", "## Preposition Attachment", ""])
    for item, count in prep_head_edges.most_common(30):
        lines.append(f"- `{item}` = {count}")

    lines.extend(["", "## Glue Word Attachment", ""])
    for item, count in glue_head_edges.most_common(30):
        lines.append(f"- `{item}` = {count}")

    lines.extend(["", "## Glue Word Dependency Context", ""])
    for item, count in glue_dep_edges.most_common(30):
        lines.append(f"- `{item}` = {count}")

    lines.extend(["", "## Operationish Word Context", ""])
    for item, count in operationish_edges.most_common(30):
        lines.append(f"- `{item}` = {count}")

    lines.extend(["", "## Preposition Attachment By Operation", ""])
    for op in sorted(operation_prep_edges):
        lines.append(f"- `{op}`: {compact_counts(operation_prep_edges[op], 10)}")

    lines.extend(["", "## Boundary Crossing Examples", ""])
    for record, note in boundary_cross_examples:
        lines.extend(example_lines(record, note))

    lines.extend(["", "## Residue Glue Attached To SOURCE Examples", ""])
    for record, note in glue_examples:
        lines.extend(example_lines(record, note))

    lines.extend(["", "## Direction Word Examples", ""])
    for record, note in direction_examples:
        lines.extend(example_lines(record, note))

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "The parser signal should be treated as weak supervision, not authority.",
            "Useful signal appears where residue glue attaches syntactically to SOURCE material, because that marks the exact attachment question the residue baseline cannot answer.",
            "The next useful test is a small classifier for token label or boundary transitions using residue features plus POS/dependency context, measured against the residue-only baseline buckets.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    records = load_records(args.input)
    write_report(records, args.report_out)
    print(f"records={len(records)}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
