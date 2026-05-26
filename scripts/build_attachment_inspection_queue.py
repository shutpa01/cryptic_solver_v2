"""Create a compact manual inspection queue for operation attachment labels.

The operation attachment slice is intentionally broad and noisy. This report
selects a small number of readable examples for each weak label so the design
discussion can proceed one case at a time rather than as a table dump.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "operation_attachment_slice_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "operation_attachment_inspection_queue_2026-05-17.md"

LABEL_ORDER = [
    "OPERATOR_SCOPE",
    "LOCATOR_SCOPE",
    "DIRECTION_OR_ORIENTATION",
    "ORDER_OR_POSITION",
    "CONTAINER_RELATION",
    "CONNECTOR_OR_SURFACE",
    "UNCLASSIFIED_RESIDUE",
]


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def source_text(row: dict[str, Any]) -> str:
    return "; ".join(f"`{s['text']}`" for s in row.get("source_spans", [])) or "`none`"


def parser_heads(row: dict[str, Any]) -> str:
    heads = row.get("parser_source_heads") or []
    if not heads:
        return "`none`"
    return ", ".join(f"`{h['token']}->{h['head']}:{h['dep']}`" for h in heads)


def score_row(row: dict[str, Any]) -> tuple[int, int, str]:
    heads = len(row.get("parser_source_heads") or [])
    residue_len = len(row.get("residue_run", {}).get("clean_words", []))
    return (1 if heads else 0, min(residue_len, 4), row.get("clue", ""))


def choose_examples(rows: list[dict[str, Any]], limit: int) -> list[dict[str, Any]]:
    chosen = []
    seen_ops = set()
    for row in sorted(rows, key=score_row, reverse=True):
        op = row.get("operation")
        if op in seen_ops and len(seen_ops) < 4:
            continue
        chosen.append(row)
        seen_ops.add(op)
        if len(chosen) >= limit:
            break
    return chosen


def write_report(rows: list[dict[str, Any]], path: Path, limit: int) -> None:
    by_label: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_label[row.get("weak_attachment_label", "UNKNOWN")].append(row)

    lines = [
        "# Operation Attachment Inspection Queue",
        "",
        "Date: 2026-05-17",
        "",
        "This is a compact queue for manual design inspection.",
        "The labels are weak and provisional. The task is to decide whether each residue block has the right anatomical relationship to the source material.",
        "",
    ]

    for label in LABEL_ORDER:
        label_rows = by_label.get(label, [])
        if not label_rows:
            continue
        lines.extend(["", f"## {label}", ""])
        lines.append(f"Available examples: `{len(label_rows)}`")
        lines.append("")
        for row in choose_examples(label_rows, limit):
            lines.append(f"- `{row['clue']}` ({row['answer']})")
            lines.append(f"  Operation: `{row['operation']}`")
            lines.append(f"  Residue: `{row['residue_run']['text']}`")
            lines.append(f"  Sources: {source_text(row)}")
            lines.append(f"  Relationship: `{row['block_relationship']}`")
            lines.append(f"  Scope status: `{row['scope_status']}`")
            if row.get("needs_split"):
                split = ", ".join(
                    f"{item['text']}->{item['weak_attachment_label']}"
                    for item in row.get("token_annotations", [])
                )
                lines.append(f"  Needs split: `{split}`")
            lines.append(f"  Parser source heads: {parser_heads(row)}")

    lines.extend(
        [
            "",
            "## Inspection Questions",
            "",
            "- Is the residue label right, or is it actually source-internal/surface material?",
            "- If it is an operation, what exact source span does it operate on?",
            "- If it is a locator, what operation is it chained to?",
            "- If it is a connector, does the surface grammar permit that reading?",
            "- If parser evidence disagrees with cryptic evidence, which should be preserved as primary?",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=4)
    args = parser.parse_args()

    rows = load_rows(args.input)
    write_report(rows, args.report_out, args.limit)
    print(f"rows={len(rows)}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
