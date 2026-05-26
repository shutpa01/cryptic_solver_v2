"""Detect definition-by-example marker leakage into wordplay residue.

Structured definition spans are useful but not always complete. This report
finds DBE-style words such as "say", "perhaps", and "maybe" that are labelled
as residue, especially near definition boundaries.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "grammar_feature_scaffold_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "definition_marker_leakage_2026-05-17.md"

DBE_MARKERS = {
    "maybe",
    "perhaps",
    "say",
    "example",
    "eg",
    "e.g",
    "like",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def find_leaks(record: dict[str, Any]) -> list[dict[str, Any]]:
    tokens = record.get("tokens", [])
    leaks = []
    for i, token in enumerate(tokens):
        clean = token.get("clean")
        if clean not in DBE_MARKERS:
            continue
        if token.get("label") != "RESIDUE":
            continue
        prev_label = tokens[i - 1].get("label") if i > 0 else "<START>"
        next_label = tokens[i + 1].get("label") if i < len(tokens) - 1 else "<END>"
        near_def = prev_label == "DEF" or next_label == "DEF"
        leaks.append(
            {
                "index": i,
                "text": token.get("text"),
                "prev_label": prev_label,
                "next_label": next_label,
                "near_def": near_def,
            }
        )
    return leaks


def label_signature(record: dict[str, Any]) -> str:
    return " ".join((t.get("label") or "?")[0] for t in record.get("tokens", []))


def write_report(records: list[dict[str, Any]], path: Path) -> None:
    leak_records = []
    marker_counts = Counter()
    near_def_count = 0
    by_operation = Counter()

    for record in records:
        leaks = find_leaks(record)
        if not leaks:
            continue
        leak_records.append((record, leaks))
        by_operation[record.get("operation")] += 1
        for leak in leaks:
            marker_counts[leak["text"].lower()] += 1
            if leak["near_def"]:
                near_def_count += 1

    lines = [
        "# Definition Marker Leakage",
        "",
        "Date: 2026-05-17",
        "",
        "This report finds definition-by-example markers that appear as `RESIDUE` rather than travelling with the definition.",
        "These are important because they can poison residue attachment labels.",
        "",
        f"Records with possible leakage: `{len(leak_records)}`",
        f"Leaked markers near a definition boundary: `{near_def_count}`",
        "",
        "## Marker Counts",
        "",
    ]
    lines.append(", ".join(f"`{k}`={v}" for k, v in marker_counts.most_common()) or "`none`")

    lines.extend(["", "## By Operation", ""])
    lines.append(", ".join(f"`{k}`={v}" for k, v in by_operation.most_common()) or "`none`")

    lines.extend(["", "## Examples", ""])
    for record, leaks in leak_records[:40]:
        lines.append(f"- `{record['clue']}` ({record['answer']})")
        lines.append(f"  Operation: `{record['operation']}`")
        lines.append(f"  Labels: `{label_signature(record)}`")
        for leak in leaks:
            lines.append(
                f"  Marker: `{leak['text']}` with context `{leak['prev_label']}<RESIDUE>{leak['next_label']}`"
            )

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "DBE marker leakage should be treated as a definition-span quality issue before it is treated as wordplay residue.",
            "A future block graph should allow a `DBE_MARKER` or `DEF_MODIFIER` node attached to `DEF_BLOCK`.",
            "This is another example of why block anatomy must be preserved rather than flattened into source/residue labels.",
            "",
        ]
    )

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
