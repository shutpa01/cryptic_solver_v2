"""Mine residue phrases against structured assembly operations.

This is part of the GT V2 science work. It asks a narrow question:

Once structured source pieces and definitions have been removed, what residue
phrases remain, and how strongly do they correlate with assembly operations?
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "clean_boundary_training_slice_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "residue_operation_signals_2026-05-17.md"


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def primary_type(record: dict[str, Any]) -> str:
    types = record.get("wordplay_types") or ["blank"]
    return types[0] if types else "blank"


def assembly_op(record: dict[str, Any]) -> str:
    assembly = record.get("assembly") or {}
    if isinstance(assembly, dict):
        return assembly.get("op") or primary_type(record)
    return primary_type(record)


def residue_text(record: dict[str, Any]) -> str:
    return " ".join(t["text"].lower() for t in record.get("residue", [])).strip()


def label_signature(record: dict[str, Any]) -> str:
    return " ".join((t.get("label") or "?")[0] for t in record.get("token_labels", []))


def source_shape(record: dict[str, Any]) -> str:
    spans = []
    for piece in record.get("pieces", []):
        mapping = piece.get("mapping") or {}
        if mapping.get("status") != "mapped":
            continue
        spans.append(str(mapping.get("span", 0)))
    return "+".join(spans) if spans else "none"


def contiguous_residue_runs(record: dict[str, Any]) -> list[str]:
    runs = []
    current = []
    last_idx = None
    for item in record.get("residue", []):
        idx = item["index"]
        if last_idx is None or idx == last_idx + 1:
            current.append(item["text"].lower())
        else:
            if current:
                runs.append(" ".join(current))
            current = [item["text"].lower()]
        last_idx = idx
    if current:
        runs.append(" ".join(current))
    return runs


def examples_for(records: list[dict[str, Any]], predicate, limit: int = 4) -> list[dict[str, Any]]:
    out = []
    for record in records:
        if predicate(record):
            out.append(record)
            if len(out) >= limit:
                break
    return out


def write_report(records: list[dict[str, Any]], path: Path) -> None:
    residue_by_op: dict[str, Counter[str]] = defaultdict(Counter)
    run_by_op: dict[str, Counter[str]] = defaultdict(Counter)
    op_by_residue: dict[str, Counter[str]] = defaultdict(Counter)
    signatures_by_op: dict[str, Counter[str]] = defaultdict(Counter)
    source_shapes_by_op: dict[str, Counter[str]] = defaultdict(Counter)
    residue_len_by_op: dict[str, Counter[int]] = defaultdict(Counter)

    for record in records:
        op = assembly_op(record)
        res = residue_text(record)
        if res:
            residue_by_op[op][res] += 1
            op_by_residue[res][op] += 1
        for run in contiguous_residue_runs(record):
            run_by_op[op][run] += 1
        signatures_by_op[op][label_signature(record)] += 1
        source_shapes_by_op[op][source_shape(record)] += 1
        residue_len_by_op[op][len(record.get("residue", []))] += 1

    distinctive = []
    for residue, ops in op_by_residue.items():
        total = sum(ops.values())
        if total < 3:
            continue
        op, count = ops.most_common(1)[0]
        dominance = count / total
        if dominance >= 0.65:
            distinctive.append((dominance, total, residue, op, count))

    lines = [
        "# Residue Operation Signals",
        "",
        "Date: 2026-05-17",
        "",
        "This report mines the clean boundary slice after SOURCE and DEF tokens have been labelled.",
        "It asks which RESIDUE phrases correlate with structured assembly operations.",
        "",
        f"Records analysed: {len(records)}",
        "",
        "## Residue Length By Operation",
        "",
    ]

    for op in sorted(residue_len_by_op):
        top = ", ".join(f"{length} tokens={count}" for length, count in residue_len_by_op[op].most_common(8))
        lines.append(f"- `{op}`: {top}")

    lines.extend(["", "## Common Residue Runs By Operation", ""])
    for op in sorted(run_by_op):
        top = ", ".join(f"`{run}`={count}" for run, count in run_by_op[op].most_common(12))
        lines.append(f"- `{op}`: {top}")

    lines.extend(["", "## Distinctive Residue Signals", ""])
    for dominance, total, residue, op, count in sorted(distinctive, reverse=True)[:60]:
        lines.append(f"- `{residue}` -> `{op}` ({count}/{total}, {dominance:.0%})")

    lines.extend(["", "## Boundary Signatures By Operation", ""])
    for op in sorted(signatures_by_op):
        top = ", ".join(f"`{sig}`={count}" for sig, count in signatures_by_op[op].most_common(10))
        lines.append(f"- `{op}`: {top}")

    lines.extend(["", "## Source Span Shapes By Operation", ""])
    for op in sorted(source_shapes_by_op):
        top = ", ".join(f"`{shape}`={count}" for shape, count in source_shapes_by_op[op].most_common(10))
        lines.append(f"- `{op}`: {top}")

    lines.extend(["", "## Worked Signal Examples", ""])
    signal_terms = ["some", "reportedly", "we hear", "reconstructed", "kept by", "at front of", "picked up"]
    for term in signal_terms:
        matches = examples_for(records, lambda r, t=term: t in residue_text(r), limit=3)
        if not matches:
            continue
        lines.append(f"`{term}`")
        for record in matches:
            lines.append(f"- `{record['clue']}` ({record['answer']})")
            lines.append(f"  Operation: `{assembly_op(record)}`")
            lines.append(f"  Residue: `{residue_text(record)}`")
            lines.append(f"  Labels: `{label_signature(record)}`")
            lines.append(f"  Assembly: `{record.get('assembly')}`")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This gives us the first non-grammar baseline: residue alone often carries strong operation evidence.",
            "The grammar experiment should therefore not ask whether grammar solves everything by itself.",
            "It should ask whether grammar improves block boundary discovery and operation attachment beyond this residue lexicon baseline.",
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
