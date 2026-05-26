"""Analyse clean SOURCE/DEF/RESIDUE boundary buckets for GT V2 R&D.

This consumes the structured research cohort sample produced by
build_structured_research_cohort.py and extracts the cleanest source/type
buckets as the first boundary-learning slice.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "structured_research_cohort_sample_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "clean_boundary_bucket_analysis_2026-05-17.md"
DEFAULT_JSONL = PROJECT_ROOT / "documents" / "clean_boundary_training_slice_2026-05-17.jsonl"


CLEAN_BUCKETS = {
    ("telegraph", "charade"),
    ("telegraph", "anagram"),
    ("telegraph", "hidden"),
    ("telegraph", "hidden_reversed"),
    ("telegraph", "reversal"),
    ("telegraph", "homophone"),
    ("dailymail", "hidden"),
    ("dailymail", "anagram"),
    ("dailymail", "charade"),
    ("telegraph-toughie", "charade"),
    ("telegraph-toughie", "hidden"),
    ("telegraph-toughie", "anagram"),
}


def load_records(path: Path) -> list[dict[str, Any]]:
    records = []
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                records.append(json.loads(line))
    return records


def primary_type(record: dict[str, Any]) -> str:
    types = record.get("wordplay_types") or ["blank"]
    return types[0] if types else "blank"


def label_signature(record: dict[str, Any]) -> str:
    return " ".join((t.get("label") or "?")[0] for t in record.get("token_labels", []))


def residue_text(record: dict[str, Any]) -> str:
    return " ".join(t["text"] for t in record.get("residue", []))


def source_spans(record: dict[str, Any]) -> list[str]:
    out = []
    for piece in record.get("pieces", []):
        mapping = piece.get("mapping") or {}
        if mapping.get("status") == "mapped":
            out.append(f"{mapping.get('text')} -> {piece.get('letters')}")
    return out


def write_jsonl(records: list[dict[str, Any]], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n")


def write_report(records: list[dict[str, Any]], path: Path) -> None:
    bucket_counts = Counter()
    signature_counts = Counter()
    residue_counts = Counter()
    type_signature_counts = defaultdict(Counter)
    examples = defaultdict(list)

    for record in records:
        bucket = (record.get("source"), primary_type(record))
        bucket_counts[bucket] += 1
        sig = label_signature(record)
        signature_counts[sig] += 1
        type_signature_counts[primary_type(record)][sig] += 1
        res = residue_text(record).lower()
        if res:
            residue_counts[res] += 1
        if len(examples[bucket]) < 3:
            examples[bucket].append(record)

    lines = [
        "# Clean Boundary Bucket Analysis",
        "",
        "Date: 2026-05-17",
        "",
        "This is the first GT V2 boundary-learning slice.",
        "It uses only clean source/type buckets and only records where every structured piece maps back to clue text.",
        "",
        f"Training slice records: {len(records)}",
        "",
        "## Bucket Counts",
        "",
    ]

    for (source, typ), count in bucket_counts.most_common():
        lines.append(f"- `{source}` / `{typ}`: {count}")

    lines.extend(["", "## Common Boundary Signatures", ""])
    for sig, count in signature_counts.most_common(25):
        lines.append(f"- `{sig}`: {count}")

    lines.extend(["", "## Boundary Signatures By Type", ""])
    for typ in sorted(type_signature_counts):
        top = ", ".join(f"`{sig}`={count}" for sig, count in type_signature_counts[typ].most_common(8))
        lines.append(f"- `{typ}`: {top}")

    lines.extend(["", "## Common Residues", ""])
    for res, count in residue_counts.most_common(30):
        lines.append(f"- `{res}`: {count}")

    lines.extend(["", "## Example Records", ""])
    for bucket in sorted(examples):
        source, typ = bucket
        lines.append(f"`{source}` / `{typ}`")
        for record in examples[bucket]:
            lines.append(f"- `{record['clue']}` ({record['answer']})")
            lines.append(f"  Definition: `{record.get('definition')}`")
            lines.append(f"  Sources: {', '.join(source_spans(record))}")
            lines.append(f"  Residue: `{residue_text(record)}`")
            lines.append(f"  Labels: `{label_signature(record)}`")
            lines.append(f"  Assembly: `{record.get('assembly')}`")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This slice is not intended to solve hard clues. It gives the science project a clean baseline:",
            "can a grammar model recover obvious source spans, definition spans, and simple residue before we ask it to handle nested containers, deletions, and reversals?",
            "",
            "The next step is to add grammar tags/dependencies to this slice and test whether the grammar features predict these boundary signatures.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--jsonl-out", type=Path, default=DEFAULT_JSONL)
    args = parser.parse_args()

    records = load_records(args.input)
    clean = [
        r for r in records
        if (r.get("source"), primary_type(r)) in CLEAN_BUCKETS
        and (r.get("quality") or {}).get("complete_piece_mapping")
    ]
    write_jsonl(clean, args.jsonl_out)
    write_report(clean, args.report_out)
    print(f"input_records={len(records)}")
    print(f"clean_records={len(clean)}")
    print(f"jsonl={args.jsonl_out}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
