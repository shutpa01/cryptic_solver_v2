"""Mine the residue-baseline error space for GT V2 grammar triage targets.

This does not attempt to solve clues. It takes the residue-only baseline and
pulls out the cases where grammar should plausibly add information: glue words,
scope/direction collisions, source contamination, and low-support indicators.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import evaluate_residue_baseline as baseline


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "clean_boundary_training_slice_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "grammar_triage_targets_2026-05-17.md"

GLUE = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "having",
    "in",
    "is",
    "of",
    "on",
    "over",
    "the",
    "to",
    "with",
}

OPERATIONISH = {
    "about",
    "back",
    "broadcast",
    "broken",
    "confused",
    "dressed",
    "east",
    "excited",
    "heard",
    "losing",
    "novel",
    "oddly",
    "partly",
    "picked",
    "radio",
    "reconstructed",
    "reported",
    "reportedly",
    "reverse",
    "rising",
    "shredded",
    "some",
    "up",
}


def token_texts(record: dict[str, Any]) -> list[str]:
    return [t["text"].lower() for t in record.get("residue", [])]


def category(record: dict[str, Any], pred: baseline.Prediction) -> str:
    gold = baseline.assembly_op(record)
    predicted = pred.op or "UNKNOWN"
    tokens = set(token_texts(record))

    if gold == predicted:
        return "correct"
    if not tokens:
        return "empty-residue"
    if tokens <= GLUE:
        return "glue-only"
    if gold in {"hidden", "hidden_reversed", "reversal"} and predicted in {
        "hidden",
        "hidden_reversed",
        "reversal",
    }:
        return "direction-or-scope"
    if tokens & GLUE and tokens & OPERATIONISH:
        return "indicator-plus-glue"
    if predicted == "UNKNOWN" and tokens & OPERATIONISH:
        return "low-support-indicator"
    if tokens & GLUE:
        return "glue-dominates"
    return "other"


def example_lines(record: dict[str, Any], pred: baseline.Prediction) -> list[str]:
    feature = "none" if pred.feature is None else f"{pred.feature[0]}={pred.feature[1]}"
    pieces = []
    for piece in record.get("pieces", []):
        clue_word = piece.get("clue_word") or ""
        letters = piece.get("letters") or ""
        mechanism = piece.get("mechanism") or ""
        if clue_word or letters or mechanism:
            pieces.append(f"{clue_word}->{letters} ({mechanism})")
    return [
        f"- `{record['clue']}` ({record['answer']})",
        f"  Gold/predicted: `{baseline.assembly_op(record)}` / `{pred.op or 'UNKNOWN'}` from `{feature}`",
        f"  Residue: `{baseline.residue_text(record) or '<empty>'}`",
        f"  Labels: `{baseline.label_signature(record)}`",
        f"  Pieces: {'; '.join(pieces[:6])}",
    ]


def write_report(records: list[dict[str, Any]], path: Path, test_percent: int, min_support: int) -> None:
    train = [r for r in records if not baseline.is_test_record(r, test_percent)]
    test = [r for r in records if baseline.is_test_record(r, test_percent)]
    model = baseline.build_model(train)

    buckets: dict[str, list[tuple[dict[str, Any], baseline.Prediction]]] = defaultdict(list)
    predicted_by_category: dict[str, Counter[str]] = defaultdict(Counter)
    gold_by_category: dict[str, Counter[str]] = defaultdict(Counter)

    for record in test:
        pred = baseline.predict(record, model, min_support)
        cat = category(record, pred)
        buckets[cat].append((record, pred))
        predicted_by_category[cat][pred.op or "UNKNOWN"] += 1
        gold_by_category[cat][baseline.assembly_op(record)] += 1

    ordered_categories = [
        "glue-only",
        "glue-dominates",
        "indicator-plus-glue",
        "direction-or-scope",
        "low-support-indicator",
        "empty-residue",
        "other",
    ]

    lines = [
        "# Grammar Triage Targets",
        "",
        "Date: 2026-05-17",
        "",
        "This report turns the residue-only baseline errors into research targets.",
        "The purpose is to identify where grammar might add information beyond a residue lexicon.",
        "",
        "## Buckets",
        "",
    ]

    for cat in ordered_categories:
        examples = buckets.get(cat, [])
        if not examples:
            continue
        gold = ", ".join(f"`{op}`={count}" for op, count in gold_by_category[cat].most_common())
        pred = ", ".join(f"`{op}`={count}" for op, count in predicted_by_category[cat].most_common())
        lines.append(f"- `{cat}`: `{len(examples)}` cases. Gold: {gold}. Predicted: {pred}.")

    lines.extend(["", "## Reading The Buckets", ""])
    lines.append("- `glue-only` and `glue-dominates` are the strongest grammar candidates: the residue lexicon is mostly seeing prepositions and conjunctions, so syntax and attachment should matter.")
    lines.append("- `indicator-plus-glue` cases test whether we can keep an operation word chained to the right source block without letting nearby glue words steal the decision.")
    lines.append("- `direction-or-scope` cases are where reversal, hidden, and reversed-hidden share similar residue and need answer-aware scope/direction checks.")
    lines.append("- `low-support-indicator` is mostly a data problem: the operation signal exists, but the clean slice has not seen enough examples yet.")

    for cat in ordered_categories:
        examples = buckets.get(cat, [])
        if not examples:
            continue
        lines.extend(["", f"## {cat}", ""])
        for record, pred in examples[:5]:
            lines.extend(example_lines(record, pred))

    lines.extend(
        [
            "",
            "## Research Implication",
            "",
            "The next grammar experiment should not start by predicting clue type directly.",
            "It should start by asking whether grammar can protect block boundaries from glue words and attach operation residues to the correct source span.",
            "That keeps the experiment close to the user's core claim: grammar signature and wordplay signature emanate from the same words.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--test-percent", type=int, default=25)
    parser.add_argument("--min-support", type=int, default=3)
    args = parser.parse_args()

    records = baseline.load_records(args.input)
    write_report(records, args.report_out, args.test_percent, args.min_support)
    print(f"records={len(records)}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
