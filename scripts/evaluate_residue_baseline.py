"""Evaluate residue-only operation prediction for the GT V2 science work.

The clean boundary slice already labels definition, source, and residue tokens
from structured human explanations. This script deliberately ignores grammar
and answer mechanics, then asks how far residue phrases alone can predict the
structured assembly operation.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "clean_boundary_training_slice_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "residue_baseline_evaluation_2026-05-17.md"


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


def residue_runs(record: dict[str, Any]) -> list[str]:
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


def residue_tokens(record: dict[str, Any]) -> list[str]:
    return [t["text"].lower() for t in record.get("residue", [])]


def source_count(record: dict[str, Any]) -> int:
    return sum(1 for t in record.get("token_labels", []) if t.get("label") == "SOURCE")


def label_signature(record: dict[str, Any]) -> str:
    return " ".join((t.get("label") or "?")[0] for t in record.get("token_labels", []))


def split_key(record: dict[str, Any]) -> str:
    return f"{record.get('source')}:{record.get('id')}:{record.get('clue')}:{record.get('answer')}"


def is_test_record(record: dict[str, Any], test_percent: int) -> bool:
    digest = hashlib.sha1(split_key(record).encode("utf-8")).hexdigest()
    bucket = int(digest[:8], 16) % 100
    return bucket < test_percent


def feature_values(record: dict[str, Any]) -> list[tuple[str, str]]:
    features: list[tuple[str, str]] = []
    text = residue_text(record)
    if text:
        features.append(("full_residue", text))
    for run in residue_runs(record):
        features.append(("run", run))
    for token in residue_tokens(record):
        features.append(("token", token))
    if not features:
        features.append(("empty_residue", "<empty>"))
    return features


@dataclass
class Prediction:
    op: str | None
    confidence: float
    feature: tuple[str, str] | None
    seen: int


def build_model(records: list[dict[str, Any]]) -> dict[tuple[str, str], Counter[str]]:
    model: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in records:
        op = assembly_op(record)
        for feature in feature_values(record):
            model[feature][op] += 1
    return model


def predict(
    record: dict[str, Any],
    model: dict[tuple[str, str], Counter[str]],
    min_support: int,
) -> Prediction:
    candidates = []
    feature_rank = {"full_residue": 3, "run": 2, "token": 1, "empty_residue": 0}
    for feature in feature_values(record):
        counts = model.get(feature)
        if not counts:
            continue
        total = sum(counts.values())
        if total < min_support:
            continue
        op, count = counts.most_common(1)[0]
        confidence = count / total
        candidates.append((confidence, feature_rank.get(feature[0], 0), total, op, feature))
    if not candidates:
        return Prediction(None, 0.0, None, 0)
    confidence, _rank, total, op, feature = sorted(candidates, reverse=True)[0]
    return Prediction(op, confidence, feature, total)


def short_example(record: dict[str, Any], prediction: Prediction) -> list[str]:
    feature = "none" if prediction.feature is None else f"{prediction.feature[0]}={prediction.feature[1]}"
    return [
        f"- `{record['clue']}` ({record['answer']})",
        f"  Gold: `{assembly_op(record)}`; predicted: `{prediction.op or 'UNKNOWN'}`; confidence: `{prediction.confidence:.0%}` from `{feature}`",
        f"  Residue: `{residue_text(record) or '<empty>'}`",
        f"  Labels: `{label_signature(record)}`",
    ]


def write_report(records: list[dict[str, Any]], path: Path, test_percent: int, min_support: int) -> None:
    train = [r for r in records if not is_test_record(r, test_percent)]
    test = [r for r in records if is_test_record(r, test_percent)]
    model = build_model(train)

    correct = 0
    covered = 0
    high_conf_correct = 0
    high_conf_total = 0
    by_gold: dict[str, Counter[str]] = defaultdict(Counter)
    by_feature: dict[str, Counter[str]] = defaultdict(Counter)
    examples_correct = []
    examples_wrong = []
    examples_unknown = []
    ambiguous = []

    for record in test:
        gold = assembly_op(record)
        pred = predict(record, model, min_support)
        pred_op = pred.op or "UNKNOWN"
        by_gold[gold][pred_op] += 1
        if pred.feature:
            by_feature[pred.feature[0]][pred_op] += 1
        if pred.op is not None:
            covered += 1
        if pred.op == gold:
            correct += 1
            if len(examples_correct) < 6:
                examples_correct.append((record, pred))
        elif pred.op is None:
            if len(examples_unknown) < 6:
                examples_unknown.append((record, pred))
        else:
            if len(examples_wrong) < 8:
                examples_wrong.append((record, pred))
        if pred.confidence >= 0.8:
            high_conf_total += 1
            if pred.op == gold:
                high_conf_correct += 1
        if pred.feature and 0.35 <= pred.confidence <= 0.75 and len(ambiguous) < 8:
            ambiguous.append((record, pred))

    total = len(test)
    accuracy = correct / total if total else 0.0
    coverage = covered / total if total else 0.0
    covered_accuracy = correct / covered if covered else 0.0
    high_conf_accuracy = high_conf_correct / high_conf_total if high_conf_total else 0.0

    train_ops = Counter(assembly_op(r) for r in train)
    test_ops = Counter(assembly_op(r) for r in test)

    lines = [
        "# Residue Baseline Evaluation",
        "",
        "Date: 2026-05-17",
        "",
        "This is a control experiment for GT V2. It ignores grammar, syntax, answer letters, and block mechanics.",
        "It predicts the structured assembly operation using only residue text left after SOURCE and DEF tokens are removed.",
        "",
        "## Result",
        "",
        f"- Records: `{len(records)}`",
        f"- Train/test split: `{len(train)}` / `{len(test)}` using deterministic clue hash",
        f"- Minimum residue feature support: `{min_support}` training examples",
        f"- Test coverage: `{covered}/{total}` (`{coverage:.0%}`)",
        f"- Overall accuracy: `{correct}/{total}` (`{accuracy:.0%}`)",
        f"- Accuracy when covered: `{correct}/{covered}` (`{covered_accuracy:.0%}`)",
        f"- High-confidence accuracy: `{high_conf_correct}/{high_conf_total}` (`{high_conf_accuracy:.0%}`)",
        "",
        "## Operation Mix",
        "",
    ]

    lines.append("Train: " + ", ".join(f"`{op}`={count}" for op, count in train_ops.most_common()))
    lines.append("Test: " + ", ".join(f"`{op}`={count}" for op, count in test_ops.most_common()))

    lines.extend(["", "## Confusion By Gold Operation", ""])
    for gold in sorted(by_gold):
        total_gold = sum(by_gold[gold].values())
        preds = ", ".join(f"`{op}`={count}" for op, count in by_gold[gold].most_common())
        lines.append(f"- `{gold}` ({total_gold}): {preds}")

    lines.extend(["", "## Feature Families Used", ""])
    for family in sorted(by_feature):
        total_family = sum(by_feature[family].values())
        preds = ", ".join(f"`{op}`={count}" for op, count in by_feature[family].most_common())
        lines.append(f"- `{family}` ({total_family}): {preds}")

    lines.extend(["", "## Correct Examples", ""])
    for record, pred in examples_correct:
        lines.extend(short_example(record, pred))

    lines.extend(["", "## Wrong Examples", ""])
    for record, pred in examples_wrong:
        lines.extend(short_example(record, pred))

    lines.extend(["", "## Unknown Examples", ""])
    for record, pred in examples_unknown:
        lines.extend(short_example(record, pred))

    lines.extend(["", "## Ambiguous Residue Examples", ""])
    for record, pred in ambiguous:
        lines.extend(short_example(record, pred))

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "Residue-only prediction is useful but not sufficient. It is strongest when the residue contains a conventional indicator phrase.",
            "It struggles where the same surface phrase can perform several jobs, or where residue is mainly grammatical glue.",
            "That is exactly the space where grammar triage should earn its keep: not by replacing the residue lexicon, but by deciding attachment, scope, and whether a residue phrase is an operation, a connector, or part of a source phrase.",
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

    records = load_records(args.input)
    write_report(records, args.report_out, args.test_percent, args.min_support)
    print(f"records={len(records)}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
