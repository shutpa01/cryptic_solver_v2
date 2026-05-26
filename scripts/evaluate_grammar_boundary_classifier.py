"""Compare lexical-only and grammar-aware token boundary classifiers.

This is a deliberately small GT V2 science test. Given the supervised scaffold,
and excluding definition tokens, can we classify each wordplay token as SOURCE
or RESIDUE? The point is not to build the final model; it is to test whether
grammar features add measurable signal beyond the word itself.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "grammar_feature_scaffold_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "grammar_boundary_classifier_2026-05-17.md"
DEFAULT_SPAN_OUT = PROJECT_ROOT / "documents" / "grammar_span_boundary_experiment_2026-05-17.md"


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def split_key(record: dict[str, Any]) -> str:
    return f"{record.get('source')}:{record.get('id')}:{record.get('clue')}:{record.get('answer')}"


def is_test_record(record: dict[str, Any], test_percent: int) -> bool:
    digest = hashlib.sha1(split_key(record).encode("utf-8")).hexdigest()
    return int(digest[:8], 16) % 100 < test_percent


def grammar(token: dict[str, Any]) -> dict[str, Any]:
    return token.get("grammar") or {}


def wordplay_tokens(record: dict[str, Any]) -> list[dict[str, Any]]:
    return [t for t in record.get("tokens", []) if t.get("label") in {"SOURCE", "RESIDUE"}]


def lexical_features(tokens: list[dict[str, Any]], i: int) -> list[tuple[str, str]]:
    token = tokens[i]
    clean = token.get("clean") or ""
    prev_clean = tokens[i - 1].get("clean") if i > 0 else "<START>"
    next_clean = tokens[i + 1].get("clean") if i < len(tokens) - 1 else "<END>"
    return [
        ("clean", clean),
        ("prev_clean", prev_clean),
        ("next_clean", next_clean),
        ("position", "first" if i == 0 else "last" if i == len(tokens) - 1 else "middle"),
        ("is_glue", str(bool(token.get("is_glue_word")))),
        ("is_operationish", str(bool(token.get("is_operationish_word")))),
    ]


def grammar_features(tokens: list[dict[str, Any]], i: int) -> list[tuple[str, str]]:
    token = tokens[i]
    g = grammar(token)
    prev_g = grammar(tokens[i - 1]) if i > 0 else {}
    next_g = grammar(tokens[i + 1]) if i < len(tokens) - 1 else {}
    features = lexical_features(tokens, i)
    features.extend(
        [
            ("mid_pos", g.get("mid_pos", "?")),
            ("dep", g.get("dep", "?")),
            ("pos_dep", f"{g.get('mid_pos', '?')}:{g.get('dep', '?')}"),
            ("prev_mid", prev_g.get("mid_pos", "<START>")),
            ("next_mid", next_g.get("mid_pos", "<END>")),
            ("prev_dep", prev_g.get("dep", "<START>")),
            ("next_dep", next_g.get("dep", "<END>")),
            ("mid_context", f"{prev_g.get('mid_pos', '<START>')}<{g.get('mid_pos', '?')}>{next_g.get('mid_pos', '<END>')}"),
        ]
    )
    return features


def build_model(
    records: list[dict[str, Any]],
    feature_fn,
    min_support: int,
) -> dict[tuple[str, str], Counter[str]]:
    model: dict[tuple[str, str], Counter[str]] = defaultdict(Counter)
    for record in records:
        tokens = wordplay_tokens(record)
        for i, token in enumerate(tokens):
            gold = token["label"]
            for feature in feature_fn(tokens, i):
                model[feature][gold] += 1
    return {
        feature: counts
        for feature, counts in model.items()
        if sum(counts.values()) >= min_support
    }


def predict(tokens: list[dict[str, Any]], i: int, model, feature_fn) -> tuple[str, float, tuple[str, str] | None]:
    votes = Counter()
    best_feature = None
    best_conf = 0.0
    for feature in feature_fn(tokens, i):
        counts = model.get(feature)
        if not counts:
            continue
        total = sum(counts.values())
        label, count = counts.most_common(1)[0]
        conf = count / total
        votes[label] += conf * min(total, 20)
        if conf > best_conf:
            best_conf = conf
            best_feature = feature
    if not votes:
        return "UNKNOWN", 0.0, None
    label, score = votes.most_common(1)[0]
    total_votes = sum(votes.values())
    return label, score / total_votes if total_votes else 0.0, best_feature


def evaluate(records: list[dict[str, Any]], feature_fn, min_support: int, test_percent: int):
    train = [r for r in records if not is_test_record(r, test_percent)]
    test = [r for r in records if is_test_record(r, test_percent)]
    model = build_model(train, feature_fn, min_support)
    total = correct = covered = 0
    confusion: dict[str, Counter[str]] = defaultdict(Counter)
    by_operation: dict[str, Counter[str]] = defaultdict(Counter)
    examples_wrong = []
    examples_unknown = []
    examples_correct = []

    for record in test:
        tokens = wordplay_tokens(record)
        for i, token in enumerate(tokens):
            gold = token["label"]
            pred, conf, feature = predict(tokens, i, model, feature_fn)
            total += 1
            if pred != "UNKNOWN":
                covered += 1
            if pred == gold:
                correct += 1
                by_operation[record["operation"]]["correct"] += 1
                if len(examples_correct) < 5:
                    examples_correct.append((record, token, pred, conf, feature))
            else:
                by_operation[record["operation"]]["wrong"] += 1
                if pred == "UNKNOWN" and len(examples_unknown) < 5:
                    examples_unknown.append((record, token, pred, conf, feature))
                elif pred != "UNKNOWN" and len(examples_wrong) < 8:
                    examples_wrong.append((record, token, pred, conf, feature))
            confusion[gold][pred] += 1

    return {
        "train": len(train),
        "test": len(test),
        "model_features": len(model),
        "total": total,
        "covered": covered,
        "correct": correct,
        "confusion": confusion,
        "by_operation": by_operation,
        "examples_correct": examples_correct,
        "examples_wrong": examples_wrong,
        "examples_unknown": examples_unknown,
    }


def pct(num: int, den: int) -> str:
    return f"{(num / den):.0%}" if den else "0%"


def example_lines(items) -> list[str]:
    lines = []
    for record, token, pred, conf, feature in items:
        feature_text = "none" if feature is None else f"{feature[0]}={feature[1]}"
        wordplay = " ".join(t["text"] for t in wordplay_tokens(record))
        lines.append(f"- `{record['clue']}` ({record['answer']})")
        lines.append(f"  Token: `{token['text']}` gold `{token['label']}`, predicted `{pred}` at `{conf:.0%}` from `{feature_text}`")
        lines.append(f"  Operation: `{record['operation']}`; wordplay tokens: `{wordplay}`")
    return lines


def write_report(records: list[dict[str, Any]], path: Path, test_percent: int, min_support: int) -> None:
    lexical = evaluate(records, lexical_features, min_support, test_percent)
    grammar_eval = evaluate(records, grammar_features, min_support, test_percent)

    def summary(name: str, result: dict[str, Any]) -> list[str]:
        total = result["total"]
        covered = result["covered"]
        correct = result["correct"]
        return [
            f"- `{name}` model features: `{result['model_features']}`",
            f"- `{name}` coverage: `{covered}/{total}` (`{pct(covered, total)}`)",
            f"- `{name}` accuracy: `{correct}/{total}` (`{pct(correct, total)}`)",
            f"- `{name}` accuracy when covered: `{correct}/{covered}` (`{pct(correct, covered)}`)",
        ]

    lines = [
        "# Grammar Boundary Classifier",
        "",
        "Date: 2026-05-17",
        "",
        "This is a small control experiment over the enriched scaffold.",
        "Given wordplay tokens only, it predicts whether each token is `SOURCE` or `RESIDUE`.",
        "The comparison is lexical-only versus lexical-plus-grammar.",
        "",
        f"Train/test records: `{lexical['train']}` / `{lexical['test']}`",
        f"Minimum feature support: `{min_support}`",
        "",
        "## Result",
        "",
    ]
    lines.extend(summary("lexical", lexical))
    lines.extend(summary("grammar", grammar_eval))

    gain = grammar_eval["correct"] - lexical["correct"]
    lines.append(f"- Correct-token gain from grammar: `{gain}`")

    lines.extend(["", "## Lexical Confusion", ""])
    for gold in sorted(lexical["confusion"]):
        lines.append(f"- `{gold}`: " + ", ".join(f"`{pred}`={count}" for pred, count in lexical["confusion"][gold].most_common()))

    lines.extend(["", "## Grammar Confusion", ""])
    for gold in sorted(grammar_eval["confusion"]):
        lines.append(f"- `{gold}`: " + ", ".join(f"`{pred}`={count}" for pred, count in grammar_eval["confusion"][gold].most_common()))

    lines.extend(["", "## Grammar Result By Operation", ""])
    for op in sorted(grammar_eval["by_operation"]):
        counts = grammar_eval["by_operation"][op]
        total = counts["correct"] + counts["wrong"]
        lines.append(f"- `{op}`: `{counts['correct']}/{total}` (`{pct(counts['correct'], total)}`)")

    lines.extend(["", "## Grammar Wrong Examples", ""])
    lines.extend(example_lines(grammar_eval["examples_wrong"]))

    lines.extend(["", "## Grammar Unknown Examples", ""])
    lines.extend(example_lines(grammar_eval["examples_unknown"]))

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This is intentionally crude, but it answers a useful first question.",
            "If grammar improves token boundary classification, the next experiment should move from token labels to span labels: predicting contiguous SOURCE blocks and RESIDUE attachment.",
            "If it does not improve much, grammar may still be useful as a verifier for ambiguous cases rather than as a primary boundary finder.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def gold_spans(tokens: list[dict[str, Any]], label: str) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i, token in enumerate(tokens):
        if token["label"] == label:
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(tokens)))
    return spans


def predicted_label_sequence(tokens: list[dict[str, Any]], model, feature_fn) -> list[str]:
    labels = []
    for i, _token in enumerate(tokens):
        pred, _conf, _feature = predict(tokens, i, model, feature_fn)
        labels.append(pred if pred != "UNKNOWN" else "RESIDUE")
    return labels


def spans_from_labels(labels: list[str], target: str) -> list[tuple[int, int]]:
    spans = []
    start = None
    for i, label in enumerate(labels):
        if label == target:
            if start is None:
                start = i
        elif start is not None:
            spans.append((start, i))
            start = None
    if start is not None:
        spans.append((start, len(labels)))
    return spans


def span_text(tokens: list[dict[str, Any]], span: tuple[int, int]) -> str:
    start, end = span
    return " ".join(t["text"] for t in tokens[start:end])


def boundary_pairs(tokens: list[dict[str, Any]]) -> list[str]:
    return [f"{tokens[i]['label']}|{tokens[i + 1]['label']}" for i in range(len(tokens) - 1)]


def predict_boundary_pairs(labels: list[str]) -> list[str]:
    return [f"{labels[i]}|{labels[i + 1]}" for i in range(len(labels) - 1)]


def evaluate_span_model(records: list[dict[str, Any]], feature_fn, min_support: int, test_percent: int):
    train = [r for r in records if not is_test_record(r, test_percent)]
    test = [r for r in records if is_test_record(r, test_percent)]
    model = build_model(train, feature_fn, min_support)

    exact_source_gold = 0
    exact_source_hit = 0
    exact_residue_gold = 0
    exact_residue_hit = 0
    boundary_total = 0
    boundary_correct = 0
    by_operation: dict[str, Counter[str]] = defaultdict(Counter)
    examples = []

    for record in test:
        tokens = wordplay_tokens(record)
        if not tokens:
            continue
        pred_labels = predicted_label_sequence(tokens, model, feature_fn)
        gold_source = set(gold_spans(tokens, "SOURCE"))
        pred_source = set(spans_from_labels(pred_labels, "SOURCE"))
        gold_residue = set(gold_spans(tokens, "RESIDUE"))
        pred_residue = set(spans_from_labels(pred_labels, "RESIDUE"))

        exact_source_gold += len(gold_source)
        exact_source_hit += len(gold_source & pred_source)
        exact_residue_gold += len(gold_residue)
        exact_residue_hit += len(gold_residue & pred_residue)

        gold_boundaries = boundary_pairs(tokens)
        pred_boundaries = predict_boundary_pairs(pred_labels)
        for gold, pred in zip(gold_boundaries, pred_boundaries):
            boundary_total += 1
            if gold == pred:
                boundary_correct += 1
                by_operation[record["operation"]]["boundary_correct"] += 1
            else:
                by_operation[record["operation"]]["boundary_wrong"] += 1

        if len(examples) < 8 and (gold_source != pred_source or gold_residue != pred_residue):
            examples.append((record, tokens, pred_labels, sorted(gold_source), sorted(pred_source), sorted(gold_residue), sorted(pred_residue)))

    return {
        "train": len(train),
        "test": len(test),
        "exact_source_gold": exact_source_gold,
        "exact_source_hit": exact_source_hit,
        "exact_residue_gold": exact_residue_gold,
        "exact_residue_hit": exact_residue_hit,
        "boundary_total": boundary_total,
        "boundary_correct": boundary_correct,
        "by_operation": by_operation,
        "examples": examples,
    }


def write_span_report(records: list[dict[str, Any]], path: Path, test_percent: int, min_support: int) -> None:
    lexical = evaluate_span_model(records, lexical_features, min_support, test_percent)
    grammar_eval = evaluate_span_model(records, grammar_features, min_support, test_percent)

    def source_rate(result):
        return pct(result["exact_source_hit"], result["exact_source_gold"])

    def residue_rate(result):
        return pct(result["exact_residue_hit"], result["exact_residue_gold"])

    def boundary_rate(result):
        return pct(result["boundary_correct"], result["boundary_total"])

    lines = [
        "# Grammar Span Boundary Experiment",
        "",
        "Date: 2026-05-17",
        "",
        "This experiment moves from token labels to span boundaries.",
        "It uses the same token classifier outputs, then asks whether contiguous `SOURCE` and `RESIDUE` spans are recovered exactly.",
        "",
        f"Train/test records: `{lexical['train']}` / `{lexical['test']}`",
        f"Minimum feature support: `{min_support}`",
        "",
        "## Result",
        "",
        f"- Lexical exact SOURCE span recall: `{lexical['exact_source_hit']}/{lexical['exact_source_gold']}` (`{source_rate(lexical)}`)",
        f"- Grammar exact SOURCE span recall: `{grammar_eval['exact_source_hit']}/{grammar_eval['exact_source_gold']}` (`{source_rate(grammar_eval)}`)",
        f"- Lexical exact RESIDUE span recall: `{lexical['exact_residue_hit']}/{lexical['exact_residue_gold']}` (`{residue_rate(lexical)}`)",
        f"- Grammar exact RESIDUE span recall: `{grammar_eval['exact_residue_hit']}/{grammar_eval['exact_residue_gold']}` (`{residue_rate(grammar_eval)}`)",
        f"- Lexical boundary-pair accuracy: `{lexical['boundary_correct']}/{lexical['boundary_total']}` (`{boundary_rate(lexical)}`)",
        f"- Grammar boundary-pair accuracy: `{grammar_eval['boundary_correct']}/{grammar_eval['boundary_total']}` (`{boundary_rate(grammar_eval)}`)",
        "",
        "## Boundary Accuracy By Operation",
        "",
    ]

    for op in sorted(grammar_eval["by_operation"]):
        counts = grammar_eval["by_operation"][op]
        correct = counts["boundary_correct"]
        total = correct + counts["boundary_wrong"]
        lines.append(f"- `{op}`: `{correct}/{total}` (`{pct(correct, total)}`)")

    lines.extend(["", "## Span Error Examples", ""])
    for record, tokens, pred_labels, gold_source, pred_source, gold_residue, pred_residue in grammar_eval["examples"]:
        lines.append(f"- `{record['clue']}` ({record['answer']})")
        lines.append(f"  Operation: `{record['operation']}`")
        lines.append("  Gold SOURCE: " + "; ".join(f"`{span_text(tokens, s)}`" for s in gold_source))
        lines.append("  Pred SOURCE: " + ("; ".join(f"`{span_text(tokens, s)}`" for s in pred_source) or "`none`"))
        lines.append("  Gold RESIDUE: " + "; ".join(f"`{span_text(tokens, s)}`" for s in gold_residue))
        lines.append("  Pred RESIDUE: " + ("; ".join(f"`{span_text(tokens, s)}`" for s in pred_residue) or "`none`"))
        lines.append("  Pred labels: `" + " ".join(label[0] for label in pred_labels) + "`")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "The span task is much stricter than token classification.",
            "A single misplaced glue word can destroy an otherwise useful source span.",
            "This is why V2 should probably not begin by asking for a single hard parse.",
            "It should generate a small set of plausible block anatomies, then use answer mechanics to verify scope and assembly.",
            "",
        ]
    )

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--span-report-out", type=Path, default=DEFAULT_SPAN_OUT)
    parser.add_argument("--test-percent", type=int, default=25)
    parser.add_argument("--min-support", type=int, default=5)
    args = parser.parse_args()

    records = load_records(args.input)
    write_report(records, args.report_out, args.test_percent, args.min_support)
    write_span_report(records, args.span_report_out, args.test_percent, args.min_support)
    print(f"records={len(records)}")
    print(f"report={args.report_out}")
    print(f"span_report={args.span_report_out}")


if __name__ == "__main__":
    main()
