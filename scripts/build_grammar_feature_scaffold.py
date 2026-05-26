"""Build a GT V2 grammar-feature scaffold from the clean boundary slice.

The scaffold is deliberately descriptive, not a solver change. It stores the
same clue tokens with their supervised block labels, residue runs, source spans,
and optional spaCy grammar features when the model is available.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "clean_boundary_training_slice_2026-05-17.jsonl"
DEFAULT_JSONL = PROJECT_ROOT / "documents" / "grammar_feature_scaffold_2026-05-17.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "documents" / "grammar_feature_scaffold_2026-05-17.md"

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
    "almost",
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

MID_MAP = {
    "NN": "N",
    "NNS": "N",
    "NNP": "NP",
    "NNPS": "NP",
    "VB": "Vb",
    "VBD": "Vi",
    "VBG": "Vi",
    "VBN": "Vi",
    "VBP": "Vb",
    "VBZ": "Vi",
    "JJ": "J",
    "JJR": "J",
    "JJS": "J",
    "RB": "R",
    "RBR": "R",
    "RBS": "R",
    "IN": "P",
    "TO": "P",
    "DT": "D",
    "WDT": "D",
    "PDT": "D",
    "CC": "C",
    "RP": "RP",
    "CD": "CD",
    "PRP": "PR",
    "PRP$": "PR",
    "WP": "PR",
    "MD": "MD",
}


def load_records(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clean_token(text: str) -> str:
    return re.sub(r"^[^\w]+|[^\w]+$", "", text.lower())


def assembly_op(record: dict[str, Any]) -> str:
    assembly = record.get("assembly") or {}
    if isinstance(assembly, dict):
        return assembly.get("op") or (record.get("wordplay_types") or ["blank"])[0]
    return (record.get("wordplay_types") or ["blank"])[0]


def source_span_ids(tokens: list[dict[str, Any]]) -> dict[int, int]:
    spans = {}
    current_piece = None
    span_id = -1
    for token in tokens:
        idx = token["index"]
        piece = token.get("piece_index")
        if token.get("label") == "SOURCE":
            if piece != current_piece:
                span_id += 1
                current_piece = piece
            spans[idx] = span_id
        else:
            current_piece = None
    return spans


def residue_run_ids(tokens: list[dict[str, Any]]) -> dict[int, int]:
    runs = {}
    run_id = -1
    previous_was_residue = False
    for token in tokens:
        idx = token["index"]
        if token.get("label") == "RESIDUE":
            if not previous_was_residue:
                run_id += 1
            runs[idx] = run_id
            previous_was_residue = True
        else:
            previous_was_residue = False
    return runs


def load_spacy():
    try:
        import spacy

        return spacy.load("en_core_web_sm")
    except Exception:
        return None


def spacy_features(tokens: list[dict[str, Any]], nlp) -> dict[int, dict[str, Any]]:
    if nlp is None:
        return {}
    words = [t["text"] for t in tokens]
    doc = nlp(" ".join(words))
    features = {}
    cursor = 0
    for token in tokens:
        if cursor >= len(doc):
            break
        parsed = doc[cursor]
        features[token["index"]] = {
            "pos": parsed.pos_,
            "tag": parsed.tag_,
            "mid_pos": MID_MAP.get(parsed.tag_, "X"),
            "dep": parsed.dep_,
            "head_text": parsed.head.text,
            "head_i": parsed.head.i,
        }
        consumed = len(parsed.text)
        cursor += 1
        while consumed < len(token["text"]) and cursor < len(doc):
            consumed += len(doc[cursor].text)
            cursor += 1
    return features


def scaffold_record(record: dict[str, Any], nlp) -> dict[str, Any]:
    tokens = record.get("token_labels", [])
    source_spans = source_span_ids(tokens)
    residue_runs = residue_run_ids(tokens)
    grammar = spacy_features(tokens, nlp)
    out_tokens = []
    for i, token in enumerate(tokens):
        idx = token["index"]
        clean = clean_token(token["text"])
        item = {
            "index": idx,
            "text": token["text"],
            "clean": clean,
            "label": token.get("label"),
            "piece_index": token.get("piece_index"),
            "source_span_id": source_spans.get(idx),
            "residue_run_id": residue_runs.get(idx),
            "is_glue_word": clean in GLUE,
            "is_operationish_word": clean in OPERATIONISH,
            "prev_label": tokens[i - 1].get("label") if i > 0 else "<START>",
            "next_label": tokens[i + 1].get("label") if i < len(tokens) - 1 else "<END>",
        }
        if idx in grammar:
            item["grammar"] = grammar[idx]
        out_tokens.append(item)
    return {
        "id": record.get("id"),
        "source": record.get("source"),
        "puzzle_number": record.get("puzzle_number"),
        "clue": record.get("clue"),
        "answer": record.get("answer"),
        "operation": assembly_op(record),
        "tokens": out_tokens,
        "pieces": record.get("pieces", []),
    }


def write_outputs(records: list[dict[str, Any]], jsonl_path: Path, report_path: Path) -> None:
    nlp = load_spacy()
    scaffolded = [scaffold_record(record, nlp) for record in records]

    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    with jsonl_path.open("w", encoding="utf-8") as f:
        for record in scaffolded:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

    label_contexts: dict[str, Counter[str]] = defaultdict(Counter)
    glue_contexts = Counter()
    opish_contexts = Counter()
    source_span_lengths = Counter()
    residue_run_lengths = Counter()
    op_counts = Counter(r["operation"] for r in scaffolded)

    for record in scaffolded:
        current_source_counts = Counter()
        current_residue_counts = Counter()
        for token in record["tokens"]:
            context = f"{token['prev_label']}<{token['label']}>{token['next_label']}"
            label_contexts[token["label"]][context] += 1
            if token["is_glue_word"]:
                glue_contexts[f"{token['clean']}:{context}"] += 1
            if token["is_operationish_word"]:
                opish_contexts[f"{token['clean']}:{context}"] += 1
            if token["source_span_id"] is not None:
                current_source_counts[token["source_span_id"]] += 1
            if token["residue_run_id"] is not None:
                current_residue_counts[token["residue_run_id"]] += 1
        source_span_lengths.update(current_source_counts.values())
        residue_run_lengths.update(current_residue_counts.values())

    lines = [
        "# Grammar Feature Scaffold",
        "",
        "Date: 2026-05-17",
        "",
        "This scaffold stores supervised token boundaries from structured explanations, with optional spaCy features when available.",
        "It is an R&D artifact for block anatomy, not a solver change.",
        "",
        f"Records: `{len(scaffolded)}`",
        f"spaCy model available: `{'yes' if nlp is not None else 'no'}`",
        "",
        "## Operation Mix",
        "",
    ]
    lines.append(", ".join(f"`{op}`={count}" for op, count in op_counts.most_common()))

    lines.extend(["", "## Source Span Lengths", ""])
    lines.append(", ".join(f"`{length}` words={count}" for length, count in source_span_lengths.most_common()))

    lines.extend(["", "## Residue Run Lengths", ""])
    lines.append(", ".join(f"`{length}` words={count}" for length, count in residue_run_lengths.most_common()))

    lines.extend(["", "## Common Label Contexts", ""])
    for label in ["DEF", "SOURCE", "RESIDUE"]:
        top = ", ".join(f"`{ctx}`={count}" for ctx, count in label_contexts[label].most_common(10))
        lines.append(f"- `{label}`: {top}")

    lines.extend(["", "## Glue Word Contexts", ""])
    for item, count in glue_contexts.most_common(20):
        lines.append(f"- `{item}` = {count}")

    lines.extend(["", "## Operationish Word Contexts", ""])
    for item, count in opish_contexts.most_common(20):
        lines.append(f"- `{item}` = {count}")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "This creates the shared surface needed for the next experiment.",
            "The immediate question is whether grammar features, when added, improve source-boundary and residue-attachment decisions beyond the residue-only baseline.",
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
    write_outputs(records, args.jsonl_out, args.report_out)
    print(f"records={len(records)}")
    print(f"jsonl={args.jsonl_out}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
