"""Inspect graph candidates with no observed definition span.

This is an R&D report, not solver logic. It asks what kind of anatomy may be
hidden behind `definition_status = missing_from_structured_explanation`.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "block_graph_candidates_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "definition_gap_analysis_2026-05-17.md"

DBE_MARKERS = {"maybe", "perhaps", "say", "eg", "e.g.", "like"}


def load_graphs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def nodes_of_kind(graph: dict[str, Any], kind: str) -> list[dict[str, Any]]:
    return [n for n in graph.get("nodes", []) if n.get("kind") == kind]


def residue_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [n for n in graph.get("nodes", []) if str(n.get("node_id", "")).startswith("res_")]


def operation(graph: dict[str, Any]) -> str:
    answer = next((n for n in graph.get("nodes", []) if n.get("node_id") == "answer"), {})
    return answer.get("operation") or "unknown"


def words(text: str) -> set[str]:
    return {
        part.strip(".,;:?!()[]{}\"'").lower()
        for part in text.split()
        if part.strip(".,;:?!()[]{}\"'")
    }


def classify_gap(graph: dict[str, Any]) -> str:
    op = operation(graph)
    clue_words = words(graph.get("clue", ""))
    residue_text = " ".join(str(n.get("text", "")) for n in residue_nodes(graph))
    residue_words = words(residue_text)
    source_nodes = nodes_of_kind(graph, "SOURCE_BLOCK")

    has_question = "?" in graph.get("clue", "")
    has_dbe = bool((clue_words | residue_words) & DBE_MARKERS)
    source_mechanisms = {
        mechanism
        for node in source_nodes
        for mechanism in node.get("mechanisms", [])
    }

    if op.startswith("hidden") and len(source_nodes) == 1:
        if has_question:
            return "hidden clue with possible all-in-one or cryptic definition surface"
        return "hidden clue with definition omitted by structured explanation"
    if "last_letter" in source_mechanisms:
        return "letter-selection source may have swallowed locator or definition surface"
    if has_dbe:
        return "definition-by-example marker present but definition anchor missing"
    if has_question:
        return "question-mark surface may be carrying definition or cryptic definition force"
    if op in {"anagram", "charade"} and len(source_nodes) == 1:
        return "single mapped source with surrounding residue possibly acting as definition"
    return "unclassified definition gap"


def render_nodes(nodes: list[dict[str, Any]]) -> str:
    rendered = []
    for node in nodes:
        if not node.get("text"):
            continue
        mechanisms = ",".join(node.get("mechanisms", []))
        value = node.get("value")
        suffix_parts = []
        if mechanisms:
            suffix_parts.append(mechanisms)
        if value:
            suffix_parts.append(f"value={value}")
        suffix = f" ({'; '.join(suffix_parts)})" if suffix_parts else ""
        rendered.append(f"{node.get('text')}{suffix}")
    text = "; ".join(rendered)
    return text or "none"


def write_report(graphs: list[dict[str, Any]], path: Path) -> None:
    gaps = [g for g in graphs if g.get("definition_status") == "missing_from_structured_explanation"]
    by_class = Counter(classify_gap(g) for g in gaps)

    lines = [
        "# Definition Gap Analysis",
        "",
        "Date: 2026-05-17",
        "",
        "This report inspects graph candidates where the structured explanation did not preserve an explicit definition span.",
        "These are not automatically bad records. They may be hidden clues, all-in-one surfaces, cryptic definitions, or explanation gaps.",
        "",
        f"Definition gaps inspected: `{len(gaps)}`",
        "",
        "## Hypotheses",
        "",
    ]
    for label, count in by_class.most_common():
        lines.append(f"- `{label}`: {count}")

    lines.extend(["", "## Examples", ""])
    for graph in gaps:
        lines.append(f"- `{graph['clue']}` ({graph['answer']})")
        lines.append(f"  Operation: `{operation(graph)}`")
        lines.append(f"  Hypothesis: `{classify_gap(graph)}`")
        lines.append(f"  Source blocks: `{render_nodes(nodes_of_kind(graph, 'SOURCE_BLOCK'))}`")
        lines.append(f"  Residue blocks: `{render_nodes(residue_nodes(graph))}`")

    lines.extend(
        [
            "",
            "## Design Consequence",
            "",
            "A missing definition must remain explicit in the graph.",
            "The right model is not `no definition`; it is `definition candidate not yet located`.",
            "That allows later grammar and answer-mechanics checks to propose whole-surface, residue-surface, or implicit definition candidates without corrupting the wordplay blocks.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()

    graphs = load_graphs(args.input)
    write_report(graphs, args.report_out)
    print(f"graphs={len(graphs)}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
