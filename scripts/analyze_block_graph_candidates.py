"""Analyze first-pass GT V2 block graph candidates.

The aim is to quantify the graph anatomy artifact: node/edge mix, unresolved
residue, and examples that need design inspection.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "block_graph_candidates_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "block_graph_quality_2026-05-17.md"


def load_graphs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def residue_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [n for n in graph.get("nodes", []) if str(n.get("node_id", "")).startswith("res_")]


def compact(counter: Counter[str], limit: int = 12) -> str:
    return ", ".join(f"`{k}`={v}" for k, v in counter.most_common(limit)) or "`none`"


def write_report(graphs: list[dict[str, Any]], path: Path) -> None:
    node_kinds = Counter()
    edge_kinds = Counter()
    scope_statuses = Counter()
    status_counts = Counter()
    unresolved_by_operation: dict[str, Counter[str]] = defaultdict(Counter)
    graphs_with_unresolved = []
    graphs_with_connectors = []
    graphs_missing_definition = []
    graph_size = Counter()

    for graph in graphs:
        status_counts[graph.get("status", "unknown")] += 1
        graph_size[f"{len(graph.get('nodes', []))} nodes/{len(graph.get('edges', []))} edges"] += 1
        has_unresolved = False
        has_connector = False
        if graph.get("definition_status") != "observed" and len(graphs_missing_definition) < 20:
            graphs_missing_definition.append(graph)
        for node in graph.get("nodes", []):
            kind = node.get("kind", "unknown")
            node_kinds[kind] += 1
            if node.get("status") == "ambiguous":
                has_unresolved = True
                if kind != "DEF_BLOCK":
                    unresolved_by_operation[graph.get("nodes", [{}])[0].get("operation", "unknown")][kind] += 1
            if kind == "CONNECTOR_BLOCK":
                has_connector = True
        for edge in graph.get("edges", []):
            edge_kinds[edge.get("kind", "unknown")] += 1
            scope_statuses[edge.get("scope_status", "unknown")] += 1
            if edge.get("kind") == "UNRESOLVED":
                has_unresolved = True
        if has_unresolved and len(graphs_with_unresolved) < 10:
            graphs_with_unresolved.append(graph)
        if has_connector and len(graphs_with_connectors) < 10:
            graphs_with_connectors.append(graph)

    lines = [
        "# Block Graph Quality",
        "",
        "Date: 2026-05-17",
        "",
        "This report quantifies the first-pass block graph candidate artifact.",
        "It is intended to show where the graph representation is informative and where it remains too weak.",
        "",
        f"Graphs analysed: `{len(graphs)}`",
        "",
        "## Status",
        "",
        compact(status_counts),
        "",
        "## Node Kinds",
        "",
        compact(node_kinds, 20),
        "",
        "## Edge Kinds",
        "",
        compact(edge_kinds, 20),
        "",
        "## Scope Statuses",
        "",
        compact(scope_statuses, 20),
        "",
        "## Common Graph Sizes",
        "",
        compact(graph_size, 12),
        "",
        "## Unresolved Node Kinds By Operation",
        "",
    ]

    for operation in sorted(unresolved_by_operation):
        lines.append(f"- `{operation}`: {compact(unresolved_by_operation[operation], 10)}")

    lines.extend(["", "## Unresolved Examples", ""])
    for graph in graphs_with_unresolved:
        lines.append(f"- `{graph['clue']}` ({graph['answer']})")
        for node in residue_nodes(graph):
            if node.get("status") == "ambiguous":
                lines.append(f"  `{node['text']}` -> `{node['kind']}` / `{node['status']}`")

    lines.extend(["", "## Connector Examples", ""])
    for graph in graphs_with_connectors:
        lines.append(f"- `{graph['clue']}` ({graph['answer']})")
        for node in residue_nodes(graph):
            if node.get("kind") == "CONNECTOR_BLOCK":
                lines.append(f"  `{node['text']}` -> connector candidate")

    lines.extend(["", "## Missing Definition Examples", ""])
    if not graphs_missing_definition:
        lines.append("`none`")
    for graph in graphs_missing_definition:
        lines.append(f"- `{graph['clue']}` ({graph['answer']})")
        residue_text = "; ".join(n.get("text", "") for n in residue_nodes(graph) if n.get("text"))
        if residue_text:
            lines.append(f"  Residue nodes: `{residue_text}`")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "The graph artifact is useful if it makes unresolved anatomy explicit.",
            "The key next design pressure is reducing ambiguous `SCOPE_BLOCK` nodes by deriving operation-specific attachment from structured explanations.",
            "Connector candidates need grammar objections, especially where a preposition is orphaned or source-internal.",
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
