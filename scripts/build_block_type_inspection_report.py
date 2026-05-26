"""Build a compact inspection report grouped by GT V2 graph block type.

The graph candidate JSONL is intentionally detailed. This report selects
readable examples by node kind so we can inspect the anatomy one block type at
a time without dumping tables.
"""

from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "block_graph_candidates_2026-05-17.jsonl"
DEFAULT_OUT = PROJECT_ROOT / "documents" / "block_type_inspection_2026-05-17.md"

KIND_ORDER = [
    "DEF_BLOCK",
    "DEF_MODIFIER_BLOCK",
    "RELATION_BLOCK",
    "CONNECTOR_BLOCK",
    "LOCATOR_BLOCK",
    "POSITION_BLOCK",
    "OP_BLOCK",
    "SCOPE_BLOCK",
]


def load_graphs(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def inspectable_nodes(graph: dict[str, Any]) -> list[dict[str, Any]]:
    return [
        n
        for n in graph.get("nodes", [])
        if str(n.get("node_id", "")).startswith(("def_", "res_"))
    ]


def outgoing_edges(graph: dict[str, Any], node_id: str) -> list[dict[str, Any]]:
    return [e for e in graph.get("edges", []) if e.get("from_node") == node_id]


def node_by_id(graph: dict[str, Any]) -> dict[str, dict[str, Any]]:
    return {n.get("node_id"): n for n in graph.get("nodes", [])}


def clean_evidence(node: dict[str, Any], source: str) -> str:
    details = [e.get("detail", "") for e in node.get("evidence", []) if e.get("source") == source]
    return "; ".join(d for d in details if d) or "none"


def select_examples(items: list[tuple[dict[str, Any], dict[str, Any]]], limit: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    chosen = []
    seen_ops = set()
    for graph, node in items:
        op = graph.get("nodes", [{}])[0].get("operation")
        if op in seen_ops and len(seen_ops) < 4:
            continue
        chosen.append((graph, node))
        seen_ops.add(op)
        if len(chosen) >= limit:
            break
    return chosen


def select_ambiguous(items: list[tuple[dict[str, Any], dict[str, Any]]], limit: int) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    return [(graph, node) for graph, node in items if node.get("status") == "ambiguous"][:limit]


def write_report(graphs: list[dict[str, Any]], path: Path, limit: int) -> None:
    by_kind: dict[str, list[tuple[dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    node_counts = Counter()
    ambiguous_counts = Counter()

    for graph in graphs:
        for node in inspectable_nodes(graph):
            kind = node.get("kind", "UNKNOWN")
            node_counts[kind] += 1
            if node.get("status") == "ambiguous":
                ambiguous_counts[kind] += 1
            by_kind[kind].append((graph, node))

    lines = [
        "# Block Type Inspection",
        "",
        "Date: 2026-05-17",
        "",
        "This report samples graph nodes by block type for manual anatomy inspection.",
        "The examples are weakly generated and should be treated as design evidence, not truth.",
        "",
        "## Counts",
        "",
    ]

    for kind, count in node_counts.most_common():
        ambiguous = ambiguous_counts[kind]
        if ambiguous:
            lines.append(f"- `{kind}`: {count} nodes, {ambiguous} ambiguous")
        else:
            lines.append(f"- `{kind}`: {count} nodes")

    for kind in KIND_ORDER:
        items = by_kind.get(kind, [])
        if not items:
            continue
        lines.extend(["", f"## {kind}", ""])
        lines.append(f"Available nodes: `{len(items)}`")
        lines.append("")
        examples = select_ambiguous(items, limit) if kind == "DEF_BLOCK" else select_examples(items, limit)
        if kind == "DEF_BLOCK" and examples:
            lines.append("Ambiguous definition gaps:")
            lines.append("")
        for graph, node in examples:
            node_lookup = node_by_id(graph)
            edges = outgoing_edges(graph, node.get("node_id"))
            lines.append(f"- `{graph['clue']}` ({graph['answer']})")
            node_text = node.get("text") or "<definition gap>"
            lines.append(f"  Node: `{node_text}` / `{node.get('status')}`")
            lines.append(f"  Operation: `{graph.get('nodes', [{}])[0].get('operation')}`")
            lines.append(f"  Span: `{node.get('span')}`")
            lines.append(f"  Residue evidence: `{clean_evidence(node, 'residue_lexicon')}`")
            split = clean_evidence(node, "residue_split")
            if split != "none":
                lines.append(f"  Split evidence: `{split}`")
            if edges:
                rendered = []
                for edge in edges:
                    target = node_lookup.get(edge.get("to_node"), {})
                    target_text = target.get("text") or "<definition gap>"
                    rendered.append(
                        f"{edge.get('kind')} -> {edge.get('to_node')}:{target_text} [{edge.get('scope_status')}]"
                    )
                lines.append("  Edges: `" + "; ".join(rendered) + "`")
            else:
                lines.append("  Edges: `none`")

    lines.extend(
        [
            "",
            "## Inspection Prompts",
            "",
            "- Should this node kind exist as shown, or should it split/merge with a neighbour?",
            "- Does the edge point to the right target, or merely to the nearest available source?",
            "- Is the node really wordplay, definition modifier, source-internal phrase material, or surface grammar?",
            "- Is scope known now, or should the graph preserve `AWAITING_SCOPE` until answer verification?",
            "",
        ]
    )

    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, default=DEFAULT_IN)
    parser.add_argument("--report-out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--limit", type=int, default=5)
    args = parser.parse_args()

    graphs = load_graphs(args.input)
    write_report(graphs, args.report_out, args.limit)
    print(f"graphs={len(graphs)}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
