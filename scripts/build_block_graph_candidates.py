"""Build first-pass GT V2 block graph candidates from attachment slice rows.

This is an R&D artifact generator. It converts weak operation-attachment rows
into graph-shaped candidate anatomies: block nodes, relationship edges, and
evidence. It does not solve clues.
"""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_IN = PROJECT_ROOT / "documents" / "operation_attachment_slice_2026-05-17.jsonl"
DEFAULT_JSONL = PROJECT_ROOT / "documents" / "block_graph_candidates_2026-05-17.jsonl"
DEFAULT_REPORT = PROJECT_ROOT / "documents" / "block_graph_candidates_2026-05-17.md"


def load_rows(path: Path) -> list[dict[str, Any]]:
    with path.open(encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def clue_key(row: dict[str, Any]) -> tuple[Any, ...]:
    return (
        row.get("id"),
        row.get("source"),
        row.get("puzzle_number"),
        row.get("clue"),
        row.get("answer"),
        row.get("operation"),
    )


def evidence(source: str, detail: str, strength: str = "weak") -> dict[str, Any]:
    return {
        "source": source,
        "detail": detail,
        "supports": True,
        "opposes": False,
        "strength": strength,
    }


def source_node_id(index: int) -> str:
    return f"src_{index}"


def residue_node_id(index: int) -> str:
    return f"res_{index}"


def definition_node_id(index: int) -> str:
    return f"def_{index}"


def kind_for_label(label: str) -> str:
    if label == "OPERATOR_SCOPE":
        return "OP_BLOCK"
    if label == "LOCATOR_SCOPE":
        return "LOCATOR_BLOCK"
    if label == "ORDER_OR_POSITION":
        return "POSITION_BLOCK"
    if label == "CONTAINER_RELATION":
        return "RELATION_BLOCK"
    if label == "CONNECTOR_OR_SURFACE":
        return "CONNECTOR_BLOCK"
    if label == "DIRECTION_OR_ORIENTATION":
        return "SCOPE_BLOCK"
    if label == "DEF_MODIFIER":
        return "DEF_MODIFIER_BLOCK"
    return "SCOPE_BLOCK"


def operation_for_kind(kind: str, operation: str) -> str | None:
    if kind in {"OP_BLOCK", "RELATION_BLOCK", "SCOPE_BLOCK"}:
        return operation
    return None


def scope_status_for_label(label: str) -> str:
    if label in {"DIRECTION_OR_ORIENTATION", "ORDER_OR_POSITION"}:
        return "answer_aware_scope_needed"
    if label in {"OPERATOR_SCOPE", "CONTAINER_RELATION", "LOCATOR_SCOPE"}:
        return "weakly_scoped_from_operation"
    if label == "SOURCE_INTERNAL":
        return "candidate_source_absorption"
    if label == "DEF_MODIFIER":
        return "definition_modifier_candidate"
    if label == "CONNECTOR_OR_SURFACE":
        return "surface_only_until_grammar_check"
    return "unresolved"


def make_graph(key: tuple[Any, ...], rows: list[dict[str, Any]]) -> dict[str, Any]:
    clue_id, source, puzzle_number, clue, answer, operation = key
    first = rows[0]
    nodes = []
    edges = []

    nodes.append(
        {
            "node_id": "answer",
            "kind": "ASSEMBLY_BLOCK",
            "span": None,
            "text": answer,
            "normalised_text": str(answer).upper().replace(" ", ""),
            "source_piece_ids": [],
            "operation": operation,
            "value": answer,
            "evidence": [evidence("structured_explanation", f"assembly operation {operation}")],
            "status": "inferred",
        }
    )

    seen_sources = {}
    source_id_to_node = {}
    definition_nodes = []
    seen_definitions = {}
    for definition in first.get("definition_spans", []):
        span = tuple(definition.get("span") or [])
        text = definition.get("text") or ""
        key_span = (span, text)
        if key_span in seen_definitions:
            continue
        node_id = definition_node_id(len(seen_definitions))
        seen_definitions[key_span] = node_id
        definition_nodes.append(node_id)
        nodes.append(
            {
                "node_id": node_id,
                "kind": "DEF_BLOCK",
                "span": list(span),
                "span_space": "full_clue_tokens",
                "text": text,
                "normalised_text": text.lower(),
                "source_piece_ids": [],
                "operation": None,
                "value": answer,
                "evidence": [evidence("structured_explanation", "definition span")],
                "status": "observed",
            }
        )
        edges.append(
            {
                "edge_id": f"edge_{len(edges)}",
                "kind": "DEFINES",
                "from_node": node_id,
                "to_node": "answer",
                "evidence": [evidence("structured_explanation", "definition defines answer")],
                "scope_status": "known",
                "confidence": "weak",
                "notes": "",
            }
        )
    if not definition_nodes:
        node_id = definition_node_id(0)
        definition_nodes.append(node_id)
        nodes.append(
            {
                "node_id": node_id,
                "kind": "DEF_BLOCK",
                "span": None,
                "span_space": None,
                "text": "",
                "normalised_text": "",
                "source_piece_ids": [],
                "operation": None,
                "value": answer,
                "evidence": [evidence("structured_explanation", "no explicit definition span")],
                "status": "ambiguous",
            }
        )

    for row in rows:
        for span_index, source_span in enumerate(row.get("source_spans", [])):
            span = tuple(source_span.get("span") or [])
            text = source_span.get("text") or ""
            key_span = (span, text)
            if key_span in seen_sources:
                continue
            node_id = source_node_id(len(seen_sources))
            seen_sources[key_span] = node_id
            if source_span.get("source_span_id") is not None:
                source_id_to_node[source_span.get("source_span_id")] = node_id
            piece_indices = source_span.get("piece_indices", [])
            nodes.append(
                {
                    "node_id": node_id,
                    "kind": "SOURCE_BLOCK",
                    "span": list(span),
                    "span_space": "wordplay_tokens",
                    "text": text,
                    "normalised_text": text.lower(),
                    "source_piece_ids": piece_indices,
                    "operation": None,
                    "value": "".join(
                        str(letters)
                        for letters in source_span.get("letters", [])
                        if letters is not None
                    )
                    or None,
                    "mechanisms": source_span.get("mechanisms", []),
                    "evidence": [
                        evidence("structured_explanation", "mapped source span"),
                        evidence(
                            "structured_explanation",
                            "mechanisms "
                            + ", ".join(source_span.get("mechanisms", []) or ["unknown"]),
                        ),
                    ],
                    "status": "observed",
                }
            )
            edges.append(
                {
                    "edge_id": f"edge_{len(edges)}",
                    "kind": "CONTRIBUTES_TO",
                    "from_node": node_id,
                    "to_node": "answer",
                    "evidence": [evidence("structured_explanation", "source contributes to answer")],
                    "scope_status": "known",
                    "confidence": "weak",
                    "notes": "",
                }
            )

    for i, row in enumerate(rows):
        residue = row.get("residue_run") or {}
        if row.get("needs_split"):
            start = (residue.get("span") or [0, 0])[0]
            residue_items = []
            for offset, annotation in enumerate(row.get("token_annotations", [])):
                label = annotation.get("weak_attachment_label")
                residue_items.append(
                    {
                        "label": label,
                        "relationship": annotation.get("block_relationship"),
                        "scope_status": scope_status_for_label(label),
                        "span": [start + offset, start + offset + 1],
                        "text": annotation.get("text"),
                        "clean_words": [annotation.get("clean")],
                        "split_from": residue.get("text"),
                    }
                )
        else:
            residue_items = [
                {
                    "label": row.get("weak_attachment_label"),
                    "relationship": row.get("block_relationship"),
                    "scope_status": row.get("scope_status"),
                    "span": residue.get("span"),
                    "text": residue.get("text"),
                    "clean_words": residue.get("clean_words", []),
                    "split_from": None,
                }
            ]

        target_nodes = []
        adjacent = row.get("adjacent_source_span_ids") or {}
        for side in ("left", "right"):
            source_span_id = adjacent.get(side)
            if source_span_id in source_id_to_node:
                target_nodes.append(source_id_to_node[source_span_id])
        for head in row.get("parser_source_heads") or []:
            source_span_id = head.get("head_source_span_id")
            if source_span_id in source_id_to_node:
                target_nodes.append(source_id_to_node[source_span_id])

        base_target_nodes = list(dict.fromkeys(target_nodes))

        for item_index, item in enumerate(residue_items):
            label = item["label"]
            relationship = item["relationship"]
            scope_status = item["scope_status"]
            kind = kind_for_label(label)
            node_id = residue_node_id(i) if len(residue_items) == 1 else f"{residue_node_id(i)}_{item_index}"
            nodes.append(
                {
                    "node_id": node_id,
                    "kind": kind,
                    "span": item["span"],
                    "span_space": "wordplay_tokens",
                    "text": item["text"],
                    "normalised_text": " ".join(w for w in item.get("clean_words", []) if w),
                    "source_piece_ids": [],
                    "operation": operation_for_kind(kind, operation),
                    "value": None,
                    "evidence": [
                        evidence("operation_attachment_slice", f"weak label {label}"),
                        evidence("residue_lexicon", ", ".join(w for w in item.get("clean_words", []) if w)),
                    ],
                    "status": "ambiguous" if label == "UNCLASSIFIED_RESIDUE" else "inferred",
                }
            )
            if item.get("split_from"):
                nodes[-1]["evidence"].append(
                    evidence("residue_split", f"split from residue run {item['split_from']}")
                )
            item_target_nodes = list(base_target_nodes)
            if relationship == "MODIFIES_DEFINITION":
                item_target_nodes = definition_nodes or ["answer"]
            elif not item_target_nodes and relationship in {"OPERATES_ON", "CONTAINS", "LOCATES_WITHIN", "AWAITING_SCOPE"}:
                item_target_nodes = list(seen_sources.values())
            elif not item_target_nodes:
                item_target_nodes = ["answer"]
            item_target_nodes = list(dict.fromkeys(item_target_nodes))

            for target in item_target_nodes:
                edges.append(
                    {
                        "edge_id": f"edge_{len(edges)}",
                        "kind": relationship,
                        "from_node": node_id,
                        "to_node": target,
                        "evidence": [
                            evidence("operation_attachment_slice", f"weak relationship {relationship}"),
                            evidence("grammar_dependency", json.dumps(row.get("grammar_evidence", []), ensure_ascii=False)),
                        ],
                        "scope_status": scope_status,
                        "confidence": "weak",
                        "notes": "operation-derived weak attachment; requires inspection",
                    }
                )

    return {
        "clue_id": clue_id,
        "source": source,
        "puzzle_number": puzzle_number,
        "clue": clue,
        "answer": answer,
        "candidate_id": f"{source}:{puzzle_number}:{clue_id}:candidate_0",
        "definition_status": (
            "observed"
            if first.get("definition_spans")
            else "missing_from_structured_explanation"
        ),
        "nodes": nodes,
        "edges": edges,
        "global_evidence": [
            evidence("structured_explanation", "source spans and operation from structured explanation"),
            evidence("grammar_scaffold", "POS/dependency features from enriched scaffold"),
        ],
        "status": "candidate",
        "notes": "First-pass graph from weak operation attachment slice.",
    }


def build_graphs(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[Any, ...], list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        grouped[clue_key(row)].append(row)
    return [make_graph(key, grouped[key]) for key in grouped]


def write_outputs(graphs: list[dict[str, Any]], jsonl_path: Path, report_path: Path) -> None:
    with jsonl_path.open("w", encoding="utf-8") as f:
        for graph in graphs:
            f.write(json.dumps(graph, ensure_ascii=False) + "\n")

    lines = [
        "# Block Graph Candidates",
        "",
        "Date: 2026-05-17",
        "",
        "This is the first graph-shaped GT V2 anatomy artifact.",
        "It is generated from weak operation attachment labels and is intended for inspection, not solving.",
        "",
        f"Graphs: `{len(graphs)}`",
        "",
        "## First Examples",
        "",
    ]

    for graph in graphs[:8]:
        lines.append(f"- `{graph['clue']}` ({graph['answer']})")
        lines.append(f"  Operation: `{graph['nodes'][0]['operation']}`")
        lines.append(f"  Nodes: `{len(graph['nodes'])}`; edges: `{len(graph['edges'])}`")
        residue_nodes = [n for n in graph["nodes"] if n["node_id"].startswith("res_")]
        for node in residue_nodes[:3]:
            lines.append(f"  Residue node: `{node['text']}` -> `{node['kind']}` / `{node['status']}`")

    lines.extend(
        [
            "",
            "## Reading",
            "",
            "The useful question is now whether these graph candidates preserve the right distinctions.",
            "In particular, inspect whether `OP_BLOCK`, `LOCATOR_BLOCK`, and `CONNECTOR_BLOCK` nodes should be split, merged, or retyped.",
            "Answer verification should come later; this artifact is still anatomy research.",
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

    rows = load_rows(args.input)
    graphs = build_graphs(rows)
    write_outputs(graphs, args.jsonl_out, args.report_out)
    print(f"rows={len(rows)}")
    print(f"graphs={len(graphs)}")
    print(f"jsonl={args.jsonl_out}")
    print(f"report={args.report_out}")


if __name__ == "__main__":
    main()
