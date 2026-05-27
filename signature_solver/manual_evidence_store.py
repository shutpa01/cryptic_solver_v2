"""Persistence and display merge for manual evidence nodes.

Manual evidence is stored separately from automatic proof attempts. It survives
rerun and is merged into the final clue display at load time.

This is display-merge only. Nothing here writes to wfw_proof_attempts or
changes the clue's proof status in the database. The merge is visible to admin
users only. Public display is unchanged.
"""
from __future__ import annotations

import json
import re
import sqlite3
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLUES_DB = PROJECT_ROOT / "data" / "clues_master.db"


_NODES_DDL = """
CREATE TABLE IF NOT EXISTS manual_evidence_nodes (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clue_id INTEGER NOT NULL,
    node_type TEXT NOT NULL,
    word_indices TEXT NOT NULL,
    word_text TEXT NOT NULL,
    role TEXT,
    raw_letters TEXT,
    answer_positions TEXT,
    group_id INTEGER,
    source TEXT NOT NULL DEFAULT 'manual',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_EDGES_DDL = """
CREATE TABLE IF NOT EXISTS manual_evidence_edges (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clue_id INTEGER NOT NULL,
    from_node_id INTEGER NOT NULL,
    to_node_id INTEGER NOT NULL,
    edge_type TEXT NOT NULL,
    created_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_IDX_NODES_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_manual_evidence_nodes_clue "
    "ON manual_evidence_nodes (clue_id)"
)

_IDX_EDGES_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_manual_evidence_edges_clue "
    "ON manual_evidence_edges (clue_id)"
)

_COLOUR_TO_ROLE = {
    "blue": "piece_0",
    "pink": "piece_1",
    "yellow": "piece_2",
    "orange": "piece_3",
    "purple": "piece_4",
}

_SOURCE_STRUCTURED_RELATIONSHIPS = {
    "synonym",
    "abbreviation",
    "literal_letters",
    "first_letter",
    "last_letter",
    "proper_noun",
}

_OPERATION_STRUCTURED_RELATIONSHIPS = _SOURCE_STRUCTURED_RELATIONSHIPS | {
    "initial_letters",
    "foreign",
    "pronoun",
    "single_letter",
    "hidden_letters",
}

_STRUCTURED_PARSES_DDL = """
CREATE TABLE IF NOT EXISTS manual_structured_parses (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    clue_id INTEGER NOT NULL UNIQUE,
    parse_json TEXT NOT NULL,
    source TEXT NOT NULL DEFAULT 'human',
    status TEXT NOT NULL DEFAULT 'draft',
    created_at TEXT NOT NULL DEFAULT (datetime('now')),
    updated_at TEXT NOT NULL DEFAULT (datetime('now'))
)
"""

_IDX_STRUCTURED_PARSES_DDL = (
    "CREATE INDEX IF NOT EXISTS idx_manual_structured_parses_clue "
    "ON manual_structured_parses (clue_id)"
)


def _add_payload_json_column(conn):
    """Add payload_json column to manual_evidence_nodes if missing."""
    try:
        conn.execute(
            "ALTER TABLE manual_evidence_nodes ADD COLUMN payload_json TEXT"
        )
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise


def ensure_tables(conn=None):
    """Create manual evidence tables and indices if they are missing."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        conn.execute(_NODES_DDL)
        conn.execute(_EDGES_DDL)
        conn.execute(_IDX_NODES_DDL)
        conn.execute(_IDX_EDGES_DDL)
        _add_payload_json_column(conn)
        conn.execute(_STRUCTURED_PARSES_DDL)
        conn.execute(_IDX_STRUCTURED_PARSES_DDL)
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def write_node(clue_id, node_type, word_indices, word_text,
               role=None, raw_letters=None, answer_positions=None,
               group_id=None, payload_json=None, conn=None):
    """Insert a manual evidence node. Returns the new node id."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_tables(conn)
        letters = (raw_letters or "").upper().strip() or None
        positions_json = (
            json.dumps(list(answer_positions))
            if answer_positions is not None
            else None
        )
        pj = payload_json if payload_json else None
        cursor = conn.execute(
            """INSERT INTO manual_evidence_nodes
               (clue_id, node_type, word_indices, word_text, role,
                raw_letters, answer_positions, group_id, payload_json)
               VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)""",
            (
                clue_id,
                node_type,
                json.dumps(list(word_indices)),
                word_text,
                role or None,
                letters,
                positions_json,
                group_id,
                pj,
            ),
        )
        node_id = cursor.lastrowid
        if own:
            conn.commit()
        return node_id
    finally:
        if own:
            conn.close()


def delete_node(node_id, conn=None):
    """Delete a manual evidence node and any edges that reference it."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_tables(conn)
        conn.execute(
            "DELETE FROM manual_evidence_edges "
            "WHERE from_node_id = ? OR to_node_id = ?",
            (node_id, node_id),
        )
        conn.execute(
            "DELETE FROM manual_evidence_nodes WHERE id = ?",
            (node_id,),
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def write_edge(clue_id, from_node_id, to_node_id, edge_type, conn=None):
    """Insert a manual evidence edge. Returns the new edge id."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_tables(conn)
        cursor = conn.execute(
            """INSERT INTO manual_evidence_edges
               (clue_id, from_node_id, to_node_id, edge_type)
               VALUES (?, ?, ?, ?)""",
            (clue_id, from_node_id, to_node_id, edge_type),
        )
        edge_id = cursor.lastrowid
        if own:
            conn.commit()
        return edge_id
    finally:
        if own:
            conn.close()


def delete_edge(edge_id, conn=None):
    """Delete a manual evidence edge by id."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_tables(conn)
        conn.execute(
            "DELETE FROM manual_evidence_edges WHERE id = ?",
            (edge_id,),
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def get_edges_for_clue(clue_id, conn=None):
    """Return all manual evidence edges for a clue as a list of dicts."""
    own = conn is None
    if own:
        conn = sqlite3.connect(
            f"file:{CLUES_DB}?mode=ro", uri=True, timeout=30
        )
        conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT id, clue_id, from_node_id, to_node_id, edge_type,
                      created_at
               FROM manual_evidence_edges
               WHERE clue_id = ?
               ORDER BY id""",
            (clue_id,),
        ).fetchall()
        return [dict(row) for row in rows]
    except Exception:
        return []
    finally:
        if own:
            conn.close()


def get_nodes_for_clue(clue_id, conn=None):
    """Return all manual evidence nodes for a clue as a list of dicts."""
    own = conn is None
    if own:
        conn = sqlite3.connect(
            f"file:{CLUES_DB}?mode=ro", uri=True, timeout=30
        )
        conn.row_factory = sqlite3.Row
    try:
        col_names = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(manual_evidence_nodes)"
            ).fetchall()
        }
        pj_select = (
            "payload_json"
            if "payload_json" in col_names
            else "NULL AS payload_json"
        )
        rows = conn.execute(
            f"""SELECT id, clue_id, node_type, word_indices, word_text,
                      role, raw_letters, answer_positions, group_id,
                      {pj_select}, source, created_at
               FROM manual_evidence_nodes
               WHERE clue_id = ?
               ORDER BY id""",
            (clue_id,),
        ).fetchall()
        result = []
        for row in rows:
            d = dict(row)
            try:
                d["word_indices"] = json.loads(d["word_indices"] or "[]")
            except (ValueError, TypeError):
                d["word_indices"] = []
            try:
                ap = d.get("answer_positions")
                d["answer_positions"] = json.loads(ap) if ap else None
            except (ValueError, TypeError):
                d["answer_positions"] = None
            try:
                pj = d.get("payload_json")
                decoded = json.loads(pj) if pj else {}
                d["payload_json"] = decoded if isinstance(decoded, dict) else {}
            except (ValueError, TypeError):
                d["payload_json"] = {}
            result.append(d)
        return result
    except Exception:
        return []
    finally:
        if own:
            conn.close()


def get_graph_for_clue(clue_id, conn=None):
    """Return nodes and edges for a clue."""
    return {
        "nodes": get_nodes_for_clue(clue_id, conn=conn),
        "edges": get_edges_for_clue(clue_id, conn=conn),
    }


def _validate_edge_types(from_type, to_type, edge_type):
    """Return True if from_type and to_type are valid for this edge_type."""
    if edge_type == "input_to":
        return from_type in ("source", "transform") and to_type == "operator"
    if edge_type == "output_of":
        return from_type == "transform" and to_type == "operator"
    return False


def _clean_answer(answer):
    """Return only alpha characters from answer, uppercased."""
    return re.sub(r"[^A-Za-z]", "", answer or "").upper()


def _colour_role(node):
    """Return the display role string for a source or transform node."""
    gid = node.get("group_id")
    if gid is not None:
        return "piece_%d" % gid
    return node.get("role") or "synonym"


def _validate_source_letters(node, answer):
    """Return True if raw_letters matches the cleaned answer positions."""
    letters = (node.get("raw_letters") or "").upper()
    positions = node.get("answer_positions") or []
    if not letters or not positions:
        return True
    cleaned = _clean_answer(answer)
    if not cleaned:
        return True
    if len(letters) != len(positions):
        return False
    for i, pos in enumerate(positions):
        if pos < 0 or pos >= len(cleaned):
            return False
        if letters[i] != cleaned[pos]:
            return False
    return True


def _plain_link(pos, letter):
    return {
        "answer_index": pos,
        "letter": letter,
        "source_block": None,
        "source_span": None,
        "source_text": "",
        "source_role": None,
        "source_value": "",
        "source_input_value": "",
        "source_value_index": None,
    }


def _decode_payload(node):
    """Return the decoded payload dict for a node."""
    pj = node.get("payload_json")
    if not pj:
        return {}
    if isinstance(pj, dict):
        return pj
    try:
        result = json.loads(pj)
        return result if isinstance(result, dict) else {}
    except (ValueError, TypeError):
        return {}


def _validate_output_positions(output, positions, cleaned_answer):
    if not positions:
        return True, ""
    if len(positions) != len(output):
        return (
            False,
            "answer_positions length %d does not match output length %d"
            % (len(positions), len(output)),
        )
    for i, pos in enumerate(positions):
        if pos < 0 or pos >= len(cleaned_answer):
            return (
                False,
                "position %d out of range (answer length %d)"
                % (pos, len(cleaned_answer)),
            )
        if output[i] != cleaned_answer[pos]:
            return (
                False,
                "output[%d]=%r does not match answer[%d]=%r"
                % (i, output[i], pos, cleaned_answer[pos]),
            )
    return True, ""


def _evaluate_reversal(op_node, input_nodes, transform_node, cleaned_answer):
    """Return (True, output_letters) or (False, reason_string)."""
    if len(input_nodes) != 1:
        return False, "reversal requires exactly one input, got %d" % len(input_nodes)
    input_letters = (input_nodes[0].get("raw_letters") or "").upper()
    if not input_letters:
        return False, "input source has no raw_letters"
    computed = input_letters[::-1]
    claimed = (transform_node.get("raw_letters") or "").upper()
    if claimed and claimed != computed:
        return (
            False,
            "claimed transform %r does not match reversal of input %r "
            "(expected %r)" % (claimed, input_letters, computed),
        )
    output = claimed or computed
    return _validated_output(output, transform_node, cleaned_answer)


def _evaluate_deletion(op_node, input_nodes, transform_node, cleaned_answer):
    """Return (True, output_letters) or (False, reason_string)."""
    if len(input_nodes) != 1:
        return False, "deletion requires exactly one input, got %d" % len(input_nodes)
    input_letters = (input_nodes[0].get("raw_letters") or "").upper()
    if not input_letters:
        return False, "input source has no raw_letters"
    payload = _decode_payload(transform_node)
    delete_text = (payload.get("delete_text") or "").upper()
    delete_positions = payload.get("delete_positions")
    if delete_text:
        idx = input_letters.find(delete_text)
        if idx == -1:
            return (
                False,
                "delete_text %r not found in input %r"
                % (delete_text, input_letters),
            )
        remainder = input_letters[:idx] + input_letters[idx + len(delete_text):]
    elif delete_positions is not None:
        pos_set = set(delete_positions)
        if any(pos < 0 or pos >= len(input_letters) for pos in pos_set):
            return (
                False,
                "delete_positions contain out-of-range index for input "
                "of length %d" % len(input_letters),
            )
        remainder = "".join(
            ch for i, ch in enumerate(input_letters) if i not in pos_set
        )
    else:
        return (
            False,
            "deletion payload must specify delete_text or delete_positions",
        )
    claimed = (transform_node.get("raw_letters") or "").upper()
    if claimed and claimed != remainder:
        return (
            False,
            "claimed transform %r does not match deletion result %r"
            % (claimed, remainder),
        )
    output = claimed or remainder
    return _validated_output(output, transform_node, cleaned_answer)


def _evaluate_anagram(op_node, input_nodes, transform_node, cleaned_answer):
    """Return (True, output_letters) or (False, reason_string)."""
    if not input_nodes:
        return False, "anagram requires at least one input"
    all_input = "".join((n.get("raw_letters") or "").upper() for n in input_nodes)
    if not all_input:
        return False, "input sources have no raw_letters"
    claimed = (transform_node.get("raw_letters") or "").upper()
    if not claimed:
        return False, "anagram transform must specify raw_letters"
    if sorted(all_input) != sorted(claimed):
        return (
            False,
            "input letters %r cannot be rearranged to produce %r"
            % (all_input, claimed),
        )
    return _validated_output(claimed, transform_node, cleaned_answer)


def _evaluate_container(op_node, input_nodes, transform_node, cleaned_answer):
    """Return (True, output_letters) or (False, reason_string)."""
    claimed = (transform_node.get("raw_letters") or "").upper()
    if not claimed:
        return False, "container transform must specify raw_letters"
    payload = _decode_payload(transform_node)
    outer_id = payload.get("outer_node_id")
    inner_id = payload.get("inner_node_id")
    node_by_id = {n["id"]: n for n in input_nodes}
    if outer_id is not None and inner_id is not None:
        outer_node = node_by_id.get(outer_id)
        inner_node = node_by_id.get(inner_id)
        if outer_node is None or inner_node is None:
            return (
                False,
                "outer_node_id %r or inner_node_id %r not found among inputs"
                % (outer_id, inner_id),
            )
        outer = (outer_node.get("raw_letters") or "").upper()
        inner = (inner_node.get("raw_letters") or "").upper()
    else:
        if len(input_nodes) != 2:
            return (
                False,
                "container requires exactly 2 inputs when outer_node_id/"
                "inner_node_id are not specified; got %d input(s)"
                % len(input_nodes),
            )
        outer = (input_nodes[0].get("raw_letters") or "").upper()
        inner = (input_nodes[1].get("raw_letters") or "").upper()
    if not outer or not inner:
        return False, "outer or inner node has no raw_letters"
    if inner not in claimed:
        return (
            False,
            "inner letters %r not found as substring in claimed output %r"
            % (inner, claimed),
        )
    idx = claimed.index(inner)
    remainder = claimed[:idx] + claimed[idx + len(inner):]
    if remainder != outer:
        return (
            False,
            "removing inner %r from claimed %r leaves %r which does not "
            "match outer %r" % (inner, claimed, remainder, outer),
        )
    return _validated_output(claimed, transform_node, cleaned_answer)


def _validated_output(output, transform_node, cleaned_answer):
    ok, reason = _validate_output_positions(
        output,
        transform_node.get("answer_positions") or [],
        cleaned_answer,
    )
    if not ok:
        return False, reason
    return True, output


def merge_manual_into_display(wfw_display, manual_nodes, answer, manual_edges=None):
    """Merge manual evidence nodes and edges into an existing wfw_display dict."""
    cleaned = _clean_answer(answer)
    edges = manual_edges or []

    node_by_id = {node["id"]: node for node in manual_nodes}
    operator_inputs = {}
    transform_to_op = {}
    source_is_input = set()

    for edge in edges:
        from_id = edge["from_node_id"]
        to_id = edge["to_node_id"]
        edge_type = edge["edge_type"]
        from_node = node_by_id.get(from_id)
        to_node = node_by_id.get(to_id)
        if not from_node or not to_node:
            continue
        if edge_type == "input_to":
            operator_inputs.setdefault(to_id, []).append(from_node)
            source_is_input.add(from_id)
        elif edge_type == "output_of":
            transform_to_op[from_id] = to_id

    transform_eval = {}
    for node in manual_nodes:
        if node.get("node_type") != "transform":
            continue
        node_id = node["id"]
        op_id = transform_to_op.get(node_id)
        if op_id is None:
            transform_eval[node_id] = (
                False, None, "no operator connected to this transform")
            continue
        op_node = node_by_id.get(op_id)
        if op_node is None:
            transform_eval[node_id] = (False, None, "operator node not found")
            continue
        inputs = operator_inputs.get(op_id, [])
        op_role = (op_node.get("role") or "").lower()
        try:
            if op_role == "reversal":
                ok, payload = _evaluate_reversal(
                    op_node, inputs, node, cleaned)
            elif op_role == "deletion":
                ok, payload = _evaluate_deletion(
                    op_node, inputs, node, cleaned)
            elif op_role == "anagram":
                ok, payload = _evaluate_anagram(
                    op_node, inputs, node, cleaned)
            elif op_role == "container":
                ok, payload = _evaluate_container(
                    op_node, inputs, node, cleaned)
            else:
                ok, payload = False, "unsupported operator role: %r" % op_role
        except Exception as exc:
            ok, payload = False, "evaluation error: %s" % exc
        if ok:
            transform_eval[node_id] = (True, payload, "")
        else:
            transform_eval[node_id] = (False, None, payload)

    operator_has_transform = set(transform_to_op.values())
    operator_failed = set()
    for node in manual_nodes:
        if node.get("node_type") != "operator":
            continue
        if node["id"] not in operator_has_transform:
            operator_failed.add(node["id"])
    for transform_id, (ok, _, _) in transform_eval.items():
        if not ok:
            op_id = transform_to_op.get(transform_id)
            if op_id is not None:
                operator_failed.add(op_id)

    covered_word_indices = set()
    for node in manual_nodes:
        for idx in (node.get("word_indices") or []):
            covered_word_indices.add(idx)

    has_manual_def = any(
        node.get("node_type") == "definition" for node in manual_nodes
    )

    new_blocks = []

    for node in manual_nodes:
        if node.get("node_type") != "definition":
            continue
        word_indices = node.get("word_indices") or []
        span = [min(word_indices), max(word_indices) + 1] if word_indices else [0, 1]
        new_blocks.append({
            "block_id": "manual_def_%d" % node["id"],
            "kind": "DEF_BLOCK",
            "role": "definition",
            "text": node["word_text"],
            "value": answer,
            "input_value": "",
            "span": span,
            "token": None,
            "evidence_status": "manual",
            "evidence_reason": None,
        })

    for node in manual_nodes:
        if node.get("node_type") != "source":
            continue
        word_indices = node.get("word_indices") or []
        span = [min(word_indices), max(word_indices) + 1] if word_indices else [0, 1]
        valid = _validate_source_letters(node, answer)
        role = _colour_role(node) if valid else "source_review"
        status = "manual" if valid else "failed"
        new_blocks.append({
            "block_id": "manual_src_%d" % node["id"],
            "kind": "SOURCE_BLOCK",
            "role": role,
            "text": node["word_text"],
            "value": node.get("raw_letters") or "",
            "input_value": "",
            "span": span,
            "token": None,
            "evidence_status": status,
            "evidence_reason": None if valid else "letters do not match answer",
        })

    for node in manual_nodes:
        if node.get("node_type") != "operator":
            continue
        word_indices = node.get("word_indices") or []
        span = [min(word_indices), max(word_indices) + 1] if word_indices else [0, 1]
        node_id = node["id"]
        failed = node_id in operator_failed
        op_role = (node.get("role") or "").lower()
        block_role = (op_role + "_indicator") if op_role else "op_indicator"
        new_blocks.append({
            "block_id": "manual_op_%d" % node_id,
            "kind": "OP_BLOCK",
            "role": block_role,
            "text": node["word_text"],
            "value": op_role,
            "input_value": "",
            "span": span,
            "token": None,
            "evidence_status": "failed" if failed else "manual",
            "evidence_reason": (
                None if not failed
                else "operation failed or has no connected transform"
            ),
        })

    for node in manual_nodes:
        if node.get("node_type") != "transform":
            continue
        node_id = node["id"]
        ok, output, reason = transform_eval.get(
            node_id, (False, None, "not evaluated"))
        op_id = transform_to_op.get(node_id)
        op_node = node_by_id.get(op_id) if op_id is not None else None
        if op_node:
            word_indices = op_node.get("word_indices") or []
            span = [min(word_indices), max(word_indices) + 1] if word_indices else [0, 1]
        else:
            word_indices = node.get("word_indices") or []
            span = [min(word_indices), max(word_indices) + 1] if word_indices else [0, 1]
        role = _colour_role(node) if ok else "source_review"
        new_blocks.append({
            "block_id": "manual_tr_%d" % node_id,
            "kind": "SOURCE_BLOCK",
            "role": role,
            "text": node["word_text"],
            "value": output if ok else (node.get("raw_letters") or ""),
            "input_value": "",
            "span": span,
            "token": None,
            "evidence_status": "manual" if ok else "failed",
            "evidence_reason": None if ok else reason,
        })

    for block in (wfw_display.get("blocks") or []):
        kind = block.get("kind")
        span = block.get("span") or []
        span_indices = (
            set(range(span[0], span[1]))
            if isinstance(span, list) and len(span) == 2
            else set()
        )
        if kind == "DEF_BLOCK" and has_manual_def:
            continue
        if kind == "SOURCE_BLOCK" and span_indices & covered_word_indices:
            continue
        if kind == "REVIEW_BLOCK" and span_indices & covered_word_indices:
            continue
        new_blocks.append(block)

    wfw_display["blocks"] = new_blocks

    manual_by_pos = {}
    conflicted = set()

    def _try_claim(pos, entry):
        if pos in manual_by_pos:
            conflicted.add(pos)
        else:
            manual_by_pos[pos] = entry

    for node in manual_nodes:
        if node.get("node_type") != "transform":
            continue
        node_id = node["id"]
        ok, output, _ = transform_eval.get(node_id, (False, None, ""))
        if not ok:
            continue
        positions = node.get("answer_positions") or []
        if not positions:
            continue
        role = _colour_role(node)
        op_id = transform_to_op.get(node_id)
        op_node = node_by_id.get(op_id) if op_id is not None else None
        if op_node:
            word_indices = op_node.get("word_indices") or []
            span = [min(word_indices), max(word_indices) + 1] if word_indices else None
        else:
            span = None
        for pos in positions:
            if pos < 0 or pos >= len(cleaned):
                continue
            _try_claim(pos, {
                "answer_index": pos,
                "letter": cleaned[pos],
                "source_block": "manual_tr_%d" % node_id,
                "source_span": span,
                "source_text": node["word_text"],
                "source_role": role,
                "source_value": output or "",
                "source_input_value": "",
                "source_value_index": None,
            })

    for node in manual_nodes:
        if node.get("node_type") != "source":
            continue
        if node["id"] in source_is_input:
            continue
        if not _validate_source_letters(node, answer):
            continue
        positions = node.get("answer_positions") or []
        if not positions:
            continue
        word_indices = node.get("word_indices") or []
        span = [min(word_indices), max(word_indices) + 1] if word_indices else [0, 1]
        role = _colour_role(node)
        for pos in positions:
            if pos < 0 or pos >= len(cleaned):
                continue
            _try_claim(pos, {
                "answer_index": pos,
                "letter": cleaned[pos],
                "source_block": "manual_src_%d" % node["id"],
                "source_span": span,
                "source_text": node["word_text"],
                "source_role": role,
                "source_value": node.get("raw_letters") or "",
                "source_input_value": "",
                "source_value_index": None,
            })

    for pos in conflicted:
        manual_by_pos[pos] = _plain_link(pos, cleaned[pos])

    auto_by_pos = {}
    for link in (wfw_display.get("answer_links") or []):
        pos = link.get("answer_index")
        if pos is None or pos < 0 or pos >= len(cleaned):
            continue
        if pos not in manual_by_pos and pos not in auto_by_pos:
            auto_by_pos[pos] = link

    final_links = []
    for pos, char in enumerate(cleaned):
        if pos in manual_by_pos:
            final_links.append(manual_by_pos[pos])
        elif pos in auto_by_pos:
            final_links.append(auto_by_pos[pos])
        else:
            final_links.append(_plain_link(pos, char))
    wfw_display["answer_links"] = final_links


def write_structured_parse(clue_id, parse_dict, source="human",
                           status="draft", conn=None):
    """Save or replace the structured parse for a clue. Returns the row id."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_tables(conn)
        parse_json = json.dumps(parse_dict, ensure_ascii=False)
        existing = conn.execute(
            "SELECT id FROM manual_structured_parses "
            "WHERE clue_id = ? ORDER BY id DESC LIMIT 1",
            (clue_id,),
        ).fetchone()
        if existing:
            conn.execute(
                """UPDATE manual_structured_parses
                   SET parse_json = ?, source = ?, status = ?,
                       updated_at = datetime('now')
                   WHERE id = ?""",
                (parse_json, source, status, existing[0]),
            )
            row_id = existing[0]
        else:
            cursor = conn.execute(
                """INSERT INTO manual_structured_parses
                   (clue_id, parse_json, source, status)
                   VALUES (?, ?, ?, ?)""",
                (clue_id, parse_json, source, status),
            )
            row_id = cursor.lastrowid
        if own:
            conn.commit()
        return row_id
    finally:
        if own:
            conn.close()


def get_structured_parse(clue_id, conn=None):
    """Return the structured parse dict for a clue, or None if none exists."""
    own = conn is None
    if own:
        conn = sqlite3.connect(
            "file:%s?mode=ro" % CLUES_DB, uri=True, timeout=30
        )
    try:
        col_names = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(manual_structured_parses)"
            ).fetchall()
        }
        if not col_names:
            return None
        row = conn.execute(
            "SELECT parse_json FROM manual_structured_parses "
            "WHERE clue_id = ? ORDER BY id DESC LIMIT 1",
            (clue_id,),
        ).fetchone()
        if row is None:
            return None
        return json.loads(row[0])
    except Exception:
        return None
    finally:
        if own:
            conn.close()


def delete_structured_parse(clue_id, conn=None):
    """Delete all structured parses for a clue."""
    own = conn is None
    if own:
        conn = sqlite3.connect(str(CLUES_DB), timeout=30)
    try:
        ensure_tables(conn)
        conn.execute(
            "DELETE FROM manual_structured_parses WHERE clue_id = ?",
            (clue_id,),
        )
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()


def _container_can_make_result(outer_letters, inner_letters, result):
    """Return True if removing inner at any occurrence leaves outer."""
    if not outer_letters or not inner_letters or not result:
        return False
    search_start = 0
    while True:
        idx = result.find(inner_letters, search_start)
        if idx == -1:
            return False
        remainder = result[:idx] + result[idx + len(inner_letters):]
        if remainder == outer_letters:
            return True
        search_start = idx + 1


def _validate_structured_parse(parse_dict, answer):
    """Validate a structured parse dict against a clue answer.

    Answer boxes in parse_dict are 1-based.
    """
    errors = []
    cleaned = _clean_answer(answer)
    if not cleaned:
        errors.append("answer is empty after cleaning")
        return errors

    defn = parse_dict.get("definition") or {}
    if not defn:
        errors.append("missing definition block")
    elif not defn.get("clue_text"):
        errors.append("definition has no clue_text")

    pieces = parse_dict.get("pieces") or []
    operations = parse_dict.get("operations") or []
    relationship_set = (
        _OPERATION_STRUCTURED_RELATIONSHIPS
        if operations
        else _SOURCE_STRUCTURED_RELATIONSHIPS
    )
    piece_ids = set()
    colours_seen = {}
    box_to_piece = {}

    for piece in pieces:
        pid = piece.get("id")
        if not pid:
            errors.append("piece is missing id field")
            continue
        if pid in piece_ids:
            errors.append("duplicate piece id: %s" % pid)
        piece_ids.add(pid)

        letters = (piece.get("letters") or "").upper()
        boxes = piece.get("answer_boxes") or []
        relationship = (piece.get("relationship") or "").lower()
        mapping = piece.get("mapping", "positional")
        colour = (piece.get("colour") or "").lower()
        clue_text = (piece.get("clue_text") or "").strip()
        row_fields = {
            "clue words": clue_text,
            "relationship": relationship,
            "letters": letters,
            "answer boxes": boxes,
            "colour": colour,
        }
        if any(row_fields.values()) and not all(row_fields.values()):
            missing = [
                label for label, value in row_fields.items() if not value
            ]
            errors.append(
                "piece %s: partially filled row missing %s"
                % (pid, ", ".join(missing))
            )

        if relationship and relationship not in relationship_set:
            errors.append(
                "piece %s: unknown relationship %r (valid: %s)"
                % (pid, relationship, ", ".join(sorted(relationship_set)))
            )

        if colour and colour not in _COLOUR_TO_ROLE:
            errors.append(
                "piece %s: unknown colour %r (valid: %s)"
                % (pid, colour, ", ".join(sorted(_COLOUR_TO_ROLE)))
            )
        elif colour:
            if colour in colours_seen:
                errors.append(
                    "colour %r used by both piece %s and piece %s"
                    % (colour, colours_seen[colour], pid)
                )
            else:
                colours_seen[colour] = pid

        if mapping == "positional":
            if letters and boxes:
                if len(letters) != len(boxes):
                    errors.append(
                        "piece %s: %d boxes but %d letters"
                        % (pid, len(boxes), len(letters))
                    )
                else:
                    for i, box in enumerate(boxes):
                        idx = box - 1
                        if idx < 0 or idx >= len(cleaned):
                            errors.append(
                                "piece %s: box %d out of range "
                                "(answer length %d)" % (pid, box, len(cleaned))
                            )
                        elif letters[i] != cleaned[idx]:
                            errors.append(
                                "piece %s: letter %r at box %d does not match "
                                "answer letter %r"
                                % (pid, letters[i], box, cleaned[idx])
                            )

        for box in boxes:
            if box in box_to_piece:
                errors.append(
                    "box %d claimed by both piece %s and piece %s"
                    % (box, box_to_piece[box], pid)
                )
            else:
                box_to_piece[box] = pid

    if not operations:
        claimed = set(box_to_piece)
        expected = set(range(1, len(cleaned) + 1))
        missing = sorted(expected - claimed)
        extra = sorted(box for box in claimed if box not in expected)
        if missing:
            errors.append(
                "charade parse does not cover answer boxes: %s"
                % ",".join(str(box) for box in missing)
            )
        if extra:
            errors.append(
                "charade parse claims boxes outside answer: %s"
                % ",".join(str(box) for box in extra)
            )
    for op in operations:
        oid = op.get("id", "?")
        op_type = (op.get("type") or "").lower()
        op_colour = (op.get("colour") or "").lower()
        if op_colour and op_colour not in _COLOUR_TO_ROLE:
            errors.append(
                "operation %s: unknown colour %r (valid: %s)"
                % (oid, op_colour, ", ".join(sorted(_COLOUR_TO_ROLE)))
            )

        if op_type == "container":
            outer_id = op.get("outer_piece_id")
            inner_id = op.get("inner_piece_id")

            if not outer_id or outer_id not in piece_ids:
                errors.append(
                    "operation %s: outer_piece_id %r not in pieces"
                    % (oid, outer_id)
                )
            if not inner_id or inner_id not in piece_ids:
                errors.append(
                    "operation %s: inner_piece_id %r not in pieces"
                    % (oid, inner_id)
                )

            if outer_id in piece_ids and inner_id in piece_ids:
                outer_piece = next(p for p in pieces if p["id"] == outer_id)
                inner_piece = next(p for p in pieces if p["id"] == inner_id)
                outer_letters = (outer_piece.get("letters") or "").upper()
                inner_letters = (inner_piece.get("letters") or "").upper()
                result = (op.get("result") or "").upper()

                if result:
                    if inner_letters and inner_letters not in result:
                        errors.append(
                            "operation %s: inner %r not found as substring "
                            "in result %r" % (oid, inner_letters, result)
                        )
                    elif inner_letters and not _container_can_make_result(
                        outer_letters, inner_letters, result
                    ):
                        errors.append(
                            "operation %s: no valid position exists where "
                            "removing inner %r from result %r leaves outer %r"
                            % (oid, inner_letters, result, outer_letters)
                        )
                    if result != cleaned:
                        errors.append(
                            "operation %s: result %r does not match cleaned "
                            "answer %r" % (oid, result, cleaned)
                        )

        elif op_type == "reversal":
            input_id = op.get("input_piece_id")
            fodder = (op.get("fodder") or "").upper()
            result = (op.get("result") or "").upper()

            if not input_id or input_id not in piece_ids:
                errors.append(
                    "operation %s: input_piece_id %r not in pieces"
                    % (oid, input_id)
                )
            else:
                input_piece = next(p for p in pieces if p["id"] == input_id)
                piece_letters = (input_piece.get("letters") or "").upper()

                input_boxes = set(input_piece.get("answer_boxes") or [])
                expected_boxes = set(range(1, len(cleaned) + 1))
                if input_boxes != expected_boxes:
                    errors.append(
                        "operation %s: piece %s boxes %s must cover all "
                        "answer boxes (1..%d)"
                        % (oid, input_id, sorted(input_boxes), len(cleaned))
                    )

                if not fodder:
                    errors.append(
                        "operation %s: fodder is required for reversal" % oid
                    )
                elif not result:
                    errors.append(
                        "operation %s: result is required for reversal" % oid
                    )
                else:
                    if result != fodder[::-1]:
                        errors.append(
                            "operation %s: result %r is not the reverse of "
                            "fodder %r" % (oid, result, fodder)
                        )
                    if result != piece_letters:
                        errors.append(
                            "operation %s: result %r does not match piece %s "
                            "letters %r" % (oid, result, input_id, piece_letters)
                        )

        elif op_type == "anagram":
            input_id = op.get("input_piece_id")
            fodder = (op.get("fodder") or "").upper()
            result = (op.get("result") or "").upper()

            if not input_id or input_id not in piece_ids:
                errors.append(
                    "operation %s: input_piece_id %r not in pieces"
                    % (oid, input_id)
                )
            else:
                input_piece = next(p for p in pieces if p["id"] == input_id)
                piece_letters = (input_piece.get("letters") or "").upper()

                input_boxes = set(input_piece.get("answer_boxes") or [])
                expected_boxes = set(range(1, len(cleaned) + 1))
                if input_boxes != expected_boxes:
                    errors.append(
                        "operation %s: piece %s boxes %s must cover all "
                        "answer boxes (1..%d)"
                        % (oid, input_id, sorted(input_boxes), len(cleaned))
                    )

                if not fodder:
                    errors.append(
                        "operation %s: fodder is required for anagram" % oid
                    )
                elif not result:
                    errors.append(
                        "operation %s: result is required for anagram" % oid
                    )
                else:
                    if sorted(fodder) != sorted(result) or len(fodder) != len(result):
                        errors.append(
                            "operation %s: result %r is not an anagram of "
                            "fodder %r" % (oid, result, fodder)
                        )
                    if result != piece_letters:
                        errors.append(
                            "operation %s: result %r does not match piece %s "
                            "letters %r" % (oid, result, input_id, piece_letters)
                        )

    return errors


def _find_word_positions(phrase, clue_text):
    """Return 0-based word indices of phrase in clue_text.

    This intentionally simple matcher is enough for the first slice; click-to-fill
    clue spans can replace it later.
    """
    if not phrase or not clue_text:
        return []
    sep = re.compile(r"[\s,;:]+")
    clue_words = [w for w in sep.split(clue_text.strip()) if w]
    phrase_words = [w for w in sep.split(phrase.strip()) if w]
    if not phrase_words:
        return []
    n = len(phrase_words)
    phrase_lower = [w.lower() for w in phrase_words]
    for i in range(len(clue_words) - n + 1):
        if [w.lower() for w in clue_words[i:i + n]] == phrase_lower:
            return list(range(i, i + n))
    return []


def merge_structured_parse_into_display(wfw_display, parse_dict, answer):
    """Merge a structured parse into an existing wfw_display dict.

    Pieces own answer tile colours. Operations validate combinations but do not
    claim answer tiles.
    """
    cleaned = _clean_answer(answer)

    pieces_by_id = {p["id"]: p for p in (parse_dict.get("pieces") or [])}
    op_results = {}

    for op in (parse_dict.get("operations") or []):
        oid = op.get("id", "?")
        op_type = (op.get("type") or "").lower()

        if op_type == "container":
            outer = pieces_by_id.get(op.get("outer_piece_id"))
            inner = pieces_by_id.get(op.get("inner_piece_id"))
            if not outer or not inner:
                op_results[oid] = (False, "piece not found")
                continue
            outer_letters = (outer.get("letters") or "").upper()
            inner_letters = (inner.get("letters") or "").upper()
            result = (op.get("result") or "").upper()
            if not result:
                op_results[oid] = (False, "no result specified")
            elif inner_letters not in result:
                op_results[oid] = (
                    False,
                    "inner %r not in result %r" % (inner_letters, result),
                )
            elif not _container_can_make_result(
                outer_letters, inner_letters, result
            ):
                op_results[oid] = (
                    False,
                    "no insertion leaves outer %r" % outer_letters,
                )
            elif result != cleaned:
                op_results[oid] = (
                    False,
                    "result %r does not match answer %r" % (result, cleaned),
                )
            else:
                op_results[oid] = (True, "")
        else:
            op_results[oid] = (True, "")

    new_blocks = []
    covered_word_positions = set()

    defn = parse_dict.get("definition") or {}
    if defn.get("clue_text"):
        positions = defn.get("clue_word_positions") or []
        covered_word_positions.update(positions)
        span = [min(positions), max(positions) + 1] if positions else [0, 1]
        new_blocks.append({
            "block_id": "sp_def_%s" % defn.get("id", "def"),
            "kind": "DEF_BLOCK",
            "role": "definition",
            "text": defn["clue_text"],
            "value": answer,
            "input_value": "",
            "span": span,
            "token": None,
            "evidence_status": "manual",
            "evidence_reason": None,
        })

    for piece in (parse_dict.get("pieces") or []):
        positions = piece.get("clue_word_positions") or []
        covered_word_positions.update(positions)
        span = [min(positions), max(positions) + 1] if positions else [0, 1]
        letters = (piece.get("letters") or "").upper()
        boxes = piece.get("answer_boxes") or []
        mapping = piece.get("mapping", "positional")
        colour = (piece.get("colour") or "").lower()
        role = _COLOUR_TO_ROLE.get(colour, "synonym")

        valid = True
        reason = None
        if mapping == "positional" and letters and boxes:
            if len(letters) != len(boxes):
                valid = False
                reason = "box count %d != letter count %d" % (
                    len(boxes), len(letters)
                )
            else:
                for i, box in enumerate(boxes):
                    idx = box - 1
                    if idx < 0 or idx >= len(cleaned):
                        valid = False
                        reason = "box %d out of range" % box
                        break
                    if letters[i] != cleaned[idx]:
                        valid = False
                        reason = "letter %r at box %d != answer %r" % (
                            letters[i], box, cleaned[idx]
                        )
                        break

        new_blocks.append({
            "block_id": "sp_piece_%s" % piece["id"],
            "kind": "SOURCE_BLOCK",
            "role": role if valid else "source_review",
            "text": piece.get("clue_text", ""),
            "value": letters,
            "input_value": "",
            "span": span,
            "token": None,
            "evidence_status": "manual" if valid else "failed",
            "evidence_reason": reason,
        })

    for op in (parse_dict.get("operations") or []):
        positions = op.get("clue_word_positions") or []
        covered_word_positions.update(positions)
        span = [min(positions), max(positions) + 1] if positions else [0, 1]
        op_type = (op.get("type") or "").lower()
        ok, reason = op_results.get(op.get("id", "?"), (False, "not evaluated"))
        colour = (op.get("colour") or "").lower()
        colour_role = _COLOUR_TO_ROLE.get(colour)
        if colour_role:
            block_role = "%s_positional_indicator" % colour_role
        else:
            block_role = (op_type + "_indicator") if op_type else "op_indicator"
        new_blocks.append({
            "block_id": "sp_op_%s" % op.get("id", "op"),
            "kind": "OP_BLOCK",
            "role": block_role,
            "text": op.get("clue_text", ""),
            "value": op_type,
            "input_value": "",
            "span": span,
            "token": None,
            "evidence_status": "manual" if ok else "failed",
            "evidence_reason": None if ok else reason,
        })

    for filler in (parse_dict.get("filler") or []):
        positions = filler.get("clue_word_positions") or []
        covered_word_positions.update(positions)
        span = [min(positions), max(positions) + 1] if positions else [0, 1]
        new_blocks.append({
            "block_id": "sp_filler_%s" % filler.get("id", "f"),
            "kind": "SOURCE_BLOCK",
            "role": "structural",
            "text": filler.get("clue_text", ""),
            "value": "",
            "input_value": "",
            "span": span,
            "token": None,
            "evidence_status": "manual",
            "evidence_reason": None,
        })

    has_manual_def = bool(defn.get("clue_text"))
    for block in (wfw_display.get("blocks") or []):
        kind = block.get("kind")
        span = block.get("span") or []
        span_indices = (
            set(range(span[0], span[1]))
            if isinstance(span, list) and len(span) == 2
            else set()
        )
        if kind == "DEF_BLOCK" and has_manual_def:
            continue
        if kind in ("SOURCE_BLOCK", "REVIEW_BLOCK") and (
            span_indices & covered_word_positions
        ):
            continue
        new_blocks.append(block)

    wfw_display["blocks"] = new_blocks

    manual_by_pos = {}
    conflicted = set()

    def _try_claim(pos, entry):
        if pos in manual_by_pos:
            conflicted.add(pos)
        else:
            manual_by_pos[pos] = entry

    for piece in (parse_dict.get("pieces") or []):
        boxes = piece.get("answer_boxes") or []
        letters = (piece.get("letters") or "").upper()
        mapping = piece.get("mapping", "positional")
        colour = (piece.get("colour") or "").lower()
        role = _COLOUR_TO_ROLE.get(colour, "synonym")
        positions = piece.get("clue_word_positions") or []
        span = [min(positions), max(positions) + 1] if positions else None

        if mapping == "positional":
            if not letters or not boxes or len(letters) != len(boxes):
                continue
            for i, box in enumerate(boxes):
                idx = box - 1
                if idx < 0 or idx >= len(cleaned):
                    continue
                if letters[i] != cleaned[idx]:
                    continue
                _try_claim(idx, {
                    "answer_index": idx,
                    "letter": cleaned[idx],
                    "source_block": "sp_piece_%s" % piece["id"],
                    "source_span": span,
                    "source_text": piece.get("clue_text", ""),
                    "source_role": role,
                    "source_value": letters,
                    "source_input_value": "",
                    "source_value_index": None,
                })
        elif mapping == "block":
            for box in boxes:
                idx = box - 1
                if idx < 0 or idx >= len(cleaned):
                    continue
                _try_claim(idx, {
                    "answer_index": idx,
                    "letter": cleaned[idx],
                    "source_block": "sp_piece_%s" % piece["id"],
                    "source_span": span,
                    "source_text": piece.get("clue_text", ""),
                    "source_role": role,
                    "source_value": letters,
                    "source_input_value": "",
                    "source_value_index": None,
                })

    for pos in conflicted:
        manual_by_pos[pos] = _plain_link(pos, cleaned[pos])

    auto_by_pos = {}
    for link in (wfw_display.get("answer_links") or []):
        pos = link.get("answer_index")
        if pos is None or pos < 0 or pos >= len(cleaned):
            continue
        if pos not in manual_by_pos and pos not in auto_by_pos:
            auto_by_pos[pos] = link

    final_links = []
    for pos, char in enumerate(cleaned):
        if pos in manual_by_pos:
            final_links.append(manual_by_pos[pos])
        elif pos in auto_by_pos:
            final_links.append(auto_by_pos[pos])
        else:
            final_links.append(_plain_link(pos, char))
    wfw_display["answer_links"] = final_links
