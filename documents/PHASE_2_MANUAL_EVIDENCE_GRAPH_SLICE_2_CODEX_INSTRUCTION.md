# Phase 2 Manual Evidence Graph — Slice 2 Codex Instruction

Date: 2026-05-26
Status: revision 3 — ready for implementation

---

## Task

Extend the manual evidence graph (built in Slice 1) to support cryptic operations:
reversal, deletion, anagram, and container. After this slice, an admin can record a
complete parse graph for a clue — source nodes, operator nodes, transform nodes, and
edges connecting them — and the display merge will show coloured blocks and answer
tiles that respect the graph structure.

This remains admin-only, display-time-only. No automatic proof promotion.

---

## Non-goals

- Do not rewrite the automatic solver.
- Do not promote manual evidence into wfw_proof_attempts.
- Do not make manual evidence public.
- Do not remove or break Slice 1 direct source placement.
- Do not make rerun delete manual evidence.
- Do not depend on Flask or a browser for verification.

---

## Answers to the eight spec questions

**1. Is payload_json needed?**
Yes. Deletion needs delete_text or delete_positions. Container needs outer_node_id and
inner_node_id. Overloading raw_letters would be brittle. Add payload_json TEXT NULL to
manual_evidence_nodes using ALTER TABLE guarded against duplicate-column errors.

**2. Exact node/edge shapes**

Source (unchanged):
  node_type="source", word_indices, word_text, role (synonym/abbreviation/etc),
  raw_letters (input letters to operator, or direct letters), answer_positions (if
  standalone placement), group_id, payload_json=null

Operator:
  node_type="operator", word_indices (clue word positions), word_text (indicator text,
  e.g. "BACKS"), role="reversal"|"deletion"|"anagram"|"container", raw_letters=null,
  answer_positions=null, group_id=null, payload_json=null

Transform:
  node_type="transform", word_indices=[] (no clue words of its own), word_text
  (admin description, e.g. "MOT (reversal of TOM)"), role=null, raw_letters (output
  letters, e.g. "MOT"), answer_positions (where output sits in cleaned answer, e.g.
  [0,1,2]), group_id (same as input source for colour continuity), payload_json
  (deletion or container specifics if needed)

Edge types:
  input_to: (source_id or transform_id) → (operator_id). Source feeds the operator.
  output_of: (transform_id) → (operator_id). Transform is the result of the operator.

The edge vocabulary is minimal but unambiguous: follow output_of to find a transform's
operator; collect input_to edges on that operator to find its inputs.

**3. How merge decides whether a transform is valid**

For each transform node, merge:
  a. Follows output_of edge to find its operator.
  b. Collects all input nodes via input_to edges on that operator.
  c. Calls the appropriate evaluator (_evaluate_reversal etc.) based on operator.role.
  d. Evaluator computes the expected output from the input letters and validates it
     against transform.raw_letters (if set) and transform.answer_positions (if set).
  e. Returns (True, output_letters) on success or (False, reason_string) on failure.

**4. How failed transforms are displayed**

  - The operator's OP_BLOCK gets evidence_status="failed".
  - The transform's SOURCE_BLOCK gets evidence_status="failed" and role="source_review"
    (rose/red tile colour in atomic_parse.html).
  - The input source blocks are unaffected; they still show as evidence_status="manual".
  - Answer positions claimed by the failed transform receive plain entries
    (source_role=None, white tile).

**5. How colour flows from source through transform to answer tiles**

  The admin sets the same group_id on the transform node as on the input source node.
  Merge colours the source block with piece_{group_id} from the source node.
  Merge colours the transform block and its answer tiles with piece_{group_id} from the
  transform node. When both carry the same group_id (e.g. 0), the block and tiles are
  both piece_0 (blue), giving colour continuity through the operation.

**6. How duplicate or conflicting answer claims are handled**

  The authoritative position map from Slice 1 is preserved unchanged. Transform nodes
  and standalone source nodes all add to the same map. If two nodes claim the same
  answer position, that position is conflicted and receives a plain entry. The Slice 1
  invariant holds: exactly one tile per cleaned-answer position, no duplicates possible.

**7. What exact files change**

  - signature_solver/manual_evidence_store.py (modify existing)
  - web/routes/admin.py (extend existing)
  - web/routes/clue.py (modify two lines)
  - web/templates/clue.html (extend form)
  - web/templates/partials/manual_evidence_nodes.html (extend to show edges)

No new files.

**8. What verification commands prove the slice**

Nine checks are given at the end of this document. All run without Flask or live DB.

---

## Files that change

Five files:

1. signature_solver/manual_evidence_store.py — schema migration, edge helpers,
   evaluators, updated merge
2. web/routes/admin.py — extended node route, two new edge routes
3. web/routes/clue.py — switch to get_graph_for_clue, pass edges to merge
4. web/templates/clue.html — payload_json textarea, edge creation form
5. web/templates/partials/manual_evidence_nodes.html — edge list, prominent ids

---

## Change 1: signature_solver/manual_evidence_store.py

Make all the modifications described below. The order matters: later sections depend
on earlier ones.

### 1a. payload_json schema migration

Add this function immediately before ensure_tables:

```python
def _add_payload_json_column(conn):
    """Add payload_json column to manual_evidence_nodes if not already present.

    Called from ensure_tables. Safe to call repeatedly: silently ignores
    the OperationalError that SQLite raises when the column already exists.
    """
    try:
        conn.execute(
            "ALTER TABLE manual_evidence_nodes ADD COLUMN payload_json TEXT"
        )
    except sqlite3.OperationalError as exc:
        if "duplicate column" not in str(exc).lower():
            raise
```

Inside ensure_tables, immediately after the four conn.execute calls and before
the `if own: conn.commit()` line, insert one new call:

```python
        _add_payload_json_column(conn)
```

The full ensure_tables body becomes:

```python
    try:
        conn.execute(_NODES_DDL)
        conn.execute(_EDGES_DDL)
        conn.execute(_IDX_NODES_DDL)
        conn.execute(_IDX_EDGES_DDL)
        _add_payload_json_column(conn)
        if own:
            conn.commit()
    finally:
        if own:
            conn.close()
```

### 1b. Updated write_node

Replace the entire write_node function with this version, which accepts payload_json
and includes it in the INSERT:

```python
def write_node(clue_id, node_type, word_indices, word_text,
               role=None, raw_letters=None, answer_positions=None,
               group_id=None, payload_json=None, conn=None):
    """Insert a manual evidence node. Returns the new node id (int)."""
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
        # payload_json is stored as-is (already a JSON string) or None.
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
```

### 1c. Updated get_nodes_for_clue

Replace the get_nodes_for_clue function with this version, which decodes payload_json:

```python
def get_nodes_for_clue(clue_id, conn=None):
    """Return all manual evidence nodes for a clue as a list of dicts.

    word_indices is decoded from JSON to a Python list of ints.
    answer_positions is decoded from JSON to a Python list of ints, or None.
    payload_json is decoded from JSON to a Python dict, or {} on any error.

    Returns [] if the table does not exist yet or on any read error.
    """
    own = conn is None
    if own:
        conn = sqlite3.connect(
            f"file:{CLUES_DB}?mode=ro", uri=True, timeout=30
        )
        conn.row_factory = sqlite3.Row
    try:
        # Detect whether payload_json column exists. If the DB was created by
        # Slice 1 (before ALTER TABLE), selecting payload_json raises an
        # OperationalError and get_nodes_for_clue would silently return [],
        # hiding all existing manual evidence. PRAGMA is safe to call on a
        # read-only connection and costs one row-fetch before the main query.
        col_names = {
            row[1]
            for row in conn.execute(
                "PRAGMA table_info(manual_evidence_nodes)"
            ).fetchall()
        }
        pj_select = "payload_json" if "payload_json" in col_names else "NULL AS payload_json"
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
```

### 1d. New edge helpers

Insert these four functions immediately after delete_node and before the
display merge helpers section:

```python
def write_edge(clue_id, from_node_id, to_node_id, edge_type, conn=None):
    """Insert a manual evidence edge. Returns the new edge id (int)."""
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
    """Return all manual evidence edges for a clue as a list of dicts.

    Returns [] on any error including missing table.
    """
    own = conn is None
    if own:
        conn = sqlite3.connect(
            f"file:{CLUES_DB}?mode=ro", uri=True, timeout=30
        )
        conn.row_factory = sqlite3.Row
    try:
        rows = conn.execute(
            """SELECT id, clue_id, from_node_id, to_node_id, edge_type, created_at
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


def get_graph_for_clue(clue_id, conn=None):
    """Return nodes and edges for a clue as {"nodes": [...], "edges": [...]}."""
    return {
        "nodes": get_nodes_for_clue(clue_id, conn=conn),
        "edges": get_edges_for_clue(clue_id, conn=conn),
    }


def _validate_edge_types(from_type, to_type, edge_type):
    """Return True if from_type and to_type are valid for this edge_type.

    Rules enforced at the admin route layer:
      input_to: from must be 'source' or 'transform'; to must be 'operator'
      output_of: from must be 'transform'; to must be 'operator'

    Exported so the admin route can call it and so Check 6 can test it
    without a running Flask server.
    """
    if edge_type == "input_to":
        return from_type in ("source", "transform") and to_type == "operator"
    if edge_type == "output_of":
        return from_type == "transform" and to_type == "operator"
    return False
```

### 1e. Operation evaluators

Insert these functions immediately before merge_manual_into_display.
They are pure functions with no Flask or DB dependencies.

```python
# ---------------------------------------------------------------------------
# Operation evaluators
# ---------------------------------------------------------------------------

def _decode_payload(node):
    """Return the decoded payload dict for a node, or {} if absent or invalid."""
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


def _evaluate_reversal(op_node, input_nodes, transform_node, cleaned_answer):
    """Return (True, output_letters) or (False, reason_string).

    Reversal requires exactly one input. Computes the reverse of input.raw_letters.
    If transform.raw_letters is set and does not match the computed reverse, fails.
    If transform.answer_positions are set, validates each position against the answer.
    """
    if len(input_nodes) != 1:
        return False, "reversal requires exactly one input, got %d" % len(input_nodes)
    input_letters = (input_nodes[0].get("raw_letters") or "").upper()
    if not input_letters:
        return False, "input source has no raw_letters"
    computed = input_letters[::-1]
    claimed = (transform_node.get("raw_letters") or "").upper()
    if claimed and claimed != computed:
        return False, (
            "claimed transform %r does not match reversal of input %r (expected %r)"
            % (claimed, input_letters, computed)
        )
    output = claimed or computed
    positions = transform_node.get("answer_positions") or []
    if positions:
        if len(positions) != len(output):
            return False, (
                "answer_positions length %d does not match output length %d"
                % (len(positions), len(output))
            )
        for i, p in enumerate(positions):
            if p < 0 or p >= len(cleaned_answer):
                return False, "position %d out of range (answer length %d)" % (p, len(cleaned_answer))
            if output[i] != cleaned_answer[p]:
                return False, (
                    "output[%d]=%r does not match answer[%d]=%r"
                    % (i, output[i], p, cleaned_answer[p])
                )
    return True, output


def _evaluate_deletion(op_node, input_nodes, transform_node, cleaned_answer):
    """Return (True, output_letters) or (False, reason_string).

    Deletion requires exactly one input. The payload must specify either:
      delete_text: exact letter string to remove (first occurrence), or
      delete_positions: list of zero-based positions in the input to remove.
    If transform.raw_letters is set, the computed remainder must match it.
    If transform.answer_positions are set, validates each position against the answer.
    """
    if len(input_nodes) != 1:
        return False, "deletion requires exactly one input, got %d" % len(input_nodes)
    input_letters = (input_nodes[0].get("raw_letters") or "").upper()
    if not input_letters:
        return False, "input source has no raw_letters"
    payload = _decode_payload(transform_node)
    delete_text = (payload.get("delete_text") or "").upper()
    delete_positions = payload.get("delete_positions")  # list of ints or None
    if delete_text:
        idx = input_letters.find(delete_text)
        if idx == -1:
            return False, (
                "delete_text %r not found in input %r" % (delete_text, input_letters)
            )
        remainder = input_letters[:idx] + input_letters[idx + len(delete_text):]
    elif delete_positions is not None:
        pos_set = set(delete_positions)
        if any(p < 0 or p >= len(input_letters) for p in pos_set):
            return False, "delete_positions contain out-of-range index for input of length %d" % len(input_letters)
        remainder = "".join(
            ch for i, ch in enumerate(input_letters) if i not in pos_set
        )
    else:
        return False, (
            "deletion payload must specify delete_text or delete_positions; "
            "set payload_json to a JSON object with one of those keys"
        )
    claimed = (transform_node.get("raw_letters") or "").upper()
    if claimed and claimed != remainder:
        return False, (
            "claimed transform %r does not match deletion result %r"
            % (claimed, remainder)
        )
    output = claimed or remainder
    positions = transform_node.get("answer_positions") or []
    if positions:
        if len(positions) != len(output):
            return False, (
                "answer_positions length %d does not match output length %d"
                % (len(positions), len(output))
            )
        for i, p in enumerate(positions):
            if p < 0 or p >= len(cleaned_answer):
                return False, "position %d out of range (answer length %d)" % (p, len(cleaned_answer))
            if output[i] != cleaned_answer[p]:
                return False, (
                    "output[%d]=%r does not match answer[%d]=%r"
                    % (i, output[i], p, cleaned_answer[p])
                )
    return True, output


def _evaluate_anagram(op_node, input_nodes, transform_node, cleaned_answer):
    """Return (True, output_letters) or (False, reason_string).

    Anagram requires at least one input. transform.raw_letters must be set
    (the admin specifies which anagram arrangement was chosen).
    Validation: sorted(all input letters) == sorted(transform.raw_letters).
    If transform.answer_positions are set, validates each position against the answer.
    """
    if not input_nodes:
        return False, "anagram requires at least one input"
    all_input = "".join(
        (n.get("raw_letters") or "").upper() for n in input_nodes
    )
    if not all_input:
        return False, "input sources have no raw_letters"
    claimed = (transform_node.get("raw_letters") or "").upper()
    if not claimed:
        return False, (
            "anagram transform must specify raw_letters "
            "(the chosen anagram arrangement)"
        )
    if sorted(all_input) != sorted(claimed):
        return False, (
            "input letters %r cannot be rearranged to produce %r "
            "(sorted inputs: %s, sorted claimed: %s)"
            % (all_input, claimed, "".join(sorted(all_input)), "".join(sorted(claimed)))
        )
    positions = transform_node.get("answer_positions") or []
    if positions:
        if len(positions) != len(claimed):
            return False, (
                "answer_positions length %d does not match output length %d"
                % (len(positions), len(claimed))
            )
        for i, p in enumerate(positions):
            if p < 0 or p >= len(cleaned_answer):
                return False, "position %d out of range (answer length %d)" % (p, len(cleaned_answer))
            if claimed[i] != cleaned_answer[p]:
                return False, (
                    "output[%d]=%r does not match answer[%d]=%r"
                    % (i, claimed[i], p, cleaned_answer[p])
                )
    return True, claimed


def _evaluate_container(op_node, input_nodes, transform_node, cleaned_answer):
    """Return (True, output_letters) or (False, reason_string).

    Container requires two inputs: one outer (frame) and one inner (content).
    transform.raw_letters must be set (the explicit output). Payload may specify
    outer_node_id and inner_node_id to disambiguate which input is which; if
    absent, the first input is treated as outer and the second as inner.

    Validation: inner appears as a substring in the claimed output, and removing
    inner from the output (at that position) leaves exactly the outer letters.
    If transform.answer_positions are set, validates each position against the answer.
    """
    claimed = (transform_node.get("raw_letters") or "").upper()
    if not claimed:
        return False, (
            "container transform must specify raw_letters "
            "(the explicit combined output)"
        )
    payload = _decode_payload(transform_node)
    outer_id = payload.get("outer_node_id")
    inner_id = payload.get("inner_node_id")
    node_by_id = {n["id"]: n for n in input_nodes}
    if outer_id is not None and inner_id is not None:
        outer_node = node_by_id.get(outer_id)
        inner_node = node_by_id.get(inner_id)
        if outer_node is None or inner_node is None:
            return False, (
                "outer_node_id %r or inner_node_id %r not found among input nodes"
                % (outer_id, inner_id)
            )
        outer = (outer_node.get("raw_letters") or "").upper()
        inner = (inner_node.get("raw_letters") or "").upper()
    else:
        if len(input_nodes) != 2:
            return False, (
                "container requires exactly 2 inputs when outer_node_id/inner_node_id "
                "are not specified in payload_json; got %d input(s)" % len(input_nodes)
            )
        outer = (input_nodes[0].get("raw_letters") or "").upper()
        inner = (input_nodes[1].get("raw_letters") or "").upper()
    if not outer or not inner:
        return False, "outer or inner node has no raw_letters"
    if inner not in claimed:
        return False, "inner letters %r not found as substring in claimed output %r" % (inner, claimed)
    idx = claimed.index(inner)
    remainder = claimed[:idx] + claimed[idx + len(inner):]
    if remainder != outer:
        return False, (
            "removing inner %r from claimed %r leaves %r which does not match outer %r"
            % (inner, claimed, remainder, outer)
        )
    positions = transform_node.get("answer_positions") or []
    if positions:
        if len(positions) != len(claimed):
            return False, (
                "answer_positions length %d does not match output length %d"
                % (len(positions), len(claimed))
            )
        for i, p in enumerate(positions):
            if p < 0 or p >= len(cleaned_answer):
                return False, "position %d out of range (answer length %d)" % (p, len(cleaned_answer))
            if claimed[i] != cleaned_answer[p]:
                return False, (
                    "output[%d]=%r does not match answer[%d]=%r"
                    % (i, claimed[i], p, cleaned_answer[p])
                )
    return True, claimed
```

### 1f. Replace merge_manual_into_display entirely

Replace the entire existing merge_manual_into_display function with the version below.
The new signature adds manual_edges=None (keyword argument with default None so the
existing Slice 1 behaviour is unchanged when no edges are passed, but the clue.py
call will be updated to pass them in Change 3).

The Slice 1 invariants all hold:
- Standalone source nodes (not feeding any operator) still produce direct answer links.
- Authoritative position map still guarantees exactly one tile per cleaned-answer
  position with no duplicate positions.
- Auto REVIEW_BLOCKs and SOURCE_BLOCKs are suppressed over covered word indices.
- Auto DEF_BLOCKs are suppressed when a manual definition exists.

New in Slice 2:
- Transform nodes are evaluated via their connected operator before blocks are built.
- Operator nodes emit OP_BLOCKs.
- Transform nodes emit SOURCE_BLOCKs (showing the operation output letters). Their
  span is borrowed from the associated operator's word_indices, since transforms have
  no clue words of their own.
- Source nodes that are inputs to operators still emit SOURCE_BLOCKs but do not
  contribute directly to answer_links; the transform handles answer placement.
- Failed transforms: operator OP_BLOCK is evidence_status="failed"; transform
  SOURCE_BLOCK is evidence_status="failed" and role="source_review"; claimed
  positions receive plain tiles.

```python
def merge_manual_into_display(wfw_display, manual_nodes, answer, manual_edges=None):
    """Merge manual evidence nodes and edges into an existing wfw_display dict.

    Mutates wfw_display["blocks"] and wfw_display["answer_links"] in place.
    Has no Flask or database dependencies.

    This is display-merge only. It does not write to wfw_proof_attempts or change
    any database state. It is only called for admin page requests.

    Slice 1 behaviour is preserved:
      - Standalone source nodes (not connected to any operator as inputs)
        behave exactly as in Slice 1: SOURCE_BLOCK + direct answer links.
      - Authoritative position map guarantees exactly one tile per cleaned-answer
        position. Conflicts → plain tile.
      - Auto REVIEW_BLOCKs and SOURCE_BLOCKs suppressed over covered word indices.
      - Auto DEF_BLOCKs suppressed when manual definition exists.

    Slice 2 additions:
      - Transform nodes are evaluated before blocks are built. Valid transforms
        emit coloured SOURCE_BLOCKs and claim answer positions. Failed transforms
        emit failed SOURCE_BLOCKs and leave claimed positions plain.
      - Operator nodes emit OP_BLOCKs (failed if their transform is failed or absent).
      - Source nodes that feed operators still emit SOURCE_BLOCKs but do not
        contribute to answer_links directly; the transform does.
    """
    cleaned = _clean_answer(answer)
    edges = manual_edges or []

    # ── Index structures ──────────────────────────────────────────────────────
    node_by_id = {n["id"]: n for n in manual_nodes}

    # operator_inputs[op_id] = [input_node, ...]
    operator_inputs = {}
    # transform_to_op[transform_id] = op_id
    transform_to_op = {}
    # source_is_input: ids of source/transform nodes that feed an operator
    source_is_input = set()

    for edge in edges:
        fn = edge["from_node_id"]
        tn = edge["to_node_id"]
        et = edge["edge_type"]
        from_node = node_by_id.get(fn)
        to_node = node_by_id.get(tn)
        if not from_node or not to_node:
            continue
        if et == "input_to":
            # from_node (source or transform) feeds to_node (operator)
            operator_inputs.setdefault(tn, []).append(from_node)
            source_is_input.add(fn)
        elif et == "output_of":
            # from_node (transform) is output of to_node (operator)
            transform_to_op[fn] = tn

    # ── Evaluate transforms ───────────────────────────────────────────────────
    # Results: transform_id -> (ok: bool, output_letters: str|None, reason: str)
    transform_eval = {}
    for node in manual_nodes:
        if node.get("node_type") != "transform":
            continue
        nid = node["id"]
        op_id = transform_to_op.get(nid)
        if op_id is None:
            transform_eval[nid] = (False, None, "no operator connected to this transform")
            continue
        op_node = node_by_id.get(op_id)
        if op_node is None:
            transform_eval[nid] = (False, None, "operator node not found")
            continue
        inputs = operator_inputs.get(op_id, [])
        op_role = (op_node.get("role") or "").lower()
        try:
            if op_role == "reversal":
                ok, payload = _evaluate_reversal(op_node, inputs, node, cleaned)
            elif op_role == "deletion":
                ok, payload = _evaluate_deletion(op_node, inputs, node, cleaned)
            elif op_role == "anagram":
                ok, payload = _evaluate_anagram(op_node, inputs, node, cleaned)
            elif op_role == "container":
                ok, payload = _evaluate_container(op_node, inputs, node, cleaned)
            else:
                ok, payload = False, "unsupported operator role: %r" % op_role
        except Exception as exc:
            ok, payload = False, "evaluation error: %s" % exc
        if ok:
            transform_eval[nid] = (True, payload, "")
        else:
            transform_eval[nid] = (False, None, payload)

    # ── Which operator nodes are failed ──────────────────────────────────────
    # An operator is failed if its transform evaluation failed, or if it has
    # no transform at all.
    operator_has_transform = set(transform_to_op.values())
    operator_failed = set()
    for node in manual_nodes:
        if node.get("node_type") != "operator":
            continue
        nid = node["id"]
        if nid not in operator_has_transform:
            operator_failed.add(nid)
    for tr_id, (ok, _, _) in transform_eval.items():
        if not ok:
            op_id = transform_to_op.get(tr_id)
            if op_id is not None:
                operator_failed.add(op_id)

    # ── Covered word indices (for auto block suppression) ─────────────────────
    covered_word_indices = set()
    for node in manual_nodes:
        for idx in (node.get("word_indices") or []):
            covered_word_indices.add(idx)

    has_manual_def = any(n.get("node_type") == "definition" for n in manual_nodes)

    # ── Build blocks ──────────────────────────────────────────────────────────
    new_blocks = []

    # 1. Manual DEF_BLOCKs first.
    for node in manual_nodes:
        if node.get("node_type") != "definition":
            continue
        wi = node.get("word_indices") or []
        span = [min(wi), max(wi) + 1] if wi else [0, 1]
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

    # 2. SOURCE_BLOCKs for source nodes (all source nodes, including operator inputs).
    for node in manual_nodes:
        if node.get("node_type") != "source":
            continue
        wi = node.get("word_indices") or []
        span = [min(wi), max(wi) + 1] if wi else [0, 1]
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

    # 3. OP_BLOCKs for operator nodes.
    for node in manual_nodes:
        if node.get("node_type") != "operator":
            continue
        wi = node.get("word_indices") or []
        span = [min(wi), max(wi) + 1] if wi else [0, 1]
        nid = node["id"]
        failed = nid in operator_failed
        op_role = (node.get("role") or "").lower()
        block_role = (op_role + "_indicator") if op_role else "op_indicator"
        new_blocks.append({
            "block_id": "manual_op_%d" % nid,
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

    # 4. SOURCE_BLOCKs for transform nodes (showing the output letters).
    # Span is borrowed from the associated operator's word_indices, since
    # transform nodes have no clue words of their own.
    for node in manual_nodes:
        if node.get("node_type") != "transform":
            continue
        nid = node["id"]
        ok, output, reason = transform_eval.get(nid, (False, None, "not evaluated"))
        op_id = transform_to_op.get(nid)
        op_node = node_by_id.get(op_id) if op_id is not None else None
        if op_node:
            wi = op_node.get("word_indices") or []
            span = [min(wi), max(wi) + 1] if wi else [0, 1]
        else:
            wi = node.get("word_indices") or []
            span = [min(wi), max(wi) + 1] if wi else [0, 1]
        role = _colour_role(node) if ok else "source_review"
        new_blocks.append({
            "block_id": "manual_tr_%d" % nid,
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

    # 5. Keep existing auto blocks with Slice 1 suppression rules.
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

    # ── Build answer_links (authoritative position map) ───────────────────────
    manual_by_pos = {}
    conflicted = set()

    def _try_claim(p, entry):
        """Attempt to claim position p. Conflict if already claimed."""
        if p in manual_by_pos:
            conflicted.add(p)
        else:
            manual_by_pos[p] = entry

    # Claims from valid transform nodes.
    for node in manual_nodes:
        if node.get("node_type") != "transform":
            continue
        nid = node["id"]
        ok, output, _ = transform_eval.get(nid, (False, None, ""))
        if not ok:
            continue
        positions = node.get("answer_positions") or []
        if not positions:
            continue
        role = _colour_role(node)
        op_id = transform_to_op.get(nid)
        op_node = node_by_id.get(op_id) if op_id is not None else None
        if op_node:
            wi = op_node.get("word_indices") or []
            span = [min(wi), max(wi) + 1] if wi else None
        else:
            span = None
        for i, p in enumerate(positions):
            if p < 0 or p >= len(cleaned):
                continue
            _try_claim(p, {
                "answer_index": p,
                "letter": cleaned[p],
                "source_block": "manual_tr_%d" % nid,
                "source_span": span,
                "source_text": node["word_text"],
                "source_role": role,
                "source_value": output or "",
                "source_input_value": "",
                "source_value_index": None,
            })

    # Claims from standalone source nodes (not feeding any operator).
    for node in manual_nodes:
        if node.get("node_type") != "source":
            continue
        if node["id"] in source_is_input:
            continue  # This source feeds an operator; transform handles placement.
        if not _validate_source_letters(node, answer):
            continue  # Failed source; no answer links.
        positions = node.get("answer_positions") or []
        if not positions:
            continue
        wi = node.get("word_indices") or []
        span = [min(wi), max(wi) + 1] if wi else [0, 1]
        role = _colour_role(node)
        for p in positions:
            if p < 0 or p >= len(cleaned):
                continue
            _try_claim(p, {
                "answer_index": p,
                "letter": cleaned[p],
                "source_block": "manual_src_%d" % node["id"],
                "source_span": span,
                "source_text": node["word_text"],
                "source_role": role,
                "source_value": node.get("raw_letters") or "",
                "source_input_value": "",
                "source_value_index": None,
            })

    # Apply conflict resolution: replace conflicted positions with plain entries.
    for p in conflicted:
        manual_by_pos[p] = {
            "answer_index": p,
            "letter": cleaned[p],
            "source_block": None,
            "source_span": None,
            "source_text": "",
            "source_role": None,
            "source_value": "",
            "source_input_value": "",
            "source_value_index": None,
        }

    # Auto positions: in-range, not already claimed by manual, first occurrence.
    auto_by_pos = {}
    for link in (wfw_display.get("answer_links") or []):
        p = link.get("answer_index")
        if p is None or p < 0 or p >= len(cleaned):
            continue
        if p not in manual_by_pos and p not in auto_by_pos:
            auto_by_pos[p] = link

    # Emit exactly one entry per cleaned-answer position.
    final_links = []
    for p, ch in enumerate(cleaned):
        if p in manual_by_pos:
            final_links.append(manual_by_pos[p])
        elif p in auto_by_pos:
            final_links.append(auto_by_pos[p])
        else:
            final_links.append({
                "answer_index": p,
                "letter": ch,
                "source_block": None,
                "source_span": None,
                "source_text": "",
                "source_role": None,
                "source_value": "",
                "source_input_value": "",
                "source_value_index": None,
            })
    wfw_display["answer_links"] = final_links
```

---

## Change 2: web/routes/admin.py

Three additions. Do not change any other part of admin.py.

### 2a. Extend create_manual_evidence_node

The existing function already handles "source", "definition", "structural". Extend it
to also accept "operator" and "transform".

Locate the existing validation line:

    if node_type not in ("source", "definition", "structural"):

Replace it with:

    if node_type not in ("source", "definition", "structural", "operator", "transform"):

Locate the existing answer_positions parsing block (which currently reads):

    answer_positions = None
    if answer_positions_raw:
        try:
            parsed = [
                int(x.strip())
                for x in answer_positions_raw.split(",")
                if x.strip()
            ]
        except (ValueError, TypeError):
            abort(400)
        answer_positions = parsed or None

Immediately after that block (after `answer_positions = parsed or None`), insert the
payload_json parsing block:

```python
    payload_json_raw = (request.form.get("payload_json") or "").strip()
    payload_json = None
    if payload_json_raw:
        try:
            decoded = json.loads(payload_json_raw)
            if not isinstance(decoded, dict):
                abort(400)
            payload_json = payload_json_raw
        except (ValueError, TypeError):
            abort(400)
```

json may not be imported at module level in admin.py. Add it explicitly inside
create_manual_evidence_node immediately before the payload_json parsing block:

    import json

Locate the write_node call inside create_manual_evidence_node. Add payload_json as a
keyword argument:

```python
    write_node(
        clue_id=clue_id,
        node_type=node_type,
        word_indices=word_indices,
        word_text=word_text,
        role=role,
        raw_letters=raw_letters,
        answer_positions=answer_positions,
        group_id=group_id,
        payload_json=payload_json,
        conn=db,
    )
```

Update the import inside the function to also import get_edges_for_clue (so the
return can include the current edge list — without this, existing edges disappear
from the HTMX-refreshed list until a full page reload):

```python
    from signature_solver.manual_evidence_store import (
        write_node, get_nodes_for_clue, get_edges_for_clue,
    )
```

Update the final db.commit() + return block to fetch and pass edges:

```python
    db.commit()
    nodes = get_nodes_for_clue(clue_id, conn=db)
    edges = get_edges_for_clue(clue_id, conn=db)
    return render_template(
        "partials/manual_evidence_nodes.html",
        clue_id=clue_id,
        nodes=nodes,
        edges=edges,
    )
```

Locate the existing word_indices rejection check (immediately after the try/except
block that parses word_indices_raw). In the current Slice 1 code it reads:

    if not word_indices:
        abort(400)

Replace it with this version, which exempts transform nodes (transforms have no
clue words of their own so empty word_indices is correct and expected):

    if not word_indices and node_type != "transform":
        abort(400)

### 2b. New create_manual_evidence_edge route

Insert this function immediately after create_manual_evidence_node:

```python
@bp.route("/manual-evidence/<int:clue_id>/edge", methods=["POST"])
def create_manual_evidence_edge(clue_id):
    """Create a manual evidence edge between two existing nodes."""
    _require_admin()
    db = get_admin_db()
    clue = db.execute(
        "SELECT id FROM clues WHERE id = ?", (clue_id,)
    ).fetchone()
    if clue is None:
        abort(404)

    from_node_id_raw = (request.form.get("from_node_id") or "").strip()
    to_node_id_raw = (request.form.get("to_node_id") or "").strip()
    edge_type = (request.form.get("edge_type") or "").strip()

    if edge_type not in ("input_to", "output_of"):
        abort(400)

    try:
        from_node_id = int(from_node_id_raw)
        to_node_id = int(to_node_id_raw)
    except (ValueError, TypeError):
        abort(400)

    # Verify both nodes exist and belong to this clue, then validate types.
    # Fetch node_type so _validate_edge_types can enforce the edge constraints.
    from_row = db.execute(
        "SELECT id, node_type FROM manual_evidence_nodes WHERE id = ? AND clue_id = ?",
        (from_node_id, clue_id),
    ).fetchone()
    to_row = db.execute(
        "SELECT id, node_type FROM manual_evidence_nodes WHERE id = ? AND clue_id = ?",
        (to_node_id, clue_id),
    ).fetchone()
    if from_row is None or to_row is None:
        abort(400)

    from signature_solver.manual_evidence_store import (
        write_edge, get_nodes_for_clue, get_edges_for_clue, _validate_edge_types,
    )
    # Enforce node-type constraints:
    # input_to: from must be source or transform; to must be operator.
    # output_of: from must be transform; to must be operator.
    if not _validate_edge_types(from_row["node_type"], to_row["node_type"], edge_type):
        abort(400)
    write_edge(
        clue_id=clue_id,
        from_node_id=from_node_id,
        to_node_id=to_node_id,
        edge_type=edge_type,
        conn=db,
    )
    db.commit()
    nodes = get_nodes_for_clue(clue_id, conn=db)
    edges = get_edges_for_clue(clue_id, conn=db)
    return render_template(
        "partials/manual_evidence_nodes.html",
        clue_id=clue_id,
        nodes=nodes,
        edges=edges,
    )
```

### 2c. New delete_manual_evidence_edge route

Insert this function immediately after create_manual_evidence_edge:

```python
@bp.route("/manual-evidence/edge/<int:edge_id>/delete", methods=["POST"])
def delete_manual_evidence_edge(edge_id):
    """Delete a manual evidence edge."""
    _require_admin()
    db = get_admin_db()
    row = db.execute(
        "SELECT clue_id FROM manual_evidence_edges WHERE id = ?",
        (edge_id,),
    ).fetchone()
    if row is None:
        abort(404)
    clue_id = row["clue_id"]

    from signature_solver.manual_evidence_store import (
        delete_edge, get_nodes_for_clue, get_edges_for_clue,
    )
    delete_edge(edge_id, conn=db)
    db.commit()
    nodes = get_nodes_for_clue(clue_id, conn=db)
    edges = get_edges_for_clue(clue_id, conn=db)
    return render_template(
        "partials/manual_evidence_nodes.html",
        clue_id=clue_id,
        nodes=nodes,
        edges=edges,
    )
```

---

## Change 3: web/routes/clue.py

Locate the manual evidence block inserted in Slice 1. It currently reads:

```python
    _manual_nodes = []
    if g.get("is_admin", False):
        try:
            from signature_solver.manual_evidence_store import (
                get_nodes_for_clue as _get_manual_nodes,
                merge_manual_into_display as _merge_manual_into_display,
            )
            _manual_nodes = _get_manual_nodes(clue_id)
        except Exception:
            _manual_nodes = []
        if _manual_nodes:
            if clue_dict.get("wfw_display") is None:
                ...
            _merge_manual_into_display(
                clue_dict["wfw_display"],
                _manual_nodes,
                clue["answer"] or "",
            )
    clue_dict["manual_evidence_nodes"] = _manual_nodes
```

Replace that block with this version, which loads edges and passes them to merge:

```python
    # Manual evidence: load graph from DB and merge into wfw_display for admins.
    # This is display-merge only (Slice 2). Public visitors are unaffected.
    _manual_nodes = []
    _manual_edges = []
    if g.get("is_admin", False):
        try:
            from signature_solver.manual_evidence_store import (
                get_graph_for_clue as _get_manual_graph,
                merge_manual_into_display as _merge_manual_into_display,
            )
            _graph = _get_manual_graph(clue_id)
            _manual_nodes = _graph["nodes"]
            _manual_edges = _graph["edges"]
        except Exception:
            _manual_nodes = []
            _manual_edges = []
        if _manual_nodes:
            if clue_dict.get("wfw_display") is None:
                # Build a minimal stub so manual blocks can appear even
                # when no automatic proof exists for this clue.
                clue_dict["wfw_display"] = {
                    "status": "wfw_review",
                    "blocks": [],
                    "answer_links": [],
                    "clue_text": clue["clue_text"] or "",
                    "answer": clue["answer"] or "",
                    "tokens": [],
                    "operations": [],
                    "review_messages": [
                        "Manual evidence only — no automatic proof"
                    ],
                    "objections": [],
                    "proof_source": "manual_only",
                    "proof_row_id": None,
                    "missing_enrichments": [],
                }
            _merge_manual_into_display(
                clue_dict["wfw_display"],
                _manual_nodes,
                clue["answer"] or "",
                manual_edges=_manual_edges,
            )
    clue_dict["manual_evidence_nodes"] = _manual_nodes
    clue_dict["manual_evidence_edges"] = _manual_edges
```

No other changes to clue.py.

---

## Change 4: web/templates/clue.html

Locate the existing manual evidence form (the `<details class="mt-3">` block with
summary text "Admin: manual evidence").

Three additions:

### 4a. Add payload_json textarea to the node creation form

Inside the second grid div (the one with "Letters produced", "Answer positions", and
"Group id" fields), add a fourth label after the group_id label:

```html
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700 md:col-span-3">
                        Payload JSON (operator/transform extra data, optional)
                        <textarea name="payload_json" rows="2" placeholder='{"delete_text": "ED"}'
                                  class="rounded border border-slate-300 px-2 py-1 bg-white font-mono text-xs"></textarea>
                    </label>
```

### 4b. Add edge creation form

Insert a new `<form>` block immediately after the existing node creation form's closing
`</form>` tag and before the `{% with clue_id=... %}{% include ... %}{% endwith %}` line:

```html
            <form hx-post="/admin/manual-evidence/{{ clue.id }}/edge"
                  hx-target="#manual-evidence-list-{{ clue.id }}"
                  hx-swap="outerHTML"
                  hx-on::after-request="this.reset()"
                  class="grid grid-cols-1 gap-2 text-sm mb-3 border-t border-violet-100 pt-3">
                <p class="text-xs font-semibold text-violet-700">Add edge (connect nodes)</p>
                <div class="grid grid-cols-3 gap-2">
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        From node id
                        <input name="from_node_id" type="number" required placeholder="e.g. 3"
                               class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                    </label>
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        To node id
                        <input name="to_node_id" type="number" required placeholder="e.g. 5"
                               class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                    </label>
                    <label class="flex flex-col gap-1 text-xs font-semibold text-slate-700">
                        Edge type
                        <select name="edge_type"
                                class="rounded border border-slate-300 px-2 py-1 bg-white text-xs">
                            <option value="input_to">input_to (source → operator)</option>
                            <option value="output_of">output_of (transform → operator)</option>
                        </select>
                    </label>
                </div>
                <button type="submit"
                        class="self-start text-xs px-3 py-1 rounded border border-violet-400 text-violet-700 hover:bg-violet-100 cursor-pointer">
                    Add edge
                </button>
            </form>
```

### 4c. Update the include call to pass edges

The existing include line reads:

    {% with clue_id=clue.id, nodes=clue.manual_evidence_nodes %}

Replace it with:

    {% with clue_id=clue.id, nodes=clue.manual_evidence_nodes, edges=clue.manual_evidence_edges %}

clue_dict["manual_evidence_edges"] = _manual_edges is already included in the
Change 3 replacement block (set unconditionally, empty list for non-admins).
No separate clue.py change is needed here.

### 4d. Remove `required` from the existing word_indices input

The Slice 1 form has a word_indices input with the HTML `required` attribute. For
transform nodes the field must be left blank (word_indices=[]), so `required`
now conflicts with the expected usage.

Locate this input in the existing Slice 1 node creation form:

```html
                        <input name="word_indices" placeholder="0,1" required
                               class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
```

Replace it with (remove `required`, update placeholder to signal optionality for
transforms, keep everything else unchanged):

```html
                        <input name="word_indices" placeholder="0,1 (blank for transform)"
                               class="rounded border border-slate-300 px-2 py-1 bg-white font-normal text-xs">
```

The server-side logic already enforces the rule: empty word_indices is allowed
only when node_type == "transform"; all other node types still get abort(400).
The `required` attribute was the only client-side gate, and removing it does not
weaken the server-side check.

---

## Change 5: web/templates/partials/manual_evidence_nodes.html

Replace the entire file with the version below. The template now receives an `edges`
variable (list of dicts). Node ids are shown prominently so the admin can use them
when creating edges.

```html
{# Manual evidence node and edge list.
   Included in clue.html and returned by create/delete admin routes.
   Variables: clue_id (int), nodes (list of dicts), edges (list of dicts). #}
<div id="manual-evidence-list-{{ clue_id }}">
{% if nodes %}
<p class="text-xs text-gray-400 mt-1 mb-1">Nodes (use the id numbers to create edges)</p>
<ul class="space-y-1 mt-1">
{% for node in nodes %}
<li class="flex flex-wrap items-center gap-2 rounded border bg-white px-2 py-1.5 text-xs
    {% if node.node_type == 'source' %}border-sky-200
    {% elif node.node_type == 'definition' %}border-green-200
    {% elif node.node_type == 'operator' %}border-amber-200
    {% elif node.node_type == 'transform' %}border-violet-200
    {% else %}border-slate-200{% endif %}">
    <span class="font-mono font-bold text-gray-400 shrink-0">#{{ node.id }}</span>
    <span class="rounded px-1.5 py-0.5 font-semibold shrink-0
        {% if node.node_type == 'source' %}bg-sky-100 text-sky-700
        {% elif node.node_type == 'definition' %}bg-green-100 text-green-700
        {% elif node.node_type == 'operator' %}bg-amber-100 text-amber-700
        {% elif node.node_type == 'transform' %}bg-violet-100 text-violet-700
        {% else %}bg-slate-100 text-slate-600{% endif %}">
        {{ node.node_type }}
    </span>
    <span class="font-medium text-gray-900">{{ node.word_text }}</span>
    {% if node.role %}<span class="text-gray-500">({{ node.role }})</span>{% endif %}
    {% if node.raw_letters %}
    <span class="font-mono text-emerald-700">&#8594; {{ node.raw_letters }}</span>
    {% endif %}
    {% if node.answer_positions is not none %}
    <span class="text-sky-600 font-mono">pos {{ node.answer_positions }}</span>
    {% endif %}
    {% if node.group_id is not none %}
    <span class="text-violet-600">grp {{ node.group_id }}</span>
    {% endif %}
    {% if node.word_indices %}
    <span class="text-gray-400">idx {{ node.word_indices }}</span>
    {% endif %}
    {% if node.payload_json %}
    <span class="text-orange-600 font-mono text-xs">{{ node.payload_json }}</span>
    {% endif %}
    <form hx-post="/admin/manual-evidence/node/{{ node.id }}/delete"
          hx-target="#manual-evidence-list-{{ clue_id }}"
          hx-swap="outerHTML"
          class="ml-auto">
        <button type="submit"
                class="text-red-400 hover:text-red-600 font-bold cursor-pointer"
                title="Delete this node (also deletes attached edges)">&#10005;</button>
    </form>
</li>
{% endfor %}
</ul>
{% else %}
<p class="text-xs text-gray-400 mt-1">No manual evidence recorded.</p>
{% endif %}

{% set edges = edges if edges is defined else [] %}
{% if edges %}
<p class="text-xs text-gray-400 mt-3 mb-1">Edges</p>
<ul class="space-y-1 mt-1">
{% for edge in edges %}
<li class="flex flex-wrap items-center gap-2 rounded border border-gray-100 bg-white px-2 py-1 text-xs">
    <span class="font-mono text-gray-400">#{{ edge.id }}</span>
    <span class="font-mono text-sky-700">#{{ edge.from_node_id }}</span>
    <span class="text-gray-400">&#8594;</span>
    <span class="font-mono text-violet-700">#{{ edge.to_node_id }}</span>
    <span class="rounded bg-gray-100 px-1.5 py-0.5 text-gray-600">{{ edge.edge_type }}</span>
    <form hx-post="/admin/manual-evidence/edge/{{ edge.id }}/delete"
          hx-target="#manual-evidence-list-{{ clue_id }}"
          hx-swap="outerHTML"
          class="ml-auto">
        <button type="submit"
                class="text-red-400 hover:text-red-600 font-bold cursor-pointer"
                title="Delete this edge">&#10005;</button>
    </form>
</li>
{% endfor %}
</ul>
{% endif %}
</div>
```

---

## Verification

### Check 1 — syntax

    .venv\Scripts\python.exe -m py_compile signature_solver\manual_evidence_store.py
    .venv\Scripts\python.exe -m py_compile web\routes\admin.py
    .venv\Scripts\python.exe -m py_compile web\routes\clue.py

Expected: no output, exit 0 for each.

### Check 2 — temp DB graph round-trip

Uses a temp file. No data reaches clues_master.db.

    .venv\Scripts\python.exe -c "
    import sys, tempfile, os, sqlite3
    sys.path.insert(0, '.')
    from signature_solver.manual_evidence_store import (
        write_node, write_edge, get_graph_for_clue,
        delete_node, ensure_tables,
    )
    tmp = tempfile.mktemp(suffix='.db')
    try:
        conn = sqlite3.connect(tmp)
        conn.row_factory = sqlite3.Row
        ensure_tables(conn)

        src_id = write_node(
            clue_id=1, node_type='source', word_indices=[0], word_text='CAT',
            role='synonym', raw_letters='TOM', group_id=0, conn=conn,
        )
        op_id = write_node(
            clue_id=1, node_type='operator', word_indices=[1], word_text='BACKS',
            role='reversal', conn=conn,
        )
        tr_id = write_node(
            clue_id=1, node_type='transform', word_indices=[], word_text='MOT',
            raw_letters='MOT', answer_positions=[0, 1, 2], group_id=0, conn=conn,
        )
        conn.commit()

        e1 = write_edge(clue_id=1, from_node_id=src_id, to_node_id=op_id,
                        edge_type='input_to', conn=conn)
        e2 = write_edge(clue_id=1, from_node_id=tr_id, to_node_id=op_id,
                        edge_type='output_of', conn=conn)
        conn.commit()

        graph = get_graph_for_clue(1, conn=conn)
        nodes = graph['nodes']
        edges = graph['edges']
        assert len(nodes) == 3, 'Expected 3 nodes, got ' + str(len(nodes))
        assert len(edges) == 2, 'Expected 2 edges, got ' + str(len(edges))

        src_node = next(n for n in nodes if n['id'] == src_id)
        assert src_node['raw_letters'] == 'TOM', 'source raw_letters mismatch'
        assert src_node['word_indices'] == [0], 'source word_indices mismatch'
        assert src_node['group_id'] == 0, 'source group_id mismatch'

        tr_node = next(n for n in nodes if n['id'] == tr_id)
        assert tr_node['raw_letters'] == 'MOT', 'transform raw_letters mismatch'
        assert tr_node['answer_positions'] == [0, 1, 2], 'transform positions mismatch'

        input_edges = [e for e in edges if e['edge_type'] == 'input_to']
        output_edges = [e for e in edges if e['edge_type'] == 'output_of']
        assert len(input_edges) == 1 and input_edges[0]['from_node_id'] == src_id
        assert len(output_edges) == 1 and output_edges[0]['from_node_id'] == tr_id
        print('PASS: nodes and edges stored and retrieved correctly')

        delete_node(op_id, conn=conn)
        conn.commit()
        graph2 = get_graph_for_clue(1, conn=conn)
        assert len(graph2['edges']) == 0, 'Edges not deleted with operator node'
        print('PASS: deleting a node removes all attached edges')
    finally:
        conn.close()
        try:
            os.unlink(tmp)
        except Exception:
            pass
    "

Expected output:

    PASS: nodes and edges stored and retrieved correctly
    PASS: deleting a node removes all attached edges

### Check 3 — reversal display merge

Pure function test. No Flask, no live DB.

    .venv\Scripts\python.exe -c "
    import sys; sys.path.insert(0, '.')
    from signature_solver.manual_evidence_store import merge_manual_into_display

    wfw = {'blocks': [], 'answer_links': []}
    nodes = [
        {'id': 1, 'node_type': 'source', 'word_text': 'CAT', 'word_indices': [0],
         'role': 'synonym', 'raw_letters': 'TOM', 'answer_positions': None,
         'group_id': 0, 'payload_json': {}},
        {'id': 2, 'node_type': 'operator', 'word_text': 'BACKS', 'word_indices': [1],
         'role': 'reversal', 'raw_letters': None, 'answer_positions': None,
         'group_id': None, 'payload_json': {}},
        {'id': 3, 'node_type': 'transform', 'word_text': 'MOT', 'word_indices': [],
         'role': None, 'raw_letters': 'MOT', 'answer_positions': [0, 1, 2],
         'group_id': 0, 'payload_json': {}},
    ]
    edges = [
        {'id': 1, 'from_node_id': 1, 'to_node_id': 2, 'edge_type': 'input_to'},
        {'id': 2, 'from_node_id': 3, 'to_node_id': 2, 'edge_type': 'output_of'},
    ]
    merge_manual_into_display(wfw, nodes, 'MOTEL', manual_edges=edges)

    assert len(wfw['answer_links']) == 5, 'Expected 5 links for MOTEL, got ' + str(len(wfw['answer_links']))
    positions = [lk['answer_index'] for lk in wfw['answer_links']]
    assert positions == list(range(5)), 'Positions not 0..4: ' + str(positions)
    assert len(set(positions)) == 5, 'Duplicate positions found'

    lmap = {lk['answer_index']: lk for lk in wfw['answer_links']}
    for p in [0, 1, 2]:
        assert lmap[p]['source_role'] == 'piece_0', 'Position %d should be piece_0, got %r' % (p, lmap[p]['source_role'])
    for p in [3, 4]:
        assert lmap[p]['source_role'] is None, 'Position %d should be plain, got %r' % (p, lmap[p]['source_role'])

    src_blocks = [b for b in wfw['blocks'] if b['block_id'] == 'manual_src_1']
    assert len(src_blocks) == 1, 'CAT SOURCE_BLOCK missing'
    op_blocks = [b for b in wfw['blocks'] if b['kind'] == 'OP_BLOCK']
    assert len(op_blocks) == 1, 'BACKS OP_BLOCK missing'
    assert op_blocks[0]['evidence_status'] == 'manual', 'OP_BLOCK should be manual status'
    tr_blocks = [b for b in wfw['blocks'] if b['block_id'] == 'manual_tr_3']
    assert len(tr_blocks) == 1, 'transform SOURCE_BLOCK missing'
    assert tr_blocks[0]['evidence_status'] == 'manual', 'transform block should be manual'
    print('PASS: reversal display merge')
    "

Expected output:

    PASS: reversal display merge

### Check 4 — two-piece display merge

    .venv\Scripts\python.exe -c "
    import sys; sys.path.insert(0, '.')
    from signature_solver.manual_evidence_store import merge_manual_into_display

    wfw = {'blocks': [], 'answer_links': []}
    nodes = [
        {'id': 1, 'node_type': 'source', 'word_text': 'CAT', 'word_indices': [0],
         'role': 'synonym', 'raw_letters': 'TOM', 'answer_positions': None,
         'group_id': 0, 'payload_json': {}},
        {'id': 2, 'node_type': 'operator', 'word_text': 'BACKS', 'word_indices': [1],
         'role': 'reversal', 'raw_letters': None, 'answer_positions': None,
         'group_id': None, 'payload_json': {}},
        {'id': 3, 'node_type': 'transform', 'word_text': 'MOT', 'word_indices': [],
         'role': None, 'raw_letters': 'MOT', 'answer_positions': [0, 1, 2],
         'group_id': 0, 'payload_json': {}},
        {'id': 4, 'node_type': 'source', 'word_text': 'THE SPANISH', 'word_indices': [3, 4],
         'role': 'synonym', 'raw_letters': 'EL', 'answer_positions': [3, 4],
         'group_id': 1, 'payload_json': {}},
    ]
    edges = [
        {'id': 1, 'from_node_id': 1, 'to_node_id': 2, 'edge_type': 'input_to'},
        {'id': 2, 'from_node_id': 3, 'to_node_id': 2, 'edge_type': 'output_of'},
    ]
    merge_manual_into_display(wfw, nodes, 'MOTEL', manual_edges=edges)

    assert len(wfw['answer_links']) == 5, 'Expected 5 links, got ' + str(len(wfw['answer_links']))
    positions = [lk['answer_index'] for lk in wfw['answer_links']]
    assert positions == list(range(5)), 'Positions not 0..4: ' + str(positions)
    assert len(set(positions)) == 5

    lmap = {lk['answer_index']: lk for lk in wfw['answer_links']}
    for p in [0, 1, 2]:
        assert lmap[p]['source_role'] == 'piece_0', 'Position %d should be piece_0' % p
    for p in [3, 4]:
        assert lmap[p]['source_role'] == 'piece_1', 'Position %d should be piece_1' % p

    block_ids = {b['block_id'] for b in wfw['blocks']}
    assert 'manual_src_1' in block_ids, 'CAT SOURCE_BLOCK missing'
    assert 'manual_src_4' in block_ids, 'THE SPANISH SOURCE_BLOCK missing'
    assert 'manual_tr_3' in block_ids, 'transform SOURCE_BLOCK missing'
    assert 'manual_op_2' in block_ids, 'BACKS OP_BLOCK missing'
    print('PASS: two-piece display merge')
    "

Expected output:

    PASS: two-piece display merge

### Check 5 — failed operation does not falsely colour

    .venv\Scripts\python.exe -c "
    import sys; sys.path.insert(0, '.')
    from signature_solver.manual_evidence_store import merge_manual_into_display

    # Transform claims XYZ but reversal of TOM is MOT — mismatch → failed.
    wfw = {'blocks': [], 'answer_links': []}
    nodes = [
        {'id': 1, 'node_type': 'source', 'word_text': 'CAT', 'word_indices': [0],
         'role': 'synonym', 'raw_letters': 'TOM', 'answer_positions': None,
         'group_id': 0, 'payload_json': {}},
        {'id': 2, 'node_type': 'operator', 'word_text': 'BACKS', 'word_indices': [1],
         'role': 'reversal', 'raw_letters': None, 'answer_positions': None,
         'group_id': None, 'payload_json': {}},
        {'id': 3, 'node_type': 'transform', 'word_text': 'XYZ (wrong)', 'word_indices': [],
         'role': None, 'raw_letters': 'XYZ', 'answer_positions': [0, 1, 2],
         'group_id': 0, 'payload_json': {}},
    ]
    edges = [
        {'id': 1, 'from_node_id': 1, 'to_node_id': 2, 'edge_type': 'input_to'},
        {'id': 2, 'from_node_id': 3, 'to_node_id': 2, 'edge_type': 'output_of'},
    ]
    merge_manual_into_display(wfw, nodes, 'MOTEL', manual_edges=edges)

    assert len(wfw['answer_links']) == 5, 'Expected 5 tiles even for failed op, got ' + str(len(wfw['answer_links']))
    positions = [lk['answer_index'] for lk in wfw['answer_links']]
    assert positions == list(range(5)), 'Positions not 0..4'
    assert len(set(positions)) == 5

    lmap = {lk['answer_index']: lk for lk in wfw['answer_links']}
    for p in [0, 1, 2]:
        assert lmap[p]['source_role'] is None, 'Position %d must be plain for failed transform, got %r' % (p, lmap[p]['source_role'])

    op_blocks = [b for b in wfw['blocks'] if b['kind'] == 'OP_BLOCK']
    assert len(op_blocks) == 1 and op_blocks[0]['evidence_status'] == 'failed', 'OP_BLOCK should be failed'

    tr_blocks = [b for b in wfw['blocks'] if b['block_id'] == 'manual_tr_3']
    assert len(tr_blocks) == 1 and tr_blocks[0]['evidence_status'] == 'failed', 'transform block should be failed'
    assert tr_blocks[0]['role'] == 'source_review', 'failed transform should have source_review role'
    print('PASS: failed operation produces plain tiles and failed blocks')
    "

Expected output:

    PASS: failed operation produces plain tiles and failed blocks

### Check 6 — edge type validation rejects invalid combinations

Pure function test. No Flask, no live DB.

    .venv\Scripts\python.exe -c "
    import sys; sys.path.insert(0, '.')
    from signature_solver.manual_evidence_store import _validate_edge_types

    # Valid combinations
    assert _validate_edge_types('source', 'operator', 'input_to'), 'source->operator input_to should be valid'
    assert _validate_edge_types('transform', 'operator', 'input_to'), 'transform->operator input_to should be valid'
    assert _validate_edge_types('transform', 'operator', 'output_of'), 'transform->operator output_of should be valid'

    # Invalid combinations
    assert not _validate_edge_types('operator', 'operator', 'input_to'), 'operator->operator input_to must be rejected'
    assert not _validate_edge_types('source', 'operator', 'output_of'), 'source->operator output_of must be rejected'
    assert not _validate_edge_types('source', 'source', 'input_to'), 'source->source input_to must be rejected'
    assert not _validate_edge_types('transform', 'source', 'output_of'), 'transform->source output_of must be rejected'
    assert not _validate_edge_types('definition', 'operator', 'input_to'), 'definition->operator input_to must be rejected'

    print('PASS: edge type validation rejects invalid node type combinations')
    "

Expected output:

    PASS: edge type validation rejects invalid node type combinations

### Check 7 — get_nodes_for_clue works against Slice 1 schema (no payload_json column)

Uses a temp file. No data reaches clues_master.db.

    .venv\Scripts\python.exe -c "
    import sys, tempfile, os, sqlite3
    sys.path.insert(0, '.')
    from signature_solver.manual_evidence_store import get_nodes_for_clue

    tmp = tempfile.mktemp(suffix='.db')
    try:
        conn = sqlite3.connect(tmp)
        conn.row_factory = sqlite3.Row
        # Create Slice 1 schema: manual_evidence_nodes without payload_json column.
        conn.execute('''
            CREATE TABLE manual_evidence_nodes (
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
        ''')
        conn.execute(
            '''INSERT INTO manual_evidence_nodes
               (clue_id, node_type, word_indices, word_text, role, raw_letters)
               VALUES (1, 'source', '[0]', 'that', 'synonym', 'HE')'''
        )
        conn.commit()

        # Must not raise. Must return the node with payload_json={}.
        nodes = get_nodes_for_clue(1, conn=conn)
        assert len(nodes) == 1, 'Expected 1 node, got ' + str(len(nodes))
        assert nodes[0]['word_text'] == 'that', 'word_text mismatch'
        assert nodes[0]['raw_letters'] == 'HE', 'raw_letters mismatch'
        assert nodes[0]['payload_json'] == {}, 'payload_json should be empty dict when column absent'
        print('PASS: get_nodes_for_clue reads Slice 1 schema without error')
    finally:
        conn.close()
        try:
            os.unlink(tmp)
        except Exception:
            pass
    "

Expected output:

    PASS: get_nodes_for_clue reads Slice 1 schema without error

### Check 8 — word_indices parsing allows blank only for transform nodes

Pure function test. No Flask, no live DB. Simulates the parsing logic in
create_manual_evidence_node using the same conditional that Codex must implement.

    .venv\Scripts\python.exe -c "
    # Simulate the word_indices parsing logic from create_manual_evidence_node.
    # Returns the parsed list, or None to represent abort(400).
    def _sim(word_indices_raw, node_type):
        try:
            wi = [int(x.strip()) for x in (word_indices_raw or '').split(',') if x.strip()]
        except (ValueError, TypeError):
            wi = []
        if not wi and node_type != 'transform':
            return None  # route would abort(400)
        return wi

    # Transform: blank word_indices must be accepted.
    assert _sim('', 'transform') == [], 'blank ok for transform'
    assert _sim('0,1', 'transform') == [0, 1], 'explicit ok for transform'

    # All other node types: blank must be rejected.
    for nt in ('source', 'operator', 'definition', 'structural'):
        assert _sim('', nt) is None, 'blank must be rejected for ' + nt
        assert _sim('1', nt) == [1], 'explicit ok for ' + nt

    print('PASS: word_indices parsing allows blank only for transform nodes')
    "

Expected output:

    PASS: word_indices parsing allows blank only for transform nodes

### Check 9 — rerun safety audit

Read lines 880-900 of web/routes/admin.py (the _rerun_clue_inner upfront clear block).
Paste those lines verbatim and confirm:

a. The only DELETE statement references structured_explanations.
b. The only UPDATE statement references clues.
c. Neither manual_evidence_nodes nor manual_evidence_edges appears anywhere in those lines.

---

## Known limitations

- Homophone is not supported in Slice 2. Its validation requires checking that
  raw_letters sounds like the input letters, which needs a phonetic DB or manual
  attestation. This can be added as a Slice 3 extension without changing the graph
  schema.
- Container validation assumes the inner substring appears once. If the inner appears
  multiple times in the claimed output, the first occurrence is used. The admin can
  resolve ambiguity by supplying explicit outer_node_id and inner_node_id in payload_json.
- Transform nodes borrow the operator's span for display positioning. If an operator
  node has empty word_indices, the transform block appears at [0, 1] in the display.
  The admin should set word_indices on operator nodes.
- The merge function trusts the admin's graph structure. It validates the mathematical
  correctness of operations (letters match, reversal is correct etc.) but does not
  detect logical errors in how the graph was constructed (e.g. connecting a definition
  node as input to an operator).
- The admin UI is minimal and requires manual id lookup to create edges. This is
  acceptable for Slice 2. A more guided UI can be added in a later slice.

---

## After writing

Paste:
1. Output of Check 1.
2. Output of Check 2.
3. Output of Check 3.
4. Output of Check 4.
5. Output of Check 5.
6. Output of Check 6.
7. Output of Check 7.
8. Output of Check 8.
9. Lines 880-900 of web/routes/admin.py (for Check 9).
